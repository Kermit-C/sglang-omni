# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Incremental RedAE decoding and per-request attention and overlap-add caches."""

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import DynamicCache, Qwen3Config, Qwen3Model
from transformers.cache_utils import DynamicSlidingWindowLayer

from sglang_omni.models.firered_tts3.istft import ISTFTHead


@dataclass
class RedAEDecoderState:
    """Per-request streaming state: Qwen3 KV cache + iSTFT OLA cache."""

    qwen3_cache: DynamicCache | None = None
    istft_cache: torch.Tensor | None = None
    frames_seen: int = 0


WINDOW_SCALARS: dict[torch.device, torch.Tensor] = {}


def build_initialized_cache(
    config: Qwen3Config, device: torch.device, dtype: torch.dtype = torch.float32
) -> DynamicCache:
    """A DynamicCache whose layers are pre-initialized for device decode."""
    cache = DynamicCache(config=config)
    window_scalar = WINDOW_SCALARS.get(device)
    if window_scalar is None:
        window_scalar = torch.tensor(
            config.sliding_window, dtype=torch.long, device=device
        )
        WINDOW_SCALARS[device] = window_scalar
    else:
        pass
    empty = torch.zeros(0, dtype=dtype, device=device)
    # note (kermit): Share the device window scalar to avoid synchronizing once per cache layer.
    for layer in cache.layers:
        layer.dtype, layer.device = (dtype, device)
        layer.keys = empty
        layer.values = empty
        layer.is_initialized = True
        if isinstance(layer, DynamicSlidingWindowLayer):
            layer._sliding_window_tensor = window_scalar  # noqa: SLF001  # leading-underscore: Transformers cache attribute
        else:
            pass
    return cache


class RedAEAudioDecoder(torch.nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        upsample_rate: int = 2,
        audio_patch_size: int = 480,
        audio_sample_rate: int = 24000,
        hidden_size: int = 896,
        intermediate_size: int = 896 * 4,
        num_hidden_layers: int = 18,
        max_position_embeddings: int = 32768,
        max_window_layers: int = 0,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        sliding_window: int = 64,
        use_sliding_window: bool = True,
    ) -> None:
        super().__init__()
        self.upsample_rate: int = upsample_rate
        self.audio_patch_size: int = audio_patch_size
        self.audio_sample_rate: int = audio_sample_rate
        self.in_proj: nn.Linear = nn.Linear(in_dim, upsample_rate * hidden_size)
        self.qwen3_config: Qwen3Config = Qwen3Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            max_window_layers=max_window_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            sliding_window=sliding_window,
            use_sliding_window=use_sliding_window,
        )
        self.qwen3: Qwen3Model = Qwen3Model(self.qwen3_config)
        self.istft_head: ISTFTHead = ISTFTHead(
            dim=hidden_size,
            n_fft=audio_patch_size * 4,
            hop_length=audio_patch_size,
            padding="same",
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Decode a full segment for independent streaming comparisons."""
        xs = self.in_proj(xs)
        xs = xs.reshape(xs.shape[0], -1, self.qwen3_config.hidden_size)
        outs = self.qwen3(inputs_embeds=xs, attention_mask=None)
        xs = outs.last_hidden_state
        audio = self.istft_head(xs)
        return audio

    def forward_stream(
        self,
        xs: torch.Tensor,
        cache: RedAEDecoderState | None = None,
        last_chunk: bool = False,
    ) -> tuple[torch.Tensor, RedAEDecoderState]:
        """Incremental decode of one latent chunk."""
        if cache is None:
            cache = RedAEDecoderState(
                build_initialized_cache(
                    self.qwen3_config, device=xs.device, dtype=self.in_proj.weight.dtype
                )
            )
        else:
            pass
        xs = self.in_proj(xs)
        xs = xs.reshape(xs.shape[0], -1, self.qwen3_config.hidden_size)
        outs = self.qwen3.forward(
            inputs_embeds=xs, use_cache=True, past_key_values=cache.qwen3_cache
        )
        xs = outs.last_hidden_state
        new_qwen3_cache = outs.past_key_values
        audio_chunk, new_istft_cache = self.istft_head.forward_stream(
            xs, cache.istft_cache, last_chunk
        )
        new_cache = RedAEDecoderState(new_qwen3_cache, new_istft_cache)
        new_cache.frames_seen = cache.frames_seen + int(xs.shape[1])
        return (audio_chunk, new_cache)

    def flush_stream(self, cache: RedAEDecoderState) -> torch.Tensor:
        """Terminal OLA tail once the final chunk is already decoded."""
        if cache.istft_cache is None:
            raise RuntimeError("RedAE flush_stream requires a decoded chunk first")
        else:
            pass
        return self.istft_head.istft.flush_stream(cache.istft_cache)

    def merge_states(self, states: list[RedAEDecoderState]) -> RedAEDecoderState:
        """Merge equal-progress per-request states into one batched state."""
        merged_cache = build_initialized_cache(
            self.qwen3_config,
            device=states[0].qwen3_cache.layers[0].keys.device,
            dtype=states[0].qwen3_cache.layers[0].keys.dtype,
        )
        for layer_index, layer in enumerate(merged_cache.layers):
            first = states[0].qwen3_cache.layers[layer_index]
            layer.keys = torch.cat(
                [s.qwen3_cache.layers[layer_index].keys for s in states], dim=0
            )
            layer.values = torch.cat(
                [s.qwen3_cache.layers[layer_index].values for s in states], dim=0
            )
            if isinstance(layer, DynamicSlidingWindowLayer):
                layer.cumulative_length = first.cumulative_length
            else:
                pass
        merged = RedAEDecoderState(merged_cache)
        merged.istft_cache = torch.cat([s.istft_cache for s in states], dim=0)
        merged.frames_seen = states[0].frames_seen
        return merged

    def split_state(
        self, state: RedAEDecoderState, batch_size: int
    ) -> list[RedAEDecoderState]:
        """Split a batched state back into per-request states."""
        states = []
        for row in range(batch_size):
            row_cache = DynamicCache(config=self.qwen3_config)
            for layer_index, layer in enumerate(state.qwen3_cache.layers):
                row_layer = copy.copy(layer)
                row_layer.keys = layer.keys[row : row + 1]
                row_layer.values = layer.values[row : row + 1]
                row_cache.layers[layer_index] = row_layer
            row_state = RedAEDecoderState(row_cache)
            row_state.istft_cache = state.istft_cache[row : row + 1]
            row_state.frames_seen = state.frames_seen
            states.append(row_state)
        return states


def rematerialize_istft_window(
    decoder: RedAEAudioDecoder, device: torch.device | str
) -> None:
    """Recompute the iSTFT Hann window after a meta-device materialization."""
    istft = decoder.istft_head.istft
    istft.window = torch.hann_window(istft.win_length, device=device)
