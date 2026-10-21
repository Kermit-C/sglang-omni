# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RedAE audio decoder — the single (streaming) implementation.

Structure ported from the FireRedTTS3 reference implementation
(fireredtts3/redae/redae.py, RedAEAudioDecoder); forward_stream /
RedAEDecoderState from the RedAE streaming decoder. Only ONE decode path
exists (the single chunked implementation pattern):

* async-chunk mode: forward_stream per chunk with a per-request
  RedAEDecoderState (Qwen3 KV cache + iSTFT OLA cache) — every chunk is a
  pure increment, the SWA window provides left context internally.
* full-payload mode: the SAME forward_stream called once with the whole
  latent sequence, cache=None, last_chunk=True — sample-exact with the
  reference's non-streaming decode, so no second implementation is kept.
  The reference's full-segment forward stays in-tree only as the unit-test
  oracle (tests/unit_test/firered_tts3/).

The Qwen3 backbone is HF transformers Qwen3Model (as in the reference):
18 layers, all-SWA(64). This module runs under fp32 + eager.
"""

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import DynamicCache, Qwen3Config, Qwen3Model

from .istft import ISTFTHead


@dataclass
class RedAEDecoderState:
    """Per-request streaming state: Qwen3 KV cache + iSTFT OLA cache."""

    qwen3_cache: object = None
    istft_cache: torch.Tensor | None = None
    # Bookkeeping for sanity checks (frames seen so far at 50 Hz).
    frames_seen: int = 0


# Per-device sliding-window scalars for build_initialized_cache; the value is
# a constant, so it is moved off-host once instead of per constructed cache.
_WINDOW_SCALARS: dict[torch.device, torch.Tensor] = {}


def build_initialized_cache(
    config: Qwen3Config, device: torch.device, dtype: torch.dtype = torch.float32
) -> DynamicCache:
    """A DynamicCache whose layers are pre-initialized for device decode.

    The stock per-request construction leaves each layer uninitialized, so
    the first update pays a pageable 8-byte H2D copy per layer (the sliding
    window scalar) that stream-syncs on the decode stream. Pre-initializing
    costs only CPU-side attribute sets plus cached-allocator empty tensors;
    the sliding-window scalar is a shared constant (never mutated per
    layer). Numerics are unchanged: the first update cats the same empty
    on-device tensors the lazy path would have made.
    """
    cache = DynamicCache(config=config)
    window_scalar = _WINDOW_SCALARS.get(device)
    if window_scalar is None:
        window_scalar = torch.tensor(
            config.sliding_window, dtype=torch.long, device=device
        )
        _WINDOW_SCALARS[device] = window_scalar
    else:
        pass
    empty = torch.zeros(0, dtype=dtype, device=device)
    for layer in cache.layers:
        layer.dtype, layer.device = dtype, device
        layer.keys = empty
        layer.values = empty
        layer.is_initialized = True
        if hasattr(layer, "_sliding_window_tensor"):
            layer._sliding_window_tensor = window_scalar  # noqa: leading-underscore  # Transformers cache attribute
        else:
            pass
    return cache


class RedAEAudioDecoder(torch.nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        upsample_rate: int = 2,  # 25Hz -> 50Hz
        # Output reshape
        audio_patch_size: int = 480,  # 50Hz
        audio_sample_rate: int = 24000,
        # Qwen(mirrors encoder)
        hidden_size: int = 896,
        intermediate_size: int = 896 * 4,
        num_hidden_layers: int = 18,
        max_position_embeddings: int = 32768,
        max_window_layers: int = 0,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        sliding_window: int = 64,
        use_sliding_window: bool = True,
    ):
        super().__init__()
        self.upsample_rate = upsample_rate
        self.audio_patch_size = audio_patch_size
        self.audio_sample_rate = audio_sample_rate
        # Upsample MLP
        self.in_proj = nn.Linear(in_dim, upsample_rate * hidden_size)
        # Qwen3
        self.qwen3_config = Qwen3Config(
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
        self.qwen3 = Qwen3Model(self.qwen3_config)

        self.istft_head = ISTFTHead(
            dim=hidden_size,
            n_fft=audio_patch_size * 4,
            hop_length=audio_patch_size,
            padding="same",
        )

    def forward(
        self,
        xs: torch.Tensor,
    ):
        """Full-segment decode — REFERENCE ORACLE ONLY (unit tests).

        Production paths (both async-chunk and full-payload) go through
        forward_stream; this method is kept verbatim from the reference
        implementation so tests can diff the streaming port against it.
        """
        # Upsample
        xs = self.in_proj(xs)  # (b, t, upsample_rate, c)
        xs = xs.reshape(
            xs.shape[0], -1, self.qwen3_config.hidden_size
        )  # (b, t*upsample_rate, c)
        # Qwen3
        outs = self.qwen3(
            inputs_embeds=xs,
            attention_mask=None,
        )
        xs = outs.last_hidden_state  # (b, t*upsample_rate, c)
        audio = self.istft_head(xs)
        return audio

    def forward_stream(
        self,
        xs: torch.Tensor,
        cache: RedAEDecoderState = None,
        last_chunk: bool = False,
    ):
        """Incremental decode of one latent chunk.

        Args:
            xs: shape (b, t, c) — NEW latent frames only (25 Hz). A chunk is
                a pure increment: the Qwen3 KV cache carries the SWA(64)
                left context, the iSTFT cache the OLA overlap tail.
            cache: per-request RedAEDecoderState (None on first chunk).
            last_chunk: trim the final (win-hop) OLA tail — set on the
                request's terminal chunk so the waveform ends flush.
        Returns:
            (audio_chunk, new_cache): audio shape (b, samples).
        """
        if cache is None:
            # Pre-initialize the per-request cache: the stock DynamicCache
            # lazy init pays one pageable H2D copy per layer on the first
            # update, which stream-syncs on the decode stream.
            cache = RedAEDecoderState(
                build_initialized_cache(
                    self.qwen3_config,
                    device=xs.device,
                    dtype=self.in_proj.weight.dtype,
                )
            )
        else:
            pass
        # Upsample
        xs = self.in_proj(xs)  # (b, t, upsample_rate, c)
        xs = xs.reshape(
            xs.shape[0], -1, self.qwen3_config.hidden_size
        )  # (b, t*upsample_rate, c)
        # Forward Qwen3
        outs = self.qwen3.forward(
            inputs_embeds=xs,
            use_cache=True,
            past_key_values=cache.qwen3_cache,
        )
        xs = outs.last_hidden_state
        new_qwen3_cache = outs.past_key_values
        # iSTFT
        audio_chunk, new_istft_cache = self.istft_head.forward_stream(
            xs, cache.istft_cache, last_chunk
        )
        # Update cache
        new_cache = RedAEDecoderState(new_qwen3_cache, new_istft_cache)
        new_cache.frames_seen = cache.frames_seen + int(xs.shape[1])

        return audio_chunk, new_cache

    def flush_stream(self, cache: RedAEDecoderState) -> torch.Tensor:
        """Terminal OLA tail once the final chunk is already decoded.

        The Qwen3 stack cannot take a zero-frame input, but the tail samples
        depend only on the iSTFT cache, so the flush bypasses Qwen3 entirely.
        """
        if cache.istft_cache is None:
            raise RuntimeError("RedAE flush_stream requires a decoded chunk first")
        else:
            pass
        return self.istft_head.istft.flush_stream(cache.istft_cache)

    # Equal-progress batch merge/split (vocoder request batching).

    def merge_states(self, states: list[RedAEDecoderState]) -> RedAEDecoderState:
        """Merge equal-progress per-request states into one batched state.

        All states must sit at the same stream position: the sliding-window
        cache layers track a single cumulative_length each, so a merged cache
        can only describe requests that advanced in lockstep (the engine
        groups on it). The SWA(64) window forgets stale KV within two chunks,
        so merge/split round-trip drift stays bounded and never accumulates.
        """
        merged_cache = build_initialized_cache(
            self.qwen3_config,
            device=states[0].qwen3_cache.layers[0].keys.device,
            dtype=states[0].qwen3_cache.layers[0].keys.dtype,
        )
        for li, layer in enumerate(merged_cache.layers):
            first = states[0].qwen3_cache.layers[li]
            layer.keys = torch.cat(
                [s.qwen3_cache.layers[li].keys for s in states], dim=0
            )
            layer.values = torch.cat(
                [s.qwen3_cache.layers[li].values for s in states], dim=0
            )
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = first.cumulative_length
            else:
                pass
        merged = RedAEDecoderState(merged_cache)
        # iSTFT OLA tail: fixed (B, n_fft//2+1, win//hop-1), progress-free.
        merged.istft_cache = torch.cat([s.istft_cache for s in states], dim=0)
        merged.frames_seen = states[0].frames_seen
        return merged

    def split_state(
        self, state: RedAEDecoderState, batch_size: int
    ) -> list[RedAEDecoderState]:
        """Split a batched state back into per-request states.

        Rows are views (no copy): update() never writes in place — it
        concatenates and reassigns — and row i's view covers only its own
        slice of the batched storage, so cross-request corruption is
        impossible by construction.
        """
        states = []
        for row in range(batch_size):
            row_cache = DynamicCache(config=self.qwen3_config)
            for li, layer in enumerate(state.qwen3_cache.layers):
                row_layer = copy.copy(layer)
                row_layer.keys = layer.keys[row : row + 1]
                row_layer.values = layer.values[row : row + 1]
                row_cache.layers[li] = row_layer
            row_state = RedAEDecoderState(row_cache)
            row_state.istft_cache = state.istft_cache[row : row + 1]
            row_state.frames_seen = state.frames_seen
            states.append(row_state)
        return states


def rematerialize_istft_window(
    decoder: RedAEAudioDecoder, device: torch.device | str
) -> None:
    """Recompute the iSTFT Hann window after a meta-device materialization.

    istft_head.istft.window is a registered buffer computed in __init__; any
    load path that materializes the module from a meta state without reading
    the checkpoint key (or with an incomplete key set) leaves it as
    uninitialized memory, and the fold/envelope math then produces garbage
    audio. Recompute it exactly as __init__ does (the fishaudio
    rematerialize_audio_decoder_buffers pattern).
    """
    istft = decoder.istft_head.istft
    istft.window = torch.hann_window(istft.win_length, device=device)
