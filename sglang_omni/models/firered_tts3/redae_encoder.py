# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RedAE audio encoder (ICL prompt-audio latents for the talker).

Ported from the FireRedTTS3 reference implementation
(fireredtts3/redae/redae.py, RedAEAudioEncoder + Qwen3ClsDownsample). The
talker uses this to encode reference audio into prompt latents during
prefill preprocessing — one-shot, outside the decode hot loop. The matching
decoder lives in redae_decoder.py; both load from the checkpoint's redae/
weight subdirectory (encoder.* / decoder.* prefixes).

Backbone: HF transformers Qwen3Model (as in the reference), 18 layers
all-SWA(64) + a CLS-token 2x downsample block. The reference pins
flash_attention_2 on the encoder and runs encode under bf16 autocast; the
attention backend here follows the transformers default (sdpa), which the
production deployment parity notes already cover. The bf16 autocast wrapper
stays at the call site.
"""

import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3Model


class Qwen3ClsDownsample(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        downsample_rate: int = 2,
        # Qwen
        hidden_size: int = 896,
        intermediate_size: int = 896 * 4,
        num_hidden_layers: int = 4,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.downsample_rate = downsample_rate
        self.qwen3_config = Qwen3Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
        )
        self.qwen3 = Qwen3Model(self.qwen3_config)
        self.cls_tok = torch.nn.Parameter(torch.ones(1, 1, hidden_size))
        # Input & output proj
        self.in_proj = (
            torch.nn.Linear(in_dim, hidden_size)
            if in_dim != hidden_size
            else torch.nn.Identity()
        )
        self.out_proj = (
            torch.nn.Linear(hidden_size, out_dim)
            if out_dim != hidden_size
            else torch.nn.Identity()
        )

    def forward(self, xs: torch.Tensor):
        """Downsample latents as BERT [CLS] token.

        Args:
            xs(torch.Tensor): shape (b, t, c).
        Returns:
            ys(torch.Tensor): shape (b, t//downsample_rate, c).
        """
        assert xs.shape[1] % self.downsample_rate == 0, xs.shape
        b = xs.shape[0]
        # Patchify, (b, t, c) -> (b*t/down, down, c)
        xs = xs.reshape(-1, self.downsample_rate, self.in_dim)
        xs = self.in_proj(xs)
        cls_tok = self.cls_tok.expand(xs.shape[0], -1, -1)
        xs = torch.cat([xs, cls_tok], dim=1)  # (b*t/down, down+1, c)
        # Forward LLM
        ys = []
        for xs_chunk in torch.split(xs, 32768, dim=0):  # attention kernel limits
            outs_chunk = self.qwen3(inputs_embeds=xs_chunk)
            ys_chunk = outs_chunk.last_hidden_state[:, -1]  # (b*t/down, c)
            ys.append(ys_chunk)
        ys = torch.cat(ys, dim=0)
        ys = self.out_proj(ys)
        ys = ys.reshape(b, -1, self.out_dim)
        return ys


class RedAEAudioEncoder(torch.nn.Module):
    def __init__(
        self,
        # Output
        out_dim: int = 1024,
        # Input reshape
        audio_patch_size: int = 480,  # 50Hz
        audio_sample_rate: int = 24000,
        # Qwen
        hidden_size: int = 896,
        intermediate_size: int = 896 * 4,
        num_hidden_layers: int = 24,
        max_position_embeddings: int = 32768,
        max_window_layers: int = 0,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        sliding_window: int = 64,
        use_sliding_window: bool = True,
        # Extra downsample
        extra_downsample_rate: int = 2,  # 50Hz -> 25Hz
        downsample_num_hidden_layers: int = 4,
    ):
        super().__init__()
        self.audio_patch_size = audio_patch_size
        self.audio_sample_rate = audio_sample_rate
        self.extra_downsample_rate = extra_downsample_rate
        self.out_dim = out_dim
        self.in_proj = nn.Sequential(
            nn.Linear(self.audio_patch_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
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
        if self.extra_downsample_rate > 1:
            self.downsample = Qwen3ClsDownsample(
                in_dim=hidden_size,
                out_dim=hidden_size,
                downsample_rate=extra_downsample_rate,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_hidden_layers=downsample_num_hidden_layers,
                max_position_embeddings=max_position_embeddings,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            )
        self.out_proj = nn.Linear(hidden_size, out_dim)

    @property
    def downsample_rate(self):
        return int(self.audio_patch_size * self.extra_downsample_rate)

    @property
    def sample_rate(self):
        return self.audio_sample_rate

    @property
    def hidden_size(self):
        return self.out_dim

    def forward(self, audio: torch.Tensor):
        """
        Args:
            audio(torch.Tensor): shape (b, t)
        Returns:
            xs(torch.Tensor): shape (b, t//self.downsample_rate, self.hidden_size)
        """
        assert audio.shape[1] % self.downsample_rate == 0, audio.shape
        # Patchify
        xs = audio.unfold(
            dimension=1,
            size=self.audio_patch_size,
            step=self.audio_patch_size,
        )  # (b, num_patch, patch_size) ~ (b, t, c)
        # LLM
        xs = self.in_proj(xs)
        outs = self.qwen3(
            inputs_embeds=xs,
            attention_mask=None,
        )
        xs = outs.last_hidden_state  # (b, t, c)
        # Downsample
        if self.extra_downsample_rate > 1:
            xs = self.downsample(xs)
        xs = self.out_proj(xs)
        return xs
