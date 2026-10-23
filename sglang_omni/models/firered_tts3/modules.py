# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention, normalization, and patch-encoder blocks for FireRedTTS3."""

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.firered_tts3.rotary_embedding import apply_rotary_pos_emb


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim: int = dim

    def forward(self, x: torch.Tensor, scale: float = 1000) -> torch.Tensor:
        device = x.device
        half_embedding_size = self.dim // 2
        position_embedding = math.log(10000) / (half_embedding_size - 1)
        position_embedding = torch.exp(
            torch.arange(half_embedding_size, device=device).float()
            * -position_embedding
        )
        position_embedding = scale * x.unsqueeze(1) * position_embedding.unsqueeze(0)
        position_embedding = torch.cat(
            (position_embedding.sin(), position_embedding.cos()), dim=-1
        )
        return position_embedding


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256) -> None:
        super().__init__()
        self.time_embed: SinusPositionEmbedding = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep_hidden_states = self.time_embed(timestep)
        timestep_hidden_states = timestep_hidden_states.to(timestep.dtype)
        timestep_embedding = self.time_mlp(timestep_hidden_states)
        return timestep_embedding


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps: float = eps
        self.weight: nn.Parameter = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        else:
            pass
        return F.rms_norm(
            x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps
        )


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 4,
        dropout: float = 0.0,
        approximate: Literal["none", "tanh"] = "none",
    ) -> None:
        super().__init__()
        projection_size = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        activation = nn.GELU(approximate=approximate)
        input_projection = nn.Sequential(nn.Linear(dim, projection_size), activation)
        self.ff: nn.Sequential = nn.Sequential(
            input_projection, nn.Dropout(dropout), nn.Linear(projection_size, dim_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class Attention(nn.Module):
    def __init__(
        self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.dim: int = dim
        self.heads: int = heads
        self.inner_dim: int = dim_head * heads
        self.dropout: float = dropout
        self.to_q: nn.Linear = nn.Linear(dim, self.inner_dim)
        self.to_k: nn.Linear = nn.Linear(dim, self.inner_dim)
        self.to_v: nn.Linear = nn.Linear(dim, self.inner_dim)
        self.to_out: nn.ModuleList = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

    def project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.to_q(x), self.to_k(x), self.to_v(x))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: tuple[torch.Tensor, float | torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        query, key, value = self.project_qkv(x)
        projection_size = key.shape[-1]
        head_size = projection_size // self.heads
        query = query.view(batch_size, -1, self.heads, head_size).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_size).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_size).transpose(1, 2)
        if rope is not None:
            position_frequencies, position_scale = rope
            query_position_scale, key_position_scale = (
                (position_scale, position_scale ** (-1.0))
                if position_scale is not None
                else (1.0, 1.0)
            )
            query = apply_rotary_pos_emb(
                query, position_frequencies, query_position_scale
            )
            key = apply_rotary_pos_emb(key, position_frequencies, key_position_scale)
        else:
            pass
        x = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        x = x.transpose(1, 2).reshape(batch_size, -1, self.heads * head_size)
        x = x.to(query.dtype)
        x = self.to_out[0](x)
        x = self.to_out[1](x)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)
        else:
            pass
        return x


class DiTBlock(nn.Module):
    """Patch-encoder block: attention + FFN only (no conv, no AdaLN)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1: RMSNorm = RMSNorm(hidden_size, eps=1e-06)
        self.attn: Attention = Attention(
            dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            dropout=dropout,
        )
        self.norm2: RMSNorm = RMSNorm(hidden_size, eps=1e-06)
        self.mlp: FeedForward = FeedForward(
            dim=hidden_size, mult=mlp_ratio, dropout=dropout, approximate="tanh"
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        rope: tuple[torch.Tensor, float | torch.Tensor | None],
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask, rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """Patch-encoder output layer: RMSNorm + linear (no AdaLN)."""

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final: RMSNorm = RMSNorm(hidden_size, eps=1e-06)
        self.linear: nn.Linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_final(x)
        x = self.linear(x)
        return x
