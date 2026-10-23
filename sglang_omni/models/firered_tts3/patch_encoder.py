# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encode each latent patch into one backbone input embedding."""

import torch

from sglang_omni.models.firered_tts3.modules import DiTBlock, FinalLayer
from sglang_omni.models.firered_tts3.rotary_embedding import RotaryEmbedding


class PatchEncoder(torch.nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        patch_size: int = 4,
        hidden_size: int = 1024,
        mlp_ratio: int = 3,
        depth: int = 8,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.in_dim: int = in_dim
        self.patch_size: int = patch_size
        self.hidden_size: int = hidden_size
        self.out_dim: int = out_dim
        self.cls_tok: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros(1, 1, hidden_size)
        )
        self.rotary_embed: RotaryEmbedding = RotaryEmbedding(hidden_size // num_heads)
        self.blocks: torch.nn.ModuleList = torch.nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.in_proj: torch.nn.Linear | torch.nn.Identity = (
            torch.nn.Linear(in_dim, hidden_size)
            if in_dim != hidden_size
            else torch.nn.Identity()
        )
        self.out_proj: FinalLayer = FinalLayer(hidden_size, out_dim)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Patch encoder aggregating patch_size latents into one."""
        assert (
            inputs_embeds.shape[1] % self.patch_size == 0
        ), f"inputs_embeds.shape={inputs_embeds.shape} patch_size={self.patch_size}"
        inputs_embeds = self.in_proj(inputs_embeds)
        hidden_states = inputs_embeds.reshape(-1, self.patch_size, self.hidden_size)
        cls_tok = self.cls_tok.expand(hidden_states.shape[0], -1, -1)
        hidden_states = torch.cat([cls_tok, hidden_states], dim=1)
        rope = self.rotary_embed.forward_from_seq_len(hidden_states.shape[1])
        for block in self.blocks:
            hidden_states = block(hidden_states, None, rope)
        hidden_states = self.out_proj(hidden_states)
        hidden_states = hidden_states[:, 0]
        hidden_states = hidden_states.unsqueeze(0)
        return hidden_states
