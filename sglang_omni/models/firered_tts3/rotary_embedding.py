# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rotary position embeddings for the DiT and patch encoder."""

import torch
from einops import rearrange
from torch import arange, cat, stack
from torch.amp import autocast
from torch.nn import Module


class RotaryEmbedding(Module):
    inv_freq: torch.Tensor

    def __init__(
        self,
        dim: int,
        interpolation_factor: float = 1.0,
        base: float = 10000,
        base_rescale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        base *= base_rescale_factor ** (dim / (dim - 2))
        inv_freq = 1.0 / base ** (arange(0, dim, 2).float() / dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        assert interpolation_factor >= 1.0
        self.interpolation_factor: float = interpolation_factor

    def forward_from_seq_len(
        self, position_count: int
    ) -> tuple[torch.Tensor, float | torch.Tensor | None]:
        device = self.inv_freq.device
        t = arange(position_count, device=device)
        return self.forward(t)

    @autocast("cuda", enabled=False)
    def forward(
        self, t: torch.Tensor
    ) -> tuple[torch.Tensor, float | torch.Tensor | None]:
        if t.ndim == 1:
            t = rearrange(t, "n -> 1 n")
        else:
            pass
        position_frequencies = (
            torch.einsum("b i , j -> b i j", t.type_as(self.inv_freq), self.inv_freq)
            / self.interpolation_factor
        )
        position_frequencies = stack(
            (position_frequencies, position_frequencies), dim=-1
        )
        position_frequencies = rearrange(position_frequencies, "... d r -> ... (d r)")
        return (position_frequencies, 1.0)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_pos_emb(
    t: torch.Tensor, position_frequencies: torch.Tensor, scale: float | torch.Tensor = 1
) -> torch.Tensor:
    rotary_size, position_count, original_dtype = (
        position_frequencies.shape[-1],
        t.shape[-2],
        t.dtype,
    )
    position_frequencies = position_frequencies[:, -position_count:, :]
    scale = scale[:, -position_count:, :] if isinstance(scale, torch.Tensor) else scale
    if t.ndim == 4 and position_frequencies.ndim == 3:
        position_frequencies = rearrange(position_frequencies, "b n d -> b 1 n d")
    else:
        pass
    t, t_unrotated = (t[..., :rotary_size], t[..., rotary_size:])
    t = (
        t * position_frequencies.cos() * scale
        + rotate_half(t) * position_frequencies.sin() * scale
    )
    out = cat((t, t_unrotated), dim=-1)
    return out.type(original_dtype)
