# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rotary position embedding with XPOS-style scaling.

Ported from the FireRedTTS3 reference implementation
(fireredtts3/llm/rotary_embedding.py). Used by the DiT flow head and the
patch encoder. forward and apply_rotary_pos_emb disable autocast explicitly:
RoPE stays fp32 even when the surrounding matmuls run in half precision.
"""

import torch
from einops import rearrange
from torch import arange, cat, stack
from torch.amp import autocast
from torch.nn import Module


class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        interpolation_factor=1.0,
        base=10000,
        base_rescale_factor=1.0,
    ):
        super().__init__()
        # NTK-aware base rescaling (bloc97) for longer sequence lengths.
        base *= base_rescale_factor ** (dim / (dim - 2))

        self.rope_dim = dim
        self.rope_base = base

        inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor

    def rope_init(self):
        dim, base = self.rope_dim, self.rope_base
        self.inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = arange(seq_len, device=device)
        return self.forward(t)

    @autocast("cuda", enabled=False)
    def forward(self, t, offset=0):
        if t.ndim == 1:
            t = rearrange(t, "n -> 1 n")
        else:
            pass

        freqs = (
            torch.einsum("b i , j -> b i j", t.type_as(self.inv_freq), self.inv_freq)
            / self.interpolation_factor
        )
        freqs = stack((freqs, freqs), dim=-1)
        freqs = rearrange(freqs, "... d r -> ... (d r)")

        return freqs, 1.0


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast("cuda", enabled=False)
def apply_rotary_pos_emb(t, freqs, scale=1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[:, -seq_len:, :]
    scale = scale[:, -seq_len:, :] if isinstance(scale, torch.Tensor) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, "b n d -> b 1 n d")
    else:
        pass

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = cat((t, t_unrotated), dim=-1)

    return out.type(orig_dtype)
