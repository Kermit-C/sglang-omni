# SPDX-License-Identifier: Apache-2.0
"""A private DiT module tree with shared source tensors and packed QKV weights."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from itertools import chain

import torch

from sglang_omni.models.firered_tts3.modules import Attention


class PackedAttention(Attention):
    """Reuse attention's post-projection operations without changing its source."""

    def __init__(self, source: Attention) -> None:
        torch.nn.Module.__init__(self)
        self.dim, self.inner_dim = source.dim, source.inner_dim
        self.heads, self.dropout = source.heads, source.dropout
        self.training = source.training
        self.to_q, self.to_k, self.to_v = source.to_q, source.to_k, source.to_v
        self.to_out = source.to_out
        self.to_qkv = torch.nn.Linear(self.dim, 3 * self.inner_dim, device="meta")
        projections = (self.to_q, self.to_k, self.to_v)
        self.to_qkv.weight = torch.nn.Parameter(
            torch.cat([layer.weight.detach() for layer in projections]),
            requires_grad=False,
        )
        self.to_qkv.bias = torch.nn.Parameter(
            torch.cat([layer.bias.detach() for layer in projections]),
            requires_grad=False,
        )
        self.to_qkv.training = source.training

    def project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.to_qkv(x).chunk(3, dim=-1)


@dataclass(frozen=True, kw_only=True)
class TensorBinding:
    value: torch.Tensor
    pointer: int
    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    version: int | None


class PackedQkvModel:
    """Share immutable model tensors and validate copied QKV parameters before capture."""

    def __init__(self, source: torch.nn.Module) -> None:
        self.source = source
        original = dict(chain(source.named_parameters(), source.named_buffers()))
        self.model = copy.deepcopy(source, memo={id(t): t for t in original.values()})
        self.layers: dict[str, PackedAttention] = {}
        for name, attention in list(self.model.named_modules()):
            if name.endswith(".attn"):
                if type(attention) is not Attention:
                    raise RuntimeError(f"Unsupported packed attention: {name}")
                else:
                    packed = PackedAttention(attention)
                    self.model.set_submodule(name, packed)
                    self.layers[name] = packed
            else:
                pass
        candidate = dict(
            chain(self.model.named_parameters(), self.model.named_buffers())
        )
        if not self.layers or any(
            candidate.get(name) is not value for name, value in original.items()
        ):
            raise RuntimeError(
                "Packed DiT must preserve all original parameter and buffer bindings"
            )
        else:
            pass
        self.projections = {
            f"{name}.{role}.{parameter}": TensorBinding(
                value=value,
                pointer=value.data_ptr(),
                shape=value.shape,
                stride=value.stride(),
                dtype=value.dtype,
                device=value.device,
                version=(
                    None if torch.is_inference(value) else value._version
                ),  # noqa: SLF001  # leading-underscore: PyTorch tensor version API
            )
            for name, layer in self.layers.items()
            for role in ("to_q", "to_k", "to_v")
            for parameter, value in layer.get_submodule(role).named_parameters()
        }
        self.tensors = tuple(chain(self.model.parameters(), self.model.buffers()))

    def validate_weights(self) -> None:
        """Reject changes to source projections copied into the packed weights."""
        for name, binding in self.projections.items():
            current = self.source.get_parameter(name)
            version = (
                None if torch.is_inference(current) else current._version
            )  # noqa: SLF001  # leading-underscore: PyTorch tensor version API
            if (
                current is not binding.value
                or current.data_ptr() != binding.pointer
                or current.shape != binding.shape
                or current.stride() != binding.stride
                or current.dtype != binding.dtype
                or current.device != binding.device
                or version != binding.version
            ):
                raise RuntimeError(
                    f"Packed DiT projection changed; rebuild mode: {name}"
                )
            else:
                pass

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        for tensor in self.tensors:
            tensor.record_stream(stream)
