# SPDX-License-Identifier: Apache-2.0
"""Selected DiT GEMMs and a private compiled model with packed QKV projections."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode

from sglang_omni.models.firered_tts3.modules import Attention
from sglang_omni.models.firered_tts3.packed_qkv import PackedQkvModel

try:
    from sglang_omni.models.firered_tts3.ffn_gemm_kernel import gemm_out
except ImportError:
    gemm_out = None

# note (Codex): measured SM120 tiles; small FFN1 and M192 QKV use K128 rounding.
FFN_CONFIGS = {
    24: (16, 64, 64, 4, 3),
    48: (32, 64, 64, 4, 3),
    96: (32, 128, 64, 4, 3),
}
QKV_CONFIGS = {192: (64, 128, 64, 4, 3), 384: (64, 64, 64, 4, 3)}
FFN_LAYERS = 11
CompileBackend = Callable[
    [torch.fx.GraphModule, list[torch.Tensor]], Callable[..., tuple[torch.Tensor, ...]]
]


class GemmOutput(Protocol):
    def __call__(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        config: tuple[int, int, int, int, int],
    ) -> torch.Tensor: ...


@dataclass(frozen=True, kw_only=True)
class FfnWeight:
    name: str
    source: torch.Tensor
    bias: torch.Tensor
    bias_pointer: int
    device: torch.device
    output_size: int
    version: int | None
    bias_version: int | None

    @classmethod
    def from_linear(cls, name: str, layer: torch.nn.Linear) -> FfnWeight:
        weight, bias = layer.weight, layer.bias
        assert bias is not None
        return cls(
            name=f"{name}.weight",
            source=weight,
            bias=bias,
            bias_pointer=bias.data_ptr(),
            device=weight.device,
            output_size=weight.shape[0],
            version=(
                None if torch.is_inference(weight) else weight._version
            ),  # noqa: SLF001  # leading-underscore: PyTorch tensor version API
            bias_version=(
                None if torch.is_inference(bias) else bias._version
            ),  # noqa: SLF001  # leading-underscore: PyTorch tensor version API
        )


class FfnGemmMode(TorchDispatchMode):
    """Own small FFN1 and large packed-QKV resources for their CUDA graph lifetime.

    Inference weights are immutable: rebuild after every weight update, including
    in-place updates of inference tensors without a version counter.
    """

    def __init__(self, source: torch.nn.Module) -> None:
        super().__init__()
        self.source = source
        self.weights: dict[int, FfnWeight] = {}
        self.packed_weights: dict[int, FfnWeight] = {}
        self.packed_qkv: PackedQkvModel | None = None
        self.packed_compiled: torch.nn.Module | None = None
        self.hits = 0
        self.misses = 0
        self.enabled = False
        self.reason = "DiT GEMM optimization requires eleven CUDA BF16 Linear layers"
        self.gemm_out: GemmOutput | None = None
        layers = [
            (name, layer)
            for name, layer in source.named_modules()
            if isinstance(layer, torch.nn.Linear) and name.endswith(".mlp.ff.0.0")
        ]
        if len(layers) != FFN_LAYERS or any(
            layer.weight.shape != (3072, 1024)
            or layer.bias is None
            or layer.bias.shape != (3072,)
            or not layer.weight.is_contiguous()
            or not layer.bias.is_contiguous()
            or not layer.weight.is_cuda
            or layer.weight.dtype != torch.bfloat16
            or layer.bias.dtype != torch.bfloat16
            or layer.weight.device != layer.bias.device
            for _, layer in layers
        ):
            return
        else:
            device = layers[0][1].weight.device
            if any(layer.weight.device != device for _, layer in layers):
                return
            elif torch.cuda.get_device_capability(device) != (12, 0):
                self.reason = "DiT GEMM optimization requires SM120"
                return
            elif not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
                self.reason = (
                    "DiT GEMM optimization requires BF16 reduced precision reduction"
                )
                return
            else:
                if gemm_out is None:
                    raise RuntimeError("DiT GEMM optimization requires Triton")
                else:
                    pass
                self.gemm_out = gemm_out
                self.weights = {
                    layer.weight.data_ptr(): FfnWeight.from_linear(name, layer)
                    for name, layer in layers
                }
                if len(self.weights) != FFN_LAYERS:
                    self.reason = (
                        "DiT GEMM optimization requires independent Linear weights"
                    )
                    return
                else:
                    self.enabled = True
                    self.reason = ""

    def prepare_packed(
        self, compile_backend: CompileBackend, device: torch.device
    ) -> None:
        """Prepare large-bucket QKV before capture without changing the small FFN path."""
        if self.packed_compiled is not None:
            return
        elif not self.enabled:
            raise RuntimeError(self.reason)
        else:
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Packed QKV must be prepared before capture")
                else:
                    pass
            self.validate_weights()
            attentions = [
                (name, layer)
                for name, layer in self.source.named_modules()
                if name.endswith(".attn")
            ]
            if len(attentions) != FFN_LAYERS or any(
                type(layer) is not Attention for _, layer in attentions
            ):
                raise RuntimeError(
                    "Packed QKV requires eleven original DiT Attention layers"
                )
            else:
                pass
            for name, attention in attentions:
                for role in ("to_q", "to_k", "to_v"):
                    layer = attention.get_submodule(role)
                    if (
                        layer.weight.shape != (1024, 1024)
                        or layer.bias is None
                        or layer.bias.shape != (1024,)
                        or any(
                            t.dtype != torch.bfloat16
                            or t.device != device
                            or not t.is_contiguous()
                            for t in (layer.weight, layer.bias)
                        )
                    ):
                        raise RuntimeError(
                            f"Unsupported packed QKV parameters: {name}.{role}"
                        )
                    else:
                        pass
            packed_qkv = PackedQkvModel(self.source)
            packed_weights = {
                packed.to_qkv.weight.data_ptr(): FfnWeight.from_linear(
                    f"{name}.to_qkv", packed.to_qkv
                )
                for name, packed in packed_qkv.layers.items()
            }
            packed_compiled = torch.compile(
                packed_qkv.model, backend=compile_backend, dynamic=False
            )
            self.packed_qkv = packed_qkv
            self.packed_weights = packed_weights
            self.packed_compiled = packed_compiled

    def supports(self, rows: int) -> bool:
        return self.enabled and (
            rows in FFN_CONFIGS
            or (rows in QKV_CONFIGS and self.packed_compiled is not None)
        )

    def expected_hits(self, rows: int) -> int:
        if rows in FFN_CONFIGS:
            return len(self.weights)
        elif rows in QKV_CONFIGS:
            return len(self.packed_weights)
        else:
            return 0

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Called once per graph replay stream by the owning graph entry."""
        for entry in self.weights.values():
            entry.source.record_stream(stream)
            entry.bias.record_stream(stream)
        if self.packed_qkv is not None:
            self.packed_qkv.record_stream(stream)
        else:
            pass

    def validate_weights(self) -> None:
        """Validate cached parameters before capturing a new graph."""
        if self.packed_qkv is not None:
            self.packed_qkv.validate_weights()
        else:
            pass
        for pointer, entry in self.weights.items():
            current = self.source.get_parameter(entry.name)
            bias = self.source.get_parameter(entry.name.removesuffix("weight") + "bias")
            version = (
                None if torch.is_inference(current) else current._version
            )  # noqa: SLF001  # leading-underscore: PyTorch tensor version API
            bias_version = (
                None if torch.is_inference(bias) else bias._version
            )  # noqa: SLF001  # leading-underscore: PyTorch tensor version API
            if (
                current is not entry.source
                or bias is not entry.bias
                or current.data_ptr() != pointer
                or current.shape != (entry.output_size, 1024)
                or current.stride() != (1024, 1)
                or bias.data_ptr() != entry.bias_pointer
                or bias.shape != (entry.output_size,)
                or bias.stride() != (1,)
                or current.dtype != torch.bfloat16
                or bias.dtype != torch.bfloat16
                or current.device != entry.device
                or bias.device != entry.device
                or version != entry.version
                or bias_version != entry.bias_version
            ):
                raise RuntimeError(
                    f"DiT GEMM weights changed; rebuild mode: {entry.name}"
                )
            else:
                pass

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        kwargs = kwargs or {}
        if not self.enabled or func is not torch.ops.aten.addmm.out:
            return func(*args, **kwargs)
        else:
            fields: dict[str, object] = {"alpha": 1, "beta": 1}
            fields.update(zip(("self", "mat1", "mat2"), args))
            fields.update(kwargs)
            weight = fields.get("mat2")
            pointer = weight.data_ptr() if isinstance(weight, torch.Tensor) else None
            entry = self.weights.get(pointer)
            if entry is None:
                entry = self.packed_weights.get(pointer)
                configs = QKV_CONFIGS
            else:
                configs = FFN_CONFIGS
            if entry is None:
                return func(*args, **kwargs)
            elif len(args) > 3 or set(fields) != {
                "self",
                "mat1",
                "mat2",
                "out",
                "alpha",
                "beta",
            }:
                self.misses += 1
                return func(*args, **kwargs)
            else:
                inputs, bias, out = (fields[name] for name in ("mat1", "self", "out"))
                if not all(
                    isinstance(tensor, torch.Tensor) for tensor in (inputs, bias, out)
                ):
                    self.misses += 1
                    return func(*args, **kwargs)
                else:
                    inputs, bias, out = cast(
                        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        (inputs, bias, out),
                    )
                    weight = cast(torch.Tensor, weight)
                    if (
                        fields["alpha"] != 1
                        or fields["beta"] != 1
                        or inputs.ndim != 2
                        or inputs.shape[0] not in configs
                        or inputs.shape[1] != 1024
                        or not inputs.is_contiguous()
                        or weight.shape != (1024, 3072)
                        or weight.stride() != (1, 1024)
                        or bias.shape != (3072,)
                        or bias.stride() != (1,)
                        or bias.data_ptr() != entry.bias_pointer
                        or out.shape != (inputs.shape[0], 3072)
                        or not out.is_contiguous()
                        or any(
                            t.dtype != torch.bfloat16
                            or t.device != entry.device
                            or t.is_neg()
                            or t.is_conj()
                            for t in (inputs, weight, bias, out)
                        )
                        or (
                            torch.is_grad_enabled()
                            and any(
                                t.requires_grad for t in (inputs, weight, bias, out)
                            )
                        )
                    ):
                        self.misses += 1
                        return func(*args, **kwargs)
                    else:
                        start, end = out.data_ptr(), out.data_ptr() + out.numel() * 2
                        if any(
                            max(start, t.data_ptr())
                            < min(end, t.data_ptr() + t.numel() * 2)
                            for t in (inputs, weight, bias)
                        ):
                            self.misses += 1
                            return func(*args, **kwargs)
                        else:
                            assert self.gemm_out is not None
                            result = self.gemm_out(
                                inputs, weight, bias, out, configs[inputs.shape[0]]
                            )
                            self.hits += 1
                            return result
