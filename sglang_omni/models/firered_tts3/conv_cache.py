# SPDX-License-Identifier: Apache-2.0
"""Inference-only convolution weight caching after the original graph is compiled."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode

from sglang_omni.models.firered_tts3.conv_plan import (
    INPUT_SHAPE,
    WEIGHT_SHAPE,
    ControlledConvPlan,
)


@dataclass(frozen=True)
class CachedConvWeight:
    name: str
    source: torch.Tensor
    packed: torch.Tensor
    version: int | None


class Conv1dWeightCache(TorchDispatchMode):
    """Keep the original compiled graph and replace its external convolution calls.

    Construct after final weight loading, device placement, and dtype conversion.
    Retain the cache until captured graphs are destroyed; recreate after weights change.
    """

    def __init__(
        self, module: torch.nn.Module, *, use_controlled_plan: bool = False
    ) -> None:
        super().__init__()
        self.module = module
        self.use_controlled_plan = use_controlled_plan
        self.weights: dict[int, CachedConvWeight] = {}
        for name, layer in module.named_modules():
            if isinstance(layer, torch.nn.Conv1d):
                weight = layer.weight
                source = weight.detach()
                self.weights[weight.data_ptr()] = CachedConvWeight(
                    name=f"{name}.weight" if name else "weight",
                    source=source,
                    packed=source.unsqueeze(2).contiguous(
                        memory_format=torch.channels_last
                    ),
                    version=None if torch.is_inference(weight) else weight._version,
                )
        self.version_checks_available = all(
            entry.version is not None for entry in self.weights.values()
        )
        self.controlled_plan: ControlledConvPlan | None = None
        self.hits = 0
        self.misses = 0

    def validate_weights(self) -> None:
        """Check weights before capture; live weight updates require a new worker."""
        for pointer, entry in self.weights.items():
            current = self.module.get_parameter(entry.name)
            version = None if torch.is_inference(current) else current._version
            if (
                current.data_ptr() != pointer
                or current.shape != entry.source.shape
                or current.stride() != entry.source.stride()
                or current.dtype != entry.source.dtype
                or current.device != entry.source.device
                or version != entry.version
            ):
                raise RuntimeError(
                    f"Convolution weight changed; rebuild cache: {entry.name}"
                )

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if func not in (
            torch.ops.aten.convolution.default,
            torch.ops.aten.conv1d.default,
        ):
            return func(*args, **(kwargs or {}))
        if func is torch.ops.aten.conv1d.default:
            names = (
                "input",
                "weight",
                "bias",
                "stride",
                "padding",
                "dilation",
                "groups",
            )
        else:
            names = (
                "input",
                "weight",
                "bias",
                "stride",
                "padding",
                "dilation",
                "transposed",
                "output_padding",
                "groups",
            )
        fields: dict[str, object] = {
            "bias": None,
            "stride": [1],
            "padding": [0],
            "dilation": [1],
            "transposed": False,
            "output_padding": [0],
            "groups": 1,
        }
        fields.update(zip(names, args))
        fields.update(kwargs or {})
        inputs = cast(torch.Tensor, fields["input"])
        weight = cast(torch.Tensor, fields["weight"])
        entry = self.weights.get(weight.data_ptr())
        if entry is None:
            self.misses += 1
            return func(*args, **(kwargs or {}))
        expected = entry.source if weight.ndim == 3 else entry.source.unsqueeze(2)
        if (
            weight.shape != expected.shape
            or weight.dtype != expected.dtype
            or weight.device != expected.device
            or any(
                size != 1 and got != want
                for size, got, want in zip(
                    weight.shape, weight.stride(), expected.stride()
                )
            )
        ):
            self.misses += 1
            return func(*args, **(kwargs or {}))
        if fields["transposed"] or any(cast(Sequence[int], fields["output_padding"])):
            self.misses += 1
            return func(*args, **(kwargs or {}))
        stride = list(cast(Sequence[int], fields["stride"]))
        padding = list(cast(Sequence[int], fields["padding"]))
        dilation = list(cast(Sequence[int], fields["dilation"]))
        if inputs.ndim == weight.ndim == 3:
            inputs = inputs.unsqueeze(2)
            stride, padding, dilation = [1, *stride], [0, *padding], [1, *dilation]
            squeeze = True
        elif (
            inputs.ndim == weight.ndim == 4
            and inputs.shape[2] == weight.shape[2] == 1
            and stride[0] == dilation[0] == 1
            and padding[0] == 0
        ):
            squeeze = False
        else:
            self.misses += 1
            return func(*args, **(kwargs or {}))
        self.hits += 1
        inputs = inputs.contiguous(memory_format=torch.channels_last)
        if (
            self.use_controlled_plan
            and inputs.device.type == "cuda"
            and inputs.dtype == torch.bfloat16
            and tuple(inputs.shape) == INPUT_SHAPE
            and tuple(entry.packed.shape) == WEIGHT_SHAPE
            and fields["bias"] is None
            and stride == dilation == [1, 1]
            and padding == [0, 1]
            and fields["groups"] == 1
        ):
            # note (Codex): preserve the NCHW reduction plan for this NHWC shape.
            if self.controlled_plan is None:
                self.controlled_plan = ControlledConvPlan(inputs, entry.packed)
            result = self.controlled_plan(inputs, entry.packed)
        else:
            result = torch.ops.aten.convolution.default(
                inputs,
                entry.packed,
                fields["bias"],
                stride,
                padding,
                dilation,
                False,
                [0, 0],
                fields["groups"],
            )
        return (result.squeeze(2) if squeeze else result).contiguous()
