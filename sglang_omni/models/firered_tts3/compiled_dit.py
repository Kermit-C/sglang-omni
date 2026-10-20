# SPDX-License-Identifier: Apache-2.0
"""Execute the original compiled DiT with optional external-operation caches."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

import torch
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode

from sglang_omni.models.firered_tts3.dit import DiT
from sglang_omni.models.firered_tts3.ffngemm import QKV_CONFIGS, FfnGemmMode

# note (Codex): each bucket uses separate Dynamo entries for both backends.
DIT_RECOMPILE_LIMIT = 64


class TimeCacheUnavailable(RuntimeError):
    """The caller must retry the complete flow using the original compiled DiT."""


def finite_byte_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    else:
        return bool(torch.isfinite(actual).all().item()) and torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )


class CompiledDiTExecutor:
    """Keep dispatch modes outside Dynamo tracing and the original graph intact."""

    def __init__(self, dit: DiT, original: torch.nn.Module | None = None) -> None:
        self.source = dit
        self.active_modes: ContextVar[tuple[TorchDispatchMode, ...]] = ContextVar(
            "firered_dit_dispatch_modes", default=()
        )
        self.original_compiled = (
            torch.compile(dit, dynamic=False, options={"epilogue_fusion": False})
            if original is None
            else original
        )
        self.observed = torch.compile(dit, backend=self.compile_backend, dynamic=False)

    def compile_backend(
        self, graph_module: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        compiled = torch._inductor.compile(
            graph_module, example_inputs, options={"epilogue_fusion": False}
        )

        def execute(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
            with ExitStack() as stack:
                for mode in self.active_modes.get():
                    stack.enter_context(mode)
                return compiled(*args)

        return execute

    @contextmanager
    def dispatch(self, modes: tuple[TorchDispatchMode, ...]) -> Iterator[None]:
        token = self.active_modes.set(modes)
        try:
            yield
        finally:
            self.active_modes.reset(token)

    def original(self, *, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch._dynamo.config.patch(
            recompile_limit=max(
                DIT_RECOMPILE_LIMIT, torch._dynamo.config.recompile_limit
            ),
            fail_on_recompile_limit_hit=True,
            suppress_errors=False,
        ):
            return self.original_compiled(x=x, t=t)

    def run(self, *, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        packed_mode = next(
            (
                mode
                for mode in self.active_modes.get()
                if isinstance(mode, FfnGemmMode)
                and x.shape[0] * x.shape[1] in QKV_CONFIGS
            ),
            None,
        )
        if packed_mode is None:
            observed = self.observed
        else:
            assert packed_mode.packed_compiled is not None
            observed = packed_mode.packed_compiled
        with torch._dynamo.config.patch(
            recompile_limit=max(
                DIT_RECOMPILE_LIMIT, torch._dynamo.config.recompile_limit
            ),
            fail_on_recompile_limit_hit=True,
            suppress_errors=False,
        ):
            return observed(x=x, t=t)

    def __call__(self, *, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.original(x=x, t=t)


@dataclass(frozen=True, kw_only=True)
class GemmSignature:
    input_shape: tuple[int, ...]
    input_stride: tuple[int, ...]
    weight_shape: tuple[int, ...]
    weight_stride: tuple[int, ...]
    bias_shape: tuple[int, ...]
    bias_stride: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True, kw_only=True)
class TimeGemmValue:
    signature: GemmSignature
    output: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class TimeGemmTable:
    """Graph entries must retain this table until their graphs are destroyed."""

    model_id: int
    n_timesteps: int
    steps: tuple[Mapping[str, TimeGemmValue], ...]

    @property
    def nbytes(self) -> int:
        return sum(
            value.output.numel() * value.output.element_size()
            for step in self.steps
            for value in step.values()
        )


@dataclass(kw_only=True)
class TimeGemmCall:
    index: int
    seen: set[str] = field(default_factory=set)


class TimeGemmMode(TorchDispatchMode):
    """Copy exact original addmm outputs into their original destination buffers."""

    def __init__(
        self,
        weights: Mapping[int, str],
        *,
        phase: Literal["collect", "cached"],
        n_timesteps: int,
        table: TimeGemmTable | None = None,
    ) -> None:
        super().__init__()
        if phase == "cached" and table is None:
            raise TimeCacheUnavailable(
                "Cached time execution requires a prepared table"
            )
        self.weights = weights
        self.phase = phase
        self.n_timesteps = n_timesteps
        self.table = table
        self.collected: list[dict[str, TimeGemmValue]] = [
            {} for _ in range(n_timesteps)
        ]
        self.active_call: ContextVar[TimeGemmCall | None] = ContextVar(
            "firered_time_gemm_call", default=None
        )

    @contextmanager
    def step(self, index: int) -> Iterator[None]:
        if self.active_call.get() is not None or not 0 <= index < self.n_timesteps:
            raise TimeCacheUnavailable(f"Invalid or nested time step {index}")
        call = TimeGemmCall(index=index)
        token = self.active_call.set(call)
        try:
            yield
            missing = set(self.weights.values()) - call.seen
            if missing:
                raise TimeCacheUnavailable(
                    f"Time step {index} missed compiled addmm operations: {sorted(missing)}"
                )
        finally:
            self.active_call.reset(token)

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        kwargs = kwargs or {}
        if func is not torch.ops.aten.addmm.out:
            return func(*args, **kwargs)
        bias, inputs, weight = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor], args
        )
        name = self.weights.get(weight.data_ptr())
        if name is None:
            return func(*args, **kwargs)
        call = self.active_call.get()
        if call is None or name in call.seen:
            raise TimeCacheUnavailable(f"Unscoped or repeated time addmm: {name}")
        if kwargs.get("alpha", 1) != 1 or kwargs.get("beta", 1) != 1:
            raise TimeCacheUnavailable(f"Unsupported time addmm scaling: {name}")
        call.seen.add(name)
        out = cast(torch.Tensor, kwargs["out"])
        signature = GemmSignature(
            input_shape=tuple(inputs.shape),
            input_stride=tuple(inputs.stride()),
            weight_shape=tuple(weight.shape),
            weight_stride=tuple(weight.stride()),
            bias_shape=tuple(bias.shape),
            bias_stride=tuple(bias.stride()),
            output_shape=tuple(out.shape),
            output_stride=tuple(out.stride()),
            dtype=out.dtype,
            device=out.device,
        )
        if self.phase == "collect":
            if name in self.collected[call.index]:
                raise TimeCacheUnavailable(f"Repeated time table entry: {name}")
            result = func(*args, **kwargs)
            self.collected[call.index][name] = TimeGemmValue(
                signature=signature,
                output=out.clone(memory_format=torch.preserve_format),
            )
            return result
        else:
            assert self.table is not None
            value = self.table.steps[call.index].get(name)
            if value is None or value.signature != signature:
                raise TimeCacheUnavailable(
                    f"Time table signature changed: step={call.index}, operation={name}"
                )
            out.copy_(value.output)
            return out


@dataclass(frozen=True, kw_only=True)
class TimeParameter:
    pointer: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    version: int | None


class TimeConditionCache:
    """Prepare tables owned by their active graph entries and dispatch modes."""

    def __init__(self, dit: DiT) -> None:
        self.source = dit
        weights = {
            module.weight.data_ptr(): name
            for name, module in dit.named_modules()
            if isinstance(module, torch.nn.Linear)
            and (name.startswith("t_embedder.") or ".adaLN_modulation." in name)
        }
        expected = 2 + len(dit.blocks) + 1
        if len(weights) != expected:
            raise TimeCacheUnavailable(
                f"Expected {expected} independent time Linear weights, found {len(weights)}"
            )
        self.weights = MappingProxyType(weights)
        self.parameters = {
            name: TimeParameter(
                pointer=parameter.data_ptr(),
                shape=tuple(parameter.shape),
                stride=tuple(parameter.stride()),
                dtype=parameter.dtype,
                device=parameter.device,
                version=None if torch.is_inference(parameter) else parameter._version,
            )
            for name, parameter in dit.named_parameters()
            if name.startswith("t_embedder.") or ".adaLN_modulation." in name
        }
        self.version_checks_available = all(
            parameter.version is not None for parameter in self.parameters.values()
        )

    def validate_weights(self) -> None:
        """Recreate the runner after weight changes, including unversioned writes."""
        current = dict(self.source.named_parameters())
        for name, expected in self.parameters.items():
            parameter = current.get(name)
            if parameter is None:
                raise TimeCacheUnavailable(f"Time parameter was removed: {name}")
            actual = TimeParameter(
                pointer=parameter.data_ptr(),
                shape=tuple(parameter.shape),
                stride=tuple(parameter.stride()),
                dtype=parameter.dtype,
                device=parameter.device,
                version=None if torch.is_inference(parameter) else parameter._version,
            )
            if actual != expected:
                raise TimeCacheUnavailable(
                    f"Time parameter changed; rebuild cache: {name}"
                )

    def prepare(
        self,
        executor: CompiledDiTExecutor,
        example_x: torch.Tensor,
        t_span: torch.Tensor,
        n_timesteps: int,
    ) -> TimeGemmTable:
        self.validate_weights()
        if executor.source is not self.source:
            raise TimeCacheUnavailable(
                "Time cache and executor have different model weights"
            )
        if (
            example_x.device.type != "cuda"
            or example_x.ndim != 3
            or example_x.shape[0] <= 0
            or example_x.shape[0] % 2
            or example_x.dtype not in (torch.float32, torch.bfloat16)
            or t_span.dtype is not torch.float32
            or t_span.ndim != 1
            or n_timesteps <= 0
            or t_span.numel() != n_timesteps + 1
            or t_span.device != example_x.device
        ):
            raise TimeCacheUnavailable(
                "Unsupported time-table input or schedule schema"
            )
        if torch.cuda.is_current_stream_capturing():
            raise TimeCacheUnavailable(
                "Time tables must be prepared before graph capture"
            )
        collector = TimeGemmMode(self.weights, phase="collect", n_timesteps=n_timesteps)
        batch_size = example_x.shape[0] // 2
        with torch.inference_mode():
            for index in range(n_timesteps):
                t = t_span[index].view(-1, 1, 1).expand(batch_size, -1, -1)
                t = torch.cat([t, t], dim=0).to(example_x.dtype)
                reference = executor.original(x=example_x, t=t)
                with collector.step(index), executor.dispatch((collector,)):
                    observed = executor.run(x=example_x, t=t)
                if not finite_byte_equal(observed, reference):
                    raise TimeCacheUnavailable(
                        f"Observed original graph changed output at time step {index}"
                    )
        table = TimeGemmTable(
            model_id=id(self.source),
            n_timesteps=n_timesteps,
            steps=tuple(MappingProxyType(dict(row)) for row in collector.collected),
        )
        mode = self.mode(table)
        with torch.inference_mode():
            for index in range(n_timesteps):
                t = t_span[index].view(-1, 1, 1).expand(batch_size, -1, -1)
                t = torch.cat([t, t], dim=0).to(example_x.dtype)
                reference = executor.original(x=example_x, t=t)
                with mode.step(index), executor.dispatch((mode,)):
                    cached = executor.run(x=example_x, t=t)
                if not finite_byte_equal(cached, reference):
                    raise TimeCacheUnavailable(
                        f"Cached original graph changed output at time step {index}"
                    )
        return table

    def mode(self, table: TimeGemmTable) -> TimeGemmMode:
        if table.model_id != id(self.source):
            raise TimeCacheUnavailable("Time table belongs to different model weights")
        else:
            return TimeGemmMode(
                self.weights, phase="cached", n_timesteps=table.n_timesteps, table=table
            )
