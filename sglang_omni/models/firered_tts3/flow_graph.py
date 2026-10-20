# SPDX-License-Identifier: Apache-2.0
"""Self-captured CUDA graphs over the whole DiT Euler flow loop.

The per-patch flow step is n_timesteps Euler iterations over a tiny CFG
batch of 2 x (history+patch) x channels tokens; the loop is kernel-launch
bound, so replaying one captured graph of the entire loop recovers the
launch overhead while every fp32 red line (integration state, noise, CFG
blend, t_span) is untouched.

Semantics:

* euler_flow_loop is the single source shared by the eager path and the
  captured graph, so the two can never drift (B=1 is bit-exact, verified
  against the eager loop on exact bucket hits);
* the noise x0 is drawn OUTSIDE the graph from the per-request generator
  and copied in, keeping the deterministic draw order;
* the CFG scale rides a [B, 1, 1] fp32 placeholder so per-task scales
  (2.0 / 1.2 / per-request overrides) share one graph without recapture;
* bucket capture sizes: a batch below the next bucket is zero-padded up to
  it (batch rows are independent — attention runs over tokens, never
  across rows — so padded rows are computed and dropped), letting one
  graph cover a range of batch sizes; a batch above every bucket still
  gets its own whole graph (never split into rounds);
* capture failure disables the runner permanently (eager fallback), and
  replay is skipped while another stream capture is in progress.
"""

from __future__ import annotations

import copy
import logging
import os
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang_omni.models.firered_tts3.compiled_dit import (
    CompiledDiTExecutor,
    TimeCacheUnavailable,
    TimeConditionCache,
    TimeGemmMode,
    TimeGemmTable,
    finite_byte_equal,
)

logger = logging.getLogger(__name__)

# One graph per listed batch size; anything between buckets zero-pads up.
DEFAULT_CAPTURE_SIZES = (1, 2, 4, 8, 16)

# Eager iterations settling cuDNN/cuBLAS autotuning and lazy inits before
# each capture.
WARMUP_ITERATIONS = 3

# The DiT execution stack inside the flow loop. fp32 is the bit-exact
# reference; the bf16 modes lower only the DiT matmul body (the Euler state,
# noise, CFG blend, and t_span stay fp32); the compile modes wrap the stack
# in torch.compile. The enum is a single knob deliberately: the four stacks
# were measured as a matrix and the uncovered boolean combinations are
# measured anti-optimizations.
FLOW_MODES = ("fp32", "compile_fp32", "bf16", "compile_bf16")
FlowMode = Literal["fp32", "compile_fp32", "bf16", "compile_bf16"]


def flow_dit_dtype(mode: str) -> torch.dtype:
    if mode not in FLOW_MODES:
        raise ValueError(f"unknown flow mode {mode!r}; expected one of {FLOW_MODES}")
    return torch.bfloat16 if mode.endswith("bf16") else torch.float32


def flow_optimizations(mode: FlowMode) -> bool:
    default = "1" if mode.startswith("compile") else "0"
    time_cache = os.environ.get("FIRERED_TTS3_PRECOMPUTE_TIME", default)
    if time_cache not in ("0", "1"):
        raise ValueError("FIRERED_TTS3_PRECOMPUTE_TIME must be 0 or 1")
    return time_cache == "1"


def prepare_flow_dit(
    dit: torch.nn.Module, mode: FlowMode
) -> tuple[torch.nn.Module | CompiledDiTExecutor, torch.nn.Module]:
    """Build the mode's DiT execution stack; returns (run, fallback).

    run is what the flow loop calls; fallback is the same stack without the
    torch.compile wrapper (the same object for the eager modes) so a compile
    failure degrades to the same dtype instead of losing the mode. bf16
    modes deep-copy before casting: nn.Module.to() is in place, and the
    caller's fp32 DiT must stay fp32 for the eager fallback path.

    The compile recipe (default mode, dynamic=False, epilogue_fusion=False)
    specializes per batch shape and keeps inductor from fusing RMSNorm/RoPE
    epilogues in ways that lose fp32 precision; it never enables compile's
    own cudagraphs, which would fight the outer flow-graph capture.
    """
    if mode not in FLOW_MODES:
        raise ValueError(f"unknown flow mode {mode!r}; expected one of {FLOW_MODES}")
    time_cache = flow_optimizations(mode)
    if time_cache and not mode.startswith("compile"):
        raise ValueError("Flow time caches require a compile flow mode")
    if mode == "fp32":
        return dit, dit
    if mode.endswith("bf16"):
        dit = copy.deepcopy(dit).to(dtype=torch.bfloat16)
    if mode.startswith("compile"):
        if time_cache:
            return CompiledDiTExecutor(dit), dit
        return (
            torch.compile(dit, dynamic=False, options={"epilogue_fusion": False}),
            dit,
        )
    return dit, dit


def flow_t_span(n_timesteps: int, device: torch.device) -> torch.Tensor:
    """The cosine t_span schedule (fp32 red line), n_timesteps + 1 knots."""
    t = torch.linspace(0.0, 1.0, n_timesteps + 1, device=device, dtype=torch.float32)
    return 1.0 - torch.cos(t * 0.5 * torch.pi)


def euler_flow_loop(
    dit: torch.nn.Module | CompiledDiTExecutor,
    xt: torch.Tensor,
    cond: torch.Tensor,
    t_span: torch.Tensor,
    cfg: torch.Tensor,
    n_timesteps: int,
    patch_size: int,
    *,
    dit_dtype: torch.dtype = torch.float32,
    executor: CompiledDiTExecutor | None = None,
    dispatch_modes: tuple[TorchDispatchMode, ...] = (),
    time_mode: TimeGemmMode | None = None,
) -> torch.Tensor:
    """The Euler flow integration over xt (in place; returns the patch tail).

    xt is [B, history+patch, D], cond is [B, history+patch, C], and cfg a
    [B, 1, 1] fp32 per-row CFG scale (a zero scale rides the CFG-doubled
    shape and blends to the conditional velocity exactly). Shared by the
    eager path and the captured graph. dit_dtype != fp32 casts at the DiT
    call site only — the integration state, CFG blend, and t_span stay fp32
    (precision red line).
    """
    bsz = xt.shape[0]
    for ti in range(n_timesteps):
        t = t_span[ti]
        dt = t_span[ti + 1] - t
        x_in = torch.cat([xt, cond], dim=2)
        t_in = t.view(-1, 1, 1).expand(bsz, -1, -1)
        x_in_cfg = torch.cat([xt, cond * 0], dim=2)
        x_in = torch.cat([x_in, x_in_cfg], dim=0)
        t_in = torch.cat([t_in, t_in], dim=0)
        with (
            executor.dispatch(dispatch_modes) if executor is not None else nullcontext()
        ), (time_mode.step(ti) if time_mode is not None else nullcontext()):
            run_dit = executor.run if executor is not None else dit
            if dit_dtype is torch.float32:
                vt = run_dit(x=x_in, t=t_in)
            else:
                vt = run_dit(x=x_in.to(dit_dtype), t=t_in.to(dit_dtype)).float()
        vt_cond, vt_uncond = vt.chunk(2, dim=0)
        vt = (1.0 + cfg) * vt_cond - cfg * vt_uncond
        xt[:, -patch_size:] = (
            xt[:, -patch_size:] + dt.view(-1, 1, 1) * vt[:, -patch_size:]
        )
    return xt[:, -patch_size:]


@dataclass
class FlowGraphEntry:
    graph: torch.cuda.CUDAGraph
    # Static input; the graph updates its tail in place per replay.
    xt: torch.Tensor  # [B, history+patch, redae_dim]
    cond: torch.Tensor  # [B, history+patch, cond_channels]
    cfg: torch.Tensor  # [B, 1, 1] fp32 per-row CFG scale
    time_table: TimeGemmTable | None = None
    dispatch_modes: tuple[TorchDispatchMode, ...] = ()
    t_span: torch.Tensor | None = None
    replay_streams: dict[tuple[int, int], torch.cuda.Stream] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def record_replay_stream(self, stream: torch.cuda.Stream) -> None:
        """Register cached tensors once per replay stream; this does not serialize use."""
        key = (stream.device.index, stream.cuda_stream)
        if key in self.replay_streams:
            return
        for tensor in (self.xt, self.cond, self.cfg):
            tensor.record_stream(stream)
        if self.t_span is not None:
            self.t_span.record_stream(stream)
        if self.time_table is not None:
            for step in self.time_table.steps:
                for value in step.values():
                    value.output.record_stream(stream)
        # note (Codex): replay bypasses Python dispatch, so retain each registered stream.
        self.replay_streams[key] = stream


class FlowGraphRunner:
    """Replay the batched Euler flow loop via per-bucket CUDA graphs.

    One graph per shape key (device, bucket, cond channels) with LRU
    eviction bounded by max_graphs; the owner sizes max_graphs so the
    pre-captured buckets never evict.

    mode selects the DiT execution stack (see prepare_flow_dit; the caller
    builds the stack at the compile_model hook so the compile wrap lands
    before any graph capture). dit is the execution module; dit_fallback is
    the same stack without the compile wrapper — a capture failure of the
    compiled stack degrades to the same dtype (retry once), and only a
    second failure disables the graphs (the caller's eager loop then runs
    the fp32 reference DiT).
    """

    def __init__(
        self,
        dit: torch.nn.Module | CompiledDiTExecutor,
        t_span: torch.Tensor,
        *,
        n_timesteps: int,
        patch_size: int,
        history_length: int,
        max_graphs: int = 8,
        capture_sizes: list[int] | tuple[int, ...] | None = None,
        mode: FlowMode = "fp32",
        dit_fallback: torch.nn.Module | None = None,
        precompute_time_conditions: bool | None = None,
    ) -> None:
        self.mode = mode
        self.dit = dit
        self.dit_dtype = flow_dit_dtype(mode)
        self.dit_fallback = dit_fallback if dit_fallback is not None else dit
        time_cache = flow_optimizations(mode)
        self.precompute_time_conditions = (
            time_cache
            if precompute_time_conditions is None
            else precompute_time_conditions
        )
        self.optimize = self.precompute_time_conditions
        if self.optimize and not mode.startswith("compile"):
            raise ValueError("Flow time caches require a compile flow mode")
        if isinstance(dit, CompiledDiTExecutor):
            self.executor = dit
        elif self.optimize:
            self.executor = CompiledDiTExecutor(self.dit_fallback, original=dit)
            self.dit = self.executor
        else:
            self.executor = None
        self.time_cache: TimeConditionCache | None = None
        self.t_span = t_span.clone() if self.optimize else t_span
        self.n_timesteps = int(n_timesteps)
        self.patch_size = int(patch_size)
        self.history_length = int(history_length)
        self.capture_sizes = (
            tuple(sorted({int(s) for s in capture_sizes if int(s) > 0})) or None
            if capture_sizes
            else None
        )
        self.max_graphs = max(int(max_graphs), len(self.capture_sizes or ()))
        self.enabled = True
        self.graph_pool = torch.cuda.graph_pool_handle()
        # note: capture must run on a stream no other graph family captures
        # on: the cuBLAS workspace is cached per (handle, capture stream), so
        # two graphs captured on one stream (e.g. the torch.cuda.graph default
        # capture stream) bake the same workspace address into their kernels,
        # and concurrent replays then race on the split-k scratch — a replayed
        # GEMM can wait on a clobbered semaphore forever (device-side hang
        # observed under load). A dedicated stream per runner keeps the baked
        # workspaces distinct.
        self.capture_stream: torch.cuda.Stream | None = None
        self.cache: OrderedDict[tuple, FlowGraphEntry] = OrderedDict()
        self.counters = {
            "calls": 0,
            "hits": 0,
            "captures": 0,
            "eager": 0,
            "padded": 0,
            "optimized_captures": 0,
            "optimization_fallbacks": 0,
        }

    def stats(self) -> dict[str, object]:
        return {
            **self.counters,
            "cache_size": len(self.cache),
            "precompute_time_conditions": self.precompute_time_conditions,
            "optimization_enabled": self.optimize,
        }

    def run(
        self,
        hist_latents: torch.Tensor,  # [B, history, D]
        x0: torch.Tensor,  # [B, patch, D]
        cond: torch.Tensor,  # [B, history+patch, C]
        cfgs: torch.Tensor,  # [B] or [B, 1, 1] fp32
        n_timesteps: int,
    ) -> torch.Tensor | None:
        """One batched flow step via graph replay; None requests eager."""
        self.counters["calls"] += 1
        if (
            not self.enabled
            or int(n_timesteps) != self.n_timesteps
            or hist_latents.device.type != "cuda"
            or torch.cuda.is_current_stream_capturing()
        ):
            self.counters["eager"] += 1
            return None
        bsz = hist_latents.shape[0]
        cfgs = cfgs.reshape(bsz, 1, 1)
        target = bsz
        if self.capture_sizes:
            target = next((s for s in self.capture_sizes if s >= bsz), bsz)
        if target != bsz:
            pad = target - bsz
            hist_latents = self.pad_batch(hist_latents, pad)
            x0 = self.pad_batch(x0, pad)
            cond = self.pad_batch(cond, pad)
            cfgs = self.pad_batch(cfgs, pad)
            self.counters["padded"] += 1
        key = (hist_latents.device.index, target, tuple(cond.shape))
        entry = self.cache.get(key)
        if entry is None:
            entry = self.capture(hist_latents, x0, cond)
            if entry is None and self.dit is not self.dit_fallback:
                # The compiled stack failed to compile or capture; degrade
                # to the same dtype without torch.compile (permanently —
                # later buckets share the conclusion) before giving up on
                # the graphs entirely.
                logger.warning(
                    f"FireRedTTS3 flow mode {self.mode}: capture with the "
                    "compiled DiT failed; retrying without torch.compile"
                )
                self.dit = self.dit_fallback
                entry = self.capture(hist_latents, x0, cond)
            if entry is None:
                self.enabled = False
                self.counters["eager"] += 1
                return None
            self.counters["captures"] += 1
            self.cache[key] = entry
            while len(self.cache) > self.max_graphs:
                _, evicted = self.cache.popitem(last=False)
                del evicted
        else:
            self.cache.move_to_end(key)
            self.counters["hits"] += 1
        entry.xt[:, : self.history_length].copy_(hist_latents)
        entry.xt[:, self.history_length :].copy_(x0)
        entry.cond.copy_(cond)
        entry.cfg.copy_(cfgs)
        if entry.dispatch_modes:
            entry.record_replay_stream(torch.cuda.current_stream(entry.xt.device))
        entry.graph.replay()
        # The tail aliases the static buffer the next replay overwrites.
        return entry.xt[:, -self.patch_size :][:bsz].clone()

    @staticmethod
    def pad_batch(t: torch.Tensor, pad: int) -> torch.Tensor:
        return torch.cat([t, t.new_zeros(pad, *t.shape[1:])], dim=0)

    def capture(
        self,
        hist_latents: torch.Tensor,
        x0: torch.Tensor,
        cond: torch.Tensor,
    ) -> FlowGraphEntry | None:
        """Capture one whole-loop graph for the inputs' shape; None on failure."""
        # Placeholders must never alias the caller's tensors: contiguous()
        # returns self for an already-contiguous input, and the next copy_
        # into the placeholder would then overlap its own source.
        xt = torch.cat([hist_latents, x0], dim=1).contiguous().clone()
        cond = cond.contiguous().clone()
        cfg = torch.zeros(xt.shape[0], 1, 1, device=xt.device, dtype=torch.float32)

        executor = self.executor if self.dit is not self.dit_fallback else None
        time_table = None
        time_mode = None
        modes: tuple[TorchDispatchMode, ...] = ()

        def body(
            state: torch.Tensor,
            condition: torch.Tensor,
            scales: torch.Tensor,
            active_modes: tuple[TorchDispatchMode, ...],
        ) -> torch.Tensor:
            return euler_flow_loop(
                self.dit,
                state,
                condition,
                self.t_span,
                scales,
                self.n_timesteps,
                self.patch_size,
                dit_dtype=self.dit_dtype,
                executor=executor if active_modes else None,
                dispatch_modes=active_modes,
                time_mode=time_mode if active_modes else None,
            )

        def capture_graph(
            active_modes: tuple[TorchDispatchMode, ...]
        ) -> torch.cuda.CUDAGraph:
            for _ in range(WARMUP_ITERATIONS):
                body(xt, cond, cfg, active_modes)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            if self.capture_stream is None:
                self.capture_stream = torch.cuda.Stream(device=xt.device)
            with torch.cuda.graph(
                graph,
                pool=self.graph_pool,
                stream=self.capture_stream,
                capture_error_mode="thread_local",
            ):
                body(xt, cond, cfg, active_modes)
            return graph

        with torch.inference_mode():
            if self.optimize and executor is not None:
                try:
                    if self.precompute_time_conditions:
                        if self.time_cache is None:
                            self.time_cache = TimeConditionCache(executor.source)
                        x_in = torch.cat([xt, cond], dim=2)
                        x_uncond = torch.cat([xt, cond * 0], dim=2)
                        example = torch.cat([x_in, x_uncond], dim=0).to(self.dit_dtype)
                        time_table = self.time_cache.prepare(
                            executor, example, self.t_span, self.n_timesteps
                        )
                        time_mode = self.time_cache.mode(time_table)
                        modes += (time_mode,)
                    generator = torch.Generator(device=xt.device).manual_seed(41729)
                    for probe in range(3):
                        if probe == 0:
                            probe_xt = torch.zeros_like(xt)
                            probe_cond = torch.zeros_like(cond)
                            probe_cfg = cfg
                        else:
                            probe_xt = torch.randn(
                                xt.shape,
                                device=xt.device,
                                dtype=xt.dtype,
                                generator=generator,
                            )
                            probe_cond = torch.randn(
                                cond.shape,
                                device=cond.device,
                                dtype=cond.dtype,
                                generator=generator,
                            )
                            probe_cfg = torch.full_like(cfg, 1.2 if probe == 1 else 2.0)
                        expected = body(probe_xt.clone(), probe_cond, probe_cfg, ())
                        actual = body(probe_xt.clone(), probe_cond, probe_cfg, modes)
                        if not finite_byte_equal(actual, expected):
                            raise TimeCacheUnavailable(
                                f"Complete Flow exact gate failed for probe {probe}"
                            )
                    graph = capture_graph(modes)
                    self.counters["optimized_captures"] += 1
                except Exception:
                    logger.warning(
                        f"Flow optimization rejected for bucket={xt.shape[0]}; "
                        "retrying the original compiled DiT",
                        exc_info=True,
                    )
                    self.counters["optimization_fallbacks"] += 1
                    modes, time_table = (), None
                    try:
                        graph = capture_graph(())
                    except Exception:
                        logger.warning(
                            "Original compiled Flow capture failed", exc_info=True
                        )
                        return None
            else:
                try:
                    graph = capture_graph(())
                except Exception:
                    logger.warning(
                        f"FireRedTTS3 flow CUDA graph capture failed for xt={tuple(xt.shape)} "
                        f"cond={tuple(cond.shape)}",
                        exc_info=True,
                    )
                    return None
        logger.info(
            f"Captured FireRedTTS3 flow: xt={tuple(xt.shape)} cond={tuple(cond.shape)} "
            f"steps={self.n_timesteps} optimized={bool(modes)}"
        )
        return FlowGraphEntry(
            graph=graph,
            xt=xt,
            cond=cond,
            cfg=cfg,
            time_table=time_table,
            dispatch_modes=modes,
            t_span=self.t_span,
        )


__all__ = [
    "DEFAULT_CAPTURE_SIZES",
    "FLOW_MODES",
    "FlowGraphEntry",
    "FlowGraphRunner",
    "FlowMode",
    "euler_flow_loop",
    "flow_dit_dtype",
    "flow_t_span",
    "prepare_flow_dit",
]
