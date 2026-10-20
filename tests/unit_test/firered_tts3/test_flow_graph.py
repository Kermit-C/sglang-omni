# SPDX-License-Identifier: Apache-2.0
"""Flow-loop CUDA graph tests: graph/eager bit-exactness and fallbacks.

The graph path must be bit-exact against the shared eager loop on exact
bucket hits (B=1 is the production shape), share one graph across CFG scales
(the [B, 1, 1] placeholder), zero-pad batches up to the next bucket, and
degrade to the eager loop on capture failure / foreign stream capture /
n_timesteps mismatch. The integration test drives the runner's full
flow_step (cond window -> dit_head -> DiT loop -> patch_encoder feedback)
with and without graphs and expects torch.equal on every emitted patch.

The four-mode execution stack adds: mode validation, bf16 deepcopy
isolation (the caller's fp32 DiT is the eager fallback and must never be
mutated), bf16 graph vs bf16 eager bit-exactness, the compile-failure
degradation to the same-dtype uncompiled stack, and a compile+capture smoke
test (the compile tree must tolerate the outer flow graph capture).
"""

from __future__ import annotations

import collections
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import sglang_omni.models.firered_tts3.flow_graph as flow_graph_module
from sglang_omni.models.firered_tts3.compiled_dit import (
    DIT_RECOMPILE_LIMIT,
    CompiledDiTExecutor,
    TimeCacheUnavailable,
    TimeConditionCache,
    TimeGemmMode,
    TimeGemmTable,
    finite_byte_equal,
)
from sglang_omni.models.firered_tts3.dit import DiT
from sglang_omni.models.firered_tts3.flow_graph import (
    FLOW_MODES,
    FlowGraphRunner,
    euler_flow_loop,
    flow_t_span,
    prepare_flow_dit,
)
from sglang_omni.models.firered_tts3.model_runner import (
    FireRedTTS3ModelRunner,
    FlowState,
)
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.models.firered_tts3.weights import load_dit

from .conftest import CHECKPOINTS, requires_checkpoints
from .test_flow_step_oracle import FakeModel

BASE_CHECKPOINT = CHECKPOINTS["base"]

HISTORY_LENGTH = 8
PATCH_SIZE = 4
REDAE_DIM = 64
COND_CHANNELS = 32
N_TIMESTEPS = 10


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


@pytest.fixture
def tiny_dit(device) -> DiT:
    torch.manual_seed(0)
    return (
        DiT(
            in_channels=REDAE_DIM + COND_CHANNELS,
            out_channels=REDAE_DIM,
            mlp_ratio=2.0,
            depth=2,
            num_heads=4,
            hidden_size=64,
        )
        .to(device=device, dtype=torch.float32)
        .eval()
    )


def tiny_runner(dit: DiT, device, **overrides) -> FlowGraphRunner:
    kwargs = {
        "n_timesteps": N_TIMESTEPS,
        "patch_size": PATCH_SIZE,
        "history_length": HISTORY_LENGTH,
        "capture_sizes": (1, 2, 4),
    }
    kwargs.update(overrides)
    return FlowGraphRunner(dit, flow_t_span(N_TIMESTEPS, device), **kwargs)


def make_inputs(device, generator: torch.Generator, bsz: int):
    hist = torch.randn(
        bsz, HISTORY_LENGTH, REDAE_DIM, device=device, generator=generator
    )
    x0 = torch.randn(bsz, PATCH_SIZE, REDAE_DIM, device=device, generator=generator)
    cond = torch.randn(
        bsz,
        HISTORY_LENGTH + PATCH_SIZE,
        COND_CHANNELS,
        device=device,
        generator=generator,
    )
    return hist, x0, cond


def eager_patch(dit, hist, x0, cond, cfg, device) -> torch.Tensor:
    return euler_flow_loop(
        dit,
        torch.cat([hist, x0], dim=1),
        cond,
        flow_t_span(N_TIMESTEPS, device),
        cfg,
        N_TIMESTEPS,
        PATCH_SIZE,
    )


@pytest.mark.accelerator
def test_graph_replay_bit_exact_eager(tiny_dit, device) -> None:
    runner = tiny_runner(tiny_dit, device)
    generator = torch.Generator(device=device).manual_seed(7)
    cfg = torch.full((1, 1, 1), 2.0, device=device)
    for _ in range(3):
        hist, x0, cond = make_inputs(device, generator, 1)
        with torch.inference_mode():
            got = runner.run(hist, x0, cond, cfg, N_TIMESTEPS)
            want = eager_patch(tiny_dit, hist, x0, cond, cfg, device)
        assert got is not None
        assert torch.equal(got, want)
    stats = runner.stats()
    assert stats["calls"] == 3 and stats["captures"] == 1 and stats["hits"] == 2


@pytest.mark.accelerator
def test_cfg_placeholder_shares_one_graph(tiny_dit, device) -> None:
    runner = tiny_runner(tiny_dit, device)
    generator = torch.Generator(device=device).manual_seed(11)
    hist, x0, cond = make_inputs(device, generator, 1)
    for scale in (2.0, 1.2, 0.0):
        cfg = torch.full((1, 1, 1), scale, device=device)
        with torch.inference_mode():
            got = runner.run(hist, x0, cond, cfg, N_TIMESTEPS)
            want = eager_patch(tiny_dit, hist, x0, cond, cfg, device)
        assert torch.equal(got, want), f"cfg={scale} diverged"
    assert runner.stats()["captures"] == 1


@pytest.mark.accelerator
def test_bucket_exact_hit_bit_exact(tiny_dit, device) -> None:
    runner = tiny_runner(tiny_dit, device)
    generator = torch.Generator(device=device).manual_seed(13)
    hist, x0, cond = make_inputs(device, generator, 2)
    cfg = torch.full((2, 1, 1), 2.0, device=device)
    with torch.inference_mode():
        got = runner.run(hist, x0, cond, cfg, N_TIMESTEPS)
        want = eager_patch(tiny_dit, hist, x0, cond, cfg, device)
    assert torch.equal(got, want)
    assert runner.stats()["padded"] == 0


@pytest.mark.accelerator
def test_bucket_zero_pad_up(tiny_dit, device) -> None:
    runner = tiny_runner(tiny_dit, device)
    generator = torch.Generator(device=device).manual_seed(17)
    hist, x0, cond = make_inputs(device, generator, 3)
    cfg = torch.full((3, 1, 1), 2.0, device=device)
    with torch.inference_mode():
        got = runner.run(hist, x0, cond, cfg, N_TIMESTEPS)
        want = eager_patch(tiny_dit, hist, x0, cond, cfg, device)
    assert got.shape == (3, PATCH_SIZE, REDAE_DIM)
    # Cross-batch GEMM tiling keeps the padded replay off the bit-exact
    # contract; the drift must stay at fp32 roundoff scale.
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
    stats = runner.stats()
    assert stats["padded"] == 1 and stats["captures"] == 1


@pytest.mark.accelerator
def test_capture_failure_disables_runner(device) -> None:
    class FailingDiT(torch.nn.Module):
        def forward(self, x, t):
            raise RuntimeError("capture probe")

    runner = tiny_runner(FailingDiT(), device)
    generator = torch.Generator(device=device).manual_seed(19)
    hist, x0, cond = make_inputs(device, generator, 1)
    cfg = torch.full((1, 1, 1), 2.0, device=device)
    assert runner.run(hist, x0, cond, cfg, N_TIMESTEPS) is None
    assert not runner.enabled
    assert runner.run(hist, x0, cond, cfg, N_TIMESTEPS) is None
    stats = runner.stats()
    assert stats["eager"] == 2 and stats["captures"] == 0 and stats["calls"] == 2


@pytest.mark.accelerator
def test_foreign_stream_capture_falls_back(tiny_dit, device, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    runner = tiny_runner(tiny_dit, device)
    generator = torch.Generator(device=device).manual_seed(23)
    hist, x0, cond = make_inputs(device, generator, 1)
    cfg = torch.full((1, 1, 1), 2.0, device=device)
    assert runner.run(hist, x0, cond, cfg, N_TIMESTEPS) is None
    stats = runner.stats()
    assert stats["eager"] == 1 and stats["captures"] == 0


@pytest.mark.accelerator
def test_n_timesteps_mismatch_falls_back(tiny_dit, device) -> None:
    runner = tiny_runner(tiny_dit, device)
    generator = torch.Generator(device=device).manual_seed(29)
    hist, x0, cond = make_inputs(device, generator, 1)
    cfg = torch.full((1, 1, 1), 2.0, device=device)
    assert runner.run(hist, x0, cond, cfg, N_TIMESTEPS + 5) is None
    assert runner.stats()["eager"] == 1


@pytest.mark.accelerator
def test_lru_eviction(tiny_dit, device) -> None:
    # max_graphs is floored at len(capture_sizes) so configured buckets never
    # evict; over-bucket batch sizes capture lazily and DO evict.
    runner = tiny_runner(tiny_dit, device, max_graphs=1, capture_sizes=(1,))
    generator = torch.Generator(device=device).manual_seed(31)
    for bsz in (1, 3, 1, 3):
        hist, x0, cond = make_inputs(device, generator, bsz)
        cfg = torch.full((bsz, 1, 1), 2.0, device=device)
        assert runner.run(hist, x0, cond, cfg, N_TIMESTEPS) is not None
    stats = runner.stats()
    assert stats["cache_size"] == 1 and stats["captures"] == 4


@requires_checkpoints
@pytest.mark.accelerator
def test_flow_step_graph_matches_eager(device) -> None:
    """The runner's full flow_step with graphs is bit-exact against eager."""
    from sglang_omni.models.firered_tts3.config import load_checkpoint_hf_config

    config = load_checkpoint_hf_config(str(BASE_CHECKPOINT))
    assert config is not None
    model = FakeModel(config, device)
    steps = 8
    generator = torch.Generator(device=device).manual_seed(2024)
    prompt_latents = torch.randn(1, 24, 64, device=device, generator=generator)
    spk_emb = torch.randn(1, 512, device=device, generator=generator)
    hidden_rows = [
        torch.randn(
            24 if step == 0 else 1,
            config.hidden_size,
            device=device,
            generator=generator,
        )
        for step in range(steps)
    ]

    def make_data() -> SimpleNamespace:
        return SimpleNamespace(
            flow_state=FlowState(
                generator=torch.Generator(device=device).manual_seed(1234),
                prompt_patch_count=prompt_latents.shape[1] // config.patch_size,
                spk_cond_dit=model.spk_proj_dit(spk_emb.float()),
                backbone_cond=prompt_latents.new_zeros(
                    1, config.num_history_patches, config.hidden_size
                ),
                latents_gen=F.pad(prompt_latents, (0, 0, 2 * config.patch_size, 0)),
                patch_count=0,
            ),
            # sigmoid never reaches 1.1: the stop head (outside the graphed
            # region) stays open so random hidden rows cannot flake the
            # comparison on the decision boundary.
            state=FireRedTTS3State(
                stop_threshold=1.1, min_patches=6, n_timesteps=10, inference_cfg=2.0
            ),
            pending_feedback_queue=collections.deque(),
            latent_patches=[],
            req=SimpleNamespace(finished_reason=None),
            probe=ProfilingProbe(enabled=False),
        )

    runner = object.__new__(FireRedTTS3ModelRunner)
    runner.model = model
    runner.t_span_cache = {}
    eager_data = make_data()
    for hidden in hidden_rows:
        runner.flow_step(eager_data, hidden)

    graphs = FlowGraphRunner(
        model.dit,
        flow_t_span(10, device),
        n_timesteps=10,
        patch_size=config.patch_size,
        history_length=model.history_length,
        capture_sizes=(1,),
    )
    runner.flow_graphs = graphs
    graph_data = make_data()
    for hidden in hidden_rows:
        runner.flow_step(graph_data, hidden)

    assert len(graph_data.latent_patches) == steps
    for index, (got, want) in enumerate(
        zip(graph_data.latent_patches, eager_data.latent_patches)
    ):
        assert torch.equal(got, want), f"patch {index} diverged"
    for got, want in zip(
        graph_data.pending_feedback_queue, eager_data.pending_feedback_queue
    ):
        assert torch.equal(got, want)
    assert torch.equal(
        graph_data.flow_state.latents_gen, eager_data.flow_state.latents_gen
    )
    stats = graphs.stats()
    assert stats["calls"] == steps and stats["captures"] == 1
    assert stats["hits"] == steps - 1 and stats["eager"] == 0


@requires_checkpoints
@pytest.mark.accelerator
def test_real_dit_graph_bit_exact_and_bucket_pad(device) -> None:
    """The production DiT shape: B=1 bit-exact; B=3 padded to 4 within roundoff."""
    from sglang_omni.models.firered_tts3.config import load_checkpoint_hf_config

    config = load_checkpoint_hf_config(str(BASE_CHECKPOINT))
    assert config is not None
    dit = load_dit(str(BASE_CHECKPOINT), config, device=device)
    history = int(config.num_history_patches) * int(config.patch_size)
    cond_channels = int(config.dit_in_channels) - int(config.redae_dim)
    runner = FlowGraphRunner(
        dit,
        flow_t_span(10, device),
        n_timesteps=10,
        patch_size=int(config.patch_size),
        history_length=history,
        capture_sizes=(1, 2, 4),
    )
    generator = torch.Generator(device=device).manual_seed(37)

    def real_inputs(bsz: int):
        hist = torch.randn(
            bsz, history, config.redae_dim, device=device, generator=generator
        )
        x0 = torch.randn(
            bsz, config.patch_size, config.redae_dim, device=device, generator=generator
        )
        cond = torch.randn(
            bsz,
            history + config.patch_size,
            cond_channels,
            device=device,
            generator=generator,
        )
        return hist, x0, cond

    def eager(hist, x0, cond, cfg):
        return euler_flow_loop(
            dit,
            torch.cat([hist, x0], dim=1),
            cond,
            flow_t_span(10, device),
            cfg,
            10,
            int(config.patch_size),
        )

    hist, x0, cond = real_inputs(1)
    cfg = torch.full((1, 1, 1), 2.0, device=device)
    with torch.inference_mode():
        got = runner.run(hist, x0, cond, cfg, 10)
        want = eager(hist, x0, cond, cfg)
    assert torch.equal(got, want)

    hist, x0, cond = real_inputs(3)
    cfg = torch.full((3, 1, 1), 2.0, device=device)
    with torch.inference_mode():
        got = runner.run(hist, x0, cond, cfg, 10)
        want = eager(hist, x0, cond, cfg)
    assert got.shape == (3, int(config.patch_size), int(config.redae_dim))
    # Cross-batch GEMM tiling keeps the padded replay off the bit-exact
    # contract; measured drift ~2e-3 sits inside the ~5e-3 batched-GEMM
    # roundoff envelope.
    torch.testing.assert_close(got, want, rtol=1e-2, atol=5e-3)
    assert runner.stats()["padded"] == 1


def tiny_cpu_dit() -> DiT:
    torch.manual_seed(0)
    return (
        DiT(
            in_channels=REDAE_DIM + COND_CHANNELS,
            out_channels=REDAE_DIM,
            mlp_ratio=2.0,
            depth=2,
            num_heads=4,
            hidden_size=64,
        )
        .to(dtype=torch.float32)
        .eval()
    )


def test_flow_mode_validation() -> None:
    dit = tiny_cpu_dit()
    with pytest.raises(ValueError, match="unknown flow mode"):
        prepare_flow_dit(dit, "compile-bf16")
    with pytest.raises(ValueError, match="unknown flow mode"):
        FlowGraphRunner(
            dit,
            flow_t_span(N_TIMESTEPS, torch.device("cpu")),
            n_timesteps=N_TIMESTEPS,
            patch_size=PATCH_SIZE,
            history_length=HISTORY_LENGTH,
            mode="fp16",
        )
    from sglang_omni.models.firered_tts3.engine_builder import FireRedTTS3EngineBuilder

    with pytest.raises(ValueError, match="flow_mode"):
        FireRedTTS3EngineBuilder(flow_mode="fast")
    assert set(FLOW_MODES) == {"fp32", "compile_fp32", "bf16", "compile_bf16"}


def test_flow_mode_defaults() -> None:
    """The default is compile_bf16 at both the builder and the
    talker factory; the Instruct deploy yaml pins fp32 over it."""
    import inspect

    import sglang_omni.models.firered_tts3.stages as stages
    from sglang_omni.models.firered_tts3.engine_builder import FireRedTTS3EngineBuilder

    assert (
        inspect.signature(FireRedTTS3EngineBuilder.__init__)
        .parameters["flow_mode"]
        .default
        == "compile_bf16"
    )
    assert (
        inspect.signature(stages.create_talker_executor).parameters["flow_mode"].default
        == "compile_bf16"
    )


def test_prepare_flow_dit_bf16_deepcopy_isolation() -> None:
    source = tiny_cpu_dit()
    source_weight = next(source.parameters())
    original = source_weight.detach().clone()
    run_dit, fallback_dit = prepare_flow_dit(source, "bf16")
    assert run_dit is fallback_dit and run_dit is not source
    assert next(run_dit.parameters()).dtype == torch.bfloat16
    # The caller's fp32 DiT (the eager reference path) is never mutated.
    assert source_weight.dtype == torch.float32
    assert torch.equal(source_weight.detach(), original)
    with torch.no_grad():
        source_weight.add_(1.0)
    assert not torch.equal(next(run_dit.parameters()).float(), source_weight)


def test_prepare_flow_dit_compile_fp32_shares_the_source() -> None:
    source = tiny_cpu_dit()
    run_dit, fallback_dit = prepare_flow_dit(source, "compile_fp32")
    # The compile wrapper shares the source's fp32 parameters (no copy), so
    # the eager fallback path keeps serving the same fp32 weights.
    assert fallback_dit is source
    assert next(run_dit.parameters()).dtype == torch.float32
    assert next(run_dit.parameters()).data_ptr() == next(source.parameters()).data_ptr()


@pytest.mark.accelerator
def test_bf16_mode_graph_bit_exact_vs_bf16_eager(tiny_dit, device) -> None:
    run_dit, _ = prepare_flow_dit(tiny_dit, "bf16")
    runner = tiny_runner(run_dit, device, mode="bf16")
    generator = torch.Generator(device=device).manual_seed(41)
    hist, x0, cond = make_inputs(device, generator, 1)
    cfg = torch.full((1, 1, 1), 2.0, device=device)
    with torch.inference_mode():
        got = runner.run(hist, x0, cond, cfg, N_TIMESTEPS)
        want = euler_flow_loop(
            run_dit,
            torch.cat([hist, x0], dim=1),
            cond,
            flow_t_span(N_TIMESTEPS, device),
            cfg,
            N_TIMESTEPS,
            PATCH_SIZE,
            dit_dtype=torch.bfloat16,
        )
    assert got is not None
    # Same stack, exact bucket hit: graph replay is bit-exact vs the eager
    # loop, and the blend/integration state stayed fp32.
    assert got.dtype == torch.float32
    assert torch.equal(got, want)
    assert next(tiny_dit.parameters()).dtype == torch.float32


@pytest.mark.accelerator
def test_compile_failure_degrades_to_uncompiled_stack(tiny_dit, device) -> None:
    class BrokenCompiledDiT(torch.nn.Module):
        def __init__(self, inner: DiT) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x, t):
            raise RuntimeError("inductor probe")

    _, fallback_dit = prepare_flow_dit(tiny_dit, "compile_fp32")
    runner = tiny_runner(
        BrokenCompiledDiT(tiny_dit),
        device,
        mode="compile_fp32",
        dit_fallback=fallback_dit,
    )
    generator = torch.Generator(device=device).manual_seed(43)
    hist, x0, cond = make_inputs(device, generator, 1)
    cfg = torch.full((1, 1, 1), 2.0, device=device)
    with torch.inference_mode():
        got = runner.run(hist, x0, cond, cfg, N_TIMESTEPS)
        want = eager_patch(tiny_dit, hist, x0, cond, cfg, device)
    assert got is not None
    assert runner.dit is runner.dit_fallback
    assert torch.equal(got, want)
    stats = runner.stats()
    assert stats["captures"] == 1 and stats["eager"] == 0


@pytest.mark.accelerator
def test_compile_fp32_graph_captures_and_runs(tiny_dit, device) -> None:
    run_dit, fallback_dit = prepare_flow_dit(tiny_dit, "compile_fp32")
    runner = tiny_runner(
        run_dit,
        device,
        mode="compile_fp32",
        dit_fallback=fallback_dit,
        capture_sizes=(1,),
    )
    generator = torch.Generator(device=device).manual_seed(47)
    hist, x0, cond = make_inputs(device, generator, 1)
    cfg = torch.full((1, 1, 1), 2.0, device=device)
    with torch.inference_mode():
        got = runner.run(hist, x0, cond, cfg, N_TIMESTEPS)
        want = eager_patch(tiny_dit, hist, x0, cond, cfg, device)
    assert got is not None
    assert runner.dit is run_dit  # no degradation: compile captured cleanly
    # Inductor reassociates fp32 math; the drift stays far inside the
    # batched-GEMM roundoff envelope (~5e-3) the plan budgets for c>=2.
    torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-4)


@pytest.fixture(autouse=True)
def disable_flow_optimization_environment(monkeypatch) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "0")


@pytest.mark.parametrize("inherited_limit", [2, 96])
@pytest.mark.parametrize("should_raise", [False, True])
def test_explicit_time_cache_protects_original_compiled_call(
    monkeypatch: pytest.MonkeyPatch, inherited_limit: int, should_raise: bool
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "0")
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    config = torch._dynamo.config
    observed: list[tuple[int, bool, bool]] = []

    def config_state() -> tuple[int, bool, bool]:
        return (
            config.recompile_limit,
            config.fail_on_recompile_limit_hit,
            config.suppress_errors,
        )

    class CompiledProbe(torch.nn.Module):
        def forward(self, *, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            observed.append(config_state())
            if should_raise:
                raise RuntimeError("compiled probe failure")
            else:
                return x + t

    runner = FlowGraphRunner(
        CompiledProbe(),
        torch.tensor([0.0, 1.0]),
        n_timesteps=1,
        patch_size=1,
        history_length=0,
        mode="compile_fp32",
        dit_fallback=torch.nn.Identity(),
        precompute_time_conditions=True,
    )
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    t = torch.ones_like(x)
    initial = config_state()
    with config.patch(
        recompile_limit=inherited_limit,
        fail_on_recompile_limit_hit=False,
        suppress_errors=True,
    ):
        if should_raise:
            with pytest.raises(RuntimeError, match="compiled probe failure"):
                runner.dit(x=x, t=t)
        else:
            assert torch.equal(runner.dit(x=x, t=t), x + t)
        assert observed == [(max(DIT_RECOMPILE_LIMIT, inherited_limit), True, False)]
        assert config_state() == (inherited_limit, False, True)
    assert config_state() == initial


@pytest.fixture
def flow_dispatch_dit(tiny_dit: DiT) -> DiT:
    # note (Codex): a zero head isolates dispatch lifecycle from backend rounding.
    with torch.no_grad():
        tiny_dit.final_layer.linear.weight.zero_()
        tiny_dit.final_layer.linear.bias.zero_()
    return tiny_dit


def test_optimized_prepare_preserves_original_weights(monkeypatch) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "1")
    source = tiny_cpu_dit()
    original = {
        name: parameter.detach().clone()
        for name, parameter in source.named_parameters()
    }
    execution, fallback = prepare_flow_dit(source, "compile_bf16")
    assert isinstance(execution, CompiledDiTExecutor)
    assert execution.source is fallback and fallback is not source
    assert all(parameter.dtype == torch.bfloat16 for parameter in fallback.parameters())
    for name, parameter in source.named_parameters():
        assert parameter.dtype == torch.float32
        assert torch.equal(parameter, original[name])


@pytest.mark.parametrize("mode", ["fp32", "bf16"])
def test_flow_caches_require_compiled_execution(monkeypatch, mode: str) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "1")
    with pytest.raises(ValueError, match="require a compile flow mode"):
        prepare_flow_dit(tiny_cpu_dit(), mode)


def test_invalid_flow_cache_switches_fail(monkeypatch) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "2")
    with pytest.raises(ValueError, match="FIRERED_TTS3_PRECOMPUTE_TIME"):
        prepare_flow_dit(tiny_cpu_dit(), "compile_bf16")


@pytest.mark.accelerator
def test_failed_optimization_gate_keeps_original_compiled_graph(
    flow_dispatch_dit, device, monkeypatch, caplog
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "1")
    execution, fallback = prepare_flow_dit(flow_dispatch_dit, "compile_bf16")
    assert isinstance(execution, CompiledDiTExecutor)
    runner = tiny_runner(
        execution,
        device,
        mode="compile_bf16",
        dit_fallback=fallback,
        capture_sizes=(1,),
    )
    original_loop = flow_graph_module.euler_flow_loop
    optimized_calls = 0

    def perturb_cached_flow(*args, **kwargs) -> torch.Tensor:
        nonlocal optimized_calls
        patch = original_loop(*args, **kwargs)
        if kwargs.get("dispatch_modes"):
            optimized_calls += 1
            return patch + 1
        else:
            return patch

    monkeypatch.setattr(flow_graph_module, "euler_flow_loop", perturb_cached_flow)
    generator = torch.Generator(device=device).manual_seed(71)
    hist, noise, cond = make_inputs(device, generator, 1)
    cfg = torch.full((1, 1, 1), 1.2, device=device)
    with torch.inference_mode():
        got = runner.run(hist, noise, cond, cfg, N_TIMESTEPS)
        calls_after_capture = optimized_calls
        repeated = runner.run(hist, noise, cond, cfg, N_TIMESTEPS)
        expected = original_loop(
            execution.original,
            torch.cat([hist, noise], dim=1),
            cond,
            runner.t_span,
            cfg,
            N_TIMESTEPS,
            PATCH_SIZE,
            dit_dtype=torch.bfloat16,
        )
    assert calls_after_capture > 0 and optimized_calls == calls_after_capture
    assert "Complete Flow exact gate failed" in caplog.text
    assert got is not None and torch.equal(got, expected)
    assert torch.equal(got, repeated)
    assert runner.enabled and runner.dit is execution and runner.dit is not fallback
    assert runner.stats()["optimized_captures"] == 0
    assert runner.stats()["optimization_fallbacks"] == 1
    assert runner.stats()["captures"] == 1 and runner.stats()["eager"] == 0
    entry = next(iter(runner.cache.values()))
    assert entry.dispatch_modes == () and entry.time_table is None


@pytest.mark.accelerator
def test_optimized_flow_hits_dispatch_and_retains_time_table(
    flow_dispatch_dit, device, monkeypatch
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "1")
    cached_time_hits = collections.Counter()
    original_dispatch = TimeGemmMode.__torch_dispatch__

    def count_cached_time_hits(self, func, types, args=(), kwargs=None) -> object:
        if (
            self.phase == "cached"
            and func is torch.ops.aten.addmm.out
            and args[2].data_ptr() in self.weights
        ):
            cached_time_hits[id(self)] += 1
        return original_dispatch(self, func, types, args, kwargs)

    monkeypatch.setattr(TimeGemmMode, "__torch_dispatch__", count_cached_time_hits)
    execution, fallback = prepare_flow_dit(flow_dispatch_dit, "compile_bf16")
    assert isinstance(execution, CompiledDiTExecutor)
    runner = tiny_runner(
        execution,
        device,
        mode="compile_bf16",
        dit_fallback=fallback,
        capture_sizes=(1,),
    )
    generator = torch.Generator(device=device).manual_seed(73)
    hist, noise, cond = make_inputs(device, generator, 1)
    cfg = torch.full((1, 1, 1), 2.0, device=device)
    with torch.inference_mode():
        got = runner.run(hist, noise, cond, cfg, N_TIMESTEPS)
        expected = euler_flow_loop(
            execution.original,
            torch.cat([hist, noise], dim=1),
            cond,
            runner.t_span,
            cfg,
            N_TIMESTEPS,
            PATCH_SIZE,
            dit_dtype=torch.bfloat16,
        )
    assert got is not None and torch.equal(got, expected)
    assert runner.dit is execution and runner.enabled
    assert runner.stats()["optimized_captures"] == 1
    assert runner.stats()["optimization_fallbacks"] == 0
    assert runner.stats()["eager"] == 0
    entry = next(iter(runner.cache.values()))
    assert len(entry.dispatch_modes) == 1
    time_mode = entry.dispatch_modes[0]
    assert isinstance(time_mode, TimeGemmMode)
    assert time_mode.table is entry.time_table and entry.time_table is not None
    assert len(entry.time_table.steps) == N_TIMESTEPS
    assert cached_time_hits[id(time_mode)] > 0
    table_ref = weakref.ref(entry.time_table)
    value_ref = weakref.ref(next(iter(entry.time_table.steps[0].values())).output)
    del time_mode
    gc.collect()
    assert table_ref() is entry.time_table and value_ref() is not None
    def reject_time_weight_scan(self: TimeConditionCache) -> None:
        raise AssertionError("Time weights must not be scanned during replay")

    monkeypatch.setattr(TimeConditionCache, "validate_weights", reject_time_weight_scan)
    hits_before_replay = dict(cached_time_hits)
    with torch.inference_mode():
        repeated = runner.run(hist, noise, cond, cfg, N_TIMESTEPS)
    assert torch.equal(got, repeated)
    assert dict(cached_time_hits) == hits_before_replay
    assert runner.stats()["captures"] == 1 and runner.stats()["hits"] == 1
    torch.cuda.synchronize(device)
    runner.cache.clear()
    del entry
    gc.collect()
    assert table_ref() is None and value_ref() is None


def test_time_exact_gate_requires_finite_identical_bytes() -> None:
    reference = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert finite_byte_equal(reference, reference.clone())
    assert finite_byte_equal(reference, reference.t().contiguous().t())
    for value in (float("nan"), float("inf"), -float("inf")):
        invalid = torch.tensor([value])
        assert not finite_byte_equal(invalid, invalid.clone())
    assert not finite_byte_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not finite_byte_equal(reference, reference.double())
    assert not finite_byte_equal(reference, reference.flatten())
    changed = torch.nextafter(reference, torch.full_like(reference, float("inf")))
    assert not finite_byte_equal(reference, changed)


@pytest.mark.parametrize(
    "failure,message",
    [
        ("negative", "Invalid or nested"),
        ("past_end", "Invalid or nested"),
        ("nested", "Invalid or nested"),
        ("missing", "missed compiled addmm"),
        ("unscoped", "Unscoped or repeated"),
        ("repeated", "Unscoped or repeated"),
    ],
)
def test_time_gemm_rejects_invalid_step_and_missing_operations(
    failure: str, message: str
) -> None:
    inputs, weight, bias = torch.ones(2, 3), torch.ones(3, 4), torch.ones(4)
    output = torch.empty(2, 4)
    mode = TimeGemmMode({weight.data_ptr(): "time"}, phase="collect", n_timesteps=2)
    with pytest.raises(TimeCacheUnavailable, match=message):
        if failure in ("negative", "past_end"):
            with mode.step(-1 if failure == "negative" else 2):
                pass
        elif failure == "nested":
            with mode.step(0), mode.step(1):
                pass
        elif failure == "missing":
            with mode.step(0):
                pass
        elif failure == "unscoped":
            with mode:
                torch.addmm(bias, inputs, weight, out=output)
        else:
            with mode, mode.step(0):
                torch.addmm(bias, inputs, weight, out=output)
                torch.addmm(bias, inputs, weight, out=output)
    with mode, mode.step(1):
        torch.addmm(bias, inputs, weight, out=output)
    assert torch.equal(output, torch.full((2, 4), 4.0))


def test_time_gemm_cache_copies_snapshot_and_rejects_wrong_signature() -> None:
    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    bias, output = torch.ones(4), torch.empty(2, 4)
    weights = {weight.data_ptr(): "time"}
    collector = TimeGemmMode(weights, phase="collect", n_timesteps=1)
    with collector, collector.step(0):
        torch.addmm(bias, inputs, weight, out=output)
    expected = output.clone()
    table = TimeGemmTable(
        model_id=id(weight), n_timesteps=1, steps=(collector.collected[0],)
    )
    cached = TimeGemmMode(weights, phase="cached", n_timesteps=1, table=table)
    output.fill_(-1)
    with cached, cached.step(0):
        returned = torch.addmm(bias, inputs, weight, out=output)
    assert returned is output
    assert finite_byte_equal(output, expected)
    assert table.steps[0]["time"].output.data_ptr() != output.data_ptr()
    changed_stride = inputs.t().contiguous().t()
    with pytest.raises(TimeCacheUnavailable, match="signature changed"):
        with cached, cached.step(0):
            torch.addmm(bias, changed_stride, weight, out=output)
    missing_table = TimeGemmTable(model_id=id(weight), n_timesteps=1, steps=({},))
    missing = TimeGemmMode(weights, phase="cached", n_timesteps=1, table=missing_table)
    with pytest.raises(TimeCacheUnavailable, match="signature changed"):
        with missing, missing.step(0):
            torch.addmm(bias, inputs, weight, out=output)


@pytest.mark.parametrize("mutation", ["inplace", "replace", "stride", "bias"])
def test_time_condition_cache_detects_weight_changes(mutation: str) -> None:
    source = tiny_cpu_dit()
    cache = TimeConditionCache(source)
    cache.validate_weights()
    assert cache.version_checks_available
    layer = source.t_embedder.time_mlp[2]
    with torch.no_grad():
        if mutation == "inplace":
            layer.weight.add_(1)
        elif mutation == "replace":
            layer.weight = torch.nn.Parameter(layer.weight.clone())
        elif mutation == "stride":
            layer.weight.transpose_(0, 1)
        else:
            layer.bias.add_(1)
    with pytest.raises(TimeCacheUnavailable, match="Time parameter changed"):
        cache.validate_weights()


@pytest.mark.parametrize("mode", ["compile_fp32", "compile_bf16"])
def test_compiled_time_cache_enabled_by_default(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.delenv("FIRERED_TTS3_PRECOMPUTE_TIME", raising=False)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    execution, fallback = prepare_flow_dit(tiny_cpu_dit(), mode)
    assert isinstance(execution, CompiledDiTExecutor)
    runner = tiny_runner(
        execution, torch.device("cpu"), mode=mode, dit_fallback=fallback
    )
    assert runner.precompute_time_conditions
    assert runner.executor is execution


@pytest.mark.parametrize("mode", ["fp32", "bf16"])
def test_eager_time_cache_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.delenv("FIRERED_TTS3_PRECOMPUTE_TIME", raising=False)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    execution, fallback = prepare_flow_dit(tiny_cpu_dit(), mode)
    assert execution is fallback
    runner = tiny_runner(
        execution, torch.device("cpu"), mode=mode, dit_fallback=fallback
    )
    assert not runner.precompute_time_conditions
    assert runner.executor is None


@pytest.mark.parametrize("mode", ["compile_fp32", "compile_bf16"])
def test_compiled_time_cache_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "0")
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    execution, fallback = prepare_flow_dit(tiny_cpu_dit(), mode)
    assert not isinstance(execution, CompiledDiTExecutor)
    runner = tiny_runner(
        execution, torch.device("cpu"), mode=mode, dit_fallback=fallback
    )
    assert not runner.precompute_time_conditions
    assert runner.executor is None


__all__ = []


@pytest.mark.parametrize("with_time", [True, False])
def test_flow_entry_records_external_resources_once_per_replay_stream(
    with_time: bool,
) -> None:
    class RecordingTensor:
        def __init__(self) -> None:
            self.streams: list[object] = []

        def record_stream(self, stream) -> None:
            self.streams.append(stream)

    static = [RecordingTensor() for _ in range(3)]
    span = RecordingTensor()
    time_outputs = [RecordingTensor() for _ in range(2)] if with_time else []
    table = (
        SimpleNamespace(
            steps=tuple(
                {"gemm": SimpleNamespace(output=value)} for value in time_outputs
            )
        )
        if with_time
        else None
    )
    entry = flow_graph_module.FlowGraphEntry(
        graph=object(),
        xt=static[0],
        cond=static[1],
        cfg=static[2],
        time_table=table,
        t_span=span,
    )
    resources = static + [span] + time_outputs
    first = SimpleNamespace(device=SimpleNamespace(index=0), cuda_stream=11)
    same_native_stream = SimpleNamespace(
        device=SimpleNamespace(index=0), cuda_stream=11
    )
    second = SimpleNamespace(device=SimpleNamespace(index=0), cuda_stream=22)
    entry.record_replay_stream(first)
    entry.record_replay_stream(same_native_stream)
    assert all(tensor.streams == [first] for tensor in resources)
    assert entry.replay_streams[(0, 11)] is first
    entry.record_replay_stream(second)
    entry.record_replay_stream(second)
    assert all(tensor.streams == [first, second] for tensor in resources)
    assert entry.replay_streams[(0, 22)] is second


@pytest.mark.accelerator
def test_time_tables_follow_graph_lru_eviction(flow_dispatch_dit, device, monkeypatch):
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "1")
    execution, fallback = prepare_flow_dit(flow_dispatch_dit, "compile_bf16")
    runner = tiny_runner(
        execution,
        device,
        mode="compile_bf16",
        dit_fallback=fallback,
        capture_sizes=(1,),
        max_graphs=1,
    )
    generator = torch.Generator(device=device).manual_seed(97)
    stream = torch.cuda.Stream(device=device)
    table_refs = []
    with torch.inference_mode():
        for batch in (1, 2, 3):
            hist, noise, cond = make_inputs(device, generator, batch)
            actual = runner.run(
                hist, noise, cond, torch.ones(batch, device=device), N_TIMESTEPS
            )
            assert actual is not None and torch.equal(actual, noise)
            entry = next(iter(runner.cache.values()))
            table_refs.append(weakref.ref(entry.time_table))
            del entry
            gc.collect()
            assert table_refs[-1]() is not None
            assert all(ref() is None for ref in table_refs[:-1])
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            replayed = runner.run(
                hist, noise, cond, torch.ones(batch, device=device), N_TIMESTEPS
            )
        runner.cache.clear()
        gc.collect()
        assert all(ref() is None for ref in table_refs)
    stream.synchronize()
    assert torch.equal(replayed, noise)
