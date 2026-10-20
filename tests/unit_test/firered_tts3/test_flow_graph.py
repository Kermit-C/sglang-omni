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
from torch.utils._python_dispatch import TorchDispatchMode

import sglang_omni.models.firered_tts3.compiled_dit as compiled_dit_module
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
from sglang_omni.models.firered_tts3.conv_cache import Conv1dWeightCache
from sglang_omni.models.firered_tts3.dit import ConvBlock, DiT
from sglang_omni.models.firered_tts3.flow_graph import (
    FLOW_MODES,
    FlowGraphRunner,
    euler_flow_loop,
    flow_t_span,
    prepare_flow_dit,
)
from sglang_omni.models.firered_tts3.flow_precision import FlowPrecisionGate
from sglang_omni.models.firered_tts3.model_runner import (
    FireRedTTS3ModelRunner,
    FlowState,
)
from sglang_omni.models.firered_tts3.packed_qkv import PackedQkvModel
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
    runner._async_enabled = False
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
    monkeypatch.setenv("FIRERED_TTS3_FFN_GEMM", "0")
    monkeypatch.setenv("FIRERED_TTS3_FLOW_ATOL", "0")
    monkeypatch.setenv("FIRERED_TTS3_FLOW_RTOL", "0")
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "0")
    monkeypatch.setenv("FIREREDTTS3_CONV_LAYOUT", "conv1d")


@pytest.mark.parametrize("inherited_limit", [2, 96])
@pytest.mark.parametrize("should_raise", [False, True])
def test_explicit_time_cache_protects_original_compiled_call(
    monkeypatch: pytest.MonkeyPatch, inherited_limit: int, should_raise: bool
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "0")
    monkeypatch.setenv("FIREREDTTS3_CONV_LAYOUT", "conv1d")
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


@pytest.mark.parametrize(
    "time_cache,conv_cache", [(True, False), (False, True), (True, True)]
)
def test_optimized_prepare_preserves_original_weights(
    monkeypatch, time_cache: bool, conv_cache: bool
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", str(int(time_cache)))
    monkeypatch.setenv(
        "FIREREDTTS3_CONV_LAYOUT", "conv2d_channels_last" if conv_cache else "conv1d"
    )
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
    assert sum(
        isinstance(layer, torch.nn.Conv1d) for layer in fallback.modules()
    ) == sum(isinstance(layer, torch.nn.Conv1d) for layer in source.modules())


@pytest.mark.parametrize("mode", ["fp32", "bf16"])
@pytest.mark.parametrize("switch", ["time", "conv"])
def test_flow_caches_require_compiled_execution(
    monkeypatch, mode: str, switch: str
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "1" if switch == "time" else "0")
    monkeypatch.setenv(
        "FIREREDTTS3_CONV_LAYOUT",
        "conv2d_channels_last" if switch == "conv" else "conv1d",
    )
    with pytest.raises(ValueError, match="require a compile flow mode"):
        prepare_flow_dit(tiny_cpu_dit(), mode)


@pytest.mark.parametrize(
    "time_switch,conv_switch,message",
    [
        ("2", "conv1d", "FIRERED_TTS3_PRECOMPUTE_TIME"),
        ("0", "conv2d", "FIREREDTTS3_CONV_LAYOUT"),
    ],
)
def test_invalid_flow_cache_switches_fail(
    monkeypatch, time_switch: str, conv_switch: str, message: str
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", time_switch)
    monkeypatch.setenv("FIREREDTTS3_CONV_LAYOUT", conv_switch)
    with pytest.raises(ValueError, match=message):
        prepare_flow_dit(tiny_cpu_dit(), "compile_bf16")


@pytest.mark.accelerator
@pytest.mark.parametrize(
    "corrupted_stage,atol,conv_enabled,accepted",
    [
        ("conv", 0.0, True, False),
        ("conv", 0.5, True, True),
        ("conv", 0.125, True, False),
        ("observed", 10.0, True, False),
        ("time", 10.0, False, False),
        ("time", 10.0, True, False),
    ],
)
def test_flow_tolerance_only_relaxes_convolution_gate(
    flow_dispatch_dit: DiT,
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    corrupted_stage: str,
    atol: float,
    conv_enabled: bool,
    accepted: bool,
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_FLOW_ATOL", str(atol))
    monkeypatch.setenv("FIRERED_TTS3_FLOW_RTOL", "0")
    monkeypatch.setenv(
        "FIRERED_TTS3_PRECOMPUTE_TIME", "1" if corrupted_stage == "time" else "0"
    )
    monkeypatch.setenv(
        "FIREREDTTS3_CONV_LAYOUT",
        "conv2d_channels_last" if conv_enabled else "conv1d",
    )
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
    corrupted_calls = 0
    injected_delta = 0.25

    def perturb_selected_flow(*args, **kwargs) -> torch.Tensor:
        nonlocal corrupted_calls
        patch = original_loop(*args, **kwargs)
        modes = kwargs.get("dispatch_modes", ())
        has_time = any(isinstance(mode, TimeGemmMode) for mode in modes)
        has_conv = any(isinstance(mode, Conv1dWeightCache) for mode in modes)
        should_corrupt = (
            (
                corrupted_stage == "observed"
                and kwargs.get("executor") is not None
                and not modes
            )
            or (corrupted_stage == "time" and has_time and not has_conv)
            or (corrupted_stage == "conv" and has_conv)
        )
        if should_corrupt:
            corrupted_calls += 1
            patch.add_(injected_delta)
        return patch

    monkeypatch.setattr(flow_graph_module, "euler_flow_loop", perturb_selected_flow)
    generator = torch.Generator(device=device).manual_seed(71)
    hist, noise, cond = make_inputs(device, generator, 1)
    cfg = torch.full((1, 1, 1), 1.2, device=device)
    with torch.inference_mode():
        got = runner.run(hist, noise, cond, cfg, N_TIMESTEPS)
        calls_after_capture = corrupted_calls
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
        if accepted:
            expected = expected + injected_delta
    assert calls_after_capture > 0 and corrupted_calls == calls_after_capture
    assert got is not None and torch.equal(got, expected)
    assert torch.equal(got, repeated)
    assert runner.enabled and runner.dit is execution and runner.dit is not fallback
    stats = runner.stats()
    assert stats["optimized_captures"] == int(accepted)
    assert stats["optimization_fallbacks"] == int(not accepted)
    assert stats["captures"] == 1 and stats["eager"] == 0
    entry = next(iter(runner.cache.values()))
    if accepted:
        assert any(isinstance(mode, Conv1dWeightCache) for mode in entry.dispatch_modes)
        assert "Flow optimization rejected" not in caplog.text
    else:
        assert entry.dispatch_modes == () and entry.time_table is None
        assert "Flow optimization rejected" in caplog.text


@pytest.mark.accelerator
@pytest.mark.parametrize(
    "time_cache,conv_cache", [(True, False), (False, True), (True, True)]
)
def test_optimized_flow_hits_dispatch_and_retains_time_table(
    flow_dispatch_dit, device, monkeypatch, time_cache: bool, conv_cache: bool
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", str(int(time_cache)))
    monkeypatch.setenv(
        "FIREREDTTS3_CONV_LAYOUT", "conv2d_channels_last" if conv_cache else "conv1d"
    )
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
    assert sum(isinstance(mode, TimeGemmMode) for mode in entry.dispatch_modes) == int(
        time_cache
    )
    assert sum(
        isinstance(mode, Conv1dWeightCache) for mode in entry.dispatch_modes
    ) == int(conv_cache)
    if time_cache:
        time_mode = next(
            mode for mode in entry.dispatch_modes if isinstance(mode, TimeGemmMode)
        )
        assert time_mode.table is entry.time_table and entry.time_table is not None
        assert len(entry.time_table.steps) == N_TIMESTEPS
        assert cached_time_hits[id(time_mode)] > 0
        table_ref = weakref.ref(entry.time_table)
        value_ref = weakref.ref(next(iter(entry.time_table.steps[0].values())).output)
        del time_mode
        gc.collect()
        assert table_ref() is entry.time_table and value_ref() is not None
    else:
        assert entry.time_table is None
    if conv_cache:
        assert (
            next(
                mode
                for mode in entry.dispatch_modes
                if isinstance(mode, Conv1dWeightCache)
            ).hits
            > 0
        )

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
    if time_cache:
        assert table_ref() is None and value_ref() is None


@pytest.mark.parametrize(
    "device_name", ["cpu", pytest.param("cuda", marks=pytest.mark.accelerator)]
)
def test_conv_weight_cache_preserves_model_and_rng(device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device(device_name)
    source = ConvBlock(8, 8).to(device=device).eval()
    original = {
        name: (value.detach().clone(), value.data_ptr(), value.stride())
        for name, value in source.state_dict().items()
    }
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    cache = Conv1dWeightCache(source)
    cache.validate_weights()
    assert cache.version_checks_available
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert source.state_dict().keys() == original.keys()
    assert sum(isinstance(layer, torch.nn.Conv1d) for layer in source.modules()) == 2
    for name, value in source.state_dict().items():
        tensor, pointer, stride = original[name]
        assert torch.equal(value, tensor)
        assert value.data_ptr() == pointer and value.stride() == stride


@pytest.mark.parametrize(
    "device_name", ["cpu", pytest.param("cuda", marks=pytest.mark.accelerator)]
)
@pytest.mark.parametrize("operator", ["conv1d", "convolution_3d", "convolution_4d"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_conv_weight_cache_dispatch_contract(
    device_name: str, operator: str, with_bias: bool
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device(device_name)
    source = torch.nn.Conv1d(4, 4, 3, padding=1).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.zero_()
        source.bias.fill_(3.0)
        for channel in range(4):
            source.weight[channel, channel, 1] = 0.5
            source.weight[channel, (channel + 1) % 4, 0] = 0.25
    inputs = ((torch.arange(96, device=device) % 8) - 4).reshape(2, 4, 12)
    inputs = inputs.to(dtype=torch.bfloat16)
    cache = Conv1dWeightCache(source)
    bias = source.bias if with_bias else None

    def call(weight: torch.Tensor) -> torch.Tensor:
        if operator == "conv1d":
            return torch.ops.aten.conv1d.default(inputs, weight, bias, padding=[1])
        elif operator == "convolution_3d":
            return torch.ops.aten.convolution.default(
                inputs, weight, bias, [1], [1], [1], False, [0], 1
            )
        else:
            return torch.ops.aten.convolution.default(
                inputs.unsqueeze(2),
                weight.unsqueeze(2),
                bias,
                [1, 1],
                [0, 1],
                [1, 1],
                False,
                [0, 0],
                1,
            )

    with torch.inference_mode():
        want = call(source.weight)
        with cache:
            got = call(source.weight)
        assert cache.hits == 1 and cache.misses == 0
        unrelated_weight = source.weight.clone()
        with cache:
            unhandled = call(unrelated_weight)
    assert got.shape == want.shape and got.stride() == want.stride()
    assert got.dtype == want.dtype == torch.bfloat16
    assert torch.equal(got, want)
    assert torch.equal(unhandled, want)
    assert cache.hits == 1 and cache.misses == 1
    cache.validate_weights()


def test_conv_weight_cache_rejects_same_pointer_other_view() -> None:
    source = torch.nn.Conv1d(4, 4, 3, padding=1)
    cache = Conv1dWeightCache(source)
    different_view = source.weight.transpose(0, 1)
    assert different_view.data_ptr() == source.weight.data_ptr()
    assert different_view.shape == source.weight.shape
    inputs = torch.randn(2, 4, 12)
    with torch.inference_mode():
        want = F.conv1d(inputs, different_view, padding=1)
        with cache:
            got = F.conv1d(inputs, different_view, padding=1)
    assert torch.equal(got, want)
    assert cache.hits == 0 and cache.misses == 1


@pytest.mark.parametrize("mutation", ["inplace", "replace", "stride"])
def test_conv_weight_cache_detects_weight_changes(mutation: str) -> None:
    source = torch.nn.Conv1d(4, 4, 3, padding=1)
    cache = Conv1dWeightCache(source)
    cache.validate_weights()
    with torch.no_grad():
        if mutation == "inplace":
            source.weight.add_(1.0)
        elif mutation == "replace":
            source.weight = torch.nn.Parameter(source.weight.clone())
        else:
            source.weight.transpose_(0, 1)
    with pytest.raises(RuntimeError, match="Convolution weight changed"):
        cache.validate_weights()


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
def test_compiled_flow_caches_enabled_by_default(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.delenv("FIRERED_TTS3_FFN_GEMM", raising=False)
    monkeypatch.delenv("FIRERED_TTS3_PRECOMPUTE_TIME", raising=False)
    monkeypatch.delenv("FIREREDTTS3_CONV_LAYOUT", raising=False)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    execution, fallback = prepare_flow_dit(tiny_cpu_dit(), mode)
    assert isinstance(execution, CompiledDiTExecutor)
    runner = tiny_runner(
        execution, torch.device("cpu"), mode=mode, dit_fallback=fallback
    )
    assert runner.precompute_time_conditions and runner.optimize_convolution
    assert runner.executor is execution


@pytest.mark.parametrize("mode", ["fp32", "bf16"])
def test_eager_flow_caches_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.delenv("FIRERED_TTS3_FFN_GEMM", raising=False)
    monkeypatch.delenv("FIRERED_TTS3_PRECOMPUTE_TIME", raising=False)
    monkeypatch.delenv("FIREREDTTS3_CONV_LAYOUT", raising=False)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    execution, fallback = prepare_flow_dit(tiny_cpu_dit(), mode)
    assert execution is fallback
    runner = tiny_runner(
        execution, torch.device("cpu"), mode=mode, dit_fallback=fallback
    )
    assert not runner.precompute_time_conditions and not runner.optimize_convolution
    assert runner.executor is None


@pytest.mark.parametrize("mode", ["compile_fp32", "compile_bf16"])
@pytest.mark.parametrize(
    "disabled,time_enabled,conv_enabled",
    [("time", False, True), ("conv", True, False), ("both", False, False)],
)
def test_compiled_flow_caches_can_be_disabled_independently(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    disabled: str,
    time_enabled: bool,
    conv_enabled: bool,
) -> None:
    monkeypatch.delenv("FIRERED_TTS3_PRECOMPUTE_TIME", raising=False)
    monkeypatch.delenv("FIREREDTTS3_CONV_LAYOUT", raising=False)
    if disabled in ("time", "both"):
        monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "0")
    if disabled in ("conv", "both"):
        monkeypatch.setenv("FIREREDTTS3_CONV_LAYOUT", "conv1d")
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: None)
    execution, fallback = prepare_flow_dit(tiny_cpu_dit(), mode)
    assert isinstance(execution, CompiledDiTExecutor) == (time_enabled or conv_enabled)
    runner = tiny_runner(
        execution, torch.device("cpu"), mode=mode, dit_fallback=fallback
    )
    assert runner.precompute_time_conditions == time_enabled
    assert runner.optimize_convolution == conv_enabled
    if time_enabled or conv_enabled:
        assert runner.executor is execution
    else:
        assert runner.executor is None


@pytest.mark.parametrize("source", ["constructor", "environment"])
def test_flow_precision_explicit_exact_checks_remain_byte_exact(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_FLOW_ATOL", "0")
    monkeypatch.setenv("FIRERED_TTS3_FLOW_RTOL", "0")
    gate = (
        FlowPrecisionGate()
        if source == "constructor"
        else FlowPrecisionGate.from_environment()
    )
    reference = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    exact = gate.compare(reference.t().contiguous().t(), reference)
    assert exact.accepted and exact.byte_exact
    assert not gate.compare(torch.tensor([0.0]), torch.tensor([-0.0])).accepted
    changed = torch.nextafter(reference, torch.full_like(reference, float("inf")))
    assert not gate.compare(changed, reference).accepted


def test_flow_precision_default_budget_accepts_only_small_roundoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FIRERED_TTS3_FLOW_ATOL", raising=False)
    monkeypatch.delenv("FIRERED_TTS3_FLOW_RTOL", raising=False)
    gate = FlowPrecisionGate.from_environment()
    reference = torch.tensor([[0.0, 1.0]])
    close = torch.tensor([[0.0005, 1.005]])
    assert gate.compare(close, reference).accepted
    assert not FlowPrecisionGate().compare(close, reference).accepted
    assert not gate.compare(torch.tensor([[0.002, 1.0]]), reference).accepted
    assert not gate.compare(torch.tensor([[0.0, 1.02]]), reference).accepted


@pytest.mark.parametrize(
    "variable", ["FIRERED_TTS3_FLOW_ATOL", "FIRERED_TTS3_FLOW_RTOL"]
)
@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf", "invalid"])
def test_flow_precision_rejects_invalid_environment(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match=variable):
        FlowPrecisionGate.from_environment()


@pytest.mark.parametrize("target", ["actual", "expected", "both"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_flow_precision_rejects_nonfinite_values(target: str, value: float) -> None:
    reference = torch.ones(2, 3)
    actual = reference.clone()
    if target in ("actual", "both"):
        actual[1, 2] = value
    if target in ("expected", "both"):
        reference[1, 2] = value
    result = FlowPrecisionGate(atol=100.0, rtol=100.0).compare(actual, reference)
    assert not result.accepted and not result.finite


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "device"])
def test_flow_precision_rejects_incompatible_tensors(mismatch: str) -> None:
    reference = torch.ones(2, 3)
    if mismatch == "shape":
        actual = reference.flatten()
    elif mismatch == "dtype":
        actual = reference.double()
    else:
        actual = torch.empty_like(reference, device="meta")
    result = FlowPrecisionGate(atol=100.0, rtol=100.0).compare(actual, reference)
    assert not result.accepted and not result.schema_valid


@pytest.mark.parametrize(
    "reference,actual,atol,rtol,accepted",
    [
        (0.0, 0.125, 0.125, 0.0, True),
        (0.0, 0.125 + 2**-26, 0.125, 0.0, False),
        (2.0, 2.25, 0.0, 0.125, True),
        (2.0, 2.25 + 2**-22, 0.0, 0.125, False),
        (-2.0, -2.25, 0.0, 0.125, True),
        (2.0, 2.375, 0.125, 0.125, True),
        (2.0, 2.375 + 2**-22, 0.125, 0.125, False),
        (0.0, 2**-24, 0.0, 100.0, False),
        (2**-24, 2**-23, 0.0, 0.5, False),
    ],
)
def test_flow_precision_elementwise_tolerance_boundaries(
    reference: float,
    actual: float,
    atol: float,
    rtol: float,
    accepted: bool,
) -> None:
    result = FlowPrecisionGate(atol=atol, rtol=rtol).compare(
        torch.tensor([actual]), torch.tensor([reference])
    )
    assert result.accepted == accepted
    assert result.out_of_bounds == int(not accepted)


def test_flow_precision_rejects_one_outlier_in_one_batch_row() -> None:
    reference = torch.ones(8, 4, 64)
    actual = reference.clone()
    actual[-1, -1, -1] += 0.25
    result = FlowPrecisionGate(atol=0.125, rtol=0.0).compare(actual, reference)
    assert not result.accepted and result.out_of_bounds == 1
    assert result.rmse < 0.125 and result.max_abs == 0.25


__all__ = []


@pytest.fixture
def controlled_conv_flow(device: torch.device) -> torch.nn.Module:
    class ConvFlow(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv1d(1024, 1024, 3, padding=1, bias=False)
            with torch.no_grad():
                self.conv.weight.zero_()
                for channel in range(REDAE_DIM):
                    self.conv.weight[channel, channel, 1] = 0.125

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            padded = F.pad(x, (0, 1024 - x.shape[-1]))
            output = self.conv(padded.transpose(1, 2)).transpose(1, 2)
            return output[:, :, :REDAE_DIM]

    return ConvFlow().to(device=device).eval()


@pytest.mark.accelerator
def test_controlled_convolution_replay_owns_plan_without_weight_scan(
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
    controlled_conv_flow: torch.nn.Module,
) -> None:
    from sglang_omni.models.firered_tts3.conv_plan import ControlledConvPlan

    frontend = pytest.importorskip("cudnn")
    if (
        torch.cuda.get_device_capability(device) != (12, 0)
        or torch.backends.cudnn.version() != 92000
        or frontend.backend_version() != 92000
        or frontend.__version__ != "1.28.0"
    ):
        pytest.skip("The controlled convolution plan requires its validated CUDA stack")
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "0")
    monkeypatch.setenv("FIREREDTTS3_CONV_LAYOUT", "conv2d_channels_last")
    monkeypatch.setenv("FIRERED_TTS3_FLOW_ATOL", "0")
    monkeypatch.setenv("FIRERED_TTS3_FLOW_RTOL", "0")

    calls = 0
    original_call = ControlledConvPlan.__call__

    def count_plan_calls(
        self, inputs: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_call(self, inputs, weight)

    monkeypatch.setattr(ControlledConvPlan, "__call__", count_plan_calls)
    original_loop = flow_graph_module.euler_flow_loop
    generic_calls = 0

    def perturb_generic_convolution(*args, **kwargs) -> torch.Tensor:
        nonlocal generic_calls
        patch = original_loop(*args, **kwargs)
        if any(
            isinstance(mode, Conv1dWeightCache) and not mode.use_controlled_plan
            for mode in kwargs.get("dispatch_modes", ())
        ):
            generic_calls += 1
            patch.add_(1.0)
        return patch

    monkeypatch.setattr(
        flow_graph_module, "euler_flow_loop", perturb_generic_convolution
    )
    execution, fallback = prepare_flow_dit(controlled_conv_flow, "compile_bf16")
    runner = tiny_runner(
        execution,
        device,
        mode="compile_bf16",
        dit_fallback=fallback,
        capture_sizes=(4,),
    )
    generator = torch.Generator(device=device).manual_seed(817)
    hist, noise, cond = make_inputs(device, generator, 4)
    cfg = torch.full((4, 1, 1), 1.2, device=device)
    with torch.inference_mode():
        before = runner.run(hist, noise, cond, cfg, N_TIMESTEPS)
    assert before is not None and not torch.equal(before, noise)
    assert generic_calls > 0 and calls > 0
    assert runner.stats()["optimized_captures"] == 1
    entry = next(iter(runner.cache.values()))
    captured_mode = next(
        mode for mode in entry.dispatch_modes if isinstance(mode, Conv1dWeightCache)
    )
    assert captured_mode.use_controlled_plan
    assert captured_mode.controlled_plan is not None
    assert runner.conv_cache is not captured_mode
    assert not runner.conv_cache.use_controlled_plan
    plan_ref = weakref.ref(captured_mode.controlled_plan)
    del captured_mode, entry
    captured_plan_calls = calls
    captured_generic_calls = generic_calls

    def unexpected_scan(self) -> None:
        raise AssertionError("Graph replay must not scan immutable weights")

    monkeypatch.setattr(Conv1dWeightCache, "validate_weights", unexpected_scan)
    with torch.inference_mode():
        repeated = runner.run(hist, noise, cond, cfg, N_TIMESTEPS)
    torch.cuda.synchronize(device)
    assert finite_byte_equal(repeated, before)
    assert calls == captured_plan_calls
    assert generic_calls == captured_generic_calls
    assert runner.stats()["captures"] == 1
    assert runner.stats()["hits"] == 1
    assert runner.stats()["optimization_fallbacks"] == 0
    gc.collect()
    assert plan_ref() is not None
    runner.cache.clear()
    gc.collect()
    assert plan_ref() is None


@pytest.mark.accelerator
def test_generic_convolution_does_not_require_controlled_plan(
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
    controlled_conv_flow: torch.nn.Module,
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "0")
    monkeypatch.setenv("FIREREDTTS3_CONV_LAYOUT", "conv2d_channels_last")
    monkeypatch.setenv("FIRERED_TTS3_FLOW_ATOL", "0")
    monkeypatch.setenv("FIRERED_TTS3_FLOW_RTOL", "0")
    failed_builds = 0

    def unavailable_plan(inputs: torch.Tensor, weight: torch.Tensor):
        nonlocal failed_builds
        failed_builds += 1
        raise RuntimeError("Controlled convolution support probe")

    monkeypatch.setattr(
        "sglang_omni.models.firered_tts3.conv_cache.ControlledConvPlan",
        unavailable_plan,
    )
    execution, fallback = prepare_flow_dit(controlled_conv_flow, "compile_bf16")
    runner = tiny_runner(
        execution,
        device,
        mode="compile_bf16",
        dit_fallback=fallback,
        capture_sizes=(2, 4),
    )
    generator = torch.Generator(device=device).manual_seed(819)
    for batch in (3, 4, 2, 3):
        hist, noise, cond = make_inputs(device, generator, batch)
        cfg = torch.full((batch, 1, 1), 1.2, device=device)
        with torch.inference_mode():
            actual = runner.run(hist, noise, cond, cfg, N_TIMESTEPS)
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
        assert actual is not None and torch.equal(actual, expected)
    assert failed_builds == 0
    assert runner.enabled and runner.dit is execution
    assert runner.conv_cache is not None and not runner.conv_cache.use_controlled_plan
    assert runner.conv_cache.controlled_plan is None
    for entry in runner.cache.values():
        assert entry.dispatch_modes == (runner.conv_cache,)
    assert runner.stats()["optimized_captures"] == 2
    assert runner.stats()["optimization_fallbacks"] == 0
    assert runner.stats()["captures"] == 2 and runner.stats()["hits"] == 2
    assert runner.stats()["eager"] == 0


@pytest.mark.parametrize(
    "with_time,with_conv,with_plan",
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_flow_entry_records_external_resources_once_per_replay_stream(
    monkeypatch: pytest.MonkeyPatch,
    with_time: bool,
    with_conv: bool,
    with_plan: bool,
) -> None:
    class RecordingTensor:
        def __init__(self) -> None:
            self.streams: list[object] = []

        def record_stream(self, stream) -> None:
            self.streams.append(stream)

    class FakeConvCache:
        pass

    monkeypatch.setattr(flow_graph_module, "Conv1dWeightCache", FakeConvCache)
    static = [RecordingTensor() for _ in range(3)]
    span = RecordingTensor()
    time_outputs = [RecordingTensor() for _ in range(2)] if with_time else []
    packed_weights = [RecordingTensor() for _ in range(2)] if with_conv else []
    workspace = RecordingTensor() if with_plan else None
    table = (
        SimpleNamespace(
            steps=tuple(
                {"gemm": SimpleNamespace(output=value)} for value in time_outputs
            )
        )
        if with_time
        else None
    )
    modes = []
    if with_conv:
        mode = FakeConvCache()
        mode.weights = {
            index: SimpleNamespace(packed=value)
            for index, value in enumerate(packed_weights)
        }
        mode.controlled_plan = (
            SimpleNamespace(workspace=workspace) if workspace is not None else None
        )
        modes.append(mode)
    entry = flow_graph_module.FlowGraphEntry(
        graph=object(),
        xt=static[0],
        cond=static[1],
        cfg=static[2],
        time_table=table,
        dispatch_modes=tuple(modes),
        t_span=span,
    )
    resources = static + [span] + time_outputs + packed_weights
    if workspace is not None:
        resources.append(workspace)
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
@pytest.mark.parametrize("failure", ["support", "precision", "capture", "replay"])
def test_convolution_failure_retains_validated_time_cache(
    flow_dispatch_dit, device, monkeypatch, failure: str
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "1")
    monkeypatch.setenv("FIREREDTTS3_CONV_LAYOUT", "conv2d_channels_last")
    execution, fallback = prepare_flow_dit(flow_dispatch_dit, "compile_bf16")
    runner = tiny_runner(
        execution,
        device,
        mode="compile_bf16",
        dit_fallback=fallback,
        capture_sizes=(4,),
    )
    original_loop = flow_graph_module.euler_flow_loop
    attempted_plans = set()
    time_graph_captures = 0

    def fail_convolution(*args, **kwargs):
        nonlocal time_graph_captures
        patch = original_loop(*args, **kwargs)
        conv_modes = [
            mode
            for mode in kwargs.get("dispatch_modes", ())
            if isinstance(mode, Conv1dWeightCache)
        ]
        if conv_modes:
            attempted_plans.add(conv_modes[0].use_controlled_plan)
            if failure == "support":
                raise RuntimeError("Unsupported convolution plan")
            elif failure == "capture" and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Convolution capture failed")
            elif failure == "precision" or (
                failure == "replay" and torch.cuda.is_current_stream_capturing()
            ):
                patch.add_(1.0)
        elif torch.cuda.is_current_stream_capturing():
            time_graph_captures += 1
        return patch

    monkeypatch.setattr(flow_graph_module, "euler_flow_loop", fail_convolution)
    generator = torch.Generator(device=device).manual_seed(83)
    for batch in (3, 4):
        hist, noise, cond = make_inputs(device, generator, batch)
        cfg = torch.full((batch, 1, 1), 1.2, device=device)
        with torch.inference_mode():
            actual = runner.run(hist, noise, cond, cfg, N_TIMESTEPS)
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
        assert actual is not None and torch.equal(actual, expected)
    entry = next(iter(runner.cache.values()))
    assert entry.time_table is not None
    assert len(entry.dispatch_modes) == 1
    assert isinstance(entry.dispatch_modes[0], TimeGemmMode)
    expected_plans = (
        {False, True}
        if torch.cuda.get_device_capability(device) == (12, 0)
        else {False}
    )
    assert attempted_plans == expected_plans
    assert time_graph_captures == 1
    assert runner.stats()["optimized_captures"] == 1
    assert runner.stats()["optimization_fallbacks"] == 1
    assert runner.stats()["hits"] == 1 and runner.stats()["eager"] == 0


@pytest.mark.accelerator
@pytest.mark.parametrize("failed_probe", [1, 2])
def test_all_time_probes_must_pass_before_trying_convolution(
    flow_dispatch_dit, device, monkeypatch, failed_probe: int
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "1")
    monkeypatch.setenv("FIREREDTTS3_CONV_LAYOUT", "conv2d_channels_last")
    execution, fallback = prepare_flow_dit(flow_dispatch_dit, "compile_bf16")
    runner = tiny_runner(
        execution,
        device,
        mode="compile_bf16",
        dit_fallback=fallback,
        capture_sizes=(1,),
    )
    original_loop = flow_graph_module.euler_flow_loop
    time_calls = 0
    conv_calls = 0

    def fail_later_time_probe(*args, **kwargs):
        nonlocal time_calls, conv_calls
        patch = original_loop(*args, **kwargs)
        modes = kwargs.get("dispatch_modes", ())
        if any(isinstance(mode, Conv1dWeightCache) for mode in modes):
            conv_calls += 1
            raise RuntimeError("Convolution must not run before time validation")
        elif any(isinstance(mode, TimeGemmMode) for mode in modes):
            if time_calls == failed_probe:
                patch.add_(1.0)
            time_calls += 1
        return patch

    monkeypatch.setattr(flow_graph_module, "euler_flow_loop", fail_later_time_probe)
    hist, noise, cond = make_inputs(
        device, torch.Generator(device=device).manual_seed(89), 1
    )
    cfg = torch.ones(1, 1, 1, device=device)
    with torch.inference_mode():
        actual = runner.run(hist, noise, cond, cfg, N_TIMESTEPS)
    assert torch.equal(actual, noise)
    entry = next(iter(runner.cache.values()))
    assert entry.time_table is None and entry.dispatch_modes == ()
    assert time_calls == failed_probe + 1 and conv_calls == 0


@pytest.mark.accelerator
def test_time_tables_follow_graph_lru_eviction(flow_dispatch_dit, device, monkeypatch):
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "1")
    monkeypatch.setenv("FIREREDTTS3_CONV_LAYOUT", "conv1d")
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


@pytest.mark.accelerator
@pytest.mark.parametrize(
    "failure",
    [
        "wrong_result",
        "capture_exception",
        "unsupported_shape",
        "bounded_result",
        "cumulative_probe",
        "cumulative_replay",
    ],
)
def test_ffn_failure_retains_existing_time_and_conv_graph(
    tiny_dit: DiT,
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("FFN capture optimization requires SM120")

    # note (Codex): Diagonal power-of-two convolutions isolate FFN fallback from reduction noise.
    with torch.no_grad():
        for layer in tiny_dit.modules():
            if isinstance(layer, torch.nn.Conv1d):
                layer.weight.zero_()
                layer.bias.zero_()
                channels = torch.arange(layer.out_channels, device=device)
                layer.weight[channels, channels, layer.kernel_size[0] // 2] = 0.5
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "1")
    monkeypatch.setenv("FIREREDTTS3_CONV_LAYOUT", "conv2d_channels_last")
    cumulative = failure.startswith("cumulative_")
    atol = "0.125" if cumulative else "1" if failure == "bounded_result" else "0"
    monkeypatch.setenv("FIRERED_TTS3_FLOW_ATOL", atol)
    monkeypatch.setenv("FIRERED_TTS3_FLOW_RTOL", "0")
    monkeypatch.setenv("FIRERED_TTS3_FFN_GEMM", "0")
    execution, fallback = prepare_flow_dit(tiny_dit, "compile_bf16")
    base = tiny_runner(
        execution,
        device,
        mode="compile_bf16",
        dit_fallback=fallback,
        capture_sizes=(1,),
    )
    generator = torch.Generator(device=device).manual_seed(337)
    inputs = [make_inputs(device, generator, 1) for _ in range(2)]
    cfg = torch.full((1, 1, 1), 1.2, device=device)
    injected = []

    class FaultingFfn(TorchDispatchMode):
        def __init__(self, source: torch.nn.Module) -> None:
            super().__init__()
            self.weights = {
                layer.weight.data_ptr(): layer.weight
                for name, layer in source.named_modules()
                if isinstance(layer, torch.nn.Linear) and name.endswith(".mlp.ff.0.0")
            }
            assert self.weights
            self.hits = 0
            self.enabled = True
            self.reason = ""
            self.corrupted = 0
            self.capture_failures = 0
            injected.append(self)

        def supports(self, rows: int) -> bool:
            return rows in (24, 48, 96)

        def expected_hits(self, rows: int) -> int:
            return len(self.weights)

        def validate_weights(self) -> None:
            pass  # note (Codex): The injected failure preserves the original weights.

        def record_stream(self, stream: torch.cuda.Stream) -> None:
            for weight in self.weights.values():
                weight.record_stream(stream)

        def __torch_dispatch__(self, func, types, args=(), kwargs=None) -> torch.Tensor:
            kwargs = kwargs or {}
            fields = dict(zip(("self", "mat1", "mat2"), args))
            fields.update(kwargs)
            weight = fields.get("mat2")
            selected = (
                func is torch.ops.aten.addmm.out
                and isinstance(weight, torch.Tensor)
                and weight.data_ptr() in self.weights
            )
            if (
                selected
                and failure == "capture_exception"
                and torch.cuda.is_current_stream_capturing()
            ):
                self.capture_failures += 1
                raise RuntimeError("injected FFN capture launch failure")
            result = func(*args, **kwargs)
            if selected:
                self.hits += 1
                if failure in ("wrong_result", "bounded_result"):
                    # note (Codex): A real FFN error must reach the complete Flow gate.
                    result[:, 0].add_(0.125 if failure == "bounded_result" else 16.0)
                    self.corrupted += 1
            return result

    original_loop = flow_graph_module.euler_flow_loop

    def perturb_complete_flow(*args, **kwargs) -> torch.Tensor:
        patch = original_loop(*args, **kwargs)
        modes = kwargs.get("dispatch_modes", ())
        if cumulative:
            if any(isinstance(mode, Conv1dWeightCache) for mode in modes):
                patch.add_(3 / 32)
            if any(isinstance(mode, FaultingFfn) for mode in modes) and (
                failure == "cumulative_probe"
                or torch.cuda.is_current_stream_capturing()
            ):
                # note (Codex): each 3/32 increment fits the 1/8 budget; their sum does not.
                patch.add_(3 / 32)
        return patch

    monkeypatch.setattr(flow_graph_module, "euler_flow_loop", perturb_complete_flow)
    with torch.inference_mode():
        expected = [base.run(*values, cfg, N_TIMESTEPS) for values in inputs]
    assert all(value is not None for value in expected)
    assert base.stats()["optimized_captures"] == 1
    assert base.stats()["optimization_fallbacks"] == 0
    base_entry = next(iter(base.cache.values()))
    assert {type(mode).__name__ for mode in base_entry.dispatch_modes} == {
        "TimeGemmMode",
        "Conv1dWeightCache",
    }

    if failure != "unsupported_shape":
        monkeypatch.setattr(flow_graph_module, "FfnGemmMode", FaultingFfn)
    monkeypatch.delenv("FIRERED_TTS3_FFN_GEMM", raising=False)
    candidate = tiny_runner(
        execution,
        device,
        mode="compile_bf16",
        dit_fallback=fallback,
        capture_sizes=(1,),
    )
    with torch.inference_mode():
        got = [candidate.run(*values, cfg, N_TIMESTEPS) for values in inputs]
    assert all(value is not None for value in got)
    if failure == "bounded_result":
        assert any(
            not finite_byte_equal(actual, reference)
            for actual, reference in zip(got, expected)
        )
        assert all(
            candidate.precision_gate.compare(actual, reference).accepted
            for actual, reference in zip(got, expected)
        )
    else:
        assert all(
            finite_byte_equal(actual, reference)
            for actual, reference in zip(got, expected)
        )
    assert candidate.enabled and candidate.dit is execution
    stats = candidate.stats()
    assert stats["ffn_captures"] == int(failure == "bounded_result")
    assert stats["ffn_fallbacks"] == int(failure != "bounded_result")
    assert stats["optimized_captures"] == 1 and stats["optimization_fallbacks"] == 0
    assert stats["eager"] == 0 and stats["captures"] == 1 and stats["hits"] == 1
    entry = next(iter(candidate.cache.values()))
    expected_modes = {"TimeGemmMode", "Conv1dWeightCache"}
    if failure == "bounded_result":
        expected_modes.add("FaultingFfn")
    assert {type(mode).__name__ for mode in entry.dispatch_modes} == expected_modes
    assert entry.time_table is not None
    assert ("FFN optimization rejected" in caplog.text) is (failure != "bounded_result")
    if cumulative:
        assert injected[0].hits > 0
        stage = "ffn" if failure == "cumulative_probe" else "replay"
        assert f"tolerance gate failed for {stage}" in caplog.text
        reference = expected[0]
        assert candidate.precision_gate.compare(reference + 3 / 32, reference).accepted
        assert not candidate.precision_gate.compare(
            reference + 3 / 16, reference
        ).accepted
    elif failure == "unsupported_shape":
        assert candidate.ffn_mode is not None and not candidate.ffn_mode.enabled
    elif failure in ("wrong_result", "bounded_result"):
        assert injected[0].hits > 0 and injected[0].corrupted > 0
    else:
        assert injected[0].hits > 0 and injected[0].capture_failures == 1


def test_unsupported_ffn_source_preserves_native_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original_import = builtins.__import__

    def without_optional_kernel(name, *args, **kwargs):
        if (
            name == "triton"
            or name.startswith("triton.")
            or name.endswith("ffn_gemm_kernel")
        ):
            raise AssertionError(
                "Unsupported CPU source must not import the optional Triton kernel"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_optional_kernel)
    from sglang_omni.models.firered_tts3.ffngemm import FfnGemmMode

    mode = FfnGemmMode(tiny_cpu_dit())
    assert not mode.enabled and mode.reason and not mode.supports(24)
    inputs, weight, bias = (
        torch.arange(8.0).reshape(2, 4),
        torch.ones(4, 3),
        torch.ones(3),
    )
    expected, actual = torch.empty(2, 3), torch.empty(2, 3)
    with torch.inference_mode():
        torch.addmm(bias, inputs, weight, out=expected)
        with mode:
            torch.addmm(bias, inputs, weight, out=actual)
    assert finite_byte_equal(actual, expected)
    assert mode.hits == 0


def test_packed_qkv_preserves_source_rng_and_forward() -> None:
    source = tiny_cpu_dit()
    x = torch.randn(2, HISTORY_LENGTH + PATCH_SIZE, REDAE_DIM + COND_CHANNELS)
    t = torch.rand(2, 1, 1)
    original = {
        name: tensor.detach().clone() for name, tensor in source.state_dict().items()
    }
    with torch.no_grad():
        expected = source(x=x, t=t)
    rng = torch.random.get_rng_state().clone()
    packed = PackedQkvModel(source)
    assert torch.equal(torch.random.get_rng_state(), rng)
    with torch.no_grad():
        actual = packed.model(x=x, t=t)
        unchanged = source(x=x, t=t)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert finite_byte_equal(unchanged, expected)
    assert source.state_dict().keys() == original.keys()
    assert all(
        finite_byte_equal(value, original[name])
        for name, value in source.state_dict().items()
    )
    packed.validate_weights()


@pytest.mark.accelerator
@pytest.mark.parametrize(
    "batch_size,scenario",
    [
        (8, "success"),
        (16, "success"),
        (8, "packed_time_corruption"),
    ],
)
def test_large_packed_flow_routing_and_time_gate(
    flow_dispatch_dit: DiT,
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    batch_size: int,
    scenario: str,
) -> None:
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("DiT GEMM capture optimization requires SM120")
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "1")
    monkeypatch.setenv("FIREREDTTS3_CONV_LAYOUT", "conv2d_channels_last")
    monkeypatch.setenv("FIRERED_TTS3_FLOW_ATOL", "1")
    monkeypatch.setenv("FIRERED_TTS3_FLOW_RTOL", "0")
    monkeypatch.setenv("FIRERED_TTS3_FFN_GEMM", "0")
    execution, fallback = prepare_flow_dit(flow_dispatch_dit, "compile_bf16")
    base = tiny_runner(
        execution,
        device,
        mode="compile_bf16",
        dit_fallback=fallback,
        capture_sizes=(batch_size,),
    )
    generator = torch.Generator(device=device).manual_seed(113)
    inputs = make_inputs(device, generator, batch_size)
    cfg = torch.full((batch_size, 1, 1), 1.2, device=device)
    with torch.inference_mode():
        expected = base.run(*inputs, cfg, N_TIMESTEPS)
    assert expected is not None
    probes = []

    class PackedProbe(TorchDispatchMode):
        def __init__(self, source: torch.nn.Module) -> None:
            super().__init__()
            self.source = source
            self.packed_qkv = None
            self.packed_compiled = None
            self.weights = set()
            self.hits = 0
            self.enabled = True
            self.reason = ""
            probes.append(self)

        def prepare_packed(self, compile_backend, device: torch.device) -> None:
            self.packed_qkv = PackedQkvModel(self.source)
            self.packed_compiled = torch.compile(
                self.packed_qkv.model, backend=compile_backend, dynamic=False
            )
            self.weights = {
                layer.to_qkv.weight.data_ptr()
                for layer in self.packed_qkv.layers.values()
            }

        def supports(self, rows: int) -> bool:
            return rows in (192, 384)

        def expected_hits(self, rows: int) -> int:
            return len(self.weights)

        def validate_weights(self) -> None:
            self.packed_qkv.validate_weights()

        def record_stream(self, stream: torch.cuda.Stream) -> None:
            self.packed_qkv.record_stream(stream)

        def __torch_dispatch__(self, func, types, args=(), kwargs=None) -> object:
            kwargs = kwargs or {}
            fields = dict(zip(("self", "mat1", "mat2"), args))
            fields.update(kwargs)
            weight = fields.get("mat2")
            result = func(*args, **kwargs)
            if (
                func is torch.ops.aten.addmm.out
                and isinstance(weight, torch.Tensor)
                and weight.data_ptr() in self.weights
            ):
                self.hits += 1
            return result

    monkeypatch.setattr(flow_graph_module, "FfnGemmMode", PackedProbe)
    monkeypatch.setattr(compiled_dit_module, "FfnGemmMode", PackedProbe)
    original_loop = flow_graph_module.euler_flow_loop
    corrupted_calls = 0

    def perturb_packed_time(*args, **kwargs) -> torch.Tensor:
        nonlocal corrupted_calls
        patch = original_loop(*args, **kwargs)
        modes = kwargs.get("dispatch_modes", ())
        if (
            scenario == "packed_time_corruption"
            and any(isinstance(mode, PackedProbe) for mode in modes)
            and any(isinstance(mode, TimeGemmMode) for mode in modes)
        ):
            corrupted_calls += 1
            patch.add_(0.25)
        return patch

    monkeypatch.setattr(flow_graph_module, "euler_flow_loop", perturb_packed_time)
    monkeypatch.delenv("FIRERED_TTS3_FFN_GEMM", raising=False)
    runner = tiny_runner(
        execution,
        device,
        mode="compile_bf16",
        dit_fallback=fallback,
        capture_sizes=(batch_size,),
    )

    def reject_replay_scan() -> None:
        raise AssertionError("Captured replay must not rescan packed weights")

    with torch.inference_mode():
        got = runner.run(*inputs, cfg, N_TIMESTEPS)
        hits_after_capture = probes[0].hits
        monkeypatch.setattr(probes[0], "validate_weights", reject_replay_scan)
        repeated = runner.run(*inputs, cfg, N_TIMESTEPS)
    assert finite_byte_equal(got, expected) and finite_byte_equal(repeated, expected)
    assert hits_after_capture > 0 and probes[0].hits == hits_after_capture
    entry = next(iter(runner.cache.values()))
    failed = scenario == "packed_time_corruption"
    assert runner.stats()["ffn_captures"] == int(not failed)
    assert runner.stats()["ffn_fallbacks"] == int(failed)
    assert (
        runner.stats()["optimization_fallbacks"] == 0 and runner.stats()["eager"] == 0
    )
    assert any(isinstance(mode, PackedProbe) for mode in entry.dispatch_modes) is (
        not failed
    )
    assert any(isinstance(mode, TimeGemmMode) for mode in entry.dispatch_modes)
    assert any(isinstance(mode, Conv1dWeightCache) for mode in entry.dispatch_modes)
    assert entry.time_table is not None
    if failed:
        assert (
            corrupted_calls > 0 and "exact gate failed for packed_time" in caplog.text
        )
