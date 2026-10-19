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
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

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


__all__ = []
