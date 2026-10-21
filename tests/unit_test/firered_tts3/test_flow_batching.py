# SPDX-License-Identifier: Apache-2.0
"""Flow cross-request batching: collect per-request flow rows over one
engine step, run ONE batched DiT + PatchEncoder pass, split back in row order.

Contracts pinned here (driving the real runner post hooks with the real DiT):

* B=1 stays bit-exact against the serial path — the c=1 regression red line
  (an exact bucket hit replays the same graph on identical values);
* batched rows land on their own request: the engine consumes next_token_ids
  positionally against the request rows, so the batched pass must never
  reorder; distinct per-row inputs make a misaligned writeback diverge far
  beyond the batched-GEMM roundoff envelope (~5e-3);
* a row whose stop check fires is never collected — mixed stop/continue
  engine steps batch only the continuing rows;
* rows group by n_timesteps (a graph replays one loop count), so an
  n_timesteps-override row falls back to eager without disturbing the rest;
* the dual-stream patch-boundary contract (only patch-producing feeds
  collect a row) lives in test_dual_stream_machine.py.
"""

from __future__ import annotations

import collections
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.firered_tts3.flow_graph import FlowGraphRunner, flow_t_span
from sglang_omni.models.firered_tts3.model_runner import (
    FireRedTTS3ModelRunner,
    FlowState,
)
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.models.firered_tts3.request_builders import CONTROL_TOKEN_ID

from .conftest import CHECKPOINTS, requires_checkpoints
from .test_flow_step_oracle import FakeModel

BASE_CHECKPOINT = CHECKPOINTS["base"]

# Rows are drawn apart by O(1); the padded/exact-bucket batched-GEMM drift is
# bounded at ~5e-3 (see test_flow_graph.py's bucket tests), so these bounds
# separate a row swap (loud) from batching roundoff (quiet).
PATCH_RTOL = 1e-2
PATCH_ATOL = 5e-3


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


@pytest.fixture
def model(device) -> FakeModel:
    from sglang_omni.models.firered_tts3.config import load_checkpoint_hf_config

    config = load_checkpoint_hf_config(str(BASE_CHECKPOINT))
    assert config is not None
    return FakeModel(config, device)


def make_graphs(model, device, capture_sizes=(1, 2, 4)) -> FlowGraphRunner:
    return FlowGraphRunner(
        model.dit,
        flow_t_span(10, device),
        n_timesteps=10,
        patch_size=model.patch_size,
        history_length=model.history_length,
        capture_sizes=capture_sizes,
    )


def make_runner(model, device, *, flow_batching: bool, graphs=None):
    runner = object.__new__(FireRedTTS3ModelRunner)
    runner.async_enabled = False
    runner.model = model
    runner.t_span_cache = {}
    runner.flow_graphs = graphs
    runner.flow_batching = flow_batching
    return runner


def make_data(
    model,
    device,
    *,
    seed: int,
    stop_threshold: float = 1.1,
    n_timesteps: int = 10,
    patch_count: int = 0,
):
    """One steady-state audio-phase request (prompt-less cond window)."""
    generator = torch.Generator(device=device).manual_seed(seed + 1000)
    latents = torch.randn(
        1, model.history_length, model.redae_dim, device=device, generator=generator
    )
    spk_emb = torch.randn(1, 512, device=device, generator=generator)
    req = SimpleNamespace(finished_reason=None, is_retracted=False)
    req.finished = lambda: req.finished_reason is not None
    return SimpleNamespace(
        dual_stream=False,
        text_ar_mode="audio",
        stop_pending=False,
        flow_state=FlowState(
            generator=torch.Generator(device=device).manual_seed(seed),
            prompt_patch_count=0,
            spk_cond_dit=model.spk_proj_dit(spk_emb),
            backbone_cond=torch.zeros(
                1, model.history_patches, model.llm_hidden_size, device=device
            ),
            latents_gen=latents,
            patch_count=patch_count,
        ),
        # sigmoid never reaches 1.1: the stop head stays open so random hidden
        # rows cannot flake the comparison on the decision boundary.
        state=FireRedTTS3State(
            stop_threshold=stop_threshold,
            min_patches=6,
            n_timesteps=n_timesteps,
            inference_cfg=2.0,
        ),
        pending_feedback_queue=collections.deque(),
        latent_patches=[],
        req=req,
        probe=ProfilingProbe(enabled=False),
    )


def decode_step(runner, requests: list, hidden: torch.Tensor) -> SimpleNamespace:
    result = SimpleNamespace(
        next_token_ids=torch.zeros(len(requests), dtype=torch.long),
        logits_output=SimpleNamespace(hidden_states=hidden),
    )
    runner.post_decode(result, None, None, requests)
    return result


def serial_patches(
    model, device, datas: list, hidden: torch.Tensor, graphs=None
) -> list:
    """Reference: the legacy serial path, one post_decode per request."""
    runner = make_runner(model, device, flow_batching=False, graphs=graphs)
    for index, data in enumerate(datas):
        request = SimpleNamespace(request_id=f"r{index}", data=data)
        decode_step(runner, [request], hidden[index : index + 1])
    return [data.latent_patches[0] for data in datas]


@requires_checkpoints
@pytest.mark.accelerator
def test_batched_single_row_bit_exact(model, device) -> None:
    """Batching on, one request: every patch/feedback is torch.equal to the
    serial path (the c=1 regression contract)."""
    graphs = make_graphs(model, device)
    steps = 6
    generator = torch.Generator(device=device).manual_seed(77)
    hidden_rows = [
        torch.randn(1, model.llm_hidden_size, device=device, generator=generator)
        for _ in range(steps)
    ]
    arms = {}
    for batching in (False, True):
        runner = make_runner(model, device, flow_batching=batching, graphs=graphs)
        data = make_data(model, device, seed=5)
        request = SimpleNamespace(request_id="r0", data=data)
        for hidden in hidden_rows:
            result = decode_step(runner, [request], hidden)
            assert int(result.next_token_ids[0]) == CONTROL_TOKEN_ID
        arms[batching] = data
    serial, batched = arms[False], arms[True]
    assert len(batched.latent_patches) == steps
    for index, (got, want) in enumerate(
        zip(batched.latent_patches, serial.latent_patches)
    ):
        assert torch.equal(got, want), f"patch {index} diverged"
    for got, want in zip(batched.pending_feedback_queue, serial.pending_feedback_queue):
        assert torch.equal(got, want)
    assert torch.equal(batched.flow_state.latents_gen, serial.flow_state.latents_gen)


@requires_checkpoints
@pytest.mark.accelerator
def test_batched_rows_stay_aligned(model, device) -> None:
    """Three rows, one engine step: one padded replay, per-row writeback."""
    batch = 3
    generator = torch.Generator(device=device).manual_seed(88)
    hidden = torch.randn(
        batch, model.llm_hidden_size, device=device, generator=generator
    )
    want = serial_patches(
        model,
        device,
        [make_data(model, device, seed=i) for i in range(batch)],
        hidden,
        make_graphs(model, device),
    )

    graphs = make_graphs(model, device)
    runner = make_runner(model, device, flow_batching=True, graphs=graphs)
    datas = [make_data(model, device, seed=i) for i in range(batch)]
    requests = [
        SimpleNamespace(request_id=f"r{i}", data=data) for i, data in enumerate(datas)
    ]
    result = decode_step(runner, requests, hidden)

    assert [int(t) for t in result.next_token_ids] == [CONTROL_TOKEN_ID] * batch
    stats = graphs.stats()
    assert stats["calls"] == 1 and stats["padded"] == 1 and stats["captures"] == 1
    for index, data in enumerate(datas):
        assert data.flow_state.patch_count == 1
        assert len(data.latent_patches) == 1
        assert len(data.pending_feedback_queue) == 1
        torch.testing.assert_close(
            data.latent_patches[0], want[index], rtol=PATCH_RTOL, atol=PATCH_ATOL
        )


@requires_checkpoints
@pytest.mark.accelerator
def test_stopping_rows_are_not_collected(model, device) -> None:
    """A row whose stop check fires emits nothing and leaves the batch; the
    continuing rows still batch once, in row order."""
    weight = model.stop_head.weight  # [1, hidden]
    bias = float(model.stop_head.bias[0])
    direction = weight[0] / weight[0].norm().pow(2)
    stop_hidden = (direction * (8.0 - bias)).reshape(1, -1).to(device)

    generator = torch.Generator(device=device).manual_seed(99)
    rows_hidden = torch.randn(
        2, model.llm_hidden_size, device=device, generator=generator
    )
    hidden = torch.cat([rows_hidden[:1], stop_hidden, rows_hidden[1:]], dim=0)
    want = serial_patches(
        model,
        device,
        [make_data(model, device, seed=i) for i in (0, 2)],
        torch.cat([rows_hidden[:1], rows_hidden[1:]], dim=0),
        make_graphs(model, device),
    )

    graphs = make_graphs(model, device)
    runner = make_runner(model, device, flow_batching=True, graphs=graphs)
    datas = [
        make_data(model, device, seed=0),
        # min_patches met and the crafted hidden clears the 0.5 threshold.
        make_data(model, device, seed=1, stop_threshold=0.5, patch_count=6),
        make_data(model, device, seed=2),
    ]
    requests = [
        SimpleNamespace(request_id=f"r{i}", data=data) for i, data in enumerate(datas)
    ]
    result = decode_step(runner, requests, hidden)

    assert [int(t) for t in result.next_token_ids] == [CONTROL_TOKEN_ID] * 3
    assert datas[1].req.finished_reason is not None
    assert not datas[1].latent_patches
    assert not datas[1].pending_feedback_queue
    stats = graphs.stats()
    assert stats["calls"] == 1 and stats["padded"] == 0  # B=2 exact bucket hit
    for data, reference in zip((datas[0], datas[2]), want):
        assert data.req.finished_reason is None
        torch.testing.assert_close(
            data.latent_patches[0], reference, rtol=PATCH_RTOL, atol=PATCH_ATOL
        )


@requires_checkpoints
@pytest.mark.accelerator
def test_rows_group_by_n_timesteps(model, device) -> None:
    """An n_timesteps override cannot share the default graph: the override
    group falls back to eager while the default group replays — both B=1
    exact, so the batched pass is bit-exact against the serial path."""
    generator = torch.Generator(device=device).manual_seed(111)
    hidden = torch.randn(2, model.llm_hidden_size, device=device, generator=generator)
    want = serial_patches(
        model,
        device,
        [
            make_data(model, device, seed=0, n_timesteps=10),
            make_data(model, device, seed=1, n_timesteps=6),
        ],
        hidden,
        make_graphs(model, device),
    )

    graphs = make_graphs(model, device)
    runner = make_runner(model, device, flow_batching=True, graphs=graphs)
    datas = [
        make_data(model, device, seed=0, n_timesteps=10),
        make_data(model, device, seed=1, n_timesteps=6),
    ]
    requests = [
        SimpleNamespace(request_id=f"r{i}", data=data) for i, data in enumerate(datas)
    ]
    decode_step(runner, requests, hidden)

    for data, reference in zip(datas, want):
        assert torch.equal(data.latent_patches[0], reference)
    stats = graphs.stats()
    assert stats["calls"] == 2 and stats["captures"] == 1 and stats["eager"] == 1


@requires_checkpoints
@pytest.mark.accelerator
def test_post_decode_launch_matches_post_decode(model, device) -> None:
    """The async-decode launch hook runs the same side path as post_decode:
    identical tokens, patches, feedback, and stop state, and a None resolve
    payload (the base-class resolve is then a no-op)."""
    graphs = make_graphs(model, device)
    steps = 4
    generator = torch.Generator(device=device).manual_seed(131)
    hidden_rows = [
        torch.randn(1, model.llm_hidden_size, device=device, generator=generator)
        for _ in range(steps)
    ]
    arms = {}
    for mode in ("sync", "launch"):
        runner = make_runner(model, device, flow_batching=True, graphs=graphs)
        data = make_data(model, device, seed=5)
        request = SimpleNamespace(request_id="r0", data=data)
        for hidden in hidden_rows:
            result = SimpleNamespace(
                next_token_ids=torch.zeros(1, dtype=torch.long),
                logits_output=SimpleNamespace(hidden_states=hidden),
            )
            if mode == "sync":
                runner.post_decode(result, None, None, [request])
            else:
                launch_buf = runner.post_decode_launch(result, None, [request])
                assert launch_buf is None
                runner.post_decode_resolve(launch_buf, result, None, None, [request])
            assert int(result.next_token_ids[0]) == CONTROL_TOKEN_ID
        arms[mode] = data
    sync_data, launch_data = arms["sync"], arms["launch"]
    assert len(launch_data.latent_patches) == steps
    for index, (got, want) in enumerate(
        zip(launch_data.latent_patches, sync_data.latent_patches)
    ):
        assert torch.equal(got, want), f"patch {index} diverged"
    for got, want in zip(
        launch_data.pending_feedback_queue, sync_data.pending_feedback_queue
    ):
        assert torch.equal(got, want)
    assert torch.equal(
        launch_data.flow_state.latents_gen, sync_data.flow_state.latents_gen
    )


@requires_checkpoints
@pytest.mark.accelerator
def test_async_deferred_stop_contract(model, device) -> None:
    """Async-decode stop bridge: a launch-time stop defers to the resolve
    (the scheduler's overrun drop must not eat the finish), and the overrun
    launch for the already-stopped row runs no flow and needs no feedback."""
    model.graph_feedback_buffer = None
    runner = make_runner(model, device, flow_batching=False)
    data = make_data(model, device, seed=5, stop_threshold=0.5, patch_count=6)
    request = SimpleNamespace(request_id="r0", data=data)

    # A continuing step through the launch hook: patch produced, nothing pending.
    generator = torch.Generator(device=device).manual_seed(77)
    hidden_ok = torch.randn(
        1, model.llm_hidden_size, device=device, generator=generator
    )
    result = SimpleNamespace(
        next_token_ids=None,
        logits_output=SimpleNamespace(hidden_states=hidden_ok),
    )
    launch_buf = runner.post_decode_launch(result, None, [request])
    assert launch_buf is None
    assert not data.stop_pending and data.req.finished_reason is None
    assert len(data.latent_patches) == 1

    # The stopping step: stop head clears the 0.5 threshold (same crafted
    # hidden as test_stopping_rows_are_not_collected).
    weight = model.stop_head.weight
    bias = float(model.stop_head.bias[0])
    direction = weight[0] / weight[0].norm().pow(2)
    stop_hidden = (direction * (8.0 - bias)).reshape(1, -1).to(device)
    result = SimpleNamespace(
        next_token_ids=None,
        logits_output=SimpleNamespace(hidden_states=stop_hidden),
    )
    runner.post_decode_launch(result, None, [request])
    assert data.stop_pending and data.req.finished_reason is None
    assert len(data.latent_patches) == 1  # a stopping step emits no patch
    assert int(result.next_token_ids[0]) == CONTROL_TOKEN_ID

    # The overrun launch for the already-stopped row: before_decode feeds a
    # placeholder (the stopping step queued no feedback) and the launch runs
    # no flow; the row's output is discarded by the scheduler.
    forward_batch = SimpleNamespace(
        input_ids=torch.zeros(1, dtype=torch.long, device=device)
    )
    runner.before_decode(forward_batch, None, [request])
    assert forward_batch.input_embeds is not None
    result_overrun = SimpleNamespace(
        next_token_ids=None,
        logits_output=SimpleNamespace(hidden_states=stop_hidden),
    )
    runner.post_decode_launch(result_overrun, None, [request])
    assert len(data.latent_patches) == 1
    assert int(result_overrun.next_token_ids[0]) == CONTROL_TOKEN_ID

    # The resolve applies the deferred finish exactly once.
    runner.post_decode_resolve(None, result, None, None, [request])
    assert not data.stop_pending
    assert data.req.finished_reason is not None
