# SPDX-License-Identifier: Apache-2.0
"""Dual-stream decode machine (speaker variant): the (nt, na) group schedule."""

from __future__ import annotations

import collections
from types import SimpleNamespace

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.models.firered_tts3.model_runner import (
    DS_STAGE_PATCH,
    DS_STAGE_TEXT,
    FireRedTTS3ModelRunner,
)
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.request_builders import (
    CONTROL_TOKEN_ID,
    FireRedTTS3SGLangRequestData,
)

TEXT_PAD_ID = 151671


def make_runner(*, stop_after: int | None = None) -> FireRedTTS3ModelRunner:
    """Runner with the side path stubbed: run_flow records (hidden, count),
    stop_predicted fires once patch_count reaches stop_after."""
    runner = FireRedTTS3ModelRunner.__new__(FireRedTTS3ModelRunner)
    runner.async_enabled = False
    runner.active_requests = {}
    runner.t_span_cache = {}
    runner.flow_calls: list[tuple[int, float]] = []
    runner.stop_checks: list[int] = []

    def run_flow(data, hidden_f32, flow_rows=None) -> None:
        data.flow_state.patch_count += 1
        data.latent_patches.append(torch.zeros(1, 4, 64))
        runner.flow_calls.append((data.flow_state.patch_count, float(hidden_f32.sum())))

    def stop_predicted(data, hidden_f32) -> bool:
        runner.stop_checks.append(data.flow_state.patch_count)
        return stop_after is not None and data.flow_state.patch_count >= stop_after

    runner.run_flow = run_flow  # type: ignore[attr-defined]
    runner.stop_predicted = stop_predicted  # type: ignore[attr-defined]
    return runner


def make_request(text_ids=(101, 102, 103, 104, 105), ratio=(2, 1)) -> SimpleNamespace:
    data = FireRedTTS3SGLangRequestData(
        state=FireRedTTS3State(
            variant="speaker",
            input_ids=[11, 22, 33],
            dual_stream=True,
            text_stream_ids=list(text_ids),
            text_pad_id=TEXT_PAD_ID,
            stop_threshold=0.5,
            min_patches=0,
        ),
        dual_stream=True,
        interleave_ratio=ratio,
    )
    data.flow_state = SimpleNamespace(patch_count=0)
    data.pending_feedback_queue = collections.deque()
    data.prefill_len = 3
    data.req = SimpleNamespace(
        finished_reason=None,
        output_ids=[],
        extend_range=SimpleNamespace(length=3),
    )
    return SimpleNamespace(request_id="rid", data=data)


def make_result(batch: int = 1, hidden_rows: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        next_token_ids=torch.zeros(batch, dtype=torch.long),
        logits_output=SimpleNamespace(
            hidden_states=torch.zeros(batch * hidden_rows, 8)
        ),
    )


def decode_step(runner, request) -> int:
    result = make_result()
    runner.post_decode(result, None, None, [request])
    return int(result.next_token_ids[0])


def prefill_tail(runner, request) -> int:
    result = make_result(hidden_rows=3)
    runner.post_prefill(result, None, SimpleNamespace(is_prefill_only=False), [request])
    return int(result.next_token_ids[0])


def test_group_schedule_and_text_feeds() -> None:
    """[2,1]: text, text(flow), patch(check) repeating; pads once exhausted."""
    runner = make_runner()
    request = make_request(text_ids=(101, 102, 103))
    data = request.data

    # The prefill tail feeds the first text token and fires nothing.
    assert prefill_tail(runner, request) == 101
    assert data.text_pos == 1
    assert (data.ds_stage, data.ds_index) == (DS_STAGE_TEXT, 0)
    assert runner.flow_calls == []

    # First text feed of the group: inert.
    assert decode_step(runner, request) == 102
    assert (data.ds_stage, data.ds_index) == (DS_STAGE_TEXT, 1)
    assert runner.flow_calls == []
    assert runner.stop_checks == []

    # Last text feed: fires the flow (patch 1 emitted), patch feed follows.
    assert decode_step(runner, request) == CONTROL_TOKEN_ID
    assert (data.ds_stage, data.ds_index) == (DS_STAGE_PATCH, 0)
    assert [count for count, _ in runner.flow_calls] == [1]

    # Last (only) patch feed: stop check AFTER the patch was emitted; the
    # next group starts on a text feed.
    assert decode_step(runner, request) == 103
    assert runner.stop_checks == [1]  # patch_count includes the emitted patch
    assert (data.ds_stage, data.ds_index) == (DS_STAGE_TEXT, 0)

    # Group 2: text feed 1 (id 103), then the text stream exhausts into pads.
    assert decode_step(runner, request) == TEXT_PAD_ID
    assert decode_step(runner, request) == CONTROL_TOKEN_ID
    assert [count for count, _ in runner.flow_calls] == [1, 2]
    assert decode_step(runner, request) == TEXT_PAD_ID
    assert runner.stop_checks == [1, 2]


def test_stop_fires_after_the_patch_feed() -> None:
    """The stopping step's patch is already emitted (unlike the prefill
    form's stop-before-generate)."""
    runner = make_runner(stop_after=2)
    request = make_request()
    data = request.data

    prefill_tail(runner, request)
    decode_step(runner, request)  # text
    decode_step(runner, request)  # text + flow -> patch 1
    decode_step(runner, request)  # patch feed: check passes (1 < 2)
    decode_step(runner, request)  # text
    assert data.req.finished_reason is None
    decode_step(runner, request)  # text + flow -> patch 2
    assert data.req.finished_reason is None  # no stop on flow steps
    decode_step(runner, request)  # patch feed: stop fires
    finished = data.req.finished_reason
    assert finished is not None
    assert isinstance(finished, FINISH_MATCHED_TOKEN)
    assert len(data.latent_patches) == 2  # the stopping patch was emitted


def test_general_ratio_groups() -> None:
    """[2, 3]: text, text(flow), patch(flow), patch(flow), patch(check)."""
    runner = make_runner()
    request = make_request(ratio=(2, 3))
    data = request.data

    prefill_tail(runner, request)
    kinds: list[str] = []
    for _ in range(5):
        stage_before = (data.ds_stage, data.ds_index)
        flows_before = len(runner.flow_calls)
        checks_before = len(runner.stop_checks)
        decode_step(runner, request)
        fired = len(runner.flow_calls) > flows_before
        checked = len(runner.stop_checks) > checks_before
        kinds.append(
            f"{stage_before}:{'flow' if fired else ''}{'check' if checked else ''}"
        )
    assert kinds == [
        "('text', 0):",
        "('text', 1):flow",
        "('patch', 0):flow",
        "('patch', 1):flow",
        "('patch', 2):check",
    ]
    assert [count for count, _ in runner.flow_calls] == [1, 2, 3]
    assert (data.ds_stage, data.ds_index) == (DS_STAGE_TEXT, 0)


def make_batching_runner() -> FireRedTTS3ModelRunner:
    """Runner with the flow halves stubbed for collection tracking:
    prepare_flow_row tags the row with its request, run_flow_rows records
    each flushed batch. stop_predicted never fires."""
    runner = FireRedTTS3ModelRunner.__new__(FireRedTTS3ModelRunner)
    runner.async_enabled = False
    runner.active_requests = {}
    runner.t_span_cache = {}
    runner.flow_batching = True
    runner.flushed: list[list] = []

    def prepare_flow_row(data, hidden_f32):
        return (data, hidden_f32)

    def run_flow_rows(rows) -> None:
        runner.flushed.append([data for data, _hidden in rows])
        for data, _hidden in rows:
            data.flow_state.patch_count += 1

    runner.prepare_flow_row = prepare_flow_row  # type: ignore[attr-defined]
    runner.run_flow_rows = run_flow_rows  # type: ignore[attr-defined]
    runner.stop_predicted = lambda data, hidden_f32: False  # type: ignore[attr-defined]
    return runner


def test_batching_collects_only_patch_producing_rows() -> None:
    """One engine step, two requests: the one at an intermediate patch feed
    collects a flow row; the one at its group's LAST patch feed only runs
    the stop check and must not enter the batch."""
    runner = make_batching_runner()
    flowing = make_request(ratio=(2, 2))
    flowing.data.ds_stage = DS_STAGE_PATCH
    flowing.data.ds_index = 0  # intermediate patch feed: drives the flow
    checking = make_request(ratio=(2, 2))
    checking.data.ds_stage = DS_STAGE_PATCH
    checking.data.ds_index = 1  # last patch feed: stop check only

    result = make_result(batch=2)
    runner.post_decode(result, None, None, [flowing, checking])

    assert runner.flushed == [[flowing.data]]
    assert flowing.data.flow_state.patch_count == 1
    assert checking.data.flow_state.patch_count == 0
    assert int(result.next_token_ids[0]) == CONTROL_TOKEN_ID
    # Stop check passed: the next group opens with a text feed.
    assert int(result.next_token_ids[1]) == 101

    # Both at intermediate patch feeds: one batched flush in row order.
    runner.flushed.clear()
    flowing.data.ds_stage, flowing.data.ds_index = DS_STAGE_PATCH, 0
    checking.data.ds_stage, checking.data.ds_index = DS_STAGE_PATCH, 0
    result = make_result(batch=2)
    runner.post_decode(result, None, None, [flowing, checking])
    assert runner.flushed == [[flowing.data, checking.data]]

    # A step with no patch-producing row flushes nothing.
    checking.data.ds_stage = DS_STAGE_PATCH
    checking.data.ds_index = 1
    runner.flushed.clear()
    runner.post_decode(make_result(), None, None, [checking])
    assert runner.flushed == []
