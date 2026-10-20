# SPDX-License-Identifier: Apache-2.0
"""Repeated retractions preserve a deferred stop until decode owns cleanup."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.models.firered_tts3.model_runner import (
    DS_STAGE_PATCH,
    FireRedTTS3ModelRunner,
)
from sglang_omni.models.firered_tts3.request_builders import CONTROL_TOKEN_ID
from tests.unit_test.firered_tts3.test_retraction_reprefill import (
    make_request,
    make_result,
    make_runner,
    prefill,
    replay,
)


def make_stopped_request(
    *, dual: bool, batching: bool = False
) -> tuple[FireRedTTS3ModelRunner, SimpleNamespace]:
    runner, request = make_runner(batching=batching), make_request(
        dual=dual, task_type="tts"
    )
    data = request.data
    if dual:
        data.req.output_ids.append(prefill(runner, request))
        for _ in range(5):
            result = make_result(1)
            runner.post_decode(result, None, None, [request])
            data.req.output_ids.append(int(result.next_token_ids[0]))
        assert (data.ds_stage, data.ds_index) == (DS_STAGE_PATCH, 2)
    replay(runner, request)
    runner.stop_predicted.return_value = True
    data.req.output_ids.append(prefill(runner, request))
    assert data.stop_pending and data.req.finished_reason is None
    assert len(data.latent_patches) == (3 if dual else 0)
    runner.prepare_flow_row.reset_mock()
    runner.run_flow_rows.reset_mock()
    runner.stop_predicted.reset_mock()
    return runner, request


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("batching", [False, True])
def test_pending_stop_retraction_replays_discardable_control(
    dual: bool, batching: bool
) -> None:
    runner, request = make_stopped_request(dual=dual, batching=batching)
    data = request.data
    output_ids = list(data.req.output_ids)
    machine = (data.text_ar_mode, data.ds_stage, data.ds_index, data.text_pos)
    patches = tuple(data.latent_patches)
    feedback = tuple(data.pending_feedback_queue)
    runner.dual_stream_step = Mock(wraps=runner.dual_stream_step)
    runner.dual_stream_cond_rows = Mock(wraps=runner.dual_stream_cond_rows)
    runner.flow_step = Mock(wraps=runner.flow_step)
    runner.record_text_sample = Mock(wraps=runner.record_text_sample)

    replay(runner, request)
    cond = data.flow_state.backbone_cond
    rows = runner.build_replay_rows(runner.model, data, "cpu", torch.float32)
    assert rows.shape == (len(output_ids), 8)
    expected_control = runner.model.get_input_embeddings()(
        torch.tensor([CONTROL_TOKEN_ID])
    )
    torch.testing.assert_close(rows[-1:], expected_control)
    assert tuple(data.pending_feedback_queue) == feedback
    assert prefill(runner, request) == CONTROL_TOKEN_ID
    assert data.flow_state.backbone_cond is cond
    assert data.req.output_ids == output_ids
    assert (data.text_ar_mode, data.ds_stage, data.ds_index, data.text_pos) == machine
    assert len(data.latent_patches) == len(patches)
    assert all(
        current is patch
        for current, patch in zip(data.latent_patches, patches, strict=True)
    )
    assert tuple(data.pending_feedback_queue) == feedback
    assert data.stop_pending and data.req.finished_reason is None
    runner.dual_stream_step.assert_not_called()
    runner.dual_stream_cond_rows.assert_not_called()
    runner.flow_step.assert_not_called()
    runner.record_text_sample.assert_not_called()
    runner.prepare_flow_row.assert_not_called()
    runner.run_flow_rows.assert_not_called()
    runner.stop_predicted.assert_not_called()

    data.req.output_ids.append(CONTROL_TOKEN_ID)
    forward = SimpleNamespace(input_ids=torch.tensor([CONTROL_TOKEN_ID]))
    runner.before_decode(forward, None, [request])
    runner.post_decode(make_result(1), None, None, [request])
    assert isinstance(data.req.finished_reason, FINISH_MATCHED_TOKEN)
    assert not data.stop_pending
    runner.prepare_flow_row.assert_not_called()


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("retractions", [2, 4])
def test_pending_stop_repeated_retraction_preserves_excess(
    dual: bool, retractions: int
) -> None:
    runner, request = make_stopped_request(dual=dual)
    data = request.data
    machine = (data.text_ar_mode, data.ds_stage, data.ds_index, data.text_pos)
    patch_count = len(data.latent_patches)
    for excess in range(1, retractions + 1):
        recorded_feeds = (
            data.req.output_ids.count(CONTROL_TOKEN_ID)
            if dual
            else len(data.req.output_ids) - runner.text_output_tokens(data)
        )
        assert recorded_feeds - patch_count == excess
        replay(runner, request)
        rows = runner.build_replay_rows(runner.model, data, "cpu", torch.float32)
        assert rows.shape == (len(data.req.output_ids), 8)
        control_ids = torch.full((excess,), CONTROL_TOKEN_ID, dtype=torch.long)
        expected = runner.model.get_input_embeddings()(control_ids)
        torch.testing.assert_close(rows[-excess:], expected)
        assert not data.pending_feedback_queue
        data.req.output_ids.append(prefill(runner, request))
        assert data.req.output_ids[-1] == CONTROL_TOKEN_ID
        assert (
            data.text_ar_mode,
            data.ds_stage,
            data.ds_index,
            data.text_pos,
        ) == machine
        assert len(data.latent_patches) == patch_count
        assert data.stop_pending and data.req.finished_reason is None
    replay(runner, request)
    runner.prepare_flow_row.assert_not_called()
    runner.run_flow_rows.assert_not_called()
    runner.stop_predicted.assert_not_called()

    # Extra controls are legal only while the deferred stop is still pending.
    data.stop_pending = False
    with pytest.raises(AssertionError, match="history|more patch feeds"):
        replay(runner, request)
