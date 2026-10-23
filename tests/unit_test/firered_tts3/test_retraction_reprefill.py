# SPDX-License-Identifier: Apache-2.0
"""Retraction resumes the pending feed's post-work and defers prefill stops."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.models.firered_tts3.model_runner import (
    DS_STAGE_PATCH,
    DS_STAGE_TEXT,
    TEXT_MODE_AUDIO,
    TEXT_MODE_AUDIO_ZERO,
    TEXT_MODE_FINALIZE,
    TEXT_MODE_SOSP,
    TEXT_MODE_TEXT,
    FireRedTTS3ModelRunner,
    FlowRow,
    FlowState,
)
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.prompt_utils import (
    AUDIO_SOS_ID,
    TEXT_EOT_ID,
    TEXT_TOKEN_BUDGET,
)
from sglang_omni.models.firered_tts3.request_builders import (
    CONTROL_TOKEN_ID,
    FireRedTTS3SGLangRequestData,
)
from tests.unit_test.fixtures.firered_tts3 import ReplayModel


def make_runner(*, batching: bool = False) -> FireRedTTS3ModelRunner:
    runner = FireRedTTS3ModelRunner.__new__(FireRedTTS3ModelRunner)
    runner.async_enabled = False
    runner.active_requests = {}
    runner.model = ReplayModel()
    runner.flow_batching = batching
    runner.stop_predicted = Mock(return_value=False)

    def prepare_flow_row(
        data: FireRedTTS3SGLangRequestData, hidden: torch.Tensor
    ) -> FlowRow:
        flow = data.flow_state
        flow.backbone_cond = torch.cat(
            [flow.backbone_cond, hidden[-1:].unsqueeze(0)], dim=1
        )
        return data, hidden, torch.empty(0), torch.empty(0)

    def run_flow_rows(rows: list[FlowRow]) -> None:
        for data, _hidden, _noise, _cond in rows:
            data.flow_state.patch_count += 1
            data.latent_patches.append(
                torch.full((1, 4, 64), float(data.flow_state.patch_count))
            )
            data.pending_feedback_queue.append(torch.ones(8))

    runner.prepare_flow_row = Mock(side_effect=prepare_flow_row)
    runner.run_flow_rows = Mock(side_effect=run_flow_rows)
    return runner


def make_request(
    *, dual: bool = False, task_type: str = "voice_design"
) -> SimpleNamespace:
    data = FireRedTTS3SGLangRequestData(
        state=FireRedTTS3State(
            variant="speaker" if dual else "instruct",
            task_type=task_type,
            input_ids=[11, 22, 33],
            text_stream_ids=list(range(101, 131)),
            text_pad_id=151671,
        ),
        dual_stream=dual,
        interleave_ratio=(3, 3) if dual else None,
        prefill_len=3,
        text_ar_mode=TEXT_MODE_AUDIO if dual or task_type == "tts" else TEXT_MODE_TEXT,
    )
    data.req = SimpleNamespace(
        output_ids=[],
        prefix_indices=[],
        is_retracted=False,
        extend_range=SimpleNamespace(length=3),
        finished_reason=None,
    )
    data.req.finished = lambda: data.req.finished_reason is not None
    data.flow_state = FlowState(
        generator=torch.Generator().manual_seed(0),
        prompt_patch_count=0,
        spk_cond_dit=None,
        backbone_cond=torch.zeros(1, 2, 8),
        latents_gen=torch.zeros(1, 4, 64),
        patch_count=0,
    )
    return SimpleNamespace(request_id="retracted", data=data)


def make_result(rows: int, sampled: int = 123) -> SimpleNamespace:
    return SimpleNamespace(
        next_token_ids=torch.tensor([sampled]),
        logits_output=SimpleNamespace(
            hidden_states=torch.arange(rows, dtype=torch.float32)
            .unsqueeze(-1)
            .expand(-1, 8)
        ),
    )


def prefill(
    runner: FireRedTTS3ModelRunner, request: SimpleNamespace, sampled: int = 123
) -> int:
    data = request.data
    data.req.extend_range.length = data.prefill_len + len(data.req.output_ids)
    result = make_result(data.req.extend_range.length, sampled)
    runner.post_prefill(result, None, None, [request])
    return int(result.next_token_ids[0])


def replay(runner: FireRedTTS3ModelRunner, request: SimpleNamespace) -> torch.Tensor:
    data = request.data
    data.req.extend_range.length = data.prefill_len + len(data.req.output_ids)
    batch = SimpleNamespace(
        input_ids=torch.zeros(data.req.extend_range.length, dtype=torch.long)
    )
    runner.before_prefill(batch, None, [request])
    assert batch.input_embeds.shape == (data.req.extend_range.length, 8)
    return batch.input_embeds


@pytest.mark.parametrize(
    "samples", [[TEXT_EOT_ID], [101, TEXT_EOT_ID], [101] * TEXT_TOKEN_BUDGET]
)
def test_single_history_invariant_at_every_pause(samples: list[int]) -> None:
    runner, request = make_runner(), make_request()
    data = request.data
    for sampled in [*samples, 0, 0, 0, 0]:
        replay(runner, request)
        generated = data.req.extend_range.length - data.prefill_len
        assert (
            generated
            == len(data.req.output_ids)
            == len(data.latent_patches) + runner.text_output_tokens(data)
        )
        previous_mode = data.text_ar_mode
        previous_cond = data.flow_state.backbone_cond
        runner.prepare_flow_row.reset_mock()
        token = prefill(runner, request, sampled)
        if previous_mode in (TEXT_MODE_FINALIZE, TEXT_MODE_SOSP):
            assert token == AUDIO_SOS_ID
            assert data.text_ar_mode == (
                TEXT_MODE_SOSP
                if previous_mode == TEXT_MODE_FINALIZE
                else TEXT_MODE_AUDIO_ZERO
            )
            assert data.flow_state.backbone_cond is previous_cond
            runner.prepare_flow_row.assert_not_called()
        elif previous_mode == TEXT_MODE_AUDIO_ZERO:
            assert token == CONTROL_TOKEN_ID and data.text_ar_mode == TEXT_MODE_AUDIO
            assert runner.prepare_flow_row.call_count == 1
        else:
            pass
        data.req.output_ids.append(token)
    assert data.text_ar_mode == TEXT_MODE_AUDIO
    assert len(data.latent_patches) == 2
    replay(runner, request)


@pytest.mark.parametrize("batching", [False, True])
@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize(
    "stage,index",
    [(DS_STAGE_TEXT, i) for i in range(3)] + [(DS_STAGE_PATCH, j) for j in range(3)],
)
def test_dual_reprefill_resumes_each_pause(
    stage: str, index: int, group: int, batching: bool
) -> None:
    runner, request = make_runner(batching=batching), make_request(dual=True)
    data = request.data
    data.req.output_ids.append(prefill(runner, request))
    # Cover both the first group's zero-patch text pauses and later groups
    # whose cond history must be rebuilt before resuming the pending feed.
    for _ in range(6 * group + (index if stage == DS_STAGE_TEXT else 3 + index)):
        result = make_result(1)
        runner.post_decode(result, None, None, [request])
        data.req.output_ids.append(int(result.next_token_ids[0]))
    assert (data.ds_stage, data.ds_index) == (stage, index)
    patches_before, text_before = len(data.latent_patches), data.text_pos
    replay(runner, request)
    runner.prepare_flow_row.reset_mock()
    runner.run_flow_rows.reset_mock()
    runner.stop_predicted.reset_mock()
    token = prefill(runner, request)
    fires = (stage == DS_STAGE_TEXT and index == 2) or (
        stage == DS_STAGE_PATCH and index < 2
    )
    assert runner.prepare_flow_row.call_count == int(fires)
    assert runner.run_flow_rows.call_count == int(fires)
    assert len(data.latent_patches) == patches_before + int(fires)
    assert len(data.pending_feedback_queue) == int(fires)
    advances_text = (stage == DS_STAGE_TEXT and index < 2) or (
        stage == DS_STAGE_PATCH and index == 2
    )
    assert data.text_pos == text_before + int(advances_text)
    assert token == (
        data.state.text_stream_ids[text_before] if advances_text else CONTROL_TOKEN_ID
    )
    expected_stage = DS_STAGE_PATCH if fires else DS_STAGE_TEXT
    expected_index = (
        (index + 1)
        if (stage == DS_STAGE_TEXT and index < 2)
        or (stage == DS_STAGE_PATCH and index < 2)
        else 0
    )
    assert (data.ds_stage, data.ds_index) == (expected_stage, expected_index)
    assert runner.stop_predicted.call_count == int(
        stage == DS_STAGE_PATCH and index == 2
    )
    hidden = make_result(data.req.extend_range.length).logits_output.hidden_states
    expected_cond = torch.cat(
        [torch.zeros(1, 2, 8), runner.dual_stream_cond_rows(data, hidden)], dim=1
    )
    torch.testing.assert_close(data.flow_state.backbone_cond, expected_cond)


@pytest.mark.parametrize("ratio", [(2, 1), (3, 3)])
def test_dual_history_and_replay_invariants_at_every_pause(
    ratio: tuple[int, int]
) -> None:
    runner, request = make_runner(), make_request(dual=True)
    data = request.data
    data.interleave_ratio = ratio
    for _ in range(3 * sum(ratio) + 1):
        assert data.req.output_ids.count(CONTROL_TOKEN_ID) == len(data.latent_patches)
        runner.model.encoded_patches.clear()
        replay(runner, request)
        assert len(runner.model.encoded_patches) == len(data.latent_patches)
        for encoded, patch in zip(
            runner.model.encoded_patches, data.latent_patches, strict=True
        ):
            assert encoded is patch
        assert not data.pending_feedback_queue
        data.req.output_ids.append(prefill(runner, request))
    assert len(data.latent_patches) == 3 * ratio[1]
    assert data.text_pos == 3 * ratio[0] + 1
    assert (data.ds_stage, data.ds_index) == (DS_STAGE_TEXT, 0)


@pytest.mark.parametrize("dual", [False, True])
def test_prefill_stop_finishes_on_next_sync_decode(dual: bool) -> None:
    runner, request = make_runner(), make_request(dual=dual, task_type="tts")
    data = request.data
    if dual:
        data.req.output_ids.append(prefill(runner, request))
        for _ in range(5):
            result = make_result(1)
            runner.post_decode(result, None, None, [request])
            data.req.output_ids.append(int(result.next_token_ids[0]))
        assert (data.ds_stage, data.ds_index) == (DS_STAGE_PATCH, 2)
        replay(runner, request)
    else:
        data.state.variant = "base"
    runner.stop_predicted.return_value = True
    runner.prepare_flow_row.reset_mock()
    assert prefill(runner, request) == CONTROL_TOKEN_ID
    assert data.stop_pending and data.req.finished_reason is None
    forward = SimpleNamespace(input_ids=torch.tensor([CONTROL_TOKEN_ID]))
    runner.before_decode(forward, None, [request])
    assert forward.input_embeds.shape == (1, 8)
    runner.post_decode(make_result(1), None, None, [request])
    assert isinstance(data.req.finished_reason, FINISH_MATCHED_TOKEN)
    assert not data.stop_pending
    runner.prepare_flow_row.assert_not_called()


@pytest.mark.parametrize("defer_stop", [False, True])
@pytest.mark.parametrize("status", ["active", "retracted", "finished"])
def test_pending_stop_resolve_ownership(defer_stop: bool, status: str) -> None:
    runner, request = make_runner(), make_request(task_type="tts")
    data = request.data
    data.stop_pending = True
    data.req.is_retracted = status == "retracted"
    previous_finish = FINISH_MATCHED_TOKEN(99) if status == "finished" else None
    data.req.finished_reason = previous_finish
    result = make_result(1)
    runner.run_post_decode(result, [request], defer_stop=defer_stop)
    assert data.stop_pending is defer_stop
    if defer_stop:
        assert data.req.finished_reason is previous_finish
        runner.post_decode_resolve(None, result, None, None, [request])
    assert not data.stop_pending
    if status == "active":
        assert isinstance(data.req.finished_reason, FINISH_MATCHED_TOKEN)
    else:
        assert data.req.finished_reason is previous_finish
    assert int(result.next_token_ids[0]) == CONTROL_TOKEN_ID
    runner.prepare_flow_row.assert_not_called()
    runner.stop_predicted.assert_not_called()


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
def test_deferred_stop_survives_repeated_retraction_until_decode(
    dual: bool, batching: bool
) -> None:
    """Retraction after a prefill stop must not generate patches or consume control feeds as audio."""
    runner, request = make_stopped_request(dual=dual, batching=batching)
    data = request.data
    machine = (data.text_ar_mode, data.ds_stage, data.ds_index, data.text_pos)
    patches = tuple(data.latent_patches)
    for excess in range(1, 5):
        output_ids = list(data.req.output_ids)
        rows = replay(runner, request)[data.prefill_len :]
        expected = runner.model.get_input_embeddings()(
            torch.full((excess,), CONTROL_TOKEN_ID, dtype=torch.long)
        )
        torch.testing.assert_close(rows[-excess:], expected)
        cond = data.flow_state.backbone_cond
        assert prefill(runner, request) == CONTROL_TOKEN_ID
        assert data.flow_state.backbone_cond is cond
        assert data.req.output_ids == output_ids
        assert (
            data.text_ar_mode,
            data.ds_stage,
            data.ds_index,
            data.text_pos,
        ) == machine
        assert len(data.latent_patches) == len(patches)
        for actual, original in zip(data.latent_patches, patches, strict=True):
            assert torch.equal(actual, original)
        assert not data.pending_feedback_queue
        assert data.stop_pending and data.req.finished_reason is None
        data.req.output_ids.append(CONTROL_TOKEN_ID)
    replay(runner, request)
    runner.prepare_flow_row.assert_not_called()
    runner.run_flow_rows.assert_not_called()
    runner.stop_predicted.assert_not_called()
    forward = SimpleNamespace(input_ids=torch.tensor([CONTROL_TOKEN_ID]))
    runner.before_decode(forward, None, [request])
    assert forward.input_embeds.shape == (1, 8)
    runner.post_decode(make_result(1), None, None, [request])
    assert isinstance(data.req.finished_reason, FINISH_MATCHED_TOKEN)
    assert not data.stop_pending
    runner.prepare_flow_row.assert_not_called()
