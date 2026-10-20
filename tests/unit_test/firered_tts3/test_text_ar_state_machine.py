# SPDX-License-Identifier: Apache-2.0
"""Instruct text-AR state machine: transitions, budget, and boundary cases.

The reference machine (fireredtts3_instruct.generate): text steps sample real
tokens; sampling <|eot|> (or the 200th token) makes that token's own feed the
no-sampling finalize step; then one <|sosp|> feed; then the audio loop's
step 0 feeds <|sosp|> AGAIN (the doubled feed is the reference's actual
behavior). <|eot|> must never terminate the request.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.firered_tts3.model_runner import (
    TEXT_MODE_AUDIO,
    TEXT_MODE_AUDIO_ZERO,
    TEXT_MODE_FINALIZE,
    TEXT_MODE_SOSP,
    TEXT_MODE_TEXT,
    FireRedTTS3ModelRunner,
    FlowRow,
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
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor


def make_runner() -> FireRedTTS3ModelRunner:
    runner = FireRedTTS3ModelRunner.__new__(FireRedTTS3ModelRunner)
    runner._async_enabled = False
    runner.active_requests = {}
    runner.t_span_cache = {}
    runner.flow_calls = 0
    runner.flow_step = lambda data, hidden, flow_rows=None, defer_stop=False: setattr(  # type: ignore[attr-defined]
        runner, "flow_calls", runner.flow_calls + 1
    )
    return runner


def make_request(task_type: str = "voice_design") -> SimpleNamespace:
    data = FireRedTTS3SGLangRequestData(state=FireRedTTS3State(task_type=task_type))
    data.text_ar_mode = TEXT_MODE_TEXT
    return SimpleNamespace(request_id="rid", data=data)


def make_result(next_ids: list[int]) -> SimpleNamespace:
    batch = len(next_ids)
    return SimpleNamespace(
        next_token_ids=torch.tensor(next_ids, dtype=torch.long),
        logits_output=SimpleNamespace(hidden_states=torch.zeros(batch, 8)),
    )


def decode_step(runner, request, sampled: int) -> int:
    result = make_result([sampled])
    runner.post_decode(result, None, None, [request])
    return int(result.next_token_ids[0])


def test_text_machine_immediate_eot() -> None:
    """Empty text: the very first sampled token is EOT (degenerate plan)."""
    runner = make_runner()
    request = make_request()
    data = request.data

    # The prefill tail samples EOT right away.
    token = runner.record_text_sample(data, torch.tensor(TEXT_EOT_ID))
    assert token == TEXT_EOT_ID
    assert data.text_ar_mode == TEXT_MODE_FINALIZE
    assert data.text_token_ids == []  # EOT is never recorded
    assert data.eot_sampled is True

    # Finalize step: feeds embed(EOT), the runner steers to sosp (no sampling).
    assert decode_step(runner, request, 0) == AUDIO_SOS_ID
    assert data.text_ar_mode == TEXT_MODE_SOSP
    # The sosp feed; the machine arms the doubled feed for audio step 0.
    assert decode_step(runner, request, 0) == AUDIO_SOS_ID
    assert data.text_ar_mode == TEXT_MODE_AUDIO_ZERO
    assert runner.flow_calls == 0
    # Audio step 0: the doubled sosp feed drives the first flow step.
    assert decode_step(runner, request, 0) == CONTROL_TOKEN_ID
    assert data.text_ar_mode == TEXT_MODE_AUDIO
    assert runner.flow_calls == 1
    # Steady state.
    assert decode_step(runner, request, 0) == CONTROL_TOKEN_ID
    assert runner.flow_calls == 2


def test_text_machine_single_token() -> None:
    runner = make_runner()
    request = make_request()
    data = request.data

    token = runner.record_text_sample(data, torch.tensor(100))
    assert token == 100
    assert data.text_ar_mode == TEXT_MODE_TEXT
    assert data.text_token_ids == [100]
    assert data.text_fed == 1

    token = runner.record_text_sample(data, torch.tensor(TEXT_EOT_ID))
    assert data.text_ar_mode == TEXT_MODE_FINALIZE
    assert data.text_token_ids == [100]
    assert data.text_fed == 2


def test_text_machine_full_budget() -> None:
    """200 samples without EOT: the 200th token's feed becomes the finalize."""
    runner = make_runner()
    request = make_request()
    data = request.data

    for index in range(TEXT_TOKEN_BUDGET):
        assert data.text_ar_mode == TEXT_MODE_TEXT
        runner.record_text_sample(data, torch.tensor(1000 + index))
    assert data.text_ar_mode == TEXT_MODE_FINALIZE
    assert len(data.text_token_ids) == TEXT_TOKEN_BUDGET
    assert data.eot_sampled is False

    # The transition sequence still runs: finalize -> sosp -> audio_zero.
    assert decode_step(runner, request, 0) == AUDIO_SOS_ID
    assert decode_step(runner, request, 0) == AUDIO_SOS_ID
    assert data.text_ar_mode == TEXT_MODE_AUDIO_ZERO
    decode_step(runner, request, 0)
    assert data.text_ar_mode == TEXT_MODE_AUDIO


def test_text_machine_budget_boundary_at_199() -> None:
    """EOT arriving as the 199th sample takes the EOT branch (not the budget)."""
    runner = make_runner()
    request = make_request()
    data = request.data
    for index in range(TEXT_TOKEN_BUDGET - 1):
        runner.record_text_sample(data, torch.tensor(1000 + index))
    assert data.text_ar_mode == TEXT_MODE_TEXT
    runner.record_text_sample(data, torch.tensor(TEXT_EOT_ID))
    assert data.text_ar_mode == TEXT_MODE_FINALIZE
    assert data.eot_sampled is True
    assert len(data.text_token_ids) == TEXT_TOKEN_BUDGET - 1


def test_text_output_tokens_tracks_the_machine() -> None:
    """The output_ids layout for re-prefill: text(+eot) + sosp rows + patches."""
    runner = make_runner()
    request = make_request()
    data = request.data

    # Mid-text: only the recorded text tokens exist in output_ids.
    data.text_token_ids = [10, 11, 12]
    assert runner.text_output_tokens(data) == 3

    # After EOT: the EOT feed row joins the region.
    data.eot_sampled = True
    data.text_ar_mode = TEXT_MODE_FINALIZE
    assert runner.text_output_tokens(data) == 4

    data.text_ar_mode = TEXT_MODE_SOSP
    assert runner.text_output_tokens(data) == 5

    data.text_ar_mode = TEXT_MODE_AUDIO_ZERO
    assert runner.text_output_tokens(data) == 6

    data.text_ar_mode = TEXT_MODE_AUDIO
    assert runner.text_output_tokens(data) == 6

    # Audio-mode requests without a text phase contribute nothing.
    plain = FireRedTTS3SGLangRequestData(state=FireRedTTS3State(task_type="tts"))
    assert runner.text_output_tokens(plain) == 0

    # Degenerate empty text (EOT at the first sample): the region is the EOT
    # feed plus the two sosp rows once the machine reaches steady audio.
    empty = FireRedTTS3SGLangRequestData(
        state=FireRedTTS3State(task_type="voice_design")
    )
    empty.eot_sampled = True
    empty.text_ar_mode = TEXT_MODE_AUDIO
    assert runner.text_output_tokens(empty) == 3


def test_audio_hidden_offset_matches_layout() -> None:
    """Patch-driving hiddens start after the text region and the FIRST sosp
    (the doubled second sosp feed is the patch-0-driving audio_zero row)."""
    runner = make_runner()
    request = make_request()
    data = request.data
    data.text_token_ids = [10, 11, 12]
    data.eot_sampled = True
    data.text_ar_mode = TEXT_MODE_AUDIO
    # output_ids = [t0, t1, t2, EOT, SOSP, SOSP, 0...]; the second SOSP row
    # (index 5) is the audio_zero feed whose hidden drives patch 0.
    assert runner.audio_hidden_offset(data) == 5

    # Budget-exhausted text (no EOT) shifts nothing else.
    data.eot_sampled = False
    assert runner.audio_hidden_offset(data) == 3 + 1

    # Plain audio-mode requests (Base / Instruct tts) have no offset.
    plain = FireRedTTS3SGLangRequestData(state=FireRedTTS3State(task_type="tts"))
    assert runner.audio_hidden_offset(plain) == 0


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("prefill", [False, True])
def test_speaker_known_ids_survive_finalize(batch_size: int, prefill: bool) -> None:
    runner = make_runner()
    runner.output_processor = SGLangOutputProcessor()
    requests = []
    for index in range(batch_size):
        request = make_request("tts")
        request.request_id = f"speaker-{index}"
        request.data.state.variant = "speaker"
        request.data.text_ar_mode = TEXT_MODE_AUDIO
        request.data.flow_state = SimpleNamespace(patch_count=0)
        request.data.req = SimpleNamespace(extend_range=SimpleNamespace(length=1))
        requests.append(request)
    result = make_result([123] * batch_size)
    result.can_run_cuda_graph = False
    if prefill:
        runner.post_prefill(
            result, None, SimpleNamespace(is_prefill_only=False), requests
        )
    else:
        runner.post_decode(result, None, None, requests)
    assert result.next_token_ids.tolist() == [CONTROL_TOKEN_ID] * batch_size
    result.next_token_ids.fill_(123)
    output = runner.finalize(result, None, None, SimpleNamespace(requests=requests))
    assert output.host_token_ids.device.type == "cpu"
    assert output.host_token_ids.tolist() == [CONTROL_TOKEN_ID] * batch_size
    assert [output.outputs[r.request_id].data for r in requests] == [
        CONTROL_TOKEN_ID
    ] * batch_size
    assert all(r.data.generation_steps == 1 for r in requests)


@pytest.mark.parametrize("speaker_first", [False, True])
@pytest.mark.parametrize("sampled", [CONTROL_TOKEN_ID, 123, TEXT_EOT_ID])
@pytest.mark.parametrize("prefill", [False, True])
def test_mixed_speaker_and_text_batch_preserves_sampled_output(
    speaker_first: bool, sampled: int, prefill: bool
) -> None:
    runner = make_runner()
    runner.output_processor = SGLangOutputProcessor()
    speaker, text = make_request("tts"), make_request()
    speaker.request_id, text.request_id = "speaker", "text"
    speaker.data.state.variant = "speaker"
    speaker.data.text_ar_mode = TEXT_MODE_AUDIO
    speaker.data.flow_state = SimpleNamespace(patch_count=0)
    text.data.state.variant = "instruct"
    requests = [speaker, text] if speaker_first else [text, speaker]
    for request in requests:
        request.data.req = SimpleNamespace(extend_range=SimpleNamespace(length=1))
    result = make_result([sampled, sampled])
    result.can_run_cuda_graph = False
    if prefill:
        runner.post_prefill(
            result, None, SimpleNamespace(is_prefill_only=False), requests
        )
    else:
        runner.post_decode(result, None, None, requests)
    output = runner.finalize(result, None, None, SimpleNamespace(requests=requests))
    assert output.outputs["speaker"].data == CONTROL_TOKEN_ID
    assert output.outputs["text"].data == sampled
    assert output.host_token_ids is None
    assert text.data.text_token_ids == ([] if sampled == TEXT_EOT_ID else [sampled])
    assert text.data.eot_sampled is (sampled == TEXT_EOT_ID)
    assert runner.flow_calls == 1
    assert all(request.data.generation_steps == 1 for request in requests)


@pytest.mark.parametrize(
    "enabled,async_enabled,variant,dual_stream,mode,ids",
    [
        (False, False, "speaker", False, TEXT_MODE_AUDIO, [0]),
        (True, True, "speaker", False, TEXT_MODE_AUDIO, [0]),
        (True, False, "base", False, TEXT_MODE_AUDIO, [0]),
        (True, False, "instruct", False, TEXT_MODE_AUDIO, [0]),
        (True, False, "speaker", True, TEXT_MODE_AUDIO, [0]),
        (True, False, "speaker", False, TEXT_MODE_TEXT, [0]),
        (True, False, "speaker", False, TEXT_MODE_AUDIO, [123]),
    ],
)
def test_control_fastpath_preserves_other_token_paths(
    enabled: bool,
    async_enabled: bool,
    variant: str,
    dual_stream: bool,
    mode: str,
    ids: list[int],
) -> None:
    runner = make_runner()
    runner.control_token_fastpath = enabled
    runner._async_enabled = async_enabled
    request = make_request("tts")
    request.data.state.variant = variant
    request.data.dual_stream = dual_stream
    request.data.text_ar_mode = mode
    device_ids, host_ids = runner.make_next_token_ids(
        ids, torch.device("cpu"), [request]
    )
    assert device_ids.tolist() == ids
    assert host_ids is None


@pytest.mark.parametrize("skip", [None, False, True])
@pytest.mark.parametrize(
    "variant,dual_stream,async_enabled,min_patches,patch_count,skip_eligible",
    [
        ("speaker", False, False, 0, 0, False),
        ("speaker", False, False, 6, 5, True),
        ("speaker", False, False, 6, 6, False),
        ("speaker", False, False, 6, 7, False),
        ("speaker", False, True, 6, 5, False),
        ("speaker", True, False, 6, 5, False),
        ("base", False, False, 6, 5, False),
        ("instruct", False, False, 6, 5, False),
    ],
)
@pytest.mark.parametrize("logit", [-2.0, 0.0, 2.0])
def test_min_patch_stop_boundary_is_unchanged(
    skip: bool | None,
    variant: str,
    dual_stream: bool,
    async_enabled: bool,
    min_patches: int,
    patch_count: int,
    skip_eligible: bool,
    logit: float,
) -> None:
    runner = make_runner()
    if skip is not None:
        runner.skip_min_patch_stop = skip
    runner._async_enabled = async_enabled
    request = make_request("tts")
    data = request.data
    data.state.variant = variant
    data.dual_stream = dual_stream
    data.state.min_patches = min_patches
    data.state.stop_threshold = 0.5
    data.text_ar_mode = TEXT_MODE_AUDIO
    data.flow_state = SimpleNamespace(patch_count=patch_count)
    data.probe = SimpleNamespace(segment=lambda name: nullcontext())
    calls = []

    def stop_head(hidden: torch.Tensor) -> torch.Tensor:
        calls.append(hidden)
        return hidden.new_full((1, 1, 1), logit)

    runner.model = SimpleNamespace(stop_head=stop_head)
    assert runner.stop_predicted(data, torch.zeros(1, 8)) == (
        patch_count >= min_patches and logit >= 0
    )
    assert len(calls) == (0 if skip is not False and skip_eligible else 1)


@pytest.mark.parametrize("control_token_fastpath", [False, True])
@pytest.mark.parametrize("skip_min_patch_stop", [False, True])
def test_speaker_control_flags_preserve_independent_request_stops(
    control_token_fastpath: bool, skip_min_patch_stop: bool
) -> None:
    runner = make_runner()
    del runner.flow_step
    runner.control_token_fastpath = control_token_fastpath
    runner.skip_min_patch_stop = skip_min_patch_stop
    runner.output_processor = SGLangOutputProcessor()
    requests = [make_request("tts"), make_request("tts")]
    for index, request in enumerate(requests):
        request.request_id = f"speaker-{index}"
        request.data.state.variant = "speaker"
        request.data.state.min_patches = 6
        request.data.text_ar_mode = TEXT_MODE_AUDIO
        request.data.flow_state = SimpleNamespace(patch_count=5 + index)
        request.data.probe = SimpleNamespace(segment=lambda name: nullcontext())
        request.data.req = SimpleNamespace(finished_reason=None)
    stop_calls: list[torch.Tensor] = []
    emitted: list[FireRedTTS3SGLangRequestData] = []

    def stop_head(hidden: torch.Tensor) -> torch.Tensor:
        stop_calls.append(hidden)
        return hidden.new_full((1, 1, 1), 2.0)

    def emit_patch(
        data: FireRedTTS3SGLangRequestData,
        hidden: torch.Tensor,
        flow_rows: list[FlowRow] | None = None,
    ) -> None:
        data.flow_state.patch_count += 1
        emitted.append(data)

    runner.model = SimpleNamespace(stop_head=stop_head)
    runner.run_flow = emit_patch
    for active in (requests, requests[:1]):
        result = make_result([123] * len(active))
        result.can_run_cuda_graph = False
        runner.post_decode(result, None, None, active)
        output = runner.finalize(result, None, None, SimpleNamespace(requests=active))
        assert [output.outputs[r.request_id].data for r in active] == [
            CONTROL_TOKEN_ID
        ] * len(active)
        assert (output.host_token_ids is not None) is control_token_fastpath
        assert requests[1].data.req.finished_reason is not None
        assert (requests[0].data.req.finished_reason is not None) is (len(active) == 1)
    assert emitted == [requests[0].data]
    assert [request.data.flow_state.patch_count for request in requests] == [6, 6]
    assert [request.data.generation_steps for request in requests] == [2, 1]
    assert len(stop_calls) == (2 if skip_min_patch_stop else 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_control_token_device_snapshots_are_independent() -> None:
    runner = make_runner()
    request = make_request("tts")
    request.data.state.variant = "speaker"
    request.data.text_ar_mode = TEXT_MODE_AUDIO
    first, host = runner.make_next_token_ids([0], torch.device("cuda"), [request])
    second, next_host = runner.make_next_token_ids([0], first.device, [request])
    second.fill_(123)
    assert first.is_cuda and host.device.type == "cpu"
    assert first.tolist() == [0]
    assert host.tolist() == next_host.tolist() == [0]
