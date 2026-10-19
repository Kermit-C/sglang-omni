# SPDX-License-Identifier: Apache-2.0
"""Instruct text-AR state machine: transitions, budget, and boundary cases.

The reference machine (fireredtts3_instruct.generate): text steps sample real
tokens; sampling <|eot|> (or the 200th token) makes that token's own feed the
no-sampling finalize step; then one <|sosp|> feed; then the audio loop's
step 0 feeds <|sosp|> AGAIN (the doubled feed is the reference's actual
behavior). <|eot|> must never terminate the request.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.firered_tts3.model_runner import (
    TEXT_MODE_AUDIO,
    TEXT_MODE_AUDIO_ZERO,
    TEXT_MODE_FINALIZE,
    TEXT_MODE_SOSP,
    TEXT_MODE_TEXT,
    FireRedTTS3ModelRunner,
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


def make_runner() -> FireRedTTS3ModelRunner:
    runner = FireRedTTS3ModelRunner.__new__(FireRedTTS3ModelRunner)
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
