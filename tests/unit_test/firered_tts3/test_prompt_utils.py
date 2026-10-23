# SPDX-License-Identifier: Apache-2.0
"""Prompt formatting, Han token expansion, and resampled patch boundaries."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import pytest
import soundfile as sf

from sglang_omni.models.firered_tts3 import prompt_utils
from sglang_omni.models.firered_tts3.instruct import count_audio_patches
from tests.unit_test.fixtures.firered_tts3 import make_speaker_tokenizer


def test_base_text_template_matches_reference() -> None:
    """Reference text precedes target text inside the checkpoint text delimiters."""
    language, ref_text, text = "zh", "参考文本。", "要合成的文本。"
    actual = prompt_utils.compose_base_text(language, ref_text, text)
    assert actual == "<|zh|><|sot|>参考文本。要合成的文本。<|eot|>"


def test_speaker_chatml_rejects_bad_ratio() -> None:
    with pytest.raises(ValueError, match="interleave_ratio"):
        prompt_utils.compose_speaker_chatml(
            "<|f531_speaker|>", dual_stream=True, interleave_ratio=(0, 1)
        )


@pytest.mark.parametrize(
    "text,expected_tokens",
    [("你 好世界", ["你", "好", "世", "界"]), ("生僻", ["生僻"])],
)
def test_speaker_text_expands_han_tokens_and_preserves_unknown_characters(
    text: str, expected_tokens: list[str]
) -> None:
    tokenizer = make_speaker_tokenizer()
    encoded = prompt_utils.encode_speaker_text(tokenizer, text)
    assert encoded == [
        tokenizer.convert_tokens_to_ids(token) for token in expected_tokens
    ]


@pytest.mark.parametrize(
    "sample_rate,sample_count,cap_seconds,expected_patches",
    [
        (24000, 3840, None, 1),
        (24000, 3841, None, 2),
        (16000, 2561, None, 2),
        (44100, 7056, None, 1),
        (44100, 7057, None, 2),
        (16000, 320001, 20.0, 125),
    ],
)
def test_audio_prompt_slots_follow_resampled_patch_boundaries(
    tmp_path: Path,
    sample_rate: int,
    sample_count: int,
    cap_seconds: float | None,
    expected_patches: int,
) -> None:
    """Audio source formats must agree on patch slots after resampling and edit caps."""
    encoded_audio = io.BytesIO()
    sf.write(
        encoded_audio,
        np.zeros(sample_count, dtype=np.float32),
        sample_rate,
        format="WAV",
        subtype="FLOAT",
    )
    audio_bytes = encoded_audio.getvalue()
    audio_path = tmp_path / "reference.wav"
    audio_path.write_bytes(audio_bytes)
    audio_uri = f"data:audio/wav;base64,{base64.b64encode(audio_bytes).decode('ascii')}"
    audio_url = "https://audio.example.test/reference.wav"
    response = httpx.Response(
        200, content=audio_bytes, request=httpx.Request("GET", audio_url)
    )
    with patch("sglang_omni.utils.audio.httpx.get", return_value=response):
        for reference in (str(audio_path), audio_uri, audio_url):
            assert (
                count_audio_patches(reference, cap_seconds=cap_seconds)
                == expected_patches
            )


def test_compose_acoustic_instruction_templates() -> None:
    assert (
        prompt_utils.compose_acoustic_instruction("Speed", 0.8)
        == "adjust the speed to 0.8x"
    )
    assert (
        prompt_utils.compose_acoustic_instruction("Volume", 1.5)
        == "adjust the volume to 1.5"
    )
    assert (
        prompt_utils.compose_acoustic_instruction("Pitch", 2)
        == "shift the pitch by 2 steps"
    )
    assert (
        prompt_utils.compose_acoustic_instruction("Pitch", -1)
        == "shift the pitch by -1 step"
    )
