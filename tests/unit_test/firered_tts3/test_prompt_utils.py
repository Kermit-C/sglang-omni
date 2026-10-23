# SPDX-License-Identifier: Apache-2.0
"""Prompt formatting, Han token expansion, and resampled patch boundaries."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.firered_tts3 import prompt_utils
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


def test_count_latent_patches_arithmetic() -> None:
    # 24 kHz identity path: 1 patch = 3840 samples.
    wav = torch.zeros(1, 10000)
    _, n_patches = prompt_utils.count_latent_patches(wav, 24000)
    assert n_patches == 3  # ceil(10000 / 3840)
    wav16k = torch.zeros(1, 16000)  # 1 s
    wav24k, n_patches = prompt_utils.count_latent_patches(wav16k, 16000)
    assert wav24k.shape[1] == 24000
    assert n_patches == 7  # ceil(24000 / 3840)


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
