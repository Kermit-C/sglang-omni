# SPDX-License-Identifier: Apache-2.0
"""Golden tests for FireRedTTS3 prompt composition.

Template strings are compared byte-for-byte against the reference composers
(fireredtts3/utils/chatml.py for the four Instruct tasks, the reference
inference wrapper's format_chatml for the speaker variant, the Base generate
template), and the char-modelled speaker text encoder against the reference
WrappedQwenTextTokenizer.
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.firered_tts3 import prompt_utils

from .conftest import CHECKPOINTS, requires_checkpoints, requires_reference


@requires_reference
@pytest.mark.parametrize(
    "ours_name, oracle_name, kwargs",
    [
        (
            "compose_tts",
            "compose_generate_input_tts",
            {
                "prompt_latent_len": 7,
                "prompt_text": "参考文本。",
                "text": "要合成的文本。",
            },
        ),
        (
            "compose_voice_design",
            "compose_generate_input_voice_design",
            {"instruction": "a calm deep voice", "text": "Hello there."},
        ),
        (
            "compose_semantic_edit",
            "compose_generate_input_semantic_edit",
            {"instruction": "replace 今天 with 明天", "audio_in_latent_len": 5},
        ),
        (
            "compose_acoustic_edit",
            "compose_generate_input_acoustic_edit",
            {"instruction": "adjust the speed to 0.8x", "audio_in_latent_len": 5},
        ),
    ],
)
def test_instruct_chatml_matches_reference(
    reference_sources, ours_name: str, oracle_name: str, kwargs: dict
) -> None:
    from fireredtts3.utils import chatml as reference_chatml

    ours = getattr(prompt_utils, ours_name)(**kwargs)
    oracle = getattr(reference_chatml, oracle_name)(**kwargs)
    assert ours == oracle


@requires_reference
def test_base_text_template_matches_reference(reference_sources) -> None:
    """The Base ICL template from the reference FireRedTTS3Base.generate."""
    language, ref_text, text = "zh", "参考文本。", "要合成的文本。"
    oracle = f"<|{language}|><|sot|>{ref_text}{text}<|eot|>"
    ours = prompt_utils.compose_base_text(language, ref_text, text)
    assert ours == oracle
    # Golden literal guards against accidental template edits.
    assert ours == "<|zh|><|sot|>参考文本。要合成的文本。<|eot|>"


@requires_reference
def test_speaker_chatml_prefill_matches_reference(reference_sources) -> None:
    from diandiantts.utils.chatml import (
        LATENT_END_TOKEN,
        LATENT_HOLDER_TOKEN,
        TEXT_HOLDER_TOKEN,
        TEXT_PAD_TOKEN,
        format_chatml,
    )

    oracle = format_chatml(
        text_len=11,
        latent_len=0,
        speaker="<|f531_speaker|>",
        text_holder=TEXT_HOLDER_TOKEN,
        text_pad=TEXT_PAD_TOKEN,
        latent_holder=LATENT_HOLDER_TOKEN,
        use_interleave=False,
    )
    oracle = oracle.removesuffix("<|im_end|>\n").removesuffix(LATENT_END_TOKEN)
    ours = prompt_utils.compose_speaker_chatml("<|f531_speaker|>", text_len=11)
    assert ours == oracle


@requires_reference
def test_speaker_chatml_dual_stream_matches_reference(reference_sources) -> None:
    from diandiantts.utils.chatml import (
        LATENT_END_TOKEN,
        LATENT_HOLDER_TOKEN,
        TEXT_HOLDER_TOKEN,
        TEXT_PAD_TOKEN,
        format_chatml,
    )

    oracle = format_chatml(
        text_len=1,
        latent_len=1,  # one extra group
        speaker="<|f531_speaker|>",
        text_holder=TEXT_HOLDER_TOKEN,
        text_pad=TEXT_PAD_TOKEN,
        latent_holder=LATENT_HOLDER_TOKEN,
        use_interleave=True,
        text_latent_ratio=(2, 1),
        user_prompt_prefix="Use a 中性 tone.",
    )
    oracle = oracle.removesuffix(
        f"{TEXT_HOLDER_TOKEN}{TEXT_PAD_TOKEN}{LATENT_HOLDER_TOKEN}"
        f"{LATENT_END_TOKEN}<|im_end|>\n"
    )
    ours = prompt_utils.compose_speaker_chatml(
        "<|f531_speaker|>",
        style="Use a 中性 tone.",
        dual_stream=True,
        interleave_ratio=(2, 1),
    )
    assert ours == oracle


@requires_reference
def test_speaker_chatml_rejects_bad_ratio(reference_sources) -> None:
    with pytest.raises(ValueError, match="interleave_ratio"):
        prompt_utils.compose_speaker_chatml(
            "<|f531_speaker|>", dual_stream=True, interleave_ratio=(0, 1)
        )


@pytest.fixture(scope="module")
def speaker_tokenizer():
    if not CHECKPOINTS["speaker"].is_dir():
        pytest.skip("speaker checkpoint not present")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(CHECKPOINTS["speaker"]), local_files_only=True
    )


@pytest.fixture(scope="module")
def reference_speaker_tokenizer(reference_sources):
    if not CHECKPOINTS["speaker"].is_dir():
        pytest.skip("speaker checkpoint not present")
    from diandiantts.utils.text_tokenizer import WrappedQwenTextTokenizer

    return WrappedQwenTextTokenizer(tokenizer_dir=str(CHECKPOINTS["speaker"]))


SPEAKER_TEXTS = [
    "大家好，今天天气真不错。",
    "Hello world 你好世界",
    "Use a 中性 tone.",
    "欢迎使用语音合成系统，支持中英文混排 123 个字符。",
    "每个汉字都要拆成单字，标点保留！yes",
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
]


@requires_reference
@requires_checkpoints
@pytest.mark.parametrize("text", SPEAKER_TEXTS)
def test_encode_speaker_text_matches_reference(
    reference_sources, speaker_tokenizer, reference_speaker_tokenizer, text: str
) -> None:
    ours = prompt_utils.encode_speaker_text(speaker_tokenizer, text)
    oracle = reference_speaker_tokenizer.encode(text)
    assert ours == [int(t) for t in oracle]


@requires_reference
@requires_checkpoints
def test_speaker_prefill_ids_match_reference_masked_scatter(
    reference_sources, speaker_tokenizer, reference_speaker_tokenizer
) -> None:
    """Our token-level holder substitution == the reference masked_scatter."""
    from diandiantts.utils.chatml import TEXT_HOLDER_TOKEN, format_chatml

    text = SPEAKER_TEXTS[3]
    text_ids = reference_speaker_tokenizer.encode(text)
    chatml = format_chatml(
        text_len=len(text_ids), latent_len=0, speaker="<|f531_speaker|>"
    )
    chatml = chatml.removesuffix("<|im_end|>\n").removesuffix("<|audio_end|>")
    template_ids = torch.tensor(
        reference_speaker_tokenizer.encode(chatml), dtype=torch.long
    )
    holder_id = int(speaker_tokenizer.convert_tokens_to_ids(TEXT_HOLDER_TOKEN))
    oracle = template_ids.masked_scatter(
        template_ids == holder_id, torch.tensor(text_ids, dtype=torch.long)
    ).tolist()

    ours = prompt_utils.encode_speaker_prefill_ids(
        speaker_tokenizer, text=text, speaker_token="<|f531_speaker|>"
    )
    assert ours == oracle


@requires_reference
@requires_checkpoints
def test_speaker_dual_stream_prompt_matches_reference(
    reference_sources, speaker_tokenizer, reference_speaker_tokenizer
) -> None:
    from diandiantts.utils.chatml import TEXT_PAD_ID

    text = SPEAKER_TEXTS[0]
    prompt_ids, text_ids, text_pad_id = prompt_utils.encode_speaker_dual_stream_prompt(
        speaker_tokenizer, text=text, speaker_token="<|f531_speaker|>"
    )
    oracle_chatml = prompt_utils.compose_speaker_chatml(
        "<|f531_speaker|>", style="Use a 中性 tone.", dual_stream=True
    )
    assert prompt_ids == reference_speaker_tokenizer.encode(oracle_chatml)
    assert text_ids == reference_speaker_tokenizer.encode(text)
    assert text_pad_id == TEXT_PAD_ID


@requires_checkpoints
def test_special_token_id_collision_across_variants() -> None:
    """151669 is <|sosp|> for Base/Instruct but <|audio_holder|> for the
    speaker variant — the reason ids are resolved per variant and never
    hard-coded."""
    from transformers import AutoTokenizer

    tokenizers = {
        variant: AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        for variant, path in CHECKPOINTS.items()
    }
    assert (
        tokenizers["base"].convert_tokens_to_ids("<|sosp|>")
        == prompt_utils.AUDIO_SOS_ID
        == 151669
    )
    assert tokenizers["speaker"].convert_tokens_to_ids("<|audio_holder|>") == 151669
    for variant in ("base", "instruct"):
        tokenizer = tokenizers[variant]
        assert tokenizer.convert_tokens_to_ids("<|eot|>") == prompt_utils.TEXT_EOT_ID
        assert (
            tokenizer.convert_tokens_to_ids("<|image_pad|>")
            == prompt_utils.LATENT_IN_PAD_ID
        )
        assert (
            tokenizer.convert_tokens_to_ids("<|video_pad|>")
            == prompt_utils.LATENT_OUT_PAD_ID
        )


@requires_checkpoints
def test_instruct_chatml_tokenizes_to_expected_pad_ids() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(CHECKPOINTS["instruct"]), local_files_only=True
    )
    chatml = prompt_utils.compose_tts(3, "参考文本。", "要合成的文本。")
    ids = prompt_utils.tokenize_text(tokenizer, chatml)
    assert ids.count(prompt_utils.LATENT_OUT_PAD_ID) == 3
    assert prompt_utils.LATENT_IN_PAD_ID not in ids

    chatml = prompt_utils.compose_acoustic_edit("adjust the speed to 0.8x", 4)
    ids = prompt_utils.tokenize_text(tokenizer, chatml)
    assert ids.count(prompt_utils.LATENT_IN_PAD_ID) == 4


def test_count_latent_patches_arithmetic() -> None:
    # 24 kHz identity path: 1 patch = 3840 samples.
    wav = torch.zeros(1, 10000)
    _, n_patches = prompt_utils.count_latent_patches(wav, 24000)
    assert n_patches == 3  # ceil(10000 / 3840)
    wav16k = torch.zeros(1, 16000)  # 1 s
    wav24k, n_patches = prompt_utils.count_latent_patches(wav16k, 16000)
    assert wav24k.shape[1] == 24000
    assert n_patches == 7  # ceil(24000 / 3840)


def test_speaker_max_tokens_formulas() -> None:
    # Prefill form: max_patches + 20 (plan section 3.4).
    assert prompt_utils.speaker_max_tokens(600) == 620
    assert prompt_utils.speaker_max_tokens(400) == 420
    # Dual-stream form with the (2, 1) ratio: 3 engine steps per patch.
    assert prompt_utils.speaker_max_tokens(600, (2, 1)) == 1820
    assert prompt_utils.speaker_max_tokens(601, (2, 1)) == 1823


def test_task_sampling_kwargs() -> None:
    assert prompt_utils.task_sampling_kwargs("voice_design") == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }
    assert prompt_utils.task_sampling_kwargs("tts") == {"temperature": 0.0}
    assert prompt_utils.task_sampling_kwargs("tts", seed=7) == {
        "temperature": 0.0,
        "seed": 7,
    }
    for task in ("tts", "voice_design", "semantic_edit", "acoustic_edit"):
        assert task in prompt_utils.TASK_CFG


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
