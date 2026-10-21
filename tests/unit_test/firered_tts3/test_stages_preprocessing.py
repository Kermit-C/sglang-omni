# SPDX-License-Identifier: Apache-2.0
"""Preprocessing stage: Base template composition, budgets, validation."""

from __future__ import annotations

import pytest

from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.prompt_utils import (
    TEXT_TOKEN_BUDGET,
    compose_acoustic_edit,
    compose_base_text,
    compose_semantic_edit,
    compose_tts,
    compose_voice_design,
)
from sglang_omni.models.firered_tts3.request_builders import ENGINE_TOKEN_SLACK
from sglang_omni.models.firered_tts3.stages import preprocess_firered_tts3_payload
from sglang_omni.proto import OmniRequest, StagePayload

from .conftest import CHECKPOINTS


class RecordingTokenizer:
    """Tokenizes to one id per character so tests can count positions."""

    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        self.encoded_texts.append(text)
        return {"input_ids": [ord(char) % 151936 for char in text]}


def make_payload(
    *,
    inputs: dict | None = None,
    params: dict | None = None,
    tts_params: dict | None = None,
) -> StagePayload:
    return StagePayload(
        request_id="rid",
        request=OmniRequest(
            inputs=inputs
            or {
                "text": "欢迎使用点点语音合成系统。",
                "references": [{"audio_path": "ref.wav", "text": "今天天气很好。"}],
            },
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data={},
    )


def preprocess(payload: StagePayload, **factory_kwargs):
    tokenizer = RecordingTokenizer()
    config = FireRedTTS3Config(spk_in_dim=512, dit_in_channels=1600)
    preprocess_firered_tts3_payload(
        payload,
        tokenizer=tokenizer,
        config=config,
        n_timesteps=factory_kwargs.get("n_timesteps", 10),
        max_patches=factory_kwargs.get("max_patches", 400),
    )
    return tokenizer, payload.data


def test_preprocessing_composes_base_template() -> None:
    tokenizer, data = preprocess(make_payload())
    expected = compose_base_text(
        "Chinese", "今天天气很好。", "欢迎使用点点语音合成系统。"
    )
    assert tokenizer.encoded_texts == [expected]
    assert data["variant"] == "base"
    assert data["task_type"] == "tts"
    assert data["language"] == "Chinese"
    assert data["prompt_audio_path"] == "ref.wav"
    assert data["prompt_text"] == "今天天气很好。"
    assert data["input_ids"] == [ord(char) % 151936 for char in expected]
    assert data["n_timesteps"] == 10
    assert data["inference_cfg"] == 2.0
    assert data["stop_threshold"] == 0.5
    assert data["min_patches"] == 6
    assert data["max_patches"] == 400
    assert data["max_engine_tokens"] == 400 + ENGINE_TOKEN_SLACK
    assert data["sample_rate"] == 24000


def test_preprocessing_param_precedence() -> None:
    payload = make_payload(
        params={
            "seed": 7,
            "stage_params": {"talker": {"inference_cfg": 1.5}},
        },
        tts_params={"stop_threshold": "0.7"},
    )
    _tokenizer, data = preprocess(payload)
    assert data["seed"] == 7
    assert data["inference_cfg"] == 1.5
    assert data["stop_threshold"] == 0.7


def test_preprocessing_requires_text_and_reference() -> None:
    payload = make_payload(inputs={"text": "  "})
    with pytest.raises(ValueError, match="non-empty input text"):
        preprocess(payload)
    payload = make_payload(inputs={"text": "你好"})
    with pytest.raises(ValueError, match="reference audio"):
        preprocess(payload)
    payload = make_payload(
        inputs={"text": "你好", "references": [{"audio_path": "ref.wav"}]}
    )
    with pytest.raises(ValueError, match="transcript"):
        preprocess(payload)


def test_preprocessing_accepts_streaming_and_bad_language_rejected() -> None:
    _tokenizer, data = preprocess(make_payload(params={"stream": True}))
    assert data["stream"] is True
    _tokenizer, data = preprocess(make_payload())
    assert data["stream"] is False
    with pytest.raises(ValueError, match="unsupported FireRedTTS3 language"):
        preprocess(make_payload(tts_params={"language": "Klingon"}))
    _tokenizer, data = preprocess(make_payload(tts_params={"language": "ZH_Shanghai"}))
    assert data["language"] == "ZH_Shanghai"


def test_preprocessing_budget_follows_max_patches() -> None:
    _tokenizer, data = preprocess(make_payload(), max_patches=256)
    assert data["max_patches"] == 256
    assert data["max_engine_tokens"] == 256 + ENGINE_TOKEN_SLACK


INSTRUCT_CONFIG = FireRedTTS3Config(use_lm_head=True, dit_in_channels=1088)


def preprocess_instruct(payload: StagePayload, monkeypatch, *, patches: int = 18):
    """Instruct preprocessing with the audio patch count pinned (no IO)."""
    tokenizer = RecordingTokenizer()
    monkeypatch.setattr(
        "sglang_omni.models.firered_tts3.stages.count_audio_patches",
        lambda path, *, cap_seconds: patches,
    )
    preprocess_firered_tts3_payload(
        payload,
        tokenizer=tokenizer,
        config=INSTRUCT_CONFIG,
        n_timesteps=10,
        max_patches=400,
    )
    return tokenizer, payload.data


def instruct_payload(
    *,
    task_type: str | None = None,
    instructions: str | None = None,
    text: str = "晚上好，欢迎收听今天的节目。",
    with_reference: bool = False,
    params: dict | None = None,
    tts_params: dict | None = None,
) -> StagePayload:
    inputs: dict = {"text": text}
    if with_reference:
        inputs["references"] = [{"audio_path": "edit.wav", "text": "源音频文本。"}]
    if instructions is not None:
        inputs["instructions"] = instructions
    if task_type is not None:
        (tts_params := dict(tts_params or {}))["task_type"] = task_type
    return make_payload(inputs=inputs, params=params or {}, tts_params=tts_params)


def test_instruct_tts_composes_chatml_with_icl_slots(monkeypatch) -> None:
    tokenizer, data = preprocess_instruct(
        instruct_payload(with_reference=True), monkeypatch, patches=18
    )
    expected = compose_tts(18, "源音频文本。", "晚上好，欢迎收听今天的节目。")
    assert tokenizer.encoded_texts == [expected]
    assert data["variant"] == "instruct"
    assert data["task_type"] == "tts"
    assert data["prompt_audio_path"] == "edit.wav"
    assert data["prompt_text"] == "源音频文本。"
    assert data.get("audio_in_path") is None
    assert data["inference_cfg"] == 2.0
    assert data["text_temperature"] == 0.0
    assert data["max_engine_tokens"] == 400 + ENGINE_TOKEN_SLACK


def test_instruct_voice_design_samples_text(monkeypatch) -> None:
    tokenizer, data = preprocess_instruct(
        instruct_payload(task_type="VoiceDesign", instructions="a calm deep voice"),
        monkeypatch,
    )
    expected = compose_voice_design("a calm deep voice", "晚上好，欢迎收听今天的节目。")
    assert tokenizer.encoded_texts == [expected]
    assert data["task_type"] == "voice_design"
    assert data.get("prompt_audio_path") is None
    assert data["inference_cfg"] == 1.2
    assert (data["text_temperature"], data["text_top_p"], data["text_top_k"]) == (
        0.7,
        0.8,
        20,
    )
    # Text phase budget: 200 text tokens + finalize + sosp transition feeds.
    assert data["max_engine_tokens"] == 400 + ENGINE_TOKEN_SLACK + TEXT_TOKEN_BUDGET + 2


def test_instruct_semantic_edit_ignores_input_text(monkeypatch) -> None:
    tokenizer, data = preprocess_instruct(
        instruct_payload(
            task_type="SemanticEdit",
            instructions="Replace '公园' with '游乐园'.",
            text="",
            with_reference=True,
        ),
        monkeypatch,
        patches=18,
    )
    expected = compose_semantic_edit("Replace '公园' with '游乐园'.", 18)
    assert tokenizer.encoded_texts == [expected]
    assert data["task_type"] == "semantic_edit"
    assert data["audio_in_path"] == "edit.wav"
    assert data.get("prompt_audio_path") is None
    assert data["inference_cfg"] == 1.2
    assert data["text_temperature"] == 0.0
    assert data["max_engine_tokens"] == 400 + ENGINE_TOKEN_SLACK + TEXT_TOKEN_BUDGET + 2


def test_instruct_acoustic_edit_maps_speed_to_template(monkeypatch) -> None:
    tokenizer, data = preprocess_instruct(
        instruct_payload(
            task_type="AcousticEdit",
            text="",
            with_reference=True,
            tts_params={"speed": 0.8},
        ),
        monkeypatch,
        patches=18,
    )
    expected = compose_acoustic_edit("adjust the speed to 0.8x", 18)
    assert tokenizer.encoded_texts == [expected]
    assert data["task_type"] == "acoustic_edit"
    assert data["audio_in_path"] == "edit.wav"
    assert data["inference_cfg"] == 1.2
    # No text phase: the budget is the plain audio one.
    assert data["max_engine_tokens"] == 400 + ENGINE_TOKEN_SLACK


def test_instruct_acoustic_edit_pitch_volume_via_stage_params(monkeypatch) -> None:
    _tokenizer, data = preprocess_instruct(
        instruct_payload(
            task_type="AcousticEdit",
            text="",
            with_reference=True,
            params={
                "stage_params": {
                    "preprocessing": {
                        "acoustic_attribute": "Pitch",
                        "acoustic_value": -2,
                    }
                }
            },
        ),
        monkeypatch,
    )
    assert data["audio_in_path"] == "edit.wav"
    # The raw instruction is composed from the fixed trained template.
    from sglang_omni.models.firered_tts3.prompt_utils import (
        compose_acoustic_instruction,
    )

    assert compose_acoustic_instruction("Pitch", -2) == "shift the pitch by -2 steps"


def test_instruct_task_validation(monkeypatch) -> None:
    with pytest.raises(ValueError, match="task_type"):
        preprocess_instruct(instruct_payload(task_type="Base2"), monkeypatch)
    with pytest.raises(ValueError, match="instructions"):
        preprocess_instruct(instruct_payload(task_type="VoiceDesign"), monkeypatch)
    with pytest.raises(ValueError, match="non-empty input text"):
        preprocess_instruct(
            instruct_payload(task_type="VoiceDesign", instructions="x", text=" "),
            monkeypatch,
        )
    with pytest.raises(ValueError, match="edit source"):
        preprocess_instruct(
            instruct_payload(task_type="AcousticEdit", text=""), monkeypatch
        )
    with pytest.raises(ValueError, match="reference audio"):
        preprocess_instruct(instruct_payload(text="你好"), monkeypatch)
    with pytest.raises(ValueError, match="transcript"):
        preprocess_instruct(
            make_payload(
                inputs={
                    "text": "你好",
                    "references": [{"audio_path": "ref.wav"}],
                }
            ),
            monkeypatch,
        )


def test_instruct_explicit_sampling_overrides_task_default(monkeypatch) -> None:
    _tokenizer, data = preprocess_instruct(
        instruct_payload(
            task_type="VoiceDesign",
            instructions="a calm deep voice",
            params={"temperature": 0.0},
            tts_params={"explicit_generation_params": ["temperature"]},
        ),
        monkeypatch,
    )
    # An explicit greedy request beats the voice_design sampling default.
    assert data["text_temperature"] == 0.0
    assert (data["text_top_p"], data["text_top_k"]) == (0.8, 20)


SPEAKER_CONFIG = FireRedTTS3Config(
    text_latent_interleave_ratio=[2, 1],
    speaker_token="<|f531_speaker|>",
    dit_in_channels=1088,
    use_lm_head=False,
    redae_scale=0.4,
    generation_defaults={
        "n_timesteps": 10,
        "inference_cfg": 1.2,
        "stop_threshold": 0.5,
        "min_patches": 6,
        "max_patches": 600,
    },
)


def make_speaker_tokenizer():

    import pytest

    checkpoint = CHECKPOINTS["speaker"]
    if not checkpoint.is_dir():
        pytest.skip("speaker checkpoint not present")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)


def preprocess_speaker(
    payload: StagePayload, *, tokenizer=None, max_patches: int = 600
):
    tokenizer = tokenizer or make_speaker_tokenizer()
    preprocess_firered_tts3_payload(
        payload,
        tokenizer=tokenizer,
        config=SPEAKER_CONFIG,
        n_timesteps=10,
        max_patches=max_patches,
    )
    return payload.data


def speaker_payload(**kwargs) -> StagePayload:
    kwargs.setdefault("inputs", {"text": "晚上好，欢迎收听今天的节目。"})
    return make_payload(**kwargs)


def test_speaker_prefill_form_composition() -> None:
    from sglang_omni.models.firered_tts3.prompt_utils import (
        encode_speaker_prefill_ids,
        resolve_speaker_token_id,
    )

    tokenizer = make_speaker_tokenizer()
    data = preprocess_speaker(speaker_payload(), tokenizer=tokenizer)
    text = "晚上好，欢迎收听今天的节目。"
    assert data["variant"] == "speaker"
    assert data["task_type"] == "tts"
    assert data["dual_stream"] is False
    assert data["text_stream_ids"] == []
    assert data["text_pad_id"] == -1
    assert data["input_ids"] == encode_speaker_prefill_ids(
        tokenizer, text=text, speaker_token="<|f531_speaker|>"
    )
    holder_id = resolve_speaker_token_id(tokenizer, "<|text_holder|>")
    assert holder_id not in data["input_ids"]  # all slots were substituted
    # Prefill form budget: max_patches + slack; generation defaults from the
    # checkpoint (the Instruct per-task CFG table does not apply).
    assert data["inference_cfg"] == 1.2
    assert data["max_engine_tokens"] == 620
    assert data["max_patches"] == 600


def test_speaker_dual_stream_form_composition() -> None:
    from sglang_omni.models.firered_tts3.prompt_utils import (
        encode_speaker_dual_stream_prompt,
    )

    tokenizer = make_speaker_tokenizer()
    text = "晚上好，欢迎收听今天的节目。"
    data = preprocess_speaker(
        speaker_payload(params={"dual_stream": True}), tokenizer=tokenizer
    )
    prompt_ids, text_ids, text_pad_id = encode_speaker_dual_stream_prompt(
        tokenizer, text=text, speaker_token="<|f531_speaker|>"
    )
    assert data["dual_stream"] is True
    assert data["input_ids"] == prompt_ids
    assert data["text_stream_ids"] == text_ids
    assert data["text_pad_id"] == text_pad_id
    # The prompt stops at <|audio_start|>; the text rides the payload.
    assert len(data["input_ids"]) < 60
    # Dual-stream budget: ceil(3 * max_patches) + slack.
    assert data["max_engine_tokens"] == 1820


def test_speaker_dual_stream_style_passthrough() -> None:
    from sglang_omni.models.firered_tts3.prompt_utils import (
        compose_speaker_chatml,
        encode_speaker_text,
    )

    tokenizer = make_speaker_tokenizer()
    data = preprocess_speaker(
        speaker_payload(params={"dual_stream": True, "style": "Use a 活泼 tone."}),
        tokenizer=tokenizer,
    )
    expected = compose_speaker_chatml(
        "<|f531_speaker|>",
        style="Use a 活泼 tone.",
        dual_stream=True,
        interleave_ratio=(2, 1),
    )
    assert data["input_ids"] == encode_speaker_text(tokenizer, expected)

    # Style is a dual-stream-only prefix (reference parity); the prefill form
    # ignores it.
    plain = preprocess_speaker(speaker_payload(), tokenizer=tokenizer)
    styled = preprocess_speaker(
        speaker_payload(params={"style": "Use a 活泼 tone."}), tokenizer=tokenizer
    )
    assert styled["input_ids"] == plain["input_ids"]


def test_speaker_rejects_references_and_foreign_tasks() -> None:
    with pytest.raises(ValueError, match="text-only tts"):
        preprocess_speaker(
            speaker_payload(
                inputs={
                    "text": "你好",
                    "references": [{"audio_path": "ref.wav", "text": "参考"}],
                }
            )
        )
    with pytest.raises(ValueError, match="text-only tts"):
        preprocess_speaker(speaker_payload(tts_params={"ref_text": "参考"}))
    with pytest.raises(ValueError, match="task_type"):
        preprocess_speaker(speaker_payload(tts_params={"task_type": "VoiceDesign"}))
    with pytest.raises(ValueError, match="non-empty input text"):
        preprocess_speaker(speaker_payload(inputs={"text": "  "}))


def test_speaker_dual_stream_requires_capability() -> None:
    base_config = FireRedTTS3Config(spk_in_dim=512, dit_in_channels=1600)
    payload = speaker_payload(params={"dual_stream": True})
    # A Base checkpoint routed through the speaker composer is impossible via
    # detect_fireredtts3_variant; engine_token_budget still guards the budget.
    from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
    from sglang_omni.models.firered_tts3.stages import engine_token_budget

    del base_config
    with pytest.raises(ValueError, match="text_latent_interleave_ratio"):
        engine_token_budget(FireRedTTS3State(dual_stream=True, max_patches=600))
    assert (
        engine_token_budget(
            FireRedTTS3State(dual_stream=True, max_patches=600),
            interleave_ratio=(2, 1),
        )
        == 1820
    )


def test_speaker_max_new_tokens_override() -> None:
    data = preprocess_speaker(
        speaker_payload(params={"dual_stream": True, "max_new_tokens": 300})
    )
    assert data["max_engine_tokens"] == 300
