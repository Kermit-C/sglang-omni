# SPDX-License-Identifier: Apache-2.0
"""Preprocessing stage: Base template composition, budgets, validation."""

from __future__ import annotations

import pytest

from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.prompt_utils import compose_base_text
from sglang_omni.models.firered_tts3.stages import preprocess_firered_tts3_payload
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.utils.json import JsonValue
from tests.unit_test.fixtures.firered_tts3 import make_speaker_tokenizer


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


@pytest.mark.parametrize("max_patches,engine_tokens", [(400, 420), (256, 276)])
def test_preprocessing_composes_base_template(
    max_patches: int, engine_tokens: int
) -> None:
    tokenizer, data = preprocess(make_payload(), max_patches=max_patches)
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
    assert data["max_patches"] == max_patches
    assert data["max_engine_tokens"] == engine_tokens
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


@pytest.mark.parametrize(
    "sampling_overrides, expected_sampling",
    [
        ({"temperature": 0.0}, (0.0, 0.8, 20)),
        ({"top_p": 0.5, "top_k": 5}, (0.7, 0.5, 5)),
    ],
)
def test_instruct_explicit_sampling_overrides_task_default(
    monkeypatch: pytest.MonkeyPatch,
    sampling_overrides: dict[str, float | int],
    expected_sampling: tuple[float, float, int],
) -> None:
    """Supplied sampling fields must override VoiceDesign defaults without metadata."""
    _tokenizer, data = preprocess_instruct(
        instruct_payload(
            task_type="VoiceDesign",
            instructions="a calm deep voice",
            params=sampling_overrides,
        ),
        monkeypatch,
    )
    assert (
        data["text_temperature"],
        data["text_top_p"],
        data["text_top_k"],
    ) == expected_sampling


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


@pytest.mark.parametrize("dual_stream,budget", [(False, 620), (True, 1820)])
def test_speaker_preprocessing_composes_prompt_and_budget(
    dual_stream: bool, budget: int
) -> None:
    tokenizer = make_speaker_tokenizer()
    data = preprocess_speaker(
        speaker_payload(inputs={"text": "你 好"}, params={"dual_stream": dual_stream}),
        tokenizer=tokenizer,
    )
    assert data["variant"] == "speaker"
    assert data["task_type"] == "tts"
    assert data["dual_stream"] is dual_stream
    assert data["inference_cfg"] == 1.2
    assert data["max_engine_tokens"] == budget
    assert data["max_patches"] == 600
    if dual_stream:
        user_prompt = "Use a中性 tone. Align the interleaved text and speech chunks."
        assert data["text_stream_ids"] == [
            tokenizer.convert_tokens_to_ids(char) for char in "你好"
        ]
        assert data["text_pad_id"] == tokenizer.convert_tokens_to_ids("<|text_pad|>")
    else:
        user_prompt = "Convert text to speech.\n你好"
        assert data["text_stream_ids"] == []
        assert data["text_pad_id"] == -1
    expected = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n<|f531_speaker|><|audio_start|>"
    )
    assert tokenizer.decode(data["input_ids"]) == expected
    assert tokenizer.convert_tokens_to_ids("<|text_holder|>") not in data["input_ids"]


def test_speaker_dual_stream_style_passthrough() -> None:
    tokenizer = make_speaker_tokenizer()
    styled = preprocess_speaker(
        speaker_payload(params={"dual_stream": True, "style": "Use a 活泼 tone."}),
        tokenizer=tokenizer,
    )
    prompt = tokenizer.decode(styled["input_ids"])
    assert "Use a活泼 tone. Align the interleaved text and speech chunks." in prompt
    assert "Use a中性 tone." not in prompt
    plain = preprocess_speaker(speaker_payload(), tokenizer=tokenizer)
    styled_prefill = preprocess_speaker(
        speaker_payload(params={"style": "Use a 活泼 tone."}),
        tokenizer=tokenizer,
    )
    assert styled_prefill["input_ids"] == plain["input_ids"]


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


def test_speaker_max_new_tokens_override() -> None:
    data = preprocess_speaker(
        speaker_payload(params={"dual_stream": True, "max_new_tokens": 300})
    )
    assert data["max_engine_tokens"] == 300


@pytest.mark.parametrize(
    "task,instructions,params,tts_params,user_instruction,engine_tokens,sampling",
    [
        (
            "tts",
            None,
            {},
            {},
            "Convert text to speech.\n源音频文本。晚上好，欢迎收听今天的节目。",
            420,
            (0.0, None, None),
        ),
        (
            "VoiceDesign",
            "a calm deep voice",
            {},
            {},
            "a calm deep voice\n\n根据上述音色描述，首先整理成语音属性，再合成以下文本对应的音频：\n晚上好，欢迎收听今天的节目。",
            622,
            (0.7, 0.8, 20),
        ),
        (
            "SemanticEdit",
            "Replace '公园' with '游乐园'.",
            {},
            {},
            "Identify the content of the audio. Replace '公园' with '游乐园'.",
            622,
            (0.0, None, None),
        ),
        (
            "AcousticEdit",
            None,
            {},
            {"speed": 0.8},
            "adjust the speed to 0.8x",
            420,
            (0.0, None, None),
        ),
        (
            "AcousticEdit",
            None,
            {
                "stage_params": {
                    "preprocessing": {
                        "acoustic_attribute": "Pitch",
                        "acoustic_value": -2,
                    }
                }
            },
            {},
            "shift the pitch by -2 steps",
            420,
            (0.0, None, None),
        ),
    ],
)
def test_instruct_preprocessing_routes_tasks_to_model_prompts(
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    instructions: str | None,
    params: dict[str, JsonValue],
    tts_params: dict[str, JsonValue],
    user_instruction: str,
    engine_tokens: int,
    sampling: tuple[float, float | None, int | None],
) -> None:
    is_edit = task in ("SemanticEdit", "AcousticEdit")
    tokenizer, serialized_state = preprocess_instruct(
        instruct_payload(
            task_type=None if task == "tts" else task,
            instructions=instructions,
            text="" if is_edit else "晚上好，欢迎收听今天的节目。",
            with_reference=task != "VoiceDesign",
            params=params,
            tts_params=tts_params,
        ),
        monkeypatch,
    )
    state = FireRedTTS3State.from_dict(serialized_state)
    audio_input = "<|sosp|>" + "<|image_pad|>" * 18 + "<|eosp|>\n" if is_edit else ""
    if task == "tts":
        assistant_prefix = "<|sosp|>" + "<|video_pad|>" * 18
        assert state.prompt_audio_path == "edit.wav"
        assert state.prompt_text == "源音频文本。"
    elif task == "AcousticEdit":
        assistant_prefix = "<|sosp|>"
    else:
        assistant_prefix = "<|sot|>"
    expected = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{audio_input}{user_instruction} /no_think<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{assistant_prefix}"
    )
    assert tokenizer.encoded_texts == [expected]
    assert state.variant == "instruct"
    assert (
        state.task_type
        == {
            "tts": "tts",
            "VoiceDesign": "voice_design",
            "SemanticEdit": "semantic_edit",
            "AcousticEdit": "acoustic_edit",
        }[task]
    )
    assert state.audio_in_path == ("edit.wav" if is_edit else None)
    assert state.prompt_audio_path == ("edit.wav" if task == "tts" else None)
    assert state.inference_cfg == (2.0 if task == "tts" else 1.2)
    assert state.max_engine_tokens == engine_tokens
    assert (
        state.text_temperature,
        state.text_top_p,
        state.text_top_k,
    ) == sampling
