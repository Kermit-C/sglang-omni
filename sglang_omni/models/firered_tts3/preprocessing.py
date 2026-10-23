# SPDX-License-Identifier: Apache-2.0
"""Prepare FireRedTTS3 requests for the talker stage."""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from sglang_omni.models.firered_tts3.hf_config import (
    FireRedTTS3Config,
    detect_fireredtts3_variant,
)
from sglang_omni.models.firered_tts3.instruct import (
    compose_instruct_state,
    normalize_instruct_task,
)
from sglang_omni.models.firered_tts3.payload_types import (
    FireRedTTS3State,
    as_dict,
    first_not_none,
    resolve_request_reference,
    store_firered_tts3_state,
)
from sglang_omni.models.firered_tts3.prompt_utils import (
    TEXT_PHASE_TASKS,
    TEXT_TOKEN_BUDGET,
    compose_base_text,
    encode_speaker_dual_stream_prompt,
    encode_speaker_prefill_ids,
    speaker_max_tokens,
    tokenize_text,
)
from sglang_omni.models.firered_tts3.request_builders import ENGINE_TOKEN_SLACK
from sglang_omni.proto.request import StagePayload
from sglang_omni.utils.json import JsonValue

MAX_TEXT_CHARS = 400
TEXT_TRANSITION_FEEDS = 2
TASK_GUIDANCE_SCALES: dict[str, float] = {
    "tts": 2.0,
    "voice_design": 1.2,
    "semantic_edit": 1.2,
    "acoustic_edit": 1.2,
}

MULTI_LANG_TAGS = (
    "Chinese",
    "English",
    "Cantonese",
    "Japanese",
    "Korean",
    "Spanish",
    "French",
    "Russian",
    "Arabic",
    "Turkish",
    "Indonesian",
    "Portuguese",
    "Italian",
    "Dutch",
    "Vietnamese",
    "German",
    "Ukrainian",
    "Thai",
    "Polish",
    "Romanian",
    "Greek",
    "Czech",
    "Finnish",
    "Hindi",
)

MULTI_DIALECT_TAGS = tuple(
    f"ZH_{name}"
    for name in (
        "Anhui",
        "Fujian",
        "Gansu",
        "Guizhou",
        "Hebei",
        "Henan",
        "Hubei",
        "Hunan",
        "Jiangxi",
        "Liaoning",
        "Minnan",
        "Ningxia",
        "Shaanxi",
        "Shandong",
        "Shanghai",
        "Shanxi",
        "Sichuan",
        "Tianjin",
        "Wenzhou",
        "Wu",
        "Yunnan",
    )
)


def preprocess_firered_tts3_payload(
    payload: StagePayload,
    *,
    tokenizer: PreTrainedTokenizerBase,
    config: FireRedTTS3Config,
    n_timesteps: int,
    max_patches: int,
) -> StagePayload:
    input_values = payload.request.inputs
    request_inputs = (
        {"text": input_values}
        if isinstance(input_values, str)
        else as_dict(input_values)
    )
    request_parameters = as_dict(payload.request.params)
    tts_parameters = as_dict(as_dict(payload.request.metadata).get("tts_params"))
    stage_parameters = as_dict(request_parameters.get("stage_params"))
    talker_parameters = as_dict(stage_parameters.get("talker"))
    preprocessing_parameters = as_dict(stage_parameters.get("preprocessing"))
    should_stream = bool(request_parameters.get("stream", False))
    text = str(
        first_not_none(
            request_inputs.get("input"), request_inputs.get("text"), default=""
        )
    )
    if len(text.strip()) > MAX_TEXT_CHARS:
        raise ValueError(
            f"FireRedTTS3 input text must be at most {MAX_TEXT_CHARS} characters"
        )
    else:
        pass
    generation_defaults = dict(config.generation_defaults or {})

    def resolve_number(parameter_name: str, fallback: int | float | None) -> JsonValue:
        return first_not_none(
            request_inputs.get(parameter_name),
            tts_parameters.get(parameter_name),
            talker_parameters.get(parameter_name),
            request_parameters.get(parameter_name),
            default=fallback,
        )

    sampling_seed = resolve_number("seed", None)
    variant = detect_fireredtts3_variant(config)
    interleave_ratio = (
        (
            int(config.text_latent_interleave_ratio[0]),
            int(config.text_latent_interleave_ratio[1]),
        )
        if config.text_latent_interleave_ratio is not None
        else None
    )
    if variant == "instruct":
        task_type = normalize_instruct_task(
            first_not_none(
                request_inputs.get("task_type"),
                tts_parameters.get("task_type"),
                request_parameters.get("task_type"),
            )
        )
        request_state = compose_instruct_state(
            config,
            tokenizer=tokenizer,
            task_type=task_type,
            request_inputs=request_inputs,
            request_parameters=request_parameters,
            tts_parameters=tts_parameters,
            preprocessing_parameters=preprocessing_parameters,
            text=text,
        )
    elif variant == "base":
        request_state = compose_base_state(
            config,
            tokenizer=tokenizer,
            request_inputs=request_inputs,
            tts_parameters=tts_parameters,
            text=text,
        )
    elif variant == "speaker":
        request_state = compose_speaker_state(
            config,
            tokenizer=tokenizer,
            request_inputs=request_inputs,
            request_parameters=request_parameters,
            tts_parameters=tts_parameters,
            preprocessing_parameters=preprocessing_parameters,
            text=text,
        )
    else:
        raise AssertionError(f"unhandled FireRedTTS3 variant: {variant}")
    request_state.n_timesteps = int(
        resolve_number(
            "n_timesteps", generation_defaults.get("n_timesteps", n_timesteps)
        )
    )
    guidance_default = generation_defaults.get("inference_cfg", 2.0)
    if variant == "instruct":
        guidance_default = TASK_GUIDANCE_SCALES.get(
            request_state.task_type, guidance_default
        )
    else:
        pass
    request_state.inference_cfg = float(
        resolve_number("inference_cfg", guidance_default)
    )
    request_state.stop_threshold = float(
        resolve_number("stop_threshold", generation_defaults.get("stop_threshold", 0.5))
    )
    request_state.min_patches = int(
        resolve_number("min_patches", generation_defaults.get("min_patches", 6))
    )
    request_state.max_patches = int(resolve_number("max_patches", max_patches))
    request_state.seed = None if sampling_seed is None else int(sampling_seed)
    request_state.stream = should_stream
    request_state.max_engine_tokens = int(
        request_parameters.get("max_new_tokens")
        or engine_token_budget(request_state, interleave_ratio=interleave_ratio)
    )
    return store_firered_tts3_state(payload, request_state)


def engine_token_budget(
    request_state: FireRedTTS3State, *, interleave_ratio: tuple[int, int] | None = None
) -> int:
    """Include text and transition feeds in the request's decode budget."""
    if request_state.dual_stream:
        if interleave_ratio is None:
            raise ValueError(
                "FireRedTTS3 dual-stream requests require the checkpoint's text_latent_interleave_ratio"
            )
        else:
            pass
        return speaker_max_tokens(
            request_state.max_patches, interleave_ratio, slack=ENGINE_TOKEN_SLACK
        )
    else:
        pass
    budget = request_state.max_patches + ENGINE_TOKEN_SLACK
    if request_state.task_type in TEXT_PHASE_TASKS:
        budget += TEXT_TOKEN_BUDGET + TEXT_TRANSITION_FEEDS
    else:
        pass
    return budget


def compose_base_state(
    config: FireRedTTS3Config,
    *,
    tokenizer: PreTrainedTokenizerBase,
    request_inputs: dict[str, JsonValue],
    tts_parameters: dict[str, JsonValue],
    text: str,
) -> FireRedTTS3State:
    if not text.strip():
        raise ValueError("FireRedTTS3 requires non-empty input text")
    else:
        pass
    text = text.strip()
    prompt_audio_path, prompt_text_value = resolve_request_reference(
        request_inputs, tts_parameters
    )
    if prompt_audio_path is None:
        raise ValueError("FireRedTTS3-Base requires a reference audio (voice cloning)")
    else:
        pass
    if prompt_text_value is None or not str(prompt_text_value).strip():
        raise ValueError(
            "FireRedTTS3-Base requires the reference audio transcript (ref_text / references[0].text)"
        )
    else:
        pass
    prompt_text = str(prompt_text_value).strip()
    language = str(
        first_not_none(
            request_inputs.get("language"),
            tts_parameters.get("language"),
            default="Chinese",
        )
    )
    if language not in MULTI_LANG_TAGS + MULTI_DIALECT_TAGS:
        raise ValueError(
            f"unsupported FireRedTTS3 language {language!r}; expected one of MULTI_LANG_TAGS/MULTI_DIALECT_TAGS"
        )
    else:
        pass
    return FireRedTTS3State(
        sample_rate=int(config.audio_sample_rate),
        variant="base",
        task_type="tts",
        text=text,
        prompt_audio_path=prompt_audio_path,
        prompt_text=prompt_text,
        language=language,
        input_ids=tokenize_text(
            tokenizer, compose_base_text(language, prompt_text, text)
        ),
    )


def compose_speaker_state(
    config: FireRedTTS3Config,
    *,
    tokenizer: PreTrainedTokenizerBase,
    request_inputs: dict[str, JsonValue],
    request_parameters: dict[str, JsonValue],
    tts_parameters: dict[str, JsonValue],
    preprocessing_parameters: dict[str, JsonValue],
    text: str,
) -> FireRedTTS3State:
    """Build the prefill or interleaved speaker prompt."""
    if not text.strip():
        raise ValueError("FireRedTTS3 speaker requests require non-empty input text")
    else:
        pass
    reference_audio, prompt_text = resolve_request_reference(
        request_inputs, tts_parameters
    )
    if reference_audio is not None or (
        prompt_text is not None and str(prompt_text).strip()
    ):
        raise ValueError(
            "FireRedTTS3 speaker variant takes text-only tts; reference audio is not used (deploy a Base/Instruct checkpoint for voice cloning)"
        )
    else:
        pass
    task_type = first_not_none(
        request_inputs.get("task_type"),
        tts_parameters.get("task_type"),
        request_parameters.get("task_type"),
    )
    if task_type is not None and str(task_type).strip().lower() not in ("tts", "base"):
        raise ValueError(
            f"FireRedTTS3 speaker variant supports only text-only tts; got task_type {task_type!r}"
        )
    else:
        pass
    dual_stream = bool(
        first_not_none(
            request_parameters.get("dual_stream"),
            tts_parameters.get("dual_stream"),
            preprocessing_parameters.get("dual_stream"),
            default=False,
        )
    )
    interleave_ratio = config.text_latent_interleave_ratio
    if dual_stream and interleave_ratio is None:
        raise ValueError(
            "FireRedTTS3 dual_stream requires a checkpoint with text_latent_interleave_ratio in its config"
        )
    else:
        pass
    speaker_token = str(config.speaker_token or "<|rand_speaker|>")
    if dual_stream:
        style = first_not_none(
            request_parameters.get("style"),
            tts_parameters.get("style"),
            preprocessing_parameters.get("style"),
        )
        prompt_token_ids, text_token_ids, text_padding_token_id = (
            encode_speaker_dual_stream_prompt(
                tokenizer,
                text=text.strip(),
                speaker_token=speaker_token,
                style=None if style is None else str(style),
                interleave_ratio=(int(interleave_ratio[0]), int(interleave_ratio[1])),
            )
        )
        return FireRedTTS3State(
            sample_rate=int(config.audio_sample_rate),
            variant="speaker",
            task_type="tts",
            text=text.strip(),
            input_ids=prompt_token_ids,
            dual_stream=True,
            text_stream_ids=text_token_ids,
            text_pad_id=text_padding_token_id,
        )
    else:
        pass
    return FireRedTTS3State(
        sample_rate=int(config.audio_sample_rate),
        variant="speaker",
        task_type="tts",
        text=text.strip(),
        input_ids=encode_speaker_prefill_ids(
            tokenizer, text=text.strip(), speaker_token=speaker_token
        ),
    )
