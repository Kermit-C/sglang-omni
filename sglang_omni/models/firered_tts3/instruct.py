# SPDX-License-Identifier: Apache-2.0
"""Prepare FireRedTTS3 Instruct tasks and their prompt sampling."""

from __future__ import annotations

import base64
import io
import math
from urllib.request import urlopen

import soundfile as sf
import torch
import torchaudio
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.payload_types import (
    FireRedTTS3State,
    FireRedTTS3TaskType,
    first_not_none,
    resolve_request_reference,
)
from sglang_omni.models.firered_tts3.prompt_utils import (
    MAX_EDIT_SECONDS,
    compose_acoustic_edit,
    compose_acoustic_instruction,
    compose_semantic_edit,
    compose_tts,
    compose_voice_design,
    tokenize_text,
)
from sglang_omni.utils.json import JsonValue

AUDIO_SAMPLE_RATE = 24000
AUDIO_URL_TIMEOUT_SECONDS = 60
SAMPLES_PER_PATCH = 3840

INSTRUCT_TASK_ALIASES: dict[str, FireRedTTS3TaskType] = {
    "tts": "tts",
    "base": "tts",
    "voicedesign": "voice_design",
    "semanticedit": "semantic_edit",
    "acousticedit": "acoustic_edit",
}

ACOUSTIC_ATTRIBUTES = ("Speed", "Pitch", "Volume")


def normalize_instruct_task(value: JsonValue) -> FireRedTTS3TaskType:
    task_type = (
        INSTRUCT_TASK_ALIASES.get(
            str(value).strip().replace("_", "").replace("-", "").lower()
        )
        if value is not None
        else None
    )
    if task_type is None:
        if value is None:
            return "tts"
        else:
            pass
        raise ValueError(
            f"unsupported FireRedTTS3-Instruct task_type {value!r}; expected one of VoiceDesign / SemanticEdit / AcousticEdit (or ref_audio tts)"
        )
    else:
        pass
    return task_type


def count_audio_patches(audio_path: str, *, cap_seconds: float | None) -> int:
    """Count prompt slots after clipping and resampling the reference audio."""
    # note (kermit): Prompt counting preserves its URL timeout and SoundFile error contract.
    audio_source: str | io.BytesIO
    if audio_path.startswith(("http://", "https://")):
        with urlopen(audio_path, timeout=AUDIO_URL_TIMEOUT_SECONDS) as response:
            audio_source = io.BytesIO(response.read())
    elif audio_path.startswith("data:"):
        _, _, encoded_audio = audio_path.partition(",")
        audio_source = io.BytesIO(base64.b64decode(encoded_audio))
    else:
        audio_source = audio_path
    waveform_samples, sample_rate = sf.read(audio_source, dtype="float32")
    if waveform_samples.ndim > 1:
        waveform_samples = waveform_samples.mean(axis=1)
    else:
        pass
    waveform = torch.from_numpy(waveform_samples)
    if cap_seconds is not None:
        waveform = waveform[..., : int(sample_rate * cap_seconds)]
    else:
        pass
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        pass
    waveform = waveform[:1].float()
    if sample_rate != AUDIO_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, AUDIO_SAMPLE_RATE
        )
    else:
        pass
    return math.ceil(waveform.shape[1] / SAMPLES_PER_PATCH)


def resolve_text_sampling(
    task_type: FireRedTTS3TaskType, request_parameters: dict[str, JsonValue]
) -> tuple[float, float | None, int | None]:
    """Apply explicit parameters over the task's text-sampling defaults."""
    if task_type == "voice_design":
        text_temperature, text_top_p, text_top_k = 0.7, 0.8, 20
    else:
        text_temperature, text_top_p, text_top_k = 0.0, None, None
    if request_parameters.get("temperature") is not None:
        text_temperature = float(request_parameters["temperature"])
    else:
        pass
    if request_parameters.get("top_p") is not None:
        text_top_p = float(request_parameters["top_p"])
    else:
        pass
    if request_parameters.get("top_k") is not None:
        text_top_k = int(request_parameters["top_k"])
    else:
        pass
    return text_temperature, text_top_p, text_top_k


def resolve_acoustic_instruction(
    instruction_text: str | None,
    tts_parameters: dict[str, JsonValue],
    preprocessing_parameters: dict[str, JsonValue],
) -> str:
    """Use an explicit instruction or a trained acoustic-edit template."""
    if instruction_text is not None and instruction_text.strip():
        return instruction_text.strip()
    else:
        pass
    attribute = str(
        first_not_none(
            preprocessing_parameters.get("acoustic_attribute"),
            tts_parameters.get("acoustic_attribute"),
            default="Speed",
        )
    )
    attribute = attribute.strip().capitalize()
    if attribute not in ACOUSTIC_ATTRIBUTES:
        raise ValueError(
            f"acoustic_attribute must be one of {ACOUSTIC_ATTRIBUTES}, got {attribute!r}"
        )
    else:
        pass
    value = first_not_none(
        preprocessing_parameters.get("acoustic_value"),
        tts_parameters.get("acoustic_value"),
        tts_parameters.get("speed"),
        default=1.0,
    )
    return compose_acoustic_instruction(attribute, float(value))


def compose_instruct_state(
    config: FireRedTTS3Config,
    *,
    tokenizer: PreTrainedTokenizerBase,
    task_type: FireRedTTS3TaskType,
    request_inputs: dict[str, JsonValue],
    request_parameters: dict[str, JsonValue],
    tts_parameters: dict[str, JsonValue],
    preprocessing_parameters: dict[str, JsonValue],
    text: str,
) -> FireRedTTS3State:
    instruction_text = first_not_none(
        request_inputs.get("instructions"), tts_parameters.get("instructions")
    )
    if instruction_text is not None:
        instruction_text = str(instruction_text).strip() or None
    else:
        pass
    reference_audio, prompt_text_value = resolve_request_reference(
        request_inputs, tts_parameters
    )
    prompt_audio_path: str | None = None
    audio_input_path: str | None = None
    prompt_text: str | None = None
    if task_type == "tts":
        if not text.strip():
            raise ValueError("FireRedTTS3 tts requires non-empty input text")
        else:
            pass
        if reference_audio is None:
            raise ValueError(
                "FireRedTTS3-Instruct tts requires a reference audio (ref_audio / references[0])"
            )
        else:
            pass
        if prompt_text_value is None or not str(prompt_text_value).strip():
            raise ValueError(
                "FireRedTTS3-Instruct tts requires the reference audio transcript (ref_text / references[0].text)"
            )
        else:
            pass
        prompt_audio_path = reference_audio
        prompt_text = str(prompt_text_value).strip()
        prompt_patch_count = count_audio_patches(prompt_audio_path, cap_seconds=None)
        prompt_template = compose_tts(prompt_patch_count, prompt_text, text.strip())
    elif task_type == "voice_design":
        if not text.strip():
            raise ValueError("FireRedTTS3 voice_design requires non-empty input text")
        else:
            pass
        if not instruction_text:
            raise ValueError("FireRedTTS3 voice_design requires instructions")
        else:
            pass
        prompt_template = compose_voice_design(instruction_text, text.strip())
    elif task_type == "semantic_edit":
        if not instruction_text:
            raise ValueError("FireRedTTS3 semantic_edit requires instructions")
        else:
            pass
        audio_input_path = require_edit_source(reference_audio, task_type)
        audio_input_patch_count = count_audio_patches(
            audio_input_path, cap_seconds=MAX_EDIT_SECONDS
        )
        prompt_template = compose_semantic_edit(
            instruction_text, audio_input_patch_count
        )
    elif task_type == "acoustic_edit":
        audio_input_path = require_edit_source(reference_audio, task_type)
        instruction = resolve_acoustic_instruction(
            instruction_text, tts_parameters, preprocessing_parameters
        )
        audio_input_patch_count = count_audio_patches(
            audio_input_path, cap_seconds=MAX_EDIT_SECONDS
        )
        prompt_template = compose_acoustic_edit(instruction, audio_input_patch_count)
    else:
        raise AssertionError(f"unhandled FireRedTTS3-Instruct task: {task_type}")
    text_temperature, text_top_p, text_top_k = resolve_text_sampling(
        task_type, request_parameters
    )
    return FireRedTTS3State(
        sample_rate=int(config.audio_sample_rate),
        variant="instruct",
        task_type=task_type,
        text=text.strip(),
        prompt_audio_path=prompt_audio_path,
        prompt_text=prompt_text,
        instructions=instruction_text,
        audio_in_path=audio_input_path,
        input_ids=tokenize_text(tokenizer, prompt_template),
        text_temperature=text_temperature,
        text_top_p=text_top_p,
        text_top_k=text_top_k,
    )


def require_edit_source(
    reference_audio: str | None, task_type: FireRedTTS3TaskType
) -> str:
    if reference_audio is None:
        raise ValueError(
            f"FireRedTTS3 {task_type} requires a reference audio (the edit source)"
        )
    else:
        pass
    return reference_audio
