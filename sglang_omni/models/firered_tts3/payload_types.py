# SPDX-License-Identifier: Apache-2.0
"""Per-request state carried by the FireRedTTS3 Omni pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Variant
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire
from sglang_omni.utils.audio_payload import audio_data_uri_from_reference
from sglang_omni.utils.json import JsonValue

FireRedTTS3TaskType = Literal["tts", "voice_design", "semantic_edit", "acoustic_edit"]


@dataclass
class FireRedTTS3State(DeclarativeStateBase):
    """Cross-stage prompt, generation, and audio state."""

    sample_rate: int = wire(24000, codec="int")
    variant: FireRedTTS3Variant = wire("base", codec="str_or")
    task_type: FireRedTTS3TaskType = wire("tts", codec="str_or")
    text: str = ""
    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    instructions: str | None = None
    language: str | None = None
    audio_in_path: str | None = None
    input_ids: list[int] = wire(default_factory=list, codec="list")
    n_timesteps: int = wire(10, codec="int_or")
    inference_cfg: float = wire(2.0, codec="float")
    stop_threshold: float = wire(0.5, codec="float")
    min_patches: int = wire(6, codec="int_or")
    max_patches: int = wire(400, codec="int_or")
    # note (kermit): A zero budget lets the request builder derive the decode limit.
    max_engine_tokens: int = wire(0, codec="int_or")
    dual_stream: bool = wire(False, codec="bool")
    text_stream_ids: list[int] = wire(default_factory=list, codec="list")
    text_pad_id: int = wire(-1, codec="int")
    seed: int | None = wire(None, codec="opt_int")
    stream: bool = wire(False, codec="bool")
    text_temperature: float | None = None
    text_top_p: float | None = None
    text_top_k: int | None = None
    # note (kermit): This wire field counts latent frames for vocoder prompt trimming.
    left_context_size: int = wire(0, codec="int")
    prompt_latents: torch.Tensor | None = wire(None, codec="typed_tensor")
    audio_in_latents: torch.Tensor | None = wire(None, codec="typed_tensor")
    speaker_embedding: torch.Tensor | None = wire(None, codec="typed_tensor")
    generated_latents: torch.Tensor | None = wire(None, codec="typed_tensor")
    gen_text: str | None = None
    finish_reason: str | None = None


def load_firered_tts3_state(payload: StagePayload) -> FireRedTTS3State:
    return FireRedTTS3State.from_dict(payload.data)


def store_firered_tts3_state(
    payload: StagePayload, request_state: FireRedTTS3State
) -> StagePayload:
    payload.data = request_state.to_dict()
    return payload


__all__ = ["FireRedTTS3State", "load_firered_tts3_state", "store_firered_tts3_state"]


def first_not_none(*values: JsonValue, default: JsonValue = None) -> JsonValue:
    return next((value for value in values if value is not None), default)


def as_dict(value: JsonValue) -> dict[str, JsonValue]:
    return dict(value) if isinstance(value, dict) else {}


def reference_path(value: JsonValue) -> str | None:
    if value is None:
        return None
    elif isinstance(value, str):
        return value
    elif isinstance(value, dict):
        audio_path = (
            value.get("audio_path")
            or value.get("path")
            or audio_data_uri_from_reference(value)
        )
        return str(audio_path) if audio_path else None
    else:
        raise TypeError("FireRedTTS3 reference audio must be a path or data URI")


def resolve_request_reference(
    request_inputs: dict[str, JsonValue], tts_parameters: dict[str, JsonValue]
) -> tuple[str | None, JsonValue]:
    """Read one reference audio source and its transcript."""
    references = request_inputs.get("references")
    if isinstance(references, list) and len(references) > 1:
        raise ValueError("FireRedTTS3 accepts at most one reference audio")
    else:
        pass
    reference = (
        references[0]
        if isinstance(references, list)
        and references
        and isinstance(references[0], dict)
        else {}
    )
    audio_path = first_not_none(
        reference_path(request_inputs.get("prompt_audio_path")),
        reference_path(request_inputs.get("prompt_audio")),
        reference_path(request_inputs.get("reference_audio")),
        reference_path(tts_parameters.get("ref_audio")),
        reference_path(reference),
    )
    reference_text = first_not_none(
        request_inputs.get("prompt_text"),
        request_inputs.get("reference_text"),
        tts_parameters.get("ref_text"),
        reference.get("text"),
    )
    return audio_path, reference_text
