# SPDX-License-Identifier: Apache-2.0
"""Per-request state carried by the FireRedTTS3 Omni pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class FireRedTTS3State(DeclarativeStateBase):
    """Only cross-stage state; scheduler/runner state stays in Omni.

    Field order follows the pipeline: preprocessing fills the template and
    generation fields, reference_encode fills the conditioning tensors, the
    talker fills generated_latents / gen_text, and the vocoder consumes
    prompt_latents (SWA left context) plus generated_latents.
    """

    sample_rate: int = wire(24000, codec="int")
    variant: str = wire("base", codec="str_or")
    task_type: str = wire("tts", codec="str_or")
    text: str = ""
    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    instructions: str | None = None
    language: str | None = None
    # Edit-source audio (semantic_edit / acoustic_edit): encoded by the
    # reference_encode stage into the image_pad slots of the Instruct prompt.
    audio_in_path: str | None = None
    input_ids: list[int] = wire(default_factory=list, codec="list")
    n_timesteps: int = wire(10, codec="int_or")
    inference_cfg: float = wire(2.0, codec="float")
    stop_threshold: float = wire(0.5, codec="float")
    min_patches: int = wire(6, codec="int_or")
    max_patches: int = wire(400, codec="int_or")
    # Engine decode-step budget derived from max_patches per topology
    # (prefill-shaped: max_patches + 20; dual-stream: 3 * max_patches + 20).
    max_engine_tokens: int = wire(0, codec="int_or")
    dual_stream: bool = wire(False, codec="bool")
    # Dual-stream text input stream (speaker variant): the char-modelled text
    # ids the talker's dual-stream machine feeds nt-at-a-time between patch
    # feeds, and the <|text_pad|> id extending the stream once exhausted.
    text_stream_ids: list[int] = wire(default_factory=list, codec="list")
    text_pad_id: int = wire(-1, codec="int")
    seed: int | None = wire(None, codec="opt_int")
    stream: bool = wire(False, codec="bool")
    # Resolved text-phase sampling parameters (None = the task default of the
    # request builder: greedy). voice_design samples at 0.7/0.8/20.
    text_temperature: float | None = None
    text_top_p: float | None = None
    text_top_k: int | None = None
    # Prompt-latent frame count; the vocoder trims the matching waveform
    # prefix (left_context_size * 960 samples) from the decoded output.
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
    payload: StagePayload, state: FireRedTTS3State
) -> StagePayload:
    payload.data = state.to_dict()
    return payload


__all__ = [
    "FireRedTTS3State",
    "load_firered_tts3_state",
    "store_firered_tts3_state",
]
