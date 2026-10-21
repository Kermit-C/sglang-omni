# SPDX-License-Identifier: Apache-2.0
"""Map the FireRedTTS3 pipeline state onto SGLang's request lifecycle.

Base requests see a placeholder prompt (1 speaker slot + text tokens +
prompt-patch slots) whose embeddings before_prefill replaces with the
assembled [spk_proj_llm(spk); embed(text); patch_encoder(prompt_latents)]
row; Instruct and speaker requests carry the real chatml tokens (Instruct:
the audio latents live inline as image_pad / video_pad slots, scattered at
prefill; speaker: the text ids are scattered into the text_holder slots at
preprocessing time, or ride the payload for dual-stream requests). Audio-phase
decode steps are one latent patch each: the runner overwrites next_token_ids
with an inert control token and drives stop/finish itself (stop_head
threshold -> FINISH_MATCHED_TOKEN). Text-phase requests (Instruct
voice_design / semantic_edit) instead let the engine's real sampling through
until the text-AR machine reaches the audio phase — <|eot|> never joins the
stop set; it triggers the no-sampling finalize feed, then the doubled
<|sosp|> transition into the audio loop.

Speaker dual-stream requests (dual_stream=True, interleave checkpoints) run
the dual-stream decode machine instead: the prompt ends at <|audio_start|> and
the char-modelled text ids feed nt-at-a-time interleaved with na patch feeds
per group ((nt + na) / na engine steps per patch), text_pad extending the
stream once exhausted; the flow fires on each group's last text feed (and
intermediate patch feeds), the stop check on the last patch feed — the
reference predicts stop AFTER the patch enters the backbone, so a stopping
step HAS already emitted its patch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.firered_tts3.payload_types import (
    FireRedTTS3State,
    load_firered_tts3_state,
    store_firered_tts3_state,
)
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.models.firered_tts3.prompt_utils import TEXT_PHASE_TASKS
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import RequestOutput

# Inert in-vocab token id carried by next_token_ids / output_ids; the latent
# loop never samples from the backbone vocab, and the runner overwrites the
# row every step. The value only needs to be inside the tokenizer's vocab.
CONTROL_TOKEN_ID = 0

# Engine-step slack above max_patches: the stop check runs before a step's
# patch is produced, so a request that generates max_patches patches consumes
# one extra step for the stop decision.
ENGINE_TOKEN_SLACK = 20

# Generated patches per emitted stream chunk: the first chunk is a single
# patch (TTFP), the second pairs two, and later chunks settle on four — the
# reference codec_chunk_frames [4, 8, 16] ramp counted in patches.
STREAM_CHUNK_PATCH_RAMP = (1, 2, 4)


@dataclass(frozen=True)
class FireRedFlowResume:
    """Surviving state of a retracted request; the rest is rebuilt on re-prefill."""

    rng_state: torch.Tensor | None


@dataclass
class FireRedTTS3SGLangRequestData(SGLangARRequestData):
    state: FireRedTTS3State = field(default_factory=FireRedTTS3State)
    prompt_patch_count: int = 0
    prefill_len: int = 0
    flow_state: Any | None = None
    latent_patches: list[torch.Tensor] = field(default_factory=list)
    probe: ProfilingProbe | None = None
    engine_start_s: float = 0.0
    streamed_patches: int = 0
    stream_chunk_index: int = 0
    # Instruct text-AR machine: "text" (sampling) -> "finalize" (the EOT/budget
    # feed) -> "sosp" -> "audio_zero" (the doubled sosp feed driving patch 0)
    # -> "audio". Requests without a text phase stay in "audio" throughout.
    # The fields persist on the request data across retraction (the engine
    # keeps output_ids, so the machine rebuilds losslessly on re-prefill).
    text_ar_mode: str = "audio"
    text_fed: int = 0
    text_token_ids: list[int] = field(default_factory=list)
    eot_sampled: bool = False
    # Dual-stream machine (speaker variant): per (nt, na) group, nt text-token
    # feeds then na patch feeds. ds_stage/ds_index locate the CURRENT step
    # within its group; text_pos is the cursor into state.text_stream_ids.
    # Like the text machine, these persist across retraction and are derived
    # from output_ids on re-prefill.
    dual_stream: bool = False
    interleave_ratio: tuple[int, int] | None = None
    ds_stage: str = "text"
    ds_index: int = 0
    text_pos: int = 0
    # Async-decode stop bridge: a stop decided at a launch is deferred to that
    # step's resolve (post_decode_resolve applies it) so the scheduler's
    # overrun drop cannot eat the finish. Only ever set under lookahead.
    stop_pending: bool = False


def build_sglang_firered_tts3_request(
    payload: StagePayload,
    *,
    patch_size: int,
    vocab_size: int,
    interleave_ratio: tuple[int, int] | None = None,
) -> FireRedTTS3SGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    state = load_firered_tts3_state(payload)
    if not state.input_ids:
        raise RuntimeError("FireRedTTS3 preprocessing did not produce input_ids")
    else:
        pass
    prompt_frames = (
        0 if state.prompt_latents is None else int(state.prompt_latents.shape[1])
    )
    if prompt_frames % patch_size:
        raise ValueError(
            f"FireRedTTS3 prompt latents ({prompt_frames} frames) are not "
            f"aligned to the patch size ({patch_size})"
        )
    else:
        pass
    prompt_patch_count = prompt_frames // patch_size
    if state.variant in ("instruct", "speaker"):
        # Real chatml tokens: Instruct audio latents live INLINE as
        # image_pad / video_pad placeholder slots (scattered by the runner at
        # prefill); speaker text ids were scattered into the text_holder slots
        # by preprocessing (prefill form) or ride the payload (dual-stream).
        origin_input_ids = list(state.input_ids)
        prefill_len = len(origin_input_ids)
    else:
        # Placeholder row: 1 speaker slot + text tokens + prompt-patch slots;
        # before_prefill overwrites it with the assembled embeds.
        prefill_len = 1 + len(state.input_ids) + prompt_patch_count
        origin_input_ids = [CONTROL_TOKEN_ID] * prefill_len
    if state.dual_stream:
        if interleave_ratio is None:
            raise ValueError(
                "FireRedTTS3 dual-stream requests require a checkpoint with "
                "text_latent_interleave_ratio"
            )
        else:
            pass
        if not state.text_stream_ids or state.text_pad_id < 0:
            raise ValueError(
                "FireRedTTS3 dual-stream requests require text_stream_ids and "
                "text_pad_id from preprocessing"
            )
        else:
            pass
    else:
        pass
    max_engine_tokens = (
        state.max_engine_tokens or state.max_patches + ENGINE_TOKEN_SLACK
    )

    sampling_params = SamplingParams(
        max_new_tokens=max_engine_tokens,
        temperature=(
            state.text_temperature if state.text_temperature is not None else 0.0
        ),
        stop_token_ids=[],
    )
    if state.text_top_p is not None:
        sampling_params.top_p = state.text_top_p
    else:
        pass
    if state.text_top_k is not None:
        sampling_params.top_k = state.text_top_k
    else:
        pass
    # The seeded engine sampler drives the text phase (the flow noise keeps
    # its own per-request generator seeded in the runner).
    if state.seed is not None and sampling_params.temperature > 0:
        sampling_params.sampling_seed = int(state.seed)
    else:
        pass
    sampling_params.normalize(None)
    sampling_params.verify(int(vocab_size))
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=origin_input_ids,
        sampling_params=sampling_params,
        eos_token_ids=set(),
        vocab_size=int(vocab_size),
    )
    req.tokenizer = None
    req._input_embeds_are_projected = (
        True  # noqa: leading-underscore  # SGLang request attribute
    )
    req._codec_suppress_tokens = (
        None  # noqa: leading-underscore  # SGLang request attribute
    )
    return FireRedTTS3SGLangRequestData(
        state=state,
        stage_payload=payload,
        req=req,
        output_ids=req.output_ids,
        prompt_patch_count=prompt_patch_count,
        prefill_len=prefill_len,
        max_new_tokens=max_engine_tokens,
        input_embeds_are_projected=True,
        engine_start_s=time.perf_counter(),
        text_ar_mode=("text" if state.task_type in TEXT_PHASE_TASKS else "audio"),
        dual_stream=state.dual_stream,
        interleave_ratio=interleave_ratio if state.dual_stream else None,
    )


def apply_latent_result(
    data: FireRedTTS3SGLangRequestData,
    text_tokenizer: PreTrainedTokenizerBase | None = None,
) -> StagePayload:
    state = data.state
    if not data.latent_patches:
        raise RuntimeError("FireRedTTS3 generated no latent patches")
    else:
        pass
    state.generated_latents = torch.cat(data.latent_patches, dim=1)
    state.prompt_tokens = data.prefill_len
    state.completion_tokens = len(data.latent_patches)
    state.engine_time_s = time.perf_counter() - data.engine_start_s
    state.finish_reason = data.finish_reason
    if data.text_token_ids and text_tokenizer is not None:
        # The recorded text-phase tokens (EOT excluded, matching the
        # reference's text_gen_ids which never appends the terminator).
        state.gen_text = text_tokenizer.decode(data.text_token_ids)
    else:
        pass
    return store_firered_tts3_state(data.stage_payload, state)


def emit_latent_chunk(
    request_id: str, data: FireRedTTS3SGLangRequestData, take: int
) -> OutgoingMessage:
    patches = data.latent_patches[data.streamed_patches : data.streamed_patches + take]
    chunk = patches[0] if len(patches) == 1 else torch.cat(patches, dim=1)
    data.streamed_patches += len(patches)
    data.stream_chunk_index += 1
    metadata: dict[str, Any] = {"modality": "audio_latents", "stream": True}
    if chunk.device.type == "cuda":
        # The vocoder shares this process and reads the chunk on its own
        # stream, so it waits on the event that marks the chunk complete
        # instead of on everything the talker has enqueued.
        event = torch.cuda.Event()
        event.record()
        metadata["codes_ready_event"] = event
    else:
        pass
    return OutgoingMessage(
        request_id=request_id,
        type="stream",
        target="vocoder",
        data=chunk,
        metadata=metadata,
    )


def build_stream_output(
    request_id: str,
    data: FireRedTTS3SGLangRequestData,
    req_output: RequestOutput,
) -> list[OutgoingMessage]:
    """Ramp-sized latent chunks: 1 patch, then 2, then 4 at a time.

    Dual-stream requests emit every patch the moment it is generated (the
    reference's streaming form decodes per patch: 3120 samples on the first
    chunk, 3840 in steady state, 4560 on the terminal one).
    """
    del req_output
    if not data.state.stream:
        return []
    else:
        pass
    if data.dual_stream:
        pending = len(data.latent_patches) - data.streamed_patches
        if pending < 1:
            return []
        else:
            pass
        return [emit_latent_chunk(request_id, data, 1)]
    else:
        pass
    ramp = STREAM_CHUNK_PATCH_RAMP[
        min(data.stream_chunk_index, len(STREAM_CHUNK_PATCH_RAMP) - 1)
    ]
    pending = len(data.latent_patches) - data.streamed_patches
    if pending < ramp:
        return []
    else:
        pass
    return [emit_latent_chunk(request_id, data, ramp)]


def flush_stream_output(
    request_id: str, data: FireRedTTS3SGLangRequestData
) -> list[OutgoingMessage]:
    """Terminal drain: everything the ramp held back reaches the vocoder."""
    if not data.state.stream:
        return []
    else:
        pass
    pending = len(data.latent_patches) - data.streamed_patches
    if pending <= 0:
        return []
    else:
        pass
    return [emit_latent_chunk(request_id, data, pending)]


build_stream_output.flush = flush_stream_output  # type: ignore[attr-defined]


__all__ = [
    "CONTROL_TOKEN_ID",
    "ENGINE_TOKEN_SLACK",
    "STREAM_CHUNK_PATCH_RAMP",
    "FireRedFlowResume",
    "FireRedTTS3SGLangRequestData",
    "apply_latent_result",
    "build_sglang_firered_tts3_request",
    "build_stream_output",
]
