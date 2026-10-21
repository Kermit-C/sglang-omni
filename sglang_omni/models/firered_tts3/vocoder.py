# SPDX-License-Identifier: Apache-2.0
"""FireRedTTS3 RedAE vocoder adapters for the Omni vocoder schedulers.

The decoder consumes [prompt latents, generated latents] jointly — the SWA(64)
Qwen3 needs the prompt frames as left context — and the client stream starts
at the prompt boundary (left_context_size x 960 samples into the joint
decode), matching the reference decode-then-trim semantics. The non-streaming
path decodes the whole latent sequence by a single forward_stream call
(cache=None, last_chunk=True), the full-payload special case that is
sample-exact with the reference's full-segment decode.

The streaming path decodes incrementally with a per-request RedAEDecoderState
(Qwen3 KV cache + iSTFT OLA cache):

* The prompt latents arrive ahead of the generated chunks (streamed by the
  reference_encode stage right after encoding). With prompt_predecode on,
  they are decoded at ingest — warming the KV/OLA caches while the talker
  prefills — and their (discarded) waveform never crosses the prompt
  boundary; with it off (the default; measured faster TTFP while every stage
  shares the default CUDA stream) they are buffered and decoded jointly with
  the first generated chunk (the reference streaming decoder's chunk-0
  semantics).
* Generated chunks (ramp-sized by the talker) decode on arrival; the terminal
  flush re-emits nothing when the last chunk landed with pending latents
  (last_chunk=True covers the OLA tail) and otherwise folds the cached
  windowed frames for the tail (RedAEAudioDecoder.flush_stream).
* All emissions are trimmed against the joint-decode timeline so the streamed
  samples equal the batch path's output sample-for-sample (up to the SWA
  chunked-vs-whole fp32 noise floor, ~2e-6 measured).

The decoder runs fp32 eager (iSTFT/phase is a precision red line). Latents
arrive in the scaled working domain and are divided by redae_scale before
decode (Base: 1.0, an exact no-op).

With stream_batching on, steady-state chunks decode in equal-progress groups:
streams at the same position — equal frames-in, cache presence and decoder
progress (50 Hz frames seen) — merge their RedAEDecoderState into one batched
forward_stream and split the result back into per-row views (the SWA cache
layers track a single cumulative length, so only lockstep rows can share a
cache). The terminal flush never enters the pump (finish_stream decodes it
per request), so last_chunk is constant False inside a group. Size-1 groups
run the exact serial decode, keeping single-request output bit-exact.

With separate_decode_stream on, every decode runs on a dedicated CUDA
stream instead of the legacy default stream the talker produces on. The
talker's chunk arrives with a readiness event (recorded at emission), so
the decode stream waits on exactly the chunk's production rather than on
everything the talker has queued: on the shared stream the first chunk's
decode (a fresh-cache pass, whose per-layer cache initialization
stream-syncs) serializes behind the talker's back-to-back graph replays —
measured on L20 as c=1 TTFP 291 -> 99 ms. Chunk tensors are
default-stream allocations read on the decode stream, so they are pinned
with record_stream against early reuse; emitted waveforms are copied to
the host on the decode stream for the same ordering reason.

With incremental_cuda_graph on, warm steady-state chunk decodes replay
self-captured CUDA graphs (one per chunk width x batch bucket; see
vocoder_graph.py): the SWA cache carries exactly sliding_window-1 frames in
steady state, so the whole in_proj + Qwen3 + iSTFT step is a fixed shape.
B=1 replays are bit-exact with the eager decode. Cold decodes (fresh cache,
joint prompt+first-chunk, ramp widths below the full window) stay eager.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.firered_tts3.payload_types import (
    FireRedTTS3State,
    load_firered_tts3_state,
)
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.models.firered_tts3.redae_decoder import (
    RedAEAudioDecoder,
    RedAEDecoderState,
)
from sglang_omni.models.firered_tts3.vocoder_cold_graph import (
    DEFAULT_COLD_BATCH_SIZES as GRAPH_DEFAULT_COLD_BATCH_SIZES,
)
from sglang_omni.models.firered_tts3.vocoder_cold_graph import (
    DEFAULT_COLD_LATENT_SIZES as GRAPH_DEFAULT_COLD_LATENT_SIZES,
)
from sglang_omni.models.firered_tts3.vocoder_cold_graph import (
    RedAEVocoderColdGraphRunner,
)
from sglang_omni.models.firered_tts3.vocoder_graph import (
    DEFAULT_BATCH_SIZES as GRAPH_DEFAULT_BATCH_SIZES,
)
from sglang_omni.models.firered_tts3.vocoder_graph import (
    DEFAULT_LATENT_SIZES as GRAPH_DEFAULT_LATENT_SIZES,
)
from sglang_omni.models.firered_tts3.vocoder_graph import RedAEVocoderGraphRunner
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

# 1 latent frame (25 Hz) = 960 samples at 24 kHz.
SAMPLES_PER_LATENT_FRAME = 960

# Stream-chunk metadata keys for the reference_encode -> vocoder prompt
# handoff (carried on the first chunk of every streaming request).
PROMPT_CHUNK_METADATA_KEY = "prompt_chunk"
LEFT_CONTEXT_METADATA_KEY = "left_context_size"

# Set FIREREDTTS3_DEBUG_DUMP_DIR to save every decode's working-domain latent
# input (prompt + generated) as <request counter>.pt for oracle diffs.
DEBUG_DUMP_DIR_ENV = "FIREREDTTS3_DEBUG_DUMP_DIR"

# Warmup chunk shape: covers both the fresh-cache and the cached+terminal
# forward_stream paths (vLLM measured ~750 ms cold vs ~17 ms warm).
WARMUP_FRAMES = 16


class FireRedTTS3BatchVocoder(BatchVocoderBase):
    """Non-streaming RedAE decode: one full-payload forward_stream per item.

    decode_stream, when set, carries every decode off the legacy default
    stream so a full-payload decode overlaps the talker's in-flight steps
    instead of queueing behind them. Payload tensors arrive host-side (the
    state codec round-trips through CPU bytes), so no readiness event is
    needed on this path.
    """

    def __init__(
        self,
        decoder: RedAEAudioDecoder,
        *,
        redae_scale: float,
        sample_rate: int,
        decode_stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.decoder = decoder
        self.redae_scale = float(redae_scale)
        self.sample_rate = int(sample_rate)
        self.decode_stream = decode_stream
        self.lock = threading.Lock()
        self.debug_dump_count = 0

    def decode_stream_context(
        self,
    ) -> contextlib.nullcontext | torch.cuda.StreamContext:
        if self.decode_stream is None:
            return contextlib.nullcontext()
        else:
            pass
        return torch.cuda.stream(self.decode_stream)

    def maybe_dump_latents(self, latents: torch.Tensor) -> None:
        dump_dir = os.environ.get(DEBUG_DUMP_DIR_ENV)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            path = os.path.join(dump_dir, f"latents_{self.debug_dump_count}.pt")
            torch.save(latents.detach().cpu(), path)
            self.debug_dump_count += 1
        else:
            pass

    def prepare_item(
        self, payload: StagePayload
    ) -> tuple[FireRedTTS3State, torch.Tensor]:
        state = load_firered_tts3_state(payload)
        if state.generated_latents is None or state.generated_latents.numel() == 0:
            raise RuntimeError("FireRedTTS3 vocoder received no generated latents")
        else:
            pass
        parts = []
        if state.prompt_latents is not None and state.prompt_latents.numel():
            parts.append(state.prompt_latents)
        else:
            pass
        parts.append(state.generated_latents)
        return state, torch.cat(parts, dim=1)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode one full latent payload; returns the waveform on the host.

        The readback runs on the decode stream (when set) so it is ordered
        after the decode; callers only ever serialize the result to bytes.
        """
        if latents.ndim != 3 or latents.shape[0] != 1:
            raise ValueError(
                "FireRedTTS3 latents must have shape [1, frames, redae_dim], "
                f"got {tuple(latents.shape)}"
            )
        else:
            pass
        if latents.shape[2] != self.decoder.in_proj.in_features:
            raise ValueError(
                f"FireRedTTS3 latent width must be {self.decoder.in_proj.in_features}, "
                f"got {latents.shape[2]}"
            )
        else:
            pass
        self.maybe_dump_latents(latents)
        with self.lock, torch.no_grad(), self.decode_stream_context():
            probe = ProfilingProbe.create()
            with probe.segment("redae_decode"):
                audio, _ = self.decoder.forward_stream(
                    (latents / self.redae_scale).to(
                        device=next(self.decoder.parameters()).device,
                        dtype=torch.float32,
                    ),
                    None,
                    last_chunk=True,
                )
            probe.log("batch-vocoder")
            return audio.cpu()

    async def decode_batch(
        self, items: list[tuple[FireRedTTS3State, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, int]]:
        outputs: list[tuple[torch.Tensor, int]] = []
        for state, latents in items:
            audio = self.decode_latents(latents)
            trim = int(state.left_context_size) * SAMPLES_PER_LATENT_FRAME
            if trim > 0:
                audio = audio[:, trim:]
            else:
                pass
            outputs.append((audio, self.sample_rate))
        return outputs

    def store_result(
        self,
        payload: StagePayload,
        state: FireRedTTS3State,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        payload.data = audio_waveform_payload(
            wav,
            sample_rate=sample_rate,
            modality="audio",
            source_hint="fireredtts3",
        )
        if state.gen_text:
            # The Instruct text-phase product (voice plan / edited
            # transcript); surfaces as chunk.text on the client.
            payload.data["text"] = state.gen_text
        else:
            pass
        usage = build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        else:
            pass
        return payload

    async def decode_payload(self, payload: StagePayload) -> StagePayload:
        state, latents = self.prepare_item(payload)
        [(wav, sample_rate)] = await self.decode_batch([(state, latents)])
        return self.store_result(payload, state, wav, sample_rate)


@dataclass
class FireRedStreamState:
    """Per-request incremental decode state (coalescing swaps decoder_state
    for a row view of the group's batched state after every grouped step)."""

    decoder_state: RedAEDecoderState | None = None
    # Buffered generated chunks not yet decoded.
    pending: list[torch.Tensor] = field(default_factory=list)
    # Buffered prompt latents (prompt_predecode off: decoded jointly with the
    # first generated chunk).
    prompt_latents: torch.Tensor | None = None
    prompt_seen: bool = False
    # Client-stream origin on the decoder's emission timeline: the prompt
    # segment's samples are never emitted.
    stream_start: int = 0
    emitted_samples: int = 0
    # Readiness event of the newest talker chunk (chunks arrive in production
    # order on one talker stream, so the newest event covers every older one).
    codes_ready: Any = None
    pending_codes_ready: Any = None
    # Set by the contract latch for the chunk currently being ingested.
    incoming_prompt: bool = False


@dataclass(frozen=True)
class FireRedCoalescedStep:
    """One equal-progress group step: per-row decoder inputs as un-catted
    chunk lists, aligned with the participants order handed to run_step. The
    cat is a GPU read of the talker's chunk tensors, so it must run on the
    decode stream after the readiness waits — inside run_step, not here."""

    parts: tuple[tuple[torch.Tensor, ...], ...]  # per row: [1, frames, dim] fp32


class FireRedTTS3StreamingVocoder(
    StreamingVocoderBase[FireRedStreamState, FireRedCoalescedStep]
):
    """Incremental RedAE decode over the talker's latent chunk stream."""

    def __init__(
        self,
        decoder: RedAEAudioDecoder,
        *,
        redae_scale: float,
        sample_rate: int,
        max_batch_size: int = 4,
        max_batch_wait_ms: float = 2.0,
        stream_slots: int = 16,
        # Measured on L20 (2026-09): eager predecode LOSES TTFP at c=1 and c=8
        # (230 vs 197 ms; 1397 vs 1148 ms on the shared default stream;
        # retested after the decode-stream split — 115 vs 99 ms at c=1,
        # 316 vs 282 ms at c=8: the joint prompt+first-chunk decode amortizes
        # the launch-bound decoder's fixed cost, so two passes still lose).
        prompt_predecode: bool = False,
        # Equal-progress coalescing, measured on L20 (2026-09): c=1 stays
        # bit-exact (size-1 groups run the serial decode); c=8 stream RTF
        # 0.517 -> 0.355, TTFP 994 -> 398 ms; c=16 gap p99 1668 -> 347 ms.
        stream_batching: bool = True,
        # Run every decode on a dedicated CUDA stream, waiting only on each
        # chunk's readiness event instead of queueing behind the talker's
        # back-to-back graph replays on the legacy default stream. Measured
        # on L20 (2026-09): c=1 TTFP 291 -> 99 ms, c=8 392 -> 282 ms, with
        # c=1 output bit-exact against the shared-stream path.
        separate_decode_stream: bool = True,
        # Replay warm steady-state chunk decodes from self-captured CUDA
        # graphs (one per chunk width x batch bucket). The decode is
        # kernel-launch bound (~16 ms eager vs ~4 ms replayed for a 4-patch
        # chunk on L20); B=1 exact-bucket replays are bit-exact with eager.
        incremental_cuda_graph: bool = True,
        cuda_graph_latent_sizes: list[int] | None = None,
        cuda_graph_batch_sizes: list[int] | None = None,
        # Replay the fresh-cache joint prompt+first-chunk decode from
        # left-padded bucket graphs (vocoder_cold_graph.py). On the TTFP
        # critical path: ~36 ms eager vs ~9 ms replayed on L20 (c=1 TTFP
        # 115 -> 89 ms). Exact-width hits are bit-exact with eager; padded
        # replays differ only inside the trimmed prompt lead plus ~1e-6 fp32
        # GEMM-tiling noise. Batch buckets cap at 2: wider groups are
        # compute-bound and bucket padding loses the launch savings.
        cold_cuda_graph: bool = True,
        cold_cuda_graph_latent_sizes: list[int] | None = None,
        cold_cuda_graph_batch_sizes: list[int] | None = None,
    ) -> None:
        device = next(decoder.parameters()).device
        decode_stream = (
            torch.cuda.Stream(device=device)
            if separate_decode_stream and device.type == "cuda"
            else None
        )
        self.batch = FireRedTTS3BatchVocoder(
            decoder,
            redae_scale=redae_scale,
            sample_rate=sample_rate,
            decode_stream=decode_stream,
        )
        # Admission cap for concurrent stream states; the config layer pins it
        # to the talker's max_running_requests.
        self.stream_slots = int(stream_slots)
        self.prompt_predecode = bool(prompt_predecode)
        self.graph_runner = (
            RedAEVocoderGraphRunner(
                decoder,
                latent_sizes=(
                    tuple(cuda_graph_latent_sizes)
                    if cuda_graph_latent_sizes
                    else GRAPH_DEFAULT_LATENT_SIZES
                ),
                batch_sizes=(
                    tuple(cuda_graph_batch_sizes)
                    if cuda_graph_batch_sizes
                    else GRAPH_DEFAULT_BATCH_SIZES
                ),
            )
            if incremental_cuda_graph and device.type == "cuda"
            else None
        )
        self.can_batch_stream_chunks = bool(stream_batching)
        self.cold_graph_runner = (
            RedAEVocoderColdGraphRunner(
                decoder,
                latent_sizes=(
                    tuple(cold_cuda_graph_latent_sizes)
                    if cold_cuda_graph_latent_sizes
                    else GRAPH_DEFAULT_COLD_LATENT_SIZES
                ),
                batch_sizes=(
                    tuple(cold_cuda_graph_batch_sizes)
                    if cold_cuda_graph_batch_sizes
                    else GRAPH_DEFAULT_COLD_BATCH_SIZES
                ),
            )
            if cold_cuda_graph and device.type == "cuda"
            else None
        )
        if stream_batching:
            self.stream_chunk_batch_distinct_requests = True
            # A coalesced group may span every admitted stream: the decoder
            # is CPU-launch bound, so a wide group costs ~one row.
            self.stream_chunk_batch_max = self.stream_slots
        else:
            pass
        super().__init__(
            self.batch.decode_payload,
            batch_compute_fn=self.batch.decode_payloads,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            sample_rate=sample_rate,
            stream_source_hint="fireredtts3",
            stream_input_modality="audio_latents",
        )

    @property
    def device(self) -> torch.device:
        return next(self.batch.decoder.parameters()).device

    def create_stream_state(self, request_id: str) -> FireRedStreamState:
        del request_id
        return FireRedStreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: FireRedStreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        if origin == "payload":
            return
        else:
            pass
        metadata = source
        state.pending_codes_ready = metadata.get("codes_ready_event")
        if not metadata.get(PROMPT_CHUNK_METADATA_KEY):
            return
        else:
            pass
        if state.prompt_seen:
            raise RuntimeError(
                f"FireRedTTS3 stream for {request_id!r} carried a second "
                "prompt chunk"
            )
        else:
            pass
        state.prompt_seen = True
        left_frames = int(metadata.get(LEFT_CONTEXT_METADATA_KEY) or 0)
        if left_frames <= 0:
            raise ValueError(
                f"FireRedTTS3 prompt chunk for {request_id!r} must carry a "
                "positive left_context_size"
            )
        else:
            pass
        state.stream_start = left_frames * SAMPLES_PER_LATENT_FRAME
        state.incoming_prompt = True

    def validate_chunk(
        self, request_id: str, state: FireRedStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        del request_id
        if codes.ndim != 3 or codes.shape[0] != 1 or codes.shape[1] == 0:
            raise ValueError(
                "FireRedTTS3 latent chunks must have shape [1, frames, redae_dim] "
                f"with frames >= 1, got {tuple(codes.shape)}"
            )
        else:
            pass
        if codes.shape[-1] != self.batch.decoder.in_proj.in_features:
            raise ValueError(
                f"FireRedTTS3 latent width must be "
                f"{self.batch.decoder.in_proj.in_features}, got {codes.shape[-1]}"
            )
        else:
            pass
        if codes.device == self.device and codes.dtype == torch.float32:
            converted = codes
        else:
            # Conversion is a GPU op: run it on the decode stream so it is
            # ordered with the later decode; a CUDA source additionally waits
            # on the chunk's readiness event (latched, not yet ingested).
            with self.batch.decode_stream_context():
                event = state.pending_codes_ready or state.codes_ready
                if event is not None and codes.is_cuda:
                    torch.cuda.current_stream(self.device).wait_event(event)
                else:
                    pass
                converted = codes.to(device=self.device, dtype=torch.float32)
        decode_stream = self.batch.decode_stream
        if decode_stream is not None and converted.is_cuda:
            # Default-stream allocation read on the decode stream: pin it
            # against reuse until the decode stream passes the free point.
            converted.record_stream(decode_stream)
        else:
            pass
        return converted

    def ingest(
        self, request_id: str, state: FireRedStreamState, codes: torch.Tensor
    ) -> None:
        del request_id
        if state.pending_codes_ready is not None:
            state.codes_ready = state.pending_codes_ready
            state.pending_codes_ready = None
        else:
            pass
        if state.incoming_prompt:
            state.incoming_prompt = False
            if self.prompt_predecode:
                self.decode_prompt(state, codes)
            else:
                state.prompt_latents = codes
            return
        else:
            pass
        state.pending.append(codes)

    def decode_prompt(self, state: FireRedStreamState, prompt: torch.Tensor) -> None:
        """Decode the prompt segment at ingest; its waveform stays below the
        client-stream origin and is discarded (KV/OLA warm-up only)."""
        with self.batch.lock, torch.no_grad(), self.batch.decode_stream_context():
            self.wait_codes_ready(state)
            probe = ProfilingProbe.create()
            with probe.segment("prompt_predecode"):
                audio, decoder_state = self.batch.decoder.forward_stream(
                    prompt / self.batch.redae_scale, None, last_chunk=False
                )
            probe.log("prompt-predecode")
        state.decoder_state = decoder_state
        state.emitted_samples += int(audio.shape[-1])

    def should_decode(self, state: FireRedStreamState, *, is_final: bool) -> bool:
        # Decode on generated chunks only: a buffered prompt (predecode off)
        # alone waits for the first generated chunk to decode jointly.
        del is_final
        return bool(state.pending)

    def wait_codes_ready(self, state: FireRedStreamState) -> None:
        """Order this thread's stream after the talker's newest chunk."""
        if state.codes_ready is None:
            return
        else:
            pass
        torch.cuda.current_stream(self.device).wait_event(state.codes_ready)
        state.codes_ready = None

    def emit_window(
        self, state: FireRedStreamState, audio: torch.Tensor
    ) -> torch.Tensor | None:
        """Trim an emission to the client stream (samples at or past the
        prompt boundary); None when the emission is entirely prompt."""
        start = state.stream_start - state.emitted_samples
        state.emitted_samples += int(audio.shape[-1])
        if start <= 0:
            return audio
        else:
            pass
        if start >= int(audio.shape[-1]):
            return None
        else:
            pass
        return audio[:, start:]

    def emit_cpu_window(
        self, state: FireRedStreamState, audio: torch.Tensor
    ) -> torch.Tensor | None:
        """emit_window plus the host readback; the copy must run on the
        stream that produced audio (the caller holds that stream's context),
        otherwise the readback is unordered with the decode."""
        waveform = self.emit_window(state, audio)
        if waveform is None:
            return None
        else:
            pass
        return waveform.cpu()

    def decode_chunk(
        self,
        latents: torch.Tensor,
        decoder_state: RedAEDecoderState | None,
        *,
        last_chunk: bool,
        lead_samples: int = 0,
    ) -> tuple[torch.Tensor, RedAEDecoderState]:
        """One forward_stream, preferring a captured graph for warm shapes.

        Cold (fresh cache), ramp (window not yet saturated), off-width and
        terminal chunks fall back to the eager decode inside the runner's
        eligibility check; the terminal chunk's last_chunk trim has no
        captured shape by design (once per request). The cold graph needs
        lead_samples (the prompt's sample count, trimmed from the emission)
        to guarantee its padded head never leaks into the client stream.
        """
        if not last_chunk:
            runner = self.graph_runner
            if runner is not None and decoder_state is not None:
                result = runner.run(latents, decoder_state)
                if result is not None:
                    return result
                else:
                    pass
            else:
                pass
            cold = self.cold_graph_runner
            if cold is not None and decoder_state is None:
                result = cold.run(latents, lead_samples=lead_samples)
                if result is not None:
                    return result
                else:
                    pass
            else:
                pass
        else:
            pass
        return self.batch.decoder.forward_stream(
            latents, decoder_state, last_chunk=last_chunk
        )

    def decode_delta(
        self, request_id: str, state: FireRedStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        del request_id
        with self.batch.lock, torch.no_grad(), self.batch.decode_stream_context():
            self.wait_codes_ready(state)
            latents = self.collect_step_latents(state)
            if latents is None:
                if (
                    not is_final
                    or state.decoder_state is None
                    or state.decoder_state.istft_cache is None
                ):
                    return None
                else:
                    pass
                # Every ingested chunk was already decoded; the terminal OLA
                # tail remains. It depends only on the cached windowed iSTFT
                # frames.
                tail = self.batch.decoder.flush_stream(state.decoder_state)
                state.decoder_state = None
                return self.emit_cpu_window(state, tail)
            else:
                pass
            probe = ProfilingProbe.create()
            with probe.segment("stream_decode"):
                audio, decoder_state = self.decode_chunk(
                    latents / self.batch.redae_scale,
                    state.decoder_state,
                    last_chunk=is_final,
                    lead_samples=state.stream_start,
                )
            probe.log("stream-decode")
            state.decoder_state = None if is_final else decoder_state
            state.prompt_latents = None
            state.pending.clear()
            return self.emit_cpu_window(state, audio)

    def latent_parts(self, state: FireRedStreamState) -> list[torch.Tensor]:
        """The row's buffered decoder input, un-catted: prompt (buffered,
        predecode off) + pending chunks."""
        parts: list[torch.Tensor] = []
        if state.prompt_latents is not None:
            parts.append(state.prompt_latents)
        else:
            pass
        parts.extend(state.pending)
        return parts

    def collect_step_latents(self, state: FireRedStreamState) -> torch.Tensor | None:
        """Prompt (buffered, predecode off) + pending chunks as one input."""
        parts = self.latent_parts(state)
        if not parts:
            return None
        else:
            pass
        return torch.cat(parts, dim=1)

    def step_group_key(self, state: FireRedStreamState) -> tuple[int, bool, int]:
        """Equal-progress grouping key: frames to decode, cache presence and
        decoder progress (50 Hz frames seen). The SWA cache layers track one
        cumulative length, so only lockstep rows can share a merged cache."""
        frames = sum(int(chunk.shape[1]) for chunk in state.pending)
        if state.prompt_latents is not None:
            frames += int(state.prompt_latents.shape[1])
        else:
            pass
        decoder_state = state.decoder_state
        if decoder_state is None:
            return frames, False, 0
        else:
            pass
        return frames, True, int(decoder_state.frames_seen)

    def select_step_participants(
        self,
    ) -> list[tuple[str, FireRedStreamState]]:
        groups: dict[tuple[int, bool, int], list[tuple[str, FireRedStreamState]]] = {}
        for request_id, state in self.stream_state_items():
            if state.pending:
                groups.setdefault(self.step_group_key(state), []).append(
                    (request_id, state)
                )
            else:
                pass
        if not groups:
            return []
        else:
            pass
        # One group per step; the pump loops until no participants remain, so
        # off-progress rows decode on the following iterations of the same
        # drain. The cap bounds one group to the admitted stream count.
        return max(groups.values(), key=len)[: self.stream_chunk_batch_max]

    def build_step_plan(
        self, participants: list[tuple[str, FireRedStreamState]]
    ) -> FireRedCoalescedStep:
        parts = []
        for _request_id, state in participants:
            row_parts = self.latent_parts(state)
            assert row_parts
            parts.append(tuple(row_parts))
        first_key = self.step_group_key(participants[0][1])
        for _request_id, state in participants[1:]:
            assert self.step_group_key(state) == first_key
        return FireRedCoalescedStep(parts=tuple(parts))

    def run_step(
        self,
        participants: list[tuple[str, FireRedStreamState]],
        plan: FireRedCoalescedStep,
    ) -> dict[str, torch.Tensor]:
        states = [state for _request_id, state in participants]
        probes = [ProfilingProbe.create() for _ in participants]
        with self.batch.lock, torch.no_grad(), self.batch.decode_stream_context():
            for state in states:
                self.wait_codes_ready(state)
            latents = [torch.cat(row_parts, dim=1) for row_parts in plan.parts]
            if len(participants) == 1:
                # Size-1 groups keep the exact serial decode: single-request
                # output stays bit-exact with the non-batched path (the B=1
                # graph replay is bit-exact with eager, so the graph is
                # allowed here).
                state = states[0]
                with probes[0].segment("stream_decode"):
                    audio, decoder_state = self.decode_chunk(
                        latents[0] / self.batch.redae_scale,
                        state.decoder_state,
                        last_chunk=False,
                        lead_samples=state.stream_start,
                    )
                state.decoder_state = decoder_state
                row_audios = [audio]
            else:
                merged = None
                if states[0].decoder_state is not None:
                    merge_inputs: list[RedAEDecoderState] = []
                    for state in states:
                        # The group key pins cache presence across a step's
                        # rows, so a cached first row means every row cached.
                        assert state.decoder_state is not None
                        merge_inputs.append(state.decoder_state)
                    merged = self.batch.decoder.merge_states(merge_inputs)
                else:
                    pass
                with ProfilingProbe.shared_segment("stream_decode", probes):
                    audio, merged_state = self.decode_chunk(
                        torch.cat(latents, dim=0) / self.batch.redae_scale,
                        merged,
                        last_chunk=False,
                        lead_samples=min(s.stream_start for s in states),
                    )
                split_states = self.batch.decoder.split_state(
                    merged_state, len(participants)
                )
                for state, split_row in zip(states, split_states):
                    state.decoder_state = split_row
                row_audios = [audio[row : row + 1] for row in range(len(states))]
            for (request_id, _state), probe in zip(participants, probes):
                probe.log(f"stream-decode {request_id}")
            decoded: dict[str, torch.Tensor] = {}
            for (request_id, state), audio in zip(participants, row_audios):
                state.prompt_latents = None
                state.pending.clear()
                waveform = self.emit_cpu_window(state, audio)
                if waveform is not None:
                    decoded[request_id] = waveform
                else:
                    pass
        return decoded

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: FireRedStreamState
    ) -> dict[str, Any]:
        del state
        tts_state = load_firered_tts3_state(payload)
        self.maybe_dump_stream_latents(request_id, tts_state)
        result: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self.sample_rate,
        }
        if tts_state.gen_text:
            result["text"] = tts_state.gen_text
        else:
            pass
        usage = build_usage(tts_state)
        if usage is not None:
            result["usage"] = usage
        else:
            pass
        return result

    @staticmethod
    def maybe_dump_stream_latents(request_id: str, state: FireRedTTS3State) -> None:
        """Dump the stream's authoritative latent sequence (same layout as the
        batch path's decoder-input dump) for cross-mode oracle diffs."""
        dump_dir = os.environ.get(DEBUG_DUMP_DIR_ENV)
        if not dump_dir:
            return
        else:
            pass
        parts = [
            tensor
            for tensor in (state.prompt_latents, state.generated_latents)
            if tensor is not None and tensor.numel()
        ]
        if not parts:
            return
        else:
            pass
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"latents_stream_{request_id}.pt")
        torch.save(torch.cat(parts, dim=1).detach().cpu(), path)

    def fallback_full_decode(
        self, request_id: str, payload: StagePayload, state: FireRedStreamState
    ) -> torch.Tensor | None:
        del request_id, state
        # The terminal payload's generated_latents are the authoritative
        # record when the stream never emitted (e.g. every chunk lost).
        tts_state, latents = self.batch.prepare_item(payload)
        audio = self.batch.decode_latents(latents)
        trim = int(tts_state.left_context_size) * SAMPLES_PER_LATENT_FRAME
        if trim > 0:
            audio = audio[:, trim:]
        else:
            pass
        return audio

    def release_stream_resources(
        self, request_id: str, state: FireRedStreamState
    ) -> None:
        del request_id
        state.decoder_state = None
        state.pending.clear()
        state.prompt_latents = None

    def warmup_now(self) -> None:
        """Warm both forward_stream paths (fresh cache, then cached terminal)
        so the first served request does not pay the decoder's cold start."""
        with self.batch.lock, torch.no_grad(), self.batch.decode_stream_context():
            dummy = torch.zeros(
                1,
                WARMUP_FRAMES,
                self.batch.decoder.in_proj.in_features,
                device=self.device,
                dtype=torch.float32,
            )
            _, cache = self.batch.decoder.forward_stream(dummy, None, last_chunk=False)
            self.batch.decoder.forward_stream(dummy, cache, last_chunk=True)
            self.batch.decoder.flush_stream(cache)
            if self.graph_runner is not None:
                # Capture inside the decode stream context the replays will
                # run under; capture failures leave eager fallbacks per key.
                self.graph_runner.capture_all()
            else:
                pass
            if self.cold_graph_runner is not None:
                self.cold_graph_runner.capture_all()
            else:
                pass
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        else:
            pass
        logger.info("FireRedTTS3 streaming vocoder warmup complete")


__all__ = [
    "FireRedCoalescedStep",
    "FireRedTTS3BatchVocoder",
    "FireRedTTS3StreamingVocoder",
    "LEFT_CONTEXT_METADATA_KEY",
    "PROMPT_CHUNK_METADATA_KEY",
]
