# SPDX-License-Identifier: Apache-2.0
"""Vocoder stage tests: full-payload decode, prompt-segment trim, batching,
and the incremental streaming lifecycle."""

from __future__ import annotations

import asyncio
import queue

import numpy as np
import pytest
import torch

from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.vocoder import (
    SAMPLES_PER_LATENT_FRAME,
    FireRedTTS3BatchVocoder,
    FireRedTTS3StreamingVocoder,
)
from sglang_omni.models.firered_tts3.weights import load_redae_decoder
from sglang_omni.proto import OmniRequest, StagePayload

from .conftest import CHECKPOINTS, requires_checkpoints

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def decoder() -> object:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from sglang_omni.models.firered_tts3.config import load_checkpoint_hf_config

    config = load_checkpoint_hf_config(str(CHECKPOINTS["base"]))
    assert config is not None
    module = load_redae_decoder(str(CHECKPOINTS["base"]), config, device="cuda")
    yield module
    del module
    torch.cuda.empty_cache()


def make_vocoder(
    decoder, *, separate_decode_stream: bool = False
) -> FireRedTTS3BatchVocoder:
    stream = (
        torch.cuda.Stream(device=torch.device("cuda"))
        if separate_decode_stream
        else None
    )
    return FireRedTTS3BatchVocoder(
        decoder, redae_scale=1.0, sample_rate=24000, decode_stream=stream
    )


def make_streaming_vocoder(
    decoder,
    *,
    prompt_predecode: bool = True,
    stream_batching: bool = False,
    separate_decode_stream: bool = False,
    incremental_cuda_graph: bool = False,
    cold_cuda_graph: bool = False,
    cold_latent_sizes: tuple[int, ...] | None = None,
    cold_batch_sizes: tuple[int, ...] | None = None,
    warmup: bool = False,
):
    vocoder = FireRedTTS3StreamingVocoder(
        decoder,
        redae_scale=1.0,
        sample_rate=24000,
        prompt_predecode=prompt_predecode,
        stream_batching=stream_batching,
        separate_decode_stream=separate_decode_stream,
        incremental_cuda_graph=incremental_cuda_graph,
        cold_cuda_graph=cold_cuda_graph,
        cold_cuda_graph_latent_sizes=(
            list(cold_latent_sizes) if cold_latent_sizes else None
        ),
        cold_cuda_graph_batch_sizes=(
            list(cold_batch_sizes) if cold_batch_sizes else None
        ),
    )
    if warmup:
        vocoder.warmup_now()
    return vocoder


def payload_with(state: FireRedTTS3State) -> StagePayload:
    return StagePayload(
        request_id="rid",
        request=OmniRequest(inputs={"text": "hello"}, params={}),
        data=state.to_dict(),
    )


@pytest.mark.parametrize("finish_reason", ["stop", "length", "abort", None])
@pytest.mark.parametrize("stream", [False, True])
def test_vocoder_preserves_terminal_metadata(
    finish_reason: str | None, stream: bool
) -> None:
    state = FireRedTTS3State(
        finish_reason=finish_reason,
        gen_text="voice plan",
        completion_tokens=3,
    )
    payload = payload_with(state)
    decoder = torch.nn.Linear(1, 1)
    if stream:
        vocoder = make_streaming_vocoder(decoder)
        result = vocoder.final_result_data(
            "rid", payload, vocoder.create_stream_state("rid")
        )
    else:
        vocoder = make_vocoder(decoder)
        result = vocoder.store_result(payload, state, torch.zeros(240), 24000).data
    assert result["finish_reason"] == finish_reason
    assert result["text"] == "voice plan"
    assert result["usage"]["completion_tokens"] == 3


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_decode_latents_matches_direct_full_payload(decoder) -> None:
    vocoder = make_vocoder(decoder)
    generator = torch.Generator().manual_seed(1234)
    latents = torch.randn(1, 20, 64, generator=generator)
    with torch.no_grad():
        direct, _ = decoder.forward_stream(latents.cuda(), None, last_chunk=True)
    decoded = vocoder.decode_latents(latents)
    assert decoded.shape == direct.shape
    assert torch.equal(decoded, direct.cpu())


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_batch_decode_trims_prompt_segment(decoder) -> None:
    """prompt (8 frames) + generated (12 frames) -> trim 8 x 960 samples."""
    vocoder = make_vocoder(decoder)
    generator = torch.Generator().manual_seed(7)
    prompt = torch.randn(1, 8, 64, generator=generator)
    generated = torch.randn(1, 12, 64, generator=generator)
    state = FireRedTTS3State(
        prompt_latents=prompt,
        generated_latents=generated,
        left_context_size=8,
        prompt_tokens=5,
        completion_tokens=3,
    )
    payload = payload_with(state)
    [result] = asyncio.run(vocoder.decode_payloads([payload]))

    audio = np.frombuffer(
        result.data["audio_waveform"],
        dtype=np.dtype(result.data["audio_waveform_dtype"]),
    ).reshape(result.data["audio_waveform_shape"])
    assert audio.shape == (12 * SAMPLES_PER_LATENT_FRAME,)
    assert result.data["sample_rate"] == 24000
    assert result.data["usage"]["completion_tokens"] == 3

    # The trimmed output must equal the untrimmed decode minus its prefix.
    state_no_trim = FireRedTTS3State(
        prompt_latents=prompt,
        generated_latents=generated,
        left_context_size=0,
    )
    [full] = asyncio.run(vocoder.decode_payloads([payload_with(state_no_trim)]))
    audio_full = np.frombuffer(full.data["audio_waveform"], dtype=np.float32)
    assert np.array_equal(audio, audio_full[8 * SAMPLES_PER_LATENT_FRAME :])


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_prepare_item_requires_generated_latents(decoder) -> None:
    vocoder = make_vocoder(decoder)
    with pytest.raises(RuntimeError, match="no generated latents"):
        vocoder.prepare_item(payload_with(FireRedTTS3State()))


def drive_stream(
    vocoder: FireRedTTS3StreamingVocoder,
    prompt: torch.Tensor,
    generated: torch.Tensor,
    *,
    patch_frames: int = 4,
    flush_with_pending: bool,
) -> torch.Tensor:
    """Replay the streaming lifecycle through the public hooks: the prompt
    chunk, ramp-sized generated chunks decoded on arrival, and the terminal
    flush (either with pending latents or from the empty-pending tail)."""
    state = vocoder.create_stream_state("rid")
    left_frames = int(prompt.shape[1])
    vocoder.latch_stream_contract(
        "rid",
        state,
        {"prompt_chunk": True, "left_context_size": left_frames, "stream": True},
        origin="stream metadata",
    )
    vocoder.ingest("rid", state, vocoder.validate_chunk("rid", state, prompt))
    # The prompt alone never triggers an emission on either predecode mode.
    assert not vocoder.should_decode(state, is_final=False)

    total_patches = int(generated.shape[1]) // patch_frames
    # Ramp [1, 2, 4, ...] in patches; when flush_with_pending, the last
    # partial chunk stays undecoded until the stream-done flush.
    ramp = [1, 2] + [4] * (total_patches // 4 + 1)
    cursor = 0
    chunk_count = 0
    emissions: list[torch.Tensor] = []
    pending_at_finish = 0
    while cursor < total_patches:
        take = min(ramp[min(chunk_count, len(ramp) - 1)], total_patches - cursor)
        chunk_count += 1
        is_last = cursor + take == total_patches
        if is_last and flush_with_pending:
            pending_at_finish = take
            break
        chunk = generated[:, cursor * patch_frames : (cursor + take) * patch_frames]
        cursor += take
        vocoder.latch_stream_contract(
            "rid", state, {"stream": True}, origin="stream metadata"
        )
        vocoder.ingest("rid", state, vocoder.validate_chunk("rid", state, chunk))
        if not vocoder.should_decode(state, is_final=False):
            continue
        emission = vocoder.decode_delta("rid", state, is_final=False)
        if emission is not None:
            emissions.append(emission)
    if pending_at_finish:
        chunk = generated[:, cursor * patch_frames :]
        vocoder.latch_stream_contract(
            "rid", state, {"stream": True}, origin="stream metadata"
        )
        vocoder.ingest("rid", state, vocoder.validate_chunk("rid", state, chunk))
    tail = vocoder.decode_delta("rid", state, is_final=True)
    if tail is not None:
        emissions.append(tail)
    return torch.cat(emissions, dim=-1)


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
@pytest.mark.parametrize("prompt_predecode", [True, False])
@pytest.mark.parametrize("flush_with_pending", [True, False])
def test_streaming_matches_batch_decode(
    decoder, prompt_predecode: bool, flush_with_pending: bool
) -> None:
    """The chunked stream (prompt chunk + ramp) equals the batch decode.

    Tolerance is the chunked-vs-whole SWA fp32 noise floor (component oracle
    measured 1.92e-6); the lengths must match exactly.
    """
    generator = torch.Generator().manual_seed(11)
    prompt = torch.randn(1, 8, 64, generator=generator)
    # 7 patches: with flush_with_pending the terminal decode has latents to
    # flush; without it the tail comes from the empty-pending OLA flush.
    generated = torch.randn(1, 7 * 4, 64, generator=generator)

    batch_vocoder = make_vocoder(decoder)
    reference = batch_vocoder.decode_latents(torch.cat([prompt, generated], dim=1))
    reference = reference[:, 8 * SAMPLES_PER_LATENT_FRAME :]

    vocoder = make_streaming_vocoder(decoder, prompt_predecode=prompt_predecode)
    streamed = drive_stream(
        vocoder,
        prompt,
        generated,
        flush_with_pending=flush_with_pending,
    )
    assert streamed.shape == reference.shape
    diff = (streamed - reference).abs().max().item()
    assert diff <= 1e-5, diff
    corr = np.corrcoef(
        np.abs(np.fft.rfft(streamed[0].cpu().numpy())),
        np.abs(np.fft.rfft(reference[0].cpu().numpy())),
    )[0, 1]
    assert corr >= 0.999999


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_streaming_rejects_duplicate_or_unmarked_prompt(decoder) -> None:
    vocoder = make_streaming_vocoder(decoder)
    generator = torch.Generator().manual_seed(3)
    prompt = torch.randn(1, 8, 64, generator=generator)
    state = vocoder.create_stream_state("rid")
    metadata = {"prompt_chunk": True, "left_context_size": 8, "stream": True}
    vocoder.latch_stream_contract("rid", state, metadata, origin="stream metadata")
    with pytest.raises(RuntimeError, match="second"):
        vocoder.latch_stream_contract("rid", state, metadata, origin="stream metadata")

    fresh = vocoder.create_stream_state("rid2")
    with pytest.raises(ValueError, match="left_context_size"):
        vocoder.latch_stream_contract(
            "rid2",
            fresh,
            {"prompt_chunk": True, "left_context_size": 0, "stream": True},
            origin="stream metadata",
        )


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_streaming_release_drops_decoder_state(decoder) -> None:
    vocoder = make_streaming_vocoder(decoder)
    generator = torch.Generator().manual_seed(5)
    prompt = torch.randn(1, 8, 64, generator=generator)
    state = vocoder.create_stream_state("rid")
    vocoder.latch_stream_contract(
        "rid",
        state,
        {"prompt_chunk": True, "left_context_size": 8, "stream": True},
        origin="stream metadata",
    )
    vocoder.ingest("rid", state, vocoder.validate_chunk("rid", state, prompt))
    assert state.decoder_state is not None
    vocoder.release_stream_resources("rid", state)
    assert state.decoder_state is None
    assert state.pending == []


# ── Equal-progress coalesced stepping (stream_batching) ──


def register_state(
    vocoder: FireRedTTS3StreamingVocoder,
    request_id: str,
    *,
    chunk_frames: int | None,
    decoder,
    cached: bool = False,
) -> None:
    state = vocoder.create_stream_state(request_id)
    if cached:
        with torch.no_grad():
            _audio, state.decoder_state = decoder.forward_stream(
                torch.zeros(1, 16, 64, device=vocoder.device), None, last_chunk=False
            )
    if chunk_frames is not None:
        state.pending.append(torch.zeros(1, chunk_frames, 64, device=vocoder.device))
    vocoder.stream_states[request_id] = state


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_select_step_participants_groups_equal_progress(decoder) -> None:
    vocoder = make_streaming_vocoder(decoder, stream_batching=True)
    register_state(vocoder, "same_a", chunk_frames=16, decoder=decoder)
    register_state(vocoder, "same_b", chunk_frames=16, decoder=decoder)
    register_state(vocoder, "other_frames", chunk_frames=8, decoder=decoder)
    register_state(
        vocoder, "other_progress", chunk_frames=16, decoder=decoder, cached=True
    )
    register_state(vocoder, "no_pending", chunk_frames=None, decoder=decoder)

    selected = vocoder.select_step_participants()
    assert sorted(request_id for request_id, _state in selected) == [
        "same_a",
        "same_b",
    ]

    # Off-progress rows become eligible once the lockstep group drains.
    for request_id, state in selected:
        state.pending.clear()
    selected = vocoder.select_step_participants()
    assert [request_id for request_id, _state in selected] == ["other_frames"]


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_select_step_participants_caps_at_stream_slots(decoder) -> None:
    vocoder = FireRedTTS3StreamingVocoder(
        decoder,
        redae_scale=1.0,
        sample_rate=24000,
        stream_slots=2,
        stream_batching=True,
    )
    for index in range(3):
        register_state(vocoder, f"row{index}", chunk_frames=16, decoder=decoder)
    selected = vocoder.select_step_participants()
    assert len(selected) == 2


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_build_step_plan_rejects_mixed_progress(decoder) -> None:
    vocoder = make_streaming_vocoder(decoder, stream_batching=True)
    register_state(vocoder, "fresh", chunk_frames=16, decoder=decoder)
    register_state(vocoder, "cached", chunk_frames=16, decoder=decoder, cached=True)
    participants = [
        ("fresh", vocoder.stream_states["fresh"]),
        ("cached", vocoder.stream_states["cached"]),
    ]
    with pytest.raises(AssertionError):
        vocoder.build_step_plan(participants)


def drain_stream_emissions(
    vocoder: FireRedTTS3StreamingVocoder,
) -> dict[str, list[torch.Tensor]]:
    emissions: dict[str, list[torch.Tensor]] = {}
    while True:
        try:
            message = vocoder.outbox.get_nowait()
        except queue.Empty:
            return emissions
        assert message.type == "stream"
        assert message.metadata == {"modality": "audio"}
        payload = message.data
        assert payload["audio_waveform_dtype"] == "float32"
        waveform = np.frombuffer(payload["audio_waveform"], dtype=np.float32)
        emissions.setdefault(message.request_id, []).append(
            torch.from_numpy(waveform.copy())
        )


def drive_coalesced(
    vocoder: FireRedTTS3StreamingVocoder,
    prompts: dict[str, torch.Tensor],
    generated: dict[str, torch.Tensor],
    *,
    patch_frames: int = 4,
) -> dict[str, torch.Tensor]:
    """Replay lockstep streams through the coalesced pump: prompt chunks,
    ramp-sized generated chunks (one pump per round), then the terminal OLA
    flush per request."""
    for request_id, prompt in prompts.items():
        state = vocoder.get_or_create_stream_state(request_id)
        vocoder.latch_stream_contract(
            request_id,
            state,
            {
                "prompt_chunk": True,
                "left_context_size": int(prompt.shape[1]),
                "stream": True,
            },
            origin="stream metadata",
        )
        vocoder.ingest(
            request_id, state, vocoder.validate_chunk(request_id, state, prompt)
        )
    # Buffered prompts alone never join a step (predecode off).
    assert vocoder.select_step_participants() == []

    lengths = {int(gen.shape[1]) for gen in generated.values()}
    assert len(lengths) == 1
    total_patches = lengths.pop() // patch_frames
    ramp = [1, 2] + [4] * (total_patches // 4 + 1)
    emissions: dict[str, list[torch.Tensor]] = {rid: [] for rid in prompts}
    cursor = 0
    chunk_count = 0
    while cursor < total_patches:
        take = min(ramp[min(chunk_count, len(ramp) - 1)], total_patches - cursor)
        chunk_count += 1
        for request_id, gen in generated.items():
            state = vocoder.stream_states[request_id]
            chunk = gen[:, cursor * patch_frames : (cursor + take) * patch_frames]
            vocoder.latch_stream_contract(
                request_id, state, {"stream": True}, origin="stream metadata"
            )
            vocoder.ingest(
                request_id, state, vocoder.validate_chunk(request_id, state, chunk)
            )
        cursor += take
        vocoder.pump_streams()
        for request_id, pieces in drain_stream_emissions(vocoder).items():
            emissions[request_id].extend(pieces)
        assert all(not vocoder.stream_states[rid].pending for rid in prompts)

    for request_id in prompts:
        state = vocoder.stream_states[request_id]
        tail = vocoder.decode_delta(request_id, state, is_final=True)
        if tail is not None:
            emissions[request_id].append(tail.reshape(-1).cpu())
    return {request_id: torch.cat(pieces) for request_id, pieces in emissions.items()}


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_coalesced_pump_matches_serial_streams(decoder) -> None:
    """Two lockstep streams through the batched pump vs the serial path.

    The batched GEMM tiling drift stays bounded (the merge/split round-trip
    bound, 5e-3) and never accumulates (the SWA window forgets stale KV
    within two chunks); lengths must match exactly.
    """
    prompts = {}
    generated = {}
    for index, request_id in enumerate(("row_a", "row_b")):
        generator = torch.Generator().manual_seed(100 + index)
        prompts[request_id] = torch.randn(1, 8, 64, generator=generator)
        generated[request_id] = torch.randn(1, 7 * 4, 64, generator=generator)

    batched_vocoder = make_streaming_vocoder(
        decoder, prompt_predecode=False, stream_batching=True
    )
    coalesced = drive_coalesced(batched_vocoder, prompts, generated)

    for request_id in prompts:
        serial_vocoder = make_streaming_vocoder(
            decoder, prompt_predecode=False, stream_batching=False
        )
        serial = (
            drive_stream(
                serial_vocoder,
                prompts[request_id],
                generated[request_id],
                flush_with_pending=False,
            )
            .reshape(-1)
            .cpu()
        )
        got = coalesced[request_id]
        assert got.shape == serial.shape
        max_diff = (got - serial).abs().max().item()
        assert max_diff <= 5e-3, f"{request_id} max diff {max_diff:.3e}"


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_coalesced_single_row_bit_exact(decoder) -> None:
    """A size-1 group runs the exact serial decode (B=1 stays bit-exact)."""
    generator = torch.Generator().manual_seed(11)
    prompt = torch.randn(1, 8, 64, generator=generator)
    generated = torch.randn(1, 7 * 4, 64, generator=generator)

    batched_vocoder = make_streaming_vocoder(
        decoder, prompt_predecode=False, stream_batching=True
    )
    coalesced = drive_coalesced(batched_vocoder, {"rid": prompt}, {"rid": generated})

    serial_vocoder = make_streaming_vocoder(
        decoder, prompt_predecode=False, stream_batching=False
    )
    serial = (
        drive_stream(serial_vocoder, prompt, generated, flush_with_pending=False)
        .reshape(-1)
        .cpu()
    )
    assert coalesced["rid"].shape == serial.shape
    assert torch.equal(coalesced["rid"], serial)


# ── Dedicated decode stream (separate_decode_stream) ──


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_decode_stream_bit_exact(decoder) -> None:
    """The decode stream changes scheduling, not numerics: streamed and
    full-payload decodes on a dedicated stream are bit-identical to the
    legacy default-stream path."""
    generator = torch.Generator().manual_seed(17)
    prompt = torch.randn(1, 8, 64, generator=generator)
    generated = torch.randn(1, 7 * 4, 64, generator=generator)
    joint = torch.cat([prompt, generated], dim=1)

    batch_default = make_vocoder(decoder).decode_latents(joint)
    batch_streamed = make_vocoder(decoder, separate_decode_stream=True)
    batch_streamed = batch_streamed.decode_latents(joint)
    assert torch.equal(batch_default, batch_streamed)

    streamed = {}
    for knob in (False, True):
        vocoder = make_streaming_vocoder(
            decoder, prompt_predecode=False, separate_decode_stream=knob
        )
        streamed[knob] = drive_stream(
            vocoder, prompt, generated, flush_with_pending=True
        )
    assert torch.equal(streamed[False], streamed[True])


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_decode_stream_waits_for_chunk_event(decoder) -> None:
    """The decode stream must order after the chunk's readiness event.

    The chunk lands on the default stream only after a slow production
    chain; the event records at its end. A decode that ignored the event
    would read the chunk before the copy and diverge from the reference
    decode of the finished values.
    """
    generator = torch.Generator().manual_seed(23)
    chunk_values = torch.randn(1, 16, 64, generator=generator)
    # Slow default-stream production chain; the chunk copy lands at its end.
    scratch = torch.randn(2048, 2048, device="cuda")
    with torch.no_grad():
        for _ in range(100):
            scratch = torch.relu(scratch @ scratch * 1e-4)
        chunk = chunk_values.cuda()
        event = torch.cuda.Event()
        event.record()

    def decode_with(knob: bool) -> torch.Tensor:
        vocoder = make_streaming_vocoder(
            decoder, prompt_predecode=False, separate_decode_stream=knob
        )
        state = vocoder.create_stream_state("rid")
        metadata: dict = {"stream": True}
        if knob:
            metadata["codes_ready_event"] = event
        vocoder.latch_stream_contract("rid", state, metadata, origin="stream metadata")
        vocoder.ingest("rid", state, vocoder.validate_chunk("rid", state, chunk))
        # No synchronize here: the decode is enqueued while the production
        # chain is still in flight, so the event wait is the only ordering.
        return vocoder.decode_delta("rid", state, is_final=True)

    emission = decode_with(True)
    torch.cuda.synchronize()
    # The reference decodes the finished chunk on the default stream (FIFO
    # ordered after the production chain).
    reference = decode_with(False)
    assert torch.equal(emission, reference)


# ── Incremental CUDA graphs (warm steady-state chunk replays) ──


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_vocoder_graph_runner_bit_exact(decoder) -> None:
    """Graph replays equal the eager warm step bit for bit at B=1: audio,
    stored KV, and iSTFT cache, chained over several chunks per width."""
    from sglang_omni.models.firered_tts3.redae_decoder import RedAEDecoderState
    from sglang_omni.models.firered_tts3.vocoder_graph import RedAEVocoderGraphRunner

    runner = RedAEVocoderGraphRunner(decoder)
    runner.capture_all()
    assert len(runner.graphs) == len(runner.latent_sizes) * len(runner.batch_sizes)
    assert not runner.failed_keys

    def clone_state(state: RedAEDecoderState) -> RedAEDecoderState:
        return decoder.merge_states([state])  # deep-enough copy via cat
        # (merge of one row materializes fresh contiguous tensors)

    warm = torch.randn(1, 40, 64, device="cuda", dtype=torch.float32)
    _, saturated = decoder.forward_stream(warm, None, last_chunk=False)
    for latents_per_chunk in (4, 8, 16):
        reference = clone_state(saturated)
        graphed = clone_state(saturated)
        for _ in range(3):
            chunk = torch.randn(
                1, latents_per_chunk, 64, device="cuda", dtype=torch.float32
            )
            audio_ref, reference = decoder.forward_stream(
                chunk, reference, last_chunk=False
            )
            result = runner.run(chunk, graphed)
            assert result is not None
            audio_graph, graphed = result
            assert torch.equal(audio_graph, audio_ref)
            for layer_g, layer_r in zip(
                graphed.qwen3_cache.layers, reference.qwen3_cache.layers
            ):
                assert torch.equal(layer_g.keys, layer_r.keys)
                assert torch.equal(layer_g.values, layer_r.values)
                assert layer_g.cumulative_length == layer_r.cumulative_length
            assert torch.equal(graphed.istft_cache, reference.istft_cache)
            assert graphed.frames_seen == reference.frames_seen


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_vocoder_graph_serial_stream_bit_exact(decoder) -> None:
    """End-to-end stream lifecycle with graphs on: the serial path stays
    bit-exact with the eager vocoder (ramp chunks run eager, warm chunks
    replay, the terminal chunk stays eager)."""
    generator = torch.Generator().manual_seed(29)
    prompt = torch.randn(1, 8, 64, generator=generator)
    generated = torch.randn(1, 16 * 4, 64, generator=generator)

    eager = drive_stream(
        make_streaming_vocoder(decoder, prompt_predecode=False),
        prompt,
        generated,
        flush_with_pending=True,
    )
    graphed_vocoder = make_streaming_vocoder(
        decoder, prompt_predecode=False, incremental_cuda_graph=True, warmup=True
    )
    graphed = drive_stream(graphed_vocoder, prompt, generated, flush_with_pending=True)
    assert graphed.shape == eager.shape
    assert torch.equal(graphed, eager)
    # The run actually exercised the graphs (prompt 8 + ramp keeps the first
    # three chunks below the saturated window; the rest replay).
    stats = graphed_vocoder.graph_runner.stats()
    assert stats["replays"] > 0


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_vocoder_graph_coalesced_exact_bucket(decoder) -> None:
    """Two lockstep streams (exact B=2 bucket): the graphed pump matches the
    eager-merged pump bit for bit."""
    prompts = {}
    generated = {}
    for index, request_id in enumerate(("row_a", "row_b")):
        generator = torch.Generator().manual_seed(200 + index)
        prompts[request_id] = torch.randn(1, 8, 64, generator=generator)
        generated[request_id] = torch.randn(1, 12 * 4, 64, generator=generator)

    eager = drive_coalesced(
        make_streaming_vocoder(decoder, prompt_predecode=False, stream_batching=True),
        prompts,
        generated,
    )
    graphed = drive_coalesced(
        make_streaming_vocoder(
            decoder,
            prompt_predecode=False,
            stream_batching=True,
            incremental_cuda_graph=True,
            warmup=True,
        ),
        prompts,
        generated,
    )
    for request_id in prompts:
        assert graphed[request_id].shape == eager[request_id].shape
        assert torch.equal(graphed[request_id], eager[request_id])


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_vocoder_graph_uncaptured_width_falls_back(decoder) -> None:
    """A chunk width outside the captured grid runs eager and stays exact."""
    from sglang_omni.models.firered_tts3.vocoder_graph import RedAEVocoderGraphRunner

    generator = torch.Generator().manual_seed(31)
    # A 40-latent prompt saturates the SWA window in the joint first decode,
    # so the 8-latent ramp chunk replays and the 16-latent chunks run eager.
    prompt = torch.randn(1, 40, 64, generator=generator)
    generated = torch.randn(1, 16 * 4, 64, generator=generator)

    vocoder = make_streaming_vocoder(decoder, prompt_predecode=False)
    vocoder.graph_runner = RedAEVocoderGraphRunner(decoder, latent_sizes=(8,))
    vocoder.graph_runner.capture_all()
    graphed = drive_stream(vocoder, prompt, generated, flush_with_pending=True)

    eager = drive_stream(
        make_streaming_vocoder(decoder, prompt_predecode=False),
        prompt,
        generated,
        flush_with_pending=True,
    )
    assert torch.equal(graphed, eager)
    stats = vocoder.graph_runner.stats()
    assert stats["replays"] > 0 and stats["eager"] > 0


# ── Cold CUDA graphs (fresh-cache joint prompt+first-chunk decode) ──


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_vocoder_cold_graph_runner_exact_and_padded(decoder) -> None:
    """Cold replays: exact-width buckets are bit-exact with the eager cold
    pass (audio, stored KV, iSTFT cache, frame bookkeeping); padded replays
    match past the prompt lead to fp32 tiling noise and hand off to a warm
    eager chunk identically."""
    from sglang_omni.models.firered_tts3.vocoder_cold_graph import (
        RedAEVocoderColdGraphRunner,
    )

    runner = RedAEVocoderColdGraphRunner(
        decoder, latent_sizes=(32, 64), batch_sizes=(1,)
    )
    runner.capture_all()
    assert len(runner.graphs) == 2 and not runner.failed_keys
    lead = runner.min_lead_samples
    # The production prompt floor is 4 latents (3840 samples), far above it.
    assert lead == 1440

    for n_latents in (64, 53, 32):
        generator = torch.Generator().manual_seed(90 + n_latents)
        latents = torch.randn(1, n_latents, 64, generator=generator).cuda()
        with torch.no_grad():
            audio_ref, state_ref = decoder.forward_stream(
                latents, None, last_chunk=False
            )
        result = runner.run(latents, lead_samples=lead)
        assert result is not None
        audio_graph, state_graph = result
        assert audio_graph.shape == audio_ref.shape
        exact = n_latents in runner.latent_sizes
        if exact:
            assert torch.equal(audio_graph, audio_ref)
        else:
            torch.testing.assert_close(
                audio_graph[:, lead:], audio_ref[:, lead:], atol=1e-5, rtol=1e-4
            )
        for layer_g, layer_r in zip(
            state_graph.qwen3_cache.layers, state_ref.qwen3_cache.layers
        ):
            assert layer_g.keys.shape == layer_r.keys.shape
            assert layer_g.cumulative_length == layer_r.cumulative_length
            if exact:
                assert torch.equal(layer_g.keys, layer_r.keys)
                assert torch.equal(layer_g.values, layer_r.values)
        assert state_graph.frames_seen == state_ref.frames_seen
        # Continuation: one eager warm chunk on top of each state agrees.
        nxt = torch.randn(1, 8, 64, generator=generator).cuda()
        with torch.no_grad():
            audio_next_ref, _ = decoder.forward_stream(nxt, state_ref, last_chunk=False)
            audio_next_graph, _ = decoder.forward_stream(
                nxt, state_graph, last_chunk=False
            )
        if exact:
            assert torch.equal(audio_next_graph, audio_next_ref)
        else:
            torch.testing.assert_close(
                audio_next_graph, audio_next_ref, atol=1e-5, rtol=1e-4
            )

    # Gate rejections: too few latents, too little lead, too many rows.
    small = torch.randn(1, 16, 64, device="cuda")
    assert runner.run(small, lead_samples=lead) is None
    wide = torch.randn(1, 53, 64, device="cuda")
    assert runner.run(wide, lead_samples=lead - 1) is None
    assert runner.run(torch.randn(3, 53, 64, device="cuda"), lead_samples=lead) is None


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_vocoder_cold_graph_stream_matches_eager(decoder) -> None:
    """Stream lifecycle with the cold graph on: the joint prompt+first-chunk
    decode replays, the client-visible emission (prompt region trimmed)
    matches the eager vocoder within the fp32 tiling floor."""
    generator = torch.Generator().manual_seed(41)
    prompt = torch.randn(1, 49, 64, generator=generator)
    generated = torch.randn(1, 16 * 4, 64, generator=generator)

    eager = drive_stream(
        make_streaming_vocoder(decoder, prompt_predecode=False),
        prompt,
        generated,
        flush_with_pending=True,
    )
    graphed_vocoder = make_streaming_vocoder(
        decoder,
        prompt_predecode=False,
        cold_cuda_graph=True,
        cold_latent_sizes=(64,),
        cold_batch_sizes=(1,),
        warmup=True,
    )
    graphed = drive_stream(graphed_vocoder, prompt, generated, flush_with_pending=True)
    assert graphed.shape == eager.shape
    torch.testing.assert_close(graphed, eager, atol=1e-4, rtol=1e-3)
    stats = graphed_vocoder.cold_graph_runner.stats()
    assert stats["replays"] == 1


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_vocoder_cold_graph_short_prompt_stays_eager(decoder) -> None:
    """A prompt below the cold-graph floor keeps the eager decode, so the
    stream stays bit-exact with graphs off."""
    generator = torch.Generator().manual_seed(43)
    prompt = torch.randn(1, 8, 64, generator=generator)
    generated = torch.randn(1, 8 * 4, 64, generator=generator)

    eager = drive_stream(
        make_streaming_vocoder(decoder, prompt_predecode=False),
        prompt,
        generated,
        flush_with_pending=True,
    )
    graphed_vocoder = make_streaming_vocoder(
        decoder,
        prompt_predecode=False,
        cold_cuda_graph=True,
        cold_latent_sizes=(64,),
        cold_batch_sizes=(1,),
        warmup=True,
    )
    graphed = drive_stream(graphed_vocoder, prompt, generated, flush_with_pending=True)
    assert torch.equal(graphed, eager)
    stats = graphed_vocoder.cold_graph_runner.stats()
    assert stats["replays"] == 0 and stats["eager"] > 0


@requires_checkpoints
@requires_gpu
@pytest.mark.accelerator
def test_vocoder_cold_graph_coalesced_matches_eager(decoder) -> None:
    """Two lockstep cold streams batch into one padded replay; per-row
    emissions match the eager coalesced pump within the fp32 floor."""
    prompts = {}
    generated = {}
    for index, request_id in enumerate(("row_a", "row_b")):
        generator = torch.Generator().manual_seed(300 + index)
        prompts[request_id] = torch.randn(1, 49, 64, generator=generator)
        generated[request_id] = torch.randn(1, 8 * 4, 64, generator=generator)

    eager = drive_coalesced(
        make_streaming_vocoder(decoder, prompt_predecode=False, stream_batching=True),
        prompts,
        generated,
    )
    graphed_vocoder = make_streaming_vocoder(
        decoder,
        prompt_predecode=False,
        stream_batching=True,
        cold_cuda_graph=True,
        cold_latent_sizes=(64,),
        cold_batch_sizes=(2,),
        warmup=True,
    )
    graphed = drive_coalesced(graphed_vocoder, prompts, generated)
    for request_id in prompts:
        assert graphed[request_id].shape == eager[request_id].shape
        torch.testing.assert_close(
            graphed[request_id], eager[request_id], atol=1e-4, rtol=1e-3
        )
    stats = graphed_vocoder.cold_graph_runner.stats()
    assert stats["replays"] > 0
