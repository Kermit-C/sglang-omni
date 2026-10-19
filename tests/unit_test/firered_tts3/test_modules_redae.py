# SPDX-License-Identifier: Apache-2.0
"""Numerical oracle tests: ported RedAE encoder/decoder + iSTFT vs reference.

The encoder comparison runs under bf16 autocast (the reference encode
setting); the reference pins flash_attention_2 in its Qwen3 configs, which is
unavailable here, so the oracle is monkeypatched onto the default attention
backend — both sides then run identical kernels and must be bit-exact.

The decoder comparison is fp32 (iSTFT/phase is a precision red line):
forward_stream over the whole latent sequence in one call (the full-payload
special case) must match the reference's full-segment decode.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.firered_tts3.config import load_checkpoint_hf_config
from sglang_omni.models.firered_tts3.istft import ISTFT, ISTFTHead
from sglang_omni.models.firered_tts3.redae_decoder import (
    RedAEDecoderState,
    rematerialize_istft_window,
)
from sglang_omni.models.firered_tts3.weights import (
    build_redae_decoder,
    load_redae_decoder,
    load_redae_encoder,
    read_component_tensors,
)

from .conftest import CHECKPOINTS, REFERENCE_FIREREDTTS3, synth_audio

BASE_CHECKPOINT = CHECKPOINTS["base"]
REFERENCE_SRC = REFERENCE_FIREREDTTS3

requires_all = pytest.mark.skipif(
    not BASE_CHECKPOINT.is_dir() or not REFERENCE_SRC.is_dir(),
    reason="FireRedTTS3 checkpoint or reference sources not present",
)


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


def pad_to_multiple_of(audio: torch.Tensor, multiple_of: int) -> torch.Tensor:
    target = -(-audio.shape[-1] // multiple_of) * multiple_of
    pad_len = target - audio.shape[-1]
    if pad_len > 0:
        audio = F.pad(audio, (pad_len, 0))  # left pad
    return audio


def load_reference_encoder(monkeypatch, device):
    """Instantiate the reference encoder with its FA2 pin removed."""
    import fireredtts3.redae.redae as reference_redae

    real_config = reference_redae.Qwen3Config

    def config_without_fa2(**kwargs):
        kwargs.pop("attn_implementation", None)
        return real_config(**kwargs)

    monkeypatch.setattr(reference_redae, "Qwen3Config", config_without_fa2)
    oracle = reference_redae.RedAEAudioEncoder(out_dim=64, num_hidden_layers=18)
    state = read_component_tensors(BASE_CHECKPOINT / "redae", "encoder.")
    oracle.load_state_dict(state, strict=True)
    return oracle.to(device=device, dtype=torch.float32).eval()


def load_reference_decoder(device):
    import fireredtts3.redae.redae as reference_redae

    oracle = reference_redae.RedAEAudioDecoder()
    state = read_component_tensors(BASE_CHECKPOINT / "redae", "decoder.")
    oracle.load_state_dict(state, strict=True)
    return oracle.to(device=device, dtype=torch.float32).eval()


@requires_all
@pytest.mark.accelerator
def test_redae_encoder_matches_reference_bf16(
    device, reference_sources, monkeypatch
) -> None:
    config = load_checkpoint_hf_config(str(BASE_CHECKPOINT))
    assert config is not None
    ours = load_redae_encoder(str(BASE_CHECKPOINT), config, device=device)
    oracle = load_reference_encoder(monkeypatch, device)

    audio = synth_audio(24000, 2.0).to(device)
    audio = pad_to_multiple_of(audio, ours.downsample_rate)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        latents_ours = ours(audio)
        latents_oracle = oracle(audio)

    latents_ours = latents_ours.float()
    latents_oracle = latents_oracle.float()
    assert latents_ours.shape == (1, 50, 64)
    max_diff = (latents_ours - latents_oracle).abs().max().item()
    assert torch.equal(latents_ours, latents_oracle), f"max diff {max_diff:.3e}"
    del ours, oracle
    torch.cuda.empty_cache()


@requires_all
@pytest.mark.accelerator
def test_redae_decoder_forward_matches_reference(device, reference_sources) -> None:
    config = load_checkpoint_hf_config(str(BASE_CHECKPOINT))
    assert config is not None
    ours = load_redae_decoder(str(BASE_CHECKPOINT), config, device=device)
    oracle = load_reference_decoder(device)

    generator = torch.Generator().manual_seed(1234)
    latents = (torch.randn(1, 20, 64, generator=generator) * 2.0).to(device)
    with torch.no_grad():
        audio_ours = ours.forward(latents)
        audio_oracle = oracle.forward(latents)

    max_diff = (audio_ours - audio_oracle).abs().max().item()
    assert torch.equal(audio_ours, audio_oracle), f"max diff {max_diff:.3e}"
    del ours, oracle
    torch.cuda.empty_cache()


@requires_all
@pytest.mark.accelerator
def test_redae_decoder_forward_stream_whole_matches_reference(
    device, reference_sources
) -> None:
    """The full-payload special case: one forward_stream call, cache=None."""
    config = load_checkpoint_hf_config(str(BASE_CHECKPOINT))
    assert config is not None
    ours = load_redae_decoder(str(BASE_CHECKPOINT), config, device=device)
    oracle = load_reference_decoder(device)

    generator = torch.Generator().manual_seed(1234)
    latents = (torch.randn(1, 20, 64, generator=generator) * 2.0).to(device)
    with torch.no_grad():
        audio_stream, cache = ours.forward_stream(latents, None, last_chunk=True)
        audio_oracle = oracle.forward(latents)

    assert audio_stream.shape == audio_oracle.shape
    assert cache.frames_seen == 40  # 20 latents upsampled 2x
    max_diff = (audio_stream - audio_oracle).abs().max().item()
    assert max_diff <= 1e-6, f"max diff {max_diff:.3e}"
    del ours, oracle
    torch.cuda.empty_cache()


@requires_all
@pytest.mark.accelerator
def test_redae_decoder_chunked_streaming_tracks_whole(
    device, reference_sources
) -> None:
    """Chunked decode vs the single-call stream (SWA increment vs full mask).

    fp32 attention over a KV cache differs from the single full-segment mask
    at the 1e-4 level; the reference stream-verification tool documents this
    as the expected window, so the bound is 1e-3 with the measured value
    reported.
    """
    config = load_checkpoint_hf_config(str(BASE_CHECKPOINT))
    assert config is not None
    decoder = load_redae_decoder(str(BASE_CHECKPOINT), config, device=device)

    generator = torch.Generator().manual_seed(1234)
    latents = (torch.randn(1, 20, 64, generator=generator) * 2.0).to(device)
    with torch.no_grad():
        audio_whole, _ = decoder.forward_stream(latents, None, last_chunk=True)
        chunks = []
        cache = None
        pieces = torch.split(latents, 4, dim=1)
        for index, piece in enumerate(pieces):
            audio_chunk, cache = decoder.forward_stream(
                piece, cache, last_chunk=index == len(pieces) - 1
            )
            chunks.append(audio_chunk)
        audio_chunked = torch.cat(chunks, dim=1)

    assert audio_chunked.shape == audio_whole.shape
    max_diff = (audio_chunked - audio_whole).abs().max().item()
    assert max_diff <= 1e-3, f"max diff {max_diff:.3e}"
    del decoder
    torch.cuda.empty_cache()


@requires_all
@pytest.mark.accelerator
def test_redae_decoder_merge_split_round_trip(device, reference_sources) -> None:
    """Batched decode of two equal-progress states vs per-request serial."""
    config = load_checkpoint_hf_config(str(BASE_CHECKPOINT))
    assert config is not None
    decoder = load_redae_decoder(str(BASE_CHECKPOINT), config, device=device)

    generator = torch.Generator().manual_seed(1234)
    latents = (torch.randn(2, 16, 64, generator=generator) * 2.0).to(device)
    spans = [slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 16)]
    with torch.no_grad():
        # Serial per-request reference decode.
        serial_audio = []
        for row in range(2):
            cache = None
            pieces = []
            for index, span in enumerate(spans):
                audio, cache = decoder.forward_stream(
                    latents[row : row + 1, span],
                    cache,
                    last_chunk=index == len(spans) - 1,
                )
                pieces.append(audio)
            serial_audio.append(torch.cat(pieces, dim=1))
        # Chunk 0 per request, then merge -> batched chunk 1 -> split ->
        # per-request chunks 2-3.
        batched_audio = [[], []]
        states = []
        for row in range(2):
            audio, state = decoder.forward_stream(
                latents[row : row + 1, spans[0]], None, last_chunk=False
            )
            batched_audio[row].append(audio)
            states.append(state)
        merged = decoder.merge_states(states)
        audio_mid, merged = decoder.forward_stream(
            latents[:, spans[1]], merged, last_chunk=False
        )
        states = decoder.split_state(merged, 2)
        for row in range(2):
            batched_audio[row].append(audio_mid[row : row + 1])
            for index, span in enumerate(spans[2:], start=2):
                audio, states[row] = decoder.forward_stream(
                    latents[row : row + 1, span],
                    states[row],
                    last_chunk=index == len(spans) - 1,
                )
                batched_audio[row].append(audio)
        for row in range(2):
            got = torch.cat(batched_audio[row], dim=1)
            want = serial_audio[row]
            assert got.shape == want.shape
            max_diff = (got - want).abs().max().item()
            assert max_diff <= 5e-3, f"row {row} max diff {max_diff:.3e}"
    del decoder
    torch.cuda.empty_cache()


@requires_all
@pytest.mark.accelerator
def test_istft_matches_reference(device, reference_sources) -> None:
    from fireredtts3.redae.redae import ISTFT as ReferenceISTFT

    n_fft, hop = 1920, 480
    ours = ISTFT(n_fft=n_fft, hop_length=hop, win_length=n_fft).to(device)
    oracle = ReferenceISTFT(n_fft=n_fft, hop_length=hop, win_length=n_fft).to(device)

    generator = torch.Generator().manual_seed(1234)
    spec = torch.randn(1, n_fft // 2 + 1, 12, generator=generator).to(device)
    spec = torch.view_as_complex(torch.stack([spec, spec * 0.5], dim=-1))
    with torch.no_grad():
        audio_ours = ours(spec)
        audio_oracle = oracle(spec)
    max_diff = (audio_ours - audio_oracle).abs().max().item()
    assert torch.equal(audio_ours, audio_oracle), f"max diff {max_diff:.3e}"

    # The streaming form over one whole chunk must be the same computation.
    stream_ours, _ = ours.forward_stream(spec, None, last_chunk=True)
    max_diff = (stream_ours - audio_ours).abs().max().item()
    assert torch.equal(stream_ours, audio_ours), f"stream max diff {max_diff:.3e}"


@requires_all
@pytest.mark.accelerator
def test_istft_head_matches_reference(device, reference_sources) -> None:
    from fireredtts3.redae.redae import ISTFTHead as ReferenceISTFTHead

    state = read_component_tensors(BASE_CHECKPOINT / "redae", "decoder.istft_head.")
    ours = ISTFTHead(dim=896, n_fft=1920, hop_length=480)
    ours.load_state_dict(state, strict=True)
    ours = ours.to(device).eval()
    oracle = ReferenceISTFTHead(dim=896, n_fft=1920, hop_length=480)
    oracle.load_state_dict(state, strict=True)
    oracle = oracle.to(device).eval()

    generator = torch.Generator().manual_seed(1234)
    hidden = torch.randn(1, 12, 896, generator=generator).to(device)
    with torch.no_grad():
        audio_ours = ours(hidden)
        audio_oracle = oracle(hidden)
    max_diff = (audio_ours - audio_oracle).abs().max().item()
    assert torch.equal(audio_ours, audio_oracle), f"max diff {max_diff:.3e}"

    stream_ours, _ = ours.forward_stream(hidden, None, last_chunk=True)
    max_diff = (stream_ours - audio_ours).abs().max().item()
    assert torch.equal(stream_ours, audio_ours), f"stream max diff {max_diff:.3e}"


def test_rematerialize_istft_window_repairs_meta_materialization() -> None:
    from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config

    with torch.device("meta"):
        decoder = build_redae_decoder(FireRedTTS3Config())
    # Simulate a meta-device load that never saw the window buffer.
    decoder = decoder.to_empty(device="cpu")
    rematerialize_istft_window(decoder, "cpu")
    assert torch.equal(decoder.istft_head.istft.window, torch.hann_window(1920))


def test_istft_stream_envelope_floor_healthy_on_production_geometry() -> None:
    """The kept-region OLA envelope stays far above ENVELOPE_FLOOR for every
    chunk shape the streaming decode can emit, so the clamp is a no-op on the
    healthy path (bit-exactness) and only catches degenerate edge geometry."""
    from sglang_omni.models.firered_tts3.istft import ENVELOPE_FLOOR

    n_fft, hop = 1920, 480
    istft = ISTFT(n_fft=n_fft, hop_length=hop, win_length=n_fft)
    pad = (n_fft - hop) // 2
    tail = n_fft - hop
    cache_frames = n_fft // hop - 1

    def envelope(frames: int) -> torch.Tensor:
        window_sq = istft.window.square().expand(1, frames, -1).transpose(1, 2)
        output_size = (frames - 1) * hop + n_fft
        return torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, n_fft),
            stride=(1, hop),
        ).squeeze()

    floors = []
    for frames in range(1, 65):
        env = envelope(frames)
        if env.numel() - tail > pad:  # cache=None, not last: [pad:-tail]
            floors.append(float(env[pad : env.numel() - tail].min()))
        if env.numel() - pad > pad:  # cache=None, last: [pad:-pad]
            floors.append(float(env[pad : env.numel() - pad].min()))
        warm = envelope(frames + cache_frames)
        if warm.numel() - tail > tail:  # cached, not last: [tail:-tail]
            floors.append(float(warm[tail : warm.numel() - tail].min()))
        if warm.numel() - pad > tail:  # cached, last: [tail:-pad]
            floors.append(float(warm[tail : warm.numel() - pad].min()))
    flush_env = envelope(cache_frames)  # flush_stream: [tail:-pad]
    floors.append(float(flush_env[tail : flush_env.numel() - pad].min()))
    assert floors, "no production chunk shape produced a kept region"
    assert min(floors) > 0.1, f"kept-region envelope floor {min(floors):.3e}"
    assert ENVELOPE_FLOOR < 1e-3 * min(floors)


def test_istft_stream_envelope_clamp_keeps_output_finite() -> None:
    """A degenerate (zeroed) window collapses the fold envelope; the clamp
    keeps the normalized output finite instead of emitting NaN/inf."""
    n_fft, hop = 1920, 480
    istft = ISTFT(n_fft=n_fft, hop_length=hop, win_length=n_fft)
    istft.window = torch.zeros_like(istft.window)
    spec = torch.randn(2, n_fft // 2 + 1, 8, dtype=torch.float32)
    spec = torch.view_as_complex(torch.stack([spec, -spec], dim=-1))
    audio, cache = istft.forward_stream(spec, None, last_chunk=False)
    assert torch.isfinite(audio).all()
    assert torch.isfinite(cache).all()
    audio, _ = istft.forward_stream(spec, cache, last_chunk=True)
    assert torch.isfinite(audio).all()
    assert torch.isfinite(istft.flush_stream(cache)).all()


def test_decoder_state_defaults() -> None:
    state = RedAEDecoderState()
    assert state.qwen3_cache is None
    assert state.istft_cache is None
    assert state.frames_seen == 0
