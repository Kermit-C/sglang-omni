# SPDX-License-Identifier: Apache-2.0
"""RedAE window restoration and lossless streaming overlap-add."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.firered_tts3.istft import ISTFT
from sglang_omni.models.firered_tts3.redae_decoder import (
    RedAEAudioDecoder,
    rematerialize_istft_window,
)


def test_rematerialize_istft_window_repairs_meta_materialization(
    decoder: RedAEAudioDecoder,
) -> None:
    decoder.to(device="meta").to_empty(device="cpu")
    rematerialize_istft_window(decoder, "cpu")
    assert torch.equal(decoder.istft_head.istft.window, torch.hann_window(1920))


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


@pytest.mark.parametrize("flush_tail", [False, True])
def test_istft_streaming_preserves_samples_across_chunk_boundaries(
    flush_tail: bool,
) -> None:
    istft = ISTFT(n_fft=1920, hop_length=480, win_length=1920)
    generator = torch.Generator().manual_seed(84)
    spectrum = torch.randn(2, 961, 12, dtype=torch.complex64, generator=generator)
    expected = istft(spectrum)
    cache = None
    chunks = []
    for index, chunk in enumerate(spectrum.split(4, dim=-1)):
        audio, cache = istft.forward_stream(
            chunk, cache, last_chunk=index == 2 and not flush_tail
        )
        chunks.append(audio)
    if flush_tail:
        chunks.append(istft.flush_stream(cache))
    else:
        pass
    assert torch.equal(torch.cat(chunks, dim=-1), expected)
