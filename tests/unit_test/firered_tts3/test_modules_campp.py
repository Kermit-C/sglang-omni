# SPDX-License-Identifier: Apache-2.0
"""Numerical oracle test: ported CAM++ speaker encoder vs the reference."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.firered_tts3.weights import load_campp

from .conftest import CHECKPOINTS, REFERENCE_FIREREDTTS3, synth_audio

BASE_CHECKPOINT = CHECKPOINTS["base"]
REFERENCE_SRC = REFERENCE_FIREREDTTS3

requires_all = pytest.mark.skipif(
    not BASE_CHECKPOINT.is_dir() or not REFERENCE_SRC.is_dir(),
    reason="FireRedTTS3-Base checkpoint or reference sources not present",
)


@requires_all
@pytest.mark.accelerator
def test_campp_embedding_matches_reference(reference_sources) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from fireredtts3.campp.campp import CamppEmbedding as ReferenceCamppEmbedding

    device = torch.device("cuda")
    ours = load_campp(str(BASE_CHECKPOINT), device=device)
    oracle = ReferenceCamppEmbedding(
        model_path=str(BASE_CHECKPOINT / "campp" / "campplus_voxceleb.bin")
    )
    oracle = oracle.to(device)

    audio = synth_audio(24000, 3.0)  # (1, samples) fp32
    with torch.no_grad():
        embedding_ours = ours(audio, 24000)
        embedding_oracle = oracle(audio, 24000).to(device)

    assert embedding_ours.shape == (1, 512)
    max_diff = (embedding_ours - embedding_oracle).abs().max().item()
    assert torch.equal(embedding_ours, embedding_oracle), f"max diff {max_diff:.3e}"
