# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the FireRedTTS3 unit tests.

The numerical oracles live in the reference sources, which are not on
pythonpath; they are mounted here (single place, session scope) rather than
ad hoc per test file.

Point FIREREDTTS3_CHECKPOINT_DIR at the directory holding the FireRedTTS3-Base /
-Instruct / -Speaker checkpoints and FIREREDTTS3_REFERENCE_DIR at the local reference
sources root; tests that need either skip when they are unset.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

CHECKPOINT_ROOT = Path(os.environ.get("FIREREDTTS3_CHECKPOINT_DIR", "/nonexistent"))
CHECKPOINTS = {
    "base": CHECKPOINT_ROOT / "FireRedTTS3-Base",
    "instruct": CHECKPOINT_ROOT / "FireRedTTS3-Instruct",
    "speaker": CHECKPOINT_ROOT / "FireRedTTS3-Speaker",
}

REFERENCE_REPO = Path(os.environ.get("FIREREDTTS3_REFERENCE_DIR", "/nonexistent"))
REFERENCE_FIREREDTTS3 = REFERENCE_REPO / "FireRedTTS3"

checkpoints_present = all(path.is_dir() for path in CHECKPOINTS.values())
requires_checkpoints = pytest.mark.skipif(
    not checkpoints_present, reason="FireRedTTS3 checkpoints not present"
)

reference_present = (
    REFERENCE_FIREREDTTS3.is_dir() and (REFERENCE_REPO / "diandiantts").is_dir()
)
requires_reference = pytest.mark.skipif(
    not reference_present, reason="FireRedTTS3 reference sources not present"
)


@pytest.fixture(scope="session")
def reference_sources() -> None:
    """Mount the reference packages (fireredtts3 + diandiantts)."""
    if not reference_present:
        pytest.skip("FireRedTTS3 reference sources not present")
    for path in (str(REFERENCE_FIREREDTTS3), str(REFERENCE_REPO)):
        if path not in sys.path:
            sys.path.insert(0, path)


def pin_native_rms_norm(module) -> None:
    """Force the reference RMSNorm onto the native F.rms_norm path.

    The reference sniffs float(torch.__version__[:3]) >= 2.4, which misparses
    torch 2.13 as 2.1 and silently takes the manual fallback; the reference
    itself ran on torch >= 2.4 (native). Pinning the flag restores the
    intended computation for oracle comparisons.
    """
    for submodule in module.modules():
        if hasattr(submodule, "native_rms_norm"):
            submodule.native_rms_norm = True


def synth_audio(sample_rate: int, seconds: float):
    """Harmonic chirp + AM + noise (the reference stream-verification input)."""
    import torch

    generator = torch.Generator().manual_seed(0)
    t = torch.arange(int(sample_rate * seconds)) / sample_rate
    f0 = 120 + 60 * torch.sin(2 * torch.pi * 0.7 * t)
    phase = 2 * torch.pi * torch.cumsum(f0, dim=0) / sample_rate
    audio = sum((0.5 / k) * torch.sin(k * phase + 0.3 * k) for k in range(1, 6))
    audio = audio * (0.6 + 0.4 * torch.sin(2 * torch.pi * 3.1 * t))
    audio = audio + 0.01 * torch.randn(audio.shape, generator=generator)
    return audio.unsqueeze(0)
