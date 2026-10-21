# SPDX-License-Identifier: Apache-2.0
"""Numerical oracle tests: ported flow side path vs the FireRedTTS3 reference.

PatchEncoder and DiT run fp32 end to end (precision red line); with the same
checkpoint weights and the reference RMSNorm pinned to its intended native
path (see conftest.pin_native_rms_norm), the port must be bit-exact.
"""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.weights import (
    load_dit,
    load_patch_encoder,
    read_component_tensors,
)

from .conftest import CHECKPOINTS, REFERENCE_FIREREDTTS3, pin_native_rms_norm

BASE_CHECKPOINT = CHECKPOINTS["base"]

requires_all = pytest.mark.skipif(
    not BASE_CHECKPOINT.is_dir() or not REFERENCE_FIREREDTTS3.is_dir(),
    reason="FireRedTTS3 checkpoint or reference sources not present",
)


def load_base_config() -> FireRedTTS3Config:
    from sglang_omni.models.firered_tts3.config import load_checkpoint_hf_config

    config = load_checkpoint_hf_config(str(BASE_CHECKPOINT))
    assert config is not None
    return config


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


@requires_all
@pytest.mark.accelerator
def test_patch_encoder_matches_reference(device, reference_sources) -> None:
    from fireredtts3.llm.patch_encoder import PatchEncoder as ReferencePatchEncoder

    config = load_base_config()
    ours = load_patch_encoder(str(BASE_CHECKPOINT), config, device=device)

    oracle = ReferencePatchEncoder(
        in_dim=config.redae_dim,
        out_dim=config.hidden_size,
        patch_size=config.patch_size,
        hidden_size=config.patch_encoder_hidden_size,
        mlp_ratio=config.patch_encoder_mlp_ratio,
        depth=config.patch_encoder_depth,
        num_heads=config.patch_encoder_num_heads,
    )
    oracle.load_state_dict(
        read_component_tensors(BASE_CHECKPOINT, "patch_encoder."), strict=True
    )
    pin_native_rms_norm(oracle)
    oracle = oracle.to(device=device, dtype=torch.float32).eval()

    generator = torch.Generator(device="cpu").manual_seed(1234)
    inputs = torch.randn(1, 8, config.redae_dim, generator=generator).to(device)
    with torch.no_grad():
        out_ours = ours(inputs)
        out_oracle = oracle(inputs)

    assert out_ours.shape == (1, 2, config.hidden_size)
    max_diff = (out_ours - out_oracle).abs().max().item()
    assert torch.equal(out_ours, out_oracle), f"max diff {max_diff:.3e}"


@requires_all
@pytest.mark.accelerator
def test_dit_single_step_matches_reference(device, reference_sources) -> None:
    from fireredtts3.llm.dit import DiT as ReferenceDiT

    config = load_base_config()
    ours = load_dit(str(BASE_CHECKPOINT), config, device=device)

    oracle = ReferenceDiT(
        in_channels=config.dit_in_channels,
        out_channels=config.redae_dim,
        mlp_ratio=config.dit_mlp_ratio,
        depth=config.dit_depth,
        num_heads=config.dit_num_heads,
        hidden_size=config.dit_hidden_size,
    )
    oracle.load_state_dict(read_component_tensors(BASE_CHECKPOINT, "dit."), strict=True)
    pin_native_rms_norm(oracle)
    oracle = oracle.to(device=device, dtype=torch.float32).eval()

    generator = torch.Generator(device="cpu").manual_seed(1234)
    # DiT context is fixed at 12 frames: 8 clean history + 4 noised targets.
    x_in = torch.randn(2, 12, config.dit_in_channels, generator=generator).to(device)
    timesteps = torch.tensor([0.13, 0.87], device=device)
    with torch.no_grad():
        out_ours = ours(x_in, timesteps)
        out_oracle = oracle(x_in, timesteps)

    assert out_ours.shape == (2, 12, config.redae_dim)
    max_diff = (out_ours - out_oracle).abs().max().item()
    assert torch.equal(out_ours, out_oracle), f"max diff {max_diff:.3e}"


@requires_all
@pytest.mark.accelerator
def test_dit_instruct_width_matches_reference(device, reference_sources) -> None:
    """The 1088-channel DiT variant (no speaker channel) must also be exact."""
    from fireredtts3.llm.dit import DiT as ReferenceDiT

    instruct_checkpoint = CHECKPOINTS["instruct"]
    if not instruct_checkpoint.is_dir():
        pytest.skip("Instruct checkpoint not present")
    from sglang_omni.models.firered_tts3.config import load_checkpoint_hf_config

    config = load_checkpoint_hf_config(str(instruct_checkpoint))
    assert config is not None and config.dit_in_channels == 1088
    ours = load_dit(str(instruct_checkpoint), config, device=device)

    oracle = ReferenceDiT(
        in_channels=config.dit_in_channels,
        out_channels=config.redae_dim,
        mlp_ratio=config.dit_mlp_ratio,
        depth=config.dit_depth,
        num_heads=config.dit_num_heads,
        hidden_size=config.dit_hidden_size,
    )
    oracle.load_state_dict(
        read_component_tensors(instruct_checkpoint, "dit."), strict=True
    )
    pin_native_rms_norm(oracle)
    oracle = oracle.to(device=device, dtype=torch.float32).eval()

    generator = torch.Generator(device="cpu").manual_seed(7)
    x_in = torch.randn(1, 12, config.dit_in_channels, generator=generator).to(device)
    timesteps = torch.tensor([0.5], device=device)
    with torch.no_grad():
        out_ours = ours(x_in, timesteps)
        out_oracle = oracle(x_in, timesteps)

    max_diff = (out_ours - out_oracle).abs().max().item()
    assert torch.equal(out_ours, out_oracle), f"max diff {max_diff:.3e}"
