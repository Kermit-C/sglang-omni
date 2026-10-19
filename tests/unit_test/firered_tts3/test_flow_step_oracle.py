# SPDX-License-Identifier: Apache-2.0
"""Loop-semantics oracle: the runner's flow_step vs the reference loop.

Drives the REAL FireRedTTS3ModelRunner.flow_step with the hidden-state
trajectory recorded from the reference FireRedTTS3BaseCore.generate (same
checkpoint, same seed-1234 noise draws). The reference AR loop is chaotic
with respect to backbone kernels, so the comparison replays the reference
hidden states instead of running a second backbone: every patch the runner
emits must then be bit-exact against the reference's, and the stop decision
must land on the same step (a stopping step emits no patch). This pins the
semantic red lines: stop-before-generate with the min-patches floor, the
first-step full prompt-span cond, the sliding cond window of
num_history_patches + 1, and the per-request generator draw order.
"""

from __future__ import annotations

import collections
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.firered_tts3.model_runner import (
    FireRedTTS3ModelRunner,
    FlowState,
)
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.models.firered_tts3.weights import (
    load_dit,
    load_patch_encoder,
    read_component_tensors,
)

from .conftest import CHECKPOINTS, REFERENCE_FIREREDTTS3, pin_native_rms_norm

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


def build_reference_core(device: torch.device, monkeypatch):
    import fireredtts3.redae.redae as reference_redae
    from fireredtts3.llm import fireredtts3_base as reference_base

    real_config = reference_redae.Qwen3Config

    def config_without_fa2(**kwargs):
        kwargs.pop("attn_implementation", None)
        return real_config(**kwargs)

    # The reference pins flash_attention_2, unavailable here; sdpa keeps the
    # oracle on identical kernels to the ported-component tests.
    monkeypatch.setattr(reference_redae, "Qwen3Config", config_without_fa2)
    reference_base.Qwen3_1_7B_ConfigDict["attn_implementation"] = "sdpa"

    core_config = reference_base.FireRedTTS3BaseCoreConfig()
    core = reference_base.FireRedTTS3BaseCore(core_config)
    state = read_component_tensors(BASE_CHECKPOINT, "")
    core.load_state_dict(state, strict=True)
    pin_native_rms_norm(core)
    return core.to(device=device, dtype=torch.float32).eval()


@torch.no_grad()
def reference_loop(core, text_ids, prompt_latents, spk_emb, device, num_steps):
    """The reference generate loop, recording hidden rows and patches.

    Mirrors fireredtts3_base.FireRedTTS3BaseCore.generate step for step;
    the backbone stays fp32 without the reference's autocast so the recorded
    trajectory carries maximal precision (the comparison is about loop
    semantics, not backbone kernels). Random synthetic inputs never trip the
    stop head, so the loop runs a fixed window; the stop branch is covered
    by a separate crafted test.
    """
    patch_prompt = core.patch_encoder(prompt_latents)
    spk_llm = core.spk_proj_llm(spk_emb)
    embeds = torch.cat(
        [spk_llm.unsqueeze(1), core.backbone_llm.embed_tokens(text_ids), patch_prompt],
        dim=1,
    )
    n_timesteps, cfg = 10, 2.0
    t_span = torch.linspace(0, 1, n_timesteps + 1, device=device)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    latents_gen = F.pad(prompt_latents, (0, 0, core.history_length, 0))
    spk_cond = core.spk_proj_dit(spk_emb)
    backbone_cond = embeds.new_zeros(1, core.history_patches, embeds.shape[-1])
    from fireredtts3.utils.utils import fix_seed

    fix_seed(1234)
    cache = None
    hidden_rows = []
    patches = []
    for step in range(num_steps):
        out = core.backbone_llm(
            inputs_embeds=embeds, use_cache=True, past_key_values=cache
        )
        hidden = out.last_hidden_state
        cache = out.past_key_values
        hidden_rows.append(hidden[0].float())
        stop_score = (
            torch.sigmoid(core.stop_head(hidden[0].float()[-1:].unsqueeze(0)))
            .reshape(-1)[-1]
            .item()
        )
        assert not (stop_score >= 0.5 and step >= 6), "synthetic loop must not stop"
        if step == 0:
            rows = hidden[0].float()[-patch_prompt.shape[1] :]
        else:
            rows = hidden[0].float()[-1:]
        backbone_cond = torch.cat([backbone_cond, rows.unsqueeze(0)], dim=1)
        bb = core.dit_head(backbone_cond[:, -(core.history_patches + 1) :])
        bb = bb.repeat_interleave(core.patch_size, dim=1)
        cond = torch.cat(
            [bb, spk_cond.unsqueeze(1).expand(-1, bb.shape[1], -1)], dim=-1
        )
        hist = latents_gen[:, -core.history_length :]
        x0 = torch.randn(1, core.patch_size, core.redae_dim, device=device)
        xt = torch.cat([hist, x0], dim=1)
        for ti in range(n_timesteps):
            t_in = t_span[ti].view(-1, 1, 1)
            dt = t_span[ti + 1] - t_span[ti]
            x_in = torch.cat([xt, cond], dim=2)
            x_in_cfg = torch.cat([xt, cond * 0], dim=2)
            x_in = torch.cat([x_in, x_in_cfg], dim=0)
            t_in = t_in.expand(2, -1, -1)
            vt = core.dit(x=x_in, t=t_in)
            vt_c, vt_u = vt.chunk(2, dim=0)
            vt = (1.0 + cfg) * vt_c - cfg * vt_u
            xt[:, -core.patch_size :] = (
                xt[:, -core.patch_size :]
                + dt.view(-1, 1, 1) * vt[:, -core.patch_size :]
            )
        patch = xt[:, -core.patch_size :].clone()
        patches.append(patch)
        latents_gen = torch.cat([latents_gen, patch], dim=1)
        embeds = core.patch_encoder(patch)
    return hidden_rows, patches


class FakeModel:
    """The side-path view of FireRedTTS3SGLangModel that flow_step reads."""

    def __init__(self, config, device):
        self.patch_encoder = load_patch_encoder(
            str(BASE_CHECKPOINT), config, device=device
        )
        self.dit = load_dit(str(BASE_CHECKPOINT), config, device=device)
        side = read_component_tensors(BASE_CHECKPOINT, "")
        self.dit_head = torch.nn.Linear(config.hidden_size, config.dit_hidden_size)
        self.stop_head = torch.nn.Linear(config.hidden_size, 1)
        self.spk_proj_llm = torch.nn.Linear(config.spk_in_dim, config.hidden_size)
        self.spk_proj_dit = torch.nn.Linear(config.spk_in_dim, config.spk_in_dim)
        for name, module in (
            ("dit_head", self.dit_head),
            ("stop_head", self.stop_head),
            ("spk_proj_llm", self.spk_proj_llm),
            ("spk_proj_dit", self.spk_proj_dit),
        ):
            module.load_state_dict(
                {
                    "weight": side[f"{name}.weight"],
                    "bias": side[f"{name}.bias"],
                }
            )
            module.to(device=device, dtype=torch.float32).eval()
        self.patch_size = int(config.patch_size)
        self.history_patches = int(config.num_history_patches)
        self.history_length = self.patch_size * self.history_patches
        self.redae_dim = int(config.redae_dim)
        self.llm_hidden_size = int(config.hidden_size)

    def parameters(self):
        # The runner casts the feedback row to the backbone dtype (bf16 in
        # production); the replay does not consume the feedback, so a dummy
        # bf16 parameter is enough.
        return iter([torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))])


@requires_all
@pytest.mark.accelerator
def test_flow_step_matches_reference_loop(
    device, reference_sources, monkeypatch
) -> None:
    from sglang_omni.models.firered_tts3.config import load_checkpoint_hf_config

    config = load_checkpoint_hf_config(str(BASE_CHECKPOINT))
    assert config is not None
    core = build_reference_core(device, monkeypatch)

    generator = torch.Generator().manual_seed(0)
    prompt_latents = torch.randn(1, 24, 64, generator=generator).to(device)
    spk_emb = torch.randn(1, 512, generator=generator).to(device)
    text_ids = torch.randint(1000, 150000, (1, 17), generator=generator).to(device)

    steps = 12
    hidden_rows, ref_patches = reference_loop(
        core, text_ids, prompt_latents, spk_emb, device, steps
    )

    runner = object.__new__(FireRedTTS3ModelRunner)
    runner.model = FakeModel(config, device)
    runner.t_span_cache = {}
    data = SimpleNamespace(
        flow_state=FlowState(
            generator=torch.Generator(device=device).manual_seed(1234),
            prompt_patch_count=prompt_latents.shape[1] // config.patch_size,
            spk_cond_dit=runner.model.spk_proj_dit(spk_emb.float()),
            backbone_cond=prompt_latents.new_zeros(
                1, config.num_history_patches, config.hidden_size
            ),
            latents_gen=F.pad(prompt_latents, (0, 0, 2 * config.patch_size, 0)),
            patch_count=0,
        ),
        state=FireRedTTS3State(
            stop_threshold=0.5, min_patches=6, n_timesteps=10, inference_cfg=2.0
        ),
        pending_feedback_queue=collections.deque(),
        latent_patches=[],
        req=SimpleNamespace(finished_reason=None),
        probe=ProfilingProbe(enabled=False),
    )

    for hidden in hidden_rows:
        runner.flow_step(data, hidden.to(device=device, dtype=torch.float32))

    assert data.req.finished_reason is None
    assert len(data.latent_patches) == steps
    assert len(data.pending_feedback_queue) == steps
    for index, (got, want) in enumerate(zip(data.latent_patches, ref_patches)):
        assert torch.equal(got, want), f"patch {index} diverged"


@requires_all
@pytest.mark.accelerator
def test_flow_step_stop_branch_and_min_patches(device) -> None:
    """The stop decision precedes patch production and honors min_patches."""
    from sglang_omni.models.firered_tts3.config import load_checkpoint_hf_config

    config = load_checkpoint_hf_config(str(BASE_CHECKPOINT))
    assert config is not None
    runner = object.__new__(FireRedTTS3ModelRunner)
    runner.model = FakeModel(config, device)
    runner.t_span_cache = {}

    # A hidden row whose stop logit clears the threshold: the stop head is a
    # Linear(2048 -> 1), so a large enough multiple of its weight row does it.
    weight = runner.model.stop_head.weight  # [1, 2048]
    bias = float(runner.model.stop_head.bias[0])
    direction = weight[0] / weight[0].norm().pow(2)
    # logit = t + bias; pick t comfortably past sigmoid >= 0.5 (logit 0).
    hidden = (direction * (8.0 - bias)).reshape(1, -1).to(device)
    # The Base DiT always consumes the speaker channel.
    spk_cond = runner.model.spk_proj_dit(torch.randn(1, 512, device=device))

    def make_data(patch_count: int):
        return SimpleNamespace(
            flow_state=FlowState(
                generator=torch.Generator(device=device).manual_seed(1),
                prompt_patch_count=0,
                spk_cond_dit=spk_cond,
                backbone_cond=torch.zeros(
                    1, config.num_history_patches, config.hidden_size, device=device
                ),
                latents_gen=torch.zeros(1, 8, 64, device=device),
                patch_count=patch_count,
            ),
            state=FireRedTTS3State(
                stop_threshold=0.5, min_patches=6, n_timesteps=10, inference_cfg=2.0
            ),
            pending_feedback_queue=collections.deque(),
            latent_patches=[],
            req=SimpleNamespace(finished_reason=None),
            probe=ProfilingProbe(enabled=False),
        )

    # patch_count below min_patches: no stop, a patch is produced.
    data = make_data(patch_count=5)
    runner.flow_step(data, hidden)
    assert data.req.finished_reason is None
    assert len(data.latent_patches) == 1
    assert len(data.pending_feedback_queue) == 1

    # patch_count at min_patches: stop, no patch, no feedback queued.
    data = make_data(patch_count=6)
    runner.flow_step(data, hidden)
    assert data.req.finished_reason is not None
    assert not data.latent_patches
    assert not data.pending_feedback_queue
