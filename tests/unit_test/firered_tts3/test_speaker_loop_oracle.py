# SPDX-License-Identifier: Apache-2.0
"""Loop-semantics oracle: the runner's side path vs the reference speaker
variant.

Drives the REAL FireRedTTS3ModelRunner side path (flow_step for the prefill
form, the dual-stream machine for the streaming form) with the hidden-state
trajectory recorded from the reference DiTAR_V1 loops (same checkpoint, same
seed-1234 noise draws). The reference AR loop is chaotic with respect to
backbone kernels, so the comparison replays the reference hidden states
instead of running a second backbone: every patch the runner emits must then
be bit-exact against the reference's, and the stop decision must land on the
same step. This pins the semantic red lines: prefill-form stop-before-flow
(stopping step emits nothing) vs dual-stream stop-after-patch-feed (the
stopping step's patch is already emitted), the cond-row selection (last text
feed of each group), and the text-pad fallback.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.firered_tts3.model_runner import (
    FireRedTTS3ModelRunner,
    FlowState,
)
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.models.firered_tts3.request_builders import (
    CONTROL_TOKEN_ID,
    FireRedTTS3SGLangRequestData,
)
from sglang_omni.models.firered_tts3.weights import (
    load_dit,
    load_patch_encoder,
    read_component_tensors,
)

from .conftest import CHECKPOINTS, REFERENCE_REPO

SPEAKER_CHECKPOINT = CHECKPOINTS["speaker"]
GOLDEN_TOOLS = Path(__file__).resolve().parents[3] / "scripts" / "fireredtts3"

requires_all = pytest.mark.skipif(
    not SPEAKER_CHECKPOINT.is_dir()
    or not REFERENCE_REPO.is_dir()
    or not GOLDEN_TOOLS.is_dir(),
    reason="speaker checkpoint, reference sources or golden tools not present",
)

TEXT = "简单地说，这相当于惠普把消费领域市场拱手相让了。"


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


def build_reference_model(device: torch.device):
    """The reference DiTAR_V1 (backbone FA2 -> sdpa substitution, as in the
    component oracles); built manually to skip the hyperpyyaml config load.
    The reference's RMSNorm version sniff misparses torch 2.13 and would take
    the manual path; pin it to the native F.rms_norm path (the path the
    reference itself takes on torch >= 2.4, and the one our port matches
    bit-exactly)."""
    scripts = str(GOLDEN_TOOLS)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    if str(REFERENCE_REPO) not in sys.path:
        sys.path.insert(0, str(REFERENCE_REPO))
    from make_speaker_golden import build_model

    from .conftest import pin_native_rms_norm

    model = build_model(REFERENCE_REPO, str(device))
    pin_native_rms_norm(model)
    return model


def reference_tokenizer():
    from diandiantts.utils.text_tokenizer import WrappedQwenTextTokenizer

    return WrappedQwenTextTokenizer(
        tokenizer_dir=str(REFERENCE_REPO / "pretrained_models" / "Qwen3-1.7B")
    )


def reference_prompt_ids(tokenizer, text: str) -> torch.Tensor:
    """The reference's nonstreaming prompt: chatml with the text
    masked-scattered into the text_holder slots."""
    from diandiantts.utils.chatml import (
        LATENT_END_TOKEN,
        LATENT_HOLDER_TOKEN,
        LATENT_START_TOKEN,
        TEXT_HOLDER_ID,
        TEXT_HOLDER_TOKEN,
        TEXT_PAD_TOKEN,
        format_chatml,
    )

    text_tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    chatml = format_chatml(
        text_len=text_tokens.shape[-1],
        latent_len=0,
        speaker="<|f531_speaker|>",
        text_holder=TEXT_HOLDER_TOKEN,
        text_pad=TEXT_PAD_TOKEN,
        latent_holder=LATENT_HOLDER_TOKEN,
        latent_start_token=LATENT_START_TOKEN,
        latent_end_token=LATENT_END_TOKEN,
        use_interleave=False,
    )
    chatml = chatml.removesuffix("<|im_end|>\n").removesuffix(LATENT_END_TOKEN)
    chatml_tokens = torch.tensor(tokenizer.encode(chatml), dtype=torch.long)
    return chatml_tokens.masked_scatter(
        chatml_tokens == TEXT_HOLDER_ID, text_tokens
    ).unsqueeze(0)


@torch.no_grad()
def reference_nonstream_trajectory(model, prompt_ids, device, max_steps=600):
    """The reference generate_nonstreaming_noicl loop, recording the
    last-position hidden row, the stop decision, and the patch per step.
    The backbone runs fp32 without the reference's autocast so the recorded
    trajectory carries maximal precision (the comparison is about loop
    semantics, not backbone kernels)."""
    from diandiantts.utils.utils import fix_seed

    fix_seed(1234)
    backbone_in = model.backbone_llm.model.embed_tokens(prompt_ids.to(device))
    t_span = torch.linspace(0, 1, 11, device=device)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    backbone_cond = torch.zeros(
        1, model.history_patches, model.hidden_size, device=device
    )
    latents_gen = torch.zeros(
        1, model.history_length, model.redae.hidden_size, device=device
    )
    cache = None
    rows = []
    for step_index in range(max_steps):
        backbone_out, cache = model._backbone_one_step(
            backbone_in, cache=cache
        )  # noqa: leading-underscore  # reference oracle method
        hidden = backbone_out[0, -1].float().cpu()
        backbone_cond = torch.cat([backbone_cond, backbone_out[:, -1:]], dim=1)
        stop_score = float(
            torch.sigmoid(model.stop_head(backbone_cond[:, -1]).squeeze(-1)).item()
        )
        stopped = stop_score >= 0.5 and step_index >= 6
        patch = None
        if not stopped:
            patch = (
                model._flow_one_step(  # noqa: leading-underscore  # reference oracle method
                    backbone_cond=backbone_cond,
                    latents_gen=latents_gen,
                    t_span=t_span,
                    inference_cfg=1.2,
                )
                .float()
                .cpu()
            )
            latents_gen = torch.cat([latents_gen, patch.to(device)], dim=1)
            backbone_in = model.patch_encoder.forward(patch.to(device))
        rows.append((hidden, patch, stopped))
        if stopped:
            break
    return rows


class SpeakerSidePath:
    """The side-path view of FireRedTTS3SGLangModel that the runner reads."""

    def __init__(self, config, device):
        self.patch_encoder = load_patch_encoder(
            str(SPEAKER_CHECKPOINT), config, device=device
        )
        self.dit = load_dit(str(SPEAKER_CHECKPOINT), config, device=device)
        side = read_component_tensors(SPEAKER_CHECKPOINT, "")
        self.dit_head = torch.nn.Linear(config.hidden_size, config.dit_hidden_size)
        self.stop_head = torch.nn.Linear(config.hidden_size, 1)
        for name, module in (
            ("dit_head", self.dit_head),
            ("stop_head", self.stop_head),
        ):
            module.load_state_dict(
                {"weight": side[f"{name}.weight"], "bias": side[f"{name}.bias"]}
            )
            module.to(device=device, dtype=torch.float32).eval()
        self.patch_size = int(config.patch_size)
        self.history_patches = int(config.num_history_patches)
        self.history_length = self.patch_size * self.history_patches
        self.redae_dim = int(config.redae_dim)
        self.llm_hidden_size = int(config.hidden_size)
        self.spk_proj_dit = None

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))])


def make_runner(model) -> FireRedTTS3ModelRunner:
    runner = object.__new__(FireRedTTS3ModelRunner)
    runner.async_enabled = False
    runner.model = model
    runner.t_span_cache = {}
    return runner


def make_flow_data(device) -> SimpleNamespace:
    return SimpleNamespace(
        flow_state=FlowState(
            generator=torch.Generator(device=device).manual_seed(1234),
            prompt_patch_count=0,
            spk_cond_dit=None,
            backbone_cond=torch.zeros(1, 2, 2048, device=device),
            latents_gen=torch.zeros(1, 8, 64, device=device),
            patch_count=0,
        ),
        state=FireRedTTS3State(
            stop_threshold=0.5, min_patches=6, n_timesteps=10, inference_cfg=1.2
        ),
        pending_feedback_queue=collections.deque(),
        latent_patches=[],
        req=SimpleNamespace(finished_reason=None),
        probe=ProfilingProbe(enabled=False),
    )


@requires_all
@pytest.mark.accelerator
def test_speaker_prefill_flow_matches_reference(device, reference_sources) -> None:
    from sglang_omni.models.firered_tts3.config import load_checkpoint_hf_config

    config = load_checkpoint_hf_config(str(SPEAKER_CHECKPOINT))
    assert config is not None
    model = build_reference_model(device)
    tokenizer = reference_tokenizer()
    prompt_ids = reference_prompt_ids(tokenizer, TEXT)
    trajectory = reference_nonstream_trajectory(model, prompt_ids, device)
    # The reference stops on this utterance well before the cap.
    assert trajectory[-1][2] is True

    runner = make_runner(SpeakerSidePath(config, device))
    data = make_flow_data(device)
    for hidden, _patch, _stopped in trajectory:
        runner.flow_step(data, hidden.to(device=device).unsqueeze(0))

    ref_patches = [
        patch for _hidden, patch, _stopped in trajectory if patch is not None
    ]
    assert len(data.latent_patches) == len(ref_patches)
    assert data.req.finished_reason is not None  # stopped on the same step
    for index, (got, want) in enumerate(zip(data.latent_patches, ref_patches)):
        assert torch.equal(got.cpu(), want), f"patch {index} diverged"


@torch.no_grad()
def reference_stream_trajectory(
    model, prefix_ids, text_ids, text_pad_id, device, max_groups=40
):
    """The reference generate_streaming_noicl loop, recorded per engine-equivalent
    step: (feed_kind, feed_value, hidden_row, patch_or_none, stop_flag).

    feed_kind is "text" (feed_value the token id) or "patch" (feed_value the
    index of the generated patch being fed). The backbone runs fp32 without
    the reference's autocast (the comparison is about loop semantics).
    """
    from diandiantts.utils.utils import fix_seed

    nt, na = 2, 1
    fix_seed(1234)
    backbone_in = model.backbone_llm.model.embed_tokens(prefix_ids.to(device))
    backbone_out, cache = model._backbone_one_step(
        backbone_in, cache=None
    )  # noqa: leading-underscore  # reference oracle method
    t_span = torch.linspace(0, 1, 11, device=device)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    latents_gen = torch.zeros(
        1, model.history_length, model.redae.hidden_size, device=device
    )
    backbone_cond = torch.zeros(
        1, model.history_patches, model.hidden_size, device=device
    )
    steps: list[tuple[str, int, torch.Tensor, torch.Tensor | None, bool]] = []
    text_pos = 0
    audio_step = 0
    patches: list[torch.Tensor] = []
    stopped = False
    for _group in range(max_groups):
        if text_pos >= text_ids.shape[-1]:
            group_ids = torch.full((1, nt), text_pad_id, dtype=torch.long)
        else:
            group_ids = text_ids[:, text_pos : text_pos + nt]
        group_ids = group_ids.to(device)
        text_out, cache = (
            model._backbone_one_step(  # noqa: leading-underscore  # reference oracle method
                model.backbone_llm.model.embed_tokens(group_ids), cache=cache
            )
        )
        for position in range(nt):
            fed = int(group_ids[0, position])
            steps.append(
                ("text", fed, text_out[0, position].float().cpu(), None, False)
            )
        text_pos += nt
        # The flow's first cond row of the group is the text forward's last
        # position (the reference reassigns backbone_out to the text output).
        backbone_out = text_out
        for sub in range(na):
            backbone_cond = torch.cat([backbone_cond, backbone_out[:, -1:]], dim=1)
            patch = model._flow_one_step(  # noqa: leading-underscore  # reference oracle method
                backbone_cond=backbone_cond,
                latents_gen=latents_gen,
                t_span=t_span,
                inference_cfg=1.2,
            )
            patch = patch.float().cpu()
            patches.append(patch)
            latents_gen = torch.cat([latents_gen, patch.to(device)], dim=1)
            audio_step += 1
            # Record the flow on the LAST text step of the group.
            kind, fed, hidden, _p, _s = steps[-1]
            steps[-1] = (kind, fed, hidden, patch, False)
            patch_out, cache = (
                model._backbone_one_step(  # noqa: leading-underscore  # reference oracle method
                    model.patch_encoder.forward(patch.to(device)), cache=cache
                )
            )
            backbone_out = patch_out
            stop = False
            if sub == na - 1:
                stop_score = float(
                    torch.sigmoid(model.stop_head(patch_out[:, -1]).squeeze(-1)).item()
                )
                stop = stop_score >= 0.5 and audio_step >= 6
            steps.append(
                ("patch", len(patches) - 1, patch_out[0, -1].float().cpu(), None, stop)
            )
            if stop:
                stopped = True
                break
        if stopped:
            break
    return steps, patches


@requires_all
@pytest.mark.accelerator
def test_speaker_dual_stream_machine_matches_reference(
    device, reference_sources
) -> None:
    from transformers import AutoTokenizer

    from sglang_omni.models.firered_tts3.config import load_checkpoint_hf_config
    from sglang_omni.models.firered_tts3.prompt_utils import (
        encode_speaker_dual_stream_prompt,
    )

    config = load_checkpoint_hf_config(str(SPEAKER_CHECKPOINT))
    assert config is not None
    model = build_reference_model(device)
    tokenizer = reference_tokenizer()
    ours_tokenizer = AutoTokenizer.from_pretrained(
        str(SPEAKER_CHECKPOINT), local_files_only=True
    )
    prompt_ids, text_ids, text_pad_id = encode_speaker_dual_stream_prompt(
        ours_tokenizer, text=TEXT, speaker_token="<|f531_speaker|>"
    )
    # The reference prefix ids must equal ours (byte-parity already pinned by
    # the prompt-utils goldens; assert the id-level agreement here).
    from diandiantts.utils.chatml import TEXT_PAD_ID

    assert text_pad_id == TEXT_PAD_ID
    prefix = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)
    reference_text = torch.tensor(tokenizer.encode(TEXT), dtype=torch.long).unsqueeze(0)
    assert text_ids == [int(t) for t in reference_text[0]]

    steps, ref_patches = reference_stream_trajectory(
        model, prefix, reference_text, text_pad_id, device
    )
    assert steps[-1][4] is True  # the reference stopped inside the window

    runner = make_runner(SpeakerSidePath(config, device))
    data = FireRedTTS3SGLangRequestData(
        state=FireRedTTS3State(
            variant="speaker",
            input_ids=prompt_ids,
            dual_stream=True,
            text_stream_ids=text_ids,
            text_pad_id=text_pad_id,
            stop_threshold=0.5,
            min_patches=6,
            n_timesteps=10,
            inference_cfg=1.2,
        ),
        dual_stream=True,
        interleave_ratio=(2, 1),
    )
    data.flow_state = FlowState(
        generator=torch.Generator(device=device).manual_seed(1234),
        prompt_patch_count=0,
        spk_cond_dit=None,
        backbone_cond=torch.zeros(1, 2, 2048, device=device),
        latents_gen=torch.zeros(1, 8, 64, device=device),
        patch_count=0,
    )
    data.prefill_len = len(prompt_ids)
    data.req = SimpleNamespace(finished_reason=None, output_ids=[], extend_range=None)
    data.probe = ProfilingProbe(enabled=False)

    # Prefill tail: the machine's first feed is the first text token.
    assert runner.next_text_feed(data) == text_ids[0]

    for index, (kind, fed, hidden, patch, stopped) in enumerate(steps):
        result = SimpleNamespace(
            next_token_ids=torch.zeros(1, dtype=torch.long),
            logits_output=SimpleNamespace(hidden_states=hidden.to(device).unsqueeze(0)),
        )
        runner.post_decode(
            result, None, None, [SimpleNamespace(request_id="rid", data=data)]
        )
        next_token = int(result.next_token_ids[0])
        if patch is not None:
            # The flow fired on this step: the patch must be bit-exact.
            assert torch.equal(
                data.latent_patches[-1].cpu(), patch
            ), f"patch at step {index} diverged"
        if stopped:
            assert (
                data.req.finished_reason is not None
            ), "the runner did not stop on the reference's stop step"
            break
        # The next feed must match the reference's next recorded feed.
        if index + 1 < len(steps):
            next_kind, next_fed = steps[index + 1][0], steps[index + 1][1]
            if next_kind == "text":
                assert (
                    next_token == next_fed
                ), f"step {index}: next text feed {next_token} != {next_fed}"
            else:
                assert (
                    next_token == CONTROL_TOKEN_ID
                ), f"step {index}: expected a patch feed next"
    assert len(data.latent_patches) == len(ref_patches)
