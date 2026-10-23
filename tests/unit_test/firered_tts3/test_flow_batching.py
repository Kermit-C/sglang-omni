# SPDX-License-Identifier: Apache-2.0
"""Real CPU flow steps preserve request ordering, feedback, and stop ownership."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.models.firered_tts3.model_runner import (
    FireRedTTS3ModelRunner,
    FlowState,
)
from sglang_omni.models.firered_tts3.patch_encoder import PatchEncoder
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.models.firered_tts3.request_builders import (
    CONTROL_TOKEN_ID,
    FireRedTTS3SGLangRequestData,
)
from tests.unit_test.fixtures.firered_tts3 import tiny_cpu_dit


class SmallFlowModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(12)
        self.dit = tiny_cpu_dit()
        self.patch_encoder = PatchEncoder(64, 8, hidden_size=16, num_heads=2, depth=1)
        self.dit_head = torch.nn.Linear(8, 32)
        self.stop_head = torch.nn.Linear(8, 1)
        self.patch_size = 4
        self.history_patches = 2
        self.history_length = 8
        self.redae_dim = 64
        self.llm_hidden_size = 8


def make_runner(model: SmallFlowModel, *, batching: bool) -> FireRedTTS3ModelRunner:
    runner = FireRedTTS3ModelRunner.__new__(FireRedTTS3ModelRunner)
    runner.model = model.eval()
    runner.async_enabled = False
    runner.flow_batching = batching
    runner.flow_graphs = None
    runner.t_span_cache = {}
    runner.active_requests = {}
    return runner


def make_request(
    index: int, timesteps: int = 3, *, stopped: bool = False
) -> SimpleNamespace:
    data = FireRedTTS3SGLangRequestData(
        state=FireRedTTS3State(
            n_timesteps=timesteps,
            inference_cfg=1.2 + index,
            min_patches=6,
            stop_threshold=0.5 if stopped else 1.1,
        ),
    )
    data.probe = ProfilingProbe(enabled=False)
    data.flow_state = FlowState(
        generator=torch.Generator().manual_seed(index),
        prompt_patch_count=0,
        spk_cond_dit=None,
        backbone_cond=torch.zeros(1, 2, 8),
        latents_gen=torch.zeros(1, 8, 64),
        patch_count=6 if stopped else 0,
    )
    data.req = SimpleNamespace(finished_reason=None, is_retracted=False)
    data.req.finished = lambda: data.req.finished_reason is not None
    return SimpleNamespace(request_id=f"row-{index}", data=data)


def make_result(hidden: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        next_token_ids=torch.full((len(hidden),), 123),
        logits_output=SimpleNamespace(hidden_states=hidden),
    )


@pytest.mark.parametrize(
    "timesteps,stop_row", [([3], None), ([3, 3, 3], 1), ([3, 2, 3], None)]
)
def test_batched_decode_matches_serial_requests(
    timesteps: list[int], stop_row: int | None
) -> None:
    model = SmallFlowModel()
    generator = torch.Generator().manual_seed(88)
    hidden_steps = [
        torch.randn(len(timesteps), 8, generator=generator) for _ in range(2)
    ]
    if stop_row is not None:
        weight = model.stop_head.weight[0].detach()
        hidden_steps[0][stop_row] = weight * (
            (8.0 - model.stop_head.bias.item()) / weight.square().sum()
        )
    else:
        pass
    outcomes: list[list[SimpleNamespace]] = []
    forward_counts = []
    for batching in (False, True):
        runner = make_runner(model, batching=batching)
        requests = [
            make_request(index, steps, stopped=index == stop_row)
            for index, steps in enumerate(timesteps)
        ]
        with patch.object(model.dit, "forward", wraps=model.dit.forward) as forward:
            for hidden in hidden_steps:
                active = [
                    request for request in requests if not request.data.req.finished()
                ]
                indices = [requests.index(request) for request in active]
                result = make_result(hidden[indices])
                runner.post_decode(result, None, None, active)
                assert result.next_token_ids.tolist() == [CONTROL_TOKEN_ID] * len(
                    active
                )
            forward_counts.append(forward.call_count)
        outcomes.append(requests)
    for index, (serial, batched) in enumerate(zip(*outcomes, strict=True)):
        assert len(batched.data.latent_patches) == (0 if index == stop_row else 2)
        assert len(batched.data.pending_feedback_queue) == len(
            batched.data.latent_patches
        )
        assert batched.data.req.finished() is (index == stop_row)
        for actual, expected in zip(
            batched.data.latent_patches, serial.data.latent_patches, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-3)
        for actual, expected in zip(
            batched.data.pending_feedback_queue,
            serial.data.pending_feedback_queue,
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-3)
    assert forward_counts[1] <= forward_counts[0]
    if len(timesteps) > 1:
        assert forward_counts[1] < forward_counts[0]
    else:
        assert torch.equal(
            outcomes[0][0].data.latent_patches[0], outcomes[1][0].data.latent_patches[0]
        )


@pytest.mark.parametrize("status", ["active", "retracted", "finished"])
def test_async_stop_resolves_only_the_owned_request(status: str) -> None:
    runner = make_runner(SmallFlowModel(), batching=True)
    request = make_request(0, stopped=True)
    weight = runner.model.stop_head.weight[0].detach()
    hidden = (
        weight * ((8.0 - runner.model.stop_head.bias.item()) / weight.square().sum())
    ).unsqueeze(0)
    result = make_result(hidden)
    runner.post_decode_launch(result, None, [request])
    assert request.data.stop_pending and request.data.req.finished_reason is None
    assert request.data.latent_patches == []
    request.data.req.is_retracted = status == "retracted"
    previous_finish = FINISH_MATCHED_TOKEN(99) if status == "finished" else None
    request.data.req.finished_reason = previous_finish
    runner.post_decode_resolve(None, result, None, None, [request])
    assert not request.data.stop_pending
    if status == "active":
        assert isinstance(request.data.req.finished_reason, FINISH_MATCHED_TOKEN)
    else:
        assert request.data.req.finished_reason is previous_finish
    runner.active_requests[request.request_id] = request.data
    runner.on_request_finished(request.request_id, request.data)
    assert runner.active_requests == {} and request.data.flow_state is None
