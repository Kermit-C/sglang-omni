# SPDX-License-Identifier: Apache-2.0
"""Checkpoint routing updates actual parameters before execution becomes immutable."""

from __future__ import annotations

from typing import Literal

import pytest
import torch

import sglang_omni.models.firered_tts3.engine_builder as engine_builder_module
from sglang_omni.models.firered_tts3.engine_builder import FireRedTTS3EngineBuilder
from sglang_omni.models.firered_tts3.sglang_model import FireRedTTS3SGLangModel


@pytest.fixture
def loadable_model() -> FireRedTTS3SGLangModel:
    class Backbone(torch.nn.Linear):
        def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
            with torch.no_grad():
                for name, tensor in weights:
                    self.get_parameter(name).copy_(tensor)

    model = FireRedTTS3SGLangModel.__new__(FireRedTTS3SGLangModel)
    torch.nn.Module.__init__(model)
    model.backbone = Backbone(2, 2, bias=False)
    for name in ("patch_encoder", "dit", "dit_head", "stop_head"):
        setattr(model, name, torch.nn.Linear(2, 2, bias=False))
    model.spk_proj_llm = model.spk_proj_dit = None
    return model


def test_weight_loading_finishes_before_execution_is_locked(
    loadable_model: FireRedTTS3SGLangModel,
) -> None:
    model = loadable_model
    assert model.weight_update_blocked_reason is None
    initial = torch.ones_like(model.dit.weight)
    assert model.load_weights([("dit.weight", initial)]) == {"dit.weight"}
    assert torch.equal(model.dit.weight, initial)
    model.weights_locked = True
    updates = iter([("dit.weight", torch.zeros_like(initial))])
    with pytest.raises(RuntimeError, match="restart the worker"):
        model.load_weights(updates)
    assert next(updates)[0] == "dit.weight"
    assert torch.equal(model.dit.weight, initial)


@pytest.mark.parametrize("compile_fails", [False, True])
def test_compile_locks_updates_before_creating_execution(
    loadable_model: FireRedTTS3SGLangModel,
    monkeypatch: pytest.MonkeyPatch,
    compile_fails: bool,
) -> None:
    model = loadable_model
    builder = FireRedTTS3EngineBuilder(flow_cuda_graph=True)

    def prepare(
        dit: torch.nn.Module, mode: str
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        assert dit is model.dit and mode == builder.flow_mode
        with pytest.raises(RuntimeError, match="restart the worker"):
            model.load_weights([])
        if compile_fails:
            raise RuntimeError("compile failed")
        else:
            return dit, dit

    monkeypatch.setattr(engine_builder_module, "prepare_flow_dit", prepare)
    if compile_fails:
        with pytest.raises(RuntimeError, match="compile failed"):
            builder.compile_model(model, None)
    else:
        builder.compile_model(model, None)
    assert model.weight_update_blocked_reason is not None


def test_disabled_flow_graph_does_not_lock_weight_loading(
    loadable_model: FireRedTTS3SGLangModel,
) -> None:
    builder = FireRedTTS3EngineBuilder(flow_cuda_graph=False)
    builder.compile_model(loadable_model, None)
    assert loadable_model.weight_update_blocked_reason is None
    replacement = torch.zeros_like(loadable_model.dit.weight)
    loadable_model.load_weights([("dit.weight", replacement)])
    assert torch.equal(loadable_model.dit.weight, replacement)


@pytest.mark.parametrize(
    "backbone_prefix", ["backbone_llm.", "backbone_llm.model."], ids=["base", "chatml"]
)
def test_checkpoint_routing_loads_backbone_and_side_path(
    loadable_model: FireRedTTS3SGLangModel,
    backbone_prefix: Literal["backbone_llm.", "backbone_llm.model."],
) -> None:
    weights = [
        (f"{backbone_prefix}weight", torch.full((2, 2), 2.0)),
        ("dit.weight", torch.full((2, 2), 3.0, dtype=torch.bfloat16)),
        ("stop_head.weight", torch.full((2, 2), 4.0)),
        ("backbone_llm.lm_head.weight", torch.full((2, 2), -1.0)),
    ]
    loaded = loadable_model.load_weights(weights)
    assert loaded == {"dit.weight", "stop_head.weight"}
    inputs = torch.ones(1, 2)
    assert torch.equal(loadable_model.backbone(inputs), torch.full((1, 2), 4.0))
    assert torch.equal(loadable_model.dit(inputs), torch.full((1, 2), 6.0))
    assert torch.equal(loadable_model.stop_head(inputs), torch.full((1, 2), 8.0))
    assert loadable_model.dit.weight.dtype == torch.float32
    with pytest.raises(ValueError, match="unexpected FireRedTTS3 checkpoint key"):
        loadable_model.load_weights([("unknown.weight", torch.ones(2, 2))])
