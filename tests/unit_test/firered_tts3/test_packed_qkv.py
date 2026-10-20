# SPDX-License-Identifier: Apache-2.0
"""Packed QKV equivalence, capture-time checks, and lazy resource allocation."""
from __future__ import annotations

import pytest
import torch

import sglang_omni.models.firered_tts3.ffngemm as gemm_module
from sglang_omni.models.firered_tts3.dit import DiT
from sglang_omni.models.firered_tts3.packed_qkv import PackedQkvModel


@pytest.fixture
def packed():
    source = DiT(
        in_channels=20,
        out_channels=4,
        hidden_size=32,
        num_heads=4,
        mlp_ratio=3,
        depth=2,
    ).eval()
    return PackedQkvModel(source)


def test_private_model_keeps_source_and_forward_unchanged():
    source = DiT(
        in_channels=20,
        out_channels=4,
        hidden_size=32,
        num_heads=4,
        mlp_ratio=3,
        depth=2,
    ).eval()
    values = {name: tensor.clone() for name, tensor in source.state_dict().items()}
    parameters = dict(source.named_parameters())
    buffers = dict(source.named_buffers())
    rng = torch.get_rng_state().clone()
    packed = PackedQkvModel(source)
    assert torch.equal(torch.get_rng_state(), rng)
    assert packed.source is source and packed.model is not source
    assert all(
        packed.model.get_parameter(name) is tensor
        for name, tensor in parameters.items()
    )
    assert all(
        packed.model.get_buffer(name) is tensor for name, tensor in buffers.items()
    )
    assert all(
        torch.equal(source.state_dict()[name], tensor)
        for name, tensor in values.items()
    )
    x, time = torch.randn(2, 12, 20), torch.ones(2, 1, 1)
    with torch.inference_mode():
        expected = source(x=x, t=time)
        actual = packed.model(x=x, t=time)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    packed.validate_weights()


@pytest.mark.parametrize("parameter", ["weight", "bias"])
@pytest.mark.parametrize("mutation", ["inplace", "replace", "same_storage"])
def test_source_projection_changes_are_rejected_before_capture(
    packed: PackedQkvModel, parameter: str, mutation: str
) -> None:
    layer = packed.source.blocks[0].attn.to_q
    original = getattr(layer, parameter)
    with torch.no_grad():
        if mutation == "inplace":
            original.add_(0.125)
        elif mutation == "replace":
            setattr(layer, parameter, torch.nn.Parameter(original.clone()))
        else:
            setattr(layer, parameter, torch.nn.Parameter(original.detach()))
    with pytest.raises(RuntimeError, match="projection changed"):
        packed.validate_weights()


@pytest.mark.parametrize("mutation", ["pointer", "shape", "stride", "dtype"])
def test_projection_metadata_changes_are_rejected_before_capture(
    packed: PackedQkvModel, mutation: str
) -> None:
    weight = packed.source.blocks[0].attn.to_q.weight
    with torch.no_grad():
        if mutation == "pointer":
            weight.data = weight.detach().clone()
        elif mutation == "shape":
            weight.data = weight.detach().view(16, 64)
        elif mutation == "stride":
            weight.data = weight.detach().t()
        else:
            weight.data = weight.detach().double()
    with pytest.raises(RuntimeError, match="projection changed"):
        packed.validate_weights()


@pytest.fixture
def sm120_dit() -> DiT:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("The selected DiT GEMMs require SM120")
    with torch.device("cuda"):
        source = (
            DiT(
                in_channels=20,
                out_channels=4,
                hidden_size=1024,
                num_heads=16,
                mlp_ratio=3,
                depth=11,
            )
            .to(dtype=torch.bfloat16)
            .eval()
            .requires_grad_(False)
        )
    with torch.no_grad():
        source.blocks[0].mlp.ff[0][0].weight.zero_()
        source.blocks[0].mlp.ff[0][0].bias.fill_(0.125)
    return source


@pytest.mark.accelerator
@pytest.mark.parametrize("failure", ["packing", "compile_wrapper"])
def test_packed_preparation_failure_preserves_small_ffn(
    sm120_dit: DiT, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    mode = gemm_module.FfnGemmMode(sm120_dit)
    assert mode.enabled and all(mode.supports(rows) for rows in (24, 48, 96))
    assert not mode.supports(192) and not mode.supports(384)
    assert mode.packed_qkv is None and mode.packed_compiled is None
    assert not mode.packed_weights

    def fail(*args, **kwargs):
        raise RuntimeError("Injected packed preparation failure")

    with monkeypatch.context() as patch:
        if failure == "packing":
            patch.setattr(gemm_module, "PackedQkvModel", fail)
        else:
            patch.setattr(torch, "compile", fail)
        with pytest.raises(RuntimeError, match="Injected packed preparation"):
            mode.prepare_packed(
                lambda graph, inputs: graph.forward, next(sm120_dit.parameters()).device
            )
    assert mode.packed_qkv is None and mode.packed_compiled is None
    assert not mode.packed_weights and not mode.supports(192)
    assert all(mode.supports(rows) for rows in (24, 48, 96))
    mode.validate_weights()
    layer = sm120_dit.blocks[0].mlp.ff[0][0]
    inputs = torch.randn(24, 1024, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(24, 3072, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), mode:
        torch.addmm(layer.bias, inputs, layer.weight.t(), out=output)
    assert mode.hits == 1 and mode.misses == 0
    assert torch.equal(output, torch.full_like(output, 0.125))


@pytest.mark.accelerator
def test_large_qkv_is_prepared_once_on_demand(
    sm120_dit: DiT, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    original = gemm_module.PackedQkvModel

    def count_packing(source: DiT) -> PackedQkvModel:
        nonlocal calls
        calls += 1
        return original(source)

    monkeypatch.setattr(gemm_module, "PackedQkvModel", count_packing)
    mode = gemm_module.FfnGemmMode(sm120_dit)
    assert calls == 0 and mode.supports(24) and not mode.supports(192)

    def backend(graph, inputs):
        return graph.forward

    mode.prepare_packed(backend, next(sm120_dit.parameters()).device)
    assert calls == 1 and mode.supports(192) and mode.supports(384)
    assert mode.supports(24) and mode.supports(96)
    mode.validate_weights()
    mode.prepare_packed(backend, next(sm120_dit.parameters()).device)
    assert calls == 1
    layer = mode.packed_qkv.model.blocks[0].attn
    inputs = torch.randn(2, 12, 1024, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        actual = layer.project_qkv(inputs)
        expected = tuple(
            projection(inputs) for projection in (layer.to_q, layer.to_k, layer.to_v)
        )
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, atol=0.02, rtol=0.01)
