# SPDX-License-Identifier: Apache-2.0
"""Numerical acceptance gates for optimized flow outputs."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.firered_tts3.flow_precision import FlowPrecisionGate

pytestmark = pytest.mark.usefixtures("disable_flow_optimization_environment")


@pytest.mark.parametrize("source", ["constructor", "environment"])
def test_flow_precision_explicit_exact_checks_remain_byte_exact(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    monkeypatch.setenv("FIRERED_TTS3_FLOW_ATOL", "0")
    monkeypatch.setenv("FIRERED_TTS3_FLOW_RTOL", "0")
    gate = (
        FlowPrecisionGate()
        if source == "constructor"
        else FlowPrecisionGate.from_environment()
    )
    reference = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    exact = gate.compare(reference.t().contiguous().t(), reference)
    assert exact.accepted and exact.byte_exact
    assert not gate.compare(torch.tensor([0.0]), torch.tensor([-0.0])).accepted
    changed = torch.nextafter(reference, torch.full_like(reference, float("inf")))
    assert not gate.compare(changed, reference).accepted


def test_flow_precision_default_budget_accepts_only_small_roundoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FIRERED_TTS3_FLOW_ATOL", raising=False)
    monkeypatch.delenv("FIRERED_TTS3_FLOW_RTOL", raising=False)
    gate = FlowPrecisionGate.from_environment()
    reference = torch.tensor([[0.0, 1.0]])
    close = torch.tensor([[0.0005, 1.005]])
    assert gate.compare(close, reference).accepted
    assert not FlowPrecisionGate().compare(close, reference).accepted
    assert not gate.compare(torch.tensor([[0.002, 1.0]]), reference).accepted
    assert not gate.compare(torch.tensor([[0.0, 1.02]]), reference).accepted


@pytest.mark.parametrize(
    "variable", ["FIRERED_TTS3_FLOW_ATOL", "FIRERED_TTS3_FLOW_RTOL"]
)
@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf", "invalid"])
def test_flow_precision_rejects_invalid_environment(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match=variable):
        FlowPrecisionGate.from_environment()


@pytest.mark.parametrize("target", ["actual", "expected", "both"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_flow_precision_rejects_nonfinite_values(target: str, value: float) -> None:
    reference = torch.ones(2, 3)
    actual = reference.clone()
    if target in ("actual", "both"):
        actual[1, 2] = value
    if target in ("expected", "both"):
        reference[1, 2] = value
    result = FlowPrecisionGate(atol=100.0, rtol=100.0).compare(actual, reference)
    assert not result.accepted and not result.finite


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "device"])
def test_flow_precision_rejects_incompatible_tensors(mismatch: str) -> None:
    reference = torch.ones(2, 3)
    if mismatch == "shape":
        actual = reference.flatten()
    elif mismatch == "dtype":
        actual = reference.double()
    else:
        actual = torch.empty_like(reference, device="meta")
    result = FlowPrecisionGate(atol=100.0, rtol=100.0).compare(actual, reference)
    assert not result.accepted and not result.schema_valid


@pytest.mark.parametrize(
    "reference,actual,atol,rtol,accepted",
    [
        (0.0, 0.125, 0.125, 0.0, True),
        (0.0, 0.125 + 2**-26, 0.125, 0.0, False),
        (2.0, 2.25, 0.0, 0.125, True),
        (2.0, 2.25 + 2**-22, 0.0, 0.125, False),
        (-2.0, -2.25, 0.0, 0.125, True),
        (2.0, 2.375, 0.125, 0.125, True),
        (2.0, 2.375 + 2**-22, 0.125, 0.125, False),
        (0.0, 2**-24, 0.0, 100.0, False),
        (2**-24, 2**-23, 0.0, 0.5, False),
    ],
)
def test_flow_precision_elementwise_tolerance_boundaries(
    reference: float,
    actual: float,
    atol: float,
    rtol: float,
    accepted: bool,
) -> None:
    result = FlowPrecisionGate(atol=atol, rtol=rtol).compare(
        torch.tensor([actual]), torch.tensor([reference])
    )
    assert result.accepted == accepted
    assert result.out_of_bounds == int(not accepted)


def test_flow_precision_rejects_one_outlier_in_one_batch_row() -> None:
    reference = torch.ones(8, 4, 64)
    actual = reference.clone()
    actual[-1, -1, -1] += 0.25
    result = FlowPrecisionGate(atol=0.125, rtol=0.0).compare(actual, reference)
    assert not result.accepted and result.out_of_bounds == 1
    assert result.rmse < 0.125 and result.max_abs == 0.25
