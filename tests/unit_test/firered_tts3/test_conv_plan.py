# SPDX-License-Identifier: Apache-2.0
"""Controlled convolution plan contracts and independent CUDA graph resources."""

from __future__ import annotations

import gc
import itertools
import sys
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from sglang_omni.models.firered_tts3.conv_plan import ControlledConvPlan


@pytest.fixture
def fake_frontend(monkeypatch: pytest.MonkeyPatch):
    graphs = []
    handles = itertools.count(1)
    frontend = SimpleNamespace(
        __version__="1.28.0",
        backend_version=lambda: 92000,
        data_type=SimpleNamespace(BFLOAT16="bf16", FLOAT="fp32"),
        knob_type=SimpleNamespace(
            **{
                name: name
                for name in (
                    "TILE_SIZE",
                    "SPLIT_K_BUF",
                    "TILEK",
                    "STAGES",
                    "REDUCTION_MODE",
                    "SPLIT_K_SLC",
                    "IDX_MODE",
                    "SPECFILT",
                )
            }
        ),
        create_handle=Mock(side_effect=lambda: next(handles)),
        destroy_handle=Mock(),
        set_stream=Mock(),
        graphs=graphs,
    )

    def make_graph(**kwargs):
        graph = Mock()
        graph.configuration = kwargs
        graph.tensor.side_effect = [Mock(name="x"), Mock(name="w")]
        output = Mock(name="y")
        output.set_output.return_value = output
        output.set_dim.return_value = output
        output.set_stride.return_value = output
        graph.conv_fprop.return_value = output
        graph.get_workspace_size.return_value = 256
        graphs.append(graph)
        return graph

    frontend.pygraph = Mock(side_effect=make_graph)
    monkeypatch.setitem(sys.modules, "cudnn", frontend)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 92000)
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    current = SimpleNamespace(stream=SimpleNamespace(cuda_stream=17))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: current.stream)
    recorded = []
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda tensor, stream: recorded.append((tensor, stream)),
    )
    monkeypatch.setattr(FakeTensor, "data_ptr", lambda tensor: 4096)
    with FakeTensorMode():
        inputs = torch.empty(
            (8, 1024, 1, 12),
            dtype=torch.bfloat16,
            device="cuda:0",
            memory_format=torch.channels_last,
        )
        weight = torch.empty(
            (1024, 1024, 1, 3),
            dtype=torch.bfloat16,
            device="cuda:0",
            memory_format=torch.channels_last,
        )
        yield SimpleNamespace(
            frontend=frontend,
            inputs=inputs,
            weight=weight,
            current=current,
            recorded=recorded,
        )


def test_controlled_plan_requests_validated_math_and_all_knobs(fake_frontend) -> None:
    env = fake_frontend
    plan = ControlledConvPlan(env.inputs, env.weight)
    graph = env.frontend.graphs[0]
    assert graph.configuration == {
        "handle": plan.handle,
        "io_data_type": "bf16",
        "intermediate_data_type": "fp32",
        "compute_data_type": "fp32",
    }
    graph.conv_fprop.assert_called_once_with(
        image=plan.x, weight=plan.w, padding=[0, 1], stride=[1, 1], dilation=[1, 1]
    )
    graph.create_execution_plan.assert_called_once_with(
        23,
        {
            "TILE_SIZE": 8,
            "SPLIT_K_BUF": 2,
            "TILEK": 1,
            "STAGES": 4,
            "REDUCTION_MODE": 0,
            "SPLIT_K_SLC": 342,
            "IDX_MODE": 1,
            "SPECFILT": 0,
        },
    )
    graph.select_plan.assert_called_once_with(0)
    graph.check_support.assert_called_once_with()
    graph.build_plan_at_index.assert_called_once_with(0)
    assert plan.workspace.dtype == torch.uint8 and plan.workspace.numel() == 256
    assert plan.workspace.device == env.inputs.device


@pytest.mark.parametrize(
    "mismatch",
    [
        "input_shape",
        "input_layout",
        "input_dtype",
        "input_alignment",
        "weight_device",
    ],
)
def test_controlled_plan_rejects_unsupported_tensors_before_backend(
    fake_frontend, mismatch: str
) -> None:
    env = fake_frontend
    inputs, weight = env.inputs, env.weight
    if mismatch == "input_shape":
        inputs = torch.empty(
            (4, 1024, 1, 12),
            device="cuda:0",
            dtype=torch.bfloat16,
            memory_format=torch.channels_last,
        )
    elif mismatch == "input_layout":
        inputs = torch.empty(inputs.shape, device="cuda:0", dtype=torch.bfloat16)
    elif mismatch == "input_dtype":
        inputs = inputs.float()
    elif mismatch == "input_alignment":
        inputs.data_ptr = lambda: 4098
    else:
        weight = weight.to("cuda:1")
    with pytest.raises(RuntimeError, match="tensor metadata"):
        ControlledConvPlan(inputs, weight)
    env.frontend.create_handle.assert_not_called()


@pytest.mark.parametrize(
    "mismatch",
    [
        "capture",
        "device_capability",
        "torch_cudnn",
        "frontend_version",
        "missing_frontend",
        "cudnn_deterministic",
    ],
)
def test_controlled_plan_rejects_unvalidated_backend(
    fake_frontend, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    env = fake_frontend
    if mismatch == "capture":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    elif mismatch == "device_capability":
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    elif mismatch == "torch_cudnn":
        monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 91900)
    elif mismatch == "frontend_version":
        env.frontend.__version__ = "1.29.0"
    elif mismatch == "missing_frontend":
        monkeypatch.setitem(sys.modules, "cudnn", None)
    else:
        monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    with pytest.raises(RuntimeError, match="Controlled convolution"):
        ControlledConvPlan(env.inputs, env.weight)
    env.frontend.create_handle.assert_not_called()


def test_controlled_plan_owns_independent_resources_and_returns_fresh_outputs(
    fake_frontend,
) -> None:
    env = fake_frontend
    first = ControlledConvPlan(env.inputs, env.weight)
    second = ControlledConvPlan(env.inputs, env.weight)
    assert first.handle != second.handle and first.workspace is not second.workspace
    output = first(env.inputs, env.weight)
    first_stream = env.current.stream
    env.current.stream = SimpleNamespace(cuda_stream=29)
    second_output = second(env.inputs, env.weight)
    repeated_output = second(env.inputs, env.weight)
    assert output is not env.inputs and second_output is not output
    assert repeated_output is not second_output
    assert output.shape == env.inputs.shape and output.stride() == env.inputs.stride()
    assert output.dtype == env.inputs.dtype and output.device == env.inputs.device
    assert env.frontend.set_stream.call_args_list[0].args == (first.handle, 17)
    assert env.frontend.set_stream.call_args_list[1].args == (second.handle, 29)
    assert any(
        tensor is first.workspace and stream is first_stream
        for tensor, stream in env.recorded
    )
    assert any(
        tensor is second.workspace and stream is env.current.stream
        for tensor, stream in env.recorded
    )
    first.graph.execute.assert_called_once_with(
        {first.x: env.inputs, first.w: env.weight, first.y: output},
        first.workspace,
        handle=first.handle,
    )
    first_handle, second_handle = first.handle, second.handle
    first_ref = weakref.ref(first)
    del first
    gc.collect()
    assert first_ref() is None
    env.frontend.destroy_handle.assert_called_once_with(first_handle)
    assert second.handle == second_handle


@pytest.mark.parametrize("mismatch", ["layout", "device"])
def test_controlled_plan_revalidates_each_call(fake_frontend, mismatch: str) -> None:
    env = fake_frontend
    plan = ControlledConvPlan(env.inputs, env.weight)
    if mismatch == "layout":
        inputs = torch.empty(env.inputs.shape, device="cuda:0", dtype=torch.bfloat16)
        weight = env.weight
    else:
        inputs, weight = env.inputs.to("cuda:1"), env.weight.to("cuda:1")
    with pytest.raises(RuntimeError, match="tensor metadata|another device"):
        plan(inputs, weight)
    plan.graph.execute.assert_not_called()


def test_controlled_plan_support_failure_releases_handle(
    fake_frontend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = fake_frontend
    make_graph = env.frontend.pygraph.side_effect

    def unsupported_graph(**kwargs):
        graph = make_graph(**kwargs)

        def reject_plan():
            raise RuntimeError("unsupported plan probe")

        graph.check_support.side_effect = reject_plan
        return graph

    monkeypatch.setattr(env.frontend.pygraph, "side_effect", unsupported_graph)
    with pytest.raises(RuntimeError, match="unsupported plan probe"):
        ControlledConvPlan(env.inputs, env.weight)
    gc.collect()
    env.frontend.graphs[0].execute.assert_not_called()
    env.frontend.destroy_handle.assert_called_once_with(1)


def test_controlled_plan_execution_failure_is_not_hidden(fake_frontend) -> None:
    env = fake_frontend
    plan = ControlledConvPlan(env.inputs, env.weight)
    plan.graph.execute.side_effect = RuntimeError("execution probe")
    with pytest.raises(RuntimeError, match="execution probe"):
        plan(env.inputs, env.weight)


@pytest.mark.accelerator
def test_controlled_plans_replay_on_independent_streams() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    frontend = pytest.importorskip("cudnn")
    device = torch.device("cuda", torch.cuda.current_device())
    if (
        torch.cuda.get_device_capability(device) != (12, 0)
        or torch.backends.cudnn.version() != 92000
        or frontend.backend_version() != 92000
        or frontend.__version__ != "1.28.0"
    ):
        pytest.skip("The controlled plan requires its validated CUDA stack")
    generator = torch.Generator(device=device).manual_seed(812)
    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    retained = []
    references = []
    for stream in streams:
        inputs = torch.randn(
            (8, 1024, 12), device=device, dtype=torch.bfloat16, generator=generator
        )
        weight = (
            torch.randn(
                (1024, 1024, 3),
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            * 0.03125
        )
        expected = F.conv1d(inputs, weight, padding=1)
        packed_input = inputs.unsqueeze(2).contiguous(memory_format=torch.channels_last)
        packed_weight = weight.unsqueeze(2).contiguous(
            memory_format=torch.channels_last
        )
        plan = ControlledConvPlan(packed_input, packed_weight)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                actual = plan(packed_input, packed_weight)
        torch.cuda.synchronize(device)
        assert torch.equal(actual.squeeze(2).contiguous(), expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = plan(packed_input, packed_weight)
        retained.append((graph, plan, packed_input, packed_weight, captured))
        references.append(expected)
    assert retained[0][1].workspace.data_ptr() != retained[1][1].workspace.data_ptr()
    plan_refs = [weakref.ref(item[1]) for item in retained]
    workspace_refs = [weakref.ref(item[1].workspace) for item in retained]
    del plan, graph, packed_input, packed_weight, captured
    gc.collect()
    assert all(reference() is not None for reference in plan_refs + workspace_refs)
    for _ in range(8):
        for stream, item in zip(streams, retained):
            with torch.cuda.stream(stream):
                item[0].replay()
    torch.cuda.synchronize(device)
    for item, expected in zip(retained, references):
        assert torch.equal(item[4].squeeze(2).contiguous(), expected)
    del item
    retained.clear()
    gc.collect()
    assert all(reference() is None for reference in plan_refs + workspace_refs)
