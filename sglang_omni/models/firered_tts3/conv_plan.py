# SPDX-License-Identifier: Apache-2.0
"""Controlled cuDNN convolution for the validated SM120 BF16 shape."""

from __future__ import annotations

import weakref

import torch

# note (Codex): verified against the original NCHW plan on SM120 with cuDNN 9.20.
CUDNN_VERSION = 92000
FRONTEND_VERSION = "1.28.0"
ENGINE_ID = 23
TENSOR_ALIGNMENT = 16  # note (Codex): frontend 1.28 tensor descriptor default.
INPUT_SHAPE = (8, 1024, 1, 12)
INPUT_STRIDE = (12288, 1, 12288, 1024)
WEIGHT_SHAPE = (1024, 1024, 1, 3)
WEIGHT_STRIDE = (3072, 1, 3072, 1024)


def validate_tensors(inputs: torch.Tensor, weight: torch.Tensor) -> None:
    if (
        inputs.device.type != "cuda"
        or weight.device != inputs.device
        or inputs.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or tuple(inputs.shape) != INPUT_SHAPE
        or tuple(inputs.stride()) != INPUT_STRIDE
        or tuple(weight.shape) != WEIGHT_SHAPE
        or tuple(weight.stride()) != WEIGHT_STRIDE
        or inputs.data_ptr() % TENSOR_ALIGNMENT
        or weight.data_ptr() % TENSOR_ALIGNMENT
    ):
        raise RuntimeError("Unsupported controlled convolution tensor metadata")
    else:
        pass


class ControlledConvPlan:
    """Own one plan and workspace; calls and captured replays must be serialized."""

    def __init__(self, inputs: torch.Tensor, weight: torch.Tensor) -> None:
        validate_tensors(inputs, weight)
        with torch.cuda.device(inputs.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Controlled convolution must be prepared before capture"
                )
            else:
                pass
        if (
            not torch.backends.cudnn.enabled
            or torch.backends.cudnn.deterministic
            or torch.are_deterministic_algorithms_enabled()
        ):
            raise RuntimeError(
                "Controlled convolution requires default cuDNN determinism settings"
            )
        else:
            pass
        if (
            torch.cuda.get_device_capability(inputs.device) != (12, 0)
            or torch.backends.cudnn.version() != CUDNN_VERSION
        ):
            raise RuntimeError("Controlled convolution requires SM120 and cuDNN 9.20")
        else:
            pass
        try:
            import cudnn
        except ImportError as error:
            raise RuntimeError(
                f"Controlled convolution requires nvidia-cudnn-frontend=={FRONTEND_VERSION}"
            ) from error
        if (
            cudnn.__version__ != FRONTEND_VERSION
            or cudnn.backend_version() != CUDNN_VERSION
        ):
            raise RuntimeError(
                f"Controlled convolution requires frontend {FRONTEND_VERSION} and cuDNN 9.20"
            )
        else:
            pass
        self.frontend = cudnn
        self.device = inputs.device
        with torch.cuda.device(self.device):
            self.handle = cudnn.create_handle()
            self.release_handle = weakref.finalize(
                self, cudnn.destroy_handle, self.handle
            )
            self.graph = cudnn.pygraph(
                handle=self.handle,
                io_data_type=cudnn.data_type.BFLOAT16,
                intermediate_data_type=cudnn.data_type.FLOAT,
                compute_data_type=cudnn.data_type.FLOAT,
            )
            self.x = self.graph.tensor(dim=INPUT_SHAPE, stride=INPUT_STRIDE, name="x")
            self.w = self.graph.tensor(dim=WEIGHT_SHAPE, stride=WEIGHT_STRIDE, name="w")
            self.y = self.graph.conv_fprop(
                image=self.x,
                weight=self.w,
                padding=[0, 1],
                stride=[1, 1],
                dilation=[1, 1],
            )
            self.y.set_output(True).set_dim(INPUT_SHAPE).set_stride(INPUT_STRIDE)
            self.y.set_data_type(cudnn.data_type.BFLOAT16)
            knobs = {
                cudnn.knob_type.TILE_SIZE: 8,
                cudnn.knob_type.SPLIT_K_BUF: 2,
                cudnn.knob_type.TILEK: 1,
                cudnn.knob_type.STAGES: 4,
                cudnn.knob_type.REDUCTION_MODE: 0,
                cudnn.knob_type.SPLIT_K_SLC: 342,
                cudnn.knob_type.IDX_MODE: 1,
                cudnn.knob_type.SPECFILT: 0,
            }
            self.graph.create_execution_plan(ENGINE_ID, knobs)
            self.graph.select_plan(0)
            self.graph.check_support()
            self.graph.build_plan_at_index(0)
            self.workspace = torch.empty(
                self.graph.get_workspace_size(), dtype=torch.uint8, device=self.device
            )

    def __call__(self, inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        validate_tensors(inputs, weight)
        if inputs.device != self.device:
            raise RuntimeError("Controlled convolution plan belongs to another device")
        else:
            pass
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            self.frontend.set_stream(self.handle, stream.cuda_stream)
            inputs.record_stream(stream)
            weight.record_stream(stream)
            self.workspace.record_stream(stream)
            output = torch.empty_like(inputs, memory_format=torch.preserve_format)
            self.graph.execute(
                {self.x: inputs, self.w: weight, self.y: output},
                self.workspace,
                handle=self.handle,
            )
            return output
