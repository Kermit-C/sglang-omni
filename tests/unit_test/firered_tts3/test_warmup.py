# SPDX-License-Identifier: Apache-2.0
"""Post-init warmup contract tests.

Every GPU stage warms its cold-start paths at factory time, before the
stage publishes readiness: the reference bundle runs CAM++ + the RedAE
encoder on one real-shaped dummy clip (off the reference cache), the talker
runs the fp32 side-path modules on their hot shapes, and the vocoder runs
the fresh / cached+terminal / flush decode paths. These tests pin the
coverage and the warmup_on_init kill switch with stub modules (CPU-only).
"""

from __future__ import annotations

import inspect
import threading
from types import SimpleNamespace

import torch

from sglang_omni.models.firered_tts3 import stages
from sglang_omni.models.firered_tts3.engine_builder import FireRedTTS3EngineBuilder
from sglang_omni.models.firered_tts3.ref_encode import (
    WARMUP_AUDIO_SECONDS,
    FireRedReferenceBundle,
)
from sglang_omni.models.firered_tts3.sglang_model import FireRedTTS3SGLangModel
from sglang_omni.models.firered_tts3.vocoder import (
    WARMUP_FRAMES,
    FireRedTTS3StreamingVocoder,
)

PATCH_SIZE = 4
REDAE_DIM = 64
HIDDEN_SIZE = 16
SPK_DIM = 512


class RecordingModule(torch.nn.Module):
    """A module stub recording its call arguments; one parameter for the
    device probe."""

    def __init__(self) -> None:
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[tuple] = []

    def forward(self, *args) -> torch.Tensor:
        self.calls.append(args)
        return torch.zeros(1)


def make_bundle(*, with_campp: bool) -> FireRedReferenceBundle:
    bundle = FireRedReferenceBundle.__new__(FireRedReferenceBundle)
    bundle.device = torch.device("cpu")
    bundle.encoder = RecordingModule()
    bundle.spk_extractor = RecordingModule() if with_campp else None
    bundle.sample_rate = 24000
    bundle.pad_multiple = 3840
    bundle.lock = threading.Lock()
    return bundle


def test_reference_bundle_warmup_runs_both_encoders() -> None:
    bundle = make_bundle(with_campp=True)
    bundle.warmup()
    assert len(bundle.spk_extractor.calls) == 1
    audio, sample_rate = bundle.spk_extractor.calls[0]
    expected = int(24000 * WARMUP_AUDIO_SECONDS)
    assert audio.dtype == torch.float32
    assert audio.shape == (1, expected)
    assert expected % bundle.pad_multiple == 0
    assert sample_rate == 24000
    # The encoder consumes the same padded clip under bf16 autocast.
    assert len(bundle.encoder.calls) == 1
    assert bundle.encoder.calls[0][0] is audio


def test_reference_bundle_warmup_skips_campp_when_absent() -> None:
    bundle = make_bundle(with_campp=False)
    bundle.warmup()
    assert len(bundle.encoder.calls) == 1


def make_talker_model(*, with_spk: bool) -> FireRedTTS3SGLangModel:
    model = FireRedTTS3SGLangModel.__new__(FireRedTTS3SGLangModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(spk_in_dim=SPK_DIM if with_spk else None)
    model.patch_size = PATCH_SIZE
    model.redae_dim = REDAE_DIM
    model.history_patches = 2
    model.llm_hidden_size = HIDDEN_SIZE
    model.backbone = RecordingModule()
    model.patch_encoder = RecordingModule()
    model.dit_head = RecordingModule()
    model.stop_head = RecordingModule()
    model.spk_proj_llm = RecordingModule() if with_spk else None
    model.spk_proj_dit = RecordingModule() if with_spk else None
    return model


def test_talker_side_path_warmup_covers_hot_shapes() -> None:
    model = make_talker_model(with_spk=True)
    model.warmup_side_path()
    # patch_encoder: the prompt-assembly shape and the decode feedback shape.
    assert [tuple(call[0].shape) for call in model.patch_encoder.calls] == [
        (1, 4 * PATCH_SIZE, REDAE_DIM),
        (1, PATCH_SIZE, REDAE_DIM),
    ]
    assert [tuple(call[0].shape) for call in model.dit_head.calls] == [
        (1, model.history_patches + 1, HIDDEN_SIZE)
    ]
    assert [tuple(call[0].shape) for call in model.stop_head.calls] == [
        (1, 1, HIDDEN_SIZE)
    ]
    assert [tuple(call[0].shape) for call in model.spk_proj_llm.calls] == [(1, SPK_DIM)]
    assert [tuple(call[0].shape) for call in model.spk_proj_dit.calls] == [(1, SPK_DIM)]


def test_talker_side_path_warmup_skips_speaker_projections_when_absent() -> None:
    model = make_talker_model(with_spk=False)
    model.warmup_side_path()
    assert len(model.patch_encoder.calls) == 2
    assert model.spk_proj_llm is None and model.spk_proj_dit is None


def test_talker_setup_model_resources_runs_warmup_after_graph_capture() -> None:
    calls: list[str] = []
    model = SimpleNamespace(warmup_side_path=lambda: calls.append("warmup"))
    builder = FireRedTTS3EngineBuilder(flow_cuda_graph=False, warmup_on_init=True)
    builder.setup_model_resources(model, None, generation_cuda_graph_enabled=False)
    assert calls == ["warmup"]


def test_talker_setup_model_resources_honors_warmup_off() -> None:
    calls: list[str] = []
    model = SimpleNamespace(warmup_side_path=lambda: calls.append("warmup"))
    builder = FireRedTTS3EngineBuilder(flow_cuda_graph=False, warmup_on_init=False)
    builder.setup_model_resources(model, None, generation_cuda_graph_enabled=False)
    assert calls == []


class StubDecoder(torch.nn.Module):
    """Records forward_stream / flush_stream calls (shape, fresh, terminal)."""

    def __init__(self) -> None:
        super().__init__()
        self.probe = torch.nn.Parameter(torch.zeros(1))
        self.in_proj = torch.nn.Linear(REDAE_DIM, REDAE_DIM)
        self.stream_calls: list[tuple] = []

    def forward_stream(self, xs, cache, last_chunk=False):
        self.stream_calls.append((tuple(xs.shape), cache is None, last_chunk))
        return torch.zeros(1, 8), object()

    def flush_stream(self, cache) -> torch.Tensor:
        self.stream_calls.append(("flush", cache is not None))
        return torch.zeros(1, 4)


def test_vocoder_warmup_covers_fresh_cached_terminal_paths() -> None:
    decoder = StubDecoder()
    vocoder = FireRedTTS3StreamingVocoder(decoder, redae_scale=1.0, sample_rate=24000)
    vocoder.warmup_now()
    assert decoder.stream_calls == [
        ((1, WARMUP_FRAMES, REDAE_DIM), True, False),
        ((1, WARMUP_FRAMES, REDAE_DIM), False, True),
        ("flush", True),
    ]
    # The warmup must leave no per-request stream state behind.
    assert vocoder.stream_states == {}


def test_warmup_on_init_defaults_on_at_every_factory() -> None:
    for factory in (
        stages.create_reference_encode_executor,
        stages.create_talker_executor,
        stages.create_vocoder_executor,
    ):
        default = inspect.signature(factory).parameters["warmup_on_init"].default
        assert default is True
