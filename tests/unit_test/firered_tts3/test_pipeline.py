# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.config.sources import patches_from_shared_block
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

from .conftest import CHECKPOINTS

SPEAKER_CHECKPOINT = CHECKPOINTS["speaker"]
BASE_CHECKPOINT = CHECKPOINTS["base"]
INSTRUCT_CHECKPOINT = CHECKPOINTS["instruct"]


def test_firered_tts3_uses_framework_stage_boundaries() -> None:
    from sglang_omni.models.firered_tts3.config import FireRedTTS3PipelineConfig
    from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State

    config = FireRedTTS3PipelineConfig(model_path="model")

    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "reference_encode",
        "talker",
        "vocoder",
    ]
    assert {stage.process for stage in config.stages} == {"pipeline"}
    assert config.terminal_stages == ["vocoder"]
    assert config.required_speech_reference_count == 1
    assert config.speech_reference_text_required is True
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("FireRedTTS3ForConditionalGeneration")
        is FireRedTTS3PipelineConfig
    )
    state = FireRedTTS3State()
    assert state.n_timesteps == 10
    assert state.max_patches == 400


@pytest.mark.skipif(
    not SPEAKER_CHECKPOINT.is_dir(), reason="speaker checkpoint not present"
)
def test_firered_tts3_speaker_drops_reference_encode_stage() -> None:
    from sglang_omni.models.firered_tts3.config import FireRedTTS3PipelineConfig

    config = FireRedTTS3PipelineConfig(model_path=str(SPEAKER_CHECKPOINT))

    assert config.variant == "speaker"
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "talker",
        "vocoder",
    ]
    assert config.stage_named("preprocessing").next == "talker"
    assert config.terminal_stages == ["vocoder"]
    # The speaker variant takes no reference clip (0 rejects any supplied
    # reference with a 400 at the serving layer) and supports the dual-stream
    # opt-in.
    assert config.required_speech_reference_count == 0
    assert config.speech_reference_text_required is False
    assert config.speech_dual_stream_supported is True
    # The speaker checkpoint default budget is larger (config.json).
    assert config.talker_generation_limits() == (10, 600)


def test_firered_tts3_shares_generation_schedule_via_shared() -> None:
    """One shared entry feeds the schedule settings to both consumers."""
    from sglang_omni.models.firered_tts3.config import FireRedTTS3PipelineConfig

    config = FireRedTTS3PipelineConfig(model_path="model")
    patches = patches_from_shared_block(
        [
            {
                "select": {"stages": ["preprocessing", "talker"]},
                "factory": {"n_timesteps": 8, "max_patches": 256},
            }
        ],
        FireRedTTS3PipelineConfig,
        [stage.name for stage in config.stages],
    )
    resolved = ConfigResolver(config).resolve(patches).config

    for name in ("preprocessing", "talker"):
        kwargs = resolve_stage_typed_kwargs(resolved.stage_named(name))
        assert kwargs["n_timesteps"] == 8
        assert kwargs["max_patches"] == 256
    assert resolved.stage_factory_kwargs("preprocessing") == {
        "n_timesteps": 8,
        "max_patches": 256,
    }
    assert resolved.stage_factory_kwargs("vocoder") == {"stream_slots": 16}


def test_firered_tts3_rejects_conflicting_preprocessing_schedule() -> None:
    from sglang_omni.models.firered_tts3.config import FireRedTTS3PipelineConfig

    raw = FireRedTTS3PipelineConfig(model_path="model").model_dump()
    stages = {item["name"]: item for item in raw["stages"]}
    stages["talker"]["factory"] = {"n_timesteps": 10}
    stages["preprocessing"]["factory"] = {"n_timesteps": 8}
    with pytest.raises(ValueError, match="n_timesteps"):
        FireRedTTS3PipelineConfig(**raw)


def test_firered_tts3_rejects_mismatched_vocoder_stream_slots() -> None:
    from sglang_omni.models.firered_tts3.config import FireRedTTS3PipelineConfig

    raw = FireRedTTS3PipelineConfig(model_path="model").model_dump()
    stages = {item["name"]: item for item in raw["stages"]}
    stages["talker"]["engine"]["max_running_requests"] = 8
    stages["vocoder"]["factory"]["stream_slots"] = 16
    with pytest.raises(ValueError, match="stream_slots"):
        FireRedTTS3PipelineConfig(**raw)


def test_firered_tts3_rejects_tp() -> None:
    from sglang_omni.models.firered_tts3.config import FireRedTTS3PipelineConfig

    raw = FireRedTTS3PipelineConfig(model_path="model").model_dump()
    stage = next(item for item in raw["stages"] if item["name"] == "talker")
    stage["tp_size"] = 2
    stage["gpu"] = [0, 1]
    stage["process"] = "talker"
    with pytest.raises(ValueError, match="tp_size=1"):
        FireRedTTS3PipelineConfig(**raw)


def test_firered_tts3_model_registered_with_sglang() -> None:
    from sglang.srt.models.registry import ModelRegistry

    from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
    from sglang_omni.models.firered_tts3.sglang_model import FireRedTTS3SGLangModel

    SGLModelRunner.register_omni_model(SGLModelRunner)

    assert (
        ModelRegistry.models["FireRedTTS3ForConditionalGeneration"]
        is FireRedTTS3SGLangModel
    )


def test_firered_tts3_state_wire_round_trip() -> None:
    from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State

    state = FireRedTTS3State(
        variant="instruct",
        task_type="voice_design",
        text="hello",
        prompt_audio_path="ref.wav",
        prompt_text="reference",
        instructions="calm",
        language="zh",
        audio_in_path="edit.wav",
        input_ids=[1, 2, 3],
        n_timesteps=8,
        inference_cfg=1.2,
        stop_threshold=0.6,
        min_patches=4,
        max_patches=256,
        max_engine_tokens=788,
        dual_stream=True,
        text_stream_ids=[101, 102, 103],
        text_pad_id=151671,
        seed=7,
        stream=True,
        text_temperature=0.7,
        text_top_p=0.8,
        text_top_k=20,
        left_context_size=12,
        prompt_latents=torch.rand(1, 12, 64),
        audio_in_latents=torch.rand(1, 8, 64),
        speaker_embedding=torch.rand(1, 512),
        generated_latents=torch.rand(1, 8, 64),
        gen_text="generated",
        finish_reason="stop",
        prompt_tokens=3,
        completion_tokens=5,
        engine_time_s=0.25,
    )

    before = state.to_dict()
    restored = FireRedTTS3State.from_dict(before)
    after = restored.to_dict()

    assert set(after) == set(before)
    for f in dataclasses.fields(state):
        expected = getattr(state, f.name)
        actual = getattr(restored, f.name)
        if isinstance(expected, torch.Tensor):
            assert isinstance(actual, torch.Tensor), f.name
            assert torch.equal(actual, expected), f.name
        else:
            assert actual == expected, f.name


def test_firered_tts3_state_defaults_round_trip() -> None:
    from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State

    restored = FireRedTTS3State.from_dict(FireRedTTS3State().to_dict())

    assert restored.sample_rate == 24000
    assert restored.variant == "base"
    assert restored.input_ids == []
    assert restored.prompt_latents is None
    assert restored.seed is None


@pytest.mark.skipif(
    not BASE_CHECKPOINT.is_dir(),
    reason="Base checkpoint not present",
)
def test_firered_tts3_preprocessing_factory_builds() -> None:
    from sglang_omni.models.firered_tts3 import stages

    scheduler = stages.create_preprocessing_executor(
        str(BASE_CHECKPOINT)
    )
    assert scheduler is not None


@pytest.mark.skipif(
    not INSTRUCT_CHECKPOINT.is_dir(),
    reason="Instruct checkpoint not present",
)
def test_firered_tts3_preprocessing_factory_builds_instruct() -> None:
    from sglang_omni.models.firered_tts3 import stages

    scheduler = stages.create_preprocessing_executor(
        str(INSTRUCT_CHECKPOINT)
    )
    assert scheduler is not None


def test_firered_tts3_profiling_probe_follows_env(monkeypatch) -> None:
    from sglang_omni.models.firered_tts3 import profiling

    monkeypatch.delenv(profiling.FIREREDTTS3_PROF_ENV, raising=False)
    probe = profiling.ProfilingProbe.create()
    assert not probe.enabled
    with probe.segment("flow"):
        pass
    assert probe.segments == {}

    monkeypatch.setenv(profiling.FIREREDTTS3_PROF_ENV, "1")
    probe = profiling.ProfilingProbe.create()
    assert probe.enabled
    with probe.segment("flow"):
        pass
    assert probe.segments["flow"] >= 0.0
