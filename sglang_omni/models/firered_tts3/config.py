# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for FireRedTTS3 (Base / Instruct / speaker)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.models.firered_tts3.hf_config import (
    FireRedTTS3Config,
    FireRedTTS3Variant,
    detect_fireredtts3_variant,
)

_PKG = "sglang_omni.models.firered_tts3"

# Defaults mirror the checkpoints' generation_defaults (config.json); the
# speaker checkpoint allows longer generations than Base/Instruct.
DEFAULT_N_TIMESTEPS = 10
DEFAULT_MAX_PATCHES = 400
SPEAKER_MAX_PATCHES = 600
DEFAULT_MAX_RUNNING_REQUESTS = 16


class FireRedTTS3VocoderFactoryArgs(FactoryArgs):
    """FireRedTTS3 vocoder constructor knobs, typed like the shared ones."""

    stream_slots: int | None = Field(default=None, ge=1)
    # Decode the prompt segment the moment its chunk lands (KV/OLA warm-up
    # behind the talker's prefill) instead of jointly with the first
    # generated chunk. Measured a TTFP loss while all stages share the
    # default CUDA stream; defaults off.
    prompt_predecode: bool | None = None
    # Coalesce equal-progress streaming requests into one batched decoder
    # forward per step (the RedAE decoder is CPU-launch bound, so a group
    # costs ~one row). Size-1 groups keep the serial decode bit-exact.
    stream_batching: bool | None = None
    # Decode on a dedicated CUDA stream (waiting only on each chunk's
    # readiness event) instead of the legacy default stream shared with the
    # talker's back-to-back graph replays.
    separate_decode_stream: bool | None = None
    # Replay warm steady-state chunk decodes from self-captured CUDA graphs
    # (one per chunk width x batch bucket; B=1 exact-bucket replays are
    # bit-exact with eager). Cold/ramp/terminal chunks stay eager.
    incremental_cuda_graph: bool | None = None
    cuda_graph_latent_sizes: list[int] | None = None
    cuda_graph_batch_sizes: list[int] | None = None
    # Replay the fresh-cache joint prompt+first-chunk decode from left-padded
    # bucket graphs (TTFP path; exact-width hits bit-exact, padded replays
    # differ only inside the trimmed prompt lead plus fp32 tiling noise).
    cold_cuda_graph: bool | None = None
    cold_cuda_graph_latent_sizes: list[int] | None = None
    cold_cuda_graph_batch_sizes: list[int] | None = None
    # Run the decoder-path warmup at factory time (fresh + cached + terminal
    # forward_stream shapes) so the first served request skips the cold
    # start. Defaults on in the factory.
    warmup_on_init: bool | None = None


class FireRedTTS3VocoderStageConfig(StageConfig):
    factory: FireRedTTS3VocoderFactoryArgs = Field(
        default_factory=FireRedTTS3VocoderFactoryArgs
    )


def load_checkpoint_hf_config(model_path: str) -> FireRedTTS3Config | None:
    """Read the checkpoint config.json for local paths; None when unreadable."""
    config_path = Path(model_path).expanduser() / "config.json"
    if not config_path.is_file():
        return None
    else:
        pass
    with config_path.open(encoding="utf-8") as handle:
        return FireRedTTS3Config(**json.load(handle))


class FireRedTTS3PipelineConfig(PipelineConfig):
    """preprocessing -> reference encode -> talker (AR + flow) -> vocoder.

    The speaker variant conditions on a speaker token instead of a reference
    clip, so its deployment drops the reference_encode stage; the stage list
    is derived from the checkpoint config at validation time, keeping the
    topology static per deployment.
    """

    architecture: ClassVar[str] = "FireRedTTS3ForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "fireredtts3",
        "firered_tts3",
    )
    # Edit tasks carry their content in the reference audio, so `input` may
    # be empty; acoustic_edit consumes `speed` as the model's native
    # instruction (no encode-time post-process).
    speech_empty_input_task_types: ClassVar[frozenset[str]] = frozenset(
        {"SemanticEdit", "AcousticEdit"}
    )
    speech_native_speed_control: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "talker": EngineStageConfig,
        "vocoder": FireRedTTS3VocoderStageConfig,
    }

    model_path: str
    variant: FireRedTTS3Variant | None = None
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            next="reference_encode",
        ),
        StageConfig(
            name="reference_encode",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_reference_encode_executor",
            gpu=0,
            next="talker",
            # Prompt latents reach the vocoder as the stream's first chunk,
            # long before the talker finishes: the vocoder warms its decoder
            # state on them while the talker is still prefilling.
            stream_to=["vocoder"],
        ),
        EngineStageConfig(
            name="talker",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_talker_executor",
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        FireRedTTS3VocoderStageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    @property
    def required_speech_reference_count(self) -> int | None:
        # The speaker variant conditions on a speaker token and takes no
        # reference clip: 0 rejects any supplied reference with a 400.
        # Instruct is per-task (voice_design takes none, tts/edits take one),
        # so the static count cannot express it — the preprocessing stage
        # validates.
        if self.variant == "speaker":
            return 0
        else:
            pass
        return None if self.variant == "instruct" else 1

    @property
    def speech_dual_stream_supported(self) -> bool:
        # The dual-stream decode form (interleaved text/patch feeds) is an
        # extra_params.dual_stream opt-in gated on the speaker checkpoint's
        # text_latent_interleave_ratio capability.
        return self.variant == "speaker"

    @property
    def speech_reference_text_required(self) -> bool:
        # Same story: only the Base clone (and the Instruct tts task) needs
        # the reference transcript; edit tasks never carry one.
        return self.variant not in ("instruct", "speaker")

    def model_post_init(self, __context: dict[str, str] | None = None) -> None:
        super().model_post_init(__context)
        if any(stage.tp_size != 1 for stage in self.stages):
            raise ValueError("FireRedTTS3 currently supports tp_size=1 only")
        else:
            pass
        checkpoint_config = load_checkpoint_hf_config(self.model_path)
        if checkpoint_config is not None:
            detected = detect_fireredtts3_variant(checkpoint_config)
            if self.variant is not None and self.variant != detected:
                raise ValueError(
                    f"pinned variant {self.variant!r} disagrees with checkpoint "
                    f"{self.model_path!r} (detected {detected!r})"
                )
            else:
                pass
            self.variant = detected
        else:
            pass
        if self.variant == "speaker":
            self.drop_reference_encode_stage()
        else:
            pass
        # The talker owns the generation schedule (ODE steps and patch
        # budget); preprocessing only derives from it, so a preprocessing
        # value that conflicts with the talker value is refused. Unset
        # preprocessing keys are derived at launch from the talker.
        n_timesteps, max_patches = self.talker_generation_limits()
        preprocessing_extra = (
            self.stage_named("preprocessing").factory.model_extra or {}
        )
        for key, talker_value in (
            ("n_timesteps", n_timesteps),
            ("max_patches", max_patches),
        ):
            pinned = preprocessing_extra.get(key)
            if pinned is not None and int(pinned) != talker_value:
                raise ValueError(
                    f"FireRedTTS3 preprocessing {key!r} ({int(pinned)}) "
                    f"disagrees with talker {key!r} ({talker_value}); "
                    f"configure only talker.factory.{key} and let "
                    "preprocessing derive it, or set both to the same value"
                )
            else:
                pass
        # stream_slots must match backbone concurrency so a
        # max_running_requests override cannot outrun vocoder admission. A
        # pinned value that disagrees is refused; an unset value is derived
        # at launch (writing the derivation into the config would make it
        # look pinned on the next merge).
        stream_slots = self.stage_named("vocoder").factory.stream_slots
        if stream_slots is not None:
            derived = self.talker_max_running_requests(self.stage_named("talker"))
            if int(stream_slots) != derived:
                raise ValueError(
                    f"FireRedTTS3 vocoder stream_slots ({int(stream_slots)}) "
                    f"must equal talker max_running_requests ({derived}); "
                    "omit stream_slots to derive it from the talker"
                )
            else:
                pass
        else:
            pass

    def drop_reference_encode_stage(self) -> None:
        """Remove the reference_encode stage and rewire preprocessing."""
        if all(stage.name != "reference_encode" for stage in self.stages):
            return
        else:
            pass
        self.stages = [
            stage for stage in self.stages if stage.name != "reference_encode"
        ]
        preprocessing = self.stage_named("preprocessing")
        if preprocessing.next == "reference_encode":
            preprocessing.next = "talker"
        else:
            pass
        self.validate_general()

    def talker_generation_limits(self) -> tuple[int, int]:
        """Return (n_timesteps, max_patches) owned by the talker stage."""
        extra = self.stage_named("talker").factory.model_extra or {}
        default_max_patches = (
            SPEAKER_MAX_PATCHES if self.variant == "speaker" else DEFAULT_MAX_PATCHES
        )
        return (
            int(extra.get("n_timesteps", DEFAULT_N_TIMESTEPS)),
            int(extra.get("max_patches", default_max_patches)),
        )

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        if stage_name == "preprocessing":
            n_timesteps, max_patches = self.talker_generation_limits()
            return {"n_timesteps": n_timesteps, "max_patches": max_patches}
        else:
            pass
        if stage_name == "vocoder":
            return {
                "stream_slots": self.talker_max_running_requests(
                    self.stage_named("talker")
                )
            }
        else:
            pass
        return {}

    @staticmethod
    def talker_max_running_requests(talker: StageConfig) -> int:
        engine = talker.engine
        value = engine.max_running_requests if engine is not None else None
        if value is None:
            return DEFAULT_MAX_RUNNING_REQUESTS
        else:
            pass
        return int(value)

    @classmethod
    def generation_admission_defaults(cls) -> dict[str, Any]:
        return {
            "max_running_requests": DEFAULT_MAX_RUNNING_REQUESTS,
            "max_queued_requests": DEFAULT_MAX_RUNNING_REQUESTS,
        }

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = FireRedTTS3PipelineConfig
