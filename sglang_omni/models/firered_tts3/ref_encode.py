# SPDX-License-Identifier: Apache-2.0
"""Reference-audio conditioning for FireRedTTS3 (reference_encode stage).

One reference clip produces two artifacts, both pure functions of the audio:
the CAM++ speaker embedding (Base voice cloning) and the RedAE prompt
latents (ICL left context). They are cached content-addressed through the
shared ReferenceEncodeService so repeat references skip the serial encode.

Normalization mirrors the reference FireRedTTS3Base.generate: first channel,
fp32, resampled to 24 kHz, LEFT-padded to a whole number of latent patches
(3840 samples = 4 frames at 25 Hz each), CAM++ fp32 on the padded audio,
RedAE encode under bf16 autocast (the reference encode setting), latents
scaled into the working domain by redae_scale (Base 1.0; the vocoder divides
it back out).
"""

from __future__ import annotations

import logging
import threading

import torch

from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.payload_types import (
    load_firered_tts3_state,
    store_firered_tts3_state,
)
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.models.firered_tts3.prompt_utils import MAX_EDIT_SECONDS
from sglang_omni.models.firered_tts3.vocoder import (
    LEFT_CONTEXT_METADATA_KEY,
    PROMPT_CHUNK_METADATA_KEY,
)
from sglang_omni.models.firered_tts3.weights import load_campp, load_redae_encoder
from sglang_omni.preprocessing.cache_key import (
    hash_media_item,
    reference_path_cache_key,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.reference_encoder import (
    KeyedReferenceEncodeHook,
    ReferenceEncodeService,
)
from sglang_omni.utils.audio import load_audio

logger = logging.getLogger(__name__)

CACHE_MAX_ITEMS = 256
CACHE_MAX_BYTES = 64 * 1024 * 1024

# One clip length warms every request length: the cold start is lazy
# kernel/handle init (shape-level), not per-T state. Measured on L20: campp
# 179 ms / RedAE encode 45 ms cold vs ~120 / ~26 ms warm (7.84 s clip).
WARMUP_AUDIO_SECONDS = 8.0


class FireRedReferenceBundle:
    """CAM++ + RedAE-encoder pair for one checkpoint (fp32, autocast at call)."""

    def __init__(self, model_path: str, config: FireRedTTS3Config, device: str) -> None:
        self.config = config
        self.device = torch.device(device)
        self.encoder = load_redae_encoder(model_path, config, device=self.device)
        self.spk_extractor = (
            load_campp(model_path, device=self.device)
            if config.spk_in_dim is not None
            else None
        )
        self.sample_rate = int(config.audio_sample_rate)
        # 3840 samples: one latent patch (4 frames x 960) — the left pad keeps
        # latents aligned with the audio tail.
        self.pad_multiple = (
            int(config.audio_patch_size)
            * int(config.enc_extra_downsample_rate)
            * int(config.patch_size)
        )
        self.redae_scale = float(config.redae_scale or 1.0)
        self.lock = threading.Lock()

    def normalize(
        self, path_or_uri: str, max_seconds: float | None = None
    ) -> torch.Tensor:
        waveform = load_audio(
            path_or_uri,
            source_name="FireRedTTS3 reference",
            target_sample_rate=self.sample_rate,
            mono=True,
        )
        audio = torch.as_tensor(waveform, dtype=torch.float32).reshape(1, -1)
        if max_seconds is not None:
            # Edit sources are capped by the reference gradio contract; the
            # preprocessing stage counts pad slots on the same capped length.
            audio = audio[:, : int(self.sample_rate * max_seconds)]
        target = -(-int(audio.shape[-1]) // self.pad_multiple) * self.pad_multiple
        pad_len = target - int(audio.shape[-1])
        if pad_len > 0:
            audio = torch.nn.functional.pad(audio, (pad_len, 0))
        return audio

    @torch.no_grad()
    def encode(
        self, path_or_uri: str, max_seconds: float | None = None
    ) -> dict[str, torch.Tensor]:
        audio = self.normalize(path_or_uri, max_seconds).to(self.device)
        probe = ProfilingProbe.create()
        with self.lock:
            with probe.segment("campp"):
                spk_emb = (
                    self.spk_extractor(audio, self.sample_rate).float()
                    if self.spk_extractor is not None
                    else None
                )
            with probe.segment("redae_encode"):
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    latents = self.encoder(audio)
        probe.log(path_or_uri)
        artifact = {
            "prompt_latents": (latents.float() * self.redae_scale).cpu(),
        }
        if spk_emb is not None:
            artifact["speaker_embedding"] = spk_emb.cpu()
        return artifact

    def warmup(self) -> None:
        """Run one real-shaped dummy clip through CAM++ and the RedAE encoder.

        Runs at factory time, synchronously on the current stream (a side
        stream perturbs the cuBLAS workspace enough to shift later numerics
        run-to-run). Bypasses the reference cache: the dummy artifact is
        discarded, never stored.
        """
        samples = int(self.sample_rate * WARMUP_AUDIO_SECONDS)
        samples = -(-samples // self.pad_multiple) * self.pad_multiple
        audio = torch.zeros(1, samples, device=self.device, dtype=torch.float32)
        with self.lock, torch.no_grad():
            if self.spk_extractor is not None:
                self.spk_extractor(audio, self.sample_rate)
            # Reference encodes under bf16 autocast; warm that exact path.
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                self.encoder(audio)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        logger.info("FireRedTTS3 reference encoders warmup complete")


class FireRedReferenceHook(KeyedReferenceEncodeHook[tuple, dict, dict]):
    model_revision = ""
    encoder_id = "fireredtts3_redae_campp"
    artifact_kind = "reference_conditioning"

    def __init__(
        self,
        bundle: FireRedReferenceBundle,
        *,
        model_id: str,
        cache_enabled: bool = True,
    ) -> None:
        self.bundle = bundle
        self.model_id = model_id
        self.cache_enabled = bool(cache_enabled)
        self.encoder_config_hash = (
            f"sr{bundle.sample_rate}:pad{bundle.pad_multiple}:"
            f"scale{bundle.redae_scale}:spk{bundle.spk_extractor is not None}"
        )

    def input_key(self, item: tuple) -> str | None:
        if not self.cache_enabled:
            return None
        path, _cap_seconds = item
        return reference_path_cache_key(path, trust_stat=False) or hash_media_item(path)

    def options_key(self, item: tuple) -> str:
        _path, cap_seconds = item
        return "" if cap_seconds is None else f"cap{cap_seconds}"

    def encode_one(self, item: tuple) -> dict:
        path, cap_seconds = item
        return self.bundle.encode(path, max_seconds=cap_seconds)

    @staticmethod
    def store_artifact(artifact: dict) -> dict:
        return {
            name: tensor.detach().cpu().float().clone()
            for name, tensor in artifact.items()
        }

    @staticmethod
    def load_artifact(stored: dict) -> dict:
        return {name: tensor.clone() for name, tensor in stored.items()}


class FireRedReferenceEncoder:
    def __init__(
        self,
        bundle: FireRedReferenceBundle,
        *,
        model_id: str,
        max_batch_size: int = 1,
        max_batch_wait_ms: float = 0.0,
        cache_enabled: bool = True,
    ) -> None:
        self.bundle = bundle
        self.service = ReferenceEncodeService(
            FireRedReferenceHook(
                bundle, model_id=model_id, cache_enabled=cache_enabled
            ),
            max_items=CACHE_MAX_ITEMS,
            max_bytes=CACHE_MAX_BYTES,
            log_prefix="fireredtts3",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            batch_worker_name="fireredtts3-ref-encode-batch",
        )

    def close(self) -> None:
        self.service.close()

    def encode_payload(self, payload: StagePayload) -> StagePayload:
        state = load_firered_tts3_state(payload)
        if state.prompt_audio_path is not None:
            artifact = self.service.get_or_encode(
                (state.prompt_audio_path, None),
                desc=repr(state.prompt_audio_path),
            )
            state.prompt_latents = artifact["prompt_latents"]
            state.speaker_embedding = artifact.get("speaker_embedding")
            # Prompt-latent frame count; the vocoder trims the matching
            # waveform prefix (left_context_size x 960 samples) from the
            # decoded output.
            state.left_context_size = int(state.prompt_latents.shape[1])
        if state.audio_in_path is not None:
            # The edit source scatters into the prompt's image_pad slots; it
            # is not ICL context (no left-context trim, no prompt streaming).
            artifact = self.service.get_or_encode(
                (state.audio_in_path, MAX_EDIT_SECONDS),
                desc=repr(state.audio_in_path),
            )
            state.audio_in_latents = artifact["prompt_latents"]
        return store_firered_tts3_state(payload, state)


def prompt_stream_message(payload: StagePayload) -> OutgoingMessage | None:
    """The prompt-latent chunk handed to the vocoder ahead of the talker.

    Streaming requests only: the vocoder decodes the prompt segment as soon as
    it lands (KV/OLA warm-up behind the talker's prefill) and trims it from
    the client stream. The prompt tensor rides the local dispatcher as a CPU
    tensor from the reference cache, so no readiness event is needed.
    """
    params = payload.request.params
    if not isinstance(params, dict) or not params.get("stream"):
        return None
    state = load_firered_tts3_state(payload)
    if state.prompt_latents is None or state.prompt_latents.numel() == 0:
        return None
    return OutgoingMessage(
        request_id=payload.request_id,
        type="stream",
        target="vocoder",
        data=state.prompt_latents,
        metadata={
            "modality": "audio_latents",
            "stream": True,
            PROMPT_CHUNK_METADATA_KEY: True,
            LEFT_CONTEXT_METADATA_KEY: int(state.left_context_size),
        },
    )


__all__ = [
    "FireRedReferenceBundle",
    "FireRedReferenceEncoder",
    "FireRedReferenceHook",
    "prompt_stream_message",
]
