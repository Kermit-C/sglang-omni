# SPDX-License-Identifier: Apache-2.0
"""Reference-conditioning cache tests: hit accounting, artifact isolation,
the edit-cap key split, and the cache kill switch."""

from __future__ import annotations

import torch

from sglang_omni.models.firered_tts3.ref_encode import FireRedReferenceHook
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService


class StubReferenceBundle:
    """Hook-facing bundle surface, with per-call distinct artifacts so a
    cache hit is observable by value."""

    sample_rate = 24000
    pad_multiple = 3840
    redae_scale = 1.0
    spk_extractor = None

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(self, path: str, max_seconds: float | None = None) -> dict:
        self.encode_calls += 1
        return {"prompt_latents": torch.full((1, 2, 4), float(self.encode_calls))}


def make_service(bundle: StubReferenceBundle, *, cache_enabled: bool = True):
    return ReferenceEncodeService(
        FireRedReferenceHook(bundle, model_id="stub", cache_enabled=cache_enabled),
        max_items=8,
        max_bytes=1 << 20,
    )


def test_reference_cache_hit_reuses_stored_artifact(tmp_path) -> None:
    bundle = StubReferenceBundle()
    service = make_service(bundle)
    clip = tmp_path / "ref.wav"
    clip.write_bytes(b"ref-audio-bytes")

    first = service.get_or_encode((str(clip), None))
    second = service.get_or_encode((str(clip), None))
    assert bundle.encode_calls == 1
    assert service.stats()["hits"] == 1
    assert torch.equal(first["prompt_latents"], second["prompt_latents"])

    # Returned artifacts are per-call clones: mutating one must not pollute
    # the stored entry.
    second["prompt_latents"].fill_(-1.0)
    third = service.get_or_encode((str(clip), None))
    assert torch.equal(third["prompt_latents"], first["prompt_latents"])
    service.close()


def test_reference_cache_key_separates_edit_cap(tmp_path) -> None:
    bundle = StubReferenceBundle()
    service = make_service(bundle)
    clip = tmp_path / "ref.wav"
    clip.write_bytes(b"ref-audio-bytes")

    service.get_or_encode((str(clip), None))
    service.get_or_encode((str(clip), 30.0))
    assert bundle.encode_calls == 2
    assert service.stats()["entries"] == 2
    service.close()


def test_reference_cache_disabled_encodes_every_call(tmp_path) -> None:
    bundle = StubReferenceBundle()
    service = make_service(bundle, cache_enabled=False)
    clip = tmp_path / "ref.wav"
    clip.write_bytes(b"ref-audio-bytes")

    service.get_or_encode((str(clip), None))
    service.get_or_encode((str(clip), None))
    assert bundle.encode_calls == 2
    stats = service.stats()
    assert stats["uncacheable"] == 2
    assert stats["hits"] == 0
    assert stats["entries"] == 0
    service.close()
