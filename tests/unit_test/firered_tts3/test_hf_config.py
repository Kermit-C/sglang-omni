# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from .conftest import CHECKPOINTS

checkpoints_present = all(path.is_dir() for path in CHECKPOINTS.values())
requires_checkpoints = pytest.mark.skipif(
    not checkpoints_present, reason="FireRedTTS3 checkpoints not present"
)


def test_fireredtts3_hf_config_loads_from_minimal_config_json(tmp_path) -> None:
    from transformers import AutoConfig

    from sglang_omni.models.firered_tts3.hf_config import (
        FireRedTTS3Config,
        detect_fireredtts3_variant,
        register_fireredtts3_hf_config,
    )

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fireredtts3",
                "architectures": ["FireRedTTS3ForConditionalGeneration"],
                "spk_in_dim": 512,
            }
        ),
        encoding="utf-8",
    )

    register_fireredtts3_hf_config()
    config = AutoConfig.from_pretrained(tmp_path, local_files_only=True)

    assert isinstance(config, FireRedTTS3Config)
    assert config.name_or_path == str(tmp_path)
    assert config.hidden_size == 2048
    assert config.patch_size == 4
    assert detect_fireredtts3_variant(config) == "base"


def test_detect_fireredtts3_variant_rejects_unknown_checkpoint() -> None:
    from sglang_omni.models.firered_tts3.hf_config import (
        FireRedTTS3Config,
        detect_fireredtts3_variant,
    )

    with pytest.raises(ValueError, match="unrecognized FireRedTTS3 checkpoint"):
        detect_fireredtts3_variant(FireRedTTS3Config())


@requires_checkpoints
def test_fireredtts3_hf_config_detects_real_checkpoint_variants() -> None:
    from transformers import AutoConfig

    from sglang_omni.models.firered_tts3.hf_config import (
        FireRedTTS3Config,
        detect_fireredtts3_variant,
        register_fireredtts3_hf_config,
    )

    register_fireredtts3_hf_config()
    for expected, path in CHECKPOINTS.items():
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        assert isinstance(config, FireRedTTS3Config)
        assert config.model_type == "fireredtts3"
        assert detect_fireredtts3_variant(config) == expected


@requires_checkpoints
def test_fireredtts3_checkpoint_variant_marker_values() -> None:
    from transformers import AutoConfig

    from sglang_omni.models.firered_tts3.hf_config import register_fireredtts3_hf_config

    register_fireredtts3_hf_config()
    configs = {
        variant: AutoConfig.from_pretrained(path, local_files_only=True)
        for variant, path in CHECKPOINTS.items()
    }

    base = configs["base"]
    assert base.spk_in_dim == 512
    assert base.use_lm_head is False
    assert base.dit_in_channels == 1600
    assert base.redae_scale == 1.0
    assert base.special_tokens["text_start"] == "<|sot|>"
    assert base.generation_defaults["max_patches"] == 400

    instruct = configs["instruct"]
    assert instruct.use_lm_head is True
    assert instruct.spk_in_dim is None
    assert instruct.dit_in_channels == 1088
    assert instruct.redae_scale == 0.4
    assert instruct.special_tokens["audio_sos"] == "<|sosp|>"

    speaker = configs["speaker"]
    assert speaker.text_latent_interleave_ratio == [2, 1]
    assert speaker.speaker_token == "<|f531_speaker|>"
    assert speaker.special_tokens["audio_holder"] == "<|audio_holder|>"
    assert speaker.generation_defaults["max_patches"] == 600
