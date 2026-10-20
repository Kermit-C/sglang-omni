# SPDX-License-Identifier: Apache-2.0
"""Checkpoint key-routing dry run for the three FireRedTTS3 variants.

Every key of every shipped checkpoint must land in exactly one bucket
(backbone / lm_head / side path / redae encoder / redae decoder / campp /
explicitly skipped), and each bucket must equal its module's state-dict key
space — no missing, no unexpected. Modules are built on the meta device, so
no GPU or weight materialization is needed.
"""

from __future__ import annotations

import collections
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import Qwen3Config, Qwen3Model

import sglang_omni.models.firered_tts3.engine_builder as engine_builder_module
from sglang_omni.models.firered_tts3.engine_builder import FireRedTTS3EngineBuilder
from sglang_omni.models.firered_tts3.sglang_model import FireRedTTS3SGLangModel
from sglang_omni.models.firered_tts3.weights import (
    build_dit,
    build_patch_encoder,
    build_redae_decoder,
    build_redae_encoder,
    drop_embedding_tables,
    route_root_key,
    strip_backbone_prefix,
)

from .conftest import CHECKPOINTS, requires_checkpoints

# RedAE embed tables (151936 x 896 fp32, ~544 MiB each) are never read at
# inference (inputs_embeds only) and are skipped at load.
REDAE_ENCODER_SKIP_KEYS = {
    "qwen3.embed_tokens.weight",
    "downsample.qwen3.embed_tokens.weight",
}
REDAE_DECODER_SKIP_KEYS = {"qwen3.embed_tokens.weight"}


@pytest.fixture
def loadable_model() -> FireRedTTS3SGLangModel:
    class Backbone(torch.nn.Linear):
        def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
            with torch.no_grad():
                for name, tensor in weights:
                    self.get_parameter(name).copy_(tensor)

    model = FireRedTTS3SGLangModel.__new__(FireRedTTS3SGLangModel)
    torch.nn.Module.__init__(model)
    model.backbone = Backbone(2, 2, bias=False)
    for name in ("patch_encoder", "dit", "dit_head", "stop_head"):
        setattr(model, name, torch.nn.Linear(2, 2, bias=False))
    model.spk_proj_llm = model.spk_proj_dit = None
    return model


def test_weight_loading_finishes_before_execution_is_locked(
    loadable_model: FireRedTTS3SGLangModel,
) -> None:
    model = loadable_model
    assert model.weight_update_blocked_reason is None
    initial = torch.ones_like(model.dit.weight)
    assert model.load_weights([("dit.weight", initial)]) == {"dit.weight"}
    assert torch.equal(model.dit.weight, initial)
    model.weights_locked = True
    updates = iter([("dit.weight", torch.zeros_like(initial))])
    with pytest.raises(RuntimeError, match="restart the worker"):
        model.load_weights(updates)
    assert next(updates)[0] == "dit.weight"
    assert torch.equal(model.dit.weight, initial)


@pytest.mark.parametrize("compile_fails", [False, True])
def test_compile_locks_updates_before_creating_execution(
    loadable_model: FireRedTTS3SGLangModel,
    monkeypatch: pytest.MonkeyPatch,
    compile_fails: bool,
) -> None:
    model = loadable_model
    builder = FireRedTTS3EngineBuilder(flow_cuda_graph=True)

    def prepare(
        dit: torch.nn.Module, mode: str
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        assert dit is model.dit and mode == builder.flow_mode
        with pytest.raises(RuntimeError, match="restart the worker"):
            model.load_weights([])
        if compile_fails:
            raise RuntimeError("compile failed")
        else:
            return dit, dit

    monkeypatch.setattr(engine_builder_module, "prepare_flow_dit", prepare)
    if compile_fails:
        with pytest.raises(RuntimeError, match="compile failed"):
            builder.compile_model(model, None)
    else:
        builder.compile_model(model, None)
    assert model.weight_update_blocked_reason is not None


def test_disabled_flow_graph_does_not_lock_weight_loading(
    loadable_model: FireRedTTS3SGLangModel,
) -> None:
    builder = FireRedTTS3EngineBuilder(flow_cuda_graph=False)
    builder.compile_model(loadable_model, None)
    assert loadable_model.weight_update_blocked_reason is None
    replacement = torch.zeros_like(loadable_model.dit.weight)
    loadable_model.load_weights([("dit.weight", replacement)])
    assert torch.equal(loadable_model.dit.weight, replacement)


def checkpoint_keys(path: Path, subfolder: str = "") -> list[str]:
    folder = path / subfolder if subfolder else path
    keys: list[str] = []
    for shard in sorted(folder.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            keys.extend(handle.keys())
    return keys


def backbone_key_space() -> set[str]:
    """The bare Qwen3Model-1.7B state-dict keys (HF layout, meta device)."""
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=40960,
        rope_theta=1e6,
        tie_word_embeddings=True,
    )
    with torch.device("meta"):
        model = Qwen3Model(config)
    return set(model.state_dict().keys())


def linear_key_space() -> set[str]:
    return {"weight", "bias"}


@requires_checkpoints
@pytest.mark.parametrize(
    "variant, expected",
    [
        (
            "base",
            {
                "backbone": 310,
                "patch_encoder": 118,
                "dit": 241,
                "dit_head": 2,
                "stop_head": 2,
                "spk_proj_llm": 2,
                "spk_proj_dit": 2,
                "lm_head": 0,
                "unknown": 0,
            },
        ),
        (
            "instruct",
            {
                "backbone": 310,
                "patch_encoder": 118,
                "dit": 241,
                "dit_head": 2,
                "stop_head": 2,
                "spk_proj_llm": 0,
                "spk_proj_dit": 0,
                "lm_head": 0,
                "unknown": 0,
            },
        ),
        (
            "speaker",
            {
                "backbone": 310,
                "patch_encoder": 118,
                "dit": 241,
                "dit_head": 2,
                "stop_head": 2,
                "spk_proj_llm": 0,
                "spk_proj_dit": 0,
                # The tied lm_head clone in speaker checkpoints is dropped.
                "lm_head": 1,
                "unknown": 0,
            },
        ),
    ],
)
def test_root_checkpoint_key_routing(variant: str, expected: dict[str, int]) -> None:
    from sglang_omni.models.firered_tts3.config import load_checkpoint_hf_config

    checkpoint = CHECKPOINTS[variant]
    keys = checkpoint_keys(checkpoint)
    buckets = collections.Counter(route_root_key(key) for key in keys)
    assert sum(buckets.values()) == len(keys)
    for bucket, count in expected.items():
        assert buckets[bucket] == count, f"{variant} bucket {bucket}"

    config = load_checkpoint_hf_config(str(checkpoint))
    assert config is not None

    # Backbone keys strip onto the bare Qwen3Model key space.
    backbone_keys = {
        strip_backbone_prefix(key) for key in keys if route_root_key(key) == "backbone"
    }
    assert backbone_keys == backbone_key_space()

    # Side-path key spaces match the ported modules exactly.
    with torch.device("meta"):
        patch_encoder = build_patch_encoder(config)
        dit = build_dit(config)
    for prefix, module in (("patch_encoder.", patch_encoder), ("dit.", dit)):
        stripped = {key[len(prefix) :] for key in keys if key.startswith(prefix)}
        assert stripped == set(module.state_dict().keys()), prefix
    for prefix in ("dit_head.", "stop_head.", "spk_proj_llm.", "spk_proj_dit."):
        stripped = {key[len(prefix) :] for key in keys if key.startswith(prefix)}
        assert stripped == (linear_key_space() if stripped else set()), prefix

    # lm_head only appears in the speaker variant (tied clone, dropped at
    # load); Instruct carries no lm_head key (tied to embed_tokens).
    lm_head_keys = [key for key in keys if route_root_key(key) == "lm_head"]
    if variant == "speaker":
        assert lm_head_keys == ["backbone_llm.lm_head.weight"]
    else:
        assert lm_head_keys == []


@requires_checkpoints
def test_redae_key_routing_and_embed_skip() -> None:
    from sglang_omni.models.firered_tts3.config import load_checkpoint_hf_config

    config = load_checkpoint_hf_config(str(CHECKPOINTS["base"]))
    assert config is not None
    for variant in ("base", "instruct", "speaker"):
        keys = checkpoint_keys(CHECKPOINTS[variant], "redae")
        encoder_keys = [key for key in keys if key.startswith("encoder.")]
        decoder_keys = [key for key in keys if key.startswith("decoder.")]
        assert len(encoder_keys) == 253, variant
        assert len(decoder_keys) == 205, variant
        assert len(keys) == len(encoder_keys) + len(decoder_keys)

        with torch.device("meta"):
            encoder = build_redae_encoder(config)
            decoder = build_redae_decoder(config)
        encoder_full = set(encoder.state_dict().keys())
        decoder_full = set(decoder.state_dict().keys())
        # The checkpoint key spaces (prefix-stripped) equal the module key
        # spaces before the embed tables are dropped.
        assert {key[len("encoder.") :] for key in encoder_keys} == encoder_full
        assert {key[len("decoder.") :] for key in decoder_keys} == decoder_full
        # ... and the skip sets are exactly the redundant embed tables.
        assert {
            key[len("encoder.") :]
            for key in encoder_keys
            if key[len("encoder.") :] in REDAE_ENCODER_SKIP_KEYS
        } == REDAE_ENCODER_SKIP_KEYS
        assert {
            key[len("decoder.") :]
            for key in decoder_keys
            if key[len("decoder.") :] in REDAE_DECODER_SKIP_KEYS
        } == REDAE_DECODER_SKIP_KEYS

        drop_embedding_tables(encoder)
        drop_embedding_tables(decoder)
        assert (
            set(encoder.state_dict().keys()) == encoder_full - REDAE_ENCODER_SKIP_KEYS
        )
        assert (
            set(decoder.state_dict().keys()) == decoder_full - REDAE_DECODER_SKIP_KEYS
        )


@requires_checkpoints
def test_campp_weights_present_for_base_only() -> None:
    assert (CHECKPOINTS["base"] / "campp" / "campplus_voxceleb.bin").is_file()
    assert not (CHECKPOINTS["instruct"] / "campp").exists()
    assert not (CHECKPOINTS["speaker"] / "campp").exists()
