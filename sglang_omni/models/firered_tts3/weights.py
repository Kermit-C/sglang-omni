# SPDX-License-Identifier: Apache-2.0
"""Checkpoint key routing and component loading for FireRedTTS3.

One root safetensors carries the talker (backbone_llm.* + side path) and a
redae/ subdirectory carries the shared audio autoencoder:

* backbone_llm.model.* (Instruct / speaker, Qwen3ForCausalLM layout) and
  backbone_llm.* (Base, bare Qwen3Model layout) -> the backbone; the
  backbone_llm.lm_head.weight copy in speaker checkpoints is a tied clone and
  is dropped (Instruct checkpoints carry no lm_head key at all — the head
  is tied to embed_tokens at load).
* patch_encoder. / dit. / dit_head. / stop_head. / spk_proj_llm. /
  spk_proj_dit. -> the fp32 side path.
* redae/encoder.* and redae/decoder.* -> the RedAE modules. Every
  qwen3 embed_tokens table in redae/ (151936 x 896 each, ~544 MiB) is dead
  weight — both modules consume inputs_embeds only — so the keys are
  skipped at read time and the modules are set to None.
"""

from __future__ import annotations

from collections.abc import Container
from pathlib import Path

import torch
from safetensors import safe_open

from sglang_omni.models.firered_tts3.campp import CamppEmbedding
from sglang_omni.models.firered_tts3.dit import DiT
from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.patch_encoder import PatchEncoder
from sglang_omni.models.firered_tts3.redae_decoder import RedAEAudioDecoder
from sglang_omni.models.firered_tts3.redae_encoder import RedAEAudioEncoder

BACKBONE_LLM_PREFIX = "backbone_llm."
# Instruct / speaker nest the Qwen3Model one level deeper (Qwen3ForCausalLM).
BACKBONE_MODEL_INFIX = "model."
BACKBONE_LM_HEAD_PREFIX = "backbone_llm.lm_head."
SIDE_PATH_PREFIXES = (
    "patch_encoder.",
    "dit.",
    "dit_head.",
    "stop_head.",
    "spk_proj_llm.",
    "spk_proj_dit.",
)

CAMPP_RELATIVE_PATH = Path("campp") / "campplus_voxceleb.bin"


def route_root_key(key: str) -> str:
    """Classify a root-checkpoint key into its component bucket.

    Returns one of: "backbone", "lm_head", "patch_encoder", "dit",
    "dit_head", "stop_head", "spk_proj_llm", "spk_proj_dit", "unknown".
    """
    if key.startswith(BACKBONE_LM_HEAD_PREFIX):
        return "lm_head"
    else:
        pass
    if key.startswith(BACKBONE_LLM_PREFIX):
        return "backbone"
    else:
        pass
    for prefix in SIDE_PATH_PREFIXES:
        if key.startswith(prefix):
            return prefix.removesuffix(".")
        else:
            pass
    return "unknown"


def strip_backbone_prefix(key: str) -> str:
    """Map a backbone_llm[.model].* key onto the bare Qwen3Model key space."""
    stripped = key.removeprefix(BACKBONE_LLM_PREFIX)
    return stripped.removeprefix(BACKBONE_MODEL_INFIX)


def read_component_tensors(
    checkpoint_dir: Path,
    prefix: str,
    *,
    skip_if: Container[str] = frozenset(),
) -> dict[str, torch.Tensor]:
    """Read the prefix-matching tensors of a checkpoint dir, prefix-stripped.

    skip_if matches against the STRIPPED key. Supports both a single
    model.safetensors and sharded layouts.
    """
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(checkpoint_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if not key.startswith(prefix):
                    continue
                else:
                    pass
                stripped = key[len(prefix) :]
                if stripped in skip_if:
                    continue
                else:
                    pass
                tensors[stripped] = handle.get_tensor(key)
    return tensors


def build_dit(config: FireRedTTS3Config) -> DiT:
    return DiT(
        in_channels=config.dit_in_channels,
        out_channels=config.redae_dim,
        mlp_ratio=config.dit_mlp_ratio,
        depth=config.dit_depth,
        num_heads=config.dit_num_heads,
        hidden_size=config.dit_hidden_size,
    )


def build_patch_encoder(config: FireRedTTS3Config) -> PatchEncoder:
    return PatchEncoder(
        in_dim=config.redae_dim,
        out_dim=config.hidden_size,
        patch_size=config.patch_size,
        hidden_size=config.patch_encoder_hidden_size,
        mlp_ratio=config.patch_encoder_mlp_ratio,
        depth=config.patch_encoder_depth,
        num_heads=config.patch_encoder_num_heads,
    )


def build_redae_encoder(config: FireRedTTS3Config) -> RedAEAudioEncoder:
    return RedAEAudioEncoder(
        out_dim=config.bottleneck_dim,
        audio_patch_size=config.audio_patch_size,
        audio_sample_rate=config.audio_sample_rate,
        hidden_size=config.enc_hidden_size,
        intermediate_size=config.enc_intermediate_size,
        num_hidden_layers=config.enc_num_hidden_layers,
        max_position_embeddings=config.enc_max_position_embeddings,
        max_window_layers=config.enc_max_window_layers,
        num_attention_heads=config.enc_num_attention_heads,
        num_key_value_heads=config.enc_num_key_value_heads,
        sliding_window=config.enc_sliding_window,
        use_sliding_window=config.enc_use_sliding_window,
        extra_downsample_rate=config.enc_extra_downsample_rate,
        downsample_num_hidden_layers=config.enc_downsample_num_hidden_layers,
    )


def build_redae_decoder(config: FireRedTTS3Config) -> RedAEAudioDecoder:
    return RedAEAudioDecoder(
        in_dim=config.bottleneck_dim,
        upsample_rate=config.enc_extra_downsample_rate,
        audio_patch_size=config.audio_patch_size,
        audio_sample_rate=config.audio_sample_rate,
        hidden_size=config.dec_hidden_size,
        intermediate_size=config.dec_intermediate_size,
        num_hidden_layers=config.dec_num_hidden_layers,
        max_position_embeddings=config.dec_max_position_embeddings,
        max_window_layers=config.dec_max_window_layers,
        num_attention_heads=config.dec_num_attention_heads,
        num_key_value_heads=config.dec_num_key_value_heads,
        sliding_window=config.dec_sliding_window,
        use_sliding_window=config.dec_use_sliding_window,
    )


def drop_embedding_tables(module: torch.nn.Module) -> None:
    """Set every embed_tokens submodule to None (inputs_embeds-only modules).

    Must run BEFORE load_state_dict so the skipped checkpoint keys and the
    module key space agree.
    """
    targets = [
        parent
        for parent in module.modules()
        for child_name, _ in parent.named_children()
        if child_name == "embed_tokens"
    ]
    for parent in targets:
        parent.embed_tokens = None


def load_dit(
    model_path: str, config: FireRedTTS3Config, *, device: torch.device | str
) -> DiT:
    module = build_dit(config)
    state = read_component_tensors(Path(model_path), "dit.")
    module.load_state_dict(state, strict=True)
    return module.to(device=device, dtype=torch.float32).eval()


def load_patch_encoder(
    model_path: str, config: FireRedTTS3Config, *, device: torch.device | str
) -> PatchEncoder:
    module = build_patch_encoder(config)
    state = read_component_tensors(Path(model_path), "patch_encoder.")
    module.load_state_dict(state, strict=True)
    return module.to(device=device, dtype=torch.float32).eval()


def load_redae_encoder(
    model_path: str, config: FireRedTTS3Config, *, device: torch.device | str
) -> RedAEAudioEncoder:
    module = build_redae_encoder(config)
    drop_embedding_tables(module)
    state = read_component_tensors(
        Path(model_path) / "redae",
        "encoder.",
        skip_if={"qwen3.embed_tokens.weight", "downsample.qwen3.embed_tokens.weight"},
    )
    module.load_state_dict(state, strict=True)
    return module.to(device=device, dtype=torch.float32).eval()


def load_redae_decoder(
    model_path: str, config: FireRedTTS3Config, *, device: torch.device | str
) -> RedAEAudioDecoder:
    module = build_redae_decoder(config)
    drop_embedding_tables(module)
    state = read_component_tensors(
        Path(model_path) / "redae",
        "decoder.",
        skip_if={"qwen3.embed_tokens.weight"},
    )
    module.load_state_dict(state, strict=True)
    return module.to(device=device, dtype=torch.float32).eval()


def load_campp(model_path: str, *, device: torch.device | str) -> CamppEmbedding:
    """Load the Base variant's CAM++ speaker encoder (fp32)."""
    weights_path = Path(model_path) / CAMPP_RELATIVE_PATH
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"FireRedTTS3 CAM++ weights not found at {weights_path}"
        )
    else:
        pass
    module = CamppEmbedding()
    state = torch.load(weights_path, weights_only=True, map_location="cpu")
    module.model.load_state_dict(state, strict=True)
    return module.to(device=device, dtype=torch.float32).eval()


__all__ = [
    "BACKBONE_LLM_PREFIX",
    "BACKBONE_LM_HEAD_PREFIX",
    "BACKBONE_MODEL_INFIX",
    "CAMPP_RELATIVE_PATH",
    "SIDE_PATH_PREFIXES",
    "build_dit",
    "build_patch_encoder",
    "build_redae_decoder",
    "build_redae_encoder",
    "drop_embedding_tables",
    "load_campp",
    "load_dit",
    "load_patch_encoder",
    "load_redae_decoder",
    "load_redae_encoder",
    "read_component_tensors",
    "route_root_key",
    "strip_backbone_prefix",
]
