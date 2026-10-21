# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HuggingFace config adapter for FireRedTTS3 checkpoints.

Registers FireRedTTS3Config (model_type fireredtts3) so that
AutoConfig.from_pretrained returns the correct config class.

One flat config covers every FireRedTTS3 variant (Base / Instruct / speaker):
variant behaviour is driven entirely by field values (use_lm_head,
dit_in_channels, spk_in_dim, speaker_token, special_tokens ...), never by
subclassing. Special tokens are stored by NAME and must be resolved against
the checkpoint's frozen tokenizer at load time — token IDs collide across
variants (id 151669 is <|sosp|> in Base/Instruct checkpoints but
<|audio_holder|> in speaker ones), so IDs are never hard-coded here.
"""

from __future__ import annotations

from typing import Literal

from transformers import AutoConfig, PretrainedConfig

from sglang_omni.utils.json import JsonValue

FIREREDTTS3_MODEL_ARCH_OVERRIDE = "FireRedTTS3ForConditionalGeneration"

FireRedTTS3Variant = Literal["base", "instruct", "speaker"]


class FireRedTTS3Config(PretrainedConfig):
    """Token-free TTS: Qwen3-1.7B backbone + flow-matching DiT head +
    PatchEncoder (talker stage) and a RedAE iSTFT decoder (vocoder stage).
    """

    model_type: str = "fireredtts3"

    def __init__(
        self,
        # Backbone (Qwen3-1.7B fields, flat; mirrors the reference
        # implementation's backbone config).
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 40960,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1e6,
        tie_word_embeddings: bool = True,
        attention_bias: bool = False,
        max_window_layers: int = 28,
        use_sliding_window: bool = False,
        sliding_window: int | None = None,
        bos_token_id: int = 151643,
        eos_token_id: int = 151645,
        # Acoustic modules (official FireRedTTS3 field names).
        redae_dim: int = 64,
        num_history_patches: int = 2,
        patch_size: int = 4,
        patch_encoder_hidden_size: int = 1024,
        patch_encoder_mlp_ratio: int = 4,
        patch_encoder_depth: int = 8,
        patch_encoder_num_heads: int = 16,
        dit_mlp_ratio: int = 3,
        dit_depth: int = 11,
        dit_num_heads: int = 16,
        dit_hidden_size: int = 1024,
        # DiT input width: redae_dim + (spk_in_dim or 0) + dit_hidden_size
        # (Base: 64+512+1024=1600; Instruct / speaker: 64+1024=1088).
        dit_in_channels: int = 1088,
        # Speaker-embedding input dim; Base only, absent otherwise.
        spk_in_dim: int | None = None,
        # RedAE (vocoder stage) fields; identical across all variants.
        audio_patch_size: int = 480,
        audio_sample_rate: int = 24000,
        bottleneck_dim: int = 64,
        enc_hidden_size: int = 896,
        enc_intermediate_size: int = 3584,
        enc_max_position_embeddings: int = 32768,
        enc_max_window_layers: int = 0,
        enc_num_attention_heads: int = 14,
        enc_num_hidden_layers: int = 18,
        enc_num_key_value_heads: int = 2,
        enc_sliding_window: int = 64,
        enc_use_sliding_window: bool = True,
        enc_downsample_num_hidden_layers: int = 4,
        enc_extra_downsample_rate: int = 2,
        dec_hidden_size: int = 896,
        dec_intermediate_size: int = 3584,
        dec_max_position_embeddings: int = 32768,
        dec_max_window_layers: int = 0,
        dec_num_attention_heads: int = 14,
        dec_num_hidden_layers: int = 18,
        dec_num_key_value_heads: int = 2,
        dec_sliding_window: int = 64,
        dec_use_sliding_window: bool = True,
        # Variant switches.
        # True only when the backbone lm_head drives a text-AR phase
        # (Instruct); False for latent-only loops (Base / speaker).
        use_lm_head: bool = False,
        # Constant latent scaling: DiT / PatchEncoder operate in
        # latent * redae_scale space; the decoder divides it back out.
        redae_scale: float = 0.4,
        # Optional speaker-as-token inserted into the prompt prefix.
        speaker_token: str | None = None,
        # (n_text, n_latent) tokens per group for interleaved streaming
        # templates. Checkpoint-level (must match the training
        # distribution); None = non-interleaved.
        text_latent_interleave_ratio: list[int] | None = None,
        # Special tokens by NAME (resolve IDs from the frozen tokenizer at
        # load time, never hard-code).
        special_tokens: dict[str, str | None] | None = None,
        # Generation defaults (request-overridable).
        generation_defaults: dict[str, int | float] | None = None,
        **kwargs: JsonValue,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.max_window_layers = max_window_layers
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        # Acoustic modules
        self.redae_dim = redae_dim
        self.num_history_patches = num_history_patches
        self.patch_size = patch_size
        self.patch_encoder_hidden_size = patch_encoder_hidden_size
        self.patch_encoder_mlp_ratio = patch_encoder_mlp_ratio
        self.patch_encoder_depth = patch_encoder_depth
        self.patch_encoder_num_heads = patch_encoder_num_heads
        self.dit_mlp_ratio = dit_mlp_ratio
        self.dit_depth = dit_depth
        self.dit_num_heads = dit_num_heads
        self.dit_hidden_size = dit_hidden_size
        self.dit_in_channels = dit_in_channels
        self.spk_in_dim = spk_in_dim
        # RedAE (vocoder stage)
        self.audio_patch_size = audio_patch_size
        self.audio_sample_rate = audio_sample_rate
        self.bottleneck_dim = bottleneck_dim
        self.enc_hidden_size = enc_hidden_size
        self.enc_intermediate_size = enc_intermediate_size
        self.enc_max_position_embeddings = enc_max_position_embeddings
        self.enc_max_window_layers = enc_max_window_layers
        self.enc_num_attention_heads = enc_num_attention_heads
        self.enc_num_hidden_layers = enc_num_hidden_layers
        self.enc_num_key_value_heads = enc_num_key_value_heads
        self.enc_sliding_window = enc_sliding_window
        self.enc_use_sliding_window = enc_use_sliding_window
        self.enc_downsample_num_hidden_layers = enc_downsample_num_hidden_layers
        self.enc_extra_downsample_rate = enc_extra_downsample_rate
        self.dec_hidden_size = dec_hidden_size
        self.dec_intermediate_size = dec_intermediate_size
        self.dec_max_position_embeddings = dec_max_position_embeddings
        self.dec_max_window_layers = dec_max_window_layers
        self.dec_num_attention_heads = dec_num_attention_heads
        self.dec_num_hidden_layers = dec_num_hidden_layers
        self.dec_num_key_value_heads = dec_num_key_value_heads
        self.dec_sliding_window = dec_sliding_window
        self.dec_use_sliding_window = dec_use_sliding_window
        # Variant switches
        self.use_lm_head = use_lm_head
        self.redae_scale = redae_scale
        self.speaker_token = speaker_token
        self.text_latent_interleave_ratio = text_latent_interleave_ratio
        self.special_tokens = special_tokens
        self.generation_defaults = generation_defaults
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


def detect_fireredtts3_variant(config: FireRedTTS3Config) -> FireRedTTS3Variant:
    """Classify a checkpoint into its FireRedTTS3 variant from config fields."""
    if config.text_latent_interleave_ratio is not None:
        return "speaker"
    else:
        pass
    if config.spk_in_dim is not None:
        return "base"
    else:
        pass
    if config.use_lm_head:
        return "instruct"
    else:
        pass
    raise ValueError(
        "unrecognized FireRedTTS3 checkpoint: none of the variant markers "
        "(text_latent_interleave_ratio / spk_in_dim / use_lm_head) is set"
    )


def register_fireredtts3_hf_config() -> None:
    """Register the local FireRedTTS3 config before SGLang loads ModelConfig."""
    AutoConfig.register("fireredtts3", FireRedTTS3Config, exist_ok=True)


__all__ = [
    "FIREREDTTS3_MODEL_ARCH_OVERRIDE",
    "FireRedTTS3Config",
    "FireRedTTS3Variant",
    "detect_fireredtts3_variant",
    "register_fireredtts3_hf_config",
]
