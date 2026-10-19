# SPDX-License-Identifier: Apache-2.0
"""SGLang model for FireRedTTS3: Qwen3 backbone plus the fp32 flow side path.

The AR backbone is the stock SGLang Qwen3ForCausalLM (stage dtype, bf16 in
production). Everything else — patch_encoder / dit / dit_head / stop_head /
(Base) spk_proj_llm / spk_proj_dit — is a precision-red-line fp32 side path
driven by the model runner between backbone steps (1 generated latent patch
= 1 engine decode position; the PatchEncoder output is fed back as the next
step's input embedding).

The RedAE encoder/decoder and CAM++ live in the reference_encode / vocoder
stages, not here. Weight routing for the root checkpoint (backbone_llm.* +
side-path prefixes, the speaker variant's tied lm_head copy dropped) is
shared with the
stage-side loaders in weights.py.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from torch import nn
from transformers import Qwen3Config

from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.weights import (
    build_dit,
    build_patch_encoder,
    route_root_key,
    strip_backbone_prefix,
)
from sglang_omni.models.weight_loader import default_weight_loader

# Backbone fields are carried flat on FireRedTTS3Config, mirroring the
# reference's Qwen3-1.7B config.
BACKBONE_CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "tie_word_embeddings",
    "attention_bias",
    "max_window_layers",
    "use_sliding_window",
    "sliding_window",
    "bos_token_id",
    "eos_token_id",
)


def backbone_qwen3_config(config: FireRedTTS3Config) -> Qwen3Config:
    return Qwen3Config(
        **{name: getattr(config, name) for name in BACKBONE_CONFIG_FIELDS}
    )


class FireRedTTS3SGLangModel(nn.Module):
    """Compose the SGLang Qwen3 backbone with the FireRedTTS3 side path."""

    # note: class-level default so partially-constructed instances (tests
    # building via __new__) resolve the graph-feedback probe to disabled.
    graph_feedback_buffer_value: torch.Tensor | None = None
    # Set by the runner before every forward: text-phase (Instruct text-AR)
    # rows need real full-vocab logits from the tied lm_head; audio-phase
    # rows keep the dummy 1-wide logits (the runner overwrites their sampled
    # token). The decode CUDA graph bakes whichever branch runs at capture,
    # so with the feedback buffer installed decode always takes the lm_head
    # branch (see forward); this flag then only steers eager prefill.
    text_logits_requested: bool = False

    def __init__(self, config: Any, quant_config: Any = None, prefix: str = "") -> None:
        super().__init__()
        if not isinstance(config, FireRedTTS3Config):
            raise ValueError(
                "FireRedTTS3 requires the registered FireRedTTS3Config; got "
                f"{type(config).__name__}"
            )
        self.config = config
        self.backbone = Qwen3ForCausalLM(
            backbone_qwen3_config(config),
            quant_config=quant_config,
            prefix=f"{prefix}.backbone" if prefix else "backbone",
        )
        # Precision red line: the side path is pinned to fp32 before any
        # weight is copied, so the loader's stage-dtype context cannot round
        # it down (copy_ casts to the parameter dtype).
        self.patch_encoder = build_patch_encoder(config).to(dtype=torch.float32)
        self.dit = build_dit(config).to(dtype=torch.float32)
        self.dit_head = nn.Linear(config.hidden_size, config.dit_hidden_size)
        self.stop_head = nn.Linear(config.hidden_size, 1)
        self.spk_proj_llm = (
            nn.Linear(config.spk_in_dim, config.hidden_size)
            if config.spk_in_dim is not None
            else None
        )
        self.spk_proj_dit = (
            nn.Linear(config.spk_in_dim, config.spk_in_dim)
            if config.spk_in_dim is not None
            else None
        )
        for module in (
            self.dit_head,
            self.stop_head,
            self.spk_proj_llm,
            self.spk_proj_dit,
        ):
            if module is not None:
                module.to(dtype=torch.float32)

        self.llm_hidden_size = int(config.hidden_size)
        self.redae_dim = int(config.redae_dim)
        self.patch_size = int(config.patch_size)
        self.history_patches = int(config.num_history_patches)
        self.history_length = self.history_patches * self.patch_size
        self.redae_scale = float(config.redae_scale or 1.0)
        generation = dict(config.generation_defaults or {})
        self.default_n_timesteps = int(generation.get("n_timesteps", 10))
        self.default_inference_cfg = float(generation.get("inference_cfg", 2.0))
        self.default_stop_threshold = float(generation.get("stop_threshold", 0.5))
        self.default_min_patches = int(generation.get("min_patches", 6))
        self.default_max_patches = int(generation.get("max_patches", 400))
        self.graph_feedback_buffer_value: torch.Tensor | None = None

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    @property
    def graph_feedback_buffer(self) -> torch.Tensor | None:
        return self.graph_feedback_buffer_value

    def enable_graph_feedback(self, max_batch_size: int) -> None:
        """Route decode feedback embeddings through a persistent buffer.

        SGLang's decode CUDA graph stages forward_batch.input_embeds only for
        dFlash draft workers, so a graph replay would silently ignore the
        per-step latent feedback. Owning the static buffer model-side bakes
        this address into the capture; before_decode copies each step's
        feedback into it, and the same read path serves eager decode.
        """
        if max_batch_size <= 0:
            raise ValueError("FireRedTTS3 graph feedback buffer needs a positive size")
        parameter = next(self.backbone.parameters())
        self.graph_feedback_buffer_value = torch.zeros(
            (int(max_batch_size), self.llm_hidden_size),
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def warmup_side_path(self) -> None:
        """Warm the fp32 side-path modules with their hot shapes.

        The engine's profiling dummy run covers the backbone and the flow
        graph capture covers the DiT loop, but patch_encoder / dit_head /
        stop_head / spk_proj run only from the runner hooks, so the first
        served request would otherwise pay their lazy init (measured on L20:
        a few ms across prefill_embeds and cond — small, but on the TTFP
        critical path). Runs synchronously on the current stream; the dummy
        tensors are zeros, so no RNG stream is touched.
        """
        device = next(self.backbone.parameters()).device
        with torch.inference_mode():
            prompt = torch.zeros(1, 4 * self.patch_size, self.redae_dim, device=device)
            self.patch_encoder(prompt)
            feedback = torch.zeros(1, self.patch_size, self.redae_dim, device=device)
            self.patch_encoder(feedback)
            hidden = torch.zeros(
                1, self.history_patches + 1, self.llm_hidden_size, device=device
            )
            self.dit_head(hidden)
            self.stop_head(hidden[:, -1:])
            if self.spk_proj_llm is not None and self.spk_proj_dit is not None:
                spk = torch.zeros(1, int(self.config.spk_in_dim), device=device)
                self.spk_proj_llm(spk)
                self.spk_proj_dit(spk)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        side_params = dict(self.side_path_parameters())
        loaded: set[str] = set()
        skipped = 0
        for name, tensor in weights:
            bucket = route_root_key(name)
            if bucket == "backbone":
                backbone_weights.append((strip_backbone_prefix(name), tensor))
                continue
            if bucket == "lm_head":
                # Speaker checkpoints ship a tied lm_head copy; the head is
                # tied to embed_tokens at load, so the copy is dropped.
                skipped += 1
                continue
            if bucket == "unknown":
                raise ValueError(f"unexpected FireRedTTS3 checkpoint key {name!r}")
            parameter = side_params.get(name)
            assert parameter is not None, (
                f"FireRedTTS3 checkpoint key {name!r} routed to {bucket} but no "
                "such side-path parameter exists"
            )
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, tensor)
            loaded.add(name)
        self.backbone.load_weights(backbone_weights)
        # Pin the side path to fp32 and onto the backbone device after the
        # loader's stage-dtype pass; the backbone keeps the stage dtype.
        device = next(self.backbone.parameters()).device
        for module in self.side_path_modules():
            module.to(device=device, dtype=torch.float32)
        return loaded

    def side_path_modules(self) -> list[nn.Module]:
        modules: list[nn.Module] = [
            self.patch_encoder,
            self.dit,
            self.dit_head,
            self.stop_head,
        ]
        if self.spk_proj_llm is not None:
            modules.append(self.spk_proj_llm)
        if self.spk_proj_dit is not None:
            modules.append(self.spk_proj_dit)
        return modules

    def side_path_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        for module, prefix in (
            (self.patch_encoder, "patch_encoder"),
            (self.dit, "dit"),
            (self.dit_head, "dit_head"),
            (self.stop_head, "stop_head"),
            (self.spk_proj_llm, "spk_proj_llm"),
            (self.spk_proj_dit, "spk_proj_dit"),
        ):
            if module is None:
                continue
            for name, parameter in module.named_parameters():
                yield f"{prefix}.{name}", parameter

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        del kwargs
        if (
            self.graph_feedback_buffer_value is not None
            and forward_batch.forward_mode.is_decode()
        ):
            # Same source for capture, replay, and eager decode; rows beyond
            # the live batch are padding and their outputs are discarded.
            input_embeds = self.graph_feedback_buffer_value[: input_ids.shape[0]]
        elif input_embeds is None:
            input_embeds = forward_batch.input_embeds
        hidden_states = self.backbone.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        if self.config.use_lm_head and (
            self.text_logits_requested
            or (
                self.graph_feedback_buffer_value is not None
                and forward_batch.forward_mode.is_decode()
            )
        ):
            # Text-AR sampling rows: real full-vocab logits via the tied
            # lm_head. The hidden states ride through per the batch's
            # capture mode (FULL for prefill, last-position for decode).
            # A captured decode graph replays one baked branch, so with the
            # feedback buffer installed every decode step computes real
            # logits: text-phase steps stay samplable under replay, and the
            # audio phase pays one discarded lm_head GEMM per step.
            return self.backbone.logits_processor(
                input_ids, hidden_states, self.backbone.lm_head, forward_batch
            )
        # The runner consumes hidden states only and overwrites
        # next_token_ids; dummy logits just satisfy the contract.
        if forward_batch.forward_mode.is_extend():
            extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
            request_count = (
                int(extend_seq_lens.numel()) if extend_seq_lens is not None else 1
            )
        else:
            request_count = int(hidden_states.shape[0])
        return LogitsProcessorOutput(
            next_token_logits=hidden_states.new_empty((request_count, 1)),
            hidden_states=hidden_states,
        )


EntryClass = FireRedTTS3SGLangModel

__all__ = ["EntryClass", "FireRedTTS3SGLangModel", "backbone_qwen3_config"]
