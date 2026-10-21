# SPDX-License-Identifier: Apache-2.0
"""FireRedTTS3 SGLang engine builder (talker stage)."""

from __future__ import annotations

import logging
import time
from typing import Protocol

import torch
from sglang.srt.server_args import ServerArgs

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.firered_tts3.compiled_dit import CompiledDiTExecutor
from sglang_omni.models.firered_tts3.flow_graph import (
    FLOW_MODES,
    FlowGraphRunner,
    FlowMode,
    prepare_flow_dit,
)
from sglang_omni.models.firered_tts3.model_runner import FireRedTTS3ModelRunner
from sglang_omni.models.firered_tts3.request_builders import (
    FireRedTTS3SGLangRequestData,
)
from sglang_omni.models.firered_tts3.sglang_model import FireRedTTS3SGLangModel
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SchedulerExtras,
    TtsEngineBuilder,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor

logger = logging.getLogger(__name__)

# Prompt (1 speaker slot + text + prompt-patch slots) plus a decode step per
# patch stay far below this cap; it only bounds scheduler bookkeeping.
CONTEXT_LENGTH = 8192


class FireRedRequestBuilder(Protocol):
    def __call__(self, payload: StagePayload) -> FireRedTTS3SGLangRequestData: ...


class FireRedResultAdapter(Protocol):
    def __call__(self, data: FireRedTTS3SGLangRequestData) -> StagePayload: ...


class FireRedAbortCallback(Protocol):
    def __call__(self, request_id: str) -> None: ...


class FireRedTTS3EngineBuilder(TtsEngineBuilder[FireRedTTS3SGLangRequestData]):
    model_name = "fireredtts3"
    context_length = CONTEXT_LENGTH

    def __init__(
        self,
        *,
        n_timesteps: int = 10,
        max_patches: int = 400,
        max_running_requests: int = 16,
        flow_cuda_graph: bool = True,
        flow_cuda_graph_capture_sizes: list[int] | None = None,
        flow_cuda_graph_max_graphs: int = 8,
        flow_batching: bool = True,
        control_token_fastpath: bool = True,
        skip_min_patch_stop: bool = True,
        flow_mode: FlowMode = "compile_bf16",
        warmup_on_init: bool = True,
        # One-step-lookahead decode loop (framework event_loop_async_decode):
        # the launch of step N+1 overlaps the resolve of step N. The closed
        # AR/flow loop makes the side path launch-bound, so this only hides
        # scheduler host work. Measured on L20 (2026-09, on top of the
        # decode-stream fix): c=1 is the sync fast path (bit-exact), c=8 RTF
        # 0.338 vs 0.340-0.346 and c=16 RTF unchanged — inside run variance,
        # so it stays off by default.
        enable_async_decode: bool = False,
        async_decode_min_batch_size: int = 2,
    ) -> None:
        from sglang_omni.models.firered_tts3.hf_config import (
            FIREREDTTS3_MODEL_ARCH_OVERRIDE,
        )

        self.model_arch_override = FIREREDTTS3_MODEL_ARCH_OVERRIDE
        self.n_timesteps = int(n_timesteps)
        self.max_patches = int(max_patches)
        self.max_running_requests = int(max_running_requests)
        self.flow_cuda_graph = bool(flow_cuda_graph)
        self.flow_cuda_graph_capture_sizes = (
            sorted({int(s) for s in flow_cuda_graph_capture_sizes if int(s) > 0})
            or None
            if flow_cuda_graph_capture_sizes
            else None
        )
        self.flow_cuda_graph_max_graphs = int(flow_cuda_graph_max_graphs)
        # Cross-request flow batching (one batched DiT replay per engine
        # step; default on). Bit-exact at B=1 (exact
        # bucket hit); c>=2 drift is the bounded cross-batch GEMM tiling
        # envelope.
        self.flow_batching = bool(flow_batching)
        self.control_token_fastpath = bool(control_token_fastpath)
        self.skip_min_patch_stop = bool(skip_min_patch_stop)
        # The DiT execution stack inside the flow loop (see
        # flow_graph.prepare_flow_dit). compile_bf16 is the measured
        # throughput default (L20 Base c=1 RTF 0.353 -> 0.223,
        # stop trajectories identical); fp32 is the bit-exact reference.
        # Instruct pins fp32 in its deploy yaml: every non-bit-exact mode
        # can flip its knife-edge stop head (measured).
        if flow_mode not in FLOW_MODES:
            raise ValueError(
                f"FireRedTTS3 flow_mode {flow_mode!r} invalid; "
                f"expected one of {FLOW_MODES}"
            )
        else:
            pass
        self.flow_mode: FlowMode = flow_mode
        # Post-init side-path warmup (patch_encoder / dit_head / stop_head /
        # spk_proj) on the assembly-line tail; the engine dummy run and the
        # flow graph capture already cover the backbone and the DiT loop.
        self.warmup_on_init = bool(warmup_on_init)
        self.enable_async_decode = bool(enable_async_decode)
        self.async_decode_min_batch_size = int(async_decode_min_batch_size)
        if min(self.n_timesteps, self.max_patches, self.max_running_requests) <= 0:
            raise ValueError("FireRedTTS3 batching limits must be positive")
        else:
            pass
        if self.flow_cuda_graph_max_graphs <= 0:
            raise ValueError("FireRedTTS3 flow_cuda_graph_max_graphs must be positive")
        else:
            pass
        if self.async_decode_min_batch_size <= 0:
            raise ValueError("FireRedTTS3 async_decode_min_batch_size must be positive")
        else:
            pass
        self.model_runner: FireRedTTS3ModelRunner | None = None
        self.flow_graphs: FlowGraphRunner | None = None
        self.flow_dit_stack: (
            tuple[torch.nn.Module | CompiledDiTExecutor, torch.nn.Module] | None
        ) = None

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir
        from sglang_omni.models.firered_tts3.hf_config import (
            register_fireredtts3_hf_config,
        )

        register_fireredtts3_hf_config()

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            # The decode CUDA graph replays the backbone step; the per-step
            # latent feedback reaches it through the model-owned persistent
            # buffer (setup_model installs it before capture). Refuse the
            # graph only via an explicit engine.disable_cuda_graph override.
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            # Requests share no reusable prefix (each carries its own latent
            # conditioning); request_builders asserts prefix_len == 0.
            "disable_radix_cache": True,
            "enable_torch_compile": False,
            "max_running_requests": self.max_running_requests,
            "chunked_prefill_size": 0,
            "mem_fraction_static": 0.20,
            # The Instruct text phase seeds top_p/top_k sampling per request;
            # the flashinfer backend rejects seeded filtered sampling.
            "sampling_backend": "pytorch",
            "dtype": dtype,
            "trust_remote_code": False,
        }

    def adjust_overrides(
        self, overrides: dict[str, int | float | str | bool | None]
    ) -> None:
        if int(overrides.get("tp_size", 1)) != 1:
            raise ValueError("FireRedTTS3 supports tp_size=1 only")
        else:
            pass
        requested = int(
            overrides.get("max_running_requests", self.max_running_requests)
        )
        if requested <= 0:
            raise ValueError("FireRedTTS3 max_running_requests must be positive")
        else:
            pass
        self.max_running_requests = requested
        overrides["disable_radix_cache"] = True
        overrides["chunked_prefill_size"] = 0
        if bool(overrides.get("enable_torch_compile", False)):
            raise ValueError(
                "FireRedTTS3 compiles its flow side path directly; SGLang "
                "backbone compile is disabled"
            )
        else:
            pass
        if not bool(overrides.get("disable_cuda_graph", True)):
            # The decode graph must be captured with hidden states (FULL); its
            # can_run gate requires an exact hidden-mode match with the
            # runner's per-step request.
            overrides["enable_return_hidden_states"] = True
        else:
            pass

    def setup_model(
        self,
        *,
        model_worker: ModelWorker,
        checkpoint_dir: str,
        device: str,
        gpu_id: int,
        server_args: ServerArgs,
    ) -> None:
        del checkpoint_dir, device, gpu_id
        from sglang.srt.runtime_context import get_exec, get_schedule

        model = model_worker.model_runner.model
        model.eval()
        logger.info(
            "FireRedTTS3 talker dtypes: backbone_embed=%s qkv0=%s patch_encoder=%s",
            next(model.backbone.parameters()).dtype,
            model.backbone.model.layers[0].self_attn.qkv_proj.weight.dtype,
            next(model.patch_encoder.parameters()).dtype,
        )
        if bool(get_exec().graph.disable_cuda_graph):
            logger.info("FireRedTTS3 backbone decode: eager")
            return
        else:
            pass
        from sglang_omni.scheduling.generation_batch_policy import (
            get_decode_cuda_graph_max_bs,
        )

        # Installed before init_cuda_graphs so capture bakes the buffer
        # address into the decode graph. Generously sized: rows are tiny
        # (hidden_size elements) and capture may pad above
        # max_running_requests.
        max_running_requests = int(get_schedule().max_running_requests)
        model.enable_graph_feedback(
            max(
                max_running_requests,
                int(get_decode_cuda_graph_max_bs(server_args) or 0),
                256,
            )
        )
        logger.info(
            "FireRedTTS3 backbone decode: SGLang CUDA graph with model-owned "
            "feedback buffer"
        )

    def compile_model(
        self, model: FireRedTTS3SGLangModel, server_args: ServerArgs
    ) -> None:
        """Build the flow mode's DiT stack, before any graph capture.

        The assembly line runs this hook before init_sglang_cuda_graphs and
        setup_model_resources, so the torch.compile wrap lands ahead of both
        the engine decode graph and the flow-loop graphs (compile's own
        cudagraphs stay off; the two graph kinds never capture into each
        other). The wrap itself is lazy — inductor compiles per bucket shape
        inside the flow pre-capture warmup, so capture doubles as the
        compile warm-up and no served request pays the compile tax.
        """
        del server_args
        if not self.flow_cuda_graph:
            logger.info(
                f"FireRedTTS3 flow_mode={self.flow_mode} ignored: the flow "
                "CUDA graph is disabled"
            )
            return
        else:
            pass
        started = time.perf_counter()
        model.weights_locked = True
        self.flow_dit_stack = prepare_flow_dit(model.dit, self.flow_mode)
        logger.info(
            f"FireRedTTS3 flow mode {self.flow_mode}: DiT stack prepared in "
            f"{time.perf_counter() - started:.1f}s (lazy compile on capture)"
        )

    def setup_model_resources(
        self,
        model: FireRedTTS3SGLangModel,
        server_args: ServerArgs,
        *,
        generation_cuda_graph_enabled: bool,
    ) -> None:
        """Pre-capture the flow-loop CUDA graphs, after SGLang's own graphs.

        The assembly line runs this after init_sglang_cuda_graphs, so the
        engine decode graph and this model-local graph never capture into
        each other. The flow DiT stack is built in compile_model, which
        runs earlier, preserving the compile-then-capture order.
        """
        del server_args, generation_cuda_graph_enabled
        if self.flow_cuda_graph:
            self.capture_flow_graphs(model)
        else:
            logger.info("FireRedTTS3 flow CUDA graph: disabled")
        if self.warmup_on_init:
            model.warmup_side_path()
            logger.info("FireRedTTS3 talker side-path warmup complete")
        else:
            pass

    def capture_flow_graphs(self, model: FireRedTTS3SGLangModel) -> None:
        """Pre-capture the flow-loop CUDA graphs for the hot batch buckets."""
        from sglang.srt.runtime_context import get_schedule

        from sglang_omni.models.firered_tts3.flow_graph import (
            DEFAULT_CAPTURE_SIZES,
            FlowGraphRunner,
            flow_t_span,
        )

        dit, dit_fallback = (
            self.flow_dit_stack
            if self.flow_dit_stack is not None
            else (model.dit, model.dit)
        )
        device = next(model.dit.parameters()).device
        if device.type != "cuda":
            logger.warning(
                "FireRedTTS3 flow CUDA graph needs a CUDA device; staying eager"
            )
            return
        else:
            pass
        cond_channels = int(model.config.dit_hidden_size) + int(
            model.config.spk_in_dim or 0
        )
        max_running = int(get_schedule().max_running_requests)
        # Buckets above the admission limit can never be hit; an over-bucket
        # batch (not possible below the limit) would capture lazily instead.
        sizes = [
            size
            for size in (self.flow_cuda_graph_capture_sizes or DEFAULT_CAPTURE_SIZES)
            if size <= max_running
        ]
        runner = FlowGraphRunner(
            dit,
            flow_t_span(self.n_timesteps, device),
            n_timesteps=self.n_timesteps,
            patch_size=model.patch_size,
            history_length=model.history_length,
            max_graphs=max(self.flow_cuda_graph_max_graphs, len(sizes)),
            capture_sizes=sizes,
            mode=self.flow_mode,
            dit_fallback=dit_fallback,
        )
        reserved_before = torch.cuda.memory_reserved(device)
        started = time.perf_counter()
        for size in sizes:
            hist = torch.zeros(
                size, model.history_length, model.redae_dim, device=device
            )
            noise = torch.zeros(size, model.patch_size, model.redae_dim, device=device)
            cond = torch.zeros(
                size,
                model.history_length + model.patch_size,
                cond_channels,
                device=device,
            )
            cfgs = cond.new_full((size, 1, 1), 2.0)
            if runner.run(hist, noise, cond, cfgs, self.n_timesteps) is None:
                logger.warning(
                    "FireRedTTS3 flow CUDA graph pre-capture failed; "
                    "the flow side path stays eager"
                )
                return
            else:
                pass
        self.flow_graphs = runner
        reserved_mib = (torch.cuda.memory_reserved(device) - reserved_before) / 2**20
        logger.info(
            f"FireRedTTS3 flow CUDA graphs captured: sizes={sizes} "
            f"mode={self.flow_mode} reserved +{reserved_mib:.0f} MiB "
            f"in {time.perf_counter() - started:.1f}s"
        )

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> FireRedTTS3ModelRunner:
        from sglang_omni.models.firered_tts3.model_runner import FireRedTTS3ModelRunner

        self.model_runner = FireRedTTS3ModelRunner(
            model_worker,
            output_proc,
            flow_graphs=self.flow_graphs,
            flow_batching=self.flow_batching,
            control_token_fastpath=self.control_token_fastpath,
            skip_min_patch_stop=self.skip_min_patch_stop,
        )
        return self.model_runner

    def make_adapters(
        self, model: FireRedTTS3SGLangModel
    ) -> tuple[FireRedRequestBuilder, FireRedResultAdapter]:
        from transformers import AutoTokenizer

        from sglang_omni.models.firered_tts3.request_builders import (
            apply_latent_result,
            build_sglang_firered_tts3_request,
        )

        # The tokenizer decodes the text-phase tokens into gen_text at apply
        # time (the Instruct plan / edited transcript the client sees).
        text_tokenizer = (
            AutoTokenizer.from_pretrained(self.checkpoint_dir, local_files_only=True)
            if bool(getattr(model.config, "use_lm_head", False))
            else None
        )
        # The dual-stream decode form (speaker checkpoints) validates and
        # paces itself against the checkpoint's interleave ratio.
        raw_ratio = getattr(model.config, "text_latent_interleave_ratio", None)
        interleave_ratio = (
            (int(raw_ratio[0]), int(raw_ratio[1])) if raw_ratio is not None else None
        )

        def _build_request(payload: StagePayload) -> FireRedTTS3SGLangRequestData:
            return build_sglang_firered_tts3_request(
                payload,
                patch_size=model.patch_size,
                vocab_size=int(model.config.vocab_size),
                interleave_ratio=interleave_ratio,
            )

        def _apply_result(data: FireRedTTS3SGLangRequestData) -> StagePayload:
            return apply_latent_result(data, text_tokenizer=text_tokenizer)

        return _build_request, _apply_result

    def make_abort_callback(self) -> FireRedAbortCallback | None:
        assert self.model_runner is not None
        return self.model_runner.reset_request

    def extra_scheduler_kwargs(self) -> SchedulerExtras[FireRedTTS3SGLangRequestData]:
        from sglang_omni.models.firered_tts3.request_builders import build_stream_output

        return {
            "stream_output_builder": build_stream_output,
            "enable_async_decode": self.enable_async_decode,
            "async_decode_min_batch_size": self.async_decode_min_batch_size,
        }


__all__ = ["FireRedTTS3EngineBuilder"]
