# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the FireRedTTS3 pipeline.

preprocessing (CPU): Base ICL template compose + tokenize + patch budget;
reference_encode (GPU, light): CAM++ + RedAE encoder conditioning artifacts
through the shared content-addressed ReferenceEncodeService, with the prompt
latents streamed to the vocoder ahead of the talker (prompt pre-decode);
talker: the SGLang AR engine with the fused flow side path, streaming
ramp-sized latent chunks to the vocoder; vocoder: RedAE decoder incremental
streaming plus the BatchVocoder full-payload path.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from sglang_omni.models.firered_tts3.flow_graph import FlowMode
from sglang_omni.models.firered_tts3.hf_config import (
    FireRedTTS3Config,
    detect_fireredtts3_variant,
)
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.prompt_utils import (
    MAX_EDIT_SECONDS,
    TASK_CFG,
    TEXT_PHASE_TASKS,
    TEXT_TOKEN_BUDGET,
    compose_acoustic_edit,
    compose_acoustic_instruction,
    compose_base_text,
    compose_semantic_edit,
    compose_tts,
    compose_voice_design,
    count_latent_patches,
    encode_speaker_dual_stream_prompt,
    encode_speaker_prefill_ids,
    load_audio,
    speaker_max_tokens,
    tokenize_text,
)
from sglang_omni.models.firered_tts3.request_builders import ENGINE_TOKEN_SLACK
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_data_uri_from_reference
from sglang_omni.utils.checkpoint import resolve_checkpoint

logger = logging.getLogger(__name__)

# Reference FireRedTTS3Base.generate language tags (fireredtts3/utils/
# text_tokenizer.py); a request language must map to exactly one of these.
MULTI_LANG_TAGS = (
    "Chinese",
    "English",
    "Cantonese",
    "Japanese",
    "Korean",
    "Spanish",
    "French",
    "Russian",
    "Arabic",
    "Turkish",
    "Indonesian",
    "Portuguese",
    "Italian",
    "Dutch",
    "Vietnamese",
    "German",
    "Ukrainian",
    "Thai",
    "Polish",
    "Romanian",
    "Greek",
    "Czech",
    "Finnish",
    "Hindi",
)
MULTI_DIALECT_TAGS = tuple(
    f"ZH_{name}"
    for name in (
        "Anhui",
        "Fujian",
        "Gansu",
        "Guizhou",
        "Hebei",
        "Henan",
        "Hubei",
        "Hunan",
        "Jiangxi",
        "Liaoning",
        "Minnan",
        "Ningxia",
        "Shaanxi",
        "Shandong",
        "Shanghai",
        "Shanxi",
        "Sichuan",
        "Tianjin",
        "Wenzhou",
        "Wu",
        "Yunnan",
    )
)

MAX_TEXT_CHARS = 400

# The serving task_type spellings (normalized: lowercase, no separators) map
# onto the reference's four Instruct tasks; "base"/"tts" both mean ICL
# cloning (the Qwen3-TTS convention of inferring tts from a reference).
INSTRUCT_TASK_ALIASES = {
    "tts": "tts",
    "base": "tts",
    "voicedesign": "voice_design",
    "semanticedit": "semantic_edit",
    "acousticedit": "acoustic_edit",
}
ACOUSTIC_ATTRIBUTES = ("Speed", "Pitch", "Volume")


def resolve_request_reference(
    inputs: dict[str, Any], tts_params: dict[str, Any]
) -> tuple[str | None, Any]:
    """The single reference clip (path/data URI) and its transcript, if any."""
    references = inputs.get("references")
    if isinstance(references, list) and len(references) > 1:
        raise ValueError("FireRedTTS3 accepts at most one reference audio")
    reference = (
        references[0]
        if isinstance(references, list)
        and references
        and isinstance(references[0], dict)
        else {}
    )
    audio = first_not_none(
        reference_path(inputs.get("prompt_audio_path")),
        reference_path(inputs.get("prompt_audio")),
        reference_path(inputs.get("reference_audio")),
        reference_path(tts_params.get("ref_audio")),
        reference_path(reference),
    )
    text = first_not_none(
        inputs.get("prompt_text"),
        inputs.get("reference_text"),
        tts_params.get("ref_text"),
        reference.get("text"),
    )
    return audio, text


def normalize_instruct_task(value: Any) -> str:
    task = (
        INSTRUCT_TASK_ALIASES.get(
            str(value).strip().replace("_", "").replace("-", "").lower()
        )
        if value is not None
        else None
    )
    if task is None:
        if value is None:
            return "tts"
        raise ValueError(
            f"unsupported FireRedTTS3-Instruct task_type {value!r}; expected one "
            "of VoiceDesign / SemanticEdit / AcousticEdit (or ref_audio tts)"
        )
    return task


def count_audio_patches(audio_path: str, *, cap_seconds: float | None) -> int:
    """Whole latent patches of a clip — pure arithmetic on the 24 kHz length.

    The reference_encode stage left-pads to whole patches before encoding, so
    the encoded patch count matches ceil(samples / 3840) exactly; the chatml
    pad-slot count must agree (the talker's masked_scatter checks it).
    """
    wav, sr = load_audio(audio_path)
    if cap_seconds is not None:
        wav = wav[..., : int(sr * cap_seconds)]
    _, patches = count_latent_patches(wav, sr)
    return patches


def resolve_text_sampling(
    task: str, params: dict[str, Any], tts_params: dict[str, Any]
) -> tuple[float, float | None, int | None]:
    """Text-phase sampling: the reference defaults per task, with explicit
    client generation fields (tracked by the serving layer) taking precedence.
    """
    if task == "voice_design":
        temperature, top_p, top_k = 0.7, 0.8, 20
    else:
        temperature, top_p, top_k = 0.0, None, None
    explicit = set(tts_params.get("explicit_generation_params") or [])
    if "temperature" in explicit:
        temperature = float(params["temperature"])
    if "top_p" in explicit:
        top_p = float(params["top_p"])
    if "top_k" in explicit:
        top_k = int(params["top_k"])
    return temperature, top_p, top_k


def resolve_acoustic_instruction(
    instructions: str | None,
    tts_params: dict[str, Any],
    preprocessing_params: dict[str, Any],
) -> str:
    """Acoustic edits take the fixed trained templates; an explicit
    instructions field overrides as a raw template (reference behavior)."""
    if instructions is not None and instructions.strip():
        return instructions.strip()
    attribute = str(
        first_not_none(
            preprocessing_params.get("acoustic_attribute"),
            tts_params.get("acoustic_attribute"),
            default="Speed",
        )
    )
    attribute = attribute.strip().capitalize()
    if attribute not in ACOUSTIC_ATTRIBUTES:
        raise ValueError(
            f"acoustic_attribute must be one of {ACOUSTIC_ATTRIBUTES}, "
            f"got {attribute!r}"
        )
    value = first_not_none(
        preprocessing_params.get("acoustic_value"),
        tts_params.get("acoustic_value"),
        tts_params.get("speed"),
        default=1.0,
    )
    return compose_acoustic_instruction(attribute, float(value))


def first_not_none(*values: Any, default: Any = None) -> Any:
    return next((value for value in values if value is not None), default)


def as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def request_inputs(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"text": value}
    return as_dict(value)


def reference_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        path = (
            value.get("audio_path")
            or value.get("path")
            or audio_data_uri_from_reference(value)
        )
        return str(path) if path else None
    raise TypeError("FireRedTTS3 reference audio must be a path or data URI")


def preprocess_firered_tts3_payload(
    payload: StagePayload,
    *,
    tokenizer: Any,
    config: FireRedTTS3Config,
    n_timesteps: int,
    max_patches: int,
) -> StagePayload:
    from sglang_omni.models.firered_tts3.payload_types import store_firered_tts3_state

    inputs = request_inputs(payload.request.inputs)
    params = as_dict(payload.request.params)
    tts_params = as_dict(as_dict(payload.request.metadata).get("tts_params"))
    stage_params = as_dict(params.get("stage_params"))
    engine_params = as_dict(stage_params.get("talker"))
    preprocessing_params = as_dict(stage_params.get("preprocessing"))

    stream = bool(params.get("stream", False))

    text = str(first_not_none(inputs.get("input"), inputs.get("text"), default=""))
    if len(text.strip()) > MAX_TEXT_CHARS:
        raise ValueError(
            f"FireRedTTS3 input text must be at most {MAX_TEXT_CHARS} characters"
        )

    generation = dict(config.generation_defaults or {})

    def resolve_number(key: str, fallback: Any) -> Any:
        return first_not_none(
            inputs.get(key),
            tts_params.get(key),
            engine_params.get(key),
            params.get(key),
            default=fallback,
        )

    seed = first_not_none(
        inputs.get("seed"),
        tts_params.get("seed"),
        engine_params.get("seed"),
        params.get("seed"),
    )

    variant = detect_fireredtts3_variant(config)
    interleave_ratio = (
        (
            int(config.text_latent_interleave_ratio[0]),
            int(config.text_latent_interleave_ratio[1]),
        )
        if config.text_latent_interleave_ratio is not None
        else None
    )
    if variant == "instruct":
        task = normalize_instruct_task(
            first_not_none(
                inputs.get("task_type"),
                tts_params.get("task_type"),
                params.get("task_type"),
            )
        )
        state = compose_instruct_state(
            config,
            tokenizer=tokenizer,
            task=task,
            inputs=inputs,
            params=params,
            tts_params=tts_params,
            preprocessing_params=preprocessing_params,
            text=text,
        )
    elif variant == "base":
        task = "tts"
        state = compose_base_state(
            config,
            tokenizer=tokenizer,
            inputs=inputs,
            tts_params=tts_params,
            text=text,
        )
    elif variant == "speaker":
        task = "tts"
        state = compose_speaker_state(
            config,
            tokenizer=tokenizer,
            inputs=inputs,
            params=params,
            tts_params=tts_params,
            preprocessing_params=preprocessing_params,
            text=text,
        )
    else:
        raise AssertionError(f"unhandled FireRedTTS3 variant: {variant}")

    state.n_timesteps = int(
        resolve_number("n_timesteps", generation.get("n_timesteps", n_timesteps))
    )
    # The per-task CFG table is the Instruct reference's; Base/speaker tts
    # requests take the checkpoint's generation_defaults (2.0 / 1.2).
    cfg_default = generation.get("inference_cfg", 2.0)
    if variant == "instruct":
        cfg_default = TASK_CFG.get(state.task_type, cfg_default)
    state.inference_cfg = float(resolve_number("inference_cfg", cfg_default))
    state.stop_threshold = float(
        resolve_number("stop_threshold", generation.get("stop_threshold", 0.5))
    )
    state.min_patches = int(
        resolve_number("min_patches", generation.get("min_patches", 6))
    )
    state.max_patches = int(resolve_number("max_patches", max_patches))
    state.seed = None if seed is None else int(seed)
    state.stream = stream
    state.max_engine_tokens = int(
        params.get("max_new_tokens")
        or engine_token_budget(state, interleave_ratio=interleave_ratio)
    )
    return store_firered_tts3_state(payload, state)


def engine_token_budget(
    state: FireRedTTS3State, *, interleave_ratio: tuple[int, int] | None = None
) -> int:
    """Decode-step budget: one per patch (dual-stream: (nt + na) / na per
    patch), plus the text-AR and transition feeds for Instruct text-phase
    tasks (200 text + finalize + sosp)."""
    if state.dual_stream:
        if interleave_ratio is None:
            raise ValueError(
                "FireRedTTS3 dual-stream requests require the checkpoint's "
                "text_latent_interleave_ratio"
            )
        return speaker_max_tokens(state.max_patches, interleave_ratio)
    budget = state.max_patches + ENGINE_TOKEN_SLACK
    if state.task_type in TEXT_PHASE_TASKS:
        budget += TEXT_TOKEN_BUDGET + 2
    return budget


def compose_base_state(
    config: FireRedTTS3Config,
    *,
    tokenizer: Any,
    inputs: dict[str, Any],
    tts_params: dict[str, Any],
    text: str,
) -> FireRedTTS3State:
    if not text.strip():
        raise ValueError("FireRedTTS3 requires non-empty input text")
    text = text.strip()
    prompt_audio, prompt_text_value = resolve_request_reference(inputs, tts_params)
    if prompt_audio is None:
        raise ValueError("FireRedTTS3-Base requires a reference audio (voice cloning)")
    if prompt_text_value is None or not str(prompt_text_value).strip():
        raise ValueError(
            "FireRedTTS3-Base requires the reference audio transcript "
            "(ref_text / references[0].text)"
        )
    prompt_text = str(prompt_text_value).strip()

    language = str(
        first_not_none(
            inputs.get("language"), tts_params.get("language"), default="Chinese"
        )
    )
    if language not in MULTI_LANG_TAGS + MULTI_DIALECT_TAGS:
        raise ValueError(
            f"unsupported FireRedTTS3 language {language!r}; expected one of "
            "MULTI_LANG_TAGS/MULTI_DIALECT_TAGS"
        )

    return FireRedTTS3State(
        sample_rate=int(config.audio_sample_rate),
        variant="base",
        task_type="tts",
        text=text,
        prompt_audio_path=prompt_audio,
        prompt_text=prompt_text,
        language=language,
        input_ids=tokenize_text(
            tokenizer, compose_base_text(language, prompt_text, text)
        ),
    )


def compose_speaker_state(
    config: FireRedTTS3Config,
    *,
    tokenizer: Any,
    inputs: dict[str, Any],
    params: dict[str, Any],
    tts_params: dict[str, Any],
    preprocessing_params: dict[str, Any],
    text: str,
) -> FireRedTTS3State:
    """Speaker variant (text_latent_interleave_ratio in the checkpoint
    config): text-only tts — a speaker token carries the voice, so there is
    no reference audio and no text-AR phase.

    The decode form is a request-level opt-in (extra_params dual_stream);
    the output mode (stream) is orthogonal. The reference passes its style
    prefix only on the streaming (dual-stream) call site, so style applies
    to that form only.
    """
    if not text.strip():
        raise ValueError("FireRedTTS3 speaker requests require non-empty input text")
    reference_audio, prompt_text = resolve_request_reference(inputs, tts_params)
    if reference_audio is not None or (
        prompt_text is not None and str(prompt_text).strip()
    ):
        raise ValueError(
            "FireRedTTS3 speaker variant takes text-only tts; reference audio "
            "is not used (deploy a Base/Instruct checkpoint for voice cloning)"
        )
    task = first_not_none(
        inputs.get("task_type"), tts_params.get("task_type"), params.get("task_type")
    )
    if task is not None and str(task).strip().lower() not in ("tts", "base"):
        raise ValueError(
            f"FireRedTTS3 speaker variant supports only text-only tts; "
            f"got task_type {task!r}"
        )
    dual_stream = bool(
        first_not_none(
            params.get("dual_stream"),
            tts_params.get("dual_stream"),
            preprocessing_params.get("dual_stream"),
            default=False,
        )
    )
    ratio = config.text_latent_interleave_ratio
    if dual_stream and ratio is None:
        raise ValueError(
            "FireRedTTS3 dual_stream requires a checkpoint with "
            "text_latent_interleave_ratio in its config"
        )
    speaker_token = str(config.speaker_token or "<|rand_speaker|>")
    if dual_stream:
        style = first_not_none(
            params.get("style"),
            tts_params.get("style"),
            preprocessing_params.get("style"),
        )
        prompt_ids, text_ids, text_pad_id = encode_speaker_dual_stream_prompt(
            tokenizer,
            text=text.strip(),
            speaker_token=speaker_token,
            style=None if style is None else str(style),
            interleave_ratio=(int(ratio[0]), int(ratio[1])),
        )
        return FireRedTTS3State(
            sample_rate=int(config.audio_sample_rate),
            variant="speaker",
            task_type="tts",
            text=text.strip(),
            input_ids=prompt_ids,
            dual_stream=True,
            text_stream_ids=text_ids,
            text_pad_id=text_pad_id,
        )
    return FireRedTTS3State(
        sample_rate=int(config.audio_sample_rate),
        variant="speaker",
        task_type="tts",
        text=text.strip(),
        input_ids=encode_speaker_prefill_ids(
            tokenizer, text=text.strip(), speaker_token=speaker_token
        ),
    )


def compose_instruct_state(
    config: FireRedTTS3Config,
    *,
    tokenizer: Any,
    task: str,
    inputs: dict[str, Any],
    params: dict[str, Any],
    tts_params: dict[str, Any],
    preprocessing_params: dict[str, Any],
    text: str,
) -> FireRedTTS3State:
    instructions = first_not_none(
        inputs.get("instructions"), tts_params.get("instructions")
    )
    if instructions is not None:
        instructions = str(instructions).strip() or None
    reference_audio, prompt_text_value = resolve_request_reference(inputs, tts_params)

    prompt_audio: str | None = None
    audio_in: str | None = None
    prompt_text: str | None = None
    if task == "tts":
        if not text.strip():
            raise ValueError("FireRedTTS3 tts requires non-empty input text")
        if reference_audio is None:
            raise ValueError(
                "FireRedTTS3-Instruct tts requires a reference audio "
                "(ref_audio / references[0])"
            )
        if prompt_text_value is None or not str(prompt_text_value).strip():
            raise ValueError(
                "FireRedTTS3-Instruct tts requires the reference audio "
                "transcript (ref_text / references[0].text)"
            )
        prompt_audio = reference_audio
        prompt_text = str(prompt_text_value).strip()
        prompt_patches = count_audio_patches(prompt_audio, cap_seconds=None)
        chatml = compose_tts(prompt_patches, prompt_text, text.strip())
    elif task == "voice_design":
        if not text.strip():
            raise ValueError("FireRedTTS3 voice_design requires non-empty input text")
        if not instructions:
            raise ValueError("FireRedTTS3 voice_design requires instructions")
        chatml = compose_voice_design(instructions, text.strip())
    elif task == "semantic_edit":
        if not instructions:
            raise ValueError("FireRedTTS3 semantic_edit requires instructions")
        audio_in = require_edit_source(reference_audio, task)
        edit_patches = count_audio_patches(audio_in, cap_seconds=MAX_EDIT_SECONDS)
        chatml = compose_semantic_edit(instructions, edit_patches)
    elif task == "acoustic_edit":
        audio_in = require_edit_source(reference_audio, task)
        instruction = resolve_acoustic_instruction(
            instructions, tts_params, preprocessing_params
        )
        edit_patches = count_audio_patches(audio_in, cap_seconds=MAX_EDIT_SECONDS)
        chatml = compose_acoustic_edit(instruction, edit_patches)
    else:
        raise AssertionError(f"unhandled FireRedTTS3-Instruct task: {task}")

    temperature, top_p, top_k = resolve_text_sampling(task, params, tts_params)
    return FireRedTTS3State(
        sample_rate=int(config.audio_sample_rate),
        variant="instruct",
        task_type=task,
        text=text.strip(),
        prompt_audio_path=prompt_audio,
        prompt_text=prompt_text,
        instructions=instructions,
        audio_in_path=audio_in,
        input_ids=tokenize_text(tokenizer, chatml),
        text_temperature=temperature,
        text_top_p=top_p,
        text_top_k=top_k,
    )


def require_edit_source(reference_audio: str | None, task: str) -> str:
    if reference_audio is None:
        raise ValueError(
            f"FireRedTTS3 {task} requires a reference audio (the edit source)"
        )
    return reference_audio


def create_preprocessing_executor(
    model_path: str,
    *,
    n_timesteps: int = 10,
    max_patches: int = 400,
    max_concurrency: int = 8,
) -> SimpleScheduler:
    from transformers import AutoTokenizer

    from sglang_omni.models.firered_tts3.config import load_checkpoint_hf_config
    from sglang_omni.models.firered_tts3.hf_config import register_fireredtts3_hf_config

    register_fireredtts3_hf_config()
    root = str(resolve_checkpoint(model_path))
    config = load_checkpoint_hf_config(root)
    if config is None:
        raise FileNotFoundError(f"FireRedTTS3 config.json not found at {root}")
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)

    def _preprocess(payload: StagePayload) -> StagePayload:
        return preprocess_firered_tts3_payload(
            payload,
            tokenizer=tokenizer,
            config=config,
            n_timesteps=n_timesteps,
            max_patches=max_patches,
        )

    return SimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_reference_encode_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_concurrency: int = 8,
    max_batch_size: int = 1,
    max_batch_wait_ms: float = 4.0,
    warmup_on_init: bool = True,
    reference_cache: bool = True,
) -> SimpleScheduler:
    from sglang_omni.models.firered_tts3.config import load_checkpoint_hf_config
    from sglang_omni.models.firered_tts3.ref_encode import (
        FireRedReferenceBundle,
        FireRedReferenceEncoder,
    )
    from sglang_omni.utils.device import resolve_concrete_device

    concrete_device = resolve_concrete_device(device, gpu_id)
    if concrete_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("FireRedTTS3 reference encode requires CUDA")
    root = str(resolve_checkpoint(model_path))
    config = load_checkpoint_hf_config(root)
    if config is None:
        raise FileNotFoundError(f"FireRedTTS3 config.json not found at {root}")
    bundle = FireRedReferenceBundle(root, config, str(concrete_device))
    if warmup_on_init:
        # The first served request would otherwise pay the encoders' cold
        # start in its TTFP; the bundle warms them with a dummy clip, off
        # the reference cache.
        bundle.warmup()
    encoder = FireRedReferenceEncoder(
        bundle,
        model_id=root,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        cache_enabled=reference_cache,
    )

    def encode_and_stream_prompt(payload: StagePayload) -> StagePayload:
        from sglang_omni.models.firered_tts3.ref_encode import prompt_stream_message

        result = encoder.encode_payload(payload)
        # The prompt chunk rides ahead of the talker's latent chunks so the
        # vocoder can warm its decoder state during the talker's prefill. The
        # scheduler name resolves at call time (construction finishes first).
        chunk = prompt_stream_message(result)
        if chunk is not None:
            scheduler.outbox.put(chunk)
        return result

    scheduler = SimpleScheduler(
        encode_and_stream_prompt,
        max_concurrency=max_concurrency,
        shutdown_callback=encoder.close,
    )
    return scheduler


def create_talker_executor(
    model_path: str,
    *,
    precision: str = "bfloat16",
    n_timesteps: int = 10,
    max_patches: int = 400,
    flow_cuda_graph: bool = True,
    flow_cuda_graph_capture_sizes: list[int] | None = None,
    flow_cuda_graph_max_graphs: int = 8,
    flow_batching: bool = True,
    flow_mode: FlowMode = "compile_bf16",
    warmup_on_init: bool = True,
    enable_async_decode: bool = False,
    async_decode_min_batch_size: int = 2,
    device: str | None = None,
    gpu_id: int | None = None,
    server_args_overrides: dict[str, Any] | None = None,
) -> OmniScheduler:
    from sglang_omni.models.firered_tts3.engine_builder import FireRedTTS3EngineBuilder

    if not torch.cuda.is_available():
        raise RuntimeError("FireRedTTS3 talker requires CUDA")
    return FireRedTTS3EngineBuilder(
        n_timesteps=n_timesteps,
        max_patches=max_patches,
        flow_cuda_graph=flow_cuda_graph,
        flow_cuda_graph_capture_sizes=flow_cuda_graph_capture_sizes,
        flow_cuda_graph_max_graphs=flow_cuda_graph_max_graphs,
        flow_batching=flow_batching,
        flow_mode=flow_mode,
        warmup_on_init=warmup_on_init,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=precision,
        server_args_overrides=server_args_overrides,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_batch_size: int = 4,
    max_batch_wait_ms: float = 2.0,
    stream_slots: int = 16,
    prompt_predecode: bool = False,
    stream_batching: bool = True,
    separate_decode_stream: bool = True,
    incremental_cuda_graph: bool = True,
    cuda_graph_latent_sizes: list[int] | None = None,
    cuda_graph_batch_sizes: list[int] | None = None,
    cold_cuda_graph: bool = True,
    cold_cuda_graph_latent_sizes: list[int] | None = None,
    cold_cuda_graph_batch_sizes: list[int] | None = None,
    warmup_on_init: bool = True,
) -> StreamingVocoderBase:
    from sglang_omni.models.firered_tts3.config import load_checkpoint_hf_config
    from sglang_omni.models.firered_tts3.vocoder import FireRedTTS3StreamingVocoder
    from sglang_omni.models.firered_tts3.weights import load_redae_decoder
    from sglang_omni.utils.device import resolve_concrete_device

    if not torch.cuda.is_available():
        raise RuntimeError("FireRedTTS3 vocoder requires CUDA")
    concrete_device = resolve_concrete_device(device, gpu_id)
    root = str(resolve_checkpoint(model_path))
    config = load_checkpoint_hf_config(root)
    if config is None:
        raise FileNotFoundError(f"FireRedTTS3 config.json not found at {root}")
    decoder = load_redae_decoder(root, config, device=concrete_device)
    # stream_slots bounds concurrent stream states; admission is owned by the
    # talker's max_running_requests (config.py enforces the match).
    scheduler = FireRedTTS3StreamingVocoder(
        decoder,
        redae_scale=float(config.redae_scale or 1.0),
        sample_rate=int(config.audio_sample_rate),
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        stream_slots=stream_slots,
        prompt_predecode=prompt_predecode,
        stream_batching=stream_batching,
        separate_decode_stream=separate_decode_stream,
        incremental_cuda_graph=incremental_cuda_graph,
        cuda_graph_latent_sizes=cuda_graph_latent_sizes,
        cuda_graph_batch_sizes=cuda_graph_batch_sizes,
        cold_cuda_graph=cold_cuda_graph,
        cold_cuda_graph_latent_sizes=cold_cuda_graph_latent_sizes,
        cold_cuda_graph_batch_sizes=cold_cuda_graph_batch_sizes,
    )
    # Factory construction completes before the stage publishes readiness, so
    # the warmup cannot overlap request-time GPU work from colocated stages.
    if warmup_on_init:
        scheduler.warmup_now()
    return scheduler


__all__ = [
    "create_preprocessing_executor",
    "create_reference_encode_executor",
    "create_talker_executor",
    "create_vocoder_executor",
    "preprocess_firered_tts3_payload",
]
