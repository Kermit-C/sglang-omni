# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FireRedTTS3 prompt composition — templates, tokenization, patch budgets.

Everything the preprocessing stage needs to build engine prompts for all
three variants lives here, depending only on torch/torchaudio/soundfile and
the standard library so the (CPU-only) preprocessing stage stays cheap:

* ChatML composers for the four Instruct tasks — byte-identical to the
  reference fireredtts3/utils/chatml.py (verified by string diff tests).
* ChatML composer + char-modelled text encoder for the speaker variant
  (text_latent_interleave_ratio config) — byte-identical to the reference
  inference wrapper: text is an INPUT stream interleaved with the generated
  audio patches (no text-AR phase, no reference audio).
* Latent-patch arithmetic: the prompt's pad-slot count must match what the
  talker's prefill scatter produces, and both are pure arithmetic on the
  24 kHz resampled length (1 patch = 4 latent frames = 3840 samples).
* Per-task sampling defaults (reference inference settings: voice_design
  samples its text at 0.7/0.8/20; the rest run greedy — the audio phase's
  2-slot stop signal is temperature-robust either way).
"""

from __future__ import annotations

import base64
import io
import math
from functools import lru_cache
from typing import Any
from urllib.request import urlopen

import torch

SAMPLE_RATE = 24000
LATENT_HZ = 25  # 1 latent frame = 960 samples
PATCH_FRAMES = 4  # 1 patch = 4 latent frames = 3840 samples
MAX_EDIT_SECONDS = 20.0  # reference gradio cap on edit-source audio
MAX_TEXT_CHARS = 400

# Special-token ids for the Base/Instruct templates (one frozen tokenizer
# shared by those two variants; the speaker variant resolves ids per name —
# its tokenizer assigns 151669 to audio_holder, not sosp).
TEXT_EOT_ID = 151677  # <|eot|>  — terminates the text phase
AUDIO_SOS_ID = 151669  # <|sosp|> — opens the audio phase
LATENT_IN_PAD_ID = 151655  # <|image_pad|> — audio-input (edit source) slots
LATENT_OUT_PAD_ID = 151656  # <|video_pad|> — ICL audio-output slots

CHATML_LATENT_IN_PAD_SYM = "<|image_pad|>"
CHATML_LATENT_OUT_PAD_SYM = "<|video_pad|>"

# Reference DiT CFG per task (reference fireredtts3_instruct defaults).
TASK_CFG = {"tts": 2.0, "voice_design": 1.2, "semantic_edit": 1.2, "acoustic_edit": 1.2}

# Instruct text-AR phase (reference fireredtts3_instruct.generate): the model
# free-writes the voice plan (voice_design) or the edited transcript
# (semantic_edit) before the audio phase. The reference caps the text loop at
# range(200) — 200 sampled tokens, the 200th token's feed doubling as the
# no-sampling finalize feed.
TEXT_TOKEN_BUDGET = 200
TEXT_PHASE_TASKS = frozenset({"voice_design", "semantic_edit"})

# Output-token budget: 200 text tokens + 2 transition feeds + 400 audio
# patches (engine max_tokens counts every sampled token, including the
# 2-slot control dummies).
DEFAULT_MAX_TOKENS = 650

# Speaker-variant special tokens (strings; ids resolve per variant — the
# speaker checkpoint's frozen tokenizer carries these, Base/Instruct's do
# not, so a missing id is a clear "wrong variant" error rather than a silent
# mismatch).
SPEAKER_AUDIO_HOLDER = "<|audio_holder|>"
SPEAKER_TEXT_HOLDER = "<|text_holder|>"
SPEAKER_TEXT_PAD = "<|text_pad|>"
SPEAKER_AUDIO_START = "<|audio_start|>"
SPEAKER_AUDIO_END = "<|audio_end|>"

__all__ = [
    "AUDIO_SOS_ID",
    "CHATML_LATENT_IN_PAD_SYM",
    "CHATML_LATENT_OUT_PAD_SYM",
    "DEFAULT_MAX_TOKENS",
    "LATENT_HZ",
    "LATENT_IN_PAD_ID",
    "LATENT_OUT_PAD_ID",
    "MAX_EDIT_SECONDS",
    "MAX_TEXT_CHARS",
    "PATCH_FRAMES",
    "SAMPLE_RATE",
    "SPEAKER_AUDIO_END",
    "SPEAKER_AUDIO_HOLDER",
    "SPEAKER_AUDIO_START",
    "SPEAKER_TEXT_HOLDER",
    "SPEAKER_TEXT_PAD",
    "TASK_CFG",
    "TEXT_EOT_ID",
    "TEXT_PHASE_TASKS",
    "TEXT_TOKEN_BUDGET",
    "compose_acoustic_edit",
    "compose_acoustic_instruction",
    "compose_base_text",
    "compose_semantic_edit",
    "compose_speaker_chatml",
    "compose_tts",
    "compose_voice_design",
    "convert_to_chatml",
    "count_latent_patches",
    "encode_speaker_dual_stream_prompt",
    "encode_speaker_prefill_ids",
    "encode_speaker_text",
    "is_cjk_char",
    "load_audio",
    "resolve_speaker_token_id",
    "speaker_max_tokens",
    "strip_space_before_han",
    "task_sampling_kwargs",
    "tokenize_text",
]


# --- ChatML prompt composition (reference fireredtts3/utils/chatml.py) ---


def convert_to_chatml(
    text_in: str,
    latent_in_len: int = 0,
    text_out: str = "",
    latent_out_len: int = 0,
) -> str:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    input_message = {
        "role": "user",
        "content": [{"type": "text", "text": text_in + " /no_think"}],
    }
    if latent_in_len > 0:
        input_message["content"].insert(
            0, {"type": "audio", "audio": CHATML_LATENT_IN_PAD_SYM * latent_in_len}
        )
    messages.append(input_message)
    output_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": "<think>\n\n</think>\n\n" + text_out}],
    }
    if latent_out_len > 0:
        output_message["content"].append(
            {"type": "audio", "audio": CHATML_LATENT_OUT_PAD_SYM * latent_out_len}
        )
    messages.append(output_message)

    chatml_str_list: list[str] = []
    for msg in messages:
        if isinstance(msg["content"], str):
            chatml_str_list.append(
                f'<|im_start|>{msg["role"]}\n{msg["content"]}<|im_end|>\n'
            )
            continue
        chatml_str_list.append(f'<|im_start|>{msg["role"]}\n')
        for content in msg["content"]:
            if content["type"] == "text":
                chatml_str_list.append(content["text"])
            else:
                chatml_str_list.append(f"<|sosp|>{content['audio']}<|eosp|>\n")
        chatml_str_list.append("<|im_end|>\n")
    return "".join(chatml_str_list)


def compose_tts(prompt_latent_len: int, prompt_text: str, text: str) -> str:
    text_in = "Convert text to speech.\n{}".format(prompt_text + text)
    chatml = convert_to_chatml(text_in=text_in, latent_out_len=prompt_latent_len)
    return chatml.removesuffix("<|eosp|>\n<|im_end|>\n")


def compose_voice_design(instruction: str, text: str) -> str:
    text_in = (
        "{}\n\n根据上述音色描述，首先整理成语音属性，再合成以下文本对应的音频：\n{}"
    ).format(instruction, text)
    chatml = convert_to_chatml(text_in=text_in, text_out="<|sot|>")
    return chatml.removesuffix("<|im_end|>\n")


def compose_semantic_edit(instruction: str, audio_in_latent_len: int) -> str:
    text_in = "Identify the content of the audio. {}".format(instruction.strip())
    chatml = convert_to_chatml(
        text_in=text_in, latent_in_len=audio_in_latent_len, text_out="<|sot|>"
    )
    return chatml.removesuffix("<|im_end|>\n")


def compose_acoustic_edit(instruction: str, audio_in_latent_len: int) -> str:
    chatml = convert_to_chatml(
        text_in=instruction, latent_in_len=audio_in_latent_len, latent_out_len=1
    )
    return chatml.removesuffix("<|video_pad|><|eosp|>\n<|im_end|>\n")


def compose_acoustic_instruction(attribute: str, value: float) -> str:
    """Acoustic edits only accept the templates the model was trained on."""
    if attribute == "Speed":
        return f"adjust the speed to {value:.1f}x"
    if attribute == "Volume":
        return f"adjust the volume to {value:.1f}"
    steps = int(round(value))
    return f"shift the pitch by {steps} step{'' if abs(steps) == 1 else 's'}"


def compose_base_text(language: str, ref_text: str, text: str) -> str:
    """FireRedTTS3-Base (ICL voice-clone) text template.

    Mirrors the reference FireRedTTS3Base.generate:
    <|{language}|><|sot|>{ref_text}{text}<|eot|>.
    """
    return f"<|{language}|><|sot|>{ref_text}{text}<|eot|>"


def compose_speaker_chatml(
    speaker_token: str,
    *,
    style: str = "",
    text_len: int = 0,
    dual_stream: bool = False,
    interleave_ratio: tuple[int, int] = (2, 1),
) -> str:
    """Speaker-variant ChatML prefix (reference inference-wrapper
    format_chatml alignment, verified by string diff).

    The prompt always ENDS at <|audio_start|> — the reference composes one
    placeholder interleave group after it and strips the whole tail:

    * prefill form (dual_stream=False): user = "{style} Convert text to
      speech.\\n" + <|text_holder|>*text_len; the text ids are scattered
      over the holder slots at the TOKEN level by encode_speaker_prefill_ids
      (the reference tokenizes the template, then masked_scatters the
      standalone-encoded text — holder special tokens are hard BPE
      boundaries, so this is exact).
    * dual-stream form (dual_stream=True, the reference's streaming
      template): the text is NOT in the prompt. user = "{style} Align the
      interleaved text and speech chunks." and the text tokens ride the
      request payload for the talker's dual-stream decode machine (their
      positions depend on how many patches were generated, so they cannot
      be prefilled).

    style defaults mirror the reference call sites: no prefix for the
    prefill form, "Use a 中性 tone." for the dual-stream form.
    """
    if dual_stream:
        user_prompt = f"{style} Align the interleaved text and speech chunks.".strip()
        nt, na = interleave_ratio
        if nt < 1 or na < 1:
            raise ValueError(
                f"FireRedTTS3 interleave_ratio {interleave_ratio!r} invalid; "
                "expected [text_steps, audio_steps] with both >= 1"
            )
        # One placeholder group (1 real text slot + pads + audio slots),
        # exactly what the reference's compose_interleave(text_len=1,
        # latent_len=1) yields for the ratio — then stripped whole.
        assistant_out = (
            SPEAKER_TEXT_HOLDER
            + SPEAKER_TEXT_PAD * (nt - 1)
            + SPEAKER_AUDIO_HOLDER * na
        )
        suffix = f"{assistant_out}{SPEAKER_AUDIO_END}<|im_end|>\n"
    else:
        user_prompt = f"{style} Convert text to speech.".strip()
        user_prompt = "{}\n{}".format(user_prompt, SPEAKER_TEXT_HOLDER * text_len)
        assistant_out = ""
        suffix = f"{SPEAKER_AUDIO_END}<|im_end|>\n"
    chatml = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        f"{speaker_token}{SPEAKER_AUDIO_START}{assistant_out}"
        f"{SPEAKER_AUDIO_END}<|im_end|>\n"
    )
    return chatml.removesuffix(suffix)


# --- speaker-variant text encoding (char-modelled Chinese) ---
#
# The reference's tokenizer wrapper normalizes (strip spaces before Han
# characters — otherwise Qwen encodes "汉 字" into strange pieces) and then
# EXPANDS every vocab token that decodes to >= 2 CJK characters into its
# per-character tokens (Chinese is char-modelled in this variant). The
# expansion is applied lazily per token here (identical semantics — the
# reference precomputes the same map over the whole vocab and only ever
# hits tokens present in the text).


def is_cjk_char(c: str) -> bool:
    return (
        "一" <= c <= "鿿"
        or "㐀" <= c <= "䶿"
        or "豈" <= c <= "﫿"
        or "\U00020000" <= c <= "\U0002a6df"
    )


@lru_cache(maxsize=1)
def space_before_han_pattern():
    # regex is an optional dependency; the fallback spells out the
    # reference's own CJK test ranges for the stdlib re engine.
    try:
        import regex

        return regex.compile(r"\s+(?=\p{Han})")
    except ImportError:
        import re

        return re.compile(r"\s+(?=[一-鿿㐀-䶿豈-﫿\U00020000-\U0002a6df])")


def strip_space_before_han(text: str) -> str:
    return space_before_han_pattern().sub("", text)


def encode_speaker_text(tokenizer, text: str) -> list[int]:
    """Char-modelled text encoding for the speaker variant.

    strip spaces before Han characters -> tokenize (no special tokens) ->
    expand multi-char CJK vocab tokens to per-character ids. A character
    that fails to encode cleanly (unk) keeps the original multi-char token,
    exactly the reference wrapper's guard. Use for BOTH the template string
    and the standalone text (the reference runs its whole ChatML template
    through the same wrapper — e.g. the "中性" in the default streaming
    style splits into two tokens).
    """
    tokens = tokenizer(strip_space_before_han(text), add_special_tokens=False)[
        "input_ids"
    ]
    unk = getattr(tokenizer, "unk_token_id", None)
    out: list[int] = []
    for tid in tokens:
        decoded = tokenizer.decode([int(tid)]).strip()
        if len(decoded) < 2 or not all(is_cjk_char(c) for c in decoded):
            out.append(int(tid))
            continue
        char_ids: list[int] = []
        for ch in decoded:
            ids = tokenizer(ch, add_special_tokens=False)["input_ids"]
            if unk is not None and unk in ids:
                char_ids = []
                break
            char_ids.extend(int(i) for i in ids)
        if char_ids:
            out.extend(char_ids)
        else:
            out.append(int(tid))
    return out


def resolve_speaker_token_id(tokenizer, token: str) -> int:
    """Resolve a speaker-variant special-token id, failing loudly on a variant
    whose frozen tokenizer lacks the token (Base/Instruct)."""
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid is None:
        raise ValueError(
            f"FireRedTTS3 speaker requests require a tokenizer with the "
            f"{token} token — this model's tokenizer has none (the speaker "
            "variant is a separate checkpoint)"
        )
    return int(tid)


# --- audio loading + patch arithmetic ---


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Load audio from a file path, URL, or data URI -> ([channels, samples], sr)."""
    if path.startswith(("http://", "https://")):
        with urlopen(path, timeout=60) as response:
            audio_obj: Any = io.BytesIO(response.read())
    elif path.startswith("data:"):
        _, _, encoded = path.partition(",")
        audio_obj = io.BytesIO(base64.b64decode(encoded))
    else:
        audio_obj = path
    import soundfile as sf  # optional dependency of the light importer

    wav_np, sr = sf.read(audio_obj, dtype="float32")
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    return torch.from_numpy(wav_np), int(sr)


def count_latent_patches(wav: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
    """Resample to 24 kHz and count whole latent patches.

    The talker left-pads the audio to a whole number of 3840-sample patches
    before encoding, so the patch count is pure arithmetic on the resampled
    length — no encoder needed client-side. Returns
    (mono 24 kHz wav [1, samples], n_patches).
    """
    import torchaudio  # optional dependency of the light importer

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    wav = wav[:1].float()
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    multiple = SAMPLE_RATE // LATENT_HZ * PATCH_FRAMES  # 3840 samples/patch
    return wav, math.ceil(wav.shape[1] / multiple)


def tokenize_text(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


# --- speaker-variant request encoding ---


def encode_speaker_prefill_ids(
    tokenizer,
    *,
    text: str,
    speaker_token: str,
    style: str | None = None,
) -> list[int]:
    """Prefill-form speaker prompt: ChatML with the char-modelled text ids
    scattered into the <|text_holder|> slots (the reference's masked_scatter;
    holder special tokens are hard BPE boundaries, so token-level
    substitution is exact)."""
    text_ids = encode_speaker_text(tokenizer, text)
    if not text_ids:
        raise ValueError("FireRedTTS3 speaker text encoded to zero tokens")
    chatml = compose_speaker_chatml(
        speaker_token,
        style="" if style is None else style,
        text_len=len(text_ids),
    )
    ids = encode_speaker_text(tokenizer, chatml)
    holder_id = resolve_speaker_token_id(tokenizer, SPEAKER_TEXT_HOLDER)
    prompt_ids: list[int] = []
    fed = 0
    for tid in ids:
        if int(tid) == holder_id:
            prompt_ids.append(text_ids[fed])
            fed += 1
        else:
            prompt_ids.append(int(tid))
    if fed != len(text_ids):
        raise ValueError(
            f"FireRedTTS3 speaker template carries {fed} text slots "
            f"for {len(text_ids)} text tokens — composition bug"
        )
    return prompt_ids


def encode_speaker_dual_stream_prompt(
    tokenizer,
    *,
    text: str,
    speaker_token: str,
    style: str | None = None,
    interleave_ratio: tuple[int, int] = (2, 1),
) -> tuple[list[int], list[int], int]:
    """Dual-stream speaker prompt: (prompt_ids, text_stream_ids, text_pad_id).

    The prompt holds ONLY the prefix through <|audio_start|>; the
    char-modelled text ids feed the talker's dual-stream decode machine
    (nt text feeds + na patch feeds per na patches) and <|text_pad|> ids
    extend the text stream once it is exhausted (reference behaviour).
    """
    if not text.strip():
        raise ValueError("FireRedTTS3 speaker requests require non-empty text")
    text_ids = encode_speaker_text(tokenizer, text)
    if not text_ids:
        raise ValueError("FireRedTTS3 speaker text encoded to zero tokens")
    chatml = compose_speaker_chatml(
        speaker_token,
        style="Use a 中性 tone." if style is None else style,
        dual_stream=True,
        interleave_ratio=interleave_ratio,
    )
    prompt_ids = encode_speaker_text(tokenizer, chatml)
    text_pad_id = resolve_speaker_token_id(tokenizer, SPEAKER_TEXT_PAD)
    return prompt_ids, text_ids, text_pad_id


def speaker_max_tokens(
    max_patches: int,
    interleave_ratio: tuple[int, int] | None = None,
    slack: int = 20,
) -> int:
    """Output-token budget for a speaker-checkpoint request.

    Prefill form: one sampled token per generated patch. Dual-stream form:
    the decode machine consumes nt text feeds + na patch feeds per na
    patches — (nt + na) / na tokens per patch. Independent of the output
    mode (sync/async).
    """
    if interleave_ratio is None:
        return int(max_patches) + slack
    nt, na = interleave_ratio
    per_patch = (int(nt) + int(na)) / int(na)
    return int(math.ceil(per_patch * int(max_patches))) + slack


# --- sampling defaults ---


def task_sampling_kwargs(task: str, seed: int | None = None) -> dict[str, Any]:
    """Per-task sampling fields (reference inference settings).

    voice_design SAMPLES its text (temp 0.7 / top_p 0.8 / top_k 20); the
    other tasks run greedy (semantic_edit's text is greedy by reference
    design; tts / acoustic_edit have no text phase). The audio phase's
    2-slot stop signal is margin-robust, so these stay in force for the
    whole request. Callers fold in execution knobs (max_tokens, seed,
    detokenize=False) when constructing the engine sampling params.
    """
    if task == "voice_design":
        kw: dict[str, Any] = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
    else:
        kw = {"temperature": 0.0}
    if seed is not None:
        kw["seed"] = seed
    return kw
