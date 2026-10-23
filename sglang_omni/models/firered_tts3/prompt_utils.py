# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FireRedTTS3 prompt templates and speaker text encoding."""

from __future__ import annotations

import math

import regex
from transformers import PreTrainedTokenizerBase

MAX_EDIT_SECONDS = 20.0

# note (kermit): Base/Instruct share these ids; speaker checkpoints resolve tokens by name.
TEXT_EOT_ID = 151677
AUDIO_SOS_ID = 151669
LATENT_IN_PAD_ID = 151655
LATENT_OUT_PAD_ID = 151656

CHATML_LATENT_IN_PAD_SYM = "<|image_pad|>"
CHATML_LATENT_OUT_PAD_SYM = "<|video_pad|>"

TEXT_TOKEN_BUDGET = 200
TEXT_PHASE_TASKS = frozenset({"voice_design", "semantic_edit"})

SPEAKER_TEXT_HOLDER = "<|text_holder|>"
SPEAKER_TEXT_PAD = "<|text_pad|>"
SPEAKER_AUDIO_START = "<|audio_start|>"
SPACE_BEFORE_HAN_PATTERN = regex.compile(r"\s+(?=\p{Han})")


def convert_to_chatml(
    text_in: str,
    latent_in_len: int = 0,
    text_out: str = "",
    latent_out_len: int = 0,
) -> str:
    audio_input = (
        f"<|sosp|>{CHATML_LATENT_IN_PAD_SYM * latent_in_len}<|eosp|>\n"
        if latent_in_len > 0
        else ""
    )
    audio_output = (
        f"<|sosp|>{CHATML_LATENT_OUT_PAD_SYM * latent_out_len}<|eosp|>\n"
        if latent_out_len > 0
        else ""
    )
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{audio_input}{text_in} /no_think<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{text_out}{audio_output}<|im_end|>\n"
    )


def compose_tts(prompt_latent_len: int, prompt_text: str, text: str) -> str:
    text_in = f"Convert text to speech.\n{prompt_text}{text}"
    chatml = convert_to_chatml(text_in=text_in, latent_out_len=prompt_latent_len)
    return chatml.removesuffix("<|eosp|>\n<|im_end|>\n")


def compose_voice_design(instruction: str, text: str) -> str:
    text_in = f"{instruction}\n\n根据上述音色描述，首先整理成语音属性，再合成以下文本对应的音频：\n{text}"
    chatml = convert_to_chatml(text_in=text_in, text_out="<|sot|>")
    return chatml.removesuffix("<|im_end|>\n")


def compose_semantic_edit(instruction: str, audio_in_latent_len: int) -> str:
    text_in = f"Identify the content of the audio. {instruction.strip()}"
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
    else:
        pass
    if attribute == "Volume":
        return f"adjust the volume to {value:.1f}"
    else:
        pass
    steps = int(round(value))
    return f"shift the pitch by {steps} step{'' if abs(steps) == 1 else 's'}"


def compose_base_text(language: str, ref_text: str, text: str) -> str:
    """Place the reference transcript before the requested Base text."""
    return f"<|{language}|><|sot|>{ref_text}{text}<|eot|>"


def compose_speaker_chatml(
    speaker_token: str,
    *,
    style: str = "",
    text_len: int = 0,
    dual_stream: bool = False,
    interleave_ratio: tuple[int, int] = (2, 1),
) -> str:
    """Build the speaker prefix through its audio-start token."""
    if dual_stream:
        user_prompt = f"{style} Align the interleaved text and speech chunks.".strip()
        text_steps, audio_steps = interleave_ratio
        if text_steps < 1 or audio_steps < 1:
            raise ValueError(
                f"FireRedTTS3 interleave_ratio {interleave_ratio!r} invalid; "
                "expected [text_steps, audio_steps] with both >= 1"
            )
        else:
            pass
    else:
        user_prompt = f"{style} Convert text to speech.".strip()
        user_prompt = f"{user_prompt}\n{SPEAKER_TEXT_HOLDER * text_len}"
    return (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        f"{speaker_token}{SPEAKER_AUDIO_START}"
    )


def is_cjk_char(character: str) -> bool:
    return (
        "一" <= character <= "鿿"
        or "㐀" <= character <= "䶿"
        or "豈" <= character <= "﫿"
        or "\U00020000" <= character <= "\U0002a6df"
    )


def encode_speaker_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    """Expand multi-character Han tokens while preserving unencodable words."""
    text = SPACE_BEFORE_HAN_PATTERN.sub("", text)
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    unknown_token_id = tokenizer.unk_token_id
    out: list[int] = []
    for token_id in tokens:
        decoded = tokenizer.decode([int(token_id)]).strip()
        if len(decoded) < 2 or not all(is_cjk_char(c) for c in decoded):
            out.append(int(token_id))
            continue
        else:
            pass
        char_ids: list[int] = []
        for character in decoded:
            ids = tokenizer(character, add_special_tokens=False)["input_ids"]
            if unknown_token_id is not None and unknown_token_id in ids:
                char_ids = []
                break
            else:
                pass
            char_ids.extend(int(i) for i in ids)
        if char_ids:
            out.extend(char_ids)
        else:
            out.append(int(token_id))
    return out


def resolve_speaker_token_id(tokenizer: PreTrainedTokenizerBase, token: str) -> int:
    """Reject a tokenizer missing a required speaker token."""
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        raise ValueError(
            f"FireRedTTS3 speaker requests require a tokenizer with the "
            f"{token} token — this model's tokenizer has none (the speaker "
            "variant is a separate checkpoint)"
        )
    else:
        pass
    return int(token_id)


def tokenize_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def encode_speaker_prefill_ids(
    tokenizer: PreTrainedTokenizerBase,
    *,
    text: str,
    speaker_token: str,
    style: str | None = None,
) -> list[int]:
    """Replace speaker text holders with the character-level token stream."""
    text_ids = encode_speaker_text(tokenizer, text)
    if not text_ids:
        raise ValueError("FireRedTTS3 speaker text encoded to zero tokens")
    else:
        pass
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
    else:
        pass
    return prompt_ids


def encode_speaker_dual_stream_prompt(
    tokenizer: PreTrainedTokenizerBase,
    *,
    text: str,
    speaker_token: str,
    style: str | None = None,
    interleave_ratio: tuple[int, int] = (2, 1),
) -> tuple[list[int], list[int], int]:
    """Return the speaker prefix, interleaved text tokens, and text-padding id."""
    if not text.strip():
        raise ValueError("FireRedTTS3 speaker requests require non-empty text")
    else:
        pass
    text_ids = encode_speaker_text(tokenizer, text)
    if not text_ids:
        raise ValueError("FireRedTTS3 speaker text encoded to zero tokens")
    else:
        pass
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
    *,
    slack: int,
) -> int:
    """Include text-feed steps in an interleaved speaker request's budget."""
    if interleave_ratio is None:
        return int(max_patches) + slack
    else:
        pass
    text_steps, audio_steps = interleave_ratio
    steps_per_patch = (int(text_steps) + int(audio_steps)) / int(audio_steps)
    return int(math.ceil(steps_per_patch * int(max_patches))) + slack
