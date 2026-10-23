# SPDX-License-Identifier: Apache-2.0
"""Shared FireRedTTS3 test assets and slow-boundary doubles."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.firered_tts3.dit import DiT
from sglang_omni.models.firered_tts3.flow_graph import (
    FlowGraphRunner,
    euler_flow_loop,
    flow_t_span,
)
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.redae_decoder import RedAEAudioDecoder
from sglang_omni.models.firered_tts3.vocoder import FireRedTTS3StreamingVocoder
from sglang_omni.models.firered_tts3.weights import drop_embedding_tables
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage


def make_speaker_tokenizer() -> PreTrainedTokenizerFast:
    """Local tokenizer with special tokens and merged Han tokens for CPU tests."""
    special_tokens = [
        "<unk>",
        "<|im_start|>",
        "<|im_end|>",
        "<|f531_speaker|>",
        "<|audio_start|>",
        "<|audio_end|>",
        "<|audio_holder|>",
        "<|text_holder|>",
        "<|text_pad|>",
    ]
    characters = list(
        dict.fromkeys(
            "".join(chr(codepoint) for codepoint in range(32, 127))
            + "\n你好世界中性活泼晚上好，欢迎收听今天的节目。"
        )
    )
    tokens = special_tokens + characters + ["你好", "中性", "生僻"]
    tokenizer = Tokenizer(
        models.WordLevel(
            {token: index for index, token in enumerate(tokens)}, unk_token="<unk>"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        Regex(r"<\|[^|]+\|>|你好|中性|生僻|[\s\S]"), behavior="isolated"
    )
    tokenizer.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        additional_special_tokens=special_tokens[1:],
        clean_up_tokenization_spaces=False,
    )


HISTORY_LENGTH = 8
PATCH_SIZE = 4
REDAE_DIM = 64
COND_CHANNELS = 32
N_TIMESTEPS = 10


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


@pytest.fixture
def tiny_dit(device: torch.device) -> DiT:
    return tiny_cpu_dit().to(device=device)


def tiny_runner(dit: DiT, device, **overrides) -> FlowGraphRunner:
    kwargs = {
        "n_timesteps": N_TIMESTEPS,
        "patch_size": PATCH_SIZE,
        "history_length": HISTORY_LENGTH,
        "capture_sizes": (1, 2, 4),
    }
    kwargs.update(overrides)
    return FlowGraphRunner(dit, flow_t_span(N_TIMESTEPS, device), **kwargs)


def make_inputs(
    device: torch.device, generator: torch.Generator, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hist = torch.randn(
        batch_size, HISTORY_LENGTH, REDAE_DIM, device=device, generator=generator
    )
    x0 = torch.randn(
        batch_size, PATCH_SIZE, REDAE_DIM, device=device, generator=generator
    )
    cond = torch.randn(
        batch_size,
        HISTORY_LENGTH + PATCH_SIZE,
        COND_CHANNELS,
        device=device,
        generator=generator,
    )
    return hist, x0, cond


def eager_patch(
    dit: torch.nn.Module,
    hist: torch.Tensor,
    x0: torch.Tensor,
    cond: torch.Tensor,
    cfg: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return euler_flow_loop(
        dit,
        torch.cat([hist, x0], dim=1),
        cond,
        flow_t_span(N_TIMESTEPS, device),
        cfg,
        N_TIMESTEPS,
        PATCH_SIZE,
    )


def tiny_cpu_dit() -> DiT:
    torch.manual_seed(0)
    return (
        DiT(
            in_channels=REDAE_DIM + COND_CHANNELS,
            out_channels=REDAE_DIM,
            mlp_ratio=2.0,
            depth=2,
            num_heads=4,
            hidden_size=64,
        )
        .to(dtype=torch.float32)
        .eval()
    )


@pytest.fixture
def disable_flow_optimization_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIRERED_TTS3_FFN_GEMM", "0")
    monkeypatch.setenv("FIRERED_TTS3_FLOW_ATOL", "0")
    monkeypatch.setenv("FIRERED_TTS3_FLOW_RTOL", "0")
    monkeypatch.setenv("FIRERED_TTS3_PRECOMPUTE_TIME", "0")
    monkeypatch.setenv("FIREREDTTS3_CONV_LAYOUT", "conv1d")


@pytest.fixture
def flow_dispatch_dit(tiny_dit: DiT) -> DiT:
    # note (Codex): a zero head isolates dispatch lifecycle from backend rounding.
    with torch.no_grad():
        tiny_dit.final_layer.linear.weight.zero_()
        tiny_dit.final_layer.linear.bias.zero_()
    return tiny_dit


@pytest.fixture
def decoder(request: pytest.FixtureRequest) -> RedAEAudioDecoder:
    uses_cuda = request.node.get_closest_marker("accelerator") is not None
    if uses_cuda and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    else:
        pass
    torch.manual_seed(1234)
    module = RedAEAudioDecoder(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    drop_embedding_tables(module)
    return module.to(device="cuda" if uses_cuda else "cpu").eval()


def make_payload(
    prompt: torch.Tensor,
    generated: torch.Tensor,
    request_id: str = "request",
    finish_reason: str | None = "length",
) -> StagePayload:
    state = FireRedTTS3State(
        prompt_latents=prompt,
        generated_latents=generated,
        left_context_size=prompt.shape[1],
        finish_reason=finish_reason,
        gen_text="voice plan",
        completion_tokens=generated.shape[1] // 4,
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={"text": "hello"}, params={"stream": True}),
        data=state.to_dict(),
    )


def audio_tensor(payload: StagePayload) -> torch.Tensor:
    assert payload.data["audio_waveform_dtype"] == "float32"
    return torch.frombuffer(
        bytearray(payload.data["audio_waveform"]), dtype=torch.float32
    )


def run_streams(
    vocoder: FireRedTTS3StreamingVocoder,
    prompts: dict[str, torch.Tensor],
    generated: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    for request_id, prompt in prompts.items():
        vocoder.handle_streaming_new_request(
            request_id, make_payload(prompt, generated[request_id], request_id)
        )
    chunk_id = 0
    cursor = 0
    while cursor < next(iter(generated.values())).shape[1]:
        messages = []
        for request_id, prompt in prompts.items():
            is_prompt = chunk_id == 0
            frames = (4, 8, 16)[min(max(chunk_id - 1, 0), 2)]
            chunk = (
                prompt
                if is_prompt
                else generated[request_id][:, cursor : cursor + frames]
            )
            metadata = {"modality": "audio_latents", "stream": True}
            if is_prompt:
                metadata.update(prompt_chunk=True, left_context_size=prompt.shape[1])
            else:
                pass
            messages.append(
                IncomingMessage(
                    request_id,
                    "stream_chunk",
                    StreamItem(chunk_id, chunk, "talker", metadata),
                )
            )
        if vocoder.can_batch_stream_chunks:
            vocoder.handle_stream_chunk_batch(messages)
        else:
            for message in messages:
                vocoder.handle_stream_chunk(message.request_id, message.data)
        if chunk_id > 0:
            cursor += frames
        else:
            assert vocoder.outbox.empty()
        chunk_id += 1
    states = list(vocoder.stream_states.values())
    for request_id in prompts:
        vocoder.handle_stream_done(request_id)
    waveforms: dict[str, list[torch.Tensor]] = {
        request_id: [] for request_id in prompts
    }
    completed: set[str] = set()
    while not vocoder.outbox.empty():
        message = vocoder.outbox.get_nowait()
        if message.type == "stream":
            assert message.request_id not in completed
            assert isinstance(message.data, dict)
            assert message.metadata == {"modality": "audio"}
            waveforms[message.request_id].append(
                torch.frombuffer(
                    bytearray(message.data["audio_waveform"]), dtype=torch.float32
                )
            )
        else:
            assert message.type == "result", message.data
            assert isinstance(message.data, StagePayload)
            assert message.data.data["finish_reason"] == "length"
            assert message.data.data["text"] == "voice plan"
            assert (
                message.data.data["usage"]["completion_tokens"]
                == generated[message.request_id].shape[1] // 4
            )
            completed.add(message.request_id)
    assert completed == set(prompts)
    assert vocoder.stream_states == {} and vocoder.stream_payloads == {}
    assert all(state.decoder_state is None and state.pending == [] for state in states)
    return {request_id: torch.cat(chunks) for request_id, chunks in waveforms.items()}


class ReplayModel(torch.nn.Module):
    """Small CPU model that records which patches replay consumed."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.history_length = 4
        self.history_patches = 2
        self.redae_dim = 64
        self.llm_hidden_size = 8
        self.spk_proj_dit = None
        self.graph_feedback_buffer = None
        self.encoded_patches: list[torch.Tensor] = []

    def get_input_embeddings(self) -> Mock:
        return Mock(
            side_effect=lambda ids: ids.float().unsqueeze(-1).expand(*ids.shape, 8)
        )

    def patch_encoder(self, patches: torch.Tensor) -> torch.Tensor:
        self.encoded_patches.append(patches)
        return (
            patches.reshape(1, -1, 4, 64)
            .mean(dim=(2, 3))
            .unsqueeze(-1)
            .expand(-1, -1, 8)
        )
