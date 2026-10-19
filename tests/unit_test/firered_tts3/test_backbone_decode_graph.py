# SPDX-License-Identifier: Apache-2.0
"""Backbone decode CUDA graph contract tests.

The engine decode graph bakes the model's forward branch at capture time,
so the branch may depend only on capture-visible state: the model-owned
feedback buffer (installed before capture) switches decode forwards to the
buffer read path, and on lm_head checkpoints (Instruct) it also pins the
real-logits branch so text-AR steps stay samplable under replay. These
tests pin that contract with a stubbed backbone (CPU-only).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput

from sglang_omni.models.firered_tts3.engine_builder import FireRedTTS3EngineBuilder
from sglang_omni.models.firered_tts3.sglang_model import FireRedTTS3SGLangModel

HIDDEN_SIZE = 8
BATCH_SIZE = 3


def build_model(*, use_lm_head: bool, with_buffer: bool) -> FireRedTTS3SGLangModel:
    """A FireRedTTS3SGLangModel shell with a recording stub backbone."""
    model = FireRedTTS3SGLangModel.__new__(FireRedTTS3SGLangModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(use_lm_head=use_lm_head)
    model.llm_hidden_size = HIDDEN_SIZE
    recorded: dict[str, torch.Tensor] = {}

    def backbone_model(input_ids, positions, forward_batch, input_embeds=None):
        recorded["input_embeds"] = input_embeds
        return input_embeds

    real_logits = torch.full((BATCH_SIZE, 151936), 7.0)

    def logits_processor(input_ids, hidden_states, lm_head, forward_batch):
        recorded["logits_processor_called"] = True
        return LogitsProcessorOutput(
            next_token_logits=real_logits, hidden_states=hidden_states
        )

    model.backbone = SimpleNamespace(
        model=backbone_model,
        logits_processor=logits_processor,
        lm_head=object(),
    )
    model.graph_feedback_buffer_value = (
        torch.arange(BATCH_SIZE * HIDDEN_SIZE, dtype=torch.bfloat16).reshape(
            BATCH_SIZE, HIDDEN_SIZE
        )
        if with_buffer
        else None
    )
    model.text_logits_requested = False
    return model, recorded


def forward_batch_stub(*, decode: bool) -> SimpleNamespace:
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: decode, is_extend=lambda: not decode
        ),
        extend_seq_lens=torch.full((BATCH_SIZE,), 5) if not decode else None,
        input_embeds=torch.full((BATCH_SIZE, HIDDEN_SIZE), -1.0),
    )


def run_forward(model, *, decode: bool) -> LogitsProcessorOutput:
    return model.forward(
        torch.zeros(BATCH_SIZE, dtype=torch.long),
        torch.arange(BATCH_SIZE),
        forward_batch_stub(decode=decode),
    )


def test_decode_reads_feedback_buffer_rows() -> None:
    model, recorded = build_model(use_lm_head=False, with_buffer=True)
    model.forward(
        torch.zeros(BATCH_SIZE, dtype=torch.long),
        torch.arange(BATCH_SIZE),
        forward_batch_stub(decode=True),
    )
    # The graph-baked read path: decode embeds come from the persistent
    # buffer, never from forward_batch.input_embeds.
    assert torch.equal(
        recorded["input_embeds"], model.graph_feedback_buffer_value[:BATCH_SIZE]
    )


def test_decode_without_buffer_falls_back_to_forward_batch_embeds() -> None:
    model, recorded = build_model(use_lm_head=False, with_buffer=False)
    stub = forward_batch_stub(decode=True)
    model.forward(
        torch.zeros(BATCH_SIZE, dtype=torch.long), torch.arange(BATCH_SIZE), stub
    )
    assert recorded["input_embeds"] is stub.input_embeds


def test_lm_head_decode_with_buffer_always_real_logits() -> None:
    # The captured branch contract: buffer installed (graph on) + decode mode
    # must take the lm_head branch even with text_logits_requested False, so
    # text-AR steps replayed from the graph return samplable logits.
    model, recorded = build_model(use_lm_head=True, with_buffer=True)
    output = run_forward(model, decode=True)
    assert recorded.get("logits_processor_called") is True
    assert output.next_token_logits.shape[1] == 151936


def test_lm_head_decode_eager_needs_the_runner_flag() -> None:
    # Without the buffer (eager decode) the per-step flag keeps its meaning:
    # audio-phase rows keep the dummy 1-wide logits.
    model, recorded = build_model(use_lm_head=True, with_buffer=False)
    output = run_forward(model, decode=True)
    assert "logits_processor_called" not in recorded
    assert output.next_token_logits.shape == (BATCH_SIZE, 1)
    model.text_logits_requested = True
    output = run_forward(model, decode=True)
    assert recorded.get("logits_processor_called") is True
    assert output.next_token_logits.shape[1] == 151936


def test_extend_branch_is_flag_driven_even_with_buffer() -> None:
    # Prefill is never captured: the buffer must not force real logits there.
    model, recorded = build_model(use_lm_head=True, with_buffer=True)
    output = run_forward(model, decode=False)
    assert "logits_processor_called" not in recorded
    assert output.next_token_logits.shape == (BATCH_SIZE, 1)


def test_generation_defaults_turn_the_decode_graph_on() -> None:
    builder = FireRedTTS3EngineBuilder()
    defaults = builder.generation_defaults(dtype="bfloat16")
    assert defaults["disable_cuda_graph"] is False
    # adjust_overrides runs on the merged dict (defaults + yaml overrides).
    overrides = dict(defaults)
    builder.adjust_overrides(overrides)
    # The graph is captured with FULL hidden states; the runner requests the
    # matching mode per step.
    assert overrides["enable_return_hidden_states"] is True
    eager_overrides = dict(defaults, disable_cuda_graph=True)
    builder.adjust_overrides(eager_overrides)
    assert "enable_return_hidden_states" not in eager_overrides
