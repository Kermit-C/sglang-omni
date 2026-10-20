# SPDX-License-Identifier: Apache-2.0
"""Omni ModelRunner hooks for the FireRedTTS3 fused AR/flow loop.

One engine decode step = one latent patch (4 frames = 160 ms of audio) in
the prefill form; the dual-stream form (speaker checkpoints, request-level
opt-in) consumes (nt + na) / na engine steps per patch:

* before_prefill assembles the prefill row (Base: [spk_proj_llm(spk_emb);
  embed(text_ids); patch_encoder(prompt_latents)]; Instruct/speaker: embed(chatml
  tokens), Instruct with patch_encoder outputs masked-scattered into the
  image_pad / video_pad slots) and seeds the per-request noise generator. A
  request's seed reproduces the reference's global-seed draw order at
  concurrency 1 (one draw per patch, in patch order); without a seed the
  draw stream derives from the request id so concurrent requests stay
  independent.
* Instruct text-phase requests (voice_design / semantic_edit) run the
  reference text-AR machine first: prefill tail and text steps sample REAL
  tokens through the backbone lm_head (the runner leaves next_token_ids
  alone); sampling <|eot|> (or exhausting the 200-token budget) does NOT
  terminate the request — the next step feeds embed(eot) without sampling
  (finalize), then one <|sosp|> feed, and the audio loop's first step feeds
  <|sosp|> AGAIN (the reference leaves input_embeds stale across the loop
  boundary — the doubled feed is its actual behavior, replicated).
* Dual-stream requests (speaker variant, dual_stream=True) run the
  reference's streaming machine: the prompt ends at <|audio_start|>; per
  (nt, na) group the machine feeds nt text tokens (the char-modelled stream,
  <|text_pad|> once exhausted) then na patch embeddings; the group's last
  text feed (and each intermediate patch feed) appends the cond and fires
  the flow, while the last patch feed only runs the stop check — the
  reference predicts stop AFTER the patch enters the backbone, so a stopping
  step HAS already emitted its patch (the prefill form checks before
  generating and a stopping step emits nothing).
* post_prefill / post_decode run the reference side path in fp32: stop_head
  sigmoid -> cond window (first step: the full prompt-latent span's hidden
  rows; later steps: the last row, sliding window of num_history_patches +
  1) -> dit_head -> DiT n_timesteps Euler (CFG doubles the batch with a
  zeroed cond twin; only the last patch frames update) -> patch_encoder
  feedback queued for the next step. With flow_batching on, each hook first
  collects the step's per-request flow rows (stop/cond/noise stay
  per-request; a stopping row is never collected) and then runs ONE batched
  DiT + PatchEncoder pass, writing back in collection order — the DiT loop
  is launch-bound, so a B-row replay costs barely more than one row.
* before_decode moves the queued feedback into forward_batch.input_embeds
  for patch-feed rows (or the model-owned persistent buffer when the decode
  CUDA graph is on); text feeds ride the engine's token rail (the runner
  writes the feed token into next_token_ids the step before).
* Retraction keeps the RNG state (FireRedFlowResume) plus the text machine
  on the request data; the re-prefill rebuilds the flow state from the full
  hidden row and replays the decoded patches through patch_encoder (for
  dual-stream, the interleaved text/patch feed order recorded in
  output_ids). The re-prefill tail is the interrupted step's input; both
  machines resume its post-work on that hidden. Prefill stop decisions
  defer to the next decode step so the scheduler releases KV normally.
  Repeated retractions while that stop is pending replay the excess CONTROL
  feeds as discardable token embeddings; no more machine post-work runs.
  Async decode with retraction remains unverified.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.firered_tts3.flow_graph import (
    FlowGraphRunner,
    euler_flow_loop,
    flow_t_span,
)
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.models.firered_tts3.prompt_utils import (
    AUDIO_SOS_ID,
    LATENT_IN_PAD_ID,
    LATENT_OUT_PAD_ID,
    TEXT_EOT_ID,
    TEXT_TOKEN_BUDGET,
)
from sglang_omni.models.firered_tts3.request_builders import (
    CONTROL_TOKEN_ID,
    FireRedFlowResume,
    FireRedTTS3SGLangRequestData,
)
from sglang_omni.sampling.seed import derive_sampling_seed
from sglang_omni.scheduling.types import SchedulerRequest

# Text-AR machine stages (the reference's generate() mapped onto engine
# steps): TEXT samples; FINALIZE feeds embed(eot-or-last-token) without
# sampling; SOSP feeds embed(sosp); AUDIO_ZERO is the audio loop's step 0
# (the doubled sosp feed whose hidden drives the first stop check + patch);
# AUDIO is the steady fused loop.
TEXT_MODE_TEXT = "text"
TEXT_MODE_FINALIZE = "finalize"
TEXT_MODE_SOSP = "sosp"
TEXT_MODE_AUDIO_ZERO = "audio_zero"
TEXT_MODE_AUDIO = "audio"

# Dual-stream machine stages (speaker variant): per (nt, na) group, nt text
# feeds then na patch feeds. The group's LAST text feed (and every
# intermediate patch feed) drives a flow step; the LAST patch feed only runs
# the stop check (the reference predicts stop AFTER the patch enters the
# backbone, so a stopping step has already emitted its patch).
DS_STAGE_TEXT = "text"
DS_STAGE_PATCH = "patch"


class FlowState:
    """Per-request fused-loop state (fp32; rebuilt from scratch on re-prefill)."""

    def __init__(
        self,
        *,
        generator: torch.Generator,
        prompt_patch_count: int,
        spk_cond_dit: torch.Tensor | None,
        backbone_cond: torch.Tensor,
        latents_gen: torch.Tensor,
        patch_count: int,
    ) -> None:
        self.generator = generator
        self.prompt_patch_count = prompt_patch_count
        self.spk_cond_dit = spk_cond_dit
        self.backbone_cond = backbone_cond
        self.latents_gen = latents_gen
        self.patch_count = patch_count


# One collected flow row: (request data, history latents, noise, DiT cond).
# The DiT Euler loop is kernel-launch bound, so the per-engine-step rows of
# concurrent requests are cat'ed into one batched replay (bucket zero-pad as
# needed) instead of replaying once per request.
FlowRow = tuple[FireRedTTS3SGLangRequestData, torch.Tensor, torch.Tensor, torch.Tensor]


class FireRedTTS3ModelRunner(ModelRunner):
    """Use the shared SGLang forward path and own only the flow side path."""

    # note: class-level defaults so tests building the runner via __new__
    # resolve the flow-graph probe to disabled and the flow path to serial.
    flow_graphs: FlowGraphRunner | None = None
    flow_batching: bool = False
    control_token_fastpath: bool = True
    skip_min_patch_stop: bool = True
    prefill_launch_at: float | None = None

    def __init__(
        self,
        tp_worker,
        output_processor,
        *,
        flow_graphs: FlowGraphRunner | None = None,
        flow_batching: bool = False,
        control_token_fastpath: bool = True,
        skip_min_patch_stop: bool = True,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.active_requests: dict[str, FireRedTTS3SGLangRequestData] = {}
        self.t_span_cache: dict[int, torch.Tensor] = {}
        # Host timestamp at the end of before_prefill; post_prefill turns it
        # into the engine_prefill segment (the prefill forward's wall time).
        self.prefill_launch_at: float | None = None
        if flow_graphs is not None:
            self.flow_graphs = flow_graphs
        self.flow_batching = flow_batching
        self.control_token_fastpath = control_token_fastpath
        self.skip_min_patch_stop = skip_min_patch_stop

    def t_span(self, n_timesteps: int, device: torch.device) -> torch.Tensor:
        span = self.t_span_cache.get(n_timesteps)
        if span is None or span.device != device:
            span = flow_t_span(n_timesteps, device)
            self.t_span_cache[n_timesteps] = span
        return span

    def before_prefill(self, forward_batch, schedule_batch, requests: list) -> None:
        del schedule_batch
        if not requests:
            return
        self.release_retracted_flow_states()
        model = self.model
        device = forward_batch.input_ids.device
        embed_dtype = next(model.parameters()).dtype
        model.text_logits_requested = any(
            request.data.text_ar_mode == TEXT_MODE_TEXT for request in requests
        )
        rows = []
        materialized = []
        try:
            for request in requests:
                data = request.data
                if data.flow_state is not None and not isinstance(
                    data.flow_state, FireRedFlowResume
                ):
                    self.suspend_request_data(data)
                self.active_requests.pop(request.request_id, None)
                resume = data.flow_state
                state = data.state
                if state.variant == "speaker":
                    # The speaker token carries the voice; speaker requests
                    # must not carry reference-conditioning tensors.
                    if (
                        state.prompt_latents is not None
                        or state.speaker_embedding is not None
                    ):
                        raise RuntimeError(
                            "FireRedTTS3 speaker-variant requests carry no "
                            "reference-conditioning tensors (prompt_latents / "
                            "speaker_embedding)"
                        )
                elif state.variant != "instruct" and (
                    state.prompt_latents is None or state.speaker_embedding is None
                ):
                    raise RuntimeError(
                        "FireRedTTS3 request is missing its reference-conditioning "
                        "tensors (prompt_latents / speaker_embedding)"
                    )
                prompt_latents = (
                    state.prompt_latents.to(device=device, dtype=torch.float32)
                    if state.prompt_latents is not None
                    else None
                )
                spk_emb = (
                    state.speaker_embedding.to(device=device, dtype=torch.float32)
                    if state.speaker_embedding is not None
                    else None
                )
                generator = torch.Generator(device=device)
                if (
                    isinstance(resume, FireRedFlowResume)
                    and resume.rng_state is not None
                ):
                    generator.set_state(resume.rng_state)
                elif state.seed is not None:
                    generator.manual_seed(int(state.seed))
                else:
                    generator.manual_seed(
                        derive_sampling_seed("fireredtts3-flow", request.request_id)
                    )
                if prompt_latents is not None:
                    latents_gen = F.pad(prompt_latents, (0, 0, model.history_length, 0))
                else:
                    latents_gen = torch.zeros(
                        1, model.history_length, model.redae_dim, device=device
                    )
                if data.latent_patches:
                    latents_gen = torch.cat([latents_gen, *data.latent_patches], dim=1)
                data.flow_state = FlowState(
                    generator=generator,
                    prompt_patch_count=data.prompt_patch_count,
                    spk_cond_dit=(
                        model.spk_proj_dit(spk_emb)
                        if model.spk_proj_dit is not None
                        else None
                    ),
                    backbone_cond=torch.zeros(
                        1,
                        model.history_patches,
                        model.llm_hidden_size,
                        device=device,
                    ),
                    latents_gen=latents_gen,
                    patch_count=len(data.latent_patches),
                )
                data.probe = ProfilingProbe.create()
                self.active_requests[request.request_id] = data
                materialized.append((request.request_id, data))

                with torch.inference_mode(), data.probe.segment("prefill_embeds"):
                    if state.variant == "base":
                        prompt_embeds = self.build_base_prefill_embeds(
                            model, state, prompt_latents, spk_emb, device
                        )
                    else:
                        # Instruct / speaker: real chatml token rows (speaker
                        # text ids were scattered into the holder slots at
                        # preprocessing; Instruct audio latents scatter here).
                        prompt_embeds = self.build_chatml_prefill_embeds(
                            model, state, device
                        )
                    if prompt_embeds.shape[1] != data.prefill_len:
                        raise RuntimeError(
                            f"FireRedTTS3 prefill row is {prompt_embeds.shape[1]} "
                            f"positions, expected {data.prefill_len}"
                        )
                    request_rows = [prompt_embeds[0].to(embed_dtype)]
                    replay_rows = self.build_replay_rows(
                        model, data, device, embed_dtype
                    )
                    if replay_rows is not None:
                        request_rows.append(replay_rows)
                prefix_len = len(data.req.prefix_indices)
                req_len = int(data.req.extend_range.length)
                generated = req_len - data.prefill_len
                assert prefix_len == 0, "FireRedTTS3 radix prefix reuse is disabled"
                if data.dual_stream:
                    # output_ids record the interleaved feeds: text tokens as
                    # themselves, patch feeds as the control token, including
                    # the pending feed. Every flow-firing step returns one
                    # control token and appends one patch. While a deferred
                    # stop is pending, each re-prefill returns another control
                    # token without a patch; the stopped machine emits nothing.
                    patch_feeds = sum(
                        1
                        for token in data.req.output_ids
                        if int(token) == CONTROL_TOKEN_ID
                    )
                    excess = patch_feeds - len(data.latent_patches)
                    assert (
                        generated == len(data.req.output_ids)
                        and excess >= 0
                        and (excess == 0 or data.stop_pending)
                    ), "FireRedTTS3 dual-stream re-prefill history is inconsistent"
                else:
                    excess = (
                        len(data.req.output_ids)
                        - self.text_output_tokens(data)
                        - len(data.latent_patches)
                    )
                    assert generated == len(data.req.output_ids) and (
                        excess >= 0 and (excess == 0 or data.stop_pending)
                    ), "FireRedTTS3 re-prefill history must match generated tokens"
                rows.append(torch.cat(request_rows, dim=0))
            forward_batch.input_embeds = torch.cat(rows, dim=0)
            if any(data.probe.enabled for _rid, data in materialized):
                self.prefill_launch_at = time.perf_counter()
        except BaseException:
            for request_id, data in materialized:
                self.active_requests.pop(request_id, None)
                self.clear_request_data(data)
            raise

    def build_base_prefill_embeds(
        self, model, state, prompt_latents, spk_emb, device
    ) -> torch.Tensor:
        """Base ICL row: [spk_proj_llm(spk_emb); embed(text); pe(prompt)]."""
        text_ids = torch.tensor(
            state.input_ids, dtype=torch.long, device=device
        ).unsqueeze(0)
        return torch.cat(
            [
                model.spk_proj_llm(spk_emb).unsqueeze(1),
                model.get_input_embeddings()(text_ids).to(torch.float32),
                model.patch_encoder(prompt_latents),
            ],
            dim=1,
        )

    def build_chatml_prefill_embeds(self, model, state, device) -> torch.Tensor:
        """Instruct / speaker row: embed(chatml tokens) with patch_encoder
        outputs masked-scattered into the image_pad (edit source, first —
        reference order) and video_pad (ICL prompt) slots. Speaker prompts
        carry no latent slots, so the row is the plain token embedding."""
        text_ids = torch.tensor(
            state.input_ids, dtype=torch.long, device=device
        ).unsqueeze(0)
        embeds = model.get_input_embeddings()(text_ids)  # [1, T, H] bf16
        for latents, pad_id, what in (
            (state.audio_in_latents, LATENT_IN_PAD_ID, "audio input"),
            (state.prompt_latents, LATENT_OUT_PAD_ID, "ICL audio"),
        ):
            if latents is None:
                continue
            patch_embeds = model.patch_encoder(
                latents.to(device=device, dtype=torch.float32)
            )
            mask = text_ids == pad_id
            slots = int(mask.sum())
            if slots != int(patch_embeds.shape[1]):
                raise RuntimeError(
                    f"FireRedTTS3 {what} slot mismatch: the prompt carries "
                    f"{slots} placeholder tokens but the audio encodes to "
                    f"{int(patch_embeds.shape[1])} patches"
                )
            embeds = embeds.masked_scatter(
                mask.unsqueeze(-1),
                patch_embeds.reshape(-1).to(embeds.dtype),
            )
        return embeds.to(torch.float32)

    def build_replay_rows(
        self, model, data: FireRedTTS3SGLangRequestData, device, embed_dtype
    ) -> torch.Tensor | None:
        """Re-prefill replay rows for the already-generated region: text /
        sosp tokens embed through the token table, audio patches through
        patch_encoder (one row per engine step). Dual-stream requests replay
        the interleaved feed order recorded in output_ids (a CONTROL row
        consumes the next generated patch); when the machine paused at a
        patch feed, that patch's embedding re-queues as the next feedback.
        Sync CONTROL entries match latent_patches 1:1 until a deferred stop;
        later CONTROL feeds replay as discardable token embeddings. Re-queuing
        is only a fallback for async output_ids lagging one token behind."""
        if data.dual_stream:
            return self.build_dual_stream_replay_rows(model, data, device, embed_dtype)
        text_tokens = self.text_output_tokens(data)
        rows = []
        if text_tokens:
            ids = torch.tensor(
                data.req.output_ids[:text_tokens], dtype=torch.long, device=device
            )
            rows.append(model.get_input_embeddings()(ids).to(embed_dtype))
        if data.latent_patches:
            rows.append(
                model.patch_encoder(torch.cat(data.latent_patches, dim=1))[0].to(
                    embed_dtype
                )
            )
        excess = len(data.req.output_ids) - text_tokens - len(data.latent_patches)
        if data.stop_pending and excess > 0:
            # A deferred stop's extra feeds have no patches; their hidden
            # rows are discarded by post_prefill and the finishing decode.
            ids = torch.full(
                (excess,), CONTROL_TOKEN_ID, dtype=torch.long, device=device
            )
            rows.append(model.get_input_embeddings()(ids).to(embed_dtype))
        if not rows:
            return None
        return torch.cat(rows, dim=0)

    def build_dual_stream_replay_rows(
        self, model, data: FireRedTTS3SGLangRequestData, device, embed_dtype
    ) -> torch.Tensor | None:
        output_ids = [int(token) for token in data.req.output_ids]
        if not output_ids:
            return None
        embed_tokens = model.get_input_embeddings()
        rows: list[torch.Tensor] = []
        text_buffer: list[int] = []
        patch_cursor = 0

        def flush_text() -> None:
            if text_buffer:
                ids = torch.tensor(text_buffer, dtype=torch.long, device=device)
                rows.append(embed_tokens(ids).to(embed_dtype))
                text_buffer.clear()

        for token in output_ids:
            if token == CONTROL_TOKEN_ID:
                flush_text()
                if data.stop_pending and patch_cursor == len(data.latent_patches):
                    # Stopped requests may accumulate CONTROL feeds across
                    # retractions; these rows never drive flow or feedback.
                    ids = torch.tensor(
                        [CONTROL_TOKEN_ID], dtype=torch.long, device=device
                    )
                    rows.append(embed_tokens(ids).to(embed_dtype))
                else:
                    assert patch_cursor < len(data.latent_patches), (
                        "FireRedTTS3 dual-stream replay found more patch feeds "
                        "than generated patches"
                    )
                    rows.append(
                        model.patch_encoder(data.latent_patches[patch_cursor])[0].to(
                            embed_dtype
                        )
                    )
                    patch_cursor += 1
            else:
                text_buffer.append(token)
        flush_text()
        # A patch whose flow already fired but whose feed was interrupted by
        # the retraction: its embedding is the next decode step's input.
        pending = len(data.latent_patches) - patch_cursor
        assert pending in (0, 1), (
            "FireRedTTS3 dual-stream replay must leave at most one fed-back "
            f"patch pending, got {pending}"
        )
        if pending:
            assert data.ds_stage == DS_STAGE_PATCH, (
                "FireRedTTS3 dual-stream replay expects the machine at a patch "
                f"feed when a patch is pending, got {data.ds_stage!r}"
            )
            data.pending_feedback_queue.append(
                model.patch_encoder(data.latent_patches[-1])[0, 0].to(dtype=embed_dtype)
            )
        return torch.cat(rows, dim=0)

    @staticmethod
    def text_output_tokens(data: FireRedTTS3SGLangRequestData) -> int:
        """Generated output_ids consumed by the text machine: the recorded
        text tokens (+ the EOT feed) and the finalize/sosp transition rows."""
        if not data.text_token_ids and not data.eot_sampled:
            # Never ran a text phase (Base, Instruct tts/acoustic_edit, or a
            # text request retracted before its first sample).
            return 0
        n_text = len(data.text_token_ids) + int(data.eot_sampled)
        if data.text_ar_mode in (TEXT_MODE_AUDIO_ZERO, TEXT_MODE_AUDIO):
            n_text += 2
        elif data.text_ar_mode == TEXT_MODE_SOSP:
            n_text += 1
        return n_text

    def before_decode(
        self, forward_batch, schedule_batch, requests: list, *, is_lookahead=False
    ) -> None:
        del schedule_batch, is_lookahead
        if not requests:
            return
        model = self.model
        model.text_logits_requested = any(
            request.data.text_ar_mode == TEXT_MODE_TEXT for request in requests
        )
        rows = []
        for row, request in enumerate(requests):
            data = request.data
            # Patch-embedding feeds: the fused loop's steady state (AUDIO)
            # pops the queued PatchEncoder feedback; the dual-stream machine
            # pops it on its patch feeds.
            feed_is_patch = (
                not data.dual_stream and data.text_ar_mode == TEXT_MODE_AUDIO
            ) or (data.dual_stream and data.ds_stage == DS_STAGE_PATCH)
            if feed_is_patch:
                queue = data.pending_feedback_queue
                if queue:
                    rows.append(queue.popleft())
                elif data.stop_pending:
                    # A deferred stop queued no feedback. The next post hook
                    # discards this row (sync) or leaves it to resolve
                    # (async), so any in-shape embedding satisfies the
                    # engine's contract.
                    rows.append(
                        torch.zeros(
                            model.llm_hidden_size,
                            device=forward_batch.input_ids.device,
                            dtype=next(model.parameters()).dtype,
                        )
                    )
                else:
                    raise RuntimeError(
                        "FireRedTTS3 decode is missing its latent feedback"
                    )
            else:
                # Text / finalize / sosp feeds: embed the engine's incoming
                # token (the sampled text token, EOT, the sosp overwrite, or
                # the dual-stream text feed placed by the previous step).
                token_id = int(forward_batch.input_ids[row])
                rows.append(
                    model.get_input_embeddings()(
                        torch.tensor(
                            [token_id],
                            dtype=torch.long,
                            device=forward_batch.input_ids.device,
                        )
                    )[0]
                )
        buffer = model.graph_feedback_buffer
        if buffer is not None:
            # stack writes in forward-batch row order on the same stream as the
            # graph or eager forward; forward() reads this buffer directly.
            torch.stack(rows, out=buffer[: len(rows)])
        else:
            forward_batch.input_embeds = torch.stack(rows).to(
                device=forward_batch.input_ids.device,
                dtype=next(model.parameters()).dtype,
            )

    def requested_capture_hidden_mode_prefill(self, schedule_batch, requests: list):
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        return CaptureHiddenMode.FULL

    def requested_capture_hidden_mode_decode(self, schedule_batch, requests: list):
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        # Decode runs one position per request, so FULL and LAST return the
        # same rows; the decode CUDA graph is captured with FULL
        # (enable_return_hidden_states) and load_batch rejects a batch asking
        # for more than the captured mode, so request FULL when the graph is on.
        if self.model.graph_feedback_buffer is not None:
            return CaptureHiddenMode.FULL
        return CaptureHiddenMode.LAST

    def sample_before_post_prefill(
        self, forward_batch, schedule_batch, requests: list
    ) -> bool:
        del forward_batch, schedule_batch
        # Text-phase rows must be really sampled (lm_head logits) before the
        # post hook runs, so the machine can read the sampled token.
        return any(request.data.text_ar_mode == TEXT_MODE_TEXT for request in requests)

    def sample_before_post_decode(
        self, forward_batch, schedule_batch, requests: list
    ) -> bool:
        del forward_batch, schedule_batch
        return any(request.data.text_ar_mode == TEXT_MODE_TEXT for request in requests)

    def post_prefill(
        self, result, forward_batch, schedule_batch, requests: list
    ) -> None:
        del forward_batch
        if bool(getattr(schedule_batch, "is_prefill_only", False)):
            return
        if not requests:
            return
        if self.prefill_launch_at is not None:
            # The wall between the before_prefill handoff and this hook is the
            # engine's prefill forward (plus in-engine bookkeeping); one
            # forward serves the whole extend batch, so every request's probe
            # accumulates the same span.
            engine_prefill = time.perf_counter() - self.prefill_launch_at
            self.prefill_launch_at = None
            for request in requests:
                probe = request.data.probe
                if probe is not None and probe.enabled:
                    probe.segments["engine_prefill"] = engine_prefill
        hidden = self.hidden_states(result)
        if hidden.ndim == 3:
            hidden = hidden.reshape(-1, hidden.shape[-1])
        elif hidden.ndim != 2:
            raise RuntimeError(
                f"FireRedTTS3 expected rank-2/3 prefill hidden, got {hidden.ndim}"
            )
        offset = 0
        next_token_ids = []
        flow_rows: list[FlowRow] | None = [] if self.flow_batching else None
        for row, request in enumerate(requests):
            data = request.data
            length = int(data.req.extend_range.length)
            request_hidden = hidden[offset : offset + length]
            if request_hidden.size(0) != length:
                raise RuntimeError("FireRedTTS3 prefill hidden rows are incomplete")
            offset += length
            hidden_f32 = request_hidden.to(torch.float32)
            if data.stop_pending:
                # A previously deferred stop leaves no machine post-work.
                # Each re-prefill returns an extra CONTROL without a patch;
                # the next decode resolves the finish and discards its row.
                next_token_ids.append(CONTROL_TOKEN_ID)
                continue
            flow = data.flow_state
            if data.text_ar_mode == TEXT_MODE_TEXT:
                # The prefill tail is the reference's first text sampling
                # (its loop iter 0); no flow side path yet.
                next_token_ids.append(
                    self.record_text_sample(data, result.next_token_ids[row])
                )
                continue
            if data.dual_stream:
                if not data.req.output_ids:
                    # First prefill: the machine starts on the next decode
                    # step's text feed; the prompt tail fires nothing
                    # (generate_streaming_noicl leaves the prefix unused).
                    next_token_ids.append(self.next_text_feed(data))
                    continue
                # The re-prefill tail is the interrupted step's input. Rebuild
                # the earlier cond rows before resuming its post-work; a flow
                # step appends the tail itself.
                if flow.patch_count > 0:
                    flow.backbone_cond = torch.cat(
                        [
                            flow.backbone_cond,
                            self.dual_stream_cond_rows(data, hidden_f32),
                        ],
                        dim=1,
                    )
                next_token_ids.append(
                    self.dual_stream_step(
                        data, hidden_f32[-1:], flow_rows, defer_stop=True
                    )
                )
                continue
            mode = data.text_ar_mode
            if mode == TEXT_MODE_FINALIZE:
                # The interrupted finalize fed embed(EOT/last-token) without
                # sampling; resume the same transition as run_post_decode.
                data.text_ar_mode = TEXT_MODE_SOSP
                next_token_ids.append(AUDIO_SOS_ID)
                continue
            if mode == TEXT_MODE_SOSP:
                data.text_ar_mode = TEXT_MODE_AUDIO_ZERO
                next_token_ids.append(AUDIO_SOS_ID)
                continue
            if mode == TEXT_MODE_AUDIO_ZERO:
                data.text_ar_mode = TEXT_MODE_AUDIO
            if flow.patch_count > 0:
                # Re-prefill after retraction: rebuild the cond history from
                # the full hidden row — the prompt-latent span, then one row
                # per completed patch-driving step — before this step's own
                # append. Text-machine requests interleave the text/sosp
                # region before the patch rows (audio_zero drives patch 0).
                span_start = data.prefill_len - flow.prompt_patch_count
                audio_start = data.prefill_len + self.audio_hidden_offset(data)
                rebuilt = [
                    flow.backbone_cond,
                    hidden_f32[span_start : data.prefill_len].unsqueeze(0),
                    hidden_f32[audio_start:-1].unsqueeze(0),
                ]
                flow.backbone_cond = torch.cat(rebuilt, dim=1)
            # Prefill skips already-finished requests before KV cleanup;
            # resolve a stop on the next decode step instead.
            self.flow_step(data, hidden_f32, flow_rows, defer_stop=True)
            next_token_ids.append(CONTROL_TOKEN_ID)
        if flow_rows:
            self.run_flow_rows(flow_rows)
        if offset != hidden.size(0):
            raise RuntimeError("FireRedTTS3 prefill hidden rows do not match requests")
        result.next_token_ids, host_token_ids = self.make_next_token_ids(
            next_token_ids, hidden.device, requests
        )
        if host_token_ids is not None:
            self.stage_token_ids(result, host_token_ids)

    def is_sync_speaker_audio(self, data: FireRedTTS3SGLangRequestData) -> bool:
        return (
            not self._async_enabled
            and data.state.variant == "speaker"
            and not data.dual_stream
            and data.text_ar_mode == TEXT_MODE_AUDIO
        )

    def make_next_token_ids(
        self,
        next_token_ids: list[int],
        device: torch.device,
        requests: list[SchedulerRequest],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Separate known reporting IDs from the device's control-token input."""
        if (
            self.control_token_fastpath
            and all(self.is_sync_speaker_audio(request.data) for request in requests)
            and all(token == CONTROL_TOKEN_ID for token in next_token_ids)
        ):
            return (
                torch.full(
                    (len(next_token_ids),),
                    CONTROL_TOKEN_ID,
                    dtype=torch.long,
                    device=device,
                ),
                torch.tensor(next_token_ids, dtype=torch.long, device="cpu"),
            )
        else:
            return torch.tensor(next_token_ids, dtype=torch.long, device=device), None

    def audio_hidden_offset(self, data: FireRedTTS3SGLangRequestData) -> int:
        """Generated-region rows before the patch-driving hiddens. Reached
        only from the re-prefill cond rebuild (patch_count > 0, so the
        machine is in steady audio): the text/EOT rows and the FIRST sosp
        feed precede it; the doubled second sosp feed IS the
        patch-0-driving audio_zero row and stays inside the slice."""
        if data.text_ar_mode == TEXT_MODE_AUDIO and (
            data.text_token_ids or data.eot_sampled
        ):
            return len(data.text_token_ids) + int(data.eot_sampled) + 1
        return 0

    def record_text_sample(
        self, data: FireRedTTS3SGLangRequestData, sampled: torch.Tensor
    ) -> int:
        """One text-phase sampling event (reference text loop iteration).

        The sampled token rides through to the next feed unchanged. On EOT
        (or the 200-token budget) the machine moves to FINALIZE: that token's
        own feed becomes the no-sampling finalize step, matching the
        reference's post-loop embed(EOT) feed. EOT itself is never recorded
        (the reference breaks before appending it to text_gen_ids).
        """
        token = int(sampled)
        data.text_fed += 1
        if token == TEXT_EOT_ID:
            data.eot_sampled = True
            data.text_ar_mode = TEXT_MODE_FINALIZE
            return token
        data.text_token_ids.append(token)
        if data.text_fed >= TEXT_TOKEN_BUDGET:
            data.text_ar_mode = TEXT_MODE_FINALIZE
        return token

    def post_decode(
        self, result, forward_batch, schedule_batch, requests: list
    ) -> None:
        del forward_batch, schedule_batch
        self.run_post_decode(result, requests)

    def post_decode_launch(self, result, forward_batch, requests: list) -> None:
        """Async-decode launch half: the flow side path runs in full at launch
        (same work as the sync post_decode). Stop decisions DEFER to the
        resolve via defer_stop: the scheduler's overrun drop snapshots
        finished-state before the resolve, so a finish marked at launch would
        be mistaken for a prior-step overrun and its finish would never be
        processed (the request would leak out of the running batch unemitted).
        """
        del forward_batch
        self.run_post_decode(result, requests, defer_stop=True)
        return None

    def post_decode_resolve(
        self, launch_buf, result, forward_batch, schedule_batch, requests: list
    ) -> None:
        """Apply the launch-time stop decisions: the scheduler processes the
        finish (stream emission, KV release) after this hook within the same
        resolve. Retracted rows are dropped by the caller; never mark them."""
        del launch_buf, result, forward_batch, schedule_batch
        for request in requests:
            data = request.data
            if not data.stop_pending:
                continue
            data.stop_pending = False
            if not data.req.is_retracted and not data.req.finished():
                data.req.finished_reason = FINISH_MATCHED_TOKEN(CONTROL_TOKEN_ID)

    def run_post_decode(
        self, result, requests: list, *, defer_stop: bool = False
    ) -> None:
        """One decode step's flow side path: stop check -> cond window ->
        DiT Euler -> PatchEncoder feedback, then the control-token rail.
        Shared by the sync post_decode and the async post_decode_launch."""
        if not requests:
            return
        hidden = self.hidden_states(result)
        if hidden.ndim == 3:
            hidden = hidden[:, -1]
        elif hidden.ndim != 2:
            raise RuntimeError(
                f"FireRedTTS3 expected rank-2/3 decode hidden, got {hidden.ndim}"
            )
        next_token_ids = []
        flow_rows: list[FlowRow] | None = [] if self.flow_batching else None
        for row, request in enumerate(requests):
            data = request.data
            if data.stop_pending:
                # A stop deferred by prefill or an earlier async launch emits
                # no more patches. Async resolves in post_decode_resolve;
                # sync finishes here so decode owns the KV cleanup.
                next_token_ids.append(CONTROL_TOKEN_ID)
                if not defer_stop:
                    data.stop_pending = False
                    if not data.req.is_retracted and not data.req.finished():
                        data.req.finished_reason = FINISH_MATCHED_TOKEN(
                            CONTROL_TOKEN_ID
                        )
                continue
            if data.dual_stream:
                next_token_ids.append(
                    self.dual_stream_step(
                        data,
                        hidden[row : row + 1].to(torch.float32),
                        flow_rows,
                        defer_stop=defer_stop,
                    )
                )
                continue
            mode = data.text_ar_mode
            if mode == TEXT_MODE_TEXT:
                next_token_ids.append(
                    self.record_text_sample(data, result.next_token_ids[row])
                )
            elif mode == TEXT_MODE_FINALIZE:
                # This step fed embed(EOT/last-token); nothing was sampled
                # (the reference runs the finalize feed without sampling).
                data.text_ar_mode = TEXT_MODE_SOSP
                next_token_ids.append(AUDIO_SOS_ID)
            elif mode == TEXT_MODE_SOSP:
                data.text_ar_mode = TEXT_MODE_AUDIO_ZERO
                next_token_ids.append(AUDIO_SOS_ID)
            else:
                if mode == TEXT_MODE_AUDIO_ZERO:
                    data.text_ar_mode = TEXT_MODE_AUDIO
                self.flow_step(
                    data,
                    hidden[row : row + 1].to(torch.float32),
                    flow_rows,
                    defer_stop=defer_stop,
                )
                next_token_ids.append(CONTROL_TOKEN_ID)
        if flow_rows:
            self.run_flow_rows(flow_rows)
        result.next_token_ids, host_token_ids = self.make_next_token_ids(
            next_token_ids, hidden.device, requests
        )
        if host_token_ids is not None:
            self.stage_token_ids(result, host_token_ids)

    def dual_stream_cond_rows(
        self, data: FireRedTTS3SGLangRequestData, hidden_f32: torch.Tensor
    ) -> torch.Tensor:
        """Cond-history rows from a re-prefill's full hidden row: per group,
        the text-last feed (group offset nt - 1) and the intermediate patch
        feeds (offsets nt .. nt + na - 2) — one row per fired flow, in
        temporal order."""
        nt, na = data.interleave_ratio
        stride = nt + na
        rows: list[int] = []
        remaining = data.flow_state.patch_count
        group = 0
        while remaining > 0:
            base = data.prefill_len + group * stride
            rows.append(base + nt - 1)
            remaining -= 1
            for k in range(na - 1):
                if remaining == 0:
                    break
                rows.append(base + nt + k)
                remaining -= 1
            group += 1
        return hidden_f32[rows].unsqueeze(0)

    def next_text_feed(self, data: FireRedTTS3SGLangRequestData) -> int:
        """The next text-stream feed token: the char-modelled text id, or
        <|text_pad|> once the text is exhausted (reference behaviour)."""
        ids = data.state.text_stream_ids
        pos = data.text_pos
        data.text_pos += 1
        return int(ids[pos]) if pos < len(ids) else int(data.state.text_pad_id)

    def dual_stream_step(
        self,
        data: FireRedTTS3SGLangRequestData,
        hidden_f32: torch.Tensor,
        flow_rows: list[FlowRow] | None = None,
        *,
        defer_stop: bool = False,
    ) -> int:
        """One engine step of the dual-stream machine (reference
        generate_streaming_noicl); returns the next feed's token id.

        (ds_stage, ds_index) locate the CURRENT step (the feed that produced
        this hidden). Per (nt, na) group: the first nt - 1 text feeds are
        inert; the last text feed's hidden appends the cond and drives the
        flow; the first na - 1 patch feeds do the same; the last patch feed
        only runs the stop check (its patch was emitted one step earlier).
        """
        nt, na = data.interleave_ratio
        if data.ds_stage == DS_STAGE_TEXT:
            if data.ds_index >= nt - 1:
                # Last text feed of the group: cond append + flow.
                self.run_flow(data, hidden_f32, flow_rows)
                data.ds_stage = DS_STAGE_PATCH
                data.ds_index = 0
                return CONTROL_TOKEN_ID
            data.ds_index += 1
            return self.next_text_feed(data)
        if data.ds_index >= na - 1:
            # Last patch feed of the group: stop check only.
            data.ds_stage = DS_STAGE_TEXT
            data.ds_index = 0
            if self.stop_predicted(data, hidden_f32):
                self.mark_stop(data, defer=defer_stop)
                return CONTROL_TOKEN_ID
            return self.next_text_feed(data)
        # Intermediate patch feed: its hidden drives the next flow.
        self.run_flow(data, hidden_f32, flow_rows)
        data.ds_index += 1
        return CONTROL_TOKEN_ID

    def flow_step(
        self,
        data: FireRedTTS3SGLangRequestData,
        hidden_f32: torch.Tensor,
        flow_rows: list[FlowRow] | None = None,
        *,
        defer_stop: bool = False,
    ) -> None:
        """Prefill-form side-path step: the stop check runs BEFORE the flow
        (a stopping step emits no patch and is never collected)."""
        if self.stop_predicted(data, hidden_f32):
            self.mark_stop(data, defer=defer_stop)
            return
        self.run_flow(data, hidden_f32, flow_rows)

    @staticmethod
    def mark_stop(data: FireRedTTS3SGLangRequestData, *, defer: bool) -> None:
        """Record a stop decision. Prefill and async launches defer the finish;
        sync decode marks it directly (async uses post_decode_resolve)."""
        if defer:
            data.stop_pending = True
        else:
            data.req.finished_reason = FINISH_MATCHED_TOKEN(CONTROL_TOKEN_ID)

    def stop_predicted(
        self, data: FireRedTTS3SGLangRequestData, hidden_f32: torch.Tensor
    ) -> bool:
        """stop_head sigmoid on the span's last row (fp32 red line), gated by
        the min-patch floor. Dual-stream callers run this AFTER the patch
        feed (the patch was already emitted); prefill-form callers run it
        before the flow."""
        model = self.model
        flow = data.flow_state
        state = data.state
        if (
            self.skip_min_patch_stop
            and self.is_sync_speaker_audio(data)
            and flow.patch_count < state.min_patches
        ):
            return False
        with torch.inference_mode(), data.probe.segment("stop_head"):
            stop_logits = model.stop_head(hidden_f32[-1:].unsqueeze(0))
            stop_score = float(torch.sigmoid(stop_logits).reshape(-1)[-1].item())
        return (
            stop_score >= state.stop_threshold and flow.patch_count >= state.min_patches
        )

    def run_flow(
        self,
        data: FireRedTTS3SGLangRequestData,
        hidden_f32: torch.Tensor,
        flow_rows: list[FlowRow] | None = None,
    ) -> None:
        """One flow step over the request's new hidden rows [T, H] fp32.

        With flow_rows given (flow cross-request batching on), only the
        per-request half runs here — cond append, dit_head, the deterministic
        noise draw — and the row is collected; the caller flushes all rows
        of the engine step in ONE batched DiT + PatchEncoder pass
        (run_flow_rows). Otherwise the row flushes immediately (the serial
        path, bit-identical to a B=1 batch on an exact bucket hit).
        """
        row = self.prepare_flow_row(data, hidden_f32)
        if flow_rows is not None:
            flow_rows.append(row)
            return
        self.run_flow_rows([row])

    def prepare_flow_row(
        self, data: FireRedTTS3SGLangRequestData, hidden_f32: torch.Tensor
    ) -> FlowRow:
        """The per-request flow-step materials: cond append -> dit_head ->
        history window -> noise draw (outside any graph, from the request's
        own generator, preserving the deterministic draw order)."""
        model = self.model
        flow = data.flow_state
        with torch.inference_mode(), data.probe.segment("cond"):
            if flow.patch_count == 0 and flow.prompt_patch_count > 0:
                cond_rows = hidden_f32[-flow.prompt_patch_count :]
            else:
                cond_rows = hidden_f32[-1:]
            flow.backbone_cond = torch.cat(
                [flow.backbone_cond, cond_rows.unsqueeze(0)], dim=1
            )
            cond_window = flow.backbone_cond[:, -(model.history_patches + 1) :]
            bb_cond = model.dit_head(cond_window)
            bb_cond = bb_cond.repeat_interleave(model.patch_size, dim=1)
            if flow.spk_cond_dit is not None:
                spk = flow.spk_cond_dit.unsqueeze(1).expand(-1, bb_cond.shape[1], -1)
                cond = torch.cat([bb_cond, spk], dim=-1)
            else:
                cond = bb_cond
            hist = flow.latents_gen[:, -model.history_length :]
            x0 = torch.empty(
                1,
                model.patch_size,
                model.redae_dim,
                device=cond.device,
                dtype=torch.float32,
            )
            x0.normal_(generator=flow.generator)
        return (data, hist, x0, cond)

    def run_flow_rows(self, rows: list[FlowRow]) -> None:
        """The batched flow half over the engine step's collected rows:
        one DiT Euler replay (bucket zero-pad as needed) + one PatchEncoder
        pass, then the per-request writeback in collection order.

        Rows are grouped by n_timesteps — a graph replays one loop count, so
        a request overriding the ODE steps cannot share a replay with the
        defaults. cfg scales ride the per-row [B, 1, 1] placeholder, so
        per-task scales (2.0 / 1.2) do share one replay.
        """
        model = self.model
        groups: dict[int, list[FlowRow]] = {}
        for row in rows:
            groups.setdefault(int(row[0].state.n_timesteps), []).append(row)
        feedback_dtype = next(model.parameters()).dtype
        for n_timesteps, group in groups.items():
            with torch.inference_mode():
                hists = torch.cat([row[1] for row in group])
                x0s = torch.cat([row[2] for row in group])
                conds = torch.cat([row[3] for row in group])
                # Per-row CFG scales; cfg <= 0 rides the CFG-doubled shape
                # and blends to the conditional velocity exactly, so one
                # code path covers every scale.
                cfgs = torch.tensor(
                    [float(row[0].state.inference_cfg) for row in group],
                    dtype=torch.float32,
                    device=conds.device,
                ).view(-1, 1, 1)
                probes = [row[0].probe for row in group]
                with ProfilingProbe.shared_segment("dit_flow", probes):
                    graphs = self.flow_graphs
                    patches = (
                        graphs.run(hists, x0s, conds, cfgs, n_timesteps)
                        if graphs is not None
                        else None
                    )
                    if patches is None:
                        patches = euler_flow_loop(
                            model.dit,
                            torch.cat([hists, x0s], dim=1),
                            conds,
                            self.t_span(n_timesteps, conds.device),
                            cfgs,
                            n_timesteps,
                            model.patch_size,
                        )
                with ProfilingProbe.shared_segment("patch_encoder", probes):
                    # Batched rows aggregate to [1, B, llm_hidden] (the
                    # encoder's trailing unsqueeze); one row per request.
                    feedbacks = model.patch_encoder(patches)
                for index, (data, _hist, _x0, _cond) in enumerate(group):
                    flow = data.flow_state
                    patch = patches[index : index + 1]
                    flow.latents_gen = torch.cat([flow.latents_gen, patch], dim=1)
                    flow.patch_count += 1
                    data.latent_patches.append(patch)
                    data.pending_feedback_queue.append(
                        feedbacks[0, index].to(dtype=feedback_dtype)
                    )

    @staticmethod
    def hidden_states(result) -> torch.Tensor:
        logits_output = getattr(result, "logits_output", None)
        hidden = getattr(logits_output, "hidden_states", None)
        if hidden is None:
            hidden = getattr(result, "hidden_states", None)
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError(
                "FireRedTTS3 SGLang forward did not return hidden states"
            )
        return hidden

    def on_request_finished(self, request_id: str, req_data) -> None:
        data = self.active_requests.pop(request_id, None)
        if data is not None and data.probe is not None:
            data.probe.log(request_id)
        self.clear_request_data(req_data)

    def reset_request(self, request_id: str) -> None:
        req_data = self.active_requests.pop(request_id, None)
        if req_data is not None:
            self.clear_request_data(req_data)

    def release_retracted_flow_states(self) -> None:
        for req_data in self.active_requests.values():
            if (
                req_data.flow_state is not None
                and not isinstance(req_data.flow_state, FireRedFlowResume)
                and req_data.req.is_retracted
            ):
                self.suspend_request_data(req_data)

    def suspend_request_data(self, req_data: FireRedTTS3SGLangRequestData) -> None:
        flow = req_data.flow_state
        assert flow is not None and not isinstance(flow, FireRedFlowResume)
        req_data.flow_state = FireRedFlowResume(flow.generator.get_state())
        req_data.pending_feedback_queue.clear()

    def clear_request_data(self, req_data: FireRedTTS3SGLangRequestData) -> None:
        req_data.pending_feedback_queue.clear()
        req_data.flow_state = None
        req_data.stop_pending = False


__all__ = [
    "DS_STAGE_PATCH",
    "DS_STAGE_TEXT",
    "FireRedTTS3ModelRunner",
    "FlowRow",
    "FlowState",
]
