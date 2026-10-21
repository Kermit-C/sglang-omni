# SPDX-License-Identifier: Apache-2.0
"""Self-captured CUDA graphs over the RedAE decoder's cold (fresh-cache) pass.

A streaming request's first decode runs the prompt latents and the first
generated chunk jointly through an empty SWA cache (vocoder.py). That pass is
kernel-launch bound like the warm steady state (~36 ms eager for a 53-latent
joint decode on L20) but was left eager by the warm graphs: the frame count
varies with the prompt length and the cache starts empty.

The cold pass becomes graphable with left padding:

* latents are left-padded with zeros to the bucket width W (frames F = 2W
  after the in_proj upsample); pad rows sit at the front, so the stored
  SWA window (the last sliding_window-1 frames) is all-real whenever
  2N >= sliding_window-1 — the replay gate.
* positions are remapped per replay: pad frames get position 0, real frames
  get 0..2N-1, so RoPE sees exactly the unpadded geometry.
* the bool sliding+causal mask is base geometry (captured per bucket) ANDed
  with a per-replay pad block (pad columns hidden from real queries; pad
  queries stay plain-causal so their rows stay finite). Real-query geometry
  is then identical to the eager fresh-cache mask.
* outputs are trimmed out of the padded buffers: the real audio emission is
  a slice (pad-frame audio contaminates only the first win-hop real samples
  — the OLA lookahead span — which always lands inside the prompt region the
  vocoder trims; the caller passes the prompt's sample count as lead_samples
  so the runner can refuse layouts where it would leak), the stored KV is
  the all-real last-63 window, and cumulative_length/frames_seen are set to
  the REAL frame count so the next (warm) chunk's eager or graphed step sees
  the same state the eager cold pass would have produced.

An exact bucket hit (N == W, no pads) is bit-exact with the eager cold pass;
padded replays differ only by fp32 GEMM tiling noise (~1e-6, far below the
accepted chunked-vs-whole SWA floor). Padded batch rows are garbage whose
outputs are sliced away, mirroring the warm runner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from transformers.masking_utils import create_sliding_window_causal_mask

from sglang_omni.models.firered_tts3.redae_decoder import (
    RedAEAudioDecoder,
    RedAEDecoderState,
    build_initialized_cache,
)
from sglang_omni.models.firered_tts3.vocoder_graph import WARMUP_ITERATIONS

logger = logging.getLogger(__name__)

# Joint-decode widths (25 Hz latent frames; prompt + first generated chunks).
# Above the top bucket the cold pass stays eager (a once-per-request cost).
DEFAULT_COLD_LATENT_SIZES = (32, 48, 64, 96, 128)
# The launch-bound regime only: at B>=3 the fp32 decode turns compute-bound
# and bucket padding inflates the replay above eager (measured on L20:
# B=1/2 replay 5.1/7.7 ms vs eager 13.4/13.7 ms; B=3 padded-to-4 is a wash
# at ~13.5 ms both ways). Wider groups stay eager.
DEFAULT_COLD_BATCH_SIZES = (1, 2)


@dataclass
class ColdGraphEntry:
    graph: torch.cuda.CUDAGraph
    latents: torch.Tensor  # [B, W, in_dim] fp32
    positions: torch.Tensor  # [1, 2W] int64
    mask: torch.Tensor  # [1, 1, 2W, 2W] bool
    mask_base: torch.Tensor  # [2W, 2W] bool, sliding+causal geometry
    row_ids: torch.Tensor  # [2W, 1] int64
    col_ids: torch.Tensor  # [1, 2W] int64
    audio: torch.Tensor  # [B, 2W*hop - (win-hop) - pad_head] fp32
    kv_out: torch.Tensor  # [2*layers, B, kv_heads, stored, head_dim] fp32
    istft_out: torch.Tensor  # [B, n_fft, win//hop-1] fp32
    keepalives: list[object] = field(default_factory=list)


class RedAEVocoderColdGraphRunner:
    """Replay the RedAE decoder's fresh-cache joint decode per (width, bucket)."""

    def __init__(
        self,
        decoder: RedAEAudioDecoder,
        *,
        latent_sizes: tuple[int, ...] = DEFAULT_COLD_LATENT_SIZES,
        batch_sizes: tuple[int, ...] = DEFAULT_COLD_BATCH_SIZES,
    ) -> None:
        self.decoder = decoder
        self.device = next(decoder.parameters()).device
        config = decoder.qwen3_config
        self.num_layers = int(config.num_hidden_layers)
        attn = decoder.qwen3.layers[0].self_attn
        self.num_kv_heads = int(attn.k_proj.out_features // attn.head_dim)
        self.head_dim = int(attn.head_dim)
        self.stored_frames = int(config.sliding_window) - 1
        # The stored window must be all-real: 2N >= stored frames.
        self.min_latents = -(-self.stored_frames // 2)
        istft = decoder.istft_head.istft
        # Real samples below this lead can differ from eager (OLA lookahead
        # plus the window-envelope head ramp); they must fall inside the
        # trimmed prompt region.
        self.min_lead_samples = int(istft.win_length) - int(istft.hop_length)
        self.latent_sizes = tuple(
            sorted(
                {int(size) for size in latent_sizes if int(size) >= self.min_latents}
            )
        )
        self.batch_sizes = tuple(
            sorted({int(size) for size in batch_sizes if int(size) > 0})
        )
        self.enabled = (
            self.device.type == "cuda"
            and bool(self.latent_sizes)
            and bool(self.batch_sizes)
        )
        self.graphs: dict[tuple[int, int], ColdGraphEntry] = {}
        self.failed_keys: set[tuple[int, int]] = set()
        self.graph_pool = None
        # Capture stream dedicated to this runner; see flow_graph's note on
        # why capture streams must not be shared across graph families.
        self.capture_stream: torch.cuda.Stream | None = None
        self.counters = {"calls": 0, "replays": 0, "eager": 0, "captures": 0}

    def stats(self) -> dict[str, object]:
        return {
            **self.counters,
            "graphs": sorted(self.graphs),
            "failed_keys": sorted(self.failed_keys),
        }

    def capture_all(self) -> None:
        """Capture the configured grid; call at warmup time, never per request."""
        if not self.enabled:
            return
        else:
            pass
        for batch_size in sorted(self.batch_sizes, reverse=True):
            for latent_size in sorted(self.latent_sizes, reverse=True):
                key = (latent_size, batch_size)
                if key in self.graphs or key in self.failed_keys:
                    continue
                else:
                    pass
                entry = self.capture(latent_size, batch_size)
                if entry is None:
                    self.failed_keys.add(key)
                    continue
                else:
                    pass
                self.graphs[key] = entry
                self.counters["captures"] += 1
        if not self.graphs:
            self.enabled = False
            logger.warning(
                "FireRedTTS3 vocoder cold CUDA graphs: no key captured; eager only"
            )
            return
        else:
            pass
        logger.info(
            f"FireRedTTS3 vocoder cold CUDA graphs captured: {sorted(self.graphs)}"
            + (f" (skipped: {sorted(self.failed_keys)})" if self.failed_keys else "")
        )

    def capture(self, latent_size: int, batch_size: int) -> ColdGraphEntry | None:
        """Capture one (width, bucket) cold decode; None on failure."""
        decoder = self.decoder
        config = decoder.qwen3_config
        hidden = int(config.hidden_size)
        frames = latent_size * int(decoder.upsample_rate)
        istft = decoder.istft_head.istft
        latents = torch.zeros(
            batch_size,
            latent_size,
            decoder.in_proj.in_features,
            device=self.device,
            dtype=torch.float32,
        )
        positions = torch.arange(
            frames, device=self.device, dtype=torch.long
        ).unsqueeze(0)
        # Fresh-cache non-terminal emission: (frames-1)*hop + win, minus the
        # head pad (win-hop)//2 and the (win-hop) OLA tail -> frames*hop minus
        # the head pad. The buffer holds the padded emission; the real one is
        # sliced out per replay.
        pad_head = (istft.win_length - istft.hop_length) // 2
        emission = frames * istft.hop_length - pad_head
        audio = torch.zeros(
            batch_size, emission, device=self.device, dtype=torch.float32
        )
        kv_out = torch.zeros(
            2 * self.num_layers,
            batch_size,
            self.num_kv_heads,
            self.stored_frames,
            self.head_dim,
            device=self.device,
            dtype=torch.float32,
        )
        cache_frames = istft.win_length // istft.hop_length - 1
        istft_out = torch.zeros(
            batch_size,
            istft.n_fft,
            cache_frames,
            device=self.device,
            dtype=torch.float32,
        )
        graph_cache = build_initialized_cache(config, device=self.device)
        # The captured mask is the fresh-cache sliding+causal geometry (the
        # eager cold mask); the pad block is applied into the buffer per replay.
        mask = create_sliding_window_causal_mask(
            config=config,
            inputs_embeds=torch.zeros(1, frames, hidden, device=self.device),
            attention_mask=None,
            past_key_values=graph_cache,
            position_ids=None,
        )
        assert mask is not None and mask.dtype == torch.bool
        mask = mask.clone()
        mask_base = mask[0, 0].clone()
        row_ids = torch.arange(frames, device=self.device).unsqueeze(1)
        col_ids = torch.arange(frames, device=self.device).unsqueeze(0)
        mask_dict = {"full_attention": None, "sliding_attention": mask}

        def body() -> None:
            xs = decoder.in_proj(latents)
            xs = xs.reshape(xs.shape[0], -1, hidden)
            outs = decoder.qwen3.forward(
                inputs_embeds=xs,
                use_cache=True,
                past_key_values=graph_cache,
                position_ids=positions,
                attention_mask=mask_dict,
            )
            chunk, new_istft_cache = decoder.istft_head.forward_stream(
                outs.last_hidden_state, None, False
            )
            audio.copy_(chunk)
            istft_out.copy_(new_istft_cache)
            for index, layer in enumerate(graph_cache.layers):
                kv_out[2 * index].copy_(layer.keys)
                kv_out[2 * index + 1].copy_(layer.values)

        try:
            with torch.no_grad():
                # Same-stream synchronized warmups (autotune/cuFFT plans
                # outside the capture). Every iteration restarts from the
                # fresh cache: a prior run's update leaves the stored window
                # behind, and the cold capture must record the empty-cache
                # pass.
                empty = torch.zeros(0, device=self.device, dtype=torch.float32)
                for _ in range(WARMUP_ITERATIONS + 1):
                    for layer in graph_cache.layers:
                        layer.keys = empty
                        layer.values = empty
                        layer.cumulative_length = 0
                    body()
                torch.cuda.synchronize()
                for layer in graph_cache.layers:
                    layer.keys = empty
                    layer.values = empty
                    layer.cumulative_length = 0
                if self.graph_pool is None:
                    self.graph_pool = torch.cuda.graph_pool_handle()
                else:
                    pass
                if self.capture_stream is None:
                    self.capture_stream = torch.cuda.Stream(device=self.device)
                else:
                    pass
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph,
                    pool=self.graph_pool,
                    stream=self.capture_stream,
                    capture_error_mode="thread_local",
                ):
                    body()
        except Exception:
            logger.warning(
                f"FireRedTTS3 vocoder cold CUDA graph capture failed for "
                f"latents={latent_size} batch={batch_size}; eager fallback",
                exc_info=True,
            )
            return None
        return ColdGraphEntry(
            graph=graph,
            latents=latents,
            positions=positions,
            mask=mask,
            mask_base=mask_base,
            row_ids=row_ids,
            col_ids=col_ids,
            audio=audio,
            kv_out=kv_out,
            istft_out=istft_out,
            keepalives=[graph_cache, mask_dict],
        )

    def find_entry(
        self, latents: torch.Tensor, lead_samples: int
    ) -> ColdGraphEntry | None:
        """The captured entry this cold decode can replay; None for eager."""
        if (
            not self.enabled
            or latents.device.type != "cuda"
            or latents.dtype != torch.float32
            or torch.cuda.is_current_stream_capturing()
            or lead_samples < self.min_lead_samples
        ):
            return None
        else:
            pass
        batch_size, latent_size = int(latents.shape[0]), int(latents.shape[1])
        if latent_size < self.min_latents:
            return None
        else:
            pass
        bucket = next((size for size in self.batch_sizes if size >= batch_size), None)
        if bucket is None:
            return None
        else:
            pass
        width = next((size for size in self.latent_sizes if size >= latent_size), None)
        if width is None:
            return None
        else:
            pass
        return self.graphs.get((width, bucket))

    def run(
        self, latents: torch.Tensor, *, lead_samples: int
    ) -> tuple[torch.Tensor, RedAEDecoderState] | None:
        """One fresh-cache joint decode via graph replay; None requests eager.

        Returns (the real audio emission [B, samples], the advanced state
        carrying the all-real stored window and the real frame count). The
        audio aliases the entry's static buffer: consume it (host copy)
        before the next replay of the same entry.
        """
        self.counters["calls"] += 1
        entry = self.find_entry(latents, lead_samples)
        if entry is None:
            self.counters["eager"] += 1
            return None
        else:
            pass
        batch_size, latent_size = int(latents.shape[0]), int(latents.shape[1])
        width = int(entry.latents.shape[1])
        upsample = int(self.decoder.upsample_rate)
        real_frames = latent_size * upsample
        pad_frames = width * upsample - real_frames
        try:
            with torch.no_grad():
                entry.latents.zero_()
                entry.latents[:batch_size, width - latent_size :, :].copy_(latents)
                if pad_frames:
                    entry.positions[0, :pad_frames].zero_()
                else:
                    pass
                torch.arange(
                    real_frames,
                    dtype=torch.long,
                    device=self.device,
                    out=entry.positions[0, pad_frames:],
                )
                pad_block = (entry.col_ids >= pad_frames) | (entry.row_ids < pad_frames)
                torch.logical_and(entry.mask_base, pad_block, out=entry.mask[0, 0])
                entry.graph.replay()
            self.counters["replays"] += 1
        except Exception:
            # The caller's state is untouched; disable and let eager answer.
            self.enabled = False
            self.counters["eager"] += 1
            logger.warning(
                "FireRedTTS3 vocoder cold CUDA graph replay failed; disabling graphs",
                exc_info=True,
            )
            return None
        kv_snapshot = entry.kv_out[:, :batch_size].clone()
        istft_snapshot = entry.istft_out[:batch_size].clone()
        cache = build_initialized_cache(
            self.decoder.qwen3_config, device=self.device, dtype=torch.float32
        )
        for index, layer in enumerate(cache.layers):
            layer.keys = kv_snapshot[2 * index]
            layer.values = kv_snapshot[2 * index + 1]
            layer.cumulative_length = real_frames
        new_state = RedAEDecoderState(cache)
        new_state.istft_cache = istft_snapshot
        new_state.frames_seen = real_frames
        hop = int(self.decoder.istft_head.istft.hop_length)
        # The padded emission is its own fresh-cache trim of the padded fold;
        # the real emission (eager coordinates) is exactly its tail: real
        # sample r sits at padded-emitted index pad_frames*hop + r. Samples
        # below (win-hop)//2 mix eager's head envelope ramp with the padded
        # interior normalization and never match; the lead gate keeps them
        # inside the trimmed prompt region.
        start = pad_frames * hop
        length = real_frames * hop - self.min_lead_samples // 2
        return entry.audio[:batch_size, start : start + length], new_state


__all__ = [
    "DEFAULT_COLD_BATCH_SIZES",
    "DEFAULT_COLD_LATENT_SIZES",
    "RedAEVocoderColdGraphRunner",
]
