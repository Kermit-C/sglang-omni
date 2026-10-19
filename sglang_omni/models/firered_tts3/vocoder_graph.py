# SPDX-License-Identifier: Apache-2.0
"""Self-captured CUDA graphs over the RedAE decoder's steady-state chunks.

A warm chunk decode (in_proj -> 18-layer SWA(64) Qwen3 -> ISTFTHead+iSTFT,
all fp32) is kernel-launch bound: ~16 ms eager on L20 for a 4-patch chunk at
any chunk width, vs ~4 ms as a captured graph. The steady state is a fixed
shape point: every SWA layer's stored cache is exactly sliding_window-1
frames once the stream is saturated, the attention mask over the stored
window + new frames is position-invariant (the relative causal+sliding
geometry never changes), and only RoPE positions advance per chunk.

Semantics:

* the capture body is the same HF module sequence forward_stream runs
  (in_proj, Qwen3Model.forward, ISTFTHead.forward_stream with
  last_chunk=False), so a replay is bit-exact with the eager warm step at
  B=1 (verified torch.equal on audio, stored KV, and iSTFT cache);
* the per-chunk-varying inputs ride static buffers: latents, the stored KV
  (one stacked [2*layers, B, kv_heads, stored, head_dim] tensor per batch
  bucket, shared across chunk widths), the iSTFT OLA cache, and the position
  ids; the mask is captured per key exactly as the eager path builds it;
* the stored-KV/iSTFT updates are materialized into static output buffers
  inside the graph (the HF cache update's slice views live in the graph
  pool at fixed addresses; the in-graph copies make the handoff explicit),
  then cloned out per replay so later replays cannot overwrite live state;
* every tensor a graph reads is owned by its entry — a captured graph
  records raw addresses, and a freed static buffer is silent
  use-after-free (the allocator reuses the block for unrelated work);
* buckets pad B up with garbage rows (every op is row-local, so padded rows
  never contaminate real ones); chunk width is never padded because frame
  padding would advance the causal decoder state;
* capture happens at warmup time for the whole configured grid; a failed
  key is skipped, a replay failure disables the runner — both fall back to
  the eager forward_stream.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache
from transformers.masking_utils import create_sliding_window_causal_mask

from sglang_omni.models.firered_tts3.redae_decoder import (
    RedAEAudioDecoder,
    RedAEDecoderState,
    build_initialized_cache,
)

logger = logging.getLogger(__name__)

# Chunk widths (25 Hz latent frames) matching the talker's 1/2/4-patch ramp.
DEFAULT_LATENT_SIZES = (4, 8, 16)
# Batch buckets; a smaller group pads up with garbage rows whose outputs are
# dropped (row-local ops only). Capped at 8: the graphs recover kernel-launch
# overhead, which dominates only while the decode stays launch-bound — at
# B=16 the fp32 step turns compute-bound (~15 ms either way on L20), where a
# monolithic replay only adds scheduling burstiness against the talker.
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)

# Eager iterations settling cuDNN/cuBLAS autotuning and cuFFT plans before
# each capture (same-stream, synchronized — see flow_graph).
WARMUP_ITERATIONS = 3


@dataclass
class BucketBuffers:
    """Fixed-address decode state shared by every chunk-width graph of one
    batch bucket (replays are serialized on the vocoder's decode stream)."""

    kv_in: torch.Tensor  # [2*layers, B, kv_heads, stored, head_dim] fp32
    kv_out: torch.Tensor  # same layout
    istft_in: torch.Tensor  # [B, n_fft, win//hop-1] fp32
    istft_out: torch.Tensor


@dataclass
class VocoderGraphEntry:
    graph: torch.cuda.CUDAGraph
    latents: torch.Tensor  # [B, latents_per_chunk, in_dim]
    positions: torch.Tensor  # [1, 2*latents_per_chunk] int64
    mask: torch.Tensor  # [B, 1, frames, stored+frames] bool
    audio: torch.Tensor  # [B, frames*hop] fp32
    buffers: BucketBuffers
    # Strong references to the remaining objects whose tensors the graph
    # addresses (the HF cache whose layer attrs point into kv_in, and the
    # mask dict handed to the model).
    keepalives: list[object] = field(default_factory=list)


class RedAEVocoderGraphRunner:
    """Replay the RedAE decoder's warm step via per-(width, bucket) graphs."""

    def __init__(
        self,
        decoder: RedAEAudioDecoder,
        *,
        latent_sizes: tuple[int, ...] = DEFAULT_LATENT_SIZES,
        batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
    ) -> None:
        self.decoder = decoder
        self.device = next(decoder.parameters()).device
        config = decoder.qwen3_config
        self.num_layers = int(config.num_hidden_layers)
        attn = decoder.qwen3.layers[0].self_attn
        self.num_kv_heads = int(attn.k_proj.out_features // attn.head_dim)
        self.head_dim = int(attn.head_dim)
        # A saturated SWA layer stores exactly window-1 frames (update keeps
        # the last sliding_window-1 so stored+new stays within the window).
        self.stored_frames = int(config.sliding_window) - 1
        self.latent_sizes = tuple(
            sorted({int(size) for size in latent_sizes if int(size) > 0})
        )
        self.batch_sizes = tuple(
            sorted({int(size) for size in batch_sizes if int(size) > 0})
        )
        self.enabled = self.device.type == "cuda"
        self.graphs: dict[tuple[int, int], VocoderGraphEntry] = {}
        self.buckets: dict[int, BucketBuffers] = {}
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
        for batch_size in sorted(self.batch_sizes, reverse=True):
            for latent_size in sorted(self.latent_sizes, reverse=True):
                key = (latent_size, batch_size)
                if key in self.graphs or key in self.failed_keys:
                    continue
                entry = self.capture(latent_size, batch_size)
                if entry is None:
                    self.failed_keys.add(key)
                    continue
                self.graphs[key] = entry
                self.counters["captures"] += 1
        if not self.graphs:
            self.enabled = False
            logger.warning(
                "FireRedTTS3 vocoder CUDA graphs: no key captured; eager only"
            )
            return
        logger.info(
            f"FireRedTTS3 vocoder CUDA graphs captured: {sorted(self.graphs)}"
            + (f" (skipped: {sorted(self.failed_keys)})" if self.failed_keys else "")
        )

    def bucket_buffers(self, batch_size: int) -> BucketBuffers:
        buffers = self.buckets.get(batch_size)
        if buffers is None:
            shape = (
                2 * self.num_layers,
                batch_size,
                self.num_kv_heads,
                self.stored_frames,
                self.head_dim,
            )
            istft = self.decoder.istft_head.istft
            cache_frames = istft.win_length // istft.hop_length - 1
            buffers = BucketBuffers(
                kv_in=torch.zeros(shape, device=self.device, dtype=torch.float32),
                kv_out=torch.zeros(shape, device=self.device, dtype=torch.float32),
                istft_in=torch.zeros(
                    batch_size,
                    istft.n_fft,
                    cache_frames,
                    device=self.device,
                    dtype=torch.float32,
                ),
                istft_out=torch.zeros(
                    batch_size,
                    istft.n_fft,
                    cache_frames,
                    device=self.device,
                    dtype=torch.float32,
                ),
            )
            self.buckets[batch_size] = buffers
        return buffers

    def capture(self, latent_size: int, batch_size: int) -> VocoderGraphEntry | None:
        """Capture one (width, bucket) graph; None on failure (eager fallback)."""
        decoder = self.decoder
        config = decoder.qwen3_config
        hidden = int(config.hidden_size)
        frames = latent_size * int(decoder.upsample_rate)
        buffers = self.bucket_buffers(batch_size)
        latents = torch.zeros(
            batch_size,
            latent_size,
            decoder.in_proj.in_features,
            device=self.device,
            dtype=torch.float32,
        )
        positions = torch.zeros(1, frames, dtype=torch.long, device=self.device)
        audio = torch.zeros(
            batch_size,
            frames * decoder.istft_head.hop_length,
            device=self.device,
            dtype=torch.float32,
        )
        graph_cache = build_initialized_cache(config, device=self.device)
        for index, layer in enumerate(graph_cache.layers):
            layer.keys = buffers.kv_in[2 * index]
            layer.values = buffers.kv_in[2 * index + 1]
            # Any saturated value works: the mask geometry is provably
            # cumulative-invariant once the stored window is full.
            layer.cumulative_length = self.stored_frames + 1
        # The mask exactly as the eager path builds it for this state; the
        # sliding+causal pattern is constant across steady-state positions.
        mask = create_sliding_window_causal_mask(
            config=config,
            inputs_embeds=torch.zeros(
                batch_size, frames, hidden, device=self.device, dtype=torch.float32
            ),
            attention_mask=None,
            past_key_values=graph_cache,
            position_ids=positions,
        )
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
                outs.last_hidden_state, buffers.istft_in, False
            )
            audio.copy_(chunk)
            buffers.istft_out.copy_(new_istft_cache)
            for index, layer in enumerate(graph_cache.layers):
                buffers.kv_out[2 * index].copy_(layer.keys)
                buffers.kv_out[2 * index + 1].copy_(layer.values)

        try:
            with torch.no_grad():
                # Warm runs stay on the current (decode) stream, synchronized:
                # autotuning and cuFFT plan creation must happen outside the
                # capture, and side-stream warmup perturbs the cuBLAS
                # workspace enough to shift later numerics run-to-run.
                for _ in range(WARMUP_ITERATIONS):
                    body()
                torch.cuda.synchronize()
                for index, layer in enumerate(graph_cache.layers):
                    layer.keys = buffers.kv_in[2 * index]
                    layer.values = buffers.kv_in[2 * index + 1]
                if self.graph_pool is None:
                    self.graph_pool = torch.cuda.graph_pool_handle()
                if self.capture_stream is None:
                    self.capture_stream = torch.cuda.Stream(device=self.device)
                graph = torch.cuda.CUDAGraph()
                # thread_local: colocated stages may keep using CUDA on other
                # threads while this capture records.
                with torch.cuda.graph(
                    graph,
                    pool=self.graph_pool,
                    stream=self.capture_stream,
                    capture_error_mode="thread_local",
                ):
                    body()
        except Exception:
            logger.warning(
                f"FireRedTTS3 vocoder CUDA graph capture failed for "
                f"latents={latent_size} batch={batch_size}; eager fallback",
                exc_info=True,
            )
            return None
        return VocoderGraphEntry(
            graph=graph,
            latents=latents,
            positions=positions,
            mask=mask,
            audio=audio,
            buffers=buffers,
            keepalives=[graph_cache, mask_dict],
        )

    def find_entry(
        self, latents: torch.Tensor, state: RedAEDecoderState
    ) -> VocoderGraphEntry | None:
        """The captured entry this warm step can replay; None for eager."""
        if (
            not self.enabled
            or latents.device.type != "cuda"
            or latents.dtype != torch.float32
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        batch_size, latent_size = int(latents.shape[0]), int(latents.shape[1])
        bucket = next((size for size in self.batch_sizes if size >= batch_size), None)
        if bucket is None:
            return None
        entry = self.graphs.get((latent_size, bucket))
        if entry is None or state.qwen3_cache is None or state.istft_cache is None:
            return None
        keys = state.qwen3_cache.layers[0].keys
        if (
            keys is None
            or keys.ndim != 4
            or int(keys.shape[0]) != batch_size
            or int(keys.shape[-2]) != self.stored_frames
        ):
            # Ramp-up (stored window not yet full) stays eager.
            return None
        return entry

    def run(
        self, latents: torch.Tensor, state: RedAEDecoderState
    ) -> tuple[torch.Tensor, RedAEDecoderState] | None:
        """One warm chunk decode via graph replay; None requests eager.

        Returns (audio rows [B, samples], the advanced batched state). The
        audio aliases the entry's static buffer: consume it (host copy)
        before the next replay of the same entry.
        """
        self.counters["calls"] += 1
        entry = self.find_entry(latents, state)
        if entry is None:
            self.counters["eager"] += 1
            return None
        batch_size = int(latents.shape[0])
        bucket = int(entry.latents.shape[0])
        buffers = entry.buffers
        frames = int(entry.positions.shape[1])
        try:
            # no_grad: the staging copies reject autograd-tracked sources
            # (the serving paths always run under no_grad; tests may not).
            with torch.no_grad():
                entry.latents[:batch_size].copy_(latents)
                cache_layers = state.qwen3_cache.layers
                kv_tensors = []
                for layer in cache_layers:
                    kv_tensors.append(layer.keys)
                    kv_tensors.append(layer.values)
                if batch_size == bucket:
                    torch.stack(kv_tensors, dim=0, out=buffers.kv_in)
                else:
                    buffers.kv_in[:, :batch_size].copy_(torch.stack(kv_tensors, dim=0))
                buffers.istft_in[:batch_size].copy_(state.istft_cache)
                torch.arange(
                    state.frames_seen,
                    state.frames_seen + frames,
                    dtype=torch.long,
                    device=self.device,
                    out=entry.positions[0],
                )
                entry.graph.replay()
            self.counters["replays"] += 1
        except Exception:
            # The caller's state is untouched; disable and let eager answer.
            self.enabled = False
            self.counters["eager"] += 1
            logger.warning(
                "FireRedTTS3 vocoder CUDA graph replay failed; disabling graphs",
                exc_info=True,
            )
            return None
        kv_snapshot = buffers.kv_out[:, :batch_size].clone()
        istft_snapshot = buffers.istft_out[:batch_size].clone()
        new_frames_seen = state.frames_seen + frames
        merged_cache = DynamicCache(config=self.decoder.qwen3_config)
        for index, layer in enumerate(merged_cache.layers):
            source = copy.copy(cache_layers[index])
            source.keys = kv_snapshot[2 * index]
            source.values = kv_snapshot[2 * index + 1]
            source.cumulative_length = new_frames_seen
            merged_cache.layers[index] = source
        new_state = RedAEDecoderState(merged_cache)
        new_state.istft_cache = istft_snapshot
        new_state.frames_seen = new_frames_seen
        return entry.audio[:batch_size], new_state


__all__ = [
    "DEFAULT_BATCH_SIZES",
    "DEFAULT_LATENT_SIZES",
    "RedAEVocoderGraphRunner",
]
