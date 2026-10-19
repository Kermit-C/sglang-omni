# FireRedTTS3

FireRedTTS3 is a token-free continuous-latent text-to-speech family. Audio is modeled end to end as 64-dim continuous latents at 25 Hz — there is no RVQ codebook and no audio token sampler. Each autoregressive step of the Qwen3-1.7B backbone produces one latent patch (4 frames = 160 ms of audio): the backbone hidden state conditions a flow-matching DiT that solves a 10-step Euler ODE for the patch, a PatchEncoder folds the patch back into the next backbone input embedding, and a stop head decides when to end. A RedAE encoder/decoder pair converts between 24 kHz waveform and latents; the decoder runs as the streaming vocoder stage.

Because the backbone emits no token logits in the audio phase, `temperature` / `top_p` / `top_k` do not shape the audio — use the solver knobs (`n_timesteps`, `inference_cfg`, `stop_threshold`) instead. The exception is the Instruct variant's text phase (voice_design / semantic_edit), where the backbone genuinely samples text and those knobs apply.

| Component | Spec |
|---|---|
| Backbone | Qwen3 1.7B decoder (28 L, hidden=2048, GQA 16/8) |
| Acoustic head | Flow-matching DiT (11 L, hidden=1024, 16 heads, AdaLN) + PatchEncoder (8 L) |
| Latent patch | 4 frames × 64 dims (one patch = 160 ms of audio, 3840 samples) |
| Vocoder | RedAE decoder (18 L Qwen3, SWA 64) + iSTFT head |
| Sample rate | 24 kHz |
| Solver | Euler, `n_timesteps=10` with classifier-free guidance |

## Variants

One architecture, three checkpoints. The variant is detected from the checkpoint's `config.json` at launch; each variant has its own example config.

| | Base | Instruct | Speaker |
|---|---|---|---|
| Example config | `examples/configs/firered_tts3_base.yaml` | `examples/configs/firered_tts3_instruct.yaml` | `examples/configs/firered_tts3_speaker.yaml` |
| Voice condition | Reference clip: CAM++ speaker embedding + in-context prompt latents | Reference clip as in-context prompt (tts task) or edit source | Built-in speaker token; no reference audio |
| Tasks | Voice cloning only | `tts`, `VoiceDesign`, `SemanticEdit`, `AcousticEdit` | Text-only `tts` |
| Default CFG | 2.0 | 2.0 (tts) / 1.2 (design, edits) | 1.2 |
| `max_patches` | 400 (~64 s) | 400 | 600 (~96 s) |
| Default `flow_mode` | `compile_bf16` | `fp32` (see below) | `compile_bf16` |
| Dual-stream decode form | no | no | opt-in via `extra_params.dual_stream` |
| Pipeline stages | preprocessing → reference_encode → talker → vocoder | same as Base | preprocessing → talker → vocoder (no reference_encode) |

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then launch the server with the config matching your checkpoint:

```bash
sgl-omni serve \
  --model-path /path/to/FireRedTTS3-Base \
  --config examples/configs/firered_tts3_base.yaml \
  --allowed-local-media-path docs/_static/audio \
  --port 8000
```

For Instruct or Speaker, swap the checkpoint and the config for `firered_tts3_instruct.yaml` / `firered_tts3_speaker.yaml`. The example configs pin `model_path`, so `--model-path` is only needed to serve a different checkpoint location.

Every GPU stage warms its cold-start paths at startup (encoders, DiT compile, CUDA graph capture, vocoder decode shapes), so the first served requests are already warm. The DiT compile for a concurrency bucket that warmup did not cover runs lazily on first use — a one-time ~45-85 s pause per process (~45 s with a warm torch-inductor FX cache, which persists across restarts).

The examples below read local clips; allow their directories with `--allowed-local-media-path`, or fetch reference audio over HTTP with `--allowed-media-domain`.

## Synthesizing Speech

### Voice Cloning (Base, and the Instruct tts task)

Base takes exactly one reference clip plus its transcript; the transcript is prefixed to the input so the model can align prompt audio with prompt text. A wrong transcript hurts cloning quality.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FireRedTTS3-Base",
    "input": "欢迎使用点点语音合成系统，这是一段零样本音色克隆测试。",
    "language": "Chinese",
    "references": [{
      "audio_path": "docs/_static/audio/ref_voice.wav",
      "text": "今天天气很好，我们一起去公园散步吧。"
    }],
    "seed": 42
  }' \
  --output output.wav
```

`ref_audio` / `ref_text` are accepted as a shorthand for `references[0].audio_path` / `references[0].text`. `language` defaults to `Chinese`; the checkpoint tags cover 24 languages plus 21 Chinese dialect tags (`ZH_Sichuan`, `ZH_Shanghai`, ...).

### Streaming

Streaming emits raw 24 kHz 16-bit mono PCM as latent patches decode, cutting time-to-first-audio to ~80 ms warm at concurrency 1. Set `"stream": true` and `"response_format": "pcm"`:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FireRedTTS3-Base",
    "input": "欢迎使用点点语音合成系统，这是一段零样本音色克隆测试。",
    "references": [{
      "audio_path": "docs/_static/audio/ref_voice.wav",
      "text": "今天天气很好，我们一起去公园散步吧。"
    }],
    "stream": true,
    "response_format": "pcm",
    "seed": 42
  }' \
  --output output.pcm

ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav
```

The WebSocket protocol at `/v1/audio/speech/stream` (session.config → input.text → input.commit / input.done) works the same way; `ref_audio` / `ref_text` / `language` / `seed` ride in the `session.config` message. See [TTS Model Usage](../basic_usage/tts.md#websocket-speech-streaming) for the full event flow.

### Instruction Tasks (Instruct)

The Instruct checkpoint adds three tasks on top of cloning. `task_type` accepts `VoiceDesign`, `SemanticEdit`, `AcousticEdit` (case- and separator-insensitive; `Base`/`TTS` mean the cloning task).

`VoiceDesign` synthesizes a voice from a text description — no reference audio, `instructions` required:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FireRedTTS3-Instruct",
    "input": "晚上好，欢迎收听今天的节目。这段声音由开源引擎实时合成。",
    "task_type": "VoiceDesign",
    "instructions": "A young woman with a gentle voice, speaking slowly.",
    "seed": 42
  }' \
  --output voice_design.wav
```

`SemanticEdit` rewrites the content of an existing clip (the reference audio is the edit source, capped at 20 s); `input` may be empty:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FireRedTTS3-Instruct",
    "input": "",
    "task_type": "SemanticEdit",
    "instructions": "Replace 公园 with 游乐园.",
    "ref_audio": "docs/_static/audio/ref_voice.wav",
    "seed": 42
  }' \
  --output semantic_edit.wav
```

`AcousticEdit` keeps the content and adjusts one acoustic attribute. The instruction comes from `instructions` verbatim, or is composed from `acoustic_attribute` (`Speed` / `Pitch` / `Volume`) and a value — the top-level `speed` field doubles as the Speed value:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FireRedTTS3-Instruct",
    "input": "",
    "task_type": "AcousticEdit",
    "ref_audio": "docs/_static/audio/ref_voice.wav",
    "speed": 0.8,
    "seed": 42
  }' \
  --output acoustic_edit.wav
```

VoiceDesign and SemanticEdit run a text autoregressive phase first (the model free-writes the voice plan / edited transcript, up to 200 tokens), so their first audio arrives later than a pure cloning request of the same length. Text-phase sampling defaults to the reference values (VoiceDesign: temperature 0.7, top_p 0.8, top_k 20; everything else greedy) and explicit `temperature` / `top_p` / `top_k` request fields override them.

### Speaker Variant and Dual-Stream

The Speaker variant conditions on a built-in speaker token — requests are text-only, and passing a reference audio is rejected with HTTP 400:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FireRedTTS3-Speaker",
    "input": "欢迎使用点点语音合成系统，这是一段零样本音色克隆测试。",
    "seed": 42
  }' \
  --output speaker.wav
```

The dual-stream decode form interleaves text and latent feeds (2 text tokens per patch group) and emits each patch's audio immediately — the reference's low-latency streaming inference form. Opt in per request with `extra_params.dual_stream` (serving rejects it on non-speaker checkpoints), optionally with an `extra_params.style` prefix:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FireRedTTS3-Speaker",
    "input": "欢迎使用点点语音合成系统，这是一段零样本音色克隆测试。",
    "extra_params": {"dual_stream": true, "style": "轻松自然地讲述"},
    "stream": true,
    "response_format": "pcm",
    "seed": 42
  }' \
  --output speaker_dual.pcm
```

`dual_stream` selects the decode form and is orthogonal to `stream` (the output mode): a non-streaming dual-stream request returns one WAV. Incremental text upload mid-generation (extending the text while audio is already streaming) is not supported.

### Request Parameters

Top-level fields (the endpoint drops unknown top-level keys silently — model knobs belong in `stage_params` / `tts_params` / `extra_params`):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | served model | Served model identifier |
| `input` | string | (required) | Text to synthesize; at most 400 characters. May be empty for SemanticEdit / AcousticEdit |
| `references` | list | variant | One item with `audio_path` + `text`. Required for Base and the Instruct tts task (both fields) and the edit tasks (audio only); forbidden for the Speaker variant and VoiceDesign |
| `ref_audio` / `ref_text` | string | `null` | Shorthand for `references[0]` |
| `response_format` | string | `"wav"` | Output format; `pcm` required when `stream` is true |
| `stream` | bool | `false` | Stream raw 24 kHz PCM chunks |
| `seed` | int | `null` | Per-request flow-sampler seed; fixed seed + fixed request is bit-exact |
| `language` | string | `"Chinese"` | Language tag for Base (24 languages + 21 ZH dialect tags) |
| `task_type` | string | `"tts"` | Instruct tasks: `VoiceDesign` / `SemanticEdit` / `AcousticEdit` (`Base`/`TTS` = cloning) |
| `instructions` | string | `null` | VoiceDesign description, SemanticEdit instruction, or a raw AcousticEdit template |
| `speed` | float | `1.0` | Consumed natively by AcousticEdit as the Speed value ("adjust the speed to Nx"); ignored by other tasks |
| `temperature` / `top_p` / `top_k` | | task defaults | Instruct text-phase sampling overrides only; the audio phase never samples |
| `max_new_tokens` | int | derived | Engine decode-step budget; rarely needed — the pipeline derives it from `max_patches` (dual-stream: `3 * max_patches + 20`) |

Solver and stop knobs ride under `stage_params.talker` (canonical) or `tts_params`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_timesteps` | int | `10` | Euler ODE steps per latent patch |
| `inference_cfg` | float | Base 2.0, Speaker 1.2, Instruct per task | Classifier-free guidance strength |
| `stop_threshold` | float | `0.5` | Stop-head probability that ends generation |
| `min_patches` | int | `6` | Patches generated before the stop head may fire (~1 s floor) |
| `max_patches` | int | `400` (Speaker `600`) | Patch budget; the deployment config's value caps requests |
| `seed` | int | `null` | Same as the top-level seed |

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "FireRedTTS3-Base",
    "input": "欢迎使用点点语音合成系统。",
    "ref_audio": "docs/_static/audio/ref_voice.wav",
    "ref_text": "今天天气很好，我们一起去公园散步吧。",
    "seed": 42,
    "stage_params": {"talker": {"n_timesteps": 10, "inference_cfg": 2.0, "max_patches": 200}}
  }' \
  --output output.wav
```

Speaker-only `extra_params` (validated at the serving layer):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dual_stream` | bool | `false` | Dual-stream decode form (2:1 text/latent interleave); 400 on non-speaker checkpoints |
| `style` | string | `null` | Style prefix for the dual-stream prompt; applies to the dual-stream form only |

AcousticEdit composition knobs under `stage_params.preprocessing` (or `tts_params`): `acoustic_attribute` (`Speed` / `Pitch` / `Volume`) and `acoustic_value` (float). An explicit `instructions` string overrides composition entirely.

### Flow Execution Modes

`flow_mode` is a launch-time talker factory setting (`stages.talker.factory.flow_mode` in the config), selecting the DiT execution stack inside the flow loop:

| Mode | What changes | Use |
|---|---|---|
| `fp32` | nothing — the reference stack | Bit-exact deployments; the Instruct default |
| `compile_fp32` | torch.compile over the fp32 DiT | Faster fp32 |
| `bf16` | DiT matmul body in bf16, eager | (measured slower than compile_bf16; kept for A/B) |
| `compile_bf16` | bf16 body + torch.compile | The throughput stack; Base / Speaker default |

Only the DiT matmul body lowers to bf16: the Euler state, noise, CFG blend, and time schedule stay fp32. Instruct pins `fp32` because its stop head sits near the decision threshold — every non-bit-exact mode measurably flips stop trajectories on some utterances (stopping 2-3 patches early). Base and Speaker default to `compile_bf16`, which keeps the c=1 stop trajectory identical and RTF ~35% lower than fp32.

Compile cost is per process and per batch bucket: the first request(s) that hit a new concurrency shape pay a one-time ~45-85 s torch.compile (~45 s with a warm inductor FX cache, which persists across restarts). Startup warmup pre-compiles the single-request shapes.

### Performance

Base voice cloning, one fixed Chinese request (~4.8 s audio), 1x L20, streaming path, warm service at the final commit. The vLLM column is the reference port's final tuned stack on 1x H20 (a longer ~7.8 s utterance); L20 and H20 measure within a few percent of each other on this stack's other TTS models.

| Metric | vLLM (H20) | SGLang-Omni (L20) |
|---|---:|---:|
| RTF, concurrency 1 | 0.309 | 0.230 |
| RTF, concurrency 8 | 0.213 | 0.342 |
| TTFP, concurrency 1 (warm) | 181-190 ms | 82 ms |

Concurrency-1 latency and TTFP beat the vLLM port clearly; vLLM still leads on c=8 RTF (its per-step engine overhead amortizes differently), so high-concurrency throughput remains the open frontier.

### Known Limitations

- Single GPU only (`tp_size=1`); one variant per deployment.
- End-to-end parity with the fp32 reference is statistical, not bitwise: any bf16 backbone (including the reference vLLM port) diverges chaotically after a few patches. At a fixed seed every request is deterministic and bit-exact against itself, and the c=1 (exact-bucket) paths stay bit-exact across the CUDA-graph and compile stacks; c>=2 batched flow output drifts within the known batched-GEMM envelope.
- Instruct VoiceDesign / SemanticEdit pay a text-AR phase (up to 200 tokens) before the first audio chunk.
- The Speaker variant takes no reference audio, and incremental text upload mid-generation is not supported.
- Concurrency above ~16 needs a VRAM budget review: each captured flow-graph batch row costs ~1.2 GB.
- `speed` is consumed only by AcousticEdit; it does not post-process other tasks' audio.
