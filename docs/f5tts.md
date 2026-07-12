# F5-TTS Engine

**F5-TTS** is a flow-matching text-to-speech model using a **Diffusion Transformer
(DiT)** backbone with ConvNeXt V2. It is the architecture behind
[Habibi-TTS](https://github.com/SWivid/Habibi-TTS) (Arabic dialectal TTS) and
several community fine-tunes. Like ZipVoice, it is an **iterative** engine — it runs
an Euler ODE sampling loop rather than a single-pass graph.

## Upstream

| | |
|---|---|
| Repo | <https://github.com/SWivid/F5-TTS> |
| Paper | *F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching* — [arXiv:2410.06885](https://arxiv.org/abs/2410.06885) |
| ONNX export | <https://github.com/DakeQQ/F5-TTS-ONNX> |
| Habibi-TTS | <https://github.com/SWivid/Habibi-TTS> — Arabic dialectal fine-tune |
| Languages | Multilingual (trained on Emilia: ZH, EN, + more) |
| ONNX weights | [`SWivid/F5-TTS`](https://huggingface.co/SWivid/F5-TTS) (PyTorch), ONNX via DakeQQ's export script |

## Architecture

F5-TTS pairs a **DiT** (Diffusion Transformer) backbone with **conditional flow
matching**. It uses **in-context** voice cloning: the reference audio + its
transcription are prepended to the target, and the model generates new speech in
the reference voice.

Three ONNX graphs make up the runtime pipeline (as exported by
[DakeQQ/F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX)):

```
text ─ tokenizer ─► text_ids ─┐
ref.wav ──────────────────────┤
                               ▼
        preprocess ─► noise, rope, cat_mel_text, ref_signal_len
                               │
        transformer × NFE ─────┘  (Euler ODE loop)
                               │
                               ▼
        decode ─► waveform      (Vocos + ISTFT)
```

### ONNX I/O

**`preprocess`** — reference audio + combined text → initial states

| Name | Type | Shape |
|---|---|---|
| `audio` | float32 | `[1, 1, audio_len]` |
| `text_ids` | int32 | `[1, text_len]` (ref_text + gen_text concatenated) |
| `max_duration` | int64 | `[1]` |
| → `noise` | float32 | `[1, max_duration, 100]` |
| → `rope_cos_q` | float32 | `[2, num_head, max_duration, head_dim]` |
| → `rope_sin_q` | float32 | `[2, num_head, max_duration, head_dim]` |
| → `rope_cos_k` | float32 | `[2, num_head, head_dim, max_duration]` |
| → `rope_sin_k` | float32 | `[2, num_head, head_dim, max_duration]` |
| → `cat_mel_text` | float32 | `[1, max_duration, n_mels + text_embed]` |
| → `cat_mel_text_drop` | float32 | `[1, max_duration, n_mels + text_embed]` |
| → `ref_signal_len` | int64 | scalar |

**`transformer`** — iterative denoising (one step per call)

| Name | Type | Shape |
|---|---|---|
| `noise` | float32 | `[1, max_duration, 100]` |
| `rope_cos_q` | float32 | `[2, num_head, max_duration, head_dim]` |
| `rope_sin_q` | float32 | `[2, num_head, max_duration, head_dim]` |
| `rope_cos_k` | float32 | `[2, num_head, head_dim, max_duration]` |
| `rope_sin_k` | float32 | `[2, num_head, head_dim, max_duration]` |
| `cat_mel_text` | float32 | `[1, max_duration, n_mels + text_embed]` |
| `cat_mel_text_drop` | float32 | `[1, max_duration, n_mels + text_embed]` |
| `time_step` | int32 | `[1]` |
| → `denoised` | float32 | `[1, max_duration, 100]` |
| → `time_step` | int32 | `[1]` |

**`decode`** — mel to waveform (Vocos + ISTFT baked in)

| Name | Type | Shape |
|---|---|---|
| `denoised` | float32 | `[1, max_duration, 100]` |
| `ref_signal_len` | int64 | scalar |
| → `output_audio` | int16/float32 | `[1, 1, audio_len]` |

### The sampling loop

Flow matching integrates `noise` from noise (`t=0`) to data (`t=1`) using a
pre-computed sway-sampled time schedule:

```python
# Pre-computed by the transformer export:
# time_steps = linspace(0,1,NFE) + sway_coef * (cos(pi/2 * t) - 1 + t)
# delta_t = diff(time_steps)
time_step = 0
for _ in range(nfe):
    noise, time_step = transformer.run(
        noise, rope_cos_q, rope_sin_q, rope_cos_k, rope_sin_k,
        cat_mel_text, cat_mel_text_drop, time_step
    )
```

`nfe` defaults to **32** (configurable; lower = faster, higher = better quality).
`cfg_strength` (default 2.0) controls classifier-free guidance strength, applied
internally by the transformer export.

### Habibi-TTS

Habibi-TTS is a fine-tune of F5-TTS for Arabic dialectal speech. It uses the same
architecture and ONNX layout, with different weights. It provides:

- **Unified** model (recommended) — handles all dialects
- **Specialized** models — per-dialect: MSA, SAU, UAE, ALG, IRQ, EGY, MAR, etc.

The Habibi-TTS vocabulary (`vocab.txt`) is Arabic-optimized. The phoonnx tokenizer
must be configured with the matching `phoneme_id_map` from Habibi's config.

## The adapter

`F5TTSAdapter` (`phoonnx/engines/f5tts.py`) overrides
`BaseOnnxAdapter.synthesize()` — the hook that lets an engine replace the default
single-graph pipeline with its own multi-ONNX loop. The voice's primary session is
the `transformer`; the `preprocess` and `decode` graphs (and an optional standalone
Vocos vocoder) are loaded from `engine_params` in `configure()`:

```json
{
    "engine": "f5tts",
    "engine_params": {
        "preprocess_path": "F5_Preprocess.onnx",
        "decode_path": "F5_Decode.onnx",
        "nfe": 32,
        "cfg_strength": 2.0,
        "sway_coefficient": -1.0,
        "target_rms": 0.15,
        "sample_rate": 24000
    }
}
```

If a standalone Vocos vocoder is provided via `vocoder_path`, it is used instead
of the baked-in `decode` graph (preferred for flexibility):

```json
{
    "engine_params": {
        "preprocess_path": "F5_Preprocess.onnx",
        "vocoder_path": "vocos-mel-24khz.onnx",
        "vocoder_type": "vocos"
    }
}
```

## Cloning

F5-TTS is an in-context engine, so it needs the reference **and its
transcription**:

```python
voice.synthesize("A line the reference never spoke.", SynthesisConfig(
    speaker_reference="reference.wav",
    speaker_reference_text="transcription of the reference clip",
))
```

See [Voice Cloning](cloning.md) for the full API.

## Variants

| Variant | Notes |
|---|---|
| **F5-TTS** | base model (DiT + ConvNeXt V2) |
| **F5-TTS v1** | improved training and inference (2025/03) |
| **Habibi-TTS** | Arabic dialectal fine-tune (Unified + Specialized) |
| **E2-TTS** | Flat-UNet variant (same flow-matching approach) |

Any model using the F5-TTS ONNX export layout (preprocess + transformer + decode)
is supported by this adapter.
