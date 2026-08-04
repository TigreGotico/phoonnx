# F5-TTS Engine

This page is for integrators cloning voices with F5-TTS / Habibi-TTS / SILMA in
phoonnx. After reading it you can load a converted voice, clone from a reference
clip plus its transcription, control speed and Arabic dialect, and wire up the
multi-graph runtime.

> Related: [adapter architecture](../../engines.md) ·
> [configuration](../../configuration.md) · [voice cloning](../../cloning.md) ·
> [ZipVoice — the other in-context iterative engine](zipvoice.md)

## What it is

**F5-TTS** is a flow-matching text-to-speech model using a **Diffusion
Transformer (DiT)** backbone with ConvNeXt V2. It is the architecture behind
[Habibi-TTS](https://github.com/SWivid/Habibi-TTS) (Arabic dialectal TTS) and
several community fine-tunes. Like ZipVoice it is an **iterative** engine — it
runs an Euler ODE sampling loop rather than a single-pass graph, using
**in-context** cloning (reference audio + its transcription prepended to the
target).

## When to pick it

Choose F5-TTS for high-fidelity in-context cloning, especially for Arabic
dialectal speech (Habibi / NAMAA / SILMA). Like [ZipVoice](zipvoice.md) it needs
the reference's transcription; contrast with the d-vector engines (YourTTS,
StyleTTS2, Chatterbox) that clone from audio alone.

## Extras needed

Cloning (reference loading + resampling) uses `pip install phoonnx[cloning]`
(`soundfile`, `scipy`).

## Architecture

F5-TTS pairs a **DiT** backbone with **conditional flow matching**. Three ONNX
graphs make up the runtime pipeline (as exported by
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
| → `rope_cos_q` / `rope_sin_q` | float32 | `[2, num_head, max_duration, head_dim]` |
| → `rope_cos_k` / `rope_sin_k` | float32 | `[2, num_head, head_dim, max_duration]` |
| → `cat_mel_text` / `cat_mel_text_drop` | float32 | `[1, max_duration, n_mels + text_embed]` |
| → `ref_signal_len` | int64 | scalar |

**`transformer`** — iterative denoising (one step per call): `noise`, the four
`rope_*`, `cat_mel_text`, `cat_mel_text_drop`, `time_step` → `denoised`,
`time_step`.

**`decode`** — mel to waveform (Vocos + ISTFT baked in): `denoised`,
`ref_signal_len` → `output_audio` (int16 or float32 `[1, 1, audio_len]`).

### The sampling loop

Flow matching integrates `noise` from `t=0` to `t=1` using a pre-computed
sway-sampled time schedule (`nfe` defaults to **32**; lower = faster,
higher = better). `cfg_strength` (default 2.0) controls classifier-free guidance,
applied internally by the transformer export.

> **ONNX-name auto-detect fallback.** `torch.onnx.export` can rename an input
> that shares a name with an output (e.g. `time_step` → `time_step.1`). The
> adapter reads the transformer's declared input names (`session.get_inputs()`)
> and feeds by declared order rather than by hardcoded name, so it survives that
> renaming across export runs / PyTorch versions.

## The adapter

`F5TTSAdapter` (`phoonnx/engines/f5tts.py`) overrides
`BaseOnnxAdapter.synthesize()`. The voice's primary session is the
`transformer`; the `preprocess` and `decode` graphs (and an optional standalone
Vocos vocoder) load from `engine_params` in `configure()`:

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
        "sample_rate": 24000,
        "speed": 1.0
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

If **neither** `decode_path` nor `vocoder_path` is supplied, synthesis raises a
`RuntimeError`.

## Parameters

| Param | Default | Description |
|---|---|---|
| `speed` | 1.0 | Speaking rate — scales the estimated generation duration (>1 faster). Settable in `engine_params` or per call via `extra_params`. |
| `nfe` | 32 | ODE steps (quality/speed trade-off) |
| `cfg_strength` | 2.0 | Classifier-free guidance strength |
| `dialect` | `UNK` | Habibi Unified dialect control tag (see below) |

## Habibi-TTS dialect control (Unified model only)

The Unified model was trained with **dialect control tokens**: upstream wraps the
generation text as `{dialect_char}〈text〉` where each dialect maps to an
enclosed-number character (`habibi_tts/model/utils.py`):

`UNK ⓪ · MSA ① · SAU ② · UAE ③ · ALG ④ · IRQ ⑤ · EGY ⑥ · MAR ⑦`
(plus `OMN ⑧ · TUN ⑨ · LEV ⑩ · SDN ⑪ · LBY ⑫`, present in the vocab but without
training data — only the first 8 are meaningful).

The adapter applies the tag at token level (through the voice's `phoneme_id_map`)
when a `dialect` engine param is set — in the voice config
(`"engine_params": {"dialect": "UNK"}`, the default for the published unified
voice) or per call via `SynthesisConfig.extra_params={"dialect": "EGY"}`.
Specialized per-dialect models take plain untagged text.

**Unhandled dialect error.** `_wrap_dialect` raises a `ValueError` for an unknown
dialect name. If the voice's vocab lacks the control tokens (the specialized
per-dialect models, or plain F5-TTS), the tag is skipped with a warning rather
than raising.

## Obtaining / usage

Download a converted voice from
[`OpenVoiceOS/phoonnx-f5tts`](https://huggingface.co/OpenVoiceOS/phoonnx-f5tts)
and load it like any other phoonnx voice. F5-TTS is in-context, so every call
needs the reference clip **and its transcription**:

```python
from huggingface_hub import snapshot_download
from phoonnx.voice import TTSVoice, SynthesisConfig

voice_dir = snapshot_download("OpenVoiceOS/phoonnx-f5tts",
                              allow_patterns=["f5tts-v1-base/*"])
voice_dir = f"{voice_dir}/f5tts-v1-base"

voice = TTSVoice.load(
    f"{voice_dir}/model.onnx",              # the transformer graph
    config_path=f"{voice_dir}/config.json",
    engine_params={
        "preprocess_path": f"{voice_dir}/F5_Preprocess.onnx",
        "decode_path": f"{voice_dir}/F5_Decode.onnx",
    },
)

import wave
with wave.open("out.wav", "wb") as f:
    voice.synthesize_wav("A line the reference never spoke.", f,
        SynthesisConfig(
            speaker_reference="reference.wav",
            speaker_reference_text="transcription of the reference clip",
        ))
```

### OVOS plugin

```json
{
  "tts": {
    "module": "ovos-tts-plugin-phoonnx",
    "ovos-tts-plugin-phoonnx": {
      "voice": "f5tts/v1-base",
      "speaker_reference": "~/reference.wav",
      "speaker_reference_text": "transcription of the reference clip"
    }
  }
}
```

The catalog voices ship in `phoonnx/voice_index/f5tts.json`:

| voice id | model | lang |
|---|---|---|
| `f5tts/v1-base` | F5-TTS v1 base | multilingual |
| `habibi/ar-unified` | Habibi-TTS Unified (all dialects, recommended) | Arabic |
| `habibi/ar-msa` | Habibi-TTS Specialized MSA | Arabic (Modern Standard) |
| `habibi/ar-egy` | Habibi-TTS Specialized EGY | Arabic (Egyptian) |
| `habibi/ar-sau` | Habibi-TTS Specialized SAU | Arabic (Saudi) |
| `habibi/ar-uae` | Habibi-TTS Specialized UAE | Arabic (Emirati) |
| `habibi/ar-alg` | Habibi-TTS Specialized ALG | Arabic (Algerian) |
| `habibi/ar-irq` | Habibi-TTS Specialized IRQ | Arabic (Iraqi) |
| `habibi/ar-mar` | Habibi-TTS Specialized MAR | Arabic (Moroccan) |
| `namaa/ar-sa-v2` | NAMAA Saudi TTS V2 (Habibi fine-tune, non-commercial) | Arabic (Saudi) |
| `silma/v1` | SILMA TTS v1 (Apache-2.0, commercial OK) | Arabic (MSA/Fusha, full tashkeel support) |
| `silma/v1-en` | SILMA TTS v1 (same model, English listing) | English |

The model manager downloads the three ONNX graphs on first use: the
`aux_model_urls` mechanism resolves `preprocess_path` / `decode_path` to the
locally-cached files automatically.

### Licensing

The F5-TTS checkpoints are **CC-BY-NC-4.0** (non-commercial). Habibi-TTS
licensing is per model (see its
[model card](https://huggingface.co/SWivid/Habibi-TTS)): **Unified, SAU and UAE
are CC-BY-NC-SA-4.0** (restricted by the SADA and Mixat datasets), while
**ALG, EGY, IRQ, MAR and MSA are Apache-2.0**. **SILMA TTS v1 is Apache-2.0**
(commercial use allowed). **NAMAA-Saudi-TTS-V2 is CC-BY-NC-SA-4.0**
(non-commercial — inherited from its Habibi-TTS base). The ONNX conversions in
`OpenVoiceOS/phoonnx-f5tts` inherit those licenses.

## Variants

| Variant | Notes |
|---|---|
| **F5-TTS** | base model (DiT + ConvNeXt V2) |
| **F5-TTS v1** | improved training and inference |
| **Habibi-TTS** | Arabic dialectal fine-tune (Unified + Specialized) |
| **NAMAA-Saudi-TTS-V2** | Habibi-TTS fine-tune for Saudi Arabic (335M DiT, 24 kHz, char-level 2704 vocab) |
| **SILMA TTS v1** | bilingual Arabic/English 150M DiT (dim 768, depth 18) pretrained from scratch |
| **E2-TTS** | Flat-UNet variant (same flow-matching approach) |

Any model using the F5-TTS ONNX export layout (preprocess + transformer + decode)
is supported by this adapter.

## Gotchas / aliases

- **Detect aliases:** the config `engine` field may be `f5tts`, `f5-tts` or
  `habibi`.
- **Multi-graph:** the runtime needs `preprocess` + `transformer` +
  (`decode` or a Vocos `vocoder_path`); the `aux_model_urls` mechanism fetches
  the auxiliary graphs.
- **In-context cloning:** every call needs the reference clip and its
  transcription; audio alone is not enough.

## Upstream

| | |
|---|---|
| Repo | <https://github.com/SWivid/F5-TTS> |
| Paper | [arXiv:2410.06885](https://arxiv.org/abs/2410.06885) |
| ONNX export | <https://github.com/DakeQQ/F5-TTS-ONNX> |
| Ready-made ONNX voices | [`OpenVoiceOS/phoonnx-f5tts`](https://huggingface.co/OpenVoiceOS/phoonnx-f5tts) |

## Habibi quality audit (retroactive, 2026-08)

The 8 Habibi voices (`habibi/ar-unified` + 7 specialized dialects) shipped
before the WER-gate standard existed for phoonnx voice releases. This is a
retroactive quality pass to backfill that gate, run against the published
[`OpenVoiceOS/phoonnx-f5tts`](https://huggingface.co/OpenVoiceOS/phoonnx-f5tts)
mirror through the real `F5TTSAdapter` synthesis path (not a mock).

Note: the catalog has **8 Habibi voices, not 9** — 1 Unified + 7 specialized
dialects (ALG, EGY, IRQ, MAR, MSA, SAU, UAE). `namaa/ar-sa-v2` is a separate,
non-Habibi checkpoint (a NAMAA-Space fine-tune of Habibi-Unified) and is out
of scope for this pass.

### Method

- **ASR**: [`OpenVoiceOS/nemotron-3.5-asr-arabic-dialectal-v2-onnx`](https://huggingface.co/OpenVoiceOS/nemotron-3.5-asr-arabic-dialectal-v2-onnx)
  (int8, NeMo FastConformer-RNNT, dialectal-Arabic-tuned prompt), loaded through
  `onnx-asr`'s `nemo-conformer-rnnt` model type on the `feat/nemotron` branch
  (adds the raw-log-mel preprocessor this checkpoint needs). This is **not**
  Whisper — the OpenVoiceOS org also publishes `whisper-*-arabic-dialectal-v2-onnx`
  exports, but those were deliberately skipped in favor of the non-Whisper
  dialectal model, per the "never Whisper unless nothing else exists" rule.
- **Sentences**: 5 per voice — 4 shared Modern Standard Arabic sentences plus
  1 dialect-flavoured sentence per voice (sourced from the phrasing already
  used in the mirror's `samples/README.md`, e.g. Egyptian "النهاردة الجو جميل
  جداً...", Gulf/Najdi "اليوم الجو حلو مرة..."). **Only the 5th sentence per
  voice is genuinely dialectal text** — the first 4 are identical MSA prompts
  across all 8 voices, run so every voice has a apples-to-apples MSA baseline.
  This means the WER numbers below are dominated by MSA-on-MSA performance,
  not true dialectal ASR difficulty; only `idx=4` rows probe dialect-appropriate
  text through (in most cases) a dialectal ASR path.
- **Reference clip (in-context conditioning)**: F5-TTS/Habibi is in-context
  cloning — every synthesis call needs a reference clip **and its
  transcription**. No verified human reference-clip+transcript pair for any
  Habibi dialect was available locally. A single ~8s Arabic clip was pulled
  from the locally-cached `ArabicSpeech/ADI20` dataset (dialect-labeled
  Tunisian) and **self-transcribed with the same nemotron ASR model** to get
  its reference text, then reused as the one speaker reference for all 8
  voices — mirroring the original mirror's methodology of using a single
  Arabic reference clip (SILMA `ar.ref.24k.wav`) across voices, but with a
  self-transcribed (not verified/human-checked) transcript.
  **This is a real methodology weakness**: the ADI20 clip's content
  ("...باش نظر") visibly bleeds into a large fraction of the synthesized
  outputs as a spurious leading phrase (visible in the ASR hypotheses below,
  e.g. "باش نظر مرحبا بكم..."). This in-context leakage — a known F5TTS
  failure mode when the reference transcript is imperfect — inflates WER
  across the board and should be re-run with a clean, verified reference
  clip+transcript pair before these numbers are treated as final quality
  gates.
- **RTF**: wall-clock synth time / output audio duration, CPU inference
  (ser9, `onnxruntime`, shared box with other production TTS/STT/translate
  services running concurrently — these numbers are an upper bound, not a
  clean benchmark).
- **Human floor**: **not available**. No per-dialect Habibi reference dataset
  with verified transcripts was found locally or on the Habibi/NAMAA/SILMA
  HF pages (the upstream training datasets — SADA, Mixat, MGB-3/5, FLEURS,
  various `oddadmix`/dialect collections — are not bundled with matching
  transcripts in a form usable here). Dialectal Arabic ASR itself is hard and
  imperfect even on human speech, so **absolute WER/CER numbers below should
  not be read as voice-quality scores** — they should be read relative to
  each other and re-checked once a verified reference pair and/or a real
  per-dialect floor becomes available.

### Results — per-voice averages (5 sentences each)

| voice | avg WER | avg CER | avg duration (s) | avg RTF (CPU, shared box) |
|---|---|---|---|---|
| `habibi/ar-unified` | 0.33 | 0.18 | 7.1 | 11.2 |
| `habibi/ar-msa` | 0.33 | 0.21 | 6.4 | 11.4 |
| `habibi/ar-egy` | 0.34 | 0.18 | 6.6 | 12.6 |
| `habibi/ar-sau` | 0.25 | 0.11 | 6.6 | 13.1 |
| `habibi/ar-uae` | 0.38 | 0.27 | 6.4 | 12.6 |
| `habibi/ar-alg` | 0.34 | 0.16 | 6.7 | 16.6 |
| `habibi/ar-irq` | 0.35 | 0.21 | 6.6 | 12.8 |
| `habibi/ar-mar` | 0.41 | 0.22 | 6.6 | 13.3 |

No dialect stands out as catastrophically broken relative to the others —
`ar-sau` scores best (lowest WER/CER), `ar-mar` and `ar-uae` score worst, but
all 8 sit within roughly the same band once the reference-leakage artifact
above is taken into account. RTF is CPU-only and NFE=32 (F5-TTS default);
all voices are firmly non-real-time on CPU (RTF > 8), consistent with
flow-matching TTS in general — GPU or lower NFE would be needed for
interactive use.

### Results — per-sentence detail

`dialectal?` = `yes` only for `idx=4`, the one genuinely dialect-flavoured
sentence per voice; `idx=0..3` are the shared MSA prompts.

| voice | idx | dialectal? | text | ASR hypothesis | WER | CER | dur (s) | RTF |
|---|---|---|---|---|---|---|---|---|
| ar-unified | 0 | no (MSA) | التكنولوجيا الحديثة غيرت طريقة تواصل الناس حول العالم. | التكنولوجيا حديثة غيرت غيرت طريقة تواصل الناس حول العالم. | 0.25 | 0.13 | 7.2 | 8.1 |
| ar-unified | 1 | no (MSA) | مرحباً بكم في نظام الصوت المفتوح، يسعدنا انضمامكم إلينا. | مرحبا بكم في نظام الصوت المفتوح يسعدنا انضمامكم إلينا | 0.11 | 0.02 | 7.3 | 9.4 |
| ar-unified | 2 | no (MSA) | التعليم هو أساس التقدم في أي مجتمع يسعى نحو المستقبل. | ضرد التعليم هو أساس أساس التقدم في أي مجتمع يسعى نحو المستقبل. | 0.20 | 0.17 | 7.0 | 11.4 |
| ar-unified | 3 | no (MSA) | الماء هو سر الحياة على كوكب الأرض منذ آلاف السنين. | ضرد الماء هو سر الحياة على كو كوكب الأرض منذ آلاف سنين | 0.30 | 0.18 | 6.7 | 12.9 |
| ar-unified | 4 | yes | مرحباً بكم في نظام الصوت المفتوح، يسعدنا انضمامكم إلينا. | ضر مرحبا بيكم في النظام الصوت المفتوح يسعدنا وسعنا يسعدنا انضمامكم إلينا ال | 0.78 | 0.41 | 7.3 | 14.1 |
| ar-msa | 0 | no (MSA) | التكنولوجيا الحديثة غيرت طريقة تواصل الناس حول العالم. | بعشر نظرة التكنولوجيا الحديثة غيرت طريقة تواصل الناس حول العالم. | 0.25 | 0.19 | 6.8 | 8.7 |
| ar-msa | 1 | no (MSA) | مرحباً بكم في نظام الصوت المفتوح، يسعدنا انضمامكم إلينا. | باش باش نظم حبا بكم في في نظام الصوت المفتوح يسعدنا انضمامكم إلينا. | 0.67 | 0.28 | 6.8 | 10.5 |
| ar-msa | 2 | no (MSA) | التعليم هو أساس التقدم في أي مجتمع يسعى نحو المستقبل. | ظر التعليم هو أساس التقدم في أي مجتمع يسعى نحو المستقبل | 0.10 | 0.06 | 6.7 | 11.9 |
| ar-msa | 3 | no (MSA) | الماء هو سر الحياة على كوكب الأرض منذ آلاف السنين. | سنين باش نظر الماء هو سر الحياة على كوكب الأرض منذ آلاف سنين. | 0.40 | 0.31 | 6.3 | 12.4 |
| ar-msa | 4 | yes | الطقس اليوم مشمس وجميل في معظم مناطق البلاد. | باش نظرا الطقس اليوم مشمس وجميل في معظم مناطق البلاد. | 0.25 | 0.21 | 5.5 | 13.4 |
| ar-egy | 0 | no (MSA) | التكنولوجيا الحديثة غيرت طريقة تواصل الناس حول العالم. | ظلون الوجه الحديثة غير الطريقة تواصل الناس حول العالم. | 0.50 | 0.19 | 6.8 | 9.4 |
| ar-egy | 1 | no (MSA) | مرحباً بكم في نظام الصوت المفتوح، يسعدنا انضمامكم إلينا. | باش نظرة مرحبا بكم في نظام الصوت المفتوح يسعدنا انضمامكم إلينا | 0.33 | 0.19 | 6.9 | 13.2 |
| ar-egy | 2 | no (MSA) | التعليم هو أساس التقدم في أي مجتمع يسعى نحو المستقبل. | باش نظرها التعليم هو أساس التقدم في أي مجتمع يسعى نحو المستقب | 0.30 | 0.21 | 6.7 | 13.2 |
| ar-egy | 3 | no (MSA) | الماء هو سر الحياة على كوكب الأرض منذ آلاف السنين. | زر الماء سر الحياة على كوكب الأرض منذ آلاف سنين. | 0.30 | 0.16 | 6.3 | 13.4 |
| ar-egy | 4 | yes | النهاردة الجو جميل جداً في القاهرة والناس مبسوطة. | باش نظر النهاردة الجو جميل جدا في القاهرة والناس مبسوطة. | 0.25 | 0.17 | 6.1 | 13.5 |
| ar-sau | 0 | no (MSA) | التكنولوجيا الحديثة غيرت طريقة تواصل الناس حول العالم. | ضرار التكنولوجيا الحديثة غير الطريقة تواصل الناس حول العالم. | 0.38 | 0.15 | 6.8 | 11.0 |
| ar-sau | 1 | no (MSA) | مرحباً بكم في نظام الصوت المفتوح، يسعدنا انضمامكم إلينا. | ظر مرحبا بابكم في نظام الصوت المفتوح يسعدنا انضمامكم إلنا | 0.44 | 0.13 | 6.9 | 14.0 |
| ar-sau | 2 | no (MSA) | التعليم هو أساس التقدم في أي مجتمع يسعى نحو المستقبل. | التعليم هو أساس التقدم في أي مجتمع يسعى نحو المستقبل | 0.00 | 0.00 | 6.7 | 13.3 |
| ar-sau | 3 | no (MSA) | الماء هو سر الحياة على كوكب الأرض منذ آلاف السنين. | هو سر الحياة على كوكب الأرض منذ آف سنين. | 0.30 | 0.20 | 6.3 | 13.7 |
| ar-sau | 4 | yes | اليوم الجو حلو مرة في الرياض والناس طالعين يتمشون. | ضرا اليوم الجو حلو مرة في الرياض والناس طالعين يتمشون. | 0.11 | 0.08 | 6.3 | 13.7 |
| ar-uae | 0 | no (MSA) | التكنولوجيا الحديثة غيرت طريقة تواصل الناس حول العالم. | نظرك الحديثة غيرت هي الطريقة طريقة تواصل الناس حول العالم. | 0.38 | 0.40 | 6.8 | 9.6 |
| ar-uae | 1 | no (MSA) | مرحباً بكم في نظام الصوت المفتوح، يسعدنا انضمامكم إلينا. | باش نظر رحبان بكم في نظام الصوت المفتوح يسعدنا انضمامكم إلينا. | 0.44 | 0.19 | 6.9 | 12.1 |
| ar-uae | 2 | no (MSA) | التعليم هو أساس التقدم في أي مجتمع يسعى نحو المستقبل. | باش نش نظر التعليم هو أساس التقدم في أي مجتمع يسعى نحو المستقبل المستقبل | 0.40 | 0.38 | 6.7 | 12.9 |
| ar-uae | 3 | no (MSA) | الماء هو سر الحياة على كوكب الأرض منذ آلاف السنين. | الماء هو سر سر الحياة على كوكب الأرض منذ آلاف السنين. | 0.10 | 0.06 | 6.3 | 13.3 |
| ar-uae | 4 | yes | اليوم الجو وايد حلو في دبي والناس كلهم برع. | باش نظرة اليوم الجوايد حلو في في دبي والناس كلهم برع | 0.56 | 0.33 | 5.4 | 15.0 |
| ar-alg | 0 | no (MSA) | التكنولوجيا الحديثة غيرت طريقة تواصل الناس حول العالم. | ضرار شكل التكنولوجيا الحديثة غير الطريقة تواصل الناس حول العالم. | 0.50 | 0.23 | 6.8 | 11.4 |
| ar-alg | 1 | no (MSA) | مرحباً بكم في نظام الصوت المفتوح، يسعدنا انضمامكم إلينا. | باش نظر مرحبا بيكم في نظام الصوت المفتوح يسعدنا انضمامكم إلينا | 0.44 | 0.19 | 6.9 | 14.9 |
| ar-alg | 2 | no (MSA) | التعليم هو أساس التقدم في أي مجتمع يسعى نحو المستقبل. | ظر التعليم هو أساس التقدم في أي مجتمع يسعى نحو المستقب | 0.20 | 0.08 | 6.7 | 14.8 |
| ar-alg | 3 | no (MSA) | الماء هو سر الحياة على كوكب الأرض منذ آلاف السنين. | الماء هو سر الحياة على على كوكب الأرض منذ آلاف السنين. | 0.10 | 0.08 | 6.3 | 20.2 |
| ar-alg | 4 | yes | اليوم الجو شباب بزاف في الجزائر والناس خارجين يتساراو. | باش نظر اليوم الجو شباب زاف في الجزائر والناس خارجين يتساروا | 0.44 | 0.21 | 6.8 | 21.8 |
| ar-irq | 0 | no (MSA) | التكنولوجيا الحديثة غيرت طريقة تواصل الناس حول العالم. | نظرة تكليفة غير ت غير طريقة تواصل الناس حول العالم. | 0.62 | 0.36 | 6.8 | 11.3 |
| ar-irq | 1 | no (MSA) | مرحباً بكم في نظام الصوت المفتوح، يسعدنا انضمامكم إلينا. | باش نظر مرة حبان بكم في نظام الصوت المفتوح يسعدنا انضمامكم إلينا | 0.56 | 0.22 | 6.9 | 13.0 |
| ar-irq | 2 | no (MSA) | التعليم هو أساس التقدم في أي مجتمع يسعى نحو المستقبل. | نظرة التعليم هو أساس التقدم في أي مجتمع يسعى نحو المستقبل | 0.10 | 0.10 | 6.7 | 13.1 |
| ar-irq | 3 | no (MSA) | الماء هو سر الحياة على كوكب الأرض منذ آلاف السنين. | نظر الماء هو سر الحياة الحياة على كوكب الأرض منذ آلاف السنين. | 0.20 | 0.22 | 6.3 | 13.4 |
| ar-irq | 4 | yes | اليوم الجو حلو هواية ببغداد والناس طالعين يتمشون. | باش نظر اليوم الجو حلو هواية ببغداد والناس طالعين يتمشون | 0.25 | 0.17 | 6.1 | 13.4 |
| ar-mar | 0 | no (MSA) | التكنولوجيا الحديثة غيرت طريقة تواصل الناس حول العالم. | ضرار التكنولوجيا الحديثة غير طريقة تواصل الناس حول العالم. | 0.25 | 0.11 | 6.8 | 11.1 |
| ar-mar | 1 | no (MSA) | مرحباً بكم في نظام الصوت المفتوح، يسعدنا انضمامكم إلينا. | باش نظر مرحبا بيكم في نظام الصوت المفتوح يساعدنا انضمامكم إلينا | 0.56 | 0.20 | 6.9 | 13.3 |
| ar-mar | 2 | no (MSA) | التعليم هو أساس التقدم في أي مجتمع يسعى نحو المستقبل. | ضر التعليم هو أساس التقدم في في أي مجتمع يسعى نحو المستقب | 0.30 | 0.13 | 6.7 | 13.7 |
| ar-mar | 3 | no (MSA) | الماء هو سر الحياة على كوكب الأرض منذ آلاف السنين. | درال ماء باش النظرالماء هو سر الحياة على كوكب الأرض منذ آلاف السنين. | 0.40 | 0.37 | 6.3 | 14.6 |
| ar-mar | 4 | yes | اليوم الجو زوين بزاف في الرباط والناس خارجين يتسارا. | باش ضرا اليوم الجوين زوين بزاف الرباط والناس خارجين يتصارى. | 0.56 | 0.29 | 6.5 | 13.7 |

### Interpretation

- **No gate failure identified.** All 8 voices land in a broadly similar
  WER/CER band (0.25-0.41 avg WER, 0.11-0.27 avg CER); none is a dramatic
  outlier suggesting a broken checkpoint or export. This is consistent with
  (not proof of) acceptable quality, given the reference-leakage caveat above.
- **The dominant error source in this pass is methodology, not the voices**:
  the self-transcribed, non-verified reference clip visibly leaks into a
  majority of outputs as a spurious leading phrase. A follow-up pass with a
  clean, hand-verified reference clip (ideally one per dialect, spoken by a
  native speaker) would very likely lower WER/CER across the board and is
  the highest-value next step before treating any of these numbers as a
  hard release gate.
- **No trustworthy per-dialect human floor exists yet** for any of the 8
  Habibi dialects in this environment — this pass can only compare voices to
  each other, not to a ceiling. Building one (e.g. sampling a handful of
  verified-transcript clips per dialect from SADA/Mixat/MGB-3/MGB-5/FLEURS)
  is tracked as follow-up work, not done here.
- Dialectal Arabic ASR is inherently harder than MSA ASR (dialect/orthography
  mismatch, code-switching, ASR training-data scarcity per dialect), so even
  a "clean" re-run should expect WER/CER to run higher on `idx=4` dialectal
  rows than on the shared MSA rows — that gap is expected, not necessarily a
  TTS quality problem.
