# F5-TTS Engine

<!-- agent-hint
covers: how to load an F5-TTS, Habibi or SILMA voice, clone with a reference clip and its transcription, and set the dialect
source: phoonnx/engines/f5tts.py, phoonnx/voice_index/f5tts.json
generated: no
verify: pytest tests/test_f5tts.py tests/test_card_example_f5tts.py; the voice list: none, checked by reading the keys of phoonnx/voice_index/f5tts.json
see: ../../cloning.md, zipvoice.md, ../../ovos_plugin.md
-->

This page explains how to load an F5-TTS voice and clone from a reference clip and its transcription, for integrators who use phoonnx.

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
sway-sampled time schedule. The export fixes the number of steps, so `nfe` is not
a speed setting. A graph that stores `nfe_step` in its metadata always runs that
many steps; the adapter logs a warning and ignores any other `nfe`. For a graph
without that key, `nfe` (default **32**) must equal the export's step count.
`cfg_strength` (default 2.0) controls classifier-free guidance,
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
| `nfe` | 32 | ODE steps. The export fixes this value: a graph with `nfe_step` metadata ignores any other value, and for other graphs it must match the export |
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

Download a converted voice from the
[F5-TTS collection](https://huggingface.co/collections/Phoonnx/f5-tts-6af8ea7f407234f6f1f08b30)
and load it like any other phoonnx voice. Each voice lives in its own repo (see
the table below). F5-TTS is in-context, so every call needs the reference clip
**and its transcription**:

```python
from huggingface_hub import snapshot_download
from phoonnx.voice import TTSVoice, SynthesisConfig

voice_dir = snapshot_download("Phoonnx/f5tts-multi-base-v1-graphemes")

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

| voice id | Hub repo |
|---|---|
| `f5tts/v1-base` | `Phoonnx/f5tts-multi-base-v1-graphemes` |
| `habibi/ar-unified` | `Phoonnx/f5tts-ar-habibi_unified-graphemes` |
| `habibi/ar-alg` | `Phoonnx/f5tts-ar_dz-habibi_alg-graphemes` |
| `habibi/ar-egy` | `Phoonnx/f5tts-ar_eg-habibi_egy-graphemes` |
| `habibi/ar-irq` | `Phoonnx/f5tts-ar_iq-habibi_irq-graphemes` |
| `habibi/ar-mar` | `Phoonnx/f5tts-ar_ma-habibi_mar-graphemes` |
| `habibi/ar-msa` | `Phoonnx/f5tts-ar-habibi_msa-graphemes` |
| `habibi/ar-sau` | `Phoonnx/f5tts-ar_sa-habibi_sau-graphemes` |
| `habibi/ar-uae` | `Phoonnx/f5tts-ar_ae-habibi_uae-graphemes` |
| `namaa/ar-sa-v2` | `Phoonnx/f5tts-ar_sa-namaa_space-v2-graphemes` |
| `silma/v1` | `Phoonnx/f5tts-multi-silma_ai-v1-graphemes` |
| `silma/v1-en` | `Phoonnx/f5tts-multi-silma_ai-v1-graphemes` |
| `openbible/tw-asante` | `Phoonnx/f5tts-tw-asante_openbible-graphemes` |
| `openbible/ckb` | `Phoonnx/f5tts-ckb-openbible-graphemes` |
| `f5tts/sema-afro` | `Phoonnx/f5tts-multi-sema_afro-unicode` |
| `f5tts/spring` | `Phoonnx/f5tts-multi-spring-unicode` |

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
| `openbible/tw-asante` | F5-TTS OpenBible Twi (Asante), name-bound export | Twi (Asante) |
| `openbible/ckb` | F5-TTS OpenBible Central Kurdish | Central Kurdish |
| `f5tts/sema-afro` | Sema TTS Afro (from `travis-moore/sema-tts-afro`) | multilingual: English and 22 African languages |
| `f5tts/spring` | SPRING F5 (from `SPRINGLab/SPRING_F5`) | multilingual: English and 23 Indian languages |

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
(non-commercial — inherited from its Habibi-TTS base). **F5-TTS OpenBible Twi
(Asante) and OpenBible Central Kurdish are CC-BY-SA-4.0** (share-alike). **Sema TTS Afro
and SPRING F5 are Apache-2.0**. The ONNX conversions in the F5-TTS collection inherit those
licenses.

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
| Ready-made ONNX voices | [F5-TTS collection](https://huggingface.co/collections/Phoonnx/f5-tts-6af8ea7f407234f6f1f08b30) |

## Measured intelligibility of the Habibi voices

An ASR round trip measured the eight Habibi voices, with the dialectal-Arabic model
[`OpenVoiceOS/nemotron-3.5-asr-arabic-dialectal-v2-onnx`](https://huggingface.co/OpenVoiceOS/nemotron-3.5-asr-arabic-dialectal-v2-onnx).
Each voice spoke five sentences: four shared Modern Standard Arabic sentences and one
sentence in its own dialect.

| voice | avg WER | avg CER | avg duration (s) | avg RTF (CPU) |
|---|---|---|---|---|
| `habibi/ar-unified` | 0.33 | 0.18 | 7.1 | 11.2 |
| `habibi/ar-msa` | 0.33 | 0.21 | 6.4 | 11.4 |
| `habibi/ar-egy` | 0.34 | 0.18 | 6.6 | 12.6 |
| `habibi/ar-sau` | 0.25 | 0.11 | 6.6 | 13.1 |
| `habibi/ar-uae` | 0.38 | 0.27 | 6.4 | 12.6 |
| `habibi/ar-alg` | 0.34 | 0.16 | 6.7 | 16.6 |
| `habibi/ar-irq` | 0.35 | 0.21 | 6.6 | 12.8 |
| `habibi/ar-mar` | 0.41 | 0.22 | 6.6 | 13.3 |

Read these numbers as a comparison between the voices, not as quality scores:

- All voices shared one reference clip, a Tunisian clip whose transcription came from the
  same ASR model. Words of that clip leak into many outputs as a spurious first phrase, and
  this raises every WER.
- Four of the five sentences are Modern Standard Arabic, so the averages mostly measure
  Modern Standard Arabic.
- No human floor exists for these dialects, so the numbers have no ceiling to compare with.
- The RTF comes from a shared CPU host at the default 32 steps. All voices are slower than
  real time on CPU.

---

[Docs index](../../index.md) · Previous: [ArkTTS](arktts.md) · Next: [MOSS-TTS-Nano](mosstts.md)
