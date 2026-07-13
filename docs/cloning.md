# Voice Cloning

phoonnx supports **zero-shot voice cloning** — synthesizing speech in a target
speaker's voice from a short reference clip, without any per-speaker training.

```python
from phoonnx.voice import TTSVoice, SynthesisConfig

voice = TTSVoice.load("model.onnx", "model.json")
audio = voice.synthesize(
    "This sentence is spoken in the cloned voice.",
    SynthesisConfig(speaker_reference="reference.wav"),
)
```

## Two cloning paradigms

Cloning engines fall into two families, which differ in what the reference clip is
turned into:

| Paradigm | How it works | Needs the reference's text? | Language-agnostic? | Engines |
|---|---|---|---|---|
| **d-vector** | a **speaker encoder** maps the reference waveform to a fixed embedding that conditions synthesis | No | Yes | YourTTS, StyleTTS2, [Chatterbox](chatterbox.md) |
| **in-context** | the reference **audio + its transcription** are part of the model input; the model continues that voice | **Yes** | Per-phoneme (espeak/pinyin) | ZipVoice |

A cloning voice can still bundle a **default speaker**, so it works with *or* without a
reference. When a reference is given it **overrides** the default
(`reference > bundled`).

## The cloning API

All cloning is driven by three `SynthesisConfig` fields:

| Field | Type | Used by | Meaning |
|---|---|---|---|
| `speaker_reference` | `str` path or `(audio, sample_rate)` | all cloning engines | the reference clip |
| `speaker_reference_text` | `str` | in-context only | the reference's transcription |
| `speaker_reference_lang` | `str` (e.g. `"pt"`) | in-context only | the language of the transcription (defaults to the voice's `lang_code`) |

d-vector engines ignore the two `_text`/`_lang` fields; in-context engines require
`speaker_reference_text`.

## d-vector engines (YourTTS, StyleTTS2)

No transcription, any language — the speaker encoder summarizes timbre directly:

```python
audio = voice.synthesize(
    "Speak this in the cloned voice.",
    SynthesisConfig(speaker_reference="any_language_clip.wav"),
)
```

The encoders live in the **speaker-encoder registry**
(`phoonnx.engines.speaker_encoders`, mirroring the vocoder registry):

| Type | Engine | Reference → |
|---|---|---|
| `coqui_resnet` | YourTTS | 512-d speaker d-vector |
| `styletts2_style` | StyleTTS2 | 256-d style (prosody + acoustic) |

A cloning voice names its encoder in `engine_params` (`speaker_encoder_url` /
`speaker_encoder_type`); the model manager downloads it to
`engine_params["speaker_encoder_path"]` and the adapter loads it in `configure()`.

## Autoregressive engine (Chatterbox)

[Chatterbox](chatterbox.md) is a d-vector engine too (reference clip, no transcription), but
adds an **`exaggeration`** control (0.0–1.0) for expressiveness:

```python
voice.synthesize("...", SynthesisConfig(speaker_reference="ref.wav", exaggeration=0.6))
```

## In-context engine (ZipVoice)

[ZipVoice](zipvoice.md) is a flow-matching model that **infills** the target after the
reference, so it needs both the reference audio **and its transcription** — the text
aligns the audio to phonemes:

```python
audio = voice.synthesize(
    "A sentence the reference never spoke.",
    SynthesisConfig(
        speaker_reference="alice.wav",
        speaker_reference_text="hello, this is the reference clip",
    ),
)
```

### Cross-lingual cloning

Because the reference is aligned via its transcription, the reference may be in a
**different language** than the target — set `speaker_reference_lang` so the
transcription is phonemized in *its* language:

```python
# a Portuguese reference clip speaking English
SynthesisConfig(
    speaker_reference="miro_pt.wav",
    speaker_reference_text="olá, tudo bem com você",
    speaker_reference_lang="pt",
)
```

Quality depends on phoneme coverage (ZipVoice is trained mostly on English + Chinese).
If you want no-text, fully language-agnostic cloning, prefer a **d-vector** engine.

## OVOS plugin

The OpenVoiceOS plugin reads cloning settings from `mycroft.conf`:

```json
{
  "module": "ovos-tts-plugin-phoonnx",
  "ovos-tts-plugin-phoonnx": {
    "ref_wav": "/home/user/me.wav",
    "ref_text": "olá tudo bem com você",
    "ref_lang": "pt"
  }
}
```

`ref_text` / `ref_lang` are only needed for in-context engines; d-vector voices use
`ref_wav` alone.

## Adding a cloning model

Conversion/export scripts live under `scripts/conversion/`:

| Engine | Script | Produces |
|---|---|---|
| YourTTS | `scripts/conversion/yourtts/export_speaker_encoder.py` | the `coqui_resnet` encoder ONNX |
| StyleTTS2 | `scripts/conversion/styletts2/export_style_encoder.py` | the `styletts2_style` encoder ONNX |
| ZipVoice | `scripts/conversion/zipvoice/export_mel.py` + `infer_zipvoice_onnx.py` | the no-torch mel ONNX + a reference runner |

See [`engines.md`](engines.md) for how the adapters are structured (d-vector engines are
single-graph; ZipVoice is the first **iterative** engine — a flow-matching ODE loop
over multiple ONNX graphs via the overridable `BaseOnnxAdapter.synthesize()`).
