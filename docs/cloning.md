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

## Fine-tuning your own cloning voice (YourTTS)

The `yourtts` training engine (`phoonnx_train/engines/yourtts.py`) fine-tunes a
zero-shot cloning voice on your own multi-speaker data. It reuses the VITS training
pipeline end to end (`phoonnx_train/vits/`) — the only difference from plain VITS is
*what conditions the decoder*: instead of a learned per-speaker-id embedding table,
each utterance is conditioned on an external 512-d **d-vector** computed with the same
Coqui ResNet speaker encoder used at inference (`phoonnx.engines.speaker_encoders
.coqui_resnet`), so train and inference embeddings match exactly. An optional additive
language embedding (`--n-langs`) supports multilingual training.

### 1. Preprocess

Preprocessing is the same LJSpeech-style pipeline as [training.md](training.md#1-preprocessing),
with speaker names read from `metadata.csv`'s second column (do **not** pass
`--single-speaker`). A separate, per-utterance d-vector is required — compute it with
`YourttsTrainingEngine.extra_preprocess`, pointing at a Coqui ResNet ONNX speaker
encoder (the same file you plan to bundle with the resulting voice):

```python
from pathlib import Path
from phoonnx_train.engines import get_engine

engine = get_engine("yourtts")
for utt in utterances:
    extra = engine.extra_preprocess(
        utterance_audio_path=utt.audio_path,
        cache_dir=cache_dir,
        sample_rate=22050,
        speaker_encoder_path="speaker_encoder.onnx",
        language_id=0,               # bump per corpus for multilingual training
    )
    utt.extra = extra                 # -> {"d_vector_path": ..., "language_id": ...}
```

d-vectors are cached as `<cache_dir>/dvec/<hash>.pt`, next to the mel/audio cache, so
re-running preprocessing is a no-op cache hit. The resulting `dataset.jsonl` lines each
carry `d_vector_path` (and `language_id` for multilingual runs) alongside the usual
`phoneme_ids` / `audio_norm_path` / `audio_spec_path` fields — the shared
`phoonnx_train.vits.dataset.PhoonnxDataset` loads them transparently (they are optional
fields; a plain-VITS `dataset.jsonl` without them is unaffected).

### 2. Train

```bash
python phoonnx_train/train.py \
  --dataset-dir /data/preprocessed \
  --engine yourtts \
  --quality medium \
  --accelerator gpu --devices 1 \
  --batch-size 16 --max-epochs 1000
```

`quality_presets` mirror the plain VITS tiers (`x-low` / `medium` / `high` —
`hidden_channels` / `inter_channels` / `filter_channels` / `n_heads` / `n_layers`); pick
the tier by the same size/quality trade-off as any other phoonnx_train run. Internally,
`YourttsTrainingEngine.create_model` builds the same `phoonnx_train.vits.lightning
.VitsModel` used by the `vits` engine, with `external_speaker_embedding=True` and
`num_speakers=1` (speaker identity comes from the d-vector, not an id table), and
`n_langs` set from your preprocessing run if you're training multilingually.

### 3. Export

```bash
python phoonnx_train/export_onnx.py \
  /path/to/checkpoint.ckpt \
  --config /path/to/config.json \
  --engine yourtts \
  --output-dir /path/to/output/
```

The export mirrors the plain VITS ONNX graph (`input`, `input_lengths`, `scales`) but
adds `d_vector` and `langid` inputs, matching what `phoonnx.engines.yourtts
.YourTTSAdapter` feeds at inference. The voice config written alongside the `.onnx`
file sets `"engine": "yourtts"` and packages a **default speaker** in `engine_params
["d_vector"]` — the mean embedding across the training set — so the voice synthesizes
with no reference clip; a reference clip or an explicit `d_vector` at synthesis time
still overrides it, exactly like any other bundled cloning voice (see "Two cloning
paradigms" above).

### Downstream OVOS plugin config

No plugin-side changes are needed beyond the usual cloning setup — a fine-tuned
YourTTS voice is used exactly like the stock one:

```json
{
  "module": "ovos-tts-plugin-phoonnx",
  "ovos-tts-plugin-phoonnx": {
    "model": "/path/to/output/checkpoint.ckpt.onnx",
    "ref_wav": "/home/user/me.wav"
  }
}
```

Omit `ref_wav` to always use the voice's packaged default speaker.

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
