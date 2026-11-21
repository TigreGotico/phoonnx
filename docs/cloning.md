# Voice Cloning

<!-- agent-hint
covers: how to clone a voice from a reference clip in Python or the OVOS plugin, and how to fine-tune a YourTTS voice
source: phoonnx/config.py:SynthesisConfig, phoonnx/reference_audio.py, phoonnx/engines/speaker_encoders/, phoonnx/opm.py, phoonnx_train/engines/yourtts.py, phoonnx_train/preprocess.py
generated: no
verify: pytest tests/test_per_request_cloning.py
see: voice_conversion.md, training/engines/chatterbox.md, ovos_plugin.md
-->

This page shows how to clone a voice from a reference clip, for developers who use phoonnx from Python or through the OVOS plugin.

Zero-shot cloning speaks in the voice of a short reference clip, with no training for that
speaker. The page covers the reference-audio API, the cloning families, and how to
fine-tune a cloning voice.

```python
from phoonnx.voice import TTSVoice, SynthesisConfig

voice = TTSVoice.load("model.onnx", "model.json")
for chunk in voice.synthesize(
    "This sentence is spoken in the cloned voice.",
    SynthesisConfig(speaker_reference="reference.wav"),
):
    ...  # chunk.audio_float_array — see Usage for the full AudioChunk API
```

Cloning happens **during** generation. If your voice is not a cloning engine, see
[Voice conversion](voice_conversion.md). Voice conversion changes the waveform **after**
generation and works with every engine.

## Two cloning paradigms

Cloning engines fall into two families, which differ in what the reference clip is
turned into:

| Paradigm | How it works | Needs the reference's text? | Language-agnostic? | Engines |
|---|---|---|---|---|
| **d-vector** | a **speaker encoder** maps the reference waveform to a fixed embedding that conditions synthesis | No | Yes | YourTTS, StyleTTS2, [Chatterbox](training/engines/chatterbox.md), [Spark-TTS](training/engines/sparktts.md) |
| **in-context** | the reference **audio + its transcription** are part of the model input; the model continues that voice | **Yes** | Per-phoneme (espeak/pinyin) | ZipVoice, [Spark-TTS](training/engines/sparktts.md), [OmniVoice](training/engines/omnivoice.md) |
| **in-context, pre-encoded** | the reference is **codec-encoded ahead of time** and shipped with the voice; the model continues those audio tokens | **Yes** (bundled) | Per-phoneme (espeak) | NeuTTS / Akiti-TTS, [OuteTTS](training/engines/outetts.md), Llasa |
| **in-context, audio only** | the reference is **codec-encoded** into audio tokens and prepended to the model's own token stream — no encoder, no transcription | No | Yes | MOSS-TTS-Nano |

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

### Reference audio formats

When `soundfile` is installed, phoonnx reads every reference with it, WAV included, so
FLAC, OGG and MP3 references work too. `soundfile` comes with the `phoonnx[cloning]`
extra (it pulls in `soundfile` + `scipy`). Without it, phoonnx reads the reference with
the standard-library `wave` module. That path decodes 8, 16, 24 and 32-bit PCM WAV by
the sample width in the file header. Any other sample width, a float WAV, and a FLAC, OGG
or MP3 file raise `ValueError`. The message names the file and the supported formats.
Install the extra for any reference that is not a PCM WAV of those widths:

```bash
pip install --pre "phoonnx[cloning]"
```

## d-vector engines (YourTTS, StyleTTS2)

No transcription, any language — the speaker encoder summarizes timbre directly:

```python
for chunk in voice.synthesize(
    "Speak this in the cloned voice.",
    SynthesisConfig(speaker_reference="any_language_clip.wav"),
):
    ...
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

The `bsc/es-styletts2` and `bsc/ca-styletts2` voices have no speaker of their
own, so a reference clip is required. Their **named speakers** —
`bsc/ca-<name>` and `bsc/es-cml<id>`, see [BSC multispeaker](bsc_multispeaker.md) —
are the same two checkpoints with a style blob attached, so they speak without a
reference clip but still clone when you pass one. The Galician
`proxectonos/celtia-styletts2` and `proxectonos/brais-styletts2` voices are
single-speaker: they ship a default style and use the reference clip only when
you pass one (see [Galician](galician.md)).

## Autoregressive engine (Chatterbox)

[Chatterbox](training/engines/chatterbox.md) is a d-vector engine too (reference clip, no transcription), but
adds an **`exaggeration`** control (0.0–1.0) for expressiveness:

```python
chunks = list(voice.synthesize("...", SynthesisConfig(speaker_reference="ref.wav", exaggeration=0.6)))
```

The speech tokens are sampled. `extra_params={"seed": 0}` fixes the speech tokens. The decoder adds its own random noise, so two runs with one seed still give slightly different audio.

## Preset engine (NeuTTS / Akiti-TTS)

[NeuTTS](engines.md) clones in-context, but the reference is **encoded once, offline**, and
shipped with the voice rather than supplied per call. A preset is a reference
transcription plus the NeuCodec tokens of its recording; the model is given both and
continues the voice. Each bundled voice pins one preset, so no cloning arguments are
needed:

```python
chunks = list(voice.synthesize("Meda wo ase paa."))
```

Switch presets per call through `extra_params`:

```python
SynthesisConfig(extra_params={"voice": "kofi", "temperature": 0.4, "seed": 0})
```

`speaker_reference` is **not** supported here: Neuphonic publishes NeuCodec's decoder as
ONNX but not its encoder, so a fresh clip cannot be turned into reference tokens at
runtime. Passing one raises rather than silently ignoring it.

Sampling is tunable through `extra_params` — `temperature`, `top_p`, `top_k`,
`repetition_penalty` and `max_new_tokens` (the autoregressive loop runs at 50 codec
tokens per second of audio). `seed` makes a run reproducible.

## Preset engine (Llasa)

[Llasa](engines.md) is the other pre-encoded preset engine, and it reaches that design
from the opposite direction: Llasa needs **no** reference at all. Prompted with text
alone it invents a speaker, which means two calls never sound like the same person. A
preset is what makes a voice repeatable — the transcript of an utterance the model
generated, plus the XCodec2 tokens it emitted for it, replayed as the in-context prefix.

Every bundled preset is therefore machine-generated and disclosed as such
(`"synthetic": true` in `voices.json`); none is a recording of a person.

```python
chunks = list(voice.synthesize("Dealing with family secrets is never easy."))
SynthesisConfig(extra_params={"voice": "zh_male_a", "temperature": 0.9, "seed": 0})
```

`speaker_reference` is **not** supported: tokenising a fresh clip needs XCodec2's encoder
together with the w2v-BERT filterbank front end, which the bundle does not ship. Passing
one raises rather than silently ignoring it.

Sampling is tunable through `extra_params` — `temperature`, `top_p` and
`max_new_tokens` (50 codec tokens per second of audio). `seed` makes a run reproducible.
Llasa's sampler follows the **HuggingFace** warper order (temperature, then top-p),
because that is the stack upstream drives the checkpoint through; NeuTTS uses llama.cpp's
opposite order for the same reason.

## Codec-LM engine (MOSS-TTS-Nano)

MOSS-TTS-Nano clones zero-shot from a reference clip alone — no transcription and no
speaker encoder. The clip is encoded to RVQ-16 audio tokens by the model's own audio
tokenizer and put before the prompt, so cloning works across languages:

```python
chunks = list(voice.synthesize("...", SynthesisConfig(speaker_reference="alice.wav")))
```

Sampling is tunable through `extra_params`. By default the engine uses the graph that
has upstream's sampling constants compiled in (fastest); changing any of
`audio_temperature` / `audio_top_p` / `audio_top_k` / `audio_repetition_penalty`
switches to host-side sampling, and `sample_mode="greedy"` is deterministic:

```python
SynthesisConfig(speaker_reference="alice.wav",
                extra_params={"audio_temperature": 0.6, "max_new_frames": 250, "seed": 0})
```

`max_new_frames` caps the autoregressive loop at 12.5 frames per second of audio
(the default 375 is 30 s per model call).

## In-context engine (ZipVoice)

[ZipVoice](training/engines/zipvoice.md) is a flow-matching model that **infills** the target after the
reference, so it needs both the reference audio **and its transcription** — the text
aligns the audio to phonemes:

```python
for chunk in voice.synthesize(
    "A sentence the reference never spoke.",
    SynthesisConfig(
        speaker_reference="alice.wav",
        speaker_reference_text="hello, this is the reference clip",
    ),
):
    ...
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

The config keys map onto the `SynthesisConfig` fields: `ref_wav` (alias `clone_voice`,
or the explicit `speaker_reference`) sets the reference clip, `ref_text` sets
`speaker_reference_text`, and `ref_lang` sets `speaker_reference_lang`. `ref_text` /
`ref_lang` are only needed for in-context engines; d-vector voices use `ref_wav` alone.

## Fine-tuning your own cloning voice (YourTTS)

The `yourtts` training engine (`phoonnx_train/engines/yourtts.py`) fine-tunes a
zero-shot cloning voice on your own multi-speaker data. It uses the VITS training
pipeline (`phoonnx_train/vits/`). The difference from plain VITS is what conditions the
decoder. Plain VITS learns an embedding table with one row per speaker id. YourTTS
conditions each utterance on a 512-d **d-vector** from the Coqui ResNet speaker encoder
that inference also uses (`phoonnx.engines.speaker_encoders.coqui_resnet`), so the
training and inference embeddings match.

An optional additive language embedding supports multilingual training. The engine reads
its size from an `n_langs` key in the `config.json` of the preprocessed dataset.
`preprocess.py` does not write that key, so add it by hand for a multilingual run. The
engine also declares a `--n-langs` option, but `train.py` does not add it to its
command line.

### 1. Preprocess

Preprocessing is the same LJSpeech-style pipeline as [preprocessing](training/preprocess.md),
with speaker names read from `metadata.csv`'s second column (do **not** pass
`--single-speaker`). A per-utterance d-vector is required — `preprocess.py`
computes it when told which engine it is preparing data for, pointing at a
Coqui ResNet ONNX speaker encoder (the same file you plan to bundle with
the resulting voice):

```bash
python -m phoonnx_train.preprocess \
  --language en-US --input-dir data --output-dir preprocessed \
  --engine yourtts \
  --speaker-encoder-path speaker_encoder.onnx \
  --language-id 0     # bump per corpus for multilingual training
```

d-vectors are cached as `<cache_dir>/dvec/<hash>.pt`, next to the mel/audio cache, so
re-running preprocessing is a no-op cache hit. The resulting `dataset.jsonl` lines each
carry `d_vector_path` (and `language_id` for multilingual runs) alongside the usual
`phoneme_ids` / `audio_norm_path` / `audio_spec_path` fields — the shared
`phoonnx_train.vits.dataset.PhoonnxDataset` loads them (they are optional
fields; a plain-VITS `dataset.jsonl` without them is unaffected).

### Speaker-consistency loss

Point `speaker_encoder_checkpoint` (engine extra / `VitsModel` hparam) at the
released H/ASP torch checkpoint (`model_se.pth.tar` — the file the ONNX
encoder was converted from) to enable the YourTTS speaker-consistency loss:
the frozen encoder embeds the real and generated audio segments and
`c_scl` (default 9.0) times the negative cosine similarity is added to the
generator loss. Without a checkpoint the loss is off and training reduces
to d-vector-conditioned VITS.

### 2. Train

```bash
python -m phoonnx_train.train \
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
python -m phoonnx_train.export_onnx \
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
still overrides it, as for any other bundled cloning voice (see "Two cloning
paradigms" above).

### Use the fine-tuned voice

Load the exported voice from Python with `TTSVoice.load`, as in the first example on this
page. Leave out `speaker_reference` to use the packaged default speaker.

The OVOS plugin loads a voice by its catalog id, so its `voice` setting does not accept a
path to an `.onnx` file. The `ref_wav`, `ref_text` and `ref_lang` settings work for any
catalog voice that clones.

## Adding a cloning model

Conversion/export scripts live under `scripts/conversion/`:

| Engine | Script | Produces |
|---|---|---|
| YourTTS | `scripts/conversion/yourtts/export_speaker_encoder.py` | the `coqui_resnet` encoder ONNX |
| StyleTTS2 | `scripts/conversion/styletts2/export_style_encoder.py` | the `styletts2_style` encoder ONNX |
| ZipVoice | `scripts/conversion/zipvoice/export_mel.py` + `infer_zipvoice_onnx.py` | the no-torch mel ONNX + a reference runner |

See [Engines](engines.md) for the structure of the adapters. The d-vector engines use one
graph. ZipVoice is an **iterative** engine: it runs a flow-matching ODE loop over several
ONNX graphs through the overridable `BaseOnnxAdapter.synthesize()`.

---

[Docs index](index.md) · Previous: [Usage guide](usage.md) · Next: [Voice conversion](voice_conversion.md)
