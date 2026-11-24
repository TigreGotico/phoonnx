# Voice Cloning

<!-- agent-hint
covers: how to clone a voice from a reference clip in Python or the OVOS plugin, which engines clone and how, and how to fine-tune a YourTTS voice
source: phoonnx/config.py:SynthesisConfig, phoonnx/reference_audio.py:load_reference_audio, phoonnx/engines/speaker_encoders/__init__.py, phoonnx/voice.py:TTSVoice._request, phoonnx/opm.py, phoonnx/engines/neutts.py, phoonnx/engines/llasa.py, phoonnx/engines/mosstts.py, phoonnx_train/engines/yourtts.py, phoonnx_train/preprocess.py
generated: no
verify: .venv/bin/python -m doctest -o NORMALIZE_WHITESPACE docs/cloning.md
see: voice_conversion.md, training/engines/chatterbox.md, ovos_plugin.md
-->

This page shows how to clone a voice from a reference clip, for developers who use phoonnx
from Python or through the OVOS plugin.

Zero-shot cloning speaks in the voice of a short reference clip, with no training for that
speaker. The page covers the reference audio API, the cloning engines, and how to fine-tune
a cloning voice.

```python
from phoonnx.voice import TTSVoice, SynthesisConfig

voice = TTSVoice.load("model.onnx", "model.json")
for chunk in voice.synthesize(
    "This sentence is spoken in the cloned voice.",
    SynthesisConfig(speaker_reference="reference.wav"),
):
    ...  # chunk.audio_float_array, see Usage for the full AudioChunk API
```

Cloning happens during generation. If your voice is not a cloning engine, see
[Voice conversion](voice_conversion.md). Voice conversion changes the waveform after
generation, and it works with every engine.

## Cloning families

Cloning engines differ in what they make from the reference clip:

| Family | How it works | Needs the text of the reference? | Engines |
|---|---|---|---|
| d-vector | a speaker encoder maps the reference waveform to a fixed embedding that conditions synthesis | No | YourTTS, StyleTTS2, [Chatterbox](training/engines/chatterbox.md), [Spark-TTS](training/engines/sparktts.md) |
| in-context | the reference audio and its transcription are part of the model input, and the model continues that voice | Yes for ZipVoice and F5-TTS, optional for OmniVoice and Spark-TTS | ZipVoice, [F5-TTS](training/engines/f5tts.md), [OmniVoice](training/engines/omnivoice.md), [Spark-TTS](training/engines/sparktts.md) |
| in-context, pre-encoded | the reference is codec-encoded ahead of time and shipped with the voice, and the model continues those audio tokens | Yes (bundled) | NeuTTS, [OuteTTS](training/engines/outetts.md), Llasa |
| audio prompt only | the reference is encoded into audio tokens or a model state, with no transcription | No | MOSS-TTS-Nano, [Pocket TTS](pockettts.md) |

A cloning voice can also ship a default speaker, so it works with or without a reference.
A reference that you pass replaces the default speaker.

## The cloning API

Three `SynthesisConfig` fields drive cloning:

| Field | Type | Used by | Meaning |
|---|---|---|---|
| `speaker_reference` | `str` path or URL, or `(audio, sample_rate)` | all cloning engines | the reference clip |
| `speaker_reference_text` | `str` | in-context only | the transcription of the reference |
| `speaker_reference_lang` | `str` (e.g. `"pt"`) | in-context only | the language of the transcription (defaults to the `lang_code` of the voice) |

d-vector engines ignore `speaker_reference_text` and `speaker_reference_lang`. In-context
engines need `speaker_reference_text`.

### Reference audio formats

`phoonnx.reference_audio.load_reference_audio` reads the reference. It gives mono float32
audio and the sample rate:

```python
>>> import os, tempfile, wave
>>> import numpy as np
>>> from phoonnx.reference_audio import load_reference_audio
>>> path = os.path.join(tempfile.mkdtemp(), "ref.wav")
>>> with wave.open(path, "wb") as wav:
...     wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
...     wav.writeframes(np.zeros(1600, np.int16).tobytes())
>>> audio, sample_rate = load_reference_audio(path)
>>> audio.shape, audio.dtype, sample_rate
((1600,), dtype('float32'), 16000)

```

When `soundfile` is installed, phoonnx reads each reference with it, WAV included, so
FLAC, OGG and MP3 references also work. The `phoonnx[cloning]` extra installs `soundfile`
and `scipy`. Without `soundfile`, phoonnx reads the reference with the standard-library
`wave` module. That path decodes 8, 16, 24 and 32-bit PCM WAV files. Any other sample
width, a float WAV, and a FLAC, OGG or MP3 file raise `ValueError`. The message names the
file and the supported formats. Install the extra for any other reference:

```bash
pip install --pre "phoonnx[cloning]"
```

A reference can also be an http or https URL, so a client of a TTS server can send a clip
that is not on the disk of the server. phoonnx downloads at most 32 MiB in at most 30 s. It
refuses a host that resolves to a private, loopback or link-local address:

```python
>>> from phoonnx.reference_audio import load_reference_audio
>>> load_reference_audio("http://127.0.0.1/clip.wav")
Traceback (most recent call last):
    ...
ValueError: refusing to fetch a reference clip from 127.0.0.1: it resolves inside this network, and the URL came from the caller

```

## d-vector engines (YourTTS, StyleTTS2)

A d-vector engine needs no transcription, and the clip can be in any language. The speaker
encoder summarizes the timbre:

```python
for chunk in voice.synthesize(
    "Speak this in the cloned voice.",
    SynthesisConfig(speaker_reference="any_language_clip.wav"),
):
    ...
```

The encoders are in the speaker encoder registry, `phoonnx.engines.speaker_encoders`:

```python
>>> from phoonnx.engines.speaker_encoders import list_speaker_encoders
>>> list_speaker_encoders()
['coqui_resnet', 'styletts2_style']

```

| Type | Engine | The reference becomes |
|---|---|---|
| `coqui_resnet` | YourTTS | a 512-value speaker d-vector |
| `styletts2_style` | StyleTTS2 | a 256-value style (acoustic and prosodic) |

The voice index entry of a cloning voice names its encoder with `speaker_encoder_url` and
`speaker_encoder_type`. The model manager downloads the encoder to
`engine_params["speaker_encoder_path"]`, and the adapter loads it in `configure()`.

The `bsc/es-styletts2` and `bsc/ca-styletts2` voices have no speaker of their own, so they
need a reference clip. Their named speakers, `bsc/ca-<name>` and `bsc/es-cml<id>`, are the
same two checkpoints with a style file attached. They speak without a reference clip, and
they clone when you pass one. See [BSC multispeaker](bsc_multispeaker.md). The Galician
`proxectonos/celtia-styletts2` and `proxectonos/brais-styletts2` voices are single-speaker
voices. They ship a default style and use a reference clip only when you pass one. See
[Galician](galician.md).

## Autoregressive engine (Chatterbox)

[Chatterbox](training/engines/chatterbox.md) is also a d-vector engine: it takes a reference
clip and no transcription. It needs a reference on each call. It adds an `exaggeration`
control from 0.0 to 1.0 (default 0.5) for expressiveness:

```python
chunks = list(voice.synthesize("...", SynthesisConfig(speaker_reference="ref.wav", exaggeration=0.6)))
```

The speech tokens are sampled. `extra_params={"seed": 0}` fixes the speech tokens. The
decoder adds its own random noise, so two runs with one seed still give slightly different
audio.

## Preset engine (NeuTTS)

[NeuTTS](engines.md) clones in-context, but the reference is encoded once, offline, and
shipped with the voice. A preset is a reference transcription plus the NeuCodec tokens of
its recording. The model gets both and continues the voice. Each bundled voice selects one
preset, so it needs no cloning arguments:

```python
chunks = list(voice.synthesize("Meda wo ase paa."))
```

Select another preset for one call through `extra_params`:

```python
SynthesisConfig(extra_params={"voice": "kofi", "temperature": 0.4, "seed": 0})
```

NeuTTS does not support `speaker_reference`. Neuphonic publishes the NeuCodec decoder as
ONNX, but not the encoder, so phoonnx cannot turn a clip into reference tokens. A
`speaker_reference` raises a `RuntimeError`.

`extra_params` tunes the sampling with `temperature`, `top_p`, `top_k`,
`repetition_penalty` and `max_new_tokens`. The loop makes 50 codec tokens per second of
audio. `seed` makes a run reproducible.

## Preset engine (Llasa)

[Llasa](engines.md) is the other pre-encoded preset engine. Llasa needs no reference: with
text alone, it makes up a speaker, so two calls do not sound like the same person. A preset
makes a voice repeatable. It holds the transcript of an utterance that the model generated
and the XCodec2 tokens that it gave for it, and the engine uses them as the in-context
prefix.

Each bundled preset is therefore machine-generated, and its `voices.json` entry has
`"synthetic": true`. No preset is a recording of a person.

```python
chunks = list(voice.synthesize("Dealing with family secrets is never easy."))
SynthesisConfig(extra_params={"voice": "zh_male_a", "temperature": 0.9, "seed": 0})
```

Llasa does not support `speaker_reference`. A fresh clip needs the XCodec2 encoder and the
w2v-BERT filterbank front end, and the bundle ships neither. A `speaker_reference` raises a
`RuntimeError`.

`extra_params` tunes the sampling with `temperature`, `top_p` and `max_new_tokens`. The
codec makes 50 tokens per second of audio. `seed` makes a run reproducible. The Llasa
sampler applies temperature and then top-p, in the order of the Hugging Face warpers,
because upstream runs the checkpoint with that stack. NeuTTS uses the opposite order of
llama.cpp for the same reason.

## Codec-LM engine (MOSS-TTS-Nano)

MOSS-TTS-Nano clones zero-shot from a reference clip alone, with no transcription and no
speaker encoder. The audio tokenizer of the model encodes the clip to RVQ-16 audio tokens
and puts them before the prompt, so cloning works across languages:

```python
chunks = list(voice.synthesize("...", SynthesisConfig(speaker_reference="alice.wav")))
```

`extra_params` tunes the sampling. By default, the engine uses the graph with the upstream
sampling constants compiled in, which is the fastest. A change to `audio_temperature`,
`audio_top_p`, `audio_top_k` or `audio_repetition_penalty` switches to sampling on the host.
`sample_mode="greedy"` is deterministic:

```python
SynthesisConfig(speaker_reference="alice.wav",
                extra_params={"audio_temperature": 0.6, "max_new_frames": 250, "seed": 0})
```

`max_new_frames` limits the loop. The model makes 12.5 frames per second of audio, so the
default of 375 frames is 30 s per model call.

## In-context engine (ZipVoice)

[ZipVoice](training/engines/zipvoice.md) is a flow-matching model that fills in the target
after the reference. It needs the reference audio and its transcription, because the text
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

The transcription aligns the reference, so the reference can be in a different language
from the target. Set `speaker_reference_lang` so that phoonnx phonemizes the transcription
in its own language:

```python
# a Portuguese reference clip speaking English
SynthesisConfig(
    speaker_reference="miro_pt.wav",
    speaker_reference_text="olá, tudo bem com você",
    speaker_reference_lang="pt",
)
```

The quality depends on the phoneme coverage of the model. ZipVoice learned mostly English
and Chinese. For cloning with no text in any language, use a d-vector engine.

## OVOS plugin

The OpenVoiceOS plugin reads the cloning settings from `mycroft.conf`:

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

The config keys map to the `SynthesisConfig` fields. `ref_wav` (or its aliases
`clone_voice` and `speaker_reference`) sets the reference clip. `ref_text` sets
`speaker_reference_text`, and `ref_lang` sets `speaker_reference_lang`. Only in-context
engines need `ref_text` and `ref_lang`. d-vector voices use `ref_wav` alone.

## Fine-tune your own cloning voice (YourTTS)

The `yourtts` training engine (`phoonnx_train/engines/yourtts.py`) fine-tunes a zero-shot
cloning voice on your own multi-speaker data. It uses the VITS training pipeline
(`phoonnx_train/vits/`). The difference from plain VITS is what conditions the decoder.
Plain VITS learns an embedding table with one row per speaker id. YourTTS conditions each
utterance on a 512-value d-vector from the Coqui ResNet speaker encoder that inference also
uses (`phoonnx.engines.speaker_encoders.coqui_resnet`), so the training and inference
embeddings match.

An optional language embedding supports multilingual training. Its size is one more than
the highest `language_id` of the preprocessed utterances, so ids 0, 1 and 2 give three
languages. The embedding is on when the size is 2 or more. An `n_langs` key in the
`config.json` of the preprocessed dataset sets the size instead.

### 1. Preprocess

Preprocessing is the LJSpeech-style pipeline of [preprocessing](training/preprocess.md).
The speaker names come from the second column of `metadata.csv`, so do not pass
`--single-speaker`. Each utterance needs a d-vector. `preprocess.py` computes it when you
name the engine and a Coqui ResNet ONNX speaker encoder. Use the encoder file that you plan
to ship with the voice:

```bash
python -m phoonnx_train.preprocess \
  --language en-US --input-dir data --output-dir preprocessed \
  --engine yourtts \
  --speaker-encoder-path speaker_encoder.onnx \
  --language-id 0     # use another id for each corpus in multilingual training
```

The d-vectors are cached as `<cache_dir>/dvec/<hash>.pt`, next to the mel and audio cache,
so a second preprocessing run reads them from the cache. Each line of the resulting
`dataset.jsonl` has a `d_vector_path`, and a `language_id` for multilingual runs, next to
the usual `phoneme_ids`, `audio_norm_path` and `audio_spec_path` fields. The shared
`phoonnx_train.vits.dataset.PhoonnxDataset` loads them. The fields are optional, so a plain
VITS `dataset.jsonl` without them still loads.

### Speaker-consistency loss

Set the `speaker_encoder_checkpoint` parameter of `VitsModel` to the released H/ASP torch
checkpoint (`model_se.pth.tar`, the file that the ONNX encoder came from) to turn on the
YourTTS speaker-consistency loss. The frozen encoder embeds the real and the generated
audio segments. The generator loss then gets `c_scl` (default 9.0) times the negative
cosine similarity of the two. Without a checkpoint, the loss is off, and training is
d-vector-conditioned VITS.

### 2. Train

```bash
python -m phoonnx_train.train \
  --dataset-dir /data/preprocessed \
  --engine yourtts \
  --quality medium \
  --accelerator gpu --devices 1 \
  --batch-size 16 --max-epochs 1000
```

The quality presets are the plain VITS tiers `x-low`, `medium` and `high`. They set
`hidden_channels`, `inter_channels`, `filter_channels`, `n_heads` and `n_layers`. Choose the
tier by the same trade-off of size and quality as any other `phoonnx_train` run.
`YourttsTrainingEngine.create_model` builds the `VitsModel` of the `vits` engine
(`phoonnx_train.vits.lightning.VitsModel`), with `external_speaker_embedding=True` and
`num_speakers=1`, because the speaker comes from the d-vector, not from an id table.
`n_langs` comes from the language ids of the utterances.

### 3. Export

```bash
python -m phoonnx_train.export_onnx \
  /path/to/checkpoint.ckpt \
  --config /path/to/config.json \
  --engine yourtts \
  --output-dir /path/to/output/
```

The export has the inputs of the plain VITS ONNX graph (`input`, `input_lengths`,
`scales`), plus `d_vector` and `langid`, as `phoonnx.engines.yourtts.YourTTSAdapter` feeds
them. The voice config next to the `.onnx` file sets `"engine": "yourtts"`. It also holds a
default speaker in `engine_params["d_vector"]`: the mean embedding of the training set. The
voice therefore speaks with no reference clip. A reference clip, or a `d_vector` at
synthesis time, replaces the default, as for any other cloning voice with a default
speaker.

### Use the fine-tuned voice

Load the exported voice from Python with `TTSVoice.load`, as in the first example on this
page. Leave out `speaker_reference` to use the default speaker of the voice.

The OVOS plugin loads a voice by its catalog id, so its `voice` setting does not accept a
path to an `.onnx` file. The `ref_wav`, `ref_text` and `ref_lang` settings work for any
catalog voice that clones.

## Add a cloning model

The conversion and export scripts are under `scripts/conversion/`:

| Engine | Script | Produces |
|---|---|---|
| YourTTS | `scripts/conversion/yourtts/export_speaker_encoder.py` | the `coqui_resnet` encoder ONNX |
| StyleTTS2 | `scripts/conversion/styletts2/export_style_encoder.py` | the `styletts2_style` encoder ONNX |
| ZipVoice | `scripts/conversion/zipvoice/export_mel.py` + `infer_zipvoice_onnx.py` | the mel ONNX that needs no torch, and a reference runner |

See [Engines](engines.md) for the structure of the adapters. A d-vector engine runs one
synthesis graph. ZipVoice is an iterative engine: it runs a flow-matching ODE loop over
several ONNX graphs through the `BaseOnnxAdapter.synthesize()` method, which an adapter can
override.

---

[Docs index](index.md) · Previous: [SSML](ssml.md) · Next: [Voice conversion](voice_conversion.md)
