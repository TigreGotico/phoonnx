# Usage Guide

This guide is for Python developers driving phoonnx directly. After reading it you can load a
voice, synthesize to a WAV or a stream, tune synthesis, and reach the low-level phonemize /
tokenize / vocode calls.

## Loading a Voice

The main entry point is `TTSVoice.load()`. You need an ONNX model file and its accompanying JSON config:

```python
from phoonnx.voice import TTSVoice

voice = TTSVoice.load(
    model_path="model.onnx",
    config_path="model.json",   # defaults to model_path + ".json" if omitted
)
```

> **Note:** the default (`<model_path>.json`, e.g. `model.onnx.json`) does **not**
> match the voice manager's cache layout, which ships `model.onnx` + `model.json`
> side by side. Loading straight out of a cache directory needs `config_path`
> passed explicitly as above; `manager.voices[voice_id].load()` already does this
> for you.

### Loading with Overrides

You can override the phonemizer and language at load time:

```python
voice = TTSVoice.load(
    model_path="model.onnx",
    config_path="model.json",
    lang_code="en-US",
    phoneme_type_str="espeak",   # override phonemizer
    alphabet_str="ipa",
)
```

### Choosing execution providers

Pass an ordered list of ONNX Runtime execution providers to run on a GPU. The list also drives
the auxiliary graphs an engine loads (vocoders, speaker encoders):

```python
voice = TTSVoice.load(
    "model.onnx", "model.json",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

When omitted, providers come from the `PHOONNX_ONNX_PROVIDERS` environment variable, then from
auto-detection. `use_cuda=True` is a **deprecated** alias for
`providers=["CUDAExecutionProvider"]`; prefer `providers`. See the
[Configuration reference](configuration.md#execution-providers).

Set `PHOONNX_ORT_CACHE_DIR` to a writable directory to cache the ONNX Runtime-optimized graph
across process restarts, and pass `warmup=True` to `TTSVoice.load` to pay the first-inference
kernel-selection cost during load instead of on the first `synthesize` call — both cut cold-start
latency.

## Synthesizing Speech

### To a WAV File

```python
import wave
from phoonnx.voice import TTSVoice

voice = TTSVoice.load("model.onnx", "model.json")

with wave.open("output.wav", "wb") as wav_file:
    voice.synthesize_wav("Hello, world!", wav_file)
```

### Streaming Audio Chunks

`synthesize()` is a generator that yields one `AudioChunk` per sentence. This enables low-latency streaming. `SynthesisConfig(chunk=...)` changes the size of a chunk: `clause` gives the first audio sooner on a long sentence, and `paragraph` or `none` gives fewer false sentence ends.

```python
from phoonnx.voice import TTSVoice
from phoonnx.config import SynthesisConfig

voice = TTSVoice.load("model.onnx", "model.json")
syn_config = SynthesisConfig(length_scale=1.2)  # slightly slower speech

for chunk in voice.synthesize("Hello! How are you today?", syn_config=syn_config):
    print(f"Sample rate: {chunk.sample_rate}")
    print(f"Audio shape: {chunk.audio_float_array.shape}")
    # chunk.audio_int16_bytes — raw PCM bytes for playback
```

### From Phoneme IDs Directly

If you have token IDs for the voice already, put them in an `Utterance` and render it:

```python
from phoonnx.voice import Utterance

chunk = voice.render(Utterance(text="", symbols=None, ids=(1, 5, 23, 7, 2)))
audio = chunk.audio_float_array
```

### Phoneme Alignment

Every `AudioChunk` carries the `phonemes` and `phoneme_ids` that produced it.
Pass `include_alignments=True` to also get per-phoneme timing — how many audio
samples each phoneme spans — when the model exposes a duration output:

```python
for chunk in voice.synthesize("Hello world.", include_alignments=True):
    for align in chunk.phoneme_alignments or []:
        ms = align.num_samples / chunk.sample_rate * 1000
        print(f"{align.phoneme!r:6s} {ms:6.1f} ms")
```

`include_alignments=False` (the default) is a strict no-op on the audio, and
models without a duration output degrade gracefully — `phoneme_alignments`
comes back `None`. If a model was exported without a duration output but has
one to find, the first `include_alignments=True` call retrofits it on demand
via load-time graph surgery (see [Phoneme alignment](alignment.md)'s
"Runtime alignment for models exported without the flag"), so re-exporting
is optional, not required. See [Phoneme alignment](alignment.md) for the full
guide, the engine support matrix, and how to export a VITS model with
alignment.

## SynthesisConfig

`SynthesisConfig` controls synthesis parameters at runtime:

```python
from phoonnx.config import SynthesisConfig

syn_config = SynthesisConfig(
    speaker_id=0,           # for multi-speaker voices
    lang_id=0,              # for multi-language voices
    length_scale=1.0,       # 1.0 = normal speed, <1 faster, >1 slower
    noise_scale=0.667,      # generator noise (affects naturalness/variation)
    noise_w_scale=0.8,      # phoneme duration noise
    normalize_audio=True,   # normalize output to full amplitude range
    volume=1.0,             # volume multiplier
    enable_phonetic_spellings=True,  # apply user-defined word replacements
    add_diacritics=None,    # None defers to the voice config (Arabic/Hebrew)
)
```

`add_diacritics` defaults to `None`, meaning "use the voice's own setting"; set it to `True`
or `False` only to force diacritization on or off for this call. `diacritizer_model` (default
`None`) likewise inherits the voice config's choice — set it to override the Arabic
`text2tashkeel` model for this call. Cloning and autoregressive engines add more fields —
`speaker_reference`, `speaker_reference_text`, `speaker_reference_lang`, `exaggeration`,
`temperature`, `top_p`, `extra_params` — documented in [Cloning](cloning.md) and the
[Configuration reference](configuration.md#synthesisconfig).

## Inline Phoneme Input

You can bypass the phonemizer for specific words by embedding phonemes directly in double brackets:

```python
text = "The word [[θɪs]] is phonemized manually."
for chunk in voice.synthesize(text):
    ...
```

## Phonetic Spellings

Word-level respellings are applied to the text before phonemization (`enable_phonetic_spellings`,
on by default). They come from `phoonnx/locale/<lang>/phonetic_spellings.txt`, one `word: spelling`
per line (`#` starts a comment), matched as whole words, ignoring case.

Add your own without touching the package: put `<lang>.txt` files in a directory and point
`PHOONNX_PHONETIC_SPELLINGS_DIR` at it.

```
$ cat ~/.config/phoonnx/spellings/fr.txt
Thalovant: Talovant
d'accord: dakor

$ PHOONNX_PHONETIC_SPELLINGS_DIR=~/.config/phoonnx/spellings python my_app.py
```

A file is named by primary language (`fr.txt`) and/or full tag (`fr-CA.txt`); they are applied
over the packaged file, the full tag over the primary language. The directory can also be passed
to `PhoneticSpellings.from_lang(lang, user_dir=...)`.

```python
voice.phonetic_spellings  # PhoneticSpellings (.replacements dict, .apply(text)), or None
```

## SSML

`TTSVoice.synthesize_ssml()` and `synthesize_ssml_wav()` read SSML and turn it into silence, pace,
loudness, pitch, language and sound effects. Input that is not SSML is synthesized as it is.

| Tag | Effect |
| --- | --- |
| `<break time="500ms"/>`, `strength=` | silence of that length |
| `<p>`, `<s>` | a pause at the paragraph / sentence edge |
| `<prosody rate= volume= pitch=>` | pace (the voice's length scale), loudness, and pitch (`+2st`, `-10%`, `high`; Hz is ignored) |
| `<emphasis level=>` | a little slower and louder (or softer for `reduced`) |
| `<lang xml:lang=>`, `<voice>` | a multilingual voice uses that language's id for the piece; a monolingual voice says the words in its own language. `<say-as>` inside it also reads its value in that language |
| `<say-as interpret-as=...>` | `cardinal`, `number`, `ordinal`, `digits`, `fraction`, `unit`, `date` (+`format="mdy"`), `time`, `currency`, `telephone`, `characters`/`spell-out`/`letters`/`verbatim`, and `expletive` (a bleep) |
| `<sub alias=>` | substituted text |
| `<phoneme alphabet="ipa\|x-sampa" ph=>` | inline phonemes (X-SAMPA is converted to IPA) |
| `<mark name=>` | `on_mark(name, seconds)` is called as the audio that follows is produced |
| `<amazon:effect name="whispered\|drc">`, `phonation="soft"`, `vocal-tract-length="+10%"` | whisper, soft voice, dynamic range compression, a longer or shorter vocal tract |
| `<amazon:auto-breaths>`, `<amazon:breath>` | breaths between sentences, or at a point |

The voice's text frontend reads `say-as` with `TextFrontend.say_as` from scriptconv. That method
calls text2spoken through the normalisation hook of the voice. So `say-as` reads with the
settings of the voice, and in the language of the text: `<lang>` changes how a value is read.
Each value text2spoken does not read, such as `address` or `interjection`, reads as plain text,
with a warning. A text that does not hold the value (`<say-as interpret-as="date">soon</say-as>`)
also reads as plain text, with a warning.

```python
voice.frontend.say_as("3", "ordinal", lang="en-US")  # 'third'
```

Pitch, whisper, soft, compression, breaths and bleeps are applied to the finished audio
(`phoonnx.audio_fx`, numpy only), not by the model: pitch shifts move the formants with the pitch,
so keep them within a few semitones; whisper is an approximation that keeps the words and drops the
voicing. Emotions and speaking styles are not something a VITS voice can do and are read as plain
text. A document that does not parse is spoken as its text with the tags removed, and plain text
containing a stray `<` is never mistaken for SSML (`phoonnx.ssml.looks_like_ssml`). The OVOS plugin
(`phoonnx.opm`) routes SSML input here and advertises `ssml_tags`.

```python
marks = []
with wave.open("out.wav", "wb") as wav:
    voice.synthesize_ssml_wav(
        '<speak>Six degrees. <break time="400ms"/> <mark name="rain"/>'
        '<prosody rate="slow" pitch="-2st">Feels like <say-as interpret-as="cardinal">4</say-as>.</prosody>'
        '</speak>', wav, on_mark=lambda name, seconds: marks.append((name, seconds)))
```

## Multi-Speaker Voices

For voices with multiple speakers, set `speaker_id` in `SynthesisConfig`:

```python
syn_config = SynthesisConfig(speaker_id=2)
for chunk in voice.synthesize("Hello!", syn_config=syn_config):
    ...
```

## Audio Output Format

Each `AudioChunk` has the following attributes:

| Attribute | Description |
|-----------|-------------|
| `audio_float_array` | `np.ndarray` (float32), values in `[-1.0, 1.0]` |
| `audio_int16_array` | `np.ndarray` (int16), the float array clipped and scaled |
| `audio_int16_bytes` | Raw PCM bytes (int16, little-endian) |
| `sample_rate` | Sample rate in Hz (e.g. 22050) |
| `sample_width` | Bytes per sample (always 2) |
| `sample_channels` | Number of channels (always 1, mono) |

## Low-level API

`synthesize()` has two steps that you can call yourself: `prepare()` turns text into model
input, and `render()` turns model input into audio.

```python
from phoonnx.config import SynthesisConfig

config = SynthesisConfig(length_scale=1.1)
for utterance in voice.prepare("Hello world. How are you?", config):
    print(utterance.symbols)  # the phonemes of one sentence, or None
    print(utterance.ids)      # the token IDs that the engine reads
    chunk = voice.render(utterance, config)
```

`prepare()` yields one `Utterance` for each engine call, usually one for each sentence. An
`Utterance` cannot be changed after it is made. It has four fields:

| Field | Value |
|-------|-------|
| `text` | The text of this engine call's chunk. `""` when the previous utterance's text covers this chunk, and `None` when it is unknown |
| `symbols` | The phonemes or characters that the tokenizer read. `None` for an engine that reads text, such as Pocket TTS. Its adapter makes the IDs from the text |
| `ids` | The token IDs |
| `streams` | Values that some phonemizers give for each ID, such as `language_ids` from Shami |

`render()` runs the engine and then does the same audio steps as `synthesize()`: level,
volume, clip, voice conversion and super-resolution. It always returns one `AudioChunk`.
Use `render(utterance, config, alignments=True)` to get the durations of the phonemes in the
chunk.

The two steps let you do work between them. For example, you can keep the utterances of a
text and render them again with another speaker or another length scale.
The text settings of `SynthesisConfig` (diacritics, phonetic spellings, input alphabet) apply
in `prepare()`, and the engine and audio settings apply in `render()`.

`phonemize()` gives the phonemes of a text with no diacritizer, one list for each sentence.
It accepts an optional `lang`, so that a cloning reference can be phonemized in its own
language. `voice.tokenizer.tokenize(phonemes)` turns one such list into token IDs.

`phonemes_to_ids()` and `phoneme_ids_to_audio()` are deprecated. Use `voice.tokenizer.tokenize`
and `render()`. `phoneme_ids_to_audio()` returns the engine audio before the level, volume
and clip steps.
