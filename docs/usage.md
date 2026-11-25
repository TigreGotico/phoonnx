# Usage Guide

<!-- agent-hint
covers: loading a voice, synthesis to WAV or a stream, SynthesisConfig, inline phonemes, phonetic spellings, SSML, multi-speaker voices and the low-level prepare, render, phonemize and tokenize calls
source: phoonnx/voice.py:TTSVoice, phoonnx/voice.py:AudioChunk, phoonnx/voice.py:PhoneticSpellings, phoonnx/encoders.py:Utterance, phoonnx/config.py:SynthesisConfig, phoonnx/ssml.py, phoonnx/providers.py
generated: no
verify: .venv/bin/python -m doctest -o NORMALIZE_WHITESPACE docs/usage.md
see: quickstart.md, configuration.md, streaming.md
-->

This guide is for Python developers who use phoonnx directly. It shows how to load a
voice, synthesize to a WAV file or a stream, tune synthesis, and use the low-level
`prepare` and `render` calls.

## Loading a Voice

Load a voice from the catalog by its id. This English voice uses the eSpeak phonemizer, so
install the `espeak` extra first:

```bash
pip install --pre "phoonnx[espeak]"
```

Then save this script as `speak.py` and run it:

```python
import wave
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.load()
manager.merge_default_voices()

voice = manager.voices["OpenVoiceOS/pipertts_en-US_miro"].load()

with wave.open("output.wav", "wb") as wav_file:
    voice.synthesize_wav("Hello, world!", wav_file)
```

The first `load()` downloads the voice, about 64 MB, into the Hugging Face cache. Later loads
read it from the cache. The script writes `output.wav`, at the 22050 Hz rate of the voice,
mono, 2 bytes per sample:

```
$ python speak.py
$ python -c "import wave; w = wave.open('output.wav'); print(w.getframerate(), w.getnchannels(), w.getsampwidth())"
22050 1 2
```

The examples on the rest of this page continue this script and use its `voice`.

### Choosing execution providers

Pass an ordered list of ONNX Runtime execution providers to run on a GPU. The list also sets
the providers of the auxiliary graphs that an engine loads (vocoders, speaker encoders):

```python
voice = manager.voices["OpenVoiceOS/pipertts_en-US_miro"].load(
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

When you omit `providers`, phoonnx reads the `PHOONNX_ONNX_PROVIDERS` environment variable,
and then detects the providers that the installed ONNX Runtime has. See the
[Configuration reference](configuration.md#execution-providers).

The `threads` argument sets the threads per graph operator in each session of the voice.
When you omit it, phoonnx reads `PHOONNX_ORT_THREADS`, and then uses the ONNX Runtime
default of one thread per core.

Set `PHOONNX_ORT_CACHE_DIR` to a writable directory to cache the graph that ONNX Runtime
optimizes. Later processes load the optimized graph and skip the optimization pass.

### Loading from files

A voice you trained, or any `model.onnx` with its JSON config, loads with `TTSVoice.load()`:

```python
from phoonnx.voice import TTSVoice

voice = TTSVoice.load(
    model_path="model.onnx",
    config_path="model.json",   # defaults to model_path + ".json" if omitted
)
```

Without `config_path`, `TTSVoice.load` looks for `<model_path>.json`, for example
`model.onnx.json`. If that file is missing, it raises `VoiceLoadError`. Use this call for
your own files only. In the Hugging Face cache, the file names change from voice to voice,
and `manager.voices[voice_id].load()` finds them for you.

`TTSVoice.load` also takes overrides for the phonemizer and the language:

```python
voice = TTSVoice.load(
    model_path="model.onnx",
    config_path="model.json",
    lang_code="en-US",
    phoneme_type_str="espeak",   # override phonemizer
    alphabet_str="ipa",
)
```

Pass `warmup=True` to `TTSVoice.load` to run one inference during the load. The first
`synthesize` call then does not pay the kernel selection cost of ONNX Runtime.
`use_cuda=True` is a deprecated alias for `providers=["CUDAExecutionProvider"]`. Use
`providers`.

### Reading text through another frontend

A voice reads text through the frontend that its config describes. To give it another
frontend, for example another phonemizer of the same alphabet, call `with_frontend`. It
returns a new voice that shares the ONNX session, the tokenizer and the engine adapter. The
old voice does not change.

1. Make a `FrontendConfig` or a `TextFrontend` from scriptconv:

   ```python
   from scriptconv.frontend import FrontendConfig
   from scriptconv.phonemizers.enums import Alphabet, Phonemizer

   frontend = FrontendConfig(lang="en-US", phonemizer=Phonemizer.GRAPHEMES, alphabet=Alphabet.UNICODE)
   ```

2. Give it to the voice. Pass a probe text to check the symbols that the frontend gives:

   ```python
   voice = voice.with_frontend(frontend, probe_text="The quick brown fox.")
   ```

   You can also pass it to the load: `TTSVoice.load("model.onnx", frontend=frontend)`.

The frontend must give the symbols that the voice was trained on. A different alphabet, a
different `keep_punctuation`, or a probe symbol that the voice vocabulary lacks raises
`FrontendMismatchError`:

```text
phoonnx.errors.FrontendMismatchError: the frontend gives alphabet <Alphabet.IPA: 'ipa'>, and the voice reads <Alphabet.UNICODE: 'unicode'>
phoonnx.errors.FrontendMismatchError: the frontend gives symbols that the voice vocabulary lacks: ['q', 'u']
```

The new voice keeps its frontend. A later change to `voice.phonemizer` does not reach it.

## Synthesizing Speech

### To a WAV File

`synthesize_wav()` writes the whole text to an open WAV file, as `speak.py` does above:

```python
with wave.open("output.wav", "wb") as wav_file:
    voice.synthesize_wav("Hello, world!", wav_file)
```

### Streaming Audio Chunks

`synthesize()` is a generator that yields one `AudioChunk` per sentence. You can play the
first sentence while the voice makes the next one. The text frontend of the voice splits the
text into sentences. You can see the split without a voice:

```python
>>> from scriptconv.phonemizers import get_phonemizer
>>> from phoonnx.config import PhonemeType, voice_frontend
>>> frontend = voice_frontend(get_phonemizer(PhonemeType.UNICODE), "en-US")
>>> [segment.text for segment in frontend.segments("Hello! How are you today?")]
['Hello!', 'How are you today?']
>>> len(list(frontend.segments("Hello world. How are you today?")))
2

```

Each of these texts thus gives two chunks.

```python
from phoonnx.config import SynthesisConfig

syn_config = SynthesisConfig(length_scale=1.2)  # slightly slower speech

for chunk in voice.synthesize("Hello world. How are you today?", syn_config=syn_config):
    print(f"Sample rate: {chunk.sample_rate}")
    print(f"Audio shape: {chunk.audio_float_array.shape}")
    # chunk.audio_int16_bytes holds raw PCM bytes for playback
```

`SynthesisConfig(stages=Stages(chunk=...))` sets how much text one engine call reads: `"sentence"` (the
default), `"clause"`, `"paragraph"` or `"none"` (the whole text). Any other value raises a
`ValueError`. `voice.stream()` yields parts of a sentence as soon as the engine makes them,
for engines that can stream. See [Streaming](streaming.md).

### From Phoneme IDs Directly

If you have token IDs for the voice already, put them in an `Utterance` and render it:

```python
from phoonnx.voice import Utterance

chunk = voice.render(Utterance(text="", symbols=None, ids=(1, 5, 23, 7, 2)))
audio = chunk.audio_float_array
```

### Phoneme Alignment

Every `AudioChunk` carries the `phonemes` and `phoneme_ids` that produced it.
Pass `include_alignments=True` to also get the timing of each phoneme, as the number of
audio samples that it spans, when the model has a duration output:

```python
for chunk in voice.synthesize("Hello world.", include_alignments=True):
    for align in chunk.phoneme_alignments or []:
        ms = align.num_samples / chunk.sample_rate * 1000
        print(f"{align.phoneme!r:6s} {ms:6.1f} ms")
```

`include_alignments=False` (the default) does not change the audio. For a model
with no duration output, `phoneme_alignments` is `None`. If a VITS model has no duration
output but has a duration tensor inside it, the first `include_alignments=True` call adds
the output with graph surgery. This needs the `onnx` package from
`pip install --pre "phoonnx[streaming]"`. Without it, phoonnx logs an INFO message and
`phoneme_alignments` is `None`. See [Phoneme alignment](alignment.md) for the engine
support table and how to export a VITS model with alignment.

## SynthesisConfig

`SynthesisConfig` controls synthesis parameters at runtime. Each field has a default. Some
defaults are `None`, which lets the voice config decide. The other defaults are fixed values:

```python
>>> from phoonnx.config import SynthesisConfig
>>> config = SynthesisConfig()
>>> config.speaker, config.language, config.length_scale, config.noise_scale, config.noise_w_scale
(None, None, None, None, None)
>>> config.stages.diacritize, config.stages.diacritizer_model
(None, None)
>>> config.audio.normalize, config.audio.volume, config.phonetic_spellings
(True, 1.0, True)

```

This example sets each field explicitly:

```python
from phoonnx.config import AudioPost, SynthesisConfig
from scriptconv.frontend import Stages

syn_config = SynthesisConfig(
    speaker=0,              # for multi-speaker voices: an id or a name
    language=0,             # for multi-language voices: an id or a code
    length_scale=1.0,       # 1.0 = normal speed, <1 faster, >1 slower
    noise_scale=0.667,      # generator noise (affects naturalness/variation)
    noise_w_scale=0.8,      # phoneme duration noise
    audio=AudioPost(normalize=True, volume=1.0),  # full amplitude range, volume multiplier
    phonetic_spellings=True,  # apply user-defined word replacements
    stages=Stages(diacritize=None),  # None defers to the voice config (Arabic/Hebrew)
)
```

`speaker`, `language`, `length_scale`, `noise_scale` and `noise_w_scale` default to `None`,
which uses the value of the voice config. `stages.diacritize` defaults to `None`, which uses the
setting of the voice. Set it to `True` or `False` only to force diacritization on or off for
this call. `stages.diacritizer_model` (default `None`) also uses the choice of the voice config. Set
it to override the Arabic `text2tashkeel` model for this call. Cloning and autoregressive
engines use more fields, such as `reference` (a `SpeakerReference` with `audio`, `text` and
`lang`), `exaggeration`, `temperature`, `top_p` and `extra`. The config is frozen: build a
changed copy with `dataclasses.replace`. See
[Cloning](cloning.md) and the [Configuration reference](configuration.md#synthesisconfig).

## Inline Phoneme Input

You can bypass the phonemizer for specific words by embedding phonemes directly in double brackets:

```python
text = "The word [[θɪs]] is phonemized manually."
for chunk in voice.synthesize(text):
    ...
```

A block holds IPA. A voice that reads another alphabet, such as X-SAMPA, gets the block
converted to that alphabet. A voice that takes no symbol input, such as tiny-voca or an
engine that reads raw text, logs one warning that names the voice and the blocks, and
leaves the blocks out. An ARPA voice gets the block as IPA, which its vocabulary does not
hold, so the tokenizer drops the block with an out-of-vocabulary warning.

A skipped block leaves one space between the words around it, or the line break next to it
when there is one. Before a punctuation mark, it leaves nothing. An SSML `<phoneme>` on such
a voice reads its written word.

## Phonetic Spellings

Phonetic spellings replace words in the text before phonemization. The
`phonetic_spellings` setting turns them on, and it is on by default. The packaged
spellings are in `phoonnx/locale/<lang>/phonetic_spellings.txt`, one `word: spelling` per
line. A `#` starts a comment. Matches are whole words, and case does not matter.

To add your own, put `<lang>.txt` files in a directory and set
`PHOONNX_PHONETIC_SPELLINGS_DIR` to it. You can also give the directory to
`PhoneticSpellings.from_lang` as `user_dir`:

```python
>>> import pathlib, tempfile
>>> from phoonnx.voice import PhoneticSpellings
>>> user_dir = tempfile.mkdtemp()
>>> _ = pathlib.Path(user_dir, "fr.txt").write_text("Thalovant: Talovant\nd'accord: dakor\n")
>>> spellings = PhoneticSpellings.from_lang("fr-CA", user_dir=user_dir)
>>> spellings.apply("Bonjour THALOVANT, d'accord?")
'Bonjour Talovant, dakor?'

```

Name a file by the primary language (`fr.txt`) or by the full tag (`fr-CA.txt`), or both.
The user files apply over the packaged file, and the full tag applies over the primary
language. `voice.phonetic_spellings` holds the `PhoneticSpellings` of a voice, or `None`.

## SSML

`TTSVoice.synthesize_ssml()` and `synthesize_ssml_wav()` read SSML. They turn it into
silence, pace, loudness, pitch, language and sound effects. Input that is not SSML is
synthesized as it is.

```python
marks = []
with wave.open("out.wav", "wb") as wav:
    voice.synthesize_ssml_wav(
        '<speak>Six degrees. <break time="400ms"/> <mark name="rain"/>'
        '<prosody rate="slow" pitch="-2st">Feels like <say-as interpret-as="cardinal">4</say-as>.</prosody>'
        '</speak>', wav, on_mark=lambda name, seconds: marks.append((name, seconds)))
```

`on_mark(name, seconds)` is called for each `<mark>` as the audio after it is produced.
The text frontend of the voice reads `<say-as>` with `TextFrontend.say_as` from scriptconv,
in the language of the text:

```python
voice.frontend.say_as("3", "ordinal", lang="en-US")  # 'third'
```

The [SSML](ssml.md) page lists each element and attribute that phoonnx supports, with its
limits and its cost. The OVOS plugin (`phoonnx.opm`) sends SSML input to these methods.

## Multi-Speaker Voices

For voices with multiple speakers, set `speaker` in `SynthesisConfig`, as an id or as a name in
the voice's `speaker_id_map`:

```python
syn_config = SynthesisConfig(speaker=2)
for chunk in voice.synthesize("Hello!", syn_config=syn_config):
    ...
```

## Audio Output Format

An `AudioChunk` holds the audio as a float32 array in `[-1.0, 1.0]`, and it converts the
array to int16 samples on request:

```python
>>> import numpy as np
>>> from phoonnx.voice import AudioChunk
>>> chunk = AudioChunk(sample_rate=22050, sample_width=2, sample_channels=1,
...                    audio_float_array=np.array([0.0, 0.5, -1.0], dtype=np.float32))
>>> chunk.audio_int16_array
array([     0,  16383, -32767], dtype=int16)
>>> len(chunk.audio_int16_bytes)
6

```

| Attribute | Description |
|-----------|-------------|
| `audio_float_array` | `np.ndarray` (float32), values in `[-1.0, 1.0]` |
| `audio_int16_array` | `np.ndarray` (int16), the float array scaled by 32767 and clipped |
| `audio_int16_bytes` | Raw PCM bytes (int16, in the byte order of the machine) |
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
`Utterance` cannot be changed after it is made:

```python
>>> from phoonnx.voice import Utterance
>>> utterance = Utterance(text="", symbols=None, ids=(1, 5, 23, 7, 2))
>>> utterance.ids = (1,)
Traceback (most recent call last):
    ...
dataclasses.FrozenInstanceError: cannot assign to field 'ids'

```

It has five fields:

| Field | Value |
|-------|-------|
| `text` | The text of the chunk of this engine call. `""` when the text of the previous utterance covers this chunk, and `None` when it is unknown |
| `symbols` | The phonemes or characters that the tokenizer read. `None` for an engine that reads text, such as Pocket TTS. Its adapter makes the IDs from the text |
| `ids` | The token IDs |
| `streams` | Values that some phonemizers give for each ID, such as `language_ids` from Shami |
| `params` | Values that the engine adapter learned from the text of this utterance. They join the request parameters of the engine call |

`render()` runs the engine and then does the same audio steps as `synthesize()`: level,
volume, clip, voice conversion and super-resolution. It always returns one `AudioChunk`.
Use `render(utterance, config, alignments=True)` to get the durations of the phonemes in the
chunk.

The two steps let you do work between them. For example, you can keep the utterances of a
text and render them again with another speaker or another length scale. The text settings
of `SynthesisConfig` (diacritics, phonetic spellings, input alphabet, chunk) apply in
`prepare()`, and the engine and audio settings apply in `render()`.

`phonemize()` gives the phonemes of a text with no diacritizer, one list for each sentence.
It accepts an optional `lang`, so that a cloning reference can be phonemized in its own
language. `voice.tokenizer.tokenize(phonemes)` turns one such list into token IDs.

`phonemes_to_ids()` and `phoneme_ids_to_audio()` are deprecated. Use `voice.tokenizer.tokenize`
and `render()`. `phoneme_ids_to_audio()` returns the engine audio before the level, volume
and clip steps.

---

[Docs index](index.md) · Previous: [BSC multispeaker StyleTTS2](bsc_multispeaker.md) · Next: [SSML](ssml.md)
