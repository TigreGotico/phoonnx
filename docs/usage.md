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

`synthesize()` is a generator that yields one `AudioChunk` per sentence. This enables low-latency streaming:

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

If you have pre-computed phoneme IDs, you can synthesize raw audio directly:

```python
import numpy as np
phoneme_ids = [1, 5, 23, 7, 2]  # example IDs
audio: np.ndarray = voice.phoneme_ids_to_audio(phoneme_ids)
```

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

You can define custom word-level pronunciation overrides that are applied before phonemization:

```python
# phonetic_spellings is loaded from a JSON file at voice load time
# Format: {"word": "replacement_text_or_phonemes"}
voice.phonetic_spellings  # dict-like object, or None if not configured
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

`synthesize()` is built from three composable steps you can call yourself:

```python
# text -> phonemes grouped by sentence (a list of lists of phoneme strings)
chunks = voice.phonemize("Hello world", lang="en-US")

# phonemes -> integer token IDs
ids = voice.phonemes_to_ids(chunks[0])

# token IDs -> raw float32 audio (unnormalized)
audio = voice.phoneme_ids_to_audio(ids)
```

`phonemize()` accepts an optional `lang` to phonemize in a language other than the voice's own
(used for cross-lingual cloning references). `phoneme_ids_to_audio()` accepts a
`SynthesisConfig` and optional per-phoneme `language_ids`.
