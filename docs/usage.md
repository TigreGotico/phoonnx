# Usage Guide

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
    use_cuda=False,
)
```

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
    add_diacritics=True,    # add diacritics (Arabic/Hebrew models)
)
```

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
| `audio_int16_bytes` | Raw PCM bytes (int16, little-endian) |
| `sample_rate` | Sample rate in Hz (e.g. 22050) |
| `sample_width` | Bytes per sample (always 2) |
| `sample_channels` | Number of channels (always 1, mono) |
