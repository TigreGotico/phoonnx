# Quickstart

This guide is for anyone who wants speech out of phoonnx as fast as possible. In about five
minutes you will install phoonnx, download a ready-made voice, and synthesize a WAV — from
the command line and from Python.

## 1. Install

```bash
pip install phoonnx
```

The base install runs ONNX voices that use grapheme/unicode tokenization plus the full
`phoonnx-voices` command line. Voices that need a language-specific phonemizer, cloning, or
streaming pull in extra packages — see [Installation](installation.md).

## 2. Find and download a voice

Populate the local catalog, browse it, then download one voice:

```bash
phoonnx-voices update-cache
phoonnx-voices list-voices --lang pt-PT
phoonnx-voices download OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone
```

`update-cache` prints how many voices and languages are now available. `download` fetches the
model into the per-voice cache directory:

```
~/.cache/phoonnx/voices/OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone/
├── model.onnx     # the ONNX voice
└── model.json     # its configuration
```

The full command set is in the [CLI reference](cli.md); browsing and caching is covered in the
[Voice manager](voice_manager.md) guide.

## 3. Synthesize from Python

The voice manager loads a downloaded voice by ID and resolves all of its files for you:

```python
import wave
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.load()
manager.merge_default_voices()

voice = manager.voices["OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone"].load()

with wave.open("hello.wav", "wb") as wav_file:
    voice.synthesize_wav("Olá, isto é um teste.", wav_file)
```

After this runs you have `hello.wav`, a mono 16-bit PCM file at the voice's sample rate.

If you already have a `model.onnx` and `model.json` on disk (for example a voice you trained),
load them directly instead:

```python
import wave
from phoonnx.voice import TTSVoice

voice = TTSVoice.load("model.onnx", "model.json")
with wave.open("hello.wav", "wb") as wav_file:
    voice.synthesize_wav("Hello world!", wav_file)
```

> **Note:** `TTSVoice.load(model_path)` defaults `config_path` to `<model_path>.json`
> (e.g. `model.onnx.json`), not `model.json`. The voice manager's cache layout ships
> `model.onnx` + `model.json` side by side, so if you load straight from a cache
> directory (instead of via `manager.voices[voice_id].load()`, which resolves this
> for you) pass `config_path` explicitly: `TTSVoice.load(cache_dir / "model.onnx", cache_dir / "model.json")`.

## 4. Tune the output

`SynthesisConfig` controls speed, expressiveness and more per call:

```python
from phoonnx.config import SynthesisConfig

cfg = SynthesisConfig(
    length_scale=1.1,   # >1 slower, <1 faster
    noise_scale=0.667,  # generator noise / variation
    volume=1.0,
)
with wave.open("hello.wav", "wb") as wav_file:
    voice.synthesize_wav("Olá, isto é um teste.", wav_file, cfg)
```

Every knob is described in [Usage](usage.md) and the [Configuration reference](configuration.md).

## Common problems

- **`No voices found in cache. Run 'update-cache' first.`** — run `phoonnx-voices update-cache`
  before `list-voices`/`download`, or call `manager.merge_default_voices()` in Python.
- **`Voice ID '...' not found.`** — check the exact ID with `phoonnx-voices list-available`
  (bundled IDs grouped by source, no download).
- **No sound / silence** — confirm the model downloaded fully; a partial `model.onnx` fails
  to load. Re-run `download`.

## Next steps

- Run the voice inside a voice assistant: [OVOS plugin](ovos_plugin.md).
- Clone a voice from a reference clip: [Cloning](cloning.md).
- Build your own voice: [Training quickstart](training/quickstart.md).
