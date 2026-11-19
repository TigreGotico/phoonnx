# Quickstart

This guide is for anyone who wants speech out of phoonnx as fast as possible. In about five
minutes you will install phoonnx, download a ready-made voice, and synthesize a WAV from
Python.

## 1. Install

```bash
pip install --pre phoonnx
```

phoonnx ships as alpha releases, so the `--pre` flag is necessary. Without it, pip installs
an old stable release whose voice commands fail (see
[Which version to install](installation.md#which-version-to-install)). With uv, use
`uv pip install --prerelease=allow phoonnx`.

The base install runs voices that use grapheme or unicode tokenization, and it gives the
`phoonnx-voices` command line. Voices that need a language-specific phonemizer pull in an
extra, as step 5 shows.

## 2. Find and download a voice

Fill the local catalog:

```
$ phoonnx-voices update-cache
Clearing local voice cache...
Loading bundled voice indexes...

Cache updated successfully!
Total voices available: 3189
Total languages supported: 1816
```

List the voices for a language. The list starts like this:

```
$ phoonnx-voices list-voices --lang pt-PT
* OpenVoiceOS/phoonnx_pt-PT_miro_unicode (pt-PT)
* OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone (pt-PT)
* OpenVoiceOS/phoonnx_pt-PT_dii_tugaphone (pt-PT)
* OpenVoiceOS/pipertts_pt-PT_miro (pt-PT)
...
```

Download one voice:

```
$ phoonnx-voices download OpenVoiceOS/phoonnx_pt-PT_miro_unicode
Attempting to download files for: OpenVoiceOS/phoonnx_pt-PT_miro_unicode (pt-PT)
-> Downloading ONNX model and side files (this may take a while)...

Download complete. Files saved to: ~/.cache/huggingface/hub/models--Phoonnx--phoonnx_pt-PT_miro_unicode/snapshots/eb9a020da900534d7232f11a6cf5d1ae1c545219
```

The download is about 60 MB. The Hugging Face client can also print a warning about
unauthenticated requests. You can ignore it.

`download` fetches the model with `huggingface_hub` into the shared Hugging Face cache
(`~/.cache/huggingface/hub/` by default). The model and its config go into the voice's hub
snapshot folder.

The full command set is in the [CLI reference](cli.md). The [Voice manager](voice_manager.md)
guide covers browsing and caching from Python.

## 3. Synthesize from Python

The voice manager loads a downloaded voice by its id and finds all of its files for you.
Save this as `hello.py`:

```python
import wave
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.load()
manager.merge_default_voices()

voice = manager.voices["OpenVoiceOS/phoonnx_pt-PT_miro_unicode"].load()

with wave.open("hello.wav", "wb") as wav_file:
    voice.synthesize_wav("Olá, isto é um teste.", wav_file)
```

Run it, then check the file:

```
$ python hello.py
$ python -c "import wave; w = wave.open('hello.wav'); print(w.getframerate(), w.getnchannels(), w.getsampwidth())"
22050 1 2
```

`hello.wav` is a mono 16-bit PCM file at the voice's sample rate, 22050 Hz for this voice.
Play it with any audio player.

## 4. Tune the output

`SynthesisConfig` controls speed, variation and volume for each call. Add this to the end of
`hello.py`:

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

With `length_scale=1.1` the same sentence takes about 1.9 seconds instead of about 1.75. The
length changes a little from run to run, because the voice adds random variation. Every
setting is described in [Usage](usage.md) and the [Configuration reference](configuration.md).

## 5. Use a voice that needs an extra

Many voices need a language-specific phonemizer. The `tugaphone` voices for European
Portuguese need the `pt` extra. Without it, the voice load stops:

```
ImportError: tugaphone is required for the Tugaphone phonemizer. Install it with 'pip install tugaphone' (or 'pip install scriptconv[pt-phonemizers]').
```

Install the extra and download the voice:

```bash
pip install --pre "phoonnx[pt]"
phoonnx-voices download OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone
```

Then change the voice id in `hello.py` and run it again:

```python
voice = manager.voices["OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone"].load()
```

To find the extra for another voice, see
[Which extra a voice needs](installation.md#which-extra-a-voice-needs).

## Load a voice you trained

If you have a `model.onnx` and `model.json` on disk (for example a voice you trained), load
them directly:

```python
import wave
from phoonnx.voice import TTSVoice

voice = TTSVoice.load("model.onnx", "model.json")
with wave.open("hello.wav", "wb") as wav_file:
    voice.synthesize_wav("Hello world!", wav_file)
```

`TTSVoice.load(model_path)` without a config path looks for `<model_path>.json`, for example
`model.onnx.json`. When no config file exists at that path, `TTSVoice.load` raises
`phoonnx.errors.VoiceLoadError`.

Do not load a downloaded voice this way. The file names in the hub cache change from voice
to voice: one voice holds `miro_pt-PT.onnx` and `miro_pt-PT.json`, another holds
`miro_en-US.onnx` and `miro_en-US.onnx.json`. `manager.voices[voice_id].load()` finds the
right files.

## Common problems

- **`No voices found in cache. Run 'update-cache' first.`** Run `phoonnx-voices update-cache`
  before `list-voices` or `download`, or call `manager.merge_default_voices()` in Python.
- **`Voice ID '...' not found.`** Check the exact id with `phoonnx-voices list-available`
  (bundled ids grouped by source, no download).
- **No sound or silence.** Make sure that the model downloaded fully. A partial `model.onnx`
  fails to load. Run `download` again.

[Errors you may see](installation.md#errors-you-may-see) lists the Python errors for a missing
extra, an unknown voice id, a language with no voice, and a voice that is not downloaded.

## Next steps

- Run the voice inside a voice assistant: [OVOS plugin](ovos_plugin.md).
- Clone a voice from a reference clip: [Cloning](cloning.md).
- Build your own voice: [Training quickstart](training/quickstart.md).
