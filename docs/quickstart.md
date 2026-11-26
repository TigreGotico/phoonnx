# Quickstart

<!-- agent-hint
covers: install phoonnx, download one voice, speak a WAV from Python, tune it, and add an extra for a voice that needs one
source: phoonnx/cli.py, phoonnx/model_manager.py:TTSModelManager, phoonnx/voice.py:TTSVoice.load, phoonnx/voice.py:TTSVoice.synthesize_wav, phoonnx/audio.py:write_wav, phoonnx/config.py:SynthesisConfig
generated: no
verify: diff <(XDG_CACHE_HOME=.venv/verify-cache HF_HUB_OFFLINE=1 .venv/bin/phoonnx-voices update-cache | sed -E 's/: [0-9]+$/: .../') <(sed -n '/^\$ phoonnx-voices update-cache$/,/^\x60\x60\x60$/{//!p}' docs/quickstart.md) && [ $(sed -n '/^\$ phoonnx-voices list-voices --lang pt-PT$/,/^\x60\x60\x60$/{//!p}' docs/quickstart.md | wc -l) -gt 3 ] && ! grep -vxF -e '...' -f <(XDG_CACHE_HOME=.venv/verify-cache HF_HUB_OFFLINE=1 .venv/bin/phoonnx-voices list-voices --lang pt-PT | sed -E 's/^Found [0-9]+ /Found ... /') <(sed -n '/^\$ phoonnx-voices list-voices --lang pt-PT$/,/^\x60\x60\x60$/{//!p}' docs/quickstart.md) && .venv/bin/python -I -c "import re, sys, json, glob, tomllib; t = open('docs/quickstart.md').read().split(chr(10) + '-' + '->', 1)[1]; ids = set(); [ids.update(json.load(open(f))) for f in glob.glob('phoonnx/voice_index/*.json')]; x = tomllib.load(open('pyproject.toml', 'rb'))['project']['optional-dependencies']; used = set(re.findall(r'phoonnx-voices download (\S+)', t) + re.findall(r'voices\[\"([^\"]+)\"\]', t)); named = set(re.findall(r'phoonnx\[([a-z0-9-]+)\]', t)); bad = sorted(used - ids) + sorted(named - set(x)); print(bad); sys.exit(bool(bad) or not used)"
see: installation.md, choose_voice.md, usage.md, offline.md
-->

This page takes you from an empty Python environment to a spoken WAV file. It is for anyone who wants speech out of phoonnx quickly.

## 1. Install

```bash
pip install --pre phoonnx
```

phoonnx ships as alpha releases, so the `--pre` flag is necessary. Without it, pip installs
an old stable release whose voice commands fail (see
[Which version to install](installation.md#which-version-to-install)). With uv, use
`uv pip install --prerelease=allow phoonnx`.

The base install runs voices that use grapheme or unicode tokenization, and it gives the
`phoonnx-voices` command line. A voice that needs a language-specific phonemizer needs an
extra, as step 5 shows.

## 2. Find and download a voice

Fill the local catalog. The command reads the voice index files that ship with phoonnx, so it
needs no network:

```
$ phoonnx-voices update-cache
Clearing local voice cache...
Loading bundled voice indexes...

Cache updated successfully!
Total voices available: ...
Total languages supported: ...
```

The last two lines give the number of voices and language tags in the bundled index.

List the voices for a language. The list starts like this:

```
$ phoonnx-voices list-voices --lang pt-PT

Found ... voices for language 'pt-PT':
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

Download complete. Files saved to: ~/.cache/huggingface/hub/models--Phoonnx--phoonnx_pt-PT_miro_unicode/snapshots/aec14a0428e6b67a47eb54b2baa4a604d42c0063
```

The download is 63 MB. The Hugging Face client can also print a warning about
unauthenticated requests. You can ignore it.

`download` puts the model and its config into the shared Hugging Face cache, which is
`~/.cache/huggingface/hub/` by default.

The full command set is in the [CLI reference](cli.md). The [Voice manager](voice_manager.md)
page shows the same tasks from Python.

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

`hello.wav` is a mono 16-bit PCM file at the sample rate of the voice. This voice has a
sample rate of 22050 Hz. Play the file with any audio player.

## 4. Tune the output

`SynthesisConfig` sets the speed, the variation and the volume of each call. Add this to the
end of `hello.py`:

```python
from phoonnx.config import AudioPost, SynthesisConfig

cfg = SynthesisConfig(
    length_scale=1.1,   # >1 slower, <1 faster
    noise_scale=0.667,  # generator noise / variation
    audio=AudioPost(volume=1.0),
)
with wave.open("hello.wav", "wb") as wav_file:
    voice.synthesize_wav("Olá, isto é um teste.", wav_file, cfg)
```

With `length_scale=1.1` the voice speaks the sentence more slowly. The length of the audio
changes a little from run to run, because the voice adds random variation.
[Usage](usage.md) and the [Configuration reference](configuration.md) describe each setting.

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

To find the extra for another voice, see [Choose a voice](choose_voice.md#2-read-what-the-voice-needs).

## Load a voice you trained

If you have a `model.onnx` and a `model.json` on disk, for example a voice that you trained,
load them directly:

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

`list-voices` prints `No voices found in cache. Run 'update-cache' first.` when the catalog
is empty. Run `phoonnx-voices update-cache`, or call `manager.merge_default_voices()` in
Python. `download` does not need the catalog: when the id is not in the catalog, `download`
looks it up in the bundled index files.

`download` prints `Error: Voice ID '...' not found.` for an id that no index file holds.
Copy the exact id from `phoonnx-voices list-available`.

A partial download of a model fails to load. Run `download` again to complete it.

With no network, a voice that is not in the cache stops with `LocalEntryNotFoundError`.
Download the voice while the machine is online. See [Use phoonnx offline](offline.md).

[Errors you may see](installation.md#errors-you-may-see) lists the Python errors for a missing
extra, an unknown voice id, a language with no voice, and a voice that is not downloaded.

---

[Docs index](index.md) · Previous: [Installation](installation.md) · Next: [Choose a voice](choose_voice.md)
