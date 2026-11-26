[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TigreGotico/phoonnx)

<img src="thumb.png" alt="phoonnx" width="480"/>

# phoonnx

<!-- agent-hint
covers: what phoonnx is, how to install it and speak a first sentence, the extras, and where each part of the docs lives
source: pyproject.toml, phoonnx/model_manager.py:TTSModelManager, phoonnx/voice.py:TTSVoice.synthesize_wav, phoonnx/engines/__init__.py:list_engines, phoonnx/voice_index/OVOS.json, docker-compose.yml, LICENSE
generated: no
verify: .venv/bin/python -I -c "import re, sys, json, glob, tomllib; import phoonnx.engines as e; t = open('README.md').read().split(chr(10) + '-' + '->', 1)[1]; ids = set(); [ids.update(json.load(open(f))) for f in glob.glob('phoonnx/voice_index/*.json')]; extras = tomllib.load(open('pyproject.toml', 'rb'))['project']['optional-dependencies']; used = set(re.findall(r'phoonnx-voices download (\S+)', t) + re.findall(r'voices\[\"([^\"]+)\"\]', t) + re.findall(r'\"voice\": \"([^\"]+)\"', t)); named = set(re.findall(r'phoonnx\[([a-z0-9-]+)\]', t)); bad = sorted(used - ids) + sorted(named - set(extras)) + ([] if f'{len(e.list_engines())} synthesis engines' in t else ['engine count']) + ([] if open('docker-compose.yml').read().split('image: ')[1].split()[0] in t else ['image']); print(bad); sys.exit(bool(bad))"
see: docs/index.md, docs/quickstart.md, docs/installation.md, AGENTS.md
-->

phoonnx turns text into speech with ONNX voices. It loads voices from many ecosystems, such as
native phoonnx, Piper, Mimic 3, Coqui, MMS and Transformers, through one ONNX Runtime
interface, and it includes a toolkit to train your own voices.

## Install

```bash
pip install --pre phoonnx
```

phoonnx publishes alpha releases, so you must use the `--pre` flag. Without it, pip installs
the last stable release, which does not have the engine registry or most of the features on
these pages.

## Quickstart

Fill the local voice catalog, then download one voice:

```bash
phoonnx-voices update-cache
phoonnx-voices download OpenVoiceOS/phoonnx_pt-PT_miro_unicode
```

Speak a sentence with it from Python:

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

The script writes `hello.wav`. This voice reads text with no phonemizer, so the base install
is sufficient. The [Quickstart](docs/quickstart.md) gives the same steps with the output of
each command.

## What it does

| Task | Page |
|---|---|
| Install the extra that a voice needs, and set up a GPU runtime | [Installation](docs/installation.md) |
| Find a voice for a language and download it | [Choose a voice](docs/choose_voice.md), [Voice manager](docs/voice_manager.md) |
| Synthesize to a file or a stream, on CPU or GPU | [Usage](docs/usage.md), [Streaming](docs/streaming.md) |
| Run 31 synthesis engines behind one adapter registry | [Engines](docs/engines.md) |
| Clone a speaker from a reference clip | [Cloning](docs/cloning.md) |
| Speak with no network | [Use phoonnx offline](docs/offline.md) |
| Train and export your own voice | [Training quickstart](docs/training/quickstart.md) |
| Use phoonnx as the OpenVoiceOS text-to-speech plugin | [OVOS plugin](docs/ovos_plugin.md) |
| Run the `ghcr.io/tigregotico/phoonnx:dev` image as an HTTP server | [Docker TTS server](docs/docker.md), [Deployment](docs/deployment.md) |

The [documentation index](docs/index.md) lists every page in reading order.

## License

phoonnx is licensed under Apache-2.0. Copyright TigreGotico Lda. Some model weights have their own license, for example the OpenRAIL license of SuperTonic: see [SuperTonic](docs/training/engines/supertonic.md). The [Phonemizers](docs/phonemizers.md) page credits the language components.

---

[Home](docs/index.md)
