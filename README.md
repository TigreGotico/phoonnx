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

The [documentation index](docs/index.md) lists every page in reading order. To start, follow
the [Quickstart](docs/quickstart.md).

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

## Extras

Many voices need a phonemizer or an engine library that the base install does not include.
Install the extra that a voice needs, for example `pip install --pre "phoonnx[pt]"` for the
`tugaphone` voices of European Portuguese. [Choose a voice](docs/choose_voice.md) shows the
extra of each voice.

| Extras | What they add |
|---|---|
| A language code, such as `pt`, `en`, `ar`, `ja` or `zh` | The phonemizers of that language |
| `espeak`, `o2i` | Phonemizers that serve many languages |
| `cjk`, `chatterbox-multilingual` | Japanese and Chinese script tools (`pykakasi`, `spacy-pkuseg`) |
| `streaming`, `matcha`, `pockettts`, `omnivoice`, `indic-parler` | The libraries that one engine or feature needs |
| `cloning`, `vc`, `audiosr` | Reference-clip cloning, voice conversion and audio super-resolution |
| `train` and the `train-*` extras | The training toolkit and its optional parts |
| `all` | Most phonemizers, the engine libraries, cloning, voice conversion and super-resolution. The Docker image installs it. |

[Installation](docs/installation.md) describes each extra and the GPU runtimes.

## Features

| Capability | Page |
|---|---|
| Run ONNX voices on CPU or GPU (CUDA, ROCm, DirectML, CoreML, OpenVINO) | [Usage](docs/usage.md), [Configuration](docs/configuration.md) |
| Phonemizer backends for many languages, from the scriptconv packages | [Phonemizers](docs/phonemizers.md) |
| Load Piper, Mimic 3, Coqui, Transformers and MMS voices | [Architecture](docs/architecture.md) |
| 31 synthesis engines behind one adapter registry | [Engines](docs/engines.md) |
| Zero-shot voice cloning from a reference clip, for example with YourTTS, StyleTTS2, ZipVoice, F5-TTS or Chatterbox | [Cloning](docs/cloning.md) |
| Low-latency streaming for VITS voices | [Streaming](docs/streaming.md) |
| Optional 48 kHz audio super-resolution of the synthesized audio | [OVOS plugin](docs/ovos_plugin.md#audio-super-resolution) |
| Download and cache voices from Hugging Face and other sources | [Voice manager](docs/voice_manager.md) |
| Smaller int8 files for some large voices | [Precision variants](docs/precision.md) |
| Speak with no network after you download the voices | [Use phoonnx offline](docs/offline.md) |
| Train and fine-tune voices | [Training quickstart](docs/training/quickstart.md) |
| OpenVoiceOS text-to-speech plugin | [OVOS plugin](docs/ovos_plugin.md) |

## Train your own voice

The `phoonnx_train` toolkit preprocesses a single-speaker dataset, trains a VITS model on one
GPU and exports it to ONNX. phoonnx then loads the result as a voice. The
[Training quickstart](docs/training/quickstart.md) gives each step.

## OpenVoiceOS plugin

phoonnx includes the `ovos-tts-plugin-phoonnx` text-to-speech plugin. Configure it in
`mycroft.conf`:

```json
"tts": {
  "module": "ovos-tts-plugin-phoonnx",
  "ovos-tts-plugin-phoonnx": {
    "voice": "OpenVoiceOS/phoonnx_pt-PT_miro_unicode"
  }
}
```

[OVOS plugin](docs/ovos_plugin.md) lists every option.

## Docker text-to-speech server

From a checkout of this repository, run:

```bash
docker compose up
```

The compose file runs the `ghcr.io/tigregotico/phoonnx:dev` image. The `docker.yml` workflow builds
that image from the `dev` branch. [Docker TTS server](docs/docker.md) describes the image, and
[Deployment](docs/deployment.md) describes how to run it as a long-lived service.

## Documentation

The [documentation index](docs/index.md) groups the pages by task: start, voices, use,
configure, deploy, train and reference. Coding agents that work on this repository read
[AGENTS.md](AGENTS.md).

## Related projects

phoonnx is part of the [OpenVoiceOS](https://github.com/OpenVoiceOS) ecosystem:

- [ovos-core](https://github.com/OpenVoiceOS/ovos-core): the voice assistant that loads this plugin.
- [ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager): the plugin framework that `ovos-tts-plugin-phoonnx` registers with.
- [ovos-tts-server](https://github.com/OpenVoiceOS/ovos-tts-server): an HTTP server for any OVOS text-to-speech plugin, this one included.

## License and credits

phoonnx is licensed under Apache-2.0. Copyright TigreGotico Lda.

phoonnx uses the VITS model of [jaywalnut310/vits](https://github.com/jaywalnut310/vits), and it
reads the Piper, Mimic 3, Coqui, MMS and Transformers voice formats. The
[Phonemizers](docs/phonemizers.md) page credits the language components.

The SuperTonic inference code in `phoonnx/engines/supertonic.py` is adapted from the
MIT-licensed reference implementation of Supertone Inc. The SuperTonic model weights
(`Supertone/supertonic-3`) have the OpenRAIL license, not Apache-2.0. Read the model license
before commercial use.

---

Next: [Docs index](docs/index.md)
