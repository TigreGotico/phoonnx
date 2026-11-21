[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TigreGotico/phoonnx)

<img src="thumb.png" alt="phoonnx" width="480"/>

# phoonnx

<!-- agent-hint
covers: what phoonnx is, how to install it and speak a first sentence, and where each part of the docs lives
source: pyproject.toml, phoonnx/model_manager.py:TTSModelManager, phoonnx/voice.py:TTSVoice, phoonnx/opm.py, phoonnx/engines/__init__.py:list_engines, docker-compose.yml
generated: no
verify: the Python block under "Quickstart", run in a fresh venv after the two phoonnx-voices commands; python -c "import phoonnx.engines as e; print(len(e.list_engines()))" prints 31
see: docs/index.md, docs/quickstart.md, AGENTS.md
-->

phoonnx turns text into speech with ONNX voices. It loads voices from several ecosystems
(native phoonnx, Piper, Mimic3, Coqui, MMS, Transformers and others) through one
`onnxruntime` interface, and it ships a pipeline to train your own voices.

The bundled voice index holds 3189 voices for 1816 language tags.

The [documentation index](docs/index.md) lists every page. To start, follow the
[Quickstart](docs/quickstart.md).

## Quickstart

```bash
pip install --pre phoonnx
```

phoonnx ships as alpha releases, so the `--pre` flag is necessary. Without it, pip installs an
old stable release whose voice commands fail.

Download a voice from the command line:

```bash
# fill the local voice catalog, then fetch one voice
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

This voice runs on the base install. Many other voices need an extra for their phonemizer, for
example `pip install --pre "phoonnx[pt]"` for the `tugaphone` voices.
[Choose a voice](docs/choose_voice.md) shows which extra each voice needs.

The [Quickstart](docs/quickstart.md) gives the same steps with the output you should see.

## Train your own voice

Record or obtain a single-speaker dataset, preprocess it, train a VITS model on one GPU,
export it to ONNX, and load it back. The bundled `phoonnx_train` toolkit does each step. The
[Training quickstart](docs/training/quickstart.md) walks through them.

## Features

| Capability | Where |
|---|---|
| Run ONNX voices on CPU or GPU (CUDA/ROCm/DirectML/CoreML/OpenVINO) | [Usage](docs/usage.md) · [Configuration](docs/configuration.md) |
| About 50 phonemizer backends across many languages | [Phonemizers](docs/phonemizers.md) |
| Load Piper / Mimic3 / Coqui / Transformers / MMS voices | [Architecture](docs/architecture.md) |
| 31 synthesis engines behind one adapter registry | [Engines](docs/engines.md) |
| Zero-shot voice cloning (YourTTS, StyleTTS2, ZipVoice, F5-TTS, Chatterbox) | [Cloning](docs/cloning.md) |
| Low-latency streaming for VITS voices | [Streaming](docs/streaming.md) |
| Optional 48 kHz audio super-resolution over synthesized audio | [OVOS plugin](docs/ovos_plugin.md#audio-super-resolution) |
| Download and cache voices from HuggingFace and other sources | [Voice manager](docs/voice_manager.md) |
| Smaller int8 files for some large voices | [Precision variants](docs/precision.md) |
| Speak with no network once the voices are downloaded | [Use phoonnx offline](docs/offline.md) |
| Train and fine-tune voices | [Training](docs/training/quickstart.md) |
| OpenVoiceOS TTS plugin | [OVOS plugin](docs/ovos_plugin.md) |

## Documentation

The [documentation index](docs/index.md) lists every page in reading order, grouped by task:
start, voices, use, configure, deploy, train and reference. Coding agents that work on this
repository read [AGENTS.md](AGENTS.md).

## OpenVoiceOS plugin

`phoonnx` ships the `ovos-tts-plugin-phoonnx` TTS plugin. Configure it in `mycroft.conf`:

```json
"tts": {
  "module": "ovos-tts-plugin-phoonnx",
  "ovos-tts-plugin-phoonnx": {
    "voice": "OpenVoiceOS/phoonnx_pt-PT_miro_unicode"
  }
}
```

Full options in [docs/ovos_plugin.md](docs/ovos_plugin.md).

## Docker / TTS server

From a checkout of this repository:

```bash
docker compose up
```

The compose file runs the `ghcr.io/tigregotico/phoonnx:dev` image, which is built from the
`dev` branch. See [docs/docker.md](docs/docker.md). For running it as a public or long-lived
service, see [docs/deployment.md](docs/deployment.md).

## Related projects

`phoonnx` is part of the [OpenVoiceOS](https://github.com/OpenVoiceOS) ecosystem:

- [ovos-core](https://github.com/OpenVoiceOS/ovos-core): the voice assistant that loads this plugin
- [ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager): the plugin framework `ovos-tts-plugin-phoonnx` registers against
- [ovos-tts-server](https://github.com/OpenVoiceOS/ovos-tts-server): a standalone HTTP server for any OVOS TTS plugin, including this one

## License and credits

`phoonnx` is licensed under Apache-2.0. Copyright Casimiro Ferreira.

It builds on the work of others, including [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
(the VITS backbone), and interoperates with the Piper, Mimic3, Coqui, MMS and Transformers
voice formats. Language-specific components are credited on the [Phonemizers](docs/phonemizers.md)
page.

The SuperTonic inference code (`phoonnx/engines/supertonic.py`) is adapted from Supertone
Inc.'s MIT-licensed reference implementation. **SuperTonic model weights
(`Supertone/supertonic-3`) are licensed OpenRAIL-M, not Apache-2.0**. Review the model's
license before commercial use.

---

[Docs index](docs/index.md) · Next: [Installation](docs/installation.md)
