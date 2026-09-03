[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TigreGotico/phoonnx)

<img src="thumb.png" alt="phoonnx" width="480"/>

# phoonnx

Multilingual **phonemization** and **Text-to-Speech** using ONNX models. `phoonnx` loads
and runs TTS voices from several ecosystems (native phoonnx, Piper, Mimic3, Coqui, MMS,
Transformers) through one `onnxruntime`-based interface, and ships a training pipeline for
building your own voices.

It reaches **1000+ languages and voices** across the bundled voice indexes. See the full
list in [VOICES.md](./VOICES.md).

## 30-second quickstart

```bash
pip install phoonnx
```

Download a voice and synthesize a WAV from the command line:

```bash
# populate the local voice catalog, then fetch one voice
phoonnx-voices update-cache
phoonnx-voices download OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone
```

The same thing from Python:

```python
import wave
from phoonnx.voice import TTSVoice

voice = TTSVoice.load("model.onnx", "model.json")
with wave.open("hello.wav", "wb") as wav_file:
    voice.synthesize_wav("Hello world!", wav_file)
```

New here? Start with the [Quickstart](docs/quickstart.md) — it downloads a voice and
speaks a sentence end-to-end.

## Train your own voice in an afternoon

Record (or obtain) a single-speaker dataset, preprocess it, train a VITS model on one GPU,
export to ONNX, and load it back — all with the bundled `phoonnx_train` toolkit. Follow the
golden path in [docs/training/quickstart.md](docs/training/quickstart.md).

## Features

| Capability | Where |
|---|---|
| Run ONNX voices on CPU or GPU (CUDA/ROCm/DirectML/CoreML/OpenVINO) | [Usage](docs/usage.md) · [Configuration](docs/configuration.md) |
| ~40 phonemizer backends across many languages | [Phonemizers](docs/phonemizers.md) |
| Load Piper / Mimic3 / Coqui / Transformers / MMS voices | [Architecture](docs/architecture.md) |
| 14 synthesis engines behind one adapter registry | [Engines](docs/engines.md) |
| Zero-shot voice cloning (YourTTS, StyleTTS2, ZipVoice, F5-TTS, Chatterbox) | [Cloning](docs/cloning.md) |
| Low-latency streaming for VITS voices | [Streaming](docs/streaming.md) |
| Optional 48 kHz audio super-resolution over synthesized audio | [OVOS plugin](docs/ovos_plugin.md#audio-super-resolution) |
| Download and cache voices from HuggingFace and other sources | [Voice manager](docs/voice_manager.md) |
| Train and fine-tune new voices | [Training](docs/training/quickstart.md) |
| Drop-in OpenVoiceOS TTS plugin | [OVOS plugin](docs/ovos_plugin.md) |

## Documentation

The docs are organized around three reader paths — see the [documentation index](docs/README.md):

- **Use a voice** — [Installation](docs/installation.md) → [Quickstart](docs/quickstart.md) → [Usage](docs/usage.md)
- **Train a voice** — [Training quickstart](docs/training/quickstart.md) → [Datasets](docs/training/datasets.md) → [Export](docs/training/export.md)
- **Understand the internals** — [Architecture](docs/architecture.md) → [Phonemizers](docs/phonemizers.md) → [Engines](docs/engines.md)

## OpenVoiceOS plugin

`phoonnx` ships the `ovos-tts-plugin-phoonnx` TTS plugin. Configure it in `mycroft.conf`:

```json
"tts": {
  "module": "ovos-tts-plugin-phoonnx",
  "ovos-tts-plugin-phoonnx": {
    "voice": "OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone"
  }
}
```

Full options in [docs/ovos_plugin.md](docs/ovos_plugin.md).

## Docker / TTS server

```bash
docker compose up
```

See [docs/docker.md](docs/docker.md). For running it as a public or long-lived
service, see [docs/deployment.md](docs/deployment.md).

## Related projects

`phoonnx` is part of the [OpenVoiceOS](https://github.com/OpenVoiceOS) ecosystem:

- [ovos-core](https://github.com/OpenVoiceOS/ovos-core) — the voice assistant that loads this plugin
- [ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager) — the plugin framework `ovos-tts-plugin-phoonnx` registers against
- [ovos-tts-server](https://github.com/OpenVoiceOS/ovos-tts-server) — a standalone HTTP server for any OVOS TTS plugin, including this one

## License and credits

`phoonnx` is licensed under Apache-2.0. Copyright Casimiro Ferreira.

It builds on the work of others, including [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
(the VITS backbone), and interoperates with the Piper, Mimic3, Coqui, MMS and Transformers
voice formats. Language-specific components are credited on the [Phonemizers](docs/phonemizers.md)
page.

The SuperTonic inference code (`phoonnx/engines/supertonic.py`) is adapted from Supertone
Inc.'s MIT-licensed reference implementation. **SuperTonic model weights
(`Supertone/supertonic-3`) are licensed OpenRAIL-M, not Apache-2.0** — review the model's
license before commercial use.
