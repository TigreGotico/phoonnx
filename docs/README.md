# phoonnx

**phoonnx** is a multilingual, ONNX-based Text-to-Speech (TTS) inference library. It provides a unified interface to load and run VITS-style TTS models trained with a variety of phonemizers, supporting voices from multiple upstream ecosystems including Piper, Mimic3, Coqui, and OpenVoiceOS.

## Key Features

- **ONNX inference** — run TTS models with `onnxruntime` (CPU or CUDA)
- **Multilingual phonemization** — 30+ phonemizer backends for dozens of languages
- **Multi-engine support** — load voices from Piper, Mimic3, Coqui, Transformers, and native phoonnx format
- **Voice manager** — download and cache models from HuggingFace and other sources
- **Training pipeline** — preprocess datasets and train new VITS voices (`phoonnx_train`)
- **OVOS plugin** — drop-in TTS plugin for the OpenVoiceOS / Mycroft ecosystem

## Quick Start

```bash
pip install phoonnx
```

```python
from phoonnx.voice import TTSVoice, SynthesisConfig
import wave

voice = TTSVoice.load("model.onnx", "model.json")

with wave.open("output.wav", "wb") as wav_file:
    voice.synthesize_wav("Hello world!", wav_file)
```

## Project Structure

| Module | Description |
|--------|-------------|
| `phoonnx/voice.py` | Core `TTSVoice` inference class |
| `phoonnx/config.py` | `VoiceConfig`, `SynthesisConfig`, enums |
| `phoonnx/tokenizer.py` | `TTSTokenizer` and `Vocabulary` |
| `phoonnx/model_manager.py` | `TTSModelManager` and `TTSModelInfo` |
| `phoonnx/cli.py` | Command-line interface |
| `phoonnx/phonemizers/` | Phonemizer backends |
| `phoonnx_train/` | Dataset preprocessing and training |

## Documentation

- [Installation](installation.md)
- [Usage Guide](usage.md)
- [Voice Manager](voice_manager.md)
- [Phonemizers](phonemizers.md)
- [Configuration Reference](configuration.md)
- [Training](training.md)
- [CLI Reference](cli.md)
- [OVOS Plugin](ovos_plugin.md)
- [Docker / TTS Server](docker.md)
