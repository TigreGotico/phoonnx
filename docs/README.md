# phoonnx documentation

**phoonnx** is a multilingual, ONNX-based Text-to-Speech and phonemization library. This
index is the front door: pick the path that matches what you want to do, or jump to the full
table of contents below.

## Three paths

### Use a voice (about 5 minutes)
For anyone who wants speech out of an existing voice.

1. [Installation](installation.md) — pip, the extras matrix, GPU runtimes
2. [Quickstart](quickstart.md) — download a voice and synthesize a WAV, CLI and Python
3. [Usage](usage.md) — the `TTSVoice` / `SynthesisConfig` Python API
4. [Voice manager](voice_manager.md) — finding, downloading and caching voices
5. [CLI reference](cli.md) — every `phoonnx-voices` subcommand

### Train a voice (the golden path)
For anyone with a dataset who wants a new voice.

1. [Training quickstart](training/quickstart.md) — dataset → preprocess → train → export → speak
2. [Datasets](training/datasets.md) — `metadata.csv` spec, audio requirements, quality filtering
3. [Preprocess reference](training/preprocess.md) — every `preprocess.py` flag
4. [Training reference](training/training.md) — every `train.py` flag, engines, resume/fine-tune
5. [Export](training/export.md) — checkpoint → ONNX, validating the exported voice

### Understand the internals (contributors and advanced users)
For anyone extending phoonnx or debugging a voice.

1. [Architecture](architecture.md) — text → normalize → phonemize → tokenize → ONNX → audio
2. [Phonemizers](phonemizers.md) — the backend catalog and how selection works
3. [Engines](engines.md) — the ONNX adapter registry
4. [Vocoders](vocoders.md) — the vocoder registry for two-stage engines
5. [Configuration reference](configuration.md) — `VoiceConfig`, `SynthesisConfig`, `model.json`

## Task guides

- [Voice cloning](cloning.md) — zero-shot cloning from a reference clip
- [Streaming](streaming.md) — low-latency streaming for VITS voices
- [Phoneme alignment](alignment.md) — per-phoneme timing for visemes, karaoke, subtitles
- [OVOS plugin](ovos_plugin.md) — the `ovos-tts-plugin-phoonnx` TTS plugin
- [Docker / TTS server](docker.md)

## Per-engine guides

Each synthesis engine has a page covering what it is, when to pick it, the extra it needs,
how to obtain or train it, and a synthesis example:

[Matcha](training/engines/matcha.md) ·
[GlowTTS](training/engines/glowtts.md) ·
[MixerTTS](training/engines/mixertts.md) ·
[OptiSpeech](training/engines/optispeech.md) ·
[FastPitch](training/engines/fastpitch.md) ·
[ZipVoice](training/engines/zipvoice.md) ·
[Chatterbox](training/engines/chatterbox.md) ·
[F5-TTS](training/engines/f5tts.md) ·
[Shami](training/engines/shami.md) ·
[Pocket TTS](pockettts.md)

## Language notes

- [Galician (Cotovia)](galician.md)

## Reference data

- [VOICES.md](../VOICES.md) — the full generated catalog of bundled voices
