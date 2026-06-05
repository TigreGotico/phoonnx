# Mixer-TTS Engine

[Mixer-TTS](https://arxiv.org/abs/2110.03584) (NVIDIA) is a non-autoregressive,
**MLP-Mixer / FastPitch-style** acoustic model: text → 80-channel mel. Like
Matcha-TTS and GlowTTS it is **two-stage** — a separate vocoder turns the mel
into a waveform — so the adapter reuses [`phoonnx.engines.vocoders`](./vocoders.md).

## Inference

### ONNX inputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| ``token_ids`` | int64 | ``[B, T]`` | IPA symbol ids (espeak) |
| ``pace`` | float32 | ``[1]`` | speaking rate (>1 faster) |
| ``speaker`` | int32 | ``[1]`` | speaker id (single-speaker LJ = 0) |
| ``emotion`` | int32 | ``[1]`` | emotion id (0) |
| ``pitch_mul`` | float32 | ``[1]`` | pitch multiplier |
| ``pitch_add`` | float32 | ``[1]`` | pitch offset |

### ONNX output

A mel spectrogram ``mel_spec [B, 80, T]`` (the HiFi-GAN 80-channel mel).

### Parameters

| Param | Default | Description |
|-------|---------|-------------|
| ``pace`` | 1.0 | Speaking rate |
| ``pitch_mul`` | 1.0 | Pitch scale |
| ``pitch_add`` | 0.0 | Pitch offset (semitone-ish) |

## Config

Mixer-TTS uses a fixed symbol table (``[pad] + punctuation + letters + IPA``,
178 symbols) and espeak IPA input.
``phoonnx.engines.mixertts_config.voice_config_from_mixer`` builds a native
``VoiceConfig`` from that ordered list; the mirrored voices ship it as a native
``config.json`` (``engine: mixertts``).

## Voice index

Mixer-TTS voices ship in ``phoonnx/voice_index/mixertts.json``, mirrored under
``OpenVoiceOS/phoonnx-mixertts``. The reference models are the
[nipponjo/mixer-tts-pytorch](https://github.com/nipponjo/mixer-tts-pytorch)
LJSpeech checkpoints (1.74M / 3.17M / 20.6M params), 22 kHz English.

```python
from phoonnx.model_manager import TTSModelManager
m = TTSModelManager(); m.merge_default_voices()
voice = m.voices["nipponjo/mixer-tts-ljspeech-384"].load()
for chunk in voice.synthesize("Hello from Mixer-TTS."): ...
```

## Vocoder

The 80-channel mel is the HiFi-GAN natural-log mel. The indexed voices use the
parametric **Griffin-Lim** vocoder (no model file; its config carries the mel
params with ``spec_gain = ln(10)`` to invert the natural log). A matched neural
vocoder (HiFi-GAN / Vocos at the LJSpeech 22 kHz mel) can be linked per-voice via
``vocoder_url`` for higher fidelity — see [vocoders.md](./vocoders.md).

## References

- [Mixer-TTS paper](https://arxiv.org/abs/2110.03584) · [nipponjo/mixer-tts-pytorch](https://github.com/nipponjo/mixer-tts-pytorch)
- [docs/engines.md](./engines.md) · [docs/vocoders.md](./vocoders.md)
