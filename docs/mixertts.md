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

## Arabic models

The [tts_arabic](https://github.com/nipponjo/tts_arabic) Mixer-TTS models
(``nipponjo/tts-arabic-mixer80`` / ``mixer128``) are **multi-speaker** Arabic
voices using a 44-symbol **buckwalter** phoneme table. They tokenize with
phoonnx's ``mantoq`` Arabic phonemizer (``phoneme_type: mantoq``,
``alphabet: buckwalter``, ``_+_`` word separator) — which produces the same
phonemes as tts_arabic's ``phonetise_buckwalter`` (a golden test guards this).
Input may be vocalized Arabic or buckwalter transliteration; pick a speaker with
``SynthesisConfig(speaker_id=...)``.

## Vocoder

The 80-channel mel is the HiFi-GAN-compatible mel, so the indexed voices use the
**universal Vocos** vocoder (``BSC-LT/vocos-mel-22khz``, mirrored as
``OpenVoiceOS/phoonnx-vocoders/vocos-mel-22khz-univ``) — the same one the
reference repo pairs Mixer-TTS with. Griffin-Lim works too (``spec_gain = ln(10)``
inverts the natural-log mel) as a no-model-file fallback. See
[vocoders.md](./vocoders.md).

> Do not confuse it with ``alvocat-vocos-22khz`` — that is Vocos *finetuned on
> Catalan* (for the Matxa voices) and is a different mel domain.

## Training

Mixer-TTS training is a registered **training engine** (``--engine mixer``,
alias ``mixertts``): the vendored NeMo model port lives in
``phoonnx_train/mixertts/models`` (Apache-2.0 headers, with the
speaker/emotion/energy conditioning and optional LSGAN mel-patch refinement
from nipponjo/tts-arabic-pytorch), wrapped by the Lightning module in
``phoonnx_train/mixertts/lightning.py`` — mel / duration / pitch / energy MSE
+ ForwardSum (CTC) aligner loss + delayed, ramped binarization loss, exactly
the NeMo recipe (AdamW + Noam warmup). ``extra["train_gan"] = True`` enables
the (non-NeMo) LSGAN ``PatchDiscriminator`` refinement.

It shares the standard phoonnx pipeline and the FastPitch engine's F0
sidecars (pyworld DIO+StoneMask at the mel hop, on the same
trimmed/normalized cached audio the mels come from):

```bash
python -m phoonnx_train.train --dataset-dir /path/to/prep \
    --engine mixer --quality medium     # x-low=80 / medium=128 / high=384 dims
python -m phoonnx_train.export_onnx ckpt.ckpt -c config.json --engine mixer
```

Needs the ``train`` + ``train-mixer`` extras
(``pip install phoonnx[train,train-mixer]``): ``pyworld`` for F0 on top of
torch / lightning / einops / numba. Export produces the inference contract
above (mel_fmin 0 / mel_fmax 8000 recorded in the ONNX metadata).

## References

- [Mixer-TTS paper](https://arxiv.org/abs/2110.03584) · [nipponjo/mixer-tts-pytorch](https://github.com/nipponjo/mixer-tts-pytorch)
- [docs/engines.md](./engines.md) · [docs/vocoders.md](./vocoders.md)
