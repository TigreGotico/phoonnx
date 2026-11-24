# Mixer-TTS Engine

<!-- agent-hint
covers: how to load an English or Arabic Mixer-TTS voice, pair it with a vocoder, and train one
source: phoonnx/engines/mixertts.py, phoonnx/engines/mixertts_config.py, phoonnx_train/engines/mixer.py, phoonnx/voice_index/mixertts.json
generated: no
verify: pytest tests/test_mixertts.py tests/test_mixertts_training.py; the preset table: none, checked by reading phoonnx_train/engines/mixer.py:_QUALITY_PRESETS
see: fastpitch.md, ../training.md, ../../vocoders.md
-->

This page explains how to load and train Mixer-TTS voices, for voice builders and integrators.

> Related: [training reference](../training.md) ·
> [adapter architecture](../../engines.md) ·
> [configuration](../../configuration.md) · [vocoders](../../vocoders.md)

## What it is

[Mixer-TTS](https://arxiv.org/abs/2110.03584) (NVIDIA) is a non-autoregressive,
**MLP-Mixer / FastPitch-style** acoustic model: text → 80-channel mel. Like
Matcha-TTS and GlowTTS it is **two-stage** — a separate vocoder turns the mel
into a waveform — so the adapter reuses
[`phoonnx.engines.vocoders`](../../vocoders.md).

## When to pick it

Choose Mixer-TTS for a compact, fast English or Arabic acoustic model with
explicit pace / pitch / emotion control, paired with a universal Vocos vocoder.
It shares its inference contract with [FastPitch](fastpitch.md).

## Extras needed

Inference needs no engine-specific extra beyond a phonemizer (espeak for
English, `mantoq` for Arabic). Training needs `pip install --pre "phoonnx[train]"`. The
default F0 extractor, `librosa.pyin`, comes with `[train]`. The `train-mixer` extra exists but
adds no package. The WORLD extractors (`--f0-method dio` or `harvest` at preprocess) need
`phoonnx[train,train-pyworld]`.

## Inference contract

### ONNX inputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `token_ids` | int64 | `[B, T]` | IPA symbol ids (espeak) |
| `pace` | float32 | `[1]` | speaking rate (>1 faster) |
| `speaker` | int32 | `[1]` | speaker id (single-speaker LJ = 0) |
| `emotion` | int32 | `[1]` | emotion id (0) |
| `pitch_mul` | float32 | `[1]` | pitch multiplier |
| `pitch_add` | float32 | `[1]` | pitch offset |

### ONNX output

A mel spectrogram `mel_spec [B, 80, T]` (the HiFi-GAN 80-channel mel).

Parameters:

| Param | Default | Description |
|-------|---------|-------------|
| `pace` | 1.0 | Speaking rate |
| `speaker` | 0 | Speaker id (multi-speaker models) |
| `emotion` | 0 | Emotion id |
| `pitch_mul` | 1.0 | Pitch scale |
| `pitch_add` | 0.0 | Pitch offset (semitone-ish) |

## Obtaining / training

### Loading indexed voices

Mixer-TTS voices ship in `phoonnx/voice_index/mixertts.json`, mirrored under
`Phoonnx/phoonnx-mixertts`. The reference models are the
[nipponjo/mixer-tts-pytorch](https://github.com/nipponjo/mixer-tts-pytorch)
LJSpeech checkpoints (1.74M / 3.17M / 20.6M params), 22 kHz English.

```python
from phoonnx.model_manager import TTSModelManager
m = TTSModelManager(); m.merge_default_voices()
voice = m.voices["nipponjo/mixer-tts-ljspeech-384"].load()
for chunk in voice.synthesize("Hello from Mixer-TTS."): ...
```

Mixer-TTS uses a fixed symbol table (`[pad] + punctuation + letters + IPA`, 178
symbols) and espeak IPA input.
`phoonnx.engines.mixertts_config.voice_config_from_mixer` builds a native
`VoiceConfig` from that ordered list; the mirrored voices ship it as a native
`config.json` (`engine: mixertts`).

### Arabic models

The [tts_arabic](https://github.com/nipponjo/tts_arabic) Mixer-TTS models
(`nipponjo/tts-arabic-mixer80` / `mixer128`) are **multi-speaker** Arabic voices
using a 44-symbol **buckwalter** phoneme table. They tokenize with phoonnx's
`mantoq` Arabic phonemizer (`phoneme_type: mantoq`, `alphabet: buckwalter`,
`_+_` word separator) — the same phonemes as tts_arabic's
`phonetise_buckwalter` (a golden test guards this). Input may be vocalized
Arabic or buckwalter transliteration; pick a speaker with
`SynthesisConfig(speaker=...)`.

### Training

Trainable with `--engine mixer` (alias `mixertts`). The vendored NeMo model port
lives in `phoonnx_train/mixertts/models` (Apache-2.0 headers, with the
speaker/emotion/energy conditioning and optional LSGAN mel-patch refinement from
nipponjo/tts-arabic-pytorch), wrapped by the Lightning module in
`phoonnx_train/mixertts/lightning.py` — mel / duration / pitch / energy MSE +
ForwardSum (CTC) aligner loss + delayed, ramped binarization loss, exactly the
NeMo recipe (AdamW + Noam warmup). `"train_gan": true` in the `engine_params` of the
dataset's `config.json` turns on the (non-NeMo) LSGAN `PatchDiscriminator` refinement.

It shares the standard phoonnx pipeline and the FastPitch engine's F0 sidecars. Preprocess
extracts F0 at the mel hop, from the same trimmed and normalized cached audio as the mels,
with the `--f0-method` extractor (default `pyin`):

```bash
python -m phoonnx_train.train --dataset-dir /path/to/prep \
    --engine mixer --quality medium     # x-low=80 / medium=128 / high=384 dims
python -m phoonnx_train.export_onnx \
    lightning_logs/version_0/checkpoints/epoch=999-step=250000.ckpt \
    -c /path/to/prep/config.json --engine mixer
```

Export produces the inference contract above (mel_fmin 0 / mel_fmax 8000
recorded in the ONNX metadata).

## Synthesis example

```python
voice.synthesize("Hello from Mixer-TTS.",
                 SynthesisConfig(extra={"pace": 1.1}))
```

## Vocoder

The 80-channel mel is the HiFi-GAN-compatible mel, so the indexed voices use the
**universal Vocos** vocoder (`BSC-LT/vocos-mel-22khz`, mirrored as
`Phoonnx/phoonnx-vocoders/vocos-mel-22khz-univ`) — the same one the
reference repo pairs Mixer-TTS with. Griffin-Lim works too (`spec_gain = ln(10)`
inverts the natural-log mel) as a no-model-file fallback. See
[vocoders.md](../../vocoders.md).

## Gotchas / aliases

- **Detect aliases:** the config `engine` field may be `mixertts` or
  `mixer_tts`; training uses `--engine mixer` (or `mixertts`). A graph with
  `token_ids` + `pace` + `pitch_mul` inputs, or a `mel_spec` output with
  `token_ids` input, is also auto-detected.
- **Do not confuse** the universal `vocos-mel-22khz` with
  `alvocat-vocos-22khz` — that is Vocos *finetuned on Catalan* (for the Matxa
  voices) and is a different mel domain.

## References

- [Mixer-TTS paper](https://arxiv.org/abs/2110.03584) · [nipponjo/mixer-tts-pytorch](https://github.com/nipponjo/mixer-tts-pytorch)

---

[Docs index](../../index.md) · Previous: [GlowTTS](glowtts.md) · Next: [OptiSpeech](optispeech.md)
