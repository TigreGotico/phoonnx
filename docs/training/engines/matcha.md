# Matcha-TTS Engine

This page is for voice builders and integrators who want to run or train a
Matcha-TTS voice in phoonnx. After reading it you can load a Matcha voice, pair
it with the right vocoder, and train and export your own.

> Related: [training reference](../training.md) ·
> [adapter architecture](../../engines.md) ·
> [configuration](../../configuration.md) · [vocoders](../../vocoders.md)

## What it is

Matcha-TTS is a non-autoregressive acoustic model based on **conditional flow
matching**: a Transformer text encoder, a duration predictor, and a
flow-matching decoder generate mel spectrograms from phoneme sequences. It is a
**two-stage** engine — the mel model produces a mel spectrogram, and a separate
**vocoder** turns the mel into audio. The `MatchaAdapter` also handles fused
**end-to-end** models transparently (it branches on the mel model's output
rank):

- **Two-stage** — the acoustic model outputs a mel `[B, n_mels, T]` and a
  separate vocoder ONNX reconstructs the waveform.
- **End-to-end** — a fused model (e.g. `*_wavenext_e2e.onnx`,
  `matcha_*_simply.onnx`) outputs the waveform directly. No vocoder is
  configured; the adapter returns the model output as-is.

## When to pick it

Choose Matcha for fast, high-quality non-autoregressive synthesis where you want
to swap the acoustic model and vocoder independently — for example the Catalan
Matxa voices, which pair one acoustic model with whichever vocoder is licensed
and tested for it. Prefer a fused end-to-end voice, or a Wavenext/HiFi-GAN
vocoder, when you need a commercial-safe pipeline.

## Extras needed

Matcha-TTS inference requires `scipy`: `pip install phoonnx[matcha]`. Training
uses the vendored `phoonnx_train.matcha` package and needs only the `train`
extra: `pip install phoonnx[train]`.

## Obtaining / training

### Loading a voice

Catalan Matxa voices ship in `phoonnx/voice_index/BSC.json`, mirrored under
`OpenVoiceOS/`. They load like any other voice.

End-to-end (single ONNX, no vocoder):

```python
from phoonnx import TTSVoice

voice = TTSVoice.load(model_path="matxa_v2_graphemes_10_steps_wavenext.onnx")
```

Two-stage (acoustic mel model + separate vocoder), via the JSON config:

```json
{
    "engine": "matcha",
    "engine_params": {
        "vocoder_path": "mel_spec_22khz_cat.onnx",
        "vocoder_type": "vocos",
        "vocoder_config": { "n_fft": 1024, "hop_length": 256 }
    }
}
```

The voice manager wires this automatically: index entries carry `vocoder_url`
and `vocoder_type`, the separate vocoder is downloaded alongside the model, and
`engine_params.vocoder_path` is set to the local file.

Indexed voices:

| voice_id | input | vocoder | license |
|----------|-------|---------|---------|
| `OpenVoiceOS/matxa-cat-multispeaker-wavenext` | espeak ca | fused Wavenext (e2e) | gpl-3.0 |
| `OpenVoiceOS/matxa-cat-multispeaker-hifigan` | espeak ca | fused HiFi-GAN (e2e) | gpl-3.0 |
| `OpenVoiceOS/matxa-cat-multiaccent-wavenext` | espeak ca | fused Wavenext (e2e) | gpl-3.0 |
| `OpenVoiceOS/matxa-cat-central-graphemes-v2` | graphemes | fused Wavenext (e2e) | apache-2.0 |
| `OpenVoiceOS/matxa-cat-multispeaker-wavenext-2stage` | espeak ca | Wavenext (separate) | gpl-3.0 + apache-2.0 |
| `OpenVoiceOS/matxa-cat-multispeaker-vocos-2stage` | espeak ca | alVoCat Vocos (separate) | gpl-3.0 + **cc-by-nc-4.0** |

Prefer Wavenext or a fused end-to-end voice for commercial use; the alVoCat-Vocos
two-stage voice is non-commercial.

### Training

Matcha trains on the standard phoonnx preprocessed dataset (`dataset.jsonl`);
mel statistics are computed once and cached as `matcha_stats.json` next to the
dataset.

```bash
python -m phoonnx_train.train \
    --engine matcha \
    --dataset-dir ./dataset \
    --output-dir ./checkpoints
```

Quality presets:

| Preset | Encoder channels | Decoder channels | Heads / Layers |
|--------|-----------------|------------------|----------------|
| `x-low` | 128 | [192, 192] | 2 heads / 4 layers |
| `medium` | 192 | [256, 256] | 2 heads / 6 layers |
| `high` | 256 | [384, 384] | 4 heads / 8 layers |

Use `--quality <preset>` or set `quality` in `extra` config.

Architecture: a Transformer text encoder with RoPE positional embeddings,
ConvReluNorm prenet and duration predictor; a conditional flow-matching decoder
with a UNet1D estimator (ResNet + Transformer/Conformer blocks); trained with
duration (MSE), prior (Gaussian) and flow-matching (velocity-field MSE) losses.

Export the mel model to ONNX:

```bash
python -m phoonnx_train.export_onnx \
    --engine matcha \
    --checkpoint matcha.ckpt \
    --output-dir ./onnx
```

The exported ONNX file is the **mel model only** — you still need a separate
vocoder ONNX for full synthesis; the adapter chains both automatically.

Matcha normalizes mels using dataset statistics. Compute them beforehand:

```bash
python -m matcha.utils.generate_data_statistics \
    --config configs/data/your_dataset.yaml
```

Then pass `mel_mean` / `mel_std` via engine config `extra`:

```json
{
    "extra": { "mel_mean": -5.536622, "mel_std": 2.116101 }
}
```

## Synthesis example

```python
from phoonnx.model_manager import TTSModelManager

m = TTSModelManager(); m.merge_default_voices()
voice = m.voices["OpenVoiceOS/matxa-cat-multispeaker-wavenext"].load()
for chunk in voice.synthesize("Bon dia."):
    ...  # chunk.audio_float_array
```

Parameters:

| Param | Default | Description |
|-------|---------|-------------|
| `temperature` | 0.667 | Sampling temperature for the flow-matching decoder |
| `length_scale` | 1.0 | Speech rate multiplier (higher = slower) |
| `denoise` | true | Spectral-subtraction denoising — Vocos only; ignored by raw-waveform/end-to-end vocoders |

### Vocoders

`vocoder_type` selects from the pluggable vocoder registry
(`phoonnx.engines.vocoders`), which has **six** registered values — see
[vocoders.md](../../vocoders.md) for the full system:

| `vocoder_type` | Family | ONNX output | Reconstruction |
|----------------|--------|-------------|----------------|
| `vocos` | Vocos / alVoCat | STFT mag + real + imag (3 tensors) | inverse STFT overlap-add (+ optional denoise) |
| `wavenext` | Wavenext | raw waveform (1 tensor) | none — Vocos with the ISTFT head replaced by a trained linear layer |
| `hifigan` | HiFi-GAN | raw waveform (1 tensor) | none |
| `melgan` | MelGAN / multiband-MelGAN | raw waveform (1 tensor) | none |
| `griffinlim` | Griffin-Lim | none (parametric) | phase reconstruction from the mel, no model file |
| `raw` | generic 1-output ONNX | raw waveform (1 tensor) | none |

If `vocoder_type` is omitted it is auto-detected from the vocoder ONNX
(3 outputs → Vocos, 1 output → raw waveform).

Tested vocoders for the Catalan Matxa models (all 22.05 kHz, 80-bin mel):

| Vocoder | Repo / file | License |
|---------|-------------|---------|
| Wavenext | `BSC-LT/wavenext-mel` → `mel_spec_22khz_wavenext.onnx` | Apache-2.0 (commercial-safe) |
| alVoCat (Vocos) | `projecte-aina/alvocat-vocos-22khz` → `mel_spec_22khz_cat.onnx` | **CC-BY-NC-4.0 (non-commercial)** |

## Gotchas / aliases

- **Detect aliases:** the config `engine` field may be `matcha` or `matcha-tts`.
- **Blank interspersing:** the tokenizer must set `blank_id = 0` so blanks are
  already present in the phoneme sequence fed to ONNX. The adapter does **not**
  re-intersperse — it passes the tokenized sequence directly to the mel model.
- **Grapheme models** (e.g. `matxa-tts-v2-ca-central-graphemes`) have no
  phonemes: they run the pass-through graphemes/unicode phonemizer
  (`GraphemePhonemizer` / `UnicodeCodepointPhonemizer`) and the tokenizer maps
  characters → IDs. Set `phoneme_type: graphemes` (or `unicode`) /
  `alphabet: unicode`.
- **Symbol handling lives entirely in the tokenizer**, so `TTSVoice.load`
  reproduces the model's exact token IDs with no per-model code (espeak voices
  use the 178-symbol Matcha table; the grapheme voice its 199-symbol char table).

## References

- [Matcha-TTS paper (ICASSP 2024)](https://arxiv.org/abs/2309.03199)
- [Upstream repository](https://github.com/shivammehta25/Matcha-TTS)
