# Matcha-TTS Engine

<!-- agent-hint
covers: how to load a Matcha voice, pair it with a vocoder, and train and export one
source: phoonnx/engines/matcha.py, phoonnx/engines/vocoders/__init__.py, phoonnx_train/engines/matcha.py, phoonnx_train/matcha/dataset.py, phoonnx/voice_index/BSC.json
generated: no
verify: .venv/bin/python -c "import re,sys,json; from phoonnx.engines.matcha import MatchaAdapter as A; from phoonnx_train.engines.matcha import _QUALITY_PRESETS as Q; p=open('docs/training/engines/matcha.md').read().replace(chr(96),''); d=dict(re.findall(r'^\| ([a-z_]+) \| ([\d.]+) \| [A-Z]',p,re.M)); q={r[0]:tuple(map(int,r[1:])) for r in re.findall(r'^\| ([\w-]+) \| (\d+) \| \[(\d+), (\d+)\] \| (\d+) / (\d+) \|',p,re.M)}; ids=set(re.findall(r'^\| (OpenVoiceOS/matxa[\w-]+) \|',p,re.M)); idx={k for k,v in json.load(open('phoonnx/voice_index/BSC.json')).items() if v['engine']=='matcha'}; sys.exit(0 if d=={k:str(v) for k,v in A().default_params().items()} and q=={k:(v['encoder_channels'],*v['decoder_channels'],v['encoder_n_heads'],v['encoder_n_layers']) for k,v in Q.items()} and ids==idx else 1)"
see: ../training.md, ../../vocoders.md, glowtts.md
-->

This page shows how to run a Matcha-TTS voice, pair it with a vocoder, and train and export your own model.

## What it is

Matcha-TTS is a non-autoregressive acoustic model that uses conditional flow matching. A Transformer text encoder, a duration predictor and a flow-matching decoder make a mel spectrogram from phoneme IDs. A Matcha voice comes in one of two forms, and `MatchaAdapter` handles both:

- A two-stage voice has a mel model and a separate vocoder. The mel model outputs a mel spectrogram `[B, n_mels, T]`, and the vocoder ONNX makes the waveform from it.
- An end-to-end voice is one fused ONNX file that outputs the waveform. It has no vocoder, and the adapter returns the model output as audio.

The adapter finds the main output by size and checks its `n_mels` axis, so the order of the output tensors does not matter.

Pick Matcha when you want to change the acoustic model and the vocoder independently. The Catalan Matxa voices are an example: one acoustic model runs with a Wavenext or a Vocos vocoder.

## Install

Inference needs `scipy`, which the `matcha` extra installs:

```bash
pip install --pre "phoonnx[matcha]"
```

Training uses the vendored `phoonnx_train.matcha` package and needs only the `train` extra: `pip install --pre "phoonnx[train]"`.

## Load a voice

The Catalan Matxa voices are in `phoonnx/voice_index/BSC.json`. They load like any other index voice:

```python
from phoonnx.model_manager import TTSModelManager

m = TTSModelManager()
m.merge_default_voices()
voice = m.voices["OpenVoiceOS/matxa-cat-multispeaker-wavenext"].load()
for chunk in voice.synthesize("Bon dia."):
    audio = chunk.audio_float_array
```

For a two-stage voice, the index entry has a `vocoder_url` and a `vocoder_type`. The voice manager downloads the vocoder with the model, and it sets `engine_params.vocoder_path` to the local file.

To load a local end-to-end model, give its paths:

```python
from phoonnx.voice import TTSVoice

voice = TTSVoice.load(model_path="matxa_v2_graphemes_10_steps_wavenext.onnx",
                      config_path="config.json")
```

A local two-stage voice names its vocoder in the `engine_params` of its `config.json`:

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

## Vocoders

`vocoder_type` selects a vocoder from the registry in `phoonnx.engines.vocoders`. The registry has six values. [Vocoders](../../vocoders.md) describes each one.

| `vocoder_type` | Family | ONNX output | Reconstruction |
|----------------|--------|-------------|----------------|
| `vocos` | Vocos, alVoCat | magnitude, real and imaginary STFT (3 tensors) | inverse STFT overlap-add, with optional denoise |
| `wavenext` | Wavenext | waveform (1 tensor) | none |
| `hifigan` | HiFi-GAN | waveform (1 tensor) | none |
| `melgan` | MelGAN, multiband MelGAN | waveform (1 tensor) | none |
| `griffinlim` | Griffin-Lim | no model file | phase reconstruction from the mel |
| `raw` | any ONNX with one output | waveform (1 tensor) | none |

If `vocoder_type` is not set, the adapter reads the vocoder ONNX. A model with three outputs is read as Vocos, and a model with one output is read as a waveform vocoder. Griffin-Lim is never selected automatically.

## Train a model

Matcha trains on the standard phoonnx preprocessed dataset (`dataset.jsonl`), the same as VITS:

```bash
python -m phoonnx_train.train \
    --engine matcha \
    --dataset-dir ./dataset \
    --default-root-dir ./checkpoints
```

Use `--quality` to pick a preset from the table below. The default preset is `medium`. A key in `engine_params` overrides the same key of the preset.

Matcha normalizes the mels with the mean and the standard deviation of the dataset. When `mel_mean` and `mel_std` are not set, the engine computes them and caches them in `matcha_stats.json` in the dataset folder. To set them yourself, put them in the `engine_params` of the dataset's `config.json`:

```json
{
    "engine_params": { "mel_mean": -5.536622, "mel_std": 2.116101 }
}
```

The model has a Transformer text encoder with RoPE positional embeddings, a ConvReluNorm prenet and a duration predictor. The decoder is a conditional flow-matching decoder with a UNet1D estimator. The losses are a duration loss (MSE), a prior loss and a flow-matching loss.

## Export to ONNX

Export the mel model:

```bash
python -m phoonnx_train.export_onnx \
    ./checkpoints/lightning_logs/version_0/checkpoints/epoch=999-step=250000.ckpt \
    --config ./dataset/config.json \
    --engine matcha \
    --output-dir ./onnx
```

The exported file is the mel model only. For audio, you also need a vocoder ONNX. Name it in `engine_params.vocoder_path`, and the adapter runs both models.

## Notes

- The config `engine` field may be `matcha` or `matcha-tts`. A config with `model_type: matcha`, or with a `vocoder_path`, also selects this adapter.
- The tokenizer sets `blank_id = 0`, so the phoneme sequence already holds the blanks. The adapter passes the sequence to the mel model as it is.
- A grapheme model, such as `OpenVoiceOS/matxa-cat-central-graphemes-v2`, has no phonemizer. Set `phoneme_type: graphemes` (or `unicode`) and `alphabet: unicode`, and the tokenizer maps each character to an ID.

## Reference

### Indexed voices

| voice_id | Input | Vocoder | Licence |
|----------|-------|---------|---------|
| `OpenVoiceOS/matxa-cat-multispeaker-wavenext` | espeak ca | fused Wavenext | gpl-3.0 |
| `OpenVoiceOS/matxa-cat-multispeaker-hifigan` | espeak ca | fused HiFi-GAN | gpl-3.0 |
| `OpenVoiceOS/matxa-cat-multiaccent-wavenext` | espeak ca | fused Wavenext | gpl-3.0 |
| `OpenVoiceOS/matxa-cat-central-graphemes-v2` | graphemes | fused Wavenext | gpl-3.0 |
| `OpenVoiceOS/matxa-cat-multispeaker-wavenext-2stage` | espeak ca | separate Wavenext | gpl-3.0, vocoder apache-2.0 |
| `OpenVoiceOS/matxa-cat-multispeaker-vocos-2stage` | espeak ca | separate alVoCat Vocos | gpl-3.0, vocoder cc-by-nc-4.0 |

The alVoCat Vocos vocoder (`projecte-aina/alvocat-vocos-22khz`) is for non-commercial use only. The Wavenext vocoder (`BSC-LT/wavenext-mel`) is Apache-2.0.

### Synthesis parameters

| Param | Default | Description |
|-------|---------|-------------|
| `temperature` | 0.667 | Sampling temperature of the flow-matching decoder |
| `length_scale` | 1.0 | Speech rate multiplier (higher is slower) |
| `denoise` | true | Spectral-subtraction denoise. Only the Vocos vocoder uses it. |

### Quality presets

| Preset | Encoder channels | Decoder channels | Heads / Layers |
|--------|-----------------|------------------|----------------|
| `x-low` | 128 | [192, 192] | 2 / 4 |
| `medium` | 192 | [256, 256] | 2 / 6 |
| `high` | 256 | [384, 384] | 4 / 8 |

## References

- [Matcha-TTS paper (ICASSP 2024)](https://arxiv.org/abs/2309.03199)
- [Upstream repository](https://github.com/shivammehta25/Matcha-TTS)

---

[Docs index](../../index.md) · Previous: [Export to ONNX](../export.md) · Next: [GlowTTS](glowtts.md)
