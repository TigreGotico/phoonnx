# Matcha-TTS Engine

Matcha-TTS is a non-autoregressive TTS architecture based on **conditional flow matching**. It uses a Transformer text encoder, a duration predictor, and a flow-matching decoder to generate mel spectrograms from phoneme sequences.

## Inference

Matcha-TTS inference uses a **two-stage ONNX pipeline**:

1. **Mel model** — flow-matching acoustic model (phoneme IDs → mel spectrogram)
2. **Vocoder** — Vocos-style vocoder (mel → waveform)

Both models are separate ONNX files. The ``MatchaAdapter`` holds the vocoder session internally and chains inference end-to-end.

### ONNX Inputs (mel model)

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| ``x`` | int64 | ``[B, T]`` | Phoneme IDs (interspersed with blank=0) |
| ``x_lengths`` | int64 | ``[B]`` | Sequence lengths |
| ``scales`` | float32 | ``[2]`` | ``[temperature, length_scale]`` |
| ``spks`` | int64 | ``[B]`` | Speaker ID (optional) |

### ONNX Outputs (mel model)

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| ``mel`` | float32 | ``[B, n_mels, T_mel]`` | Generated mel spectrogram |
| ``mel_lengths`` | int64 | ``[B]`` | Mel lengths |

The vocoder is run internally by the adapter, producing a float32 waveform via inverse STFT with overlap-add and optional spectral denoising.

### Loading a Matcha-TTS voice

```python
from phoonnx import TTSVoice

voice = TTSVoice.load(
    model_path="matcha_multispeaker_cat_all_opset_15_10_steps.onnx",
    config_path="matcha_config.json",
)
```

The JSON config must include:

```json
{
    "engine": "matcha",
    "engine_params": {
        "vocoder_path": "mel_spec_22khz_cat.onnx",
        "vocoder_config": {
            "feature_extractor": {
                "init_args": {
                    "n_fft": 1024,
                    "hop_length": 256,
                    "sample_rate": 22050
                }
            }
        }
    }
}
```

### Text processing

Matcha-TTS uses phoonnx's standard phonemizer and tokenizer. The tokenizer must be configured with ``blank_id = 0`` so that interspersed blanks are already present in the phoneme sequence fed to ONNX. The adapter does **not** re-intersperse — it passes the tokenized sequence directly to the mel model.

### Parameters

| Param | Default | Description |
|-------|---------|-------------|
| ``temperature`` | 0.667 | Sampling temperature for the flow-matching decoder |
| ``length_scale`` | 1.0 | Speech rate multiplier (higher = slower) |
| ``denoise`` | true | Spectral subtraction denoising via vocoder bias |

## Training

Matcha-TTS training requires the upstream ``matcha-tts`` package. It is installed automatically as a development dependency when working in the shared venv.

### Quick start

```bash
python -m phoonnx_train.train \
    --engine matcha \
    --dataset-dir ./dataset \
    --output-dir ./checkpoints
```

### Quality presets

| Preset | Encoder channels | Decoder channels | Heads / Layers |
|--------|-----------------|------------------|----------------|
| ``x-low`` | 128 | [192, 192] | 2 heads / 4 layers |
| ``medium`` | 192 | [256, 256] | 2 heads / 6 layers |
| ``high`` | 256 | [384, 384] | 4 heads / 8 layers |

Use ``--quality <preset>`` or set ``quality`` in ``extra`` config.

### Architecture overview

- **Text encoder** — Transformer with RoPE positional embeddings, ConvReluNorm prenet, duration predictor
- **CFM decoder** — Conditional flow matching with UNet1D estimator (ResNet blocks + Transformer/Conformer blocks)
- **Losses** — Duration loss (MSE), prior loss (Gaussian), flow-matching loss (MSE on velocity field)

### ONNX export

After training, export the mel model to ONNX:

```bash
python -m phoonnx_train.export_onnx \
    --engine matcha \
    --checkpoint matcha.ckpt \
    --output-dir ./onnx
```

The exported ONNX file is the **mel model only**. You still need a separate vocoder ONNX (e.g. Vocos) for full speech synthesis. The inference adapter chains both automatically.

### Data statistics

Matcha-TTS normalizes mel spectrograms using dataset statistics. Compute them beforehand:

```bash
python -m matcha.utils.generate_data_statistics \
    --config configs/data/your_dataset.yaml
```

Then pass ``mel_mean`` and ``mel_std`` via engine config ``extra``:

```json
{
    "extra": {
        "mel_mean": -5.536622,
        "mel_std": 2.116101
    }
}
```

## References

- [Matcha-TTS paper (ICASSP 2024)](https://arxiv.org/abs/2309.03199)
- [Upstream repository](https://github.com/shivammehta25/Matcha-TTS)
