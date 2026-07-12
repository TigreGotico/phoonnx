# Matcha-TTS Engine

Matcha-TTS is a non-autoregressive TTS architecture based on **conditional flow matching**. It uses a Transformer text encoder, a duration predictor, and a flow-matching decoder to generate mel spectrograms from phoneme sequences.

## Inference

Matcha-TTS produces a **mel spectrogram**, not a waveform. A separate
**vocoder** turns the mel into audio. Matcha is published in two forms, and
the ``MatchaAdapter`` handles both transparently (it branches on the mel
model's output rank):

- **Two-stage** — the acoustic model outputs a mel spectrogram
  ``[B, n_mels, T]`` and a separate vocoder ONNX reconstructs the waveform.
- **End-to-end** — a fused model (e.g. ``*_wavenext_e2e.onnx``,
  ``matcha_*_simply.onnx``) outputs the waveform directly. No vocoder is
  configured; the adapter returns the model output as-is.

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
| ``mel`` | float32 | ``[B, n_mels, T_mel]`` | Generated mel spectrogram (two-stage) |
| ``mel_lengths`` | int64 | ``[B]`` | Mel lengths (padding is trimmed before the vocoder) |

End-to-end models instead emit a single waveform tensor ``[B, T]`` and need no vocoder.

## Vocoders

> See [vocoders.md](./vocoders.md) for the shared vocoder registry, the
> config-driven mel preprocessing flags, and how to use/replace/add vocoders.

Vocoders are a **pluggable registry** (``phoonnx.engines.vocoders``), parallel
to the engine registry. The acoustic model and vocoder are versioned and
swapped independently, so one Matcha voice can be paired with whichever
vocoder is licensed/tested for it.

| `vocoder_type` | Family | ONNX output | Reconstruction |
|----------------|--------|-------------|----------------|
| ``vocos`` | Vocos / alVoCat | STFT mag + real + imag (3 tensors) | inverse STFT overlap-add (+ optional denoise) |
| ``wavenext`` | Wavenext | raw waveform (1 tensor) | none — Vocos with the ISTFT head replaced by a trained linear layer |
| ``hifigan`` | HiFi-GAN | raw waveform (1 tensor) | none |

If ``vocoder_type`` is omitted it is auto-detected from the vocoder ONNX
(3 outputs → Vocos, 1 output → raw waveform).

Tested vocoders for the Catalan Matxa models (all 22.05 kHz, 80-bin mel):

| Vocoder | Repo / file | License |
|---------|-------------|---------|
| Wavenext | ``BSC-LT/wavenext-mel`` → ``mel_spec_22khz_wavenext.onnx`` | Apache-2.0 (commercial-safe) |
| alVoCat (Vocos) | ``projecte-aina/alvocat-vocos-22khz`` → ``mel_spec_22khz_cat.onnx`` | **CC-BY-NC-4.0 (non-commercial)** |

Prefer Wavenext (or a fused end-to-end model) for commercial use; alVoCat-Vocos
is non-commercial.

### Loading a Matcha-TTS voice

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

The voice manager wires this automatically: index entries carry ``vocoder_url``
and ``vocoder_type``, the separate vocoder is downloaded alongside the model,
and ``engine_params.vocoder_path`` is set to the local file. See
[Voice index](#voice-index) below.

### Text processing

Matcha-TTS uses phoonnx's standard phonemizer and tokenizer. The tokenizer must
be configured with ``blank_id = 0`` so that interspersed blanks are already
present in the phoneme sequence fed to ONNX. The adapter does **not**
re-intersperse — it passes the tokenized sequence directly to the mel model.

Grapheme/character models (e.g. ``matxa-tts-v2-ca-central-graphemes``) have no
phonemes: from phoonnx's point of view they run the pass-through
graphemes/unicode phonemizer (``GraphemePhonemizer`` / ``UnicodeCodepointPhonemizer``)
and the **tokenizer** maps characters → IDs. Set ``phoneme_type: graphemes``
(or ``unicode``) / ``alphabet: unicode``.

### Parameters

| Param | Default | Description |
|-------|---------|-------------|
| ``temperature`` | 0.667 | Sampling temperature for the flow-matching decoder |
| ``length_scale`` | 1.0 | Speech rate multiplier (higher = slower) |
| ``denoise`` | true | Spectral-subtraction denoising — Vocos only; ignored by raw-waveform/end-to-end vocoders |

## Voice index

Catalan Matxa voices ship in ``phoonnx/voice_index/BSC.json``. Each entry links
the acoustic model to its tested vocoder and records the license:

| Field | Meaning |
|-------|---------|
| ``vocoder_url`` | Separate vocoder ONNX (omit for end-to-end models) |
| ``vocoder_type`` | ``vocos`` / ``wavenext`` / ``hifigan`` (auto-detected if null) |
| ``license`` | Model (and vocoder) license, e.g. ``gpl-3.0`` or the CC-BY-NC vocoder note |
| ``verified`` | ``true`` once synthesis has been confirmed end-to-end through phoonnx |
| ``speakers`` | ``speaker_id`` → name/accent map |

The BSC entries point at **phoonnx mirrors under `OpenVoiceOS/`** — the upstream
BSC models repackaged with a phoonnx `config.json` that carries the symbol table
as the tokenizer vocab (the espeak voices use the 178-symbol Matcha table; the
grapheme voice uses its 199-symbol char table). Symbol handling lives entirely
in the **tokenizer**, so `TTSVoice.load` reproduces the model's exact token IDs
with no per-model code. All entries are ``verified: true`` (synthesis confirmed
end-to-end through the index → download → `TTSVoice.load` path).

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

## Training

Matcha-TTS inference requires ``scipy`` (install with ``pip install phoonnx[matcha]``). Training additionally requires the ``train`` extra and the upstream ``matcha-tts`` package (``pip install phoonnx[train] matcha-tts``).

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
