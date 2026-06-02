# Disentangled VITS — Timbre, Articulation, and Prosody Control

phoonnx supports an optional **disentangled** VITS architecture that replaces the single monolithic speaker embedding with three separate encoders. This enables independent control of voice identity, accent/pronunciation, and rhythm/emotion.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Training](#training)
- [Export to ONNX](#export-to-onnx)
- [Inference](#inference)
- [Python API](#python-api)
- [Emotion Control](#emotion-control)
- [LoRA and Fine-Tuning](#lora-and-fine-tuning)
- [Troubleshooting](#troubleshooting)

---

## Overview

Standard VITS uses a single speaker embedding `g` that conditions the generator, posterior encoder, normalizing flow, and duration predictor. This conflates everything about a voice into one vector, making it impossible to independently control identity, accent, or rhythm.

The disentangled architecture splits `g` into three factors:

| Factor | What it controls | Analogous to |
|---|---|---|
| **Timbre** | Voice identity — who is speaking | Speaker ID / voice cloning |
| **Articulation** | Pronunciation envelope — how phonemes are realized | Accent, dialect, coarticulation |
| **Prosody** | Rhythm and pacing — when phonemes happen | Emotion, emphasis, speed |

Each factor is produced by a dedicated lightweight `ReferenceEncoder` (CNN+BiGRU, ~50K parameters) that compresses a reference mel spectrogram into a fixed-size embedding. At inference time you can provide reference audio clips to set any combination of the three factors independently.

---

## Architecture

```
text -> enc_p -> (m_p, logs_p) -> expand by dp -> z_p -> flow -> z -> dec -> audio
                                       |           ^
                                       |           |
                         g_prosody -> dp           g_flow = proj(cat(g_timbre, g_artic, g_prosody))
                                                                       |
                                                                       v
                           audio -> enc_q(y, g_timbre) -> z -> flow(z, g_flow)

g_timbre   = TimbreEncoder(speaker_id or reference_mel)
g_artic    = ArticulationEncoder(reference_mel)
g_prosody  = ProsodyEncoder(reference_mel or emotion_label)
```

### Conditioning routing

| Sub-module | Receives | Controls |
|---|---|---|
| `enc_q` (PosteriorEncoder) | `g_timbre` | Voice quality in the latent posterior |
| `dec` (Generator) | `g_timbre` | Waveform synthesis timbre |
| `flow` (ResidualCouplingBlock) | `g_flow` = concat projection | Latent transformation, bridging prior and posterior |
| `dp` (DurationPredictor) | `g_prosody` | Phoneme durations, rhythm, pacing |
| `enc_p` projection | `g_artic` (optional add) | Phoneme realization, accent |

### ReferenceEncoder

Each encoder is a stack of strided Conv1d layers followed by a bidirectional GRU:

```
mel [B, n_mels, T]
  -> Conv1d-ReLU-LayerNorm (stride=2, x3 layers)
  -> BiGRU (1 layer)
  -> Linear projection -> [B, out_dim, 1]
```

Default hyperparameters:

| Parameter | Default | Description |
|---|---|---|
| `ref_enc_hidden_channels` | 256 | Conv feature map width |
| `ref_enc_n_layers` | 3 | Number of conv layers |
| `ref_enc_kernel_size` | 3 | Conv kernel size |
| `ref_enc_stride` | 2 | Conv stride (halves time each layer) |
| `ref_enc_n_gru_layers` | 1 | BiGRU layers |

---

## Training

### Enable disentangled mode

Add `--disentangled` to the standard training command:

```bash
python phoonnx_train/train.py \
  --dataset-dir /data/preprocessed \
  --disentangled \
  --accelerator gpu \
  --devices 1 \
  --batch-size 16 \
  --max-epochs 1000
```

### Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--disentangled` | `False` | Enable three-encoder mode |
| `--ref-enc-hidden-channels` | 256 | Reference encoder hidden channels |
| `--ref-enc-n-layers` | 3 | Number of conv layers per reference encoder |
| `--ref-enc-stride` | 2 | Conv stride |
| `--timbre-dim` | `gin_channels` | Timbre embedding dimension |
| `--artic-dim` | `gin_channels` | Articulation embedding dimension |
| `--prosody-dim` | `gin_channels` | Prosody embedding dimension |
| `--n-emotion-labels` | 0 | Number of categorical emotion labels (0 = disabled) |
| `--lambda-mi` | 0.1 | Mutual information disentanglement loss weight |
| `--lambda-cycle` | 1.0 | Cycle consistency loss weight |
| `--lambda-kl-dis` | 0.01 | KL regularization weight on disentangled latents |

### Resuming from a legacy checkpoint

Legacy checkpoints that contain `emb_g` will automatically map the speaker embedding into `timbre_enc.speaker_emb` when loaded into a disentangled model. The articulation and prosody encoders are initialized randomly.

```bash
python phoonnx_train/train.py \
  --dataset-dir /data/preprocessed \
  --disentangled \
  --resume-from-checkpoint /checkpoints/legacy.ckpt \
  ...
```

### Dataset requirements

Disentanglement works best with **multi-speaker** data. The mutual information loss expects different speakers to have different timbre embeddings while sharing articulation and prosody patterns. Single-speaker datasets can use the architecture but the disentanglement will be weaker.

During training, each batch item uses its own mel spectrogram as the reference for all three encoders. Future improvements will sample:
- **timbre ref** from the same speaker
- **articulation ref** from the same language/accent
- **prosody ref** from the same emotion class

### Disentanglement losses

Three auxiliary losses are added to the generator loss during training:

1. **Mutual Information (MI) Loss** — `loss_mi_timbre` encourages timbre embeddings to cluster by speaker; `loss_mi_artic` and `loss_mi_prosody` are negatively weighted to discourage them from encoding speaker identity.
2. **KL Regularization** — `loss_kl_dis` treats each latent as an isotropic Gaussian and penalizes deviation from N(0, I), encouraging smooth interpolation.
3. **Cycle Consistency** (partial) — a placeholder loss that swaps timbre between two speakers and verifies reconstruction quality is maintained.

---

## Export to ONNX

### End-to-end mode (default for disentangled)

The exported ONNX model includes the three reference encoders and accepts raw reference mel spectrograms as inputs:

```bash
python phoonnx_train/export_onnx.py \
  /path/to/checkpoint.ckpt \
  --config /path/to/config.json \
  --output-dir /path/to/output/ \
  --disentangled-mode end-to-end
```

ONNX inputs (in addition to standard `input`, `input_lengths`, `scales`, `sid`):

| Input | Shape | Description |
|---|---|---|
| `timbre_ref_mel` | `[B, n_mels, T_ref]` | Reference mel for voice identity |
| `artic_ref_mel` | `[B, n_mels, T_ref]` | Reference mel for accent |
| `prosody_ref_mel` | `[B, n_mels, T_ref]` | Reference mel for rhythm |
| `emotion_id` | `[B]` | Categorical emotion label (optional) |

### Pre-encoded mode (planned)

For edge deployment, the reference encoders can be run offline once per voice clone, producing tiny embedding vectors (~192 floats each). These pre-computed embeddings are passed directly to the ONNX model, eliminating the encoder overhead. This mode is not yet fully implemented in the export pipeline but is the recommended approach for Raspberry Pi / browser inference.

---

## Inference

### Python API — basic synthesis

```python
from phoonnx.voice import TTSVoice, SynthesisConfig
import wave

voice = TTSVoice.load("model.onnx", "model.json")

config = SynthesisConfig()

with wave.open("output.wav", "wb") as wav_file:
    voice.synthesize_wav("Hello world!", wav_file, config)
```

### Python API — voice cloning (swap timbre)

```python
config = SynthesisConfig(
    timbre_ref_path="/data/reference_voices/alice.wav",  # clone Alice's voice
)

with wave.open("alice_speaks.wav", "wb") as wav_file:
    voice.synthesize_wav("This is a cloned voice.", wav_file, config)
```

### Python API — accent transfer

```python
config = SynthesisConfig(
    timbre_ref_path="/data/reference_voices/alice.wav",
    artic_ref_path="/data/reference_voices/bob.wav",  # Bob's accent
)

with wave.open("alice_with_bob_accent.wav", "wb") as wav_file:
    voice.synthesize_wav("This sounds like Alice speaking with Bob's accent.", wav_file, config)
```

### Python API — emotion control

```python
config = SynthesisConfig(
    timbre_ref_path="/data/reference_voices/alice.wav",
    prosody_ref_path="/data/reference_emotions/angry.wav",  # angry rhythm
)

with wave.open("alice_angry.wav", "wb") as wav_file:
    voice.synthesize_wav("I am very upset!", wav_file, config)
```

### Python API — categorical emotion

If the model was trained with `--n-emotion-labels`:

```python
config = SynthesisConfig(
    emotion="happy",  # maps to emotion_id via config.emotion_id_map
)
```

---

## Emotion Control

The `ProsodyEncoder` supports two modes:

1. **Reference audio** — provide a prosody reference clip (`prosody_ref_path`) to clone the rhythm and pacing from another utterance.
2. **Categorical labels** — if the model was trained with `--n-emotion-labels N`, an emotion embedding table is added. At inference, an emotion string is mapped to an integer ID and the corresponding embedding is used.

Common emotion labels (if trained): `neutral`, `happy`, `sad`, `angry`, `fearful`, `surprised`, `disgusted`.

---

## LoRA and Fine-Tuning

### Per-factor LoRA (planned)

The disentangled architecture enables targeted LoRA adaptation:

| Preset | Target modules | Use case |
|---|---|---|
| `timbre-only` | `dec`, `timbre_enc` | Clone a voice, keep accent and rhythm |
| `prosody-only` | `dp`, `prosody_enc` | Adjust emotion or rhythm only |
| `articulation-only` | `flow`, `artic_enc` | Transfer accent |
| `full-acoustic` | all encoders + dec/flow/dp | Full adaptation |

These presets are not yet wired into `lora_config.py` but will be added in a follow-up.

### Fine-tuning a disentangled model

When fine-tuning a pre-trained disentangled checkpoint, you can freeze the text encoder (`enc_p`) and only adapt the acoustic components:

```bash
python phoonnx_train/train.py \
  --dataset-dir /data/new_speaker \
  --disentangled \
  --resume-from-checkpoint /checkpoints/disentangled.ckpt \
  --learning-rate 1e-4 \
  ...
```

---

## Troubleshooting

### "Disentanglement will be weak" warning

If training with `--disentangled` on a single-speaker dataset, you will see:

```
Disentangled mode is enabled but the dataset only has 1 speaker.
Timbre/articulation/prosody disentanglement will be weak.
```

This is expected. The MI loss requires multiple speakers to learn meaningful timbre separation. The model will still train but the three encoders may learn overlapping representations.

### ONNX model does not accept reference mel inputs

Ensure you exported with `--disentangled-mode end-to-end`. Without this flag, the ONNX export uses the legacy forward function and only accepts `input`, `input_lengths`, `scales`, and `sid`.

### Reference audio too short

The `ReferenceEncoder` applies three stride-2 conv layers, reducing time by 8x. A reference clip should be at least a few hundred frames (~2-3 seconds at 22050 Hz with hop_size=256) to avoid empty output. If the clip is too short, the GRU will receive an empty sequence and the embedding will be near-zero, causing the model to fall back to its learned default.

### Emotion label not found

If `emotion` is set in `SynthesisConfig` but the model was not trained with `--n-emotion-labels`, the runtime will ignore it silently. Check the ONNX model metadata: `disentangled` should be `"True"` and `n_emotion_labels` should be > 0.
