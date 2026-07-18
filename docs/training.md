# Training

phoonnx includes a full training pipeline (`phoonnx_train`) for training new VITS-style TTS models from scratch or fine-tuning existing ones.

## Overview

The training workflow has three stages:

1. **Preprocess** — phonemize a dataset and produce `dataset.jsonl` + `config.json`
2. **Train** — train a VITS model from the preprocessed data
3. **Export** — convert the PyTorch checkpoint to ONNX format

---

## 1. Preprocessing

The `preprocess.py` script phonemizes an LJSpeech-style dataset and prepares it for training.

```bash
python phoonnx_train/preprocess.py \
  --language en-US \
  --input-dir /data/my_dataset \
  --output-dir /data/preprocessed \
  --sample-rate 22050 \
  --phoneme-type espeak \
  --alphabet ipa \
  --single-speaker
```

### Key Options

| Option | Description |
|--------|-------------|
| `--language` | Language code (e.g. `en-US`, `pt-PT`) |
| `--input-dir` | Root directory of the input dataset |
| `--output-dir` | Where to write `config.json` and `dataset.jsonl` |
| `--sample-rate` | Target audio sample rate (default: 22050) |
| `--phoneme-type` | Phonemizer backend (e.g. `espeak`, `gruut`) |
| `--alphabet` | Phoneme alphabet (e.g. `ipa`, `arpa`) |
| `--single-speaker` | Treat all data as a single speaker |
| `--speaker-id INT` | Assign a fixed speaker ID (cannot use with `--single-speaker`) |
| `--prev-config PATH` | Reuse phoneme map from a prior config (for fine-tuning) |
| `--drop-extra-phonemes` | Drop phonemes not in the prev-config phoneme map |
| `--phonemizer-model ID` | Model ID for neural phonemizers (e.g. ByT5) |
| `--add-diacritics` | Add diacritics before phonemization (Arabic/Hebrew) |
| `--skip-audio` | Only phonemize text; skip audio normalization |
| `--max-workers INT` | Number of parallel workers |

### Input Dataset Format

The input dataset should follow the LJSpeech layout:

```
my_dataset/
  metadata.csv       # pipe-separated: filename|text
  wavs/
    001.wav
    002.wav
    ...
```

---

## 2. Training

```bash
python phoonnx_train/train.py \
  --dataset-dir /data/preprocessed \
  --accelerator gpu \
  --devices 1 \
  --batch-size 16 \
  --max-epochs 1000 \
  --validation-split 0.05 \
  --checkpoint-epochs 1 \
  --precision 32
```

### Resuming from a Checkpoint

```bash
python phoonnx_train/train.py \
  --dataset-dir /data/preprocessed \
  --resume-from-checkpoint /checkpoints/epoch=50.ckpt \
  ...
```

### Fine-tuning

For fine-tuning on a new speaker, preprocess your new dataset with `--prev-config` pointing to the original model's `config.json`. This reuses the existing phoneme-to-ID mapping so the vocabulary is preserved.

---

## 3. Exporting to ONNX

After training, convert the `.ckpt` checkpoint to ONNX for inference:

```bash
python phoonnx_train/export_onnx.py \
  /path/to/checkpoint.ckpt \
  --config /path/to/config.json \
  --output-dir /path/to/output/
```

### Options

| Option | Description |
|--------|-------------|
| `-c`, `--config` | Path to the model config JSON |
| `-o`, `--output-dir` | Output directory (default: current directory) |
| `-t`, `--generate-tokens` | Also write a `tokens.txt` file (for Sherpa/Mimic3) |
| `-p`, `--piper` | Also write a Piper-compatible `.json` config |

> **Note:** Piper-compatible export (`--piper`) is only valid for models trained with `alphabet=ipa` and `phoneme_type=espeak`.

---

## Training on Kaggle

A ready-to-use Kaggle notebook is provided at `phoonnx_train/train_kaggle.ipynb`. It handles environment setup, dataset download from HuggingFace, preprocessing, training, and export in a single notebook.

Key environment variables used in the notebook:

```python
os.environ["LANG"] = "en-US"
os.environ["ACCELERATOR"] = "gpu"
os.environ["PHONEMIZER"] = "espeak"
os.environ["ALPHABET"] = "ipa"
os.environ["HF_DATASET"] = "my-org/my-dataset"
os.environ["LOCAL_DATASET_PATH"] = "/kaggle/working/dataset"
os.environ["PHONEMIZED_DATASET_PATH"] = "/kaggle/working/training"
os.environ["BASE_CKPT_URL"] = "https://..."   # optional: base checkpoint for fine-tuning
os.environ["LOCAL_CKPT_PATH"] = "/kaggle/working/base.ckpt"
```

---

## FastPitch / SpeedySpeech training (`--engine fastpitch` / `--engine speedyspeech`)

`phoonnx_train` includes a training engine for the Coqui **ForwardTTS** family —
[FastPitch](https://arxiv.org/abs/2006.06873) and
[SpeedySpeech](https://arxiv.org/abs/2008.03802) are both configurations of the same
non-autoregressive text→mel model: an encoder produces per-token states, a duration
predictor expands them to frame rate, and a decoder renders an 80-channel mel.
Durations are learned **unsupervised** (no external aligner or forced-alignment step
needed) via an attention-based alignment network + monotonic alignment search, the
same recipe as upstream FastPitch.

| | FastPitch | SpeedySpeech |
|---|---|---|
| encoder/decoder | FFT-transformer | residual conv-BN |
| pitch predictor | on | off |
| relative size | larger, higher quality | smaller, faster |

The model/loss code is vendored in `phoonnx_train/fastpitch/` — a self-contained,
pure-torch port of coqui-ai/TTS's `TTS/tts/models/forward_tts.py` and the layers it
uses (© Coqui GmbH, MPL-2.0; see the license note in
`phoonnx_train/fastpitch/__init__.py`), with no dependency on the unmaintained `TTS`
package. It reuses the shared VITS preprocessing pipeline (`audio_norm_path` /
`audio_spec_path`) and derives the mel target from the cached linear spectrogram at
train time — no separate preprocessing pass is required to switch between `vits` and
`fastpitch`/`speedyspeech` on the same phonemized dataset.

### Pitch (F0) extraction

FastPitch's pitch predictor needs a frame-level F0 target. The engine implements
`extra_preprocess` (same contract as the OptiSpeech training engine) to extract F0 via
`pyworld` (DIO + StoneMask) and cache it as `<utterance>.f0.npy` next to the
spectrogram cache. F0 is computed on the same trimmed/normalized audio the mels come
from, at a frame period matched to the mel hop, so the pitch track is frame-aligned
with the mel target. Install with `phoonnx[train,train-fastpitch]`. If `pyworld`/`librosa`
aren't installed, training still runs — the pitch predictor trains without a target
loss term (a warning is logged) — but real pitch conditioning requires the extra.
SpeedySpeech does not use pitch at all.

### Selecting the variant

```sh
python -m phoonnx_train.train --engine fastpitch ...      # FastPitch (FFT-transformer, pitch on)
python -m phoonnx_train.train --engine speedyspeech ...   # SpeedySpeech (residual conv-BN, no pitch)
```

Both are the same underlying engine class (`ForwardTTSTrainingEngine`); the
`--engine` name only changes the *default* `variant`. It can also be set explicitly
via the engine's `extra` config (e.g. `variant: speedyspeech`) to mix-and-match
encoder family and pitch predictor.

### Quality presets

`--quality` selects `x-low` / `medium` / `high`, scaling `hidden_channels`,
`hidden_channels_ffn`, and the encoder/decoder layer count.

### Export

`export_onnx` emits the `token_ids` [B,T] + `pace`/`pitch_mul`/`pitch_add` [1]
control inputs (+ optional `speaker` [B] for multi-speaker models) →
`mel_spec` [B, 80, T_mel] contract, matching
`scripts/conversion/coqui_fastpitch_export/export_fp.py` (used to convert
pretrained Coqui checkpoints) and consumed by `phoonnx.engines.fastpitch.FastPitchAdapter`
(which reuses `MixerTTSAdapter`'s mel→vocoder inference path — FastPitch is a
**two-stage** model, so a separate vocoder is required at inference time, same as
Mixer-TTS/GlowTTS/Matcha).

### Downstream OVOS TTS plugin config

```json
{
  "tts": {
    "module": "ovos-tts-plugin-phoonnx",
    "ovos-tts-plugin-phoonnx": {
      "voice": "/path/to/exported/model.onnx",
      "config": "/path/to/exported/config.json",
      "engine_params": {
        "vocoder_path": "/path/to/vocoder.onnx",
        "vocoder_type": "hifigan"
      }
    }
  }
}
```

The native `config.json`'s `"engine"` field must be `"fastpitch"` (or `"fast_pitch"`)
so `phoonnx`'s engine auto-detection routes to `FastPitchAdapter` instead of
`MixerTTSAdapter` (the two share an identical ONNX I/O contract and are only told
apart by this field).
