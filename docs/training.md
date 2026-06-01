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

## 4. LoRA Fine-Tuning (Voice Adaptation)

LoRA (Low-Rank Adaptation) enables fast voice adaptation with less data (5–15 min of audio) by only training lightweight adapter layers while keeping the base model frozen. Pronunciation is preserved from the base model since the TextEncoder is frozen.

### Presets

| Preset | Rank | Target modules | Adapter size | Use case |
|---|---|---|---|---|
| `generator-only` | 4 | `dec` | ~300KB | Fastest, minimal data |
| `full-acoustic` | 8 | `dec`, `enc_q`, `flow`, `dp` | ~1MB | Best quality (default) |
| `aggressive` | 16 | all components incl. `enc_p` | ~2.5MB | Maximum expressiveness |

### Training

```bash
# Preprocess (same as regular fine-tuning)
python phoonnx_train/preprocess.py \
  --language eu-ES --input-dir /data/new_speaker \
  --prev-config /data/base/config.json \
  --single-speaker --output-dir /data/preprocessed

# LoRA train — uses base checkpoint, only adapter weights are trainable
python phoonnx_train/train.py \
  --dataset-dir /data/preprocessed \
  --resume-from-checkpoint /data/base/base.ckpt \
  --lora-scope full-acoustic \
  --max-epochs 200 --batch-size 16
```

Custom rank/alpha/targets override the preset:

```bash
python phoonnx_train/train.py \
  --dataset-dir /data/preprocessed \
  --resume-from-checkpoint /data/base/base.ckpt \
  --lora-scope full-acoustic \
  --lora-rank 12 \
  --lora-alpha 24.0 \
  --lora-target-modules "dec,enc_q" \
  --max-epochs 200
```

After training, the LoRA adapter is saved to `<output_dir>/lora_adapter/lora_adapter.pt` along with `lora_config.json`.

### Merging & Export

Merge the LoRA adapter into the base model and export to ONNX:

```bash
python phoonnx_train/merge_lora.py \
  /data/base/base.ckpt \
  --lora-adapter /data/lora_adapter/lora_adapter.pt \
  --config /data/base/config.json \
  --lora-scope full-acoustic \
  --output-dir /data/output_merged/
```

This produces:
- `merged_model.ckpt` — merged PyTorch checkpoint
- `lora_adapter.pt` — standalone adapter (for server-side hot-swap, future)
- `config.json` — config with LoRA metadata
- `merged_model.onnx` — ONNX model ready for phoonnx inference

### Deployment Modes

**Edge (merged ONNX):** A single ONNX file per voice, loaded normally with `TTSVoice.load()`. No runtime changes needed.

**Server (hot-swap, future):** Load the base ONNX + per-voice LoRA adapters. Swap voices without reloading the base model. Use `TTSVoice.load_with_lora(base_path, lora_path)`.

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
