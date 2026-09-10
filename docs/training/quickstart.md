# Training quickstart — the golden path

This guide is for someone who has never trained a TTS voice and wants to succeed on the first
try. Follow it top to bottom and you will go from a folder of audio to a trained, exported,
speaking ONNX voice. Every command is copy-pasteable; expected outputs and the common ways
each step fails are listed inline.

The five stages are: **prepare a dataset → preprocess → train → export → speak.**

## 0. Install the training pipeline

```bash
pip install "phoonnx[train]"
```

This adds PyTorch, PyTorch Lightning and librosa on top of the base install. A GPU is
strongly recommended — CPU training is not practical.

## 1. Prepare a dataset

Use the LJSpeech layout: a `metadata.csv` and a folder of WAV files.

```
my_dataset/
├── metadata.csv
└── wavs/
    ├── 0001.wav
    ├── 0002.wav
    └── ...
```

`metadata.csv` is **pipe-delimited**, one line per clip. For a single speaker:

```
0001|Hello and welcome.
0002|This is the second utterance.
```

The audio folder may be named `wav/` or `wavs/`. Aim for at least ~30 minutes of clean,
single-speaker speech to get an intelligible voice; more is better.

LJSpeech is the simplest format and the golden path followed here. Preprocess also reads JSONL,
Parquet, and Hugging Face datasets, and can merge several sources at once — see
[Datasets](datasets.md#input-formats). The full spec (multi-speaker columns, filename
resolution, audio requirements, quality filtering) is in [Datasets](datasets.md).

## 2. Preprocess

Phonemize the text and cache the audio into a training-ready dataset:

```bash
python -m phoonnx_train.preprocess \
  --input-dir my_dataset \
  --output-dir train_out \
  --language en-US \
  --phoneme-type espeak \
  --alphabet ipa \
  --sample-rate 22050 \
  --single-speaker
```

**Expected output** in `train_out/`:

```
train_out/
├── config.json          # model + dataset configuration
├── dataset.jsonl        # one normalized utterance per line, with phoneme IDs
└── cache/22050/         # normalized audio + cached spectrograms
```

The final log line reports how many utterances survived:
`Preprocessing complete. Wrote N valid utterances to dataset.jsonl.`

**Common failures**

- **`No valid utterances found in dataset.`** and the script exits with status 0 — almost
  always a `metadata.csv` problem: wrong path, wrong delimiter (must be `|`), or audio files
  that don't match the filenames. Preprocess is silent about a zero result beyond this log
  line, so always read it.
- **`Missing metadata file: .../metadata.csv`** — the file is not directly under
  `--input-dir`.
- **Rows warning `Skipping malformed row`** — a line has fewer than two `|`-separated fields.
- **Utterances dropped with `audio is too short for its text`** — the clip is shorter than its
  transcript needs; these are skipped automatically. A few is fine; many means the audio and
  transcripts are misaligned.

Recommended: add quality filters (`--filter utmos:3.0:`, `--filter wpm:80:400`, …) to drop bad
clips before training. See [Datasets](datasets.md#quality-filtering) — quality filtering needs
`phoonnx[train-eval]`.

## 3. Train

Train a VITS voice on a single GPU:

```bash
python -m phoonnx_train.train \
  --dataset-dir train_out \
  --engine vits \
  --quality medium \
  --batch-size 16 \
  --accelerator gpu \
  --devices 1 \
  --max-epochs 1000 \
  --default-root-dir train_out/runs
```

Add `--compile` for a `torch.compile` speedup (optional; tune the graph-capture mode with
`--compile-mode`). See the [training reference](training.md#train.py-options).

**Expected output** — checkpoints appear under the run directory as training progresses:

```
train_out/runs/lightning_logs/version_0/checkpoints/epoch=49-step=12345.ckpt
```

A checkpoint is saved every epoch by default (`--checkpoint-epochs`), and all are kept.

**Watch progress** with TensorBoard:

```bash
tensorboard --logdir train_out/runs/lightning_logs
```

**When to stop** — there is no fixed epoch count. Listen to samples and watch the losses
plateau; for a single-speaker `medium` voice, intelligible speech typically emerges within a
few hundred epochs and quality keeps improving with more. Stop when samples stop getting
better, then use the latest checkpoint.

**Common failures**

- **CUDA out of memory** — lower `--batch-size` (try 8, then 4). As a rough guide, a
  `medium` VITS voice at batch size 16 wants a mid-range (~12 GB) GPU; smaller cards need a
  smaller batch. You can also use `--precision 16-mixed` to cut memory.
- **`Unknown engine ...`** — the `--engine` name is not registered; see
  [Training reference](training.md#engines) for the list.
- **Very slow / no GPU used** — confirm `--accelerator gpu` and that a CUDA/ROCm ONNX-free
  PyTorch build is installed.

> There is **no** `--learning-rate` option. The optimizer schedule is set by the engine and
> its quality preset. The complete flag list is in the [Training reference](training.md).

## 4. Export to ONNX

Convert the checkpoint to an ONNX voice:

```bash
python -m phoonnx_train.export_onnx \
  train_out/runs/lightning_logs/version_0/checkpoints/epoch=999-step=250000.ckpt \
  --config train_out/config.json \
  --engine vits \
  --output-dir exported \
  --generate-tokens
```

`export_onnx` takes the checkpoint as its **single positional argument**; the output location
is set with `-o/--output-dir`, never as a second positional. **Expected output**:

```
exported/
└── model.onnx
```

Add `--piper` to also emit a Piper-compatible `.json` (only meaningful for
`phoneme_type=espeak` / `alphabet=ipa` voices). Details in [Export](export.md).

## 5. Speak

Load the exported voice with the config produced in step 2 and synthesize:

```python
import wave
from phoonnx.voice import TTSVoice

voice = TTSVoice.load("exported/model.onnx", "train_out/config.json")
with wave.open("my_voice.wav", "wb") as wav_file:
    voice.synthesize_wav("My own trained voice, speaking at last.", wav_file)
```

That's the full loop. From here:

- Fine-tune onto a new speaker or resume training: [Training reference](training.md).
- Train a different architecture (Matcha, GlowTTS, StyleTTS2, …): [engine guides](engines/matcha.md).
- Ship the voice in a voice assistant: [OVOS plugin](../ovos_plugin.md).
