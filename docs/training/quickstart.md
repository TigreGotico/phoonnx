# Training quickstart

<!-- agent-hint
covers: how to go from a folder of audio to a speaking ONNX voice, step by step
source: phoonnx_train/preprocess.py:cli, phoonnx_train/train.py:main, phoonnx_train/export_onnx.py:cli, phoonnx_train/engines/vits.py:write_sidecars
generated: no
verify: ! comm -23 <(grep -oP -- '(?<![\w-])--[a-z][a-z0-9-]*' docs/training/quickstart.md | grep -vxE -- '--(pre|logdir|learning-rate)' | sort -u) <(for m in preprocess train export_onnx; do python -m phoonnx_train.$m --help; done | grep -oP -- '(?<![\w-])--[a-z][a-z0-9-]*' | sort -u) | grep .
see: datasets.md, training.md, export.md
-->

This page takes a first-time trainer from a folder of audio to a speaking ONNX voice, step by step.

Follow the steps from top to bottom. Each step shows the command, the output you see, and the usual failures.
The five steps are: **prepare a dataset → preprocess → train → export → speak.**

## 0. Install the training pipeline

```bash
pip install --pre "phoonnx[train]"
```

This adds PyTorch, PyTorch Lightning and librosa to the base install. phoonnx ships as alpha
releases, so the `--pre` flag is necessary. Use a GPU: training on a CPU is too slow to be practical.

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

The audio folder may be named `wav/` or `wavs/`. For an intelligible voice, use at least 30
minutes of clean speech from one speaker. More data gives a better voice.

This page uses LJSpeech, the simplest format. Preprocess also reads JSONL, Parquet and Hugging
Face datasets, and it can merge several sources in one run: see
[Datasets](datasets.md#input-formats). [Datasets](datasets.md) also covers multi-speaker
columns, filename resolution, audio requirements and quality filtering.

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

- **`No valid utterances found in dataset.`** The script exits with status 0. The cause is
  usually in `metadata.csv`: a wrong path, a wrong delimiter (it must be `|`), or audio files
  that do not match the filenames. This log line is the only sign of a zero result, so always
  read it.
- **`Missing metadata file: .../metadata.csv`** — the file is not directly under
  `--input-dir`.
- **Rows warning `Skipping malformed row`** — a line has fewer than two `|`-separated fields.
- **Utterances dropped with `audio is too short for its text`.** The clip is shorter than its
  transcript needs, and preprocess skips it. A few drops are normal. Many drops mean that the
  audio and the transcripts do not match.

Add quality filters (`--filter utmos:3.0:`, `--filter wpm:80:400`) to drop bad clips before
training. Quality filtering needs `phoonnx[train-eval]`. See
[Datasets](datasets.md#quality-filtering).

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

Add `--compile` to compile the model with `torch.compile`. `--compile-mode` sets the
compile mode. See the [training reference](training.md#trainpy-options).

**Expected output.** Checkpoints appear under the run directory as training progresses:

```
train_out/runs/lightning_logs/version_0/checkpoints/epoch=49-step=12345.ckpt
```

A checkpoint is saved every epoch by default (`--checkpoint-epochs`), and all are kept.

**Watch progress** with TensorBoard:

```bash
tensorboard --logdir train_out/runs/lightning_logs
```

**When to stop.** There is no fixed epoch count. Listen to samples and watch the losses
level off. A single-speaker `medium` voice usually speaks clearly after a few hundred epochs,
and it improves with more epochs. Stop when the samples stop getting better, then use the
latest checkpoint.

**Common failures**

- **CUDA out of memory.** Lower `--batch-size` (try 8, then 4). A `medium` VITS voice at
  batch size 16 needs about 12 GB of GPU memory. A smaller card needs a smaller batch.
  `--precision 16-mixed` also uses less memory.
- **`Unknown engine ...`.** The `--engine` name is not registered. See the
  [Training reference](training.md#engines) for the list.
- **Very slow, or no GPU used.** Make sure that you set `--accelerator gpu` and that a CUDA or
  ROCm build of PyTorch is installed.

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

`export_onnx` takes the checkpoint as its **single positional argument**. Set the output
folder with `-o/--output-dir`. There is no second positional argument. **Expected output**:

```
exported/
├── epoch=999-step=250000.ckpt.onnx    # the model, named after the checkpoint
├── epoch=999-step=250000.json         # phoonnx config, always written
└── epoch=999-step=250000.tokens.txt   # sherpa-onnx tokens, from --generate-tokens
```

Add `--piper` to also write `epoch=999-step=250000.piper.json`, a Piper-format config.
It is useful only for a voice with `phoneme_type=espeak` and `alphabet=ipa`.

## 5. Speak

Load the exported voice with the config written beside it and synthesize:

```python
import wave
from phoonnx.voice import TTSVoice

voice = TTSVoice.load("exported/epoch=999-step=250000.ckpt.onnx", "exported/epoch=999-step=250000.json")
with wave.open("my_voice.wav", "wb") as wav_file:
    voice.synthesize_wav("My own trained voice, speaking at last.", wav_file)
```

## Training notebooks

The notebook
[`train_vits.ipynb`](https://github.com/TigreGotico/ml-notebooks/blob/main/tts/train_vits.ipynb)
in the `ml-notebooks` repository runs the steps of this page on Colab or Kaggle. It clones
phoonnx and runs the same `preprocess`, `train` and `export_onnx` commands. It gives a trained
and exported ONNX voice, with no local setup.

When you do not have a dataset, the notebook
[`tts_dataset_gen.ipynb`](https://github.com/TigreGotico/ml-notebooks/blob/main/tts/tts_dataset_gen.ipynb)
makes one. It speaks text in a donor voice and writes a `metadata.csv` and a `wavs/` folder in
the LJSpeech layout that [step 1](#1-prepare-a-dataset) uses.

---

[Docs index](../index.md) · Previous: [Voice sweep](../voice_sweep.md) · Next: [Datasets](datasets.md)
