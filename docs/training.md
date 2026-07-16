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

---

## StyleTTS2 training (`--engine styletts2`)

`phoonnx_train` includes a **full two-stage** engine for the
[StyleTTS2](https://arxiv.org/abs/2306.07691) architecture, porting the complete
[yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) recipe (MIT) onto the shared
training framework. It trains new models **from scratch in new languages** — the BSC
Spanish/Catalan multispeaker models are this exact recipe with a language-specific
PL-BERT — and also fine-tunes existing checkpoints onto new speakers.

The upstream model/loss/data code is vendored in `phoonnx_train/styletts2/` (imports
made package-relative; the compiled `monotonic_align` extension replaced by a pure
numpy port). Install its extra deps with `phoonnx[train,train-styletts2]`.

### Stages

Select with `stage` (engine extra / `StyleTTS2Config`):

| Stage | What trains | Recipe |
|---|---|---|
| `first` | text aligner (TMA), text encoder, style encoder, decoder | mel reconstruction with ground-truth F0/energy; s2s CE + monotonicity losses and MPD/MRSD adversarial + WavLM feature-matching after `tma_epoch` |
| `second` | PL-BERT + duration/prosody predictors, style diffusion (after `diff_epoch`), joint decoder + SLM adversarial (after `joint_epoch`) | starts from the stage-1 checkpoint (`first_stage_path`) |
| `finetune` | the `second` recipe with diffusion + joint training enabled from epoch 0 | adapt an existing checkpoint to a new speaker/dataset |

### Auxiliary models (all configurable)

| Key | Model | Where to get it |
|---|---|---|
| `asr_path` + `asr_config` | text aligner (ASRCNN) | bundled with the upstream repo (`Utils/ASR/`); trained further during TMA |
| `f0_path` | JDC pitch extractor | upstream `Utils/JDC/bst.t7` |
| `plbert_dir` | PL-BERT (config.yml + step_*.t7) | upstream `Utils/PLBERT/` for English; [BSC-LT PL-BERTs](https://huggingface.co/BSC-LT) for es/ca; train your own for a new language |
| `slm.model` (model_params) | WavLM for SLM losses | HF hub, default `microsoft/wavlm-base-plus`; disable all SLM losses with `use_slm: false` |

Set `download_aux: true` to fetch the yl4579 English aligner/pitch/PL-BERT
automatically (cached under `~/.cache/phoonnx/styletts2_aux_en`) for any path
left unset — or train your own with the engines below. Missing auxiliaries are
otherwise randomly initialized with a warning (from-scratch/CI mode).

Engine-specific keys like these ride in an `"engine_params"` dict inside the
dataset dir's `config.json` (optional for the styletts2* engines):

```json
{"engine_params": {"stage": "second", "first_stage_path": "…", "plbert_dir": "…", "download_aux": true}}
```

### Data layout

Upstream list format inside the dataset dir: `train_list.txt` / `val_list.txt` with
`filename.wav|phonemes|speaker_id` lines and audio under `wavs/` (24 kHz).

### Quality presets

`--quality` selects the model size / decoder family: `low` (halved widths,
iSTFTNet), `medium` (upstream LJSpeech recipe, iSTFTNet), `high` (upstream LibriTTS
recipe, HiFi-GAN decoder, multispeaker-ready).

### Export

`export_onnx` (via `phoonnx_train/styletts2/export.py`) accepts an upstream-layout
`.pth` (`net` dict) or a Lightning `.ckpt` saved by the engine, and emits the
two-graph zero-shot contract the `StyleTTS2Adapter` consumes — `model.onnx`
(tokens + style + speed → waveform, diffusion sampler bypassed) and
`style_encoder.onnx` (reference waveform → acoustic + prosodic style) — plus
`config.json`. Same contract as `scripts/conversion/styletts2/export_bsc.py`.

### Downstream OVOS TTS plugin config

```json
{
  "tts": {
    "module": "ovos-tts-plugin-phoonnx",
    "ovos-tts-plugin-phoonnx": {
      "voice": "/path/to/exported/model.onnx",
      "config": "/path/to/exported/config.json"
    }
  }
}
```

StyleTTS2 exports are self-contained (`model.onnx` + `style_encoder.onnx` +
`config.json`); no external vocoder is required at inference time.

---

## Training every StyleTTS2 step for a new language

Following the [BSC-LT/styletts2-spanish-multispeaker](https://huggingface.co/BSC-LT/styletts2-spanish-multispeaker)
methodology, a from-scratch model in a new language trains a language-specific
**text aligner** ([yl4579/AuxiliaryASR](https://github.com/yl4579/AuxiliaryASR)) and
**prosodic text encoder** ([yl4579/PL-BERT](https://github.com/yl4579/PL-BERT)) before
the TTS itself. `phoonnx_train` ships an engine for each, sharing the registry,
the Lightning trainer and the StyleTTS2 symbol table (`--engine` on
`python -m phoonnx_train.train`):

| Engine | Trains | Output consumed via |
|---|---|---|
| `styletts2-aligner` | ASRCNN text aligner (CTC + s2s CE, upstream AuxiliaryASR recipe) | `asr_path` + `asr_config` |
| `styletts2-plbert` | PL-BERT, `backbone: albert` (upstream, checkpoint-compatible) or `backbone: modernbert` ([BSC](https://huggingface.co/BSC-LT/PL-ModernBERT-wp-es)/[proxectonos](https://huggingface.co/proxectonos/PL-ModernBERT-gl) recipe); dual masked-phoneme + phoneme-to-grapheme heads, optional `prosodic_masking` (inverse-frequency: punctuation 40%, `!`/`?` 80%) | `plbert_dir` |
| `styletts2-pitch` | JDC pitch extractor (SmoothL1 F0 + BCE voicing, pyworld ground truth, upstream PitchExtractor recipe) | `f0_path` |

All three fine-tune too: warm-start from the English checkpoints via
`pretrained_path` (aligner/pitch) or `pretrained_dir` (PL-BERT). The trainers use
automatic optimization, so the full Lightning toolbox applies (bf16 mixed
precision, DDP, gradient accumulation); features are cached (`.mel.npy`/`.f0.npy`
next to the audio, pre-tokenized PL-BERT corpora) and audio batches are
length-bucketed to cut padding waste. `compile_model: true` enables
`torch.compile` on the aux models.

### Recipe

```bash
# 0. phonemize your data with phoonnx's own phonemizers
python -m phoonnx_train.styletts2.phonemize_corpus list \
    raw_list.txt dataset/train_list.txt --lang pt --phonemizer espeak
python -m phoonnx_train.styletts2.phonemize_corpus plbert \
    corpus.txt plbert_data --lang pt          # plain text, one sentence per line

# 1. text aligner (per language; warm-start from English via pretrained_path)
python -m phoonnx_train.train --dataset-dir dataset --engine styletts2-aligner

# 2. prosodic text encoder
python -m phoonnx_train.train --dataset-dir plbert_data --engine styletts2-plbert

# 3. pitch extractor (optional — the English JDC transfers well; BSC reused it)
python -m phoonnx_train.train --dataset-dir dataset --engine styletts2-pitch

# 4. StyleTTS2 stage first, then second, pointing at the outputs
#    (asr_path/asr_config, plbert_dir, f0_path via engine_params in config.json)
python -m phoonnx_train.train --dataset-dir dataset --engine styletts2

# 5. export the two-graph ONNX contract
python -m phoonnx_train.export_onnx --engine styletts2 ...
```

Library usage:

```python
from phoonnx_train.engines import get_engine
from phoonnx_train.engines.base import TrainingEngineConfig

engine = get_engine("styletts2-aligner")
model = engine.create_model(TrainingEngineConfig(num_symbols=178, sample_rate=24000,
                                                 extra={"batch_size": 32}),
                            dataset_paths=[Path("dataset")])
# trainer.fit(model); then:
model.save_asr_checkpoint("aligner_out")   # -> asr_path/asr_config for --engine styletts2
```
