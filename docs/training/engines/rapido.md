# Rápido Engine (training)

<!-- agent-hint
covers: how to train, fine-tune and export a Rápido voice with --engine rapido
source: phoonnx_train/engines/rapido.py, phoonnx_train/rapido/lightning.py, phoonnx_train/rapido/export_onnx.py
generated: no
verify: grep -n "learning_rate\|cond_drop\|mas_upsample\|freeze_backbone_steps\|text_only" phoonnx_train/rapido/lightning.py
see: ../training.md, ../export.md, ../../engines.md
-->

This page shows how to train or fine-tune a Rápido voice with `--engine rapido` and export it, for anyone with a prepared dataset.

> Related: [training reference](../training.md) ·
> [adapter architecture](../../engines.md) ·
> [configuration](../../configuration.md)

## What it trains

Rápido is a 21M-parameter flow-matching DiT. It predicts 64-dimensional
VoxCPM2 AudioVAE latents at 25 frames per second. The frozen AudioVAE decoder
turns the latents into 48 kHz audio. The model has four parts:

- A ConvNeXt and attention text encoder reads one embedding per token.
- A duration predictor gives each token a duration in frames.
- An aligner turns the word and frame layout into a condition for each frame.
- A DiT predicts the flow-matching velocity of the latents.

The upstream release ships no training code. The training loop in
`phoonnx_train/rapido/lightning.py` is our own.

## Extras needed

```bash
pip install --pre "phoonnx[train]"
pip install --pre "phoonnx[train-resample]"   # resampling to the AudioVAE's 16 kHz input
pip install --pre "phoonnx[pt]"               # tugaphone, for Portuguese
```

## Data and tokens

Rápido trains on the shared `preprocess.py` output (`dataset.jsonl` and
`config.json`). Preprocessing interleaves a blank token between the phonemes.
The model does not read the blanks. It reads the phonemes, the word
separator, the clause punctuation, BOS and EOS. The exported voice config turns
blank insertion off, so the runtime tokenizer gives the model the ids it was
trained on.

The engine finds words from the ids. A word starts at a token that follows a
boundary token. The boundary tokens are the word separator, the six clause
tokens `. ? ! , : ;`, BOS and EOS. A word owns the boundary tokens that follow
it, so punctuation belongs to the word before it. The clause tokens are
boundaries because a full stop, a question mark and an exclamation mark have
no space after them in the phoneme stream.

## Durations without an aligner

The engine needs no external aligner. In each training step, monotonic
alignment search (as in Matcha-TTS) aligns the tokens to the latent frames:

1. A linear prior head projects the text features into the latent space.
2. The search runs over the frames repeated four times
   (`mas_upsample`), so a token can hold a quarter of a frame. A stress mark or
   a combining tilde is a token of its own, and a 25 Hz frame is often longer
   than one such token.
3. The aligned durations train the duration predictor (squared error on
   `log(1 + frames)`), give the word and frame layout for the aligner, and give
   the prior loss its targets.

Word frame counts come from the cumulative durations rounded to whole frames.
Each word keeps at least one frame, as at inference.

## Losses

| Loss | Target |
|---|---|
| flow | velocity `z - x0` between noise `x0` (t = 0) and the normalized latents `z` (t = 1) |
| prior | normalized latents against the prior means of their aligned tokens |
| duration | aligned durations, in the log domain |

The condition is set to zero for 10% of the batch (`cond_drop`), so the model
also learns the unconditional velocity that classifier-free guidance needs.

The latents are normalized per dimension by the mean and standard deviation of
the training set. The engine computes them on the first run, stores them in
`rapido_latent_stats.json` beside the dataset and writes them into the model
buffers that the export reads. The stored statistics carry a hash of the latent
caches. When the latents change, the hash changes and the engine computes the
statistics again.

## Train

```bash
# 1. preprocess; --engine rapido caches the AudioVAE latents
python -m phoonnx_train.preprocess -i <dataset> -o <out> \
    --language pt-BR --phoneme-type tugaphone --engine rapido

# 2. train (quality: base = the released size, low = a tiny CI tier)
python -m phoonnx_train.train --dataset-dir <out> --engine rapido --quality base

# 3. export the voice
python -m phoonnx_train.export_onnx \
    lightning_logs/version_0/checkpoints/epoch=<E>-step=<S>.ckpt --engine rapido \
    --config <out>/config.json --output-dir <voice>
```

If you preprocess without `--engine rapido`, the first training run encodes
the latents instead.

The engine always trains uncompiled. The training loop never calls the model's
`forward`, so `--compile` gives no speed-up. The CLI ignores the flag for this
engine and logs a warning.

Engine options go in the `engine_params` object of the dataset `config.json`:

| Option | Default | Effect |
|---|---|---|
| `learning_rate` | `1e-4` | AdamW learning rate |
| `cond_drop` | `0.1` | share of the batch trained without the condition |
| `mas_upsample` | `4` | time resolution of the alignment search |
| `freeze_backbone_steps` | `0` | train only the text encoder, duration predictor, aligner, speakers and prior head for this many steps |
| `text_only` | `false` | train only the text encoder, duration predictor and prior head |

## Fine-tune from the released weights

Give the upstream `rapido.pt` (from `edwixx/rapido-tts`) as a warm start:

```bash
python -m phoonnx_train.train --dataset-dir <out> --engine rapido --quality base \
    --resume-from-checkpoint rapido.pt
```

`rapido.pt` holds weights and no Lightning training state. The CLI therefore
loads it as a warm start and does not resume a run. The engine loads every
tensor with the same shape. It drops the speaker table when the speaker count
differs. It keeps the letter embedding only when the dataset's
`phoneme_id_map` lists the released vocabulary in the same order. The released
model reads Portuguese letters, and a phonemized dataset has an IPA
vocabulary, so for such a dataset the engine drops the embedding and trains it
from the start. Add `--discard-encoder` to drop the embedding in all cases. The
log gives the counts of loaded and dropped tensors.

A warm start does not keep the released latent normalization: the engine
replaces the mean and standard deviation of `rapido.pt` with the statistics of
your training set.

A two-phase recipe that works for a language the released model does not speak: train with
`freeze_backbone_steps` set for the early steps, then continue with
`text_only` to adapt the duration predictor and text encoder.

---

[Docs index](../../index.md) · Previous: [SuperTonic](supertonic.md) · Next: [Indic Parler-TTS](indic_parler.md)
