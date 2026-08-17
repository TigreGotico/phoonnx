# SuperTonic Engine (training)

This page teaches, from zero, how to train a **SuperTonic** text-to-speech model
in phoonnx: what each of its three stages is, how to prepare data, how to train
and resume each stage, how to fine-tune from the released weights, and how to
export the four ONNX graphs the phoonnx `supertonic` runtime consumes.

> Related: [training reference](../training.md) ·
> [adapter architecture](../../engines.md) · [export](../export.md)

## What SuperTonic is

**SuperTonic** (Kim et al., Supertone Inc., *SupertonicTTS*,
[arXiv:2503.23108](https://arxiv.org/abs/2503.23108)) is a fast, multilingual
text-to-speech system. Unlike a single end-to-end model, it is built from **three
networks that are trained one after another**, each solving one piece of the
problem:

1. **Speech autoencoder** — learns to compress a waveform into a compact
   *latent* sequence and reconstruct the waveform from it. A *latent* here is
   just a short numeric summary of a slice of audio: instead of ~22050 numbers
   per second of sound, the autoencoder keeps a few dozen channels at a much
   lower frame rate. It is trained like a vocoder GAN — a *generator* (the
   autoencoder) versus *discriminators* that try to tell real audio from
   reconstructed audio, which pushes the reconstruction to sound natural.
2. **Text-to-latent** — learns to turn text into one of those latent sequences,
   in a chosen voice. It uses **flow matching**: think of it as learning the
   direction to "flow" a cloud of random noise, step by step, until it becomes a
   valid speech latent for the given text. At synthesis time you start from noise
   and take a handful of small steps along the learned directions.
3. **Duration predictor** — learns how *long* the spoken sentence should be (in
   seconds) from the text and the target voice, so the flow-matching stage knows
   how many latent frames to generate.

The text side is **grapheme-level and G2P-free**: SuperTonic reads raw
characters (wrapped in a `<lang>…</lang>` tag), so no phonemizer is involved. The
voice ("style") is supplied at inference as a small per-speaker JSON, so the two
text stages consume a pre-computed style vector rather than a speaker encoder.

## When to pick it

Pick SuperTonic for a fast, compact, multilingual model where you can train all
three stages, and where reading raw text (no phoneme dictionary) is an advantage.
It is an **iterative** engine like ZipVoice (a short sampling loop), but with a
separate learned duration model instead of in-context infilling.

## Install

```bash
pip install phoonnx[train]        # torch, pytorch-lightning, torchaudio
pip install phoonnx[train-resample]  # only if your audio is not already at the model sample rate
```

## Prepare data

SuperTonic is grapheme-level, so it trains directly from **raw text**, not from
phoneme ids. It therefore uses a simple filelist rather than the phoneme-oriented
`dataset.jsonl` other engines use. One utterance per line:

```
audio/utt0001.wav|Hello there, this is a test.|en
audio/utt0002.wav|Another line of speech.|en
```

- Column 1: path to an audio file, relative to a `root_dir` you pass in.
- Column 2: the transcript, exactly as it should be read.
- Column 3 (optional): a language code (`en`, `ko`, `ja`, `ar`, `pt`, … — the 31
  SuperTonic languages, defaulting to `en`).

Any `soundfile`-readable format works; audio at a different sample rate is
resampled on load.

## Train the three stages

Each stage is a separate LightningModule (`phoonnx_train/supertonic/lightning.py`)
and is trained in order, because stages 2 and 3 need the frozen stage-1
autoencoder to produce their target latents.

```python
import pytorch_lightning as pl
from phoonnx_train.supertonic.config import SuperTonicConfig
from phoonnx_train.supertonic.text import CharTokenizer
from phoonnx_train.supertonic.lightning import (
    AutoencoderModule, TextToLatentModule, DurationPredictorModule,
)

cfg = SuperTonicConfig(vocab_size=512)          # or config.tiny_config(...) for a smoke run
tok = CharTokenizer.build_from_texts(open_transcripts, langs)  # build the char->id map
files = ["train.filelist"]
root = "/path/to/audio_root"

# Stage 1 — speech autoencoder (GAN)
ae = AutoencoderModule(config=cfg, dataset=files, root_dir=root, batch_size=8)
pl.Trainer(accelerator="gpu", devices=1, max_steps=400_000).fit(ae)
ae_ckpt = "ae.ckpt"; ae.trainer.save_checkpoint(ae_ckpt)

# Stage 2 — text-to-latent (flow matching), targets from the frozen autoencoder
ttl = TextToLatentModule(config=cfg, tokenizer=tok, dataset=files, root_dir=root,
                         ae_checkpoint=ae_ckpt, batch_size=8)
pl.Trainer(accelerator="gpu", devices=1, max_steps=600_000).fit(ttl)

# Stage 3 — duration predictor
dp = DurationPredictorModule(config=cfg, tokenizer=tok, dataset=files, root_dir=root,
                             ae_checkpoint=ae_ckpt, batch_size=8)
pl.Trainer(accelerator="gpu", devices=1, max_steps=100_000).fit(dp)
```

Stage 1 measures the per-channel statistics of its own latent space at the end of
every epoch and stores them in the checkpoint. Stages 2 and 3 normalize their
targets with those statistics, so they refuse an autoencoder checkpoint that does
not carry them — train stage 1 at least to the end of one epoch before starting
either.

Via the shared CLI/registry the engine name is `supertonic` and the stage is
chosen with `stage=` in the engine's `extra` bag; `--quality low` selects a tiny
architecture for a smoke run, `--quality base` the full size.

### Resuming

Every stage writes standard Lightning checkpoints, which carry the model,
optimizer, LR-scheduler and step — plus the SuperTonic config and tokenizer, so a
checkpoint fully rebuilds the model. Resume with Lightning's `ckpt_path`:

```python
pl.Trainer(...).fit(module, ckpt_path="last.ckpt")   # continues optimizer + step
```

For standalone (non-Lightning) loops, `phoonnx_train/supertonic/checkpointing.py`
provides **atomic** full-checkpoint saves (`save_checkpoint`) and `resume_into`,
which restore model + every optimizer + scheduler + step. A truncated or corrupt
file raises a clear `CheckpointError` rather than loading half a model.

### Fine-tuning from the released `supertonic-3` weights

`phoonnx_train/supertonic/import_onnx.py` reads the initializers out of the
released ONNX graphs (`vocoder.onnx`, `text_encoder.onnx`,
`vector_estimator.onnx`, `duration_predictor.onnx`) and copies them into the
PyTorch modules by an explicit name map, tolerating the conv-1×1↔linear and
linear-transpose conventions of the export. A `PortReport` records what loaded,
what was missing, and any shape mismatch — nothing is copied under a wrong shape.
The encoder-side networks (speech encoder and the two style encoders) are absent
from the public graphs, so they always start from a fresh initialization and must
be trained.

To fine-tune onto a new language whose script is missing from a pretrained
tokenizer, grow the vocab **without disturbing existing rows** so the pretrained
embedding stays aligned:

```python
tok2 = tok.extend_with_texts(new_texts, new_langs)   # appends ids for new chars only
# then load the pretrained weights with checkpointing.load_state_dict_grow_vocab(...)
```

## Export

Export produces exactly the four graphs the `supertonic` inference engine
consumes, with the official input names, plus `tts.json` (runtime config) and
`unicode_indexer.json` (the code-point → id table):

```python
from phoonnx_train.supertonic.export_onnx import export_from_checkpoints
export_from_checkpoints(
    "exported/",
    autoencoder_ckpt="ae.ckpt",
    text_to_latent_ckpt="ttl.ckpt",
    duration_predictor_ckpt="dp.ckpt",
)
```

| Graph | Inputs | Output |
|---|---|---|
| `duration_predictor.onnx` | `text_ids`, `style_dp`, `text_mask` | `duration` |
| `text_encoder.onnx` | `text_ids`, `style_ttl`, `text_mask` | `text_emb` |
| `vector_estimator.onnx` | `noisy_latent`, `text_emb`, `style_ttl`, `text_mask`, `latent_mask`, `current_step`, `total_step` | `latent` |
| `vocoder.onnx` | `latent` | `wav` |

The two text graphs take the **pre-pooled** style tokens directly (`style_dp` /
`style_ttl`), matching the released model: the style encoders are not part of any
graph, and the runtime supplies per-voice style JSONs. The vector-estimator graph
folds one Euler integration step so the runtime loop feeds its output straight
back in as `noisy_latent`; the vocoder graph decompresses and denormalizes the
latent internally before decoding to a waveform.

## Loading the result in phoonnx

Once the four graphs plus `tts.json` and `unicode_indexer.json` are in a voice
directory, they are loaded by the phoonnx `supertonic` inference engine (added
separately). See the runtime engine's own guide for `SynthesisConfig`/voice
wiring.

## Licensing

The training code in `phoonnx_train/supertonic/` is written from scratch against
the SuperTonic paper and the public `Supertone/supertonic-3` ONNX graphs. The
released **weights** are OpenRAIL-M licensed; fine-tuning from them inherits that
license.

## Upstream

| | |
|---|---|
| Paper | *SupertonicTTS* — [arXiv:2503.23108](https://arxiv.org/abs/2503.23108) |
| Weights | [`Supertone/supertonic-3`](https://huggingface.co/Supertone/supertonic-3) (OpenRAIL-M) |
| Languages | 31 (`en`, `ko`, `ja`, `ar`, `bg`, `cs`, `da`, `de`, `el`, `es`, `et`, `fi`, `fr`, `hi`, `hr`, `hu`, `id`, `it`, `lt`, `lv`, `nl`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sv`, `tr`, `uk`, `vi`) |
