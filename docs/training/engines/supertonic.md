# SuperTonic Engine (training)

<!-- agent-hint
covers: how to train the three SuperTonic stages, fine-tune them and export the four ONNX graphs
source: phoonnx_train/supertonic/lightning.py, phoonnx_train/supertonic/text.py, phoonnx_train/supertonic/export_onnx.py, phoonnx_train/supertonic/checkpointing.py, phoonnx_train/supertonic/import_onnx.py, phoonnx_train/supertonic/dataset.py, phoonnx_train/engines/supertonic.py
generated: no
verify: .venv/bin/python -W ignore -c "import inspect, re, sys; from phoonnx_train.supertonic import lightning, text; p = open('docs/training/engines/supertonic.md').read(); src = open('phoonnx_train/supertonic/export_onnx.py').read(); norm = lambda s: re.sub(r'\s+', ' ', s); langs = re.findall(r'\x60([a-z]+)\x60', p.split(chr(10) + '| Languages |')[1].split(chr(10))[0]); bad = [] if langs == [x for x in text.AVAILABLE_LANGS if x != 'na'] else ['languages']; rows = [[re.findall(r'\x60([a-z_]+)\x60', c) for c in r.split('|')[2:4]] for r in p.split(chr(10) + '| Graph | Inputs | Output |')[1].split(chr(10) + chr(10))[0].splitlines()[2:]]; bad += [r for r in rows if norm(str(r[0]) + ', ' + str(r[1])).replace(chr(39), chr(34)) not in norm(src)]; calls = re.findall(r'(\w+Module)\(([^)]*)\)', p); bad += [(c, k) for c, args in calls for k in re.findall(r'(\w+)=', args) if k not in inspect.signature(getattr(lightning, c)).parameters]; print(len(rows), len(calls), bad); sys.exit(bool(bad) or not rows or not calls)"
see: ../training.md, ../export.md, ../../engines.md
-->

This page shows how to train a SuperTonic model from the start, stage by stage, and export it for the phoonnx `supertonic` runtime.

See also the [training reference](../training.md), the [adapter architecture](../../engines.md) and [Export](../export.md).

## What SuperTonic is

SuperTonic (Kim et al., Supertone Inc., *SupertonicTTS*, [arXiv:2503.23108](https://arxiv.org/abs/2503.23108)) is a multilingual text-to-speech system. It has three networks, which you train one after another:

1. The speech autoencoder compresses a waveform into a latent sequence and rebuilds the waveform from it. A latent is a short numeric summary of a slice of audio, with a few dozen channels at a frame rate much lower than the sample rate. It trains as a vocoder GAN: the autoencoder is the generator, and discriminators try to tell real audio from rebuilt audio.
2. The text-to-latent model turns text into a latent sequence in a given voice. It uses flow matching: it learns how to move random noise, step by step, to a speech latent for the text. At synthesis, it starts from noise and takes a few steps.
3. The duration predictor gives the length of the sentence in seconds, from the text and the voice. The text-to-latent stage then knows how many latent frames to make.

SuperTonic reads characters, not phonemes. Each text goes in a `<lang>…</lang>` tag, and no phonemizer runs. The voice ("style") comes at inference from a small JSON file for each speaker, so the two text stages read a computed style vector and not a speaker encoder.

Use SuperTonic for a small multilingual model when you can train all three stages, and when reading characters is an advantage. Like ZipVoice, it runs a short sampling loop. Unlike ZipVoice, it has a separate duration model.

## Install

```bash
pip install --pre "phoonnx[train,train-resample]"   # torch, pytorch-lightning, torchaudio
```

## 1. Prepare the data

SuperTonic reads text, not phoneme ids, so it uses a file list and not the `dataset.jsonl` of the other engines. Write one utterance per line:

```
audio/utt0001.wav|Hello there, this is a test.|en
audio/utt0002.wav|Another line of speech.|en
```

The first field is the path of an audio file, relative to the `root_dir` that you give. The second field is the transcript as it must be read. The third field is optional: a language code from the [language list](#upstream). It defaults to `en`.

Any format that `soundfile` reads works. The loader resamples audio at another sample rate.

## 2. Train the three stages

Each stage is a separate LightningModule in `phoonnx_train/supertonic/lightning.py`. Train them in order, because stages 2 and 3 use the frozen stage 1 autoencoder to make their target latents.

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

# Stage 1: speech autoencoder (GAN)
ae = AutoencoderModule(config=cfg, dataset=files, root_dir=root, batch_size=8)
pl.Trainer(accelerator="gpu", devices=1, max_steps=400_000).fit(ae)
ae_ckpt = "ae.ckpt"; ae.trainer.save_checkpoint(ae_ckpt)

# Stage 2: text-to-latent (flow matching), targets from the frozen autoencoder
ttl = TextToLatentModule(config=cfg, tokenizer=tok, dataset=files, root_dir=root,
                         ae_checkpoint=ae_ckpt, batch_size=8)
pl.Trainer(accelerator="gpu", devices=1, max_steps=600_000).fit(ttl)

# Stage 3: duration predictor
dp = DurationPredictorModule(config=cfg, tokenizer=tok, dataset=files, root_dir=root,
                             ae_checkpoint=ae_ckpt, batch_size=8)
pl.Trainer(accelerator="gpu", devices=1, max_steps=100_000).fit(dp)
```

At the end of every epoch, stage 1 measures the statistics of each channel of its latent space and stores them in the checkpoint. Stages 2 and 3 normalize their targets with these statistics, so they refuse an autoencoder checkpoint without them. Train stage 1 to the end of at least one epoch before you start stage 2 or 3.

The training engine registry has the name `supertonic`. It selects the stage with `stage` in the engine `extra` (`autoencoder`, `text_to_latent` or `duration_predictor`). `--quality low` selects a very small model for tests, and `--quality base` the full size. The `phoonnx_train.train` command cannot train it, because it gives the stages the dataset folder, and the stages read a file list. Train SuperTonic from Python, as shown above.

### Resuming

Every stage writes standard Lightning checkpoints. A checkpoint holds the model, the optimizer, the scheduler and the step, and also the SuperTonic config and the tokenizer, so it rebuilds the full model. Resume with the `ckpt_path` of Lightning:

```python
pl.Trainer(...).fit(module, ckpt_path="last.ckpt")   # continues optimizer + step
```

For training loops without Lightning, `phoonnx_train/supertonic/checkpointing.py` has `save_checkpoint`, which writes a full checkpoint atomically, and `resume_into`, which restores the models, every optimizer, the scheduler and the step. A truncated or damaged file raises `CheckpointError`, so half a model never loads.

### Fine-tuning from the released `supertonic-3` weights

`phoonnx_train/supertonic/import_onnx.py` reads the weights from the released ONNX graphs (`vocoder.onnx`, `text_encoder.onnx`, `vector_estimator.onnx` and `duration_predictor.onnx`). It copies them into the PyTorch modules through an explicit name map, and handles the 1×1 convolution, linear and transposed-linear forms of the export. A `PortReport` records what loaded, what was missing and each shape mismatch. Nothing is copied with a wrong shape. The public graphs do not hold the speech encoder or the two style encoders, so these always start from new weights and you must train them.

To fine-tune on a language whose characters a pretrained tokenizer lacks, add ids to the tokenizer and keep the existing ids, so the pretrained embedding stays aligned:

```python
tok2 = tok.extend_with_texts(new_texts, new_langs)   # appends ids only for chars the map lacks
# then load the pretrained weights with checkpointing.load_state_dict_grow_vocab(...)
```

## 3. Export

The export writes the four graphs that the `supertonic` inference engine reads, with the official input names. It also writes `tts.json`, the runtime config, and `unicode_indexer.json`, the table from code point to id:

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

The two text graphs take the pooled style tokens (`style_dp` and `style_ttl`) as the released model does. No graph holds the style encoders, and the runtime gives a style JSON for each voice. The vector estimator graph does one Euler step, so the runtime loop feeds its output back as `noisy_latent`. The vocoder graph decompresses and denormalizes the latent before it decodes the waveform.

`phoonnx_train.export_onnx --engine supertonic` passes only one checkpoint, so use `export_from_checkpoints`.

## Load the result in phoonnx

Put the four graphs, `tts.json` and `unicode_indexer.json` in one voice folder. The phoonnx `supertonic` inference engine loads them. The [engine architecture](../../engines.md) page describes the adapters.

## Licensing

The training code in `phoonnx_train/supertonic/` is written from the SuperTonic paper and the public `Supertone/supertonic-3` ONNX graphs. The released weights have the OpenRAIL-M licence, and a model fine-tuned from them has the same licence.

## Upstream

| | |
|---|---|
| Paper | *SupertonicTTS*, [arXiv:2503.23108](https://arxiv.org/abs/2503.23108) |
| Weights | [`Supertone/supertonic-3`](https://huggingface.co/Supertone/supertonic-3) (OpenRAIL-M) |
| Languages | 31 (`en`, `ko`, `ja`, `ar`, `bg`, `cs`, `da`, `de`, `el`, `es`, `et`, `fi`, `fr`, `hi`, `hr`, `hu`, `id`, `it`, `lt`, `lv`, `nl`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sv`, `tr`, `uk`, `vi`) |

The tokenizer also accepts the tag `na`.

---

[← OmniVoice](omnivoice.md) · [Home](../../index.md) · [Rápido →](rapido.md)
