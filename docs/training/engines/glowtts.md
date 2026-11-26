# GlowTTS Engine (Larynx)

<!-- agent-hint
covers: how to load a GlowTTS voice with its vocoder, convert a Larynx or Coqui voice and train a mel model
source: phoonnx/engines/glowtts.py, phoonnx/engines/glowtts_config.py, phoonnx/engines/coqui_config.py, scripts/conversion/coqui_glowtts_export/export.py, phoonnx_train/engines/glowtts.py, phoonnx/voice_index/glowtts.json
generated: no
verify: .venv/bin/python -c "import re,sys,json; from phoonnx.engines.glowtts import GlowTTSAdapter as A; from phoonnx_train.engines.glowtts import _QUALITY_PRESETS as Q; p=open('docs/training/engines/glowtts.md').read().replace(chr(96),''); d=dict(re.findall(r'^\| ([a-z_]+) \| ([\d.]+) \| [A-Z]',p,re.M)); q={r[0]:tuple(map(int,r[1:])) for r in re.findall(r'^\| ([\w-]+) \| (\d+) \| (\d+) \| (\d+) / (\d+) \| (\d+) / (\d+) \|',p,re.M)}; idx=json.load(open('phoonnx/voice_index/glowtts.json')); n=int(re.search(r'holds (\d+) voices',p).group(1)); ids=set(re.findall(r'\"((?:larynx/|coqui-)[\w-]+)\"',p)); sys.exit(0 if d=={k:str(v) for k,v in A().default_params().items()} and q=={k:(v['hidden_channels'],v['filter_channels'],v['n_heads'],v['n_layers'],v['dec_n_blocks'],v['dec_n_layers']) for k,v in Q.items()} and n==len(idx) and ids<=set(idx) else 1)"
see: matcha.md, ../training.md, ../../vocoders.md
-->

This page shows how to load a GlowTTS voice with its vocoder, convert a Larynx or Coqui voice, and train and export a GlowTTS mel model.

## What it is

GlowTTS is a flow-based acoustic model that makes a mel spectrogram from phoneme IDs. [Larynx](https://github.com/rhasspy/larynx), the project before Mimic 3 and Piper, made it well known. Like Matcha-TTS, GlowTTS is a two-stage engine: a separate vocoder turns the mel into a waveform. The adapter uses the vocoders of [`phoonnx.engines.vocoders`](../../vocoders.md).

Pick GlowTTS to run Larynx or Coqui-TTS GlowTTS voices with ONNX Runtime only, with no `coqui-tts` package. You can also train a small flow-based voice and pair it with a mel-matched vocoder or with Griffin-Lim.

## Install

Larynx voices phonemize with gruut. Install the language extra of the voice, for example `pip install --pre "phoonnx[en]"`, which installs `gruut[en]`. Training uses the `train` extra: `pip install --pre "phoonnx[train]"`.

## Load an indexed voice

`phoonnx/voice_index/glowtts.json` holds 58 voices. Each voice has its own Hub repository under `Phoonnx/`, with the model and a native config. The index entry also names a vocoder with `vocoder_url` and `vocoder_type`, and the voice manager downloads it with the model:

```python
from phoonnx.model_manager import TTSModelManager

m = TTSModelManager()
m.merge_default_voices()
voice = m.voices["larynx/en-us-ljspeech-glow_tts"].load()
for chunk in voice.synthesize("Hello from GlowTTS."):
    audio = chunk.audio_float_array
```

The Larynx voices have ids such as `"larynx/en-us-ljspeech-glow_tts"`, and the Coqui voices have ids such as `"coqui-en-ljspeech-glow-tts"`.

## Convert a voice

### Larynx

A Larynx GlowTTS voice ships a training `config.json` and a `phonemes.txt` symbol table, with one `<id> <phoneme>` pair per line. `voice_config_from_larynx` turns them into a native `VoiceConfig` with the gruut phonemizer, blank-interspersed tokens and the audio settings:

```python
import json
from phoonnx.engines.glowtts_config import voice_config_from_larynx

cfg = json.load(open("config.json"))
config = voice_config_from_larynx(cfg, open("phonemes.txt").read(), lang_code="en-us")
```

The indexed voices already ship this native `config.json` (`engine: glowtts`), so they load through the standard path.

### Coqui

`scripts/conversion/coqui_glowtts_export/export.py` converts a Coqui-TTS GlowTTS model to ONNX without the `coqui-tts` package. It vendors only the pure-torch encoder and decoder, and it gives an ONNX graph with the Larynx inputs and outputs. The paired Coqui vocoder (HiFi-GAN or multiband MelGAN) is converted in the same way. A model with no paired vocoder uses Griffin-Lim. `voice_config_from_coqui` (in `phoonnx.engines.coqui_config`, also importable from `phoonnx.engines.glowtts_config`) builds the native config, with graphemes, or with espeak when the model sets `use_phonemes`.

## Train a model

Training uses the standard `phoonnx_train` preprocessing, which GlowTTS shares with VITS. The model is a pure-torch GlowTTS in `phoonnx_train/glowtts/`, with no `coqui-tts` dependency. `phoonnx_train/glowtts/__init__.py` gives its provenance: a reimplementation of the GlowTTS paper (Kim et al. 2020), with the training math checked against jaywalnut310/glow-tts (MIT).

1. Preprocess an LJSpeech-style dataset:

   ```bash
   python -m phoonnx_train.preprocess \
     --input-dir /data/my-dataset \
     --output-dir /data/preprocessed \
     --language en-us
   ```

2. Train:

   ```bash
   python -m phoonnx_train.train \
     --dataset-dir /data/preprocessed \
     --engine glowtts \
     --quality medium \
     --batch-size 16 \
     --max-epochs 1000 \
     --default-root-dir /data/preprocessed
   ```

   Without `--default-root-dir`, PyTorch Lightning writes `lightning_logs/` in the current folder. The trainer keeps the checkpoint of every save and writes no `last.ckpt`.

3. Export the mel model:

   ```bash
   python -m phoonnx_train.export_onnx \
     /data/preprocessed/lightning_logs/version_0/checkpoints/epoch=999-step=250000.ckpt \
     --engine glowtts \
     --config /data/preprocessed/config.json \
     --output-dir ./onnx
   ```

   The export writes the mel model `<checkpoint-stem>.onnx` and its config `<checkpoint-stem>.json`. The graph has the inputs and the output of the contract below. The export never makes a vocoder.

The mel basis uses fmin 0 Hz and fmax 8000 Hz, the same as the HiFi-GAN vocoder configs.

The model has a phoneme embedding, a convolutional prenet and a Transformer text encoder that gives a Gaussian prior per token. A convolutional duration predictor sits beside the encoder. The decoder is an invertible normalizing flow between the mel and a latent. Training uses Monotonic Alignment Search, an MLE loss and an MSE duration loss.

To train a multi-speaker model, set `num_speakers` above 1 in the training config. A speaker embedding then conditions the duration predictor and the coupling layers of the flow. Its size is `gin_channels`, 512 by default.

## Run your exported model

Add the vocoder to the `engine_params` of the exported config. `GlowTTSAdapter` reads `vocoder_path` and `vocoder_type`:

```json
{
    "engine": "glowtts",
    "engine_params": {
        "vocoder_path": "hifigan.onnx",
        "vocoder_type": "hifigan"
    }
}
```

Put the config next to the exported `.onnx` mel model in a folder, then set that folder as the `voice` of `ovos-tts-plugin-phoonnx` (see [OVOS plugin](../../ovos_plugin.md)):

```json
{
  "tts": {
    "module": "ovos-tts-plugin-phoonnx",
    "ovos-tts-plugin-phoonnx": {
      "lang": "en-US",
      "voice": "/home/user/.local/share/phoonnx/voices/my-glowtts-voice"
    }
  }
}
```

## Vocoders

Each indexed voice names its vocoder (see [Vocoders](../../vocoders.md)):

- A neural vocoder (`vocoder_type: hifigan` or `melgan`) is an ONNX file in its own `Phoonnx/` repository, downloaded with the model. 51 Larynx voices share `Phoonnx/hifigan-multi-vctk-small`. The Coqui voices use the vocoder that Coqui paired with the model.
- Griffin-Lim (`vocoder_type: griffinlim`) needs no model file. The voices with no mel-matched neural vocoder use it. Its config carries the mel settings (`ref_level_db`, `spec_gain`, `max_norm`) so that a Coqui mel inverts correctly.

## Notes

- The config `engine` field may be `glowtts`, `glow_tts` or `larynx`. A config with `model_type: glow_tts` also selects this adapter.
- GlowTTS and VITS both take a `scales` input. The registry probes GlowTTS before VITS, and the probe needs a mel output.
- Larynx voices phonemize with gruut (`phoneme_type: gruut`, `alphabet: ipa`). The tokenizer puts a blank (PAD, id 0) between symbols and adds no BOS or EOS.
- A Larynx mel is signal-normalized, and a Coqui mel is log-domain. The adapter reproduces the Larynx mel post-processing for the first kind and passes a log-domain mel to the vocoder as it is.

## Reference

### ONNX inputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `input` | int64 | `[B, T]` | Phoneme IDs |
| `input_lengths` | int64 | `[B]` | Sequence lengths |
| `scales` | float32 | `[2]` | `[noise_scale, length_scale]` |
| `sid` | int64 | `[B]` | Speaker ID, only when the graph declares it |

The output is a mel spectrogram `[B, n_mels, T]`. A Larynx graph also outputs a second rank-3 tensor. The adapter takes the first rank-3 tensor with a mel axis.

### Synthesis parameters

| Param | Default | Description |
|-------|---------|-------------|
| `noise_scale` | 0.667 | Flow sampling temperature |
| `length_scale` | 1.0 | Speech rate (higher is slower) |

### Quality presets

| Preset | hidden_channels | filter_channels | heads / layers | decoder blocks / layers |
|--------|-----------------|------------------|-----------------|--------------------------|
| `x-low` | 96 | 384 | 2 / 4 | 8 / 3 |
| `medium` | 192 | 768 | 2 / 6 | 12 / 4 |
| `high` | 256 | 1024 | 4 / 8 | 12 / 4 |

## References

- [Larynx](https://github.com/rhasspy/larynx)
- [GlowTTS paper](https://arxiv.org/abs/2005.11129)

---

[← Matcha-TTS](matcha.md) · [Home](../../index.md) · [Mixer-TTS →](mixertts.md)
