# GlowTTS Engine (Larynx)

This page is for voice builders and integrators working with GlowTTS / Larynx
voices in phoonnx. After reading it you can load a GlowTTS voice with its
vocoder, convert a Larynx or Coqui voice, and train and export your own mel
model.

> Related: [training reference](../training.md) ·
> [adapter architecture](../../engines.md) ·
> [configuration](../../configuration.md) · [vocoders](../../vocoders.md) ·
> [Matcha — the other two-stage engine](matcha.md)

## What it is

GlowTTS is a **flow-based** acoustic model (text → mel spectrogram), best known
from [Larynx](https://github.com/rhasspy/larynx) — the precursor to Mimic3 and
Piper. Like Matcha-TTS it is **two-stage**: a separate vocoder (Larynx ships
HiFi-GAN) turns the mel into a waveform, so the adapter reuses
[`phoonnx.engines.vocoders`](../../vocoders.md).

## When to pick it

Choose GlowTTS to run existing Larynx or Coqui-TTS GlowTTS voices in a
pure-ONNX pipeline (no `coqui-tts` / `TTS` dependency), or to train a compact
flow-based voice where you want to pair the mel model with a mel-matched
neural vocoder or a universal Griffin-Lim fallback.

## Extras needed

Inference: phonemization uses **gruut** (`pip install gruut`, or the relevant
language extra). Training uses the `train` extra: `pip install phoonnx[train]`.

## Inference contract

### ONNX inputs (glow_tts generator)

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `input` | int64 | `[B, T]` | Phoneme IDs (gruut) |
| `input_lengths` | int64 | `[B]` | Sequence lengths |
| `scales` | float32 | `[2]` | `[noise_scale, length_scale]` |
| `sid` | int64 | `[B]` | Speaker ID (**optional**, multi-speaker models only) |

The `sid` input is fed only when the loaded graph declares it; single-speaker
voices omit it.

### ONNX outputs

A mel spectrogram `[B, n_mels, T]`. Larynx also emits an extra intermediate
tensor; the adapter finds the mel by its `n_mels` axis rather than by output
position, then runs the vocoder.

> GlowTTS shares the `scales` input with VITS, so the adapter is probed before
> VITS — it is distinguished by its **mel** (not waveform) output.

Parameters:

| Param | Default | Description |
|-------|---------|-------------|
| `noise_scale` | 0.667 | Flow sampling temperature |
| `length_scale` | 1.0 | Speech rate (higher = slower) |

## Obtaining / converting / training

### Loading an indexed voice

GlowTTS voices ship in `phoonnx/voice_index/glowtts.json`, mirrored under
`OpenVoiceOS/phoonnx-glowtts` (model + native config) with the HiFi-GAN vocoder
under `OpenVoiceOS/phoonnx-vocoders` (linked per entry via `vocoder_url`,
`vocoder_type: hifigan`):

```python
from phoonnx.model_manager import TTSModelManager

m = TTSModelManager(); m.merge_default_voices()
voice = m.voices["larynx/en-us-ljspeech-glow_tts"].load()  # downloads model + vocoder
for chunk in voice.synthesize("Hello from GlowTTS."):
    ...
```

### Larynx voice → native config

A Larynx GlowTTS voice ships a training `config.json` (audio + model params) and
a `phonemes.txt` symbol table (`<id> <phoneme>` per line, gruut IPA).
`phoonnx.engines.glowtts_config.voice_config_from_larynx` turns those into a
native phoonnx `VoiceConfig` (gruut phonemizer, blank-interspersed tokenization,
mel/audio params):

```python
import json
from phoonnx.engines.glowtts_config import voice_config_from_larynx

cfg = json.load(open("config.json"))
config = voice_config_from_larynx(cfg, open("phonemes.txt").read(), lang_code="en-us")
```

The mirrored voices ship this as a native `config.json` (`engine: glowtts`), so
they load through the standard path.

### Coqui voices

Coqui-TTS GlowTTS models (`coqui/…` ids) are converted to phoonnx ONNX without
the coqui-tts package: a standalone exporter vendors only the pure-torch
`Encoder`/`Decoder` and replicates `GlowTTS.inference` (pre-inverting the flow
1×1 convs). Their paired `default_vocoder` (HiFi-GAN / multiband-MelGAN) is
converted the same way; models with no paired vocoder use Griffin-Lim.
`phoonnx.engines.glowtts_config.voice_config_from_coqui` builds the native
config (graphemes, or espeak when `use_phonemes`).

### Training

Trainable with `--engine glowtts`. Training uses `phoonnx_train`'s standard
preprocessing pipeline (phonemization + audio normalization + linear-spectrogram
extraction, shared with VITS) and a self-contained, pure-torch GlowTTS
implementation vendored under `phoonnx_train/glowtts/` — no `coqui-tts` / `TTS`
dependency. See `phoonnx_train/glowtts/__init__.py` for the full provenance
note: a reimplementation from the published GlowTTS paper architecture (Kim et
al. 2020), with the training math audited against the original reference
implementation (jaywalnut310/glow-tts, MIT). The mel basis is pinned to fmin 0 /
fmax 8000 Hz (matching the HiFi-GAN-family vocoder configs) and recorded in the
exported ONNX metadata.

```bash
# 1. preprocess an LJSpeech-style dataset (shared with VITS)
python phoonnx_train/preprocess.py \
  --input-dir /data/my-dataset \
  --output-dir /data/preprocessed \
  --language en-us

# 2. train
python phoonnx_train/train.py \
  --dataset-dir /data/preprocessed \
  --engine glowtts \
  --quality medium \
  --batch-size 16 \
  --max-epochs 1000

# 3. export the mel model to ONNX
python phoonnx_train/export_onnx.py \
  --engine glowtts \
  --config /data/preprocessed/config.json \
  --output-dir ./onnx \
  /data/preprocessed/lightning_logs/version_0/checkpoints/last.ckpt
```

`export_onnx` produces **only the mel model** (`<checkpoint-stem>.onnx`), with
the exact input/output contract above (`input` / `input_lengths` / `scales` →
`[B, n_mels, T]` mel). You still need a **separate vocoder** ONNX to synthesize
audio; this engine never produces one.

Quality presets:

| Preset | hidden_channels | filter_channels | heads / layers | decoder blocks / layers |
|--------|-----------------|------------------|-----------------|--------------------------|
| `x-low` | 96 | 384 | 2 / 4 | 8 / 3 |
| `medium` | 192 | 768 | 2 / 6 | 12 / 4 |
| `high` | 256 | 1024 | 4 / 8 | 12 / 4 |

Architecture: a phoneme-embedding → conv prenet → Transformer text encoder
(shared with VITS's own text encoder) producing a per-token Gaussian prior plus
a conv duration predictor; an invertible normalizing-flow decoder mapping mel ↔
latent; Monotonic Alignment Search (reusing the VITS-vendored MAS kernel); MLE
loss plus an MSE duration loss.

**Multi-speaker:** set `num_speakers > 1` in the shared `TrainingEngineConfig`; a
speaker embedding conditions both the duration predictor and the flow's affine
coupling layers (`gin_channels`, default 512, overridable via `extra`).

## Synthesis example

Point a local voice `config.json` at the exported mel model and a vocoder using
the exact `engine_params` keys `GlowTTSAdapter.configure_from_params`
(`phoonnx/engines/glowtts.py`) reads — `vocoder_path` and `vocoder_type`:

```json
{
    "engine": "glowtts",
    "engine_params": {
        "vocoder_path": "hifigan.onnx",
        "vocoder_type": "hifigan"
    }
}
```

Place this `config.json` next to the exported `.onnx` mel model in a local voice
directory, then point `ovos-tts-plugin-phoonnx` at it via the plugin's `voice`
setting (see [ovos_plugin.md](../../ovos_plugin.md)):

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

GlowTTS is two-stage, so each indexed voice links a vocoder (see
[vocoders.md](../../vocoders.md)):

- **Neural** (`vocoder_type: hifigan` / `melgan`) — an ONNX vocoder under
  `OpenVoiceOS/phoonnx-vocoders`, downloaded alongside the model. Best quality;
  used where a **mel-matched** vocoder exists (Larynx HiFi-GAN; Coqui models'
  paired `default_vocoder`).
- **Griffin-Lim** (`vocoder_type: griffinlim`) — a parametric fallback (no model
  file) for voices with no mel-matched neural vocoder. Robotic but universal;
  its config carries the mel params (`ref_level_db` / `spec_gain` / `max_norm` …)
  so coqui-domain mels invert correctly.

## Gotchas / aliases

- **Detect aliases:** the config `engine` field may be `glowtts`, `glow_tts` or
  `larynx`; a config with `model_type: glow_tts` is also detected.
- **Text processing:** GlowTTS/Larynx phonemizes with **gruut**
  (`phoneme_type: gruut`, `alphabet: ipa`) and interleaves a blank (PAD, id 0)
  between symbols (`add_blank`), with no BOS/EOS. The 46-symbol table comes from
  the voice's `phonemes.txt`, folded into `phoneme_id_map`.
- **Mel domain:** Larynx glow_tts emits a signal-normalized mel; Coqui glow_tts
  emits a log-domain mel. The adapter reproduces larynx's mel post-processing so
  each domain feeds its vocoder correctly.

## References

- [Larynx](https://github.com/rhasspy/larynx) · [GlowTTS paper](https://arxiv.org/abs/2005.11129)
