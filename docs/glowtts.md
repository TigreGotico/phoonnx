# GlowTTS Engine (Larynx)

GlowTTS is a **flow-based** acoustic model (text → mel spectrogram), best known
from [Larynx](https://github.com/rhasspy/larynx) — the precursor to Mimic3 and
Piper. Like Matcha-TTS it is **two-stage**: a separate vocoder (Larynx ships
HiFi-GAN) turns the mel into a waveform, so the adapter reuses
[`phoonnx.engines.vocoders`](./engines.md).

## Inference

### ONNX inputs (glow_tts generator)

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| ``input`` | int64 | ``[B, T]`` | Phoneme IDs (gruut) |
| ``input_lengths`` | int64 | ``[B]`` | Sequence lengths |
| ``scales`` | float32 | ``[2]`` | ``[noise_scale, length_scale]`` |

### ONNX outputs

A mel spectrogram ``[B, n_mels, T]``. Larynx also emits an extra intermediate
tensor; the adapter finds the mel by its ``n_mels`` axis rather than by output
position, then runs the vocoder.

> GlowTTS shares the ``scales`` input with VITS, so the adapter is probed before
> VITS — it is distinguished by its **mel** (not waveform) output.

### Parameters

| Param | Default | Description |
|-------|---------|-------------|
| ``noise_scale`` | 0.667 | Flow sampling temperature |
| ``length_scale`` | 1.0 | Speech rate (higher = slower) |

## Config — Larynx voice → native config

A Larynx GlowTTS voice ships a training ``config.json`` (audio + model params)
and a ``phonemes.txt`` symbol table (``<id> <phoneme>`` per line, gruut IPA).
``phoonnx.engines.glowtts_config.voice_config_from_larynx`` turns those into a
native phoonnx ``VoiceConfig`` (gruut phonemizer, blank-interspersed
tokenization, mel/audio params)::

    import json
    from phoonnx.engines.glowtts_config import voice_config_from_larynx

    cfg = json.load(open("config.json"))
    config = voice_config_from_larynx(cfg, open("phonemes.txt").read(), lang_code="en-us")

The mirrored voices ship this as a native ``config.json`` (``engine: glowtts``),
so they load through the standard path.

## Voice index

GlowTTS voices ship in ``phoonnx/voice_index/glowtts.json``, mirrored under
``OpenVoiceOS/phoonnx-glowtts`` (model + native config) with the HiFi-GAN
vocoder under ``OpenVoiceOS/phoonnx-vocoders`` (linked per entry via
``vocoder_url``, ``vocoder_type: hifigan``). They load like any other voice:

```python
from phoonnx.model_manager import TTSModelManager

m = TTSModelManager(); m.merge_default_voices()
voice = m.voices["larynx/en-us-ljspeech-glow_tts"].load()  # downloads model + vocoder
for chunk in voice.synthesize("Hello from GlowTTS."):
    ...
```

## Vocoders

GlowTTS is two-stage, so each indexed voice links a vocoder. See
[vocoders.md](./vocoders.md) for the full vocoder system (types, config flags,
how to use/replace/add vocoders). In brief:

- **Neural** (``vocoder_type: hifigan`` / ``melgan``) — an ONNX vocoder under
  ``OpenVoiceOS/phoonnx-vocoders``, downloaded alongside the model. Best quality;
  used where a **mel-matched** vocoder exists (Larynx HiFi-GAN; coqui models'
  paired ``default_vocoder``).
- **Griffin-Lim** (``vocoder_type: griffinlim``) — a parametric fallback (no model
  file) for voices with no mel-matched neural vocoder. Robotic but universal; its
  config carries the mel params (``ref_level_db``/``spec_gain``/``max_norm``…) so
  coqui-domain mels invert correctly.

## Coqui voices

Coqui-TTS GlowTTS models (``coqui/…`` ids) are converted to phoonnx ONNX without
the coqui-tts package: a standalone exporter vendors only the pure-torch
``Encoder``/``Decoder`` and replicates ``GlowTTS.inference`` (pre-inverting the
flow 1×1 convs). Their paired ``default_vocoder`` (HiFi-GAN / multiband-MelGAN)
is converted the same way; models with no paired vocoder use Griffin-Lim.
``phoonnx.engines.glowtts_config.voice_config_from_coqui`` builds the native
config (graphemes, or espeak when ``use_phonemes``).

## Text processing

GlowTTS/Larynx phonemizes with **gruut** (``phoneme_type: gruut``,
``alphabet: ipa``) and interleaves a blank (PAD, id 0) between symbols
(``add_blank``), with no BOS/EOS. The 46-symbol table comes from the voice's
``phonemes.txt``, folded into the native config's ``phoneme_id_map``.

> Requires the ``gruut`` package for phonemization.

## Training

Training uses `phoonnx_train`'s standard preprocessing pipeline
(phonemization + audio normalization + linear-spectrogram extraction, shared
with VITS) and a self-contained, pure-torch GlowTTS implementation
vendored under `phoonnx_train/glowtts/` — no `coqui-tts` / `TTS` dependency.
See `phoonnx_train/glowtts/__init__.py` for the full provenance note: this
is a **reimplementation from the published GlowTTS paper architecture**
(Kim et al. 2020) and general knowledge of GlowTTS-style implementations,
**not a verified line-by-line port** of coqui-TTS (MPL-2.0) source — the
coding agent that authored it did not have network access to diff against
the actual upstream source. The training math has since been audited against
the original reference implementation (jaywalnut310/glow-tts, MIT) — see the
fidelity notes in `phoonnx_train/glowtts/__init__.py`. The mel basis is
pinned to fmin 0 / fmax 8000 Hz (matching the HiFi-GAN-family vocoder
configs) and recorded in the exported ONNX metadata.

Install the `train` extra: `pip install phoonnx[train]`.

### Quick start

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

`export_onnx` produces **only the mel model** (`<checkpoint-stem>.onnx`), with the
exact input/output contract the `GlowTTSAdapter` above expects (`input` /
`input_lengths` / `scales` → `[B, n_mels, T]` mel). You still need a
**separate vocoder** ONNX (HiFi-GAN / Vocos / Griffin-Lim — see
[vocoders.md](./vocoders.md)) to synthesize audio; this engine never
produces one.

### Quality presets

| Preset | hidden_channels | filter_channels | heads / layers | decoder blocks / layers |
|--------|-----------------|------------------|-----------------|--------------------------|
| `x-low` | 96 | 384 | 2 / 4 | 8 / 3 |
| `medium` | 192 | 768 | 2 / 6 | 12 / 4 |
| `high` | 256 | 1024 | 4 / 8 | 12 / 4 |

These roughly mirror VITS's own x-low/medium/high split in this repo (a
GlowTTS-descended architecture); the exact upstream coqui-TTS GlowTTS
dimensions were not independently verified.

### Architecture overview

- **Text encoder** — phoneme embedding → conv "prenet" → Transformer encoder
  (shared verbatim with VITS's own text encoder,
  `phoonnx_train/vits/attentions.py`) → per-token Gaussian prior (`m`, `logs`)
  + a small conv duration predictor.
- **Flow decoder** — an invertible normalizing flow (squeeze → stacked
  ActNorm → invertible 1×1 conv → WN-conditioned affine coupling →
  unsqueeze), mapping mel ↔ latent exactly, run forward at training time and
  in reverse at inference/export time.
- **Monotonic Alignment Search (MAS)** — a dynamic-programming search for the
  most probable monotonic text↔mel alignment under the current model,
  reusing the compiled MAS kernel already vendored for VITS
  (`phoonnx_train/vits/monotonic_align`), which implements the same
  algorithm GlowTTS introduced.
- **Losses** — exact negative log-likelihood of the target mel under the
  flow-transformed Gaussian prior (MLE loss), plus an MSE duration loss
  against MAS-derived log-durations.

### Multi-speaker

Set `num_speakers > 1` in the shared `TrainingEngineConfig`; a speaker
embedding conditions both the duration predictor and the flow's affine
coupling layers (`gin_channels`, default 512, overridable via `extra`).

### Downstream: OVOS `ovos-tts-plugin-phoonnx` config

Once you have the exported mel model (`glowtts.ckpt.onnx`) and a vocoder
ONNX (e.g. a HiFi-GAN vocoder — see [vocoders.md](./vocoders.md) for how to
train/obtain one), point a local voice `config.json` at both using the exact
`engine_params` keys `GlowTTSAdapter.configure_from_params`
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

Place this `config.json` next to the exported `.onnx` mel model in a local
voice directory, then point `ovos-tts-plugin-phoonnx` at it via the plugin's
`voice` setting in `mycroft.conf` (see [docs/ovos_plugin.md](./ovos_plugin.md)
for the full plugin config reference):

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

`TTSModelManager` loads `config.json` (engine="glowtts") + the mel `.onnx`
alongside it, and `GlowTTSAdapter` builds the vocoder from
`engine_params.vocoder_path`/`vocoder_type` exactly as documented in
[Vocoders](#vocoders) above.

## References

- [Larynx](https://github.com/rhasspy/larynx) · [GlowTTS paper](https://arxiv.org/abs/2005.11129)
- [docs/engines.md](./engines.md) — the engine adapter framework
- [docs/matcha.md](./matcha.md) — the other two-stage engine
- [docs/training.md](./training.md) — the shared VITS training pipeline this engine's dataset/preprocess step reuses
