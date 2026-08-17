# OptiSpeech Engine

This page is for integrators loading OptiSpeech voices in phoonnx. After reading
it you can load an OptiSpeech voice, understand its single-file ONNX contract,
and control duration / pitch / energy at synthesis time.

> Related: [adapter architecture](../../engines.md) ·
> [configuration](../../configuration.md) ·
> [training reference](../training.md)

## What it is

OptiSpeech is a non-autoregressive, **FastSpeech2-style** TTS: an acoustic model
with explicit duration / pitch / energy prediction and a GAN vocoder, exported
as a **single ONNX file** that outputs the waveform directly (no separate
vocoder stage). It embeds **all of its config inside the ONNX metadata** (no
external config file needed).

## When to pick it

Choose OptiSpeech when you want a single-file, end-to-end voice (no vocoder to
manage) with fine per-call control over speaking rate, pitch and energy — for
example the English LightSpeech / ConvNeXt-TTS voices below.

## Extras needed

Inference needs no engine-specific extra beyond a phonemizer. English voices
phonemize with espeak (`pip install phoonnx[espeak]`); Arabic-buckwalter voices
run in grapheme mode (see Gotchas).

## Inference contract

OptiSpeech uses a different ONNX I/O contract from VITS.

### ONNX inputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `x` | int64 | `[B, T]` | Phoneme IDs |
| `x_lengths` | int64 | `[B]` | Sequence lengths |
| `scales` | float32 | `[3]` | `[d_factor, p_factor, e_factor]` |
| `sids` | int64 | `[B]` | Speaker IDs (optional, multi-speaker) |
| `lids` | int64 | `[B]` | Language IDs (optional, multi-language) |

### ONNX outputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `wav` | float32 | `[B, 1, T]` | Waveform |
| `wav_lengths` | int64 | `[B]` | Sample counts |
| `durations` | int64 | `[B, T_text]` | Per-phoneme durations |

> The `scales` tensor differs from VITS, where it packs
> `[noise_scale, length_scale, noise_w_scale]`. OptiSpeech shares the
> `x` / `x_lengths` / `scales` input names with Matcha-TTS, so the adapter is
> probed **before** Matcha — it is identified by its embedded metadata and
> `wav` / `durations` outputs.

Parameters:

| Param | Default | Description |
|-------|---------|-------------|
| `d_factor` | 1.0 | Duration scale — speaking rate (>1 slower, <1 faster) |
| `p_factor` | 1.0 | Pitch scale |
| `e_factor` | 1.0 | Energy scale |

Pass them per call via `SynthesisConfig.extra_params`:

```python
voice.synthesize("hello", SynthesisConfig(extra_params={"d_factor": 0.9}))
```

## Obtaining

### Embedded metadata → native config

OptiSpeech stores its config inside the ONNX model under the `inference`
metadata key (a JSON blob): `sample_rate`, `input_symbols` (symbol→id),
`special_symbols`, `text_processor` (`add_blank` / `add_bos_eos` / `tokenizer`),
`speakers` and `languages`.
`phoonnx.engines.optispeech_config.voice_config_from_optispeech_meta` converts
that metadata into a native phoonnx `VoiceConfig` (folding `input_symbols` into
the tokenizer `phoneme_id_map` and honouring the `add_blank` / `add_bos_eos`
flags):

```python
import onnxruntime as ort
from phoonnx.engines.optispeech import OptiSpeechAdapter
from phoonnx.engines.optispeech_config import voice_config_from_optispeech_meta

sess = ort.InferenceSession("model.onnx")
meta = OptiSpeechAdapter().parse_onnx_meta(sess)
config = voice_config_from_optispeech_meta(meta)
```

The mirrored voices ship this as a **native `config.json`** alongside the model,
so they load through the standard path — `engine: optispeech` routes to
`OptiSpeechAdapter` with no runtime metadata extraction.

### Voice index

OptiSpeech voices ship in `phoonnx/voice_index/optispeech.json`, mirrored under
`OpenVoiceOS/phoonnx-optispeech` (each subfolder = `model.onnx` + native
`config.json`):

```python
from phoonnx.model_manager import TTSModelManager

m = TTSModelManager(); m.merge_default_voices()
voice = m.voices["hf_community/mush42/optispeech-lightspeech-en-us-emily"].load()
for chunk in voice.synthesize("Hello from OptiSpeech."):
    ...   # chunk.audio_float_array
```

Current entries (mirror of [`mush42/optispeech`](https://huggingface.co/mush42/optispeech), 24 kHz, en-US):

| voice_id | acoustic | speaker |
|----------|----------|---------|
| `…/optispeech-lightspeech-en-us-emily` | LightSpeech | emily |
| `…/optispeech-lightspeech-en-us-mike` | LightSpeech | mike |
| `…/optispeech-convnext-en-us-emily` | ConvNeXt-TTS | emily |

## Synthesis example

```python
voice.synthesize("Hello.", SynthesisConfig(extra_params={"d_factor": 1.0,
                                                          "p_factor": 1.0,
                                                          "e_factor": 1.0}))
```

## Training

OptiSpeech trains on the standard phoonnx preprocessed dataset. The vendored
model code lives under `phoonnx_train/optispeech/` and is constructed entirely
in plain Python — a quality preset fully determines the model, no external
config file is involved. Install the engine extra:

```bash
pip install phoonnx[train,train-optispeech]
```

```bash
python -m phoonnx_train.train \
    --engine optispeech \
    --dataset-dir ./dataset \
    --default-root-dir ./checkpoints \
    --quality medium
```

Quality presets change the constructed model (hidden width, encoder/decoder
depth and the WaveNeXt vocoder size):

| Preset | Acoustic `dim` | Encoder / decoder layers | Vocoder `dim` / layers |
|--------|----------------|--------------------------|------------------------|
| `x-low` | 128 | 2 / 2 | 256 / 4 |
| `medium` | 256 | 4 / 4 | 384 / 8 |
| `high` | 384 | 6 / 6 | 512 / 8 |

Use `--quality <preset>` or set `quality` in the `extra` config; any preset
field can be overridden individually through `extra` (e.g. `{"dim": 320}`).

Architecture: a ConvNeXt text encoder with scaled sinusoidal positional
embeddings, FastSpeech2-style duration / pitch / energy variance predictors, a
learned monotonic alignment (ForwardSum + Viterbi) with Gaussian upsampling, a
ConvNeXt decoder, and a **WaveNeXt** neural vocoder trained adversarially against
multi-period and multi-resolution discriminators. Losses combine alignment,
duration, pitch and energy terms with the GAN mel / multi-resolution-STFT /
feature-matching terms.

Training uses manual optimization (two optimizers — generator and
discriminator — each with a cosine-with-warmup schedule). Checkpoints carry the
optimizer state, scheduler state and epoch, so a run is fully resumable:

```bash
python -m phoonnx_train.train --engine optispeech --dataset-dir ./dataset \
    --default-root-dir ./checkpoints \
    --resume-from-checkpoint ./checkpoints/last.ckpt
```

To fine-tune from an existing checkpoint whose vocabulary, speaker count or
vocoder size differs, the engine warm-starts by mapping matching tensors by
name and shape and logs exactly which keys were loaded and which were skipped.

Export a trained checkpoint to the single-file ONNX described above:

```bash
python -m phoonnx_train.export_onnx \
    --engine optispeech \
    --checkpoint ./checkpoints/last.ckpt \
    --output-dir ./onnx
```

The exported `model.onnx` is self-contained: the waveform comes straight out
(no separate vocoder ONNX) and the full runtime config — sample rate, inference
defaults, symbol table and text-processor flags — is embedded under the
`inference` metadata key, which the adapter reads back automatically.

## Gotchas / aliases

- **Text processing depends on the tokenizer name.** The tokenizer flags come
  from the model's `text_processor` metadata. When `tokenizer == "arabic-buck"`
  the config builder forces `PhonemeType.GRAPHEMES`; **otherwise** it uses
  espeak (`PhonemeType.ESPEAK`, `alphabet: ipa`). So the English
  LightSpeech / ConvNeXt voices phonemize with espeak, while an
  arabic-buckwalter voice runs in grapheme mode — it does *not* phonemize with
  espeak.
- **Add-blank / BOS-EOS** flags also come from the metadata (these voices use
  `add_blank: false` / `add_bos_eos: false`), so phoonnx feeds the exact token
  sequence the model was trained on.

## References

- [OptiSpeech](https://github.com/mush42/optispeech)
