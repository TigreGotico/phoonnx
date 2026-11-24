# OptiSpeech Engine

<!-- agent-hint
covers: how to load an OptiSpeech voice, control duration, pitch and energy, and train one
source: phoonnx/engines/optispeech.py, phoonnx/engines/optispeech_config.py, phoonnx_train/engines/optispeech.py, phoonnx/voice_index/optispeech.json
generated: no
verify: .venv/bin/python -c "import re,sys,json; from phoonnx.engines.optispeech import OptiSpeechAdapter as A; from phoonnx_train.engines.optispeech import _QUALITY_PRESETS as Q; p=open('docs/training/engines/optispeech.md').read().replace(chr(96),''); d={k:float(v) for k,v in re.findall(r'^\| ([a-z_]+) \| ([\d.]+) \| [A-Z]',p,re.M)}; q={r[0]:tuple(map(int,r[1:])) for r in re.findall(r'^\| ([\w-]+) \| (\d+) \| (\d+) / (\d+) \| (\d+) / (\d+) \|',p,re.M)}; ids=set(re.findall(r'^\| (hf_community/[\w/-]+) \|',p,re.M)); sys.exit(0 if d==A().default_params() and q=={k:(v['dim'],v['encoder_num_layers'],v['decoder_num_layers'],v['vocoder_dim'],v['vocoder_num_layers']) for k,v in Q.items()} and ids==set(json.load(open('phoonnx/voice_index/optispeech.json'))) else 1)"
see: ../training.md, ../export.md, ../../engines.md
-->

This page shows how to load an OptiSpeech voice, set its rate, pitch and energy, and train and export your own model.

## What it is

OptiSpeech is a non-autoregressive TTS in the FastSpeech2 family. It has an acoustic model with explicit duration, pitch and energy prediction, and a GAN vocoder. The export is one ONNX file that outputs the waveform, so there is no separate vocoder. The file also holds its whole config in the ONNX metadata.

Pick OptiSpeech for a one-file voice with per-call control of speaking rate, pitch and energy.

## Install

Inference needs only the phonemizer of the voice. The indexed English voices phonemize with espeak:

```bash
pip install --pre "phoonnx[espeak]"
```

## Load an indexed voice

The OptiSpeech voices are in `phoonnx/voice_index/optispeech.json`. They mirror [`mush42/optispeech`](https://huggingface.co/mush42/optispeech), and each voice has its own Hub repository under `Phoonnx/` with `model.onnx` and a native `config.json`.

```python
from phoonnx.config import SynthesisConfig
from phoonnx.model_manager import TTSModelManager

m = TTSModelManager()
m.merge_default_voices()
voice = m.voices["hf_community/mush42/optispeech-lightspeech-en-us-emily"].load()
for chunk in voice.synthesize("Hello from OptiSpeech.",
                              SynthesisConfig(extra_params={"d_factor": 0.9})):
    audio = chunk.audio_float_array
```

`d_factor`, `p_factor` and `e_factor` scale the duration, the pitch and the energy. Pass them in `SynthesisConfig.extra_params`.

## Read the embedded metadata

OptiSpeech stores its config in the ONNX model under the `inference` metadata key, as JSON: `sample_rate`, `input_symbols` (symbol to ID), `special_symbols`, `text_processor` (`add_blank`, `add_bos_eos`, `tokenizer`), `speakers` and `languages`. `voice_config_from_optispeech_meta` turns that metadata into a native `VoiceConfig`. It puts `input_symbols` into the tokenizer `phoneme_id_map` and applies the `add_blank` and `add_bos_eos` flags:

```python
import onnxruntime as ort
from phoonnx.engines.optispeech import OptiSpeechAdapter
from phoonnx.engines.optispeech_config import voice_config_from_optispeech_meta

sess = ort.InferenceSession("model.onnx")
meta = OptiSpeechAdapter().parse_onnx_meta(sess)
config = voice_config_from_optispeech_meta(meta)
```

The indexed voices ship this native `config.json` beside the model. `engine: optispeech` selects `OptiSpeechAdapter`, and the voice loads through the standard path.

The tokenizer name in `text_processor` selects the phonemizer. For `arabic-buck`, the config builder sets `phoneme_type: graphemes`, so the voice does not phonemize. For any other name, it sets espeak (`alphabet: ipa`). The `add_blank` and `add_bos_eos` flags also come from the metadata, so phoonnx feeds the token sequence that the model was trained on.

## Train a model

OptiSpeech trains on the standard phoonnx preprocessed dataset. The model code is in `phoonnx_train/optispeech/`, and a quality preset sets the whole model, with no other config file.

1. Install the engine extra:

   ```bash
   pip install --pre "phoonnx[train,train-optispeech]"
   ```

2. Train:

   ```bash
   python -m phoonnx_train.train \
       --engine optispeech \
       --dataset-dir ./dataset \
       --default-root-dir ./checkpoints \
       --quality medium
   ```

   Use `--quality` to pick a preset from the table below. A key in the `engine_params` of the dataset's `config.json` overrides the same preset key, for example `{"engine_params": {"dim": 320}}`.

3. To resume a run, give its checkpoint. PyTorch Lightning writes the checkpoints under `<default-root-dir>/lightning_logs/version_<N>/checkpoints/`. A checkpoint holds the optimizer state, the scheduler state and the epoch.

   ```bash
   python -m phoonnx_train.train --engine optispeech --dataset-dir ./dataset \
       --default-root-dir ./checkpoints \
       --resume-from-checkpoint ./checkpoints/lightning_logs/version_0/checkpoints/epoch=99-step=25000.ckpt
   ```

4. Export the checkpoint:

   ```bash
   python -m phoonnx_train.export_onnx \
       ./checkpoints/lightning_logs/version_0/checkpoints/epoch=99-step=25000.ckpt \
       --config ./dataset/config.json \
       --engine optispeech \
       --output-dir ./onnx
   ```

   The export writes `model.onnx`. It outputs the waveform, and it holds the runtime config (sample rate, inference defaults, symbol table and text-processor flags) under the `inference` metadata key. The adapter reads that key when it loads the file.

To fine-tune from a checkpoint with a different vocabulary, speaker count or vocoder size, the engine warm-starts. It loads the tensors that match by name and shape, and it logs which keys it loaded and which it skipped.

The model has a ConvNeXt text encoder with scaled sinusoidal positional embeddings, FastSpeech2 duration, pitch and energy predictors, and a learned monotonic alignment (ForwardSum and Viterbi) with Gaussian upsampling. A ConvNeXt decoder feeds a WaveNeXt vocoder, which trains against multi-period and multi-resolution discriminators. Training uses manual optimization with two optimizers, one for the generator and one for the discriminator, each with a cosine schedule and warmup.

## Notes

- OptiSpeech and Matcha-TTS share the input names `x`, `x_lengths` and `scales`. The registry probes OptiSpeech first, and the probe reads the embedded metadata and the `wav` and `durations` outputs.
- The `scales` input of OptiSpeech holds `[d_factor, p_factor, e_factor]`. The `scales` input of VITS holds `[noise_scale, length_scale, noise_w_scale]`.

## Reference

### ONNX inputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `x` | int64 | `[B, T]` | Phoneme IDs |
| `x_lengths` | int64 | `[B]` | Sequence lengths |
| `scales` | float32 | `[3]` | `[d_factor, p_factor, e_factor]` |
| `sids` | int64 | `[B]` | Speaker IDs, for a multi-speaker graph |
| `lids` | int64 | `[B]` | Language IDs, for a multi-language graph |

### ONNX outputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `wav` | float32 | `[B, 1, T]` | Waveform |
| `wav_lengths` | int64 | `[B]` | Sample counts |
| `durations` | int64 | `[B, T_text]` | Duration of each phoneme |

### Synthesis parameters

| Param | Default | Description |
|-------|---------|-------------|
| `d_factor` | 1.0 | Duration scale (above 1 is slower) |
| `p_factor` | 1.0 | Pitch scale |
| `e_factor` | 1.0 | Energy scale |

### Indexed voices

| voice_id | Acoustic model | Speaker |
|----------|----------------|---------|
| `hf_community/mush42/optispeech-lightspeech-en-us-emily` | LightSpeech | emily |
| `hf_community/mush42/optispeech-lightspeech-en-us-mike` | LightSpeech | mike |
| `hf_community/mush42/optispeech-convnext-en-us-emily` | ConvNeXt-TTS | emily |

### Quality presets

| Preset | Acoustic `dim` | Encoder / decoder layers | Vocoder `dim` / layers |
|--------|----------------|--------------------------|------------------------|
| `x-low` | 128 | 2 / 2 | 256 / 4 |
| `medium` | 256 | 4 / 4 | 384 / 8 |
| `high` | 384 | 6 / 6 | 512 / 8 |

## References

- [OptiSpeech](https://github.com/mush42/optispeech)

---

[Docs index](../../index.md) · Previous: [Mixer-TTS](mixertts.md) · Next: [FastPitch](fastpitch.md)
