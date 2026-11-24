# Mixer-TTS Engine

<!-- agent-hint
covers: how to load an English or Arabic Mixer-TTS voice, pair it with a vocoder, and train one
source: phoonnx/engines/mixertts.py, phoonnx/engines/mixertts_config.py, phoonnx_train/engines/mixer.py, phoonnx/voice_index/mixertts.json
generated: no
verify: .venv/bin/python -c "import re,sys,json; from phoonnx.engines.mixertts import MixerTTSAdapter as A; from phoonnx_train.engines.mixer import _QUALITY_PRESETS as Q; p=open('docs/training/engines/mixertts.md').read().replace(chr(96),''); d={k:float(v) for k,v in re.findall(r'^\| ([a-z_]+) \| ([\d.]+) \| [A-Z]',p,re.M)}; q={k:int(v) for k,v in re.findall(r'^\| ([\w-]+) \| (\d+) \|$',p,re.M)}; idx=json.load(open('phoonnx/voice_index/mixertts.json')); v=dict(re.findall(r'^\| (nipponjo/[\w-]+) \| [^|]+ \| (\w+) \|',p,re.M)); sys.exit(0 if d=={k:float(x) for k,x in A().default_params().items()} and q=={k:x['symbols_embedding_dim'] for k,x in Q.items()} and v=={k:x['vocoder_type'] for k,x in idx.items()} else 1)"
see: fastpitch.md, ../training.md, ../../vocoders.md
-->

This page shows how to load an English or Arabic Mixer-TTS voice, pair it with a vocoder, and train and export your own model.

## What it is

[Mixer-TTS](https://arxiv.org/abs/2110.03584) (NVIDIA) is a non-autoregressive acoustic model in the FastPitch family, built from MLP-Mixer blocks. It makes an 80-channel mel spectrogram from symbol IDs. Like Matcha-TTS and GlowTTS, it is a two-stage engine: a separate vocoder from [`phoonnx.engines.vocoders`](../../vocoders.md) turns the mel into a waveform.

Pick Mixer-TTS for a small English or Arabic acoustic model with control of pace, pitch and emotion. It has the same inference inputs as [FastPitch](fastpitch.md).

## Install

Inference needs only the phonemizer of the voice: espeak for English (`pip install --pre "phoonnx[espeak]"`) and `mantoq` for Arabic (`pip install --pre "phoonnx[ar]"`). Training needs `pip install --pre "phoonnx[train]"`. The default F0 extractor, `librosa.pyin`, comes with `train`. The `train-mixer` extra exists but adds no package. The WORLD extractors (`--f0-method dio` or `harvest` at preprocess) need `phoonnx[train,train-pyworld]`.

## Load an indexed voice

The Mixer-TTS voices are in `phoonnx/voice_index/mixertts.json`. Each voice has its own Hub repository under `Phoonnx/`. The English voices are the [nipponjo/mixer-tts-pytorch](https://github.com/nipponjo/mixer-tts-pytorch) LJSpeech checkpoints at 22.05 kHz.

```python
from phoonnx.config import SynthesisConfig
from phoonnx.model_manager import TTSModelManager

m = TTSModelManager()
m.merge_default_voices()
voice = m.voices["nipponjo/mixer-tts-ljspeech-384"].load()
for chunk in voice.synthesize("Hello from Mixer-TTS.",
                              SynthesisConfig(extra={"pace": 1.1})):
    audio = chunk.audio_float_array
```

The English voices use a fixed symbol table (pad, punctuation, letters and IPA) and espeak IPA input. `phoonnx.engines.mixertts_config.voice_config_from_mixer` builds a native `VoiceConfig` from that ordered list. The indexed voices ship it as a native `config.json` (`engine: mixertts`).

### Arabic voices

The [tts_arabic](https://github.com/nipponjo/tts_arabic) Mixer-TTS models (`nipponjo/tts-arabic-mixer80` and `nipponjo/tts-arabic-mixer128`) are multi-speaker Arabic voices with a Buckwalter phoneme table. They tokenize with the `mantoq` phonemizer (`phoneme_type: mantoq`, `alphabet: buckwalter`, `_+_` word separator). The input can be vocalized Arabic or Buckwalter transliteration. Select a speaker with `SynthesisConfig(speaker=...)`.

## Train a model

Training uses `--engine mixer` (or `mixertts`). The model is a port of the NeMo Mixer-TTS in `phoonnx_train/mixertts/models`, with Apache-2.0 headers. It adds the speaker, emotion and energy conditioning and the optional LSGAN mel-patch refinement of nipponjo/tts-arabic-pytorch. The Lightning module in `phoonnx_train/mixertts/lightning.py` follows the NeMo recipe: mel, duration, pitch and energy MSE losses, a ForwardSum (CTC) aligner loss, a delayed binarization loss, and AdamW with Noam warmup. Set `"train_gan": true` in the `engine_params` of the dataset's `config.json` to add the LSGAN `PatchDiscriminator` refinement, which NeMo does not have.

Mixer-TTS uses the standard phoonnx pipeline and the F0 sidecars of the FastPitch engine. Preprocess extracts F0 at the mel hop with the `--f0-method` extractor (default `pyin`), from the same trimmed and normalized audio as the mels.

1. Train, with a preset from the table below:

   ```bash
   python -m phoonnx_train.train --dataset-dir /path/to/prep \
       --engine mixer --quality medium
   ```

2. Export to ONNX:

   ```bash
   python -m phoonnx_train.export_onnx \
       lightning_logs/version_0/checkpoints/epoch=999-step=250000.ckpt \
       -c /path/to/prep/config.json --engine mixer
   ```

The export gives the inference inputs and output of the contract below, and it records `mel_fmin` and `mel_fmax` in the ONNX metadata.

## Vocoders

Each indexed voice names its vocoder. BSC publishes the 22 kHz Vocos graph of the English voices twice, with the same bytes: `BSC-LT/vocos-mel-22khz` under Apache-2.0, and `projecte-aina/alvocat-vocos-22khz` under CC-BY-NC-4.0, for non-commercial use. The two releases differ only in their license. The `vocoder_url` of each index entry names the release that the voice loads. The Arabic voices use the `hifigan` vocoder of `Phoonnx/hifigan-ar-nipponjo`. Griffin-Lim also works, with `spec_gain = ln(10)` to invert the natural-log mel, and it needs no model file. See [Vocoders](../../vocoders.md).

## Notes

- The config `engine` field may be `mixertts` or `mixer_tts`. A graph with the inputs `token_ids`, `pace` and `pitch_mul`, or with a `mel_spec` output and a `token_ids` input, also selects this adapter.
- `SynthesisConfig.length_scale` sets `pace` to `1 / length_scale`.

## Reference

### ONNX inputs

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `token_ids` | int64 | `[B, T]` | Symbol IDs |
| `pace` | float32 | `[1]` | Speaking rate (above 1 is faster) |
| `speaker` | int32 | `[1]` | Speaker ID, from `SynthesisConfig.speaker` (0 by default) |
| `emotion` | int32 | `[1]` | Emotion ID |
| `pitch_mul` | float32 | `[1]` | Pitch multiplier |
| `pitch_add` | float32 | `[1]` | Pitch offset |

The output is a mel spectrogram `mel_spec [B, 80, T]`.

### Synthesis parameters

Set these in `SynthesisConfig.extra`.

| Param | Default | Description |
|-------|---------|-------------|
| `pace` | 1.0 | Speaking rate |
| `pitch_mul` | 1.0 | Pitch scale |
| `pitch_add` | 0.0 | Pitch offset |
| `emotion` | 0 | Emotion ID |

### Indexed voices

| voice_id | Language | Vocoder |
|----------|----------|---------|
| `nipponjo/mixer-tts-ljspeech-80` | English | `vocos` |
| `nipponjo/mixer-tts-ljspeech-128` | English | `vocos` |
| `nipponjo/mixer-tts-ljspeech-384` | English | `vocos` |
| `nipponjo/tts-arabic-mixer80` | Arabic | `hifigan` |
| `nipponjo/tts-arabic-mixer128` | Arabic | `hifigan` |

### Quality presets

The preset sets `symbols_embedding_dim`.

| Preset | symbols_embedding_dim |
|--------|-----------------------|
| `x-low` | 80 |
| `medium` | 128 |
| `high` | 384 |

## References

- [Mixer-TTS paper](https://arxiv.org/abs/2110.03584)
- [nipponjo/mixer-tts-pytorch](https://github.com/nipponjo/mixer-tts-pytorch)

---

[Docs index](../../index.md) · Previous: [GlowTTS](glowtts.md) · Next: [OptiSpeech](optispeech.md)
