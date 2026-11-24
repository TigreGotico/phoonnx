# FastPitch Engine

<!-- agent-hint
covers: how to load a FastPitch voice with its vocoder, convert a Coqui FastPitch model and train one
source: phoonnx/engines/fastpitch.py, phoonnx/engines/mixertts.py, phoonnx/engines/coqui_config.py, phoonnx_train/engines/fastpitch.py, phoonnx/voice_index/fastpitch.json, scripts/conversion/coqui_fastpitch_export/
generated: no
verify: .venv/bin/python -c "import re,sys,json; from phoonnx_train.engines.fastpitch import _QUALITY_PRESETS as Q; p=open('docs/training/engines/fastpitch.md').read().replace(chr(96),''); q={r[0]:tuple(map(int,r[1:])) for r in re.findall(r'^\| ([\w-]+) \| (\d+) \| (\d+) \| (\d+) / (\d+) \| (\d+) \|',p,re.M)}; v={k:(a,b) for k,a,b in re.findall(r'^\| ([\w-]+/[\w/-]+) \| (\w+) \| (\w+) \|$',p,re.M)}; idx=json.load(open('phoonnx/voice_index/fastpitch.json')); sys.exit(0 if q=={k:(x['hidden_channels'],x['hidden_channels_ffn'],x['encoder_num_layers'],x['decoder_num_layers'],x['num_heads']) for k,x in Q.items()} and v=={k:(x['phoneme_type'],x['vocoder_type']) for k,x in idx.items()} else 1)"
see: mixertts.md, ../training.md, ../../vocoders.md
-->

This page shows how to load a FastPitch voice, convert a Coqui FastPitch model to ONNX, and train your own model.

## What it is

[FastPitch](https://arxiv.org/abs/2006.06873) (NVIDIA) is a non-autoregressive acoustic model in the FastSpeech2 family. It makes an 80-channel mel spectrogram from symbol IDs, with control of pace and pitch. Its ONNX inputs and output are the same as those of [Mixer-TTS](mixertts.md), so `FastPitchAdapter` subclasses `MixerTTSAdapter` and uses its feed and parse code. The config `engine` field tells the two apart. FastPitch is a two-stage engine: a separate vocoder makes the waveform from the mel (see [Vocoders](../../vocoders.md)).

Pick FastPitch to run Coqui or tts_arabic FastPitch voices with ONNX Runtime only, or to train a voice with explicit pitch control. The Mixer-TTS vocoders also work with it.

## Install

Inference needs only the phonemizer of the voice: gruut for the English voices (`pip install --pre "phoonnx[en]"`) and `mantoq` for Arabic (`pip install --pre "phoonnx[ar]"`). Training needs `pip install --pre "phoonnx[train]"`. The default F0 extractor, `librosa.pyin`, comes with `train`. The `train-fastpitch` extra exists but adds no package. For the faster WORLD extractors, preprocess with `--f0-method dio` or `--f0-method harvest` and install `phoonnx[train,train-pyworld]`.

## Load an indexed voice

The FastPitch voices are in `phoonnx/voice_index/fastpitch.json`. Each voice has its own Hub repository under `Phoonnx/`, and its index entry names its vocoder. `nipponjo/tts-arabic-fastpitch` is the multi-speaker Arabic FastPitch from [nipponjo/tts_arabic](https://github.com/nipponjo/tts_arabic), with `mantoq` Buckwalter tokens and the matched tts_arabic HiFi-GAN.

```python
from phoonnx.model_manager import TTSModelManager

m = TTSModelManager()
m.merge_default_voices()
voice = m.voices["nipponjo/tts-arabic-fastpitch"].load()
for chunk in voice.synthesize("السَّلامُ عَلَيكُم."):
    audio = chunk.audio_float_array
```

The synthesis parameters (`pace`, `pitch_mul`, `pitch_add`, `emotion`) and `SynthesisConfig.speaker_id` work as on the [Mixer-TTS](mixertts.md) page:

```python
from phoonnx.config import SynthesisConfig

voice.synthesize("السَّلامُ عَلَيكُم.", SynthesisConfig(extra_params={"pace": 1.1}))
```

## Convert a Coqui FastPitch model

`scripts/conversion/coqui_fastpitch_export/export_fp.py` exports a Coqui `ForwardTTS` checkpoint to an ONNX graph with dynamic length:

```bash
python scripts/conversion/coqui_fastpitch_export/export_fp.py config.json model.pth model.onnx
```

The exporter uses a vendored copy of `ForwardTTS`, and the pretrained weights load into it unchanged. The copy makes two changes so that the graph keeps a dynamic length:

1. The `FFTransformer` attention uses a tracer-safe replacement for `torch.nn.MultiheadAttention` (`tsmha.py`), with the same weight names. The original module puts the sequence length into the graph as a constant.
2. Inference uses a mask of ones with the live input shape. The original code makes `x_lengths` from the shape of the example input, and the tracer stores it as a constant that masks every token after the example length.

The export uses `torch.onnx.export(..., dynamo=False, opset_version=14)`. Then `add_pace.py` adds a `pace` input: the tracer stores the duration scale as a constant, and the script divides that constant by the new input.

The tokens must follow Coqui's vocabulary. The `phonemizer` field of the Coqui config selects gruut or espeak, which give different IPA, and `is_sorted` sorts the symbol set before IDs are assigned. `voice_config_from_coqui` in `phoonnx.engines.coqui_config` handles both.

## Train a model

`--engine fastpitch` trains a FastPitch model with `ForwardTTSTrainingEngine`. `--engine speedyspeech` uses its subclass `SpeedySpeechTrainingEngine`, which defaults to the SpeedySpeech variant: a residual conv-BN encoder and decoder, with no pitch predictor. Training uses the standard phoonnx pipeline. FastPitch reads the F0 sidecars that preprocess writes with `--engine fastpitch`. The FastPitch section of the [training reference](../training.md) lists the steps.

```bash
python -m phoonnx_train.train --dataset-dir ./dataset --engine fastpitch --quality medium
```

The exported model is the mel model only. To run it, add a vocoder to the `engine_params` of its config (`engine: fastpitch`, with `vocoder_path` and `vocoder_type`), as for [Mixer-TTS](mixertts.md).

## Notes

- The config `engine` field must be `fastpitch` or `fast_pitch`. The adapter never auto-detects a FastPitch graph, because the graph has the same inputs and output as a Mixer-TTS graph.

## Reference

### Indexed voices

| voice_id | Phonemizer | Vocoder |
|----------|------------|---------|
| `nipponjo/tts-arabic-fastpitch` | `mantoq` | `hifigan` |
| `coqui/en-ljspeech-fast_pitch` | `gruut` | `hifigan` |
| `coqui/en-ljspeech-speedy-speech` | `gruut` | `vocos` |
| `coqui/en-vctk-fast_pitch` | `gruut` | `vocos` |

BSC publishes the 22 kHz Vocos graph of the `vocos` voices twice, with the same bytes: `BSC-LT/vocos-mel-22khz` under Apache-2.0, and `projecte-aina/alvocat-vocos-22khz` under CC-BY-NC-4.0, for non-commercial use. The two releases differ only in their license. The `vocoder_url` of each index entry names the release that the voice loads.

### Quality presets

| Preset | hidden_channels | hidden_channels_ffn | encoder / decoder layers | num_heads |
|--------|-----------------|---------------------|--------------------------|-----------|
| `x-low` | 128 | 512 | 4 / 4 | 1 |
| `medium` | 384 | 1024 | 6 / 6 | 1 |
| `high` | 512 | 1536 | 6 / 6 | 2 |

## References

- [FastPitch paper](https://arxiv.org/abs/2006.06873)
- [tts_arabic](https://github.com/nipponjo/tts_arabic)

---

[Docs index](../../index.md) · Previous: [OptiSpeech](optispeech.md) · Next: [x-vits](xvits.md)
