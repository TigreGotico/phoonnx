# FastPitch Engine

<!-- agent-hint
covers: how to load a FastPitch voice with its vocoder, convert a Coqui FastPitch model and train one
source: phoonnx/engines/fastpitch.py, phoonnx_train/engines/fastpitch.py, phoonnx/voice_index/fastpitch.json, scripts/conversion/coqui_fastpitch_export/
generated: no
verify: pytest tests/test_fastpitch.py tests/test_fastpitch_training.py; the preset table: none, checked by reading phoonnx_train/engines/fastpitch.py:_QUALITY_PRESETS
see: mixertts.md, ../training.md, ../../vocoders.md
-->

This page explains how to load, convert and train FastPitch voices, for voice builders and integrators.

> Related: [training reference](../training.md) ·
> [adapter architecture](../../engines.md) ·
> [configuration](../../configuration.md) · [vocoders](../../vocoders.md) ·
> [Mixer-TTS — the shared inference contract](mixertts.md)

## What it is

[FastPitch](https://arxiv.org/abs/2006.06873) (NVIDIA) is a non-autoregressive,
FastSpeech2-style acoustic model: text → 80-channel mel with pace/pitch control.
Its exported ONNX inference contract is **identical to
[Mixer-TTS](mixertts.md)** (`token_ids` + `pace` / `speaker` / `pitch_mul` /
`pitch_add` → `mel_spec`), so `FastPitchAdapter` subclasses `MixerTTSAdapter` and
reuses its feed/parse logic; the two are told apart by the native config `engine`
field. It is **two-stage** — the mel is voiced by a separate vocoder
([vocoders.md](../../vocoders.md)).

## When to pick it

Choose FastPitch to run existing Coqui or tts_arabic FastPitch voices in a
pure-ONNX pipeline, or to train a FastSpeech2-style voice with explicit pitch
control. Its shared Mixer-TTS contract means the same vocoders apply.

## Extras needed

Inference needs a phonemizer (gruut/espeak for English, `mantoq` for Arabic) and
no engine-specific extra. Training needs `pip install --pre "phoonnx[train]"`. The default F0
extractor, `librosa.pyin`, comes with `[train]`. The `train-fastpitch` extra exists but adds
no package. For the faster WORLD extractors, preprocess with `--f0-method dio` or
`--f0-method harvest` and install `phoonnx[train,train-pyworld]`.

## Obtaining / training

### Loading indexed voices

FastPitch voices ship in `phoonnx/voice_index/fastpitch.json`, mirrored under
`Phoonnx/phoonnx-fastpitch`. The reference model is the multi-speaker Arabic
FastPitch from [nipponjo/tts_arabic](https://github.com/nipponjo/tts_arabic)
(`mantoq` / buckwalter tokenization, paired with the matched
`tts-arabic-hifigan`).

```python
from phoonnx.model_manager import TTSModelManager
m = TTSModelManager(); m.merge_default_voices()
voice = m.voices["nipponjo/tts-arabic-fastpitch"].load()
for chunk in voice.synthesize("السَّلامُ عَلَيكُم."): ...
```

### Converting Coqui FastPitch to ONNX

Coqui's `ForwardTTS` needs three fixes to export a faithful, **truly
dynamic-length** ONNX (done in a vendored copy; pretrained weights load
unchanged):

1. **Attention** — its `FFTransformer` wraps `torch.nn.MultiheadAttention`,
   which bakes the sequence length under the legacy tracer. Replace it with a
   tracer-safe drop-in (same `in_proj_weight` / `in_proj_bias` / `out_proj`
   names, `-1` + live-shape reshapes). Numerically identical (~2e-7).
2. **Dynamic mask** — `inference()` does `x_lengths = torch.tensor(x.shape[1:2])`,
   which freezes the input length as a constant, masking out every token past
   the export example. Replace with a live-shape all-ones mask
   (`torch.ones(B, 1, x.shape[1])`) — inference is a single un-padded sequence.
   This is what makes the mel length unbounded (L=600 → 3637 frames, linear).
3. **Tokenization** — build the vocab coqui's way: `phonemizer` field (gruut vs
   espeak emit different IPA) and `is_sorted=True` (sort the symbol set before id
   assignment). Both handled by `voice_config_from_coqui`.

Then `torch.onnx.export(..., dynamo=False, opset=14)` gives a dynamic graph.
en/ljspeech is mirrored as `coqui/en-ljspeech-fast_pitch` (gruut, matched
`coqui-ljspeech-hifigan-v2`). The index also holds `coqui/en-vctk-fast_pitch` and
`coqui/en-ljspeech-speedy-speech`. The exporter is in `scripts/conversion/coqui_fastpitch_export/`.

### Training

Trainable with `--engine fastpitch` (the `ForwardTTSTrainingEngine`). The same
`ForwardTTS` engine is also registered as `--engine speedyspeech` (a subclass
defaulting to the SpeedySpeech variant — residual conv-BN encoder/decoder, no
pitch predictor). Training shares the standard phoonnx pipeline. FastPitch reads the F0
sidecars that preprocess writes with `--engine fastpitch` (see
[FastPitch / SpeedySpeech](../training.md#fastpitch--speedyspeech---engine-fastpitch----engine-speedyspeech)).

## Synthesis example

FastPitch is two-stage; point a local voice `config.json` at the exported mel
model and a vocoder (`engine: fastpitch`, `engine_params.vocoder_path` /
`vocoder_type`), then synthesize:

```python
voice.synthesize("Hello.", SynthesisConfig(extra_params={"pace": 1.0}))
```

The mel and per-call params (`pace` / `speaker` / `pitch_mul` / `pitch_add`) are
exactly as documented for [Mixer-TTS](mixertts.md).

## Gotchas / aliases

- **Detect aliases:** the config `engine` field may be `fastpitch` or
  `fast_pitch`.
- **Two-stage:** the exported model is the mel only — you must supply a separate
  vocoder ONNX (see [vocoders.md](../../vocoders.md)).
- **Training engine names:** both `fastpitch` and `speedyspeech` route to the
  same `ForwardTTSTrainingEngine`.

## References

- [FastPitch paper](https://arxiv.org/abs/2006.06873) · [tts_arabic](https://github.com/nipponjo/tts_arabic)

---

[Docs index](../../index.md) · Previous: [OptiSpeech](optispeech.md) · Next: [x-vits](xvits.md)
