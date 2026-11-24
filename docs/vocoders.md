# Vocoders

<!-- agent-hint
covers: which vocoder types exist, how a voice selects one, how a vocoder config changes the mel, and how to add a vocoder type
source: phoonnx/engines/vocoders/__init__.py, phoonnx/engines/vocoders/base.py:BaseVocoder, phoonnx/engines/vocoders/raw.py, phoonnx/engines/vocoders/vocos.py, phoonnx/engines/vocoders/griffinlim.py, phoonnx/model_manager.py:TTSModelInfo
generated: no
verify: .venv/bin/python -m doctest -o ELLIPSIS -o NORMALIZE_WHITESPACE docs/vocoders.md
see: engines.md, training/engines/glowtts.md, training/engines/matcha.md
-->

This page describes the vocoder registry in `phoonnx/engines/vocoders/`. Read it when you pair an acoustic model that makes a mel spectrogram with a vocoder, or when you add a vocoder type.

Some acoustic models, such as GlowTTS, Matcha-TTS and some OptiSpeech voices, make a mel spectrogram. A separate vocoder turns that mel into a waveform. The registry lets each of these adapters use any vocoder through one method, `mel_to_audio`.

```
text ──[acoustic model]──▶ mel ──[vocoder]──▶ waveform
```

## Use a vocoder directly

`list_vocoders` gives the registered vocoder types:

```python
>>> from phoonnx.engines.vocoders import list_vocoders
>>> list_vocoders()
['griffinlim', 'wavenext', 'hifigan', 'melgan', 'vocos', 'raw']

```

`build_vocoder` makes a vocoder. A neural vocoder needs its ONNX file, for example `build_vocoder(model_path="hifigan.onnx", vocoder_type="hifigan")`. Then `vocoder.mel_to_audio(mel)` takes a mel of shape `[B, n_mels, T]` and gives a 1-D float32 waveform.

Griffin-Lim needs no model file. Its parameters come from the config:

```python
>>> from phoonnx.engines.vocoders import build_vocoder
>>> gl = build_vocoder(vocoder_type="griffinlim",
...                    config={"sample_rate": 22050, "n_fft": 1024, "hop_length": 256, "num_mels": 80})
>>> type(gl).__name__, gl.sample_rate, gl.num_iters
('GriffinLimVocoder', 22050, 60)

```

Griffin-Lim imports `librosa` when it runs. The `train` and `all` extras install it.

When `vocoder_type` names a registered type, `build_vocoder` uses it. With no type, or an unknown one, it opens the ONNX file and asks each vocoder type in turn if the file matches. When no type matches, it logs a warning and uses Vocos.

## How a voice selects its vocoder

A voice that needs a vocoder names it in its voice index entry. This is the Turkish GlowTTS entry, with the URLs shortened:

```json
{
  "voice_id": "coqui-tr-common-voice-glow-tts",
  "engine": "glowtts",
  "model_url": ".../glowtts-tr_tr-common-graphemes/.../model.onnx",
  "config_url": ".../glowtts-tr_tr-common-graphemes/.../config.json",
  "vocoder_url": ".../hifigan-tr_tr-common/.../model.onnx",
  "vocoder_config_url": ".../hifigan-tr_tr-common/.../vocoder.json",
  "vocoder_type": "hifigan"
}
```

The model manager downloads the vocoder with the model, reads the vocoder config, and gives both to the adapter in its engine parameters. Your code does not change: `voice.synthesize(...)` gives audio.

- `vocoder_url` is the vocoder ONNX file. Griffin-Lim has no file, so leave it out.
- `vocoder_type` selects the vocoder type. When it is absent, the ONNX outputs select it: three or more outputs select Vocos, and one output selects a vocoder that gives the waveform directly.
- `vocoder_config_url` is a small `vocoder.json` with the vocoder parameters: the sample rate, the mel settings and the mel preprocessing flags.

## Mel preprocessing flags

Vocoders expect the mel in different conventions. A vocoder declares its convention with flags in its `vocoder.json`. `BaseVocoder._preprocess_mel` applies the flags before the ONNX session runs. A vocoder with no flags gets the mel as the acoustic model made it.

The `stats_norm` flag scales each mel channel with `mel_mean` and `mel_std`, as `(mel - mean) / std`. Coqui vocoders that were trained on normalized mels, such as multiband MelGAN, need it:

```python
>>> import numpy as np
>>> from phoonnx.engines.vocoders.raw import HiFiGANVocoder
>>> vocoder = HiFiGANVocoder(config={"stats_norm": True, "mel_mean": [1.0, 2.0], "mel_std": [2.0, 4.0]})
>>> vocoder._preprocess_mel(np.array([[[3.0], [6.0]]])).tolist()
[[[1.0], [1.0]]]

```

The `vocoder.json` of such a vocoder looks like this:

```json
{ "vocoder_type": "melgan", "sample_rate": 22050, "stats_norm": true,
  "mel_mean": [ ... 80 floats ... ], "mel_std": [ ... 80 floats ... ] }
```

To add a step, add a flag to the config and a branch to `_preprocess_mel`. The flag ships with the vocoder data.

## Replace a vocoder

The vocoder is a link in the index entry, so you can change it and keep the acoustic model.

- To use another neural vocoder, change `vocoder_url` and `vocoder_type`. The vocoder must use the same mel settings as the acoustic model: sample rate, FFT size, hop length, number of mels, and minimum and maximum frequency. A different mel config gives noise. If only the normalization differs, add the `stats_norm` flag.
- To use Griffin-Lim, remove `vocoder_url`, set `vocoder_type` to `"griffinlim"`, and point `vocoder_config_url` at a `vocoder.json` with the mel settings of the acoustic model. The audio sounds robotic.
- To select a vocoder in code, give the adapter a vocoder instance: `GlowTTSAdapter(vocoder=build_vocoder(...))`.

## Add a vocoder type

Make a subclass of `BaseVocoder`, write its `mel_to_audio` method, and register it:

```python
>>> from phoonnx.engines.vocoders import list_vocoders, register_vocoder
>>> from phoonnx.engines.vocoders.base import BaseVocoder
>>> class MyVocoder(BaseVocoder):
...     name = "myvoc"
...     def mel_to_audio(self, mel, denoise=False):
...         mel = self._preprocess_mel(mel)
...         return self.session.run(None, {self.input_name: mel})[0].squeeze()
>>> register_vocoder("myvoc", MyVocoder, detect_priority=35)
>>> list_vocoders()[-1]
'myvoc'

```

`detect_priority` sets the order of auto-detection: a lower value is asked first. The default is 100. The `detect` method of `BaseVocoder` returns `False`, so a type that does not override it is used only when a voice names it.

## Vocoder types

| `vocoder_type` | Class | ONNX outputs | Detection priority | Notes |
|---|---|---|---|---|
| `wavenext` | `WavenextVocoder` | waveform (1) | 30 | Vocos with a trained layer in place of the inverse STFT |
| `hifigan` | `HiFiGANVocoder` | waveform (1) | 31 | GAN vocoder |
| `melgan` | `HiFiGANVocoder` | waveform (1) | 32 | MelGAN and multiband MelGAN, with the same mel to audio contract |
| `vocos` | `VocosVocoder` | magnitude, real, imaginary (3) | 40 | inverse STFT, honours `denoise` |
| `raw` | `RawWaveformVocoder` | waveform (1) | 50 | any vocoder with one waveform output |
| `griffinlim` | `GriffinLimVocoder` | no model | 99 | parametric, never auto-detected, needs `librosa` |

The neural vocoders of the bundled index live in their own Hugging Face repositories, such as `Phoonnx/hifigan-tr_tr-common` and `Phoonnx/vocos-multi-alvocat-22khz`. The Coqui HiFi-GAN exports use the generator code in `scripts/conversion/coqui_fastpitch_export/hifigan_generator.py`, which needs no `coqui-tts` package. A vocoder works only with an acoustic model whose mel settings and normalization match it.

---

[Docs index](index.md) · Previous: [Engine architecture](engines.md) · Next: [Python API](api.md)
