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

## Text processing

GlowTTS/Larynx phonemizes with **gruut** (``phoneme_type: gruut``,
``alphabet: ipa``) and interleaves a blank (PAD, id 0) between symbols
(``add_blank``), with no BOS/EOS. The 46-symbol table comes from the voice's
``phonemes.txt``, folded into the native config's ``phoneme_id_map``.

> Requires the ``gruut`` package for phonemization.

## References

- [Larynx](https://github.com/rhasspy/larynx) · [GlowTTS paper](https://arxiv.org/abs/2005.11129)
- [docs/engines.md](./engines.md) — the engine adapter framework
- [docs/matcha.md](./matcha.md) — the other two-stage engine
