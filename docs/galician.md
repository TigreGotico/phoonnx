# Galician phonemizer (pycotovia)

<!-- agent-hint
covers: how the Galician Cotovia phonemizer works and how the Galician StyleTTS2 voices use it
source: scriptconv_copyleft.phonemizers.gl:CotoviaPhonemizer (scriptconv[gl] extra), phoonnx/voice_index/styletts2.json, scripts/conversion/styletts2/export_proxectonos_gl.py, scripts/conversion/styletts2/gl_vocab.py
generated: no
verify: HF_HUB_OFFLINE=1 .venv/bin/python -m doctest -o NORMALIZE_WHITESPACE docs/galician.md
see: phonemizers.md, bsc_multispeaker.md, cloning.md
-->

This page describes the Galician Cotovia phonemizer and the two Galician StyleTTS2 voices that use it.
It is part of the [phonemizer catalog](phonemizers.md).

The `CotoviaPhonemizer` class in `scriptconv_copyleft.phonemizers.gl` makes Galician
phonemes with [pycotovia](https://github.com/TigreGotico/pycotovia), a pure-Python
port of the [Cotovia](http://webs.uvigo.es/gtm_voz) G2P engine.

## Installation

```bash
pip install --pre "phoonnx[gl]"
```

The `gl` extra installs `scriptconv[gl]`, which brings `pycotovia`.

## Phonemize a sentence

The phonemizer gives IPA or the native Cotovia notation. The `alphabet` argument selects
the output, and `Alphabet.COTOVIA` is the default.

```python
>>> from phoonnx.config import Alphabet
>>> from scriptconv_copyleft.phonemizers.gl import CotoviaPhonemizer
>>> text = "Este é un sistema de conversión de texto a voz en lingua galega."
>>> print(CotoviaPhonemizer(alphabet=Alphabet.IPA).phonemize_string(text, "gl"))
este ɛ uŋ sistema ðe kombeɾsjon de teksto a βoθ eŋ liŋɡwa ɣaleɣa
>>> print(CotoviaPhonemizer(alphabet=Alphabet.COTOVIA).phonemize_string(text, "gl"))
este E uN sistema De kombersjon de teksto a BoT eN liNgwa GaleGa

```

Voices trained on Cotovia notation get the same strings as from the original Cotovia C
binary, because pycotovia is tested for parity with it.

The phonemizer accepts a language tag that matches `gl-ES`, such as `gl` or `gl-ES`.
Any other tag raises `UnsupportedLanguageError`.

## Mark the stressed vowels

The `model` argument selects a variant. `model="stress"` gives the Cotovia notation with
a `^` after each stressed vowel. The voice manager passes this value from the
`phonemizer_model` field of the voice config.

```python
>>> from phoonnx.config import Alphabet
>>> from scriptconv_copyleft.phonemizers.gl import CotoviaPhonemizer
>>> text = "Este é un sistema de conversión de texto a voz en lingua galega."
>>> p_stress = CotoviaPhonemizer(alphabet=Alphabet.COTOVIA, model="stress")
>>> print(p_stress.phonemize_string(text, "gl"))
e^ste E^ u^N siste^ma De^ kombersjo^n de^ te^ksto a^ Bo^T e^N li^Ngwa Gale^Ga

```

The stress variant always gives Cotovia notation. It ignores `alphabet=Alphabet.IPA`.

## Build the phonemizer from a config

A `scriptconv` `TextFrontend` builds the phonemizer that its `FrontendConfig` names:

```python
>>> from phoonnx.config import Alphabet, PhonemeType
>>> from scriptconv.frontend import FrontendConfig, TextFrontend
>>> frontend = TextFrontend(FrontendConfig(lang="gl", phonemizer=PhonemeType.COTOVIA,
...                                        alphabet=Alphabet.COTOVIA, model="stress"))
>>> type(frontend.backend).__name__
'CotoviaPhonemizer'

```

`phoonnx.config.get_phonemizer` also builds it, but that function is deprecated. Its
warning names the release that removes it.

## Galician StyleTTS2 voices (Proxecto Nós Celtia and Brais)

Two Galician StyleTTS2 voices are in the voice index. Both speak `gl-ES` at 24 kHz, and
both read Cotovia notation:

```python
>>> from phoonnx.model_manager import TTSModelManager
>>> manager = TTSModelManager()
>>> manager.merge_default_voices()
>>> for voice_id in ("proxectonos/celtia-styletts2", "proxectonos/brais-styletts2"):
...     info = manager.get_voice(voice_id)
...     print(voice_id, info.engine.value, info.lang, info.phoneme_type.value, info.speaker_encoder_type)
proxectonos/celtia-styletts2 styletts2 gl-ES cotovia styletts2_style
proxectonos/brais-styletts2 styletts2 gl-ES cotovia styletts2_style

```

Celtia is a female speaker and Brais is a male speaker. The voices come from the
Apache-2.0 checkpoints of [Proxecto Nós](https://nos.gal/gl/proxecto-nos)
(`proxectonos/Nos_StyleTTS2-Celtia-GL` and `proxectonos/Nos_StyleTTS2-Brais-GL`).
`scripts/conversion/styletts2/export_proxectonos_gl.py` converts them to ONNX. The
files are on the Hub in
[`Phoonnx/styletts2-gl_es-celtia-cotovia`](https://huggingface.co/Phoonnx/styletts2-gl_es-celtia-cotovia)
and `Phoonnx/styletts2-gl_es-brais-cotovia`.

Each voice speaks with its own speaker style by default. It also accepts a reference
clip, because the export ships a `styletts2_style` speaker encoder. See
[voice cloning](cloning.md).

```python
from phoonnx.config import SynthesisConfig
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.load()
manager.merge_default_voices()
voice = manager.get_voice("proxectonos/celtia-styletts2").load()
for chunk in voice.synthesize("Bo día, como estás?", SynthesisConfig()):
    ...
```

### Front end

The checkpoints learned Cotovia notation with stress marks, not IPA. The voice config
therefore sets `phoneme_type: cotovia`, `alphabet: cotovia` and
`phonemizer_model: stress`. The vocabulary is the 69-symbol Galician phoneset of the
upstream `phoneme_token_maps.json`, not the 178-symbol espeak set of the other
StyleTTS2 voices. `scripts/conversion/styletts2/gl_vocab.py` builds that vocabulary,
and `tests/test_styletts2.py` checks it.

The tokenizer folds the multi-character Cotovia forms onto the single ids that the
model learned: `rr` becomes `R`, `tS` becomes `W`, and a stressed vowel `V^` becomes the
accented vowel (`o^` becomes `ó`).

### Padding token

The Galician checkpoints pad with `X` (id 0), the token that the upstream training code
adds to each sample. The upstream `inference.py` uses the word separator (id 1) instead.
Do not copy that. The word separator is a trained speech symbol, so a leading id 1 makes
the voice speak an extra syllable at the start of each utterance.

### Check intelligibility

Use a Galician ASR model to check these voices. The `onnx-asr` multilingual models
`nemo-canary-1b-v2` and `parakeet-tdt-v3` do not cover Galician. Two Galician ONNX
exports of Proxecto Nós models do:
[`OpenVoiceOS/Nos_ASR-wav2vec2-xls-r-300m-gl-onnx`](https://huggingface.co/OpenVoiceOS/Nos_ASR-wav2vec2-xls-r-300m-gl-onnx)
(wav2vec2 XLS-R 300M CTC) and
[`OpenVoiceOS/proxectonos-gl-conformer-ctc-large-onnx`](https://huggingface.co/OpenVoiceOS/proxectonos-gl-conformer-ctc-large-onnx)
(NeMo Conformer-CTC large). Both run on onnxruntime CPU.

Score the speaker's own recordings with the same ASR, and report that score next to the
score of the synthesized audio. The gap between the two says how good the voice is. The
absolute word error rate says more about the ASR.

### Known front-end gaps

pycotovia marks stress on unstressed function words, such as `o` and `de`, and it drops
punctuation:

```python
>>> from phoonnx.config import Alphabet
>>> from scriptconv_copyleft.phonemizers.gl import CotoviaPhonemizer
>>> p_stress = CotoviaPhonemizer(alphabet=Alphabet.COTOVIA, model="stress")
>>> print(p_stress.phonemize_string("Ola. Que tal?", "gl"))
o^la ke^ ta^l

```

The models accept punctuation, so the voices speak with flatter prosody than the
upstream front end gives. These gaps belong to pycotovia.

### Train a Galician StyleTTS2 voice

[`proxectonos/PL-ModernBERT-gl`](https://huggingface.co/proxectonos/PL-ModernBERT-gl)
is a Galician phoneme-level BERT for the `styletts2-plbert` training engine, which
supports a ModernBERT backbone. It uses the same 69-symbol Cotovia vocabulary. The two
shipped voices do not contain it, because their checkpoints hold an ALBERT-style PL-BERT.

---

[← Orpheus](orpheus.md) · [Home](index.md) · [BSC multispeaker StyleTTS2 →](bsc_multispeaker.md)
