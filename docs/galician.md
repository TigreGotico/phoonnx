# Galician phonemizer (pycotovia)

This language note is for anyone working with Galician voices. After reading it you will know
how the Cotovia phonemizer emits IPA or native Cotovia notation and how to select the
stress-marked variant. It is part of the [phonemizer catalog](phonemizers.md).

The `CotoviaPhonemizer` class in `scriptconv.phonemizers.gl` produces Galician
phonemes using [pycotovia](https://github.com/TigreGotico/pycotovia), a pure-Python
port of the [Cotovia](http://webs.uvigo.es/gtm_voz) G2P engine.

## Output alphabets

| `Alphabet` value | Output format | Use case |
|---|---|---|
| `Alphabet.IPA` | IPA string via `pycotovia.cotovia_to_ipa` | Most phonemizer pipelines |
| `Alphabet.COTOVIA` | Raw Cotovia notation (e.g. `este e uN sistema`) | Voices trained on Cotovia alphabet output |

Voices trained on Cotovia-alphabet output (e.g. ProxectoNos, Sabela, Celtia)
receive strings identical to the original Cotovia C binary's: pycotovia is
binary-parity-tested against it (see `pycotovia/docs/parity.md`).

## Installation

```bash
pip install phoonnx[gl]
# or directly:
pip install pycotovia
```

## Usage

```python
from scriptconv.phonemizers.gl import CotoviaPhonemizer
from phoonnx.config import Alphabet

p_ipa = CotoviaPhonemizer(alphabet=Alphabet.IPA)
p_cot = CotoviaPhonemizer(alphabet=Alphabet.COTOVIA)

text = "Este é un sistema de conversión de texto a voz en lingua galega."

print(p_ipa.phonemize_string(text, "gl"))   # IPA
print(p_cot.phonemize_string(text, "gl"))   # Cotovia notation
```

## Accepted language codes

`CotoviaPhonemizer` accepts any language tag that matches `gl-ES` within a
`langcodes` distance of ≤ 10 (e.g. `"gl"`, `"gl-ES"`).

## Stress-marked variant (HiTZ voices)

The constructor takes a `model` argument that selects the notation. `model="stress"` selects
the stress-marked Cotovia notation used by the HiTZ Galician VITS voices; the default is
stressless. The voice manager passes this through from the voice's `phonemizer_model` field.

```python
p_stress = CotoviaPhonemizer(alphabet=Alphabet.COTOVIA, model="stress")
```

## Factory dispatch

```python
from phoonnx.config import PhonemeType, Alphabet, get_phonemizer
p = get_phonemizer(PhonemeType.COTOVIA, alphabet=Alphabet.IPA, model="stress")
```

## Galician StyleTTS2 voices (ProxectoNos Celtia and Brais)

Two Galician StyleTTS2 voices ship in the voice index:

| Voice id | Speaker | Engine | Sample rate |
|---|---|---|---|
| `proxectonos/celtia-styletts2` | Celtia (female) | `styletts2` | 24 kHz |
| `proxectonos/brais-styletts2` | Brais (male) | `styletts2` | 24 kHz |

They come from the Apache-2.0 checkpoints of
[Proxecto Nós](https://nos.gal/gl/proxecto-nos) (`proxectonos/Nos_StyleTTS2-Celtia-GL`
and `proxectonos/Nos_StyleTTS2-Brais-GL`), converted to ONNX by
`scripts/conversion/styletts2/export_proxectonos_gl.py` and mirrored to
[`OpenVoiceOS/phoonnx-styletts2`](https://huggingface.co/OpenVoiceOS/phoonnx-styletts2).

Each voice renders with its own speaker style out of the box. It also accepts a
reference clip, because the export ships a `styletts2_style` speaker encoder —
see [voice cloning](cloning.md).

```python
from phoonnx.config import SynthesisConfig
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.load()
voice = manager.voices["proxectonos/celtia-styletts2"].load()
for chunk in voice.synthesize("Bo día, como estás?", SynthesisConfig()):
    ...
```

### Front end

These models were trained on **Cotovía notation with stress marks**, not on IPA.
The voices therefore set `phoneme_type: cotovia`, `alphabet: cotovia` and
`phonemizer_model: stress`. Their vocabulary is the 69-symbol Galician phoneset
of the upstream `phoneme_token_maps.json`, not the 178-symbol espeak set of the
other StyleTTS2 voices. `scripts/conversion/styletts2/gl_vocab.py` builds that
vocabulary, and `tests/test_styletts2.py` checks it against the vendored table.

The tokenizer folds the multi-character Cotovía forms onto the single ids the
model was trained with: `rr` → `R`, `tS` → `W`, and a stressed vowel `V^` → the
accented vowel (`o^` → `ó`).

### Intelligibility gate

Galician ASR is the instrument for judging these voices, and the choice of model
dominates the result. `onnx-asr`'s multilingual models do not help: both
`nemo-canary-1b-v2` and `parakeet-tdt-v3` cover 25 European languages, Galician
not among them, and only `whisper-base` in that library handles `gl` at all.

Use the two Galician ONNX exports instead:

| Repo | Model | Notes |
|---|---|---|
| [`OpenVoiceOS/Nos_ASR-wav2vec2-xls-r-300m-gl-onnx`](https://huggingface.co/OpenVoiceOS/Nos_ASR-wav2vec2-xls-r-300m-gl-onnx) | wav2vec2 XLS-R 300M CTC | strongest; 5.9 % WER on Celtia's own recordings |
| [`OpenVoiceOS/proxectonos-gl-conformer-ctc-large-onnx`](https://huggingface.co/OpenVoiceOS/proxectonos-gl-conformer-ctc-large-onnx) | NeMo Conformer-CTC large | second opinion; much weaker (22-23 % on human speech) |

Both are Apache-2.0 exports of Proxecto Nós models and run on onnxruntime CPU.
Always report the **human-speech floor** — the same ASR on the speaker's own
recordings — next to the synthesized WER. The gap is the number that says
something about the voice; the absolute WER says more about the ASR.

Measured on 52 Galician sentences per voice — the two speakers' published test
splits — with each speaker's own recordings of those sentences as the floor
(16 clips for Brais, 20 for Celtia):

| Voice | wav2vec2 synth | wav2vec2 floor | gap | conformer synth | conformer floor | gap |
|---|---|---|---|---|---|---|
| Celtia | 0.170 | 0.059 | +0.111 | 0.284 | 0.233 | +0.051 |
| Brais | 0.122 | 0.057 | +0.065 | 0.275 | 0.218 | +0.057 |

Report a distribution, not only a mean. A single sentence says almost nothing:
on the pangram `O raposo veloz salta por riba do can preguiceiro preto da vella
ponte` both voices score 0.308, because the ASR merges `can preguiceiro` into
one word for either of them.

### Padding token

The Galician checkpoints pad with `X` (id 0), the token upstream's
`meldataset.py` inserts and appends on every training sample. Upstream's own
`inference.py` instead prepends the word separator (id 1). Do not copy that:
the word separator is a trained speech symbol, so a leading id 1 makes the
voice speak an extra syllable at the start of every utterance. It costs Brais
0.196 WER against 0.122, and Celtia 0.180 against 0.170.

### Known front-end gaps

`pycotovia` does not yet reproduce the Cotovía binary's transcription in two
places. Measured against the gold `phonetic_transcription` column of the
`proxectonos/Nos_Celtia-GL` test split (30 sentences):

| Comparison | Word error |
|---|---|
| Phone string, ignoring stress and vowel timbre | 1.9 % |
| Adding open/closed vowel timbre (`É`, `Ó`) | 14.2 % |
| Adding stress placement | 47.6 % |

The base transcription is correct. The remaining errors are that `pycotovia`
never emits the open vowels `É` and `Ó`, and that it marks stress on unstressed
function words that the binary leaves unmarked. Punctuation is also dropped,
although the models accept it. Fixing these belongs in `pycotovia`; until then
the voices are intelligible but their prosody is flatter than upstream's.

### Training a Galician StyleTTS2 voice

[`proxectonos/PL-ModernBERT-gl`](https://huggingface.co/proxectonos/PL-ModernBERT-gl)
is the Galician phoneme-level BERT to use with the `styletts2-plbert` training
engine, which supports ModernBERT backbones. It shares the same 69-symbol
Cotovía vocabulary. Note that the two shipped voices do **not** embed it — their
checkpoints carry an ALBERT-style PL-BERT — so it is a resource for new training
runs, not a drop-in replacement.
