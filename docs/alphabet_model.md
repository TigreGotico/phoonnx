# Alphabet model

phoonnx uses two distinct alphabet concepts that meet at the synthesis boundary.

## VoiceConfig.alphabet — model-expected alphabet

Set from the model's `config.json`; describes the script or phoneme representation
the ONNX model was trained to consume.  Callers must not override this field.

Examples: `ipa`, `hangul`, `unicode`, `cangjie`.

## SynthesisConfig.alphabet — user-input alphabet

The representation the *caller* provides to `TTSVoice.synthesize`.  `None` (default)
means "assume the model's own alphabet" — the text passes through unchanged.

Set this when the caller's text is in a different representation than the model
expects.  For example, a Korean voice trained on conjoining Jamo (`alphabet=hangul`)
but given raw Hangul syllables:

```python
from phoonnx.config import Alphabet, SynthesisConfig

audio = list(voice.synthesize(
    "안녕하세요",
    syn_config=SynthesisConfig(alphabet=Alphabet.UNICODE),
))
```

## Conversion bridge — `phoonnx.alphabet_convert.convert`

```python
convert(text, src: Alphabet, dst: Alphabet) -> str
```

Called inside `TTSVoice.synthesize` after diacritics and before `adapter.encode_text`:

```python
src = syn_config.alphabet or voice.config.alphabet   # caller's representation
text = convert(text, src=src, dst=voice.config.alphabet)
```

Returns `text` unchanged when `src == dst` or no converter is registered for the
`(src, dst)` pair (unregistered pairs log a DEBUG message and never raise).

## Registered converters

| Pair                     | Transform                                                     | Dependencies               |
|--------------------------|---------------------------------------------------------------|----------------------------|
| `(UNICODE, HANGUL)`      | Hangul syllables → conjoining Jamo (NFD)                      | none (pure Python)         |
| `(UNICODE, HIRA)`        | Kanji → hiragana via `pykakasi`                               | `pykakasi` (extra: `cjk`)  |
| `(UNICODE, CANGJIE)`     | Hanzi → Cangjie tokens via `spacy-pkuseg` + HF mapping        | `spacy-pkuseg` (extra: `cjk`) |

Install CJK optional deps:

```
pip install phoonnx[cjk]
```

## Relationship to phonemizers

Phonemizers (IPA/ARPA output) are a separate step that runs after `convert` in
`TTSVoice.phonemize`.  They already emit the model's alphabet, so `convert` is a
no-op for all IPA/ARPA voices (`src == dst`).  `convert` only fires for
script/grapheme voices where the model was trained on a normalised text form.
