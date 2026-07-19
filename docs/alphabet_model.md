# Alphabet conversion model

phoonnx uses a single, unified mechanism to transform every piece of text into
the token sequence a model expects: **alphabet conversion**.

## The two alphabets

Every voice has two alphabet fields:

| concept | field | answers | example |
|---|---|---|---|
| model input | `VoiceConfig.alphabet` | WHAT the model eats | `Alphabet.IPA` |
| conversion backend + tokenisation | `VoiceConfig.phoneme_type` | HOW to get there | `PhonemeType.ESPEAK` |
| caller's text | `SynthesisConfig.alphabet` | WHAT the user's text is | `None` (= graphemes) |

`SynthesisConfig.alphabet` defaults to `None`, which is treated as
`Alphabet.GRAPHEMES` — ordinary transcription text.  Set it only when passing
pre-converted content (e.g. an IPA string or Hangul text).

## `convert(text, lang, src, tgt)`

```python
from phoonnx.alphabet_convert import convert
from phoonnx.config import Alphabet, PhonemeType

# graphemes → IPA  (phonemization via espeak)
chunks = convert("hello world", "en", Alphabet.GRAPHEMES, Alphabet.IPA,
                 phoneme_type=PhonemeType.ESPEAK)

# graphemes → Hiragana  (script conversion via pykakasi)
hira = convert("東京", "ja", Alphabet.GRAPHEMES, Alphabet.HIRA)

# identity (src == tgt, no-op)
same = convert("already ipa", "en", Alphabet.IPA, Alphabet.IPA)
```

### Return type

- **Graphemes → phoneme alphabet edges** return a `PhonemizedChunks`
  (`list[list[str]]`) — the same shape that `TTSVoice.phonemize()` has always
  returned, so the tokenisation contract with existing engines is unbroken.
- **All other edges** return a plain `str`.

## Conversion graph and chain support

The registry `ALPHABET_CONVERTERS` maps `(src_alphabet, tgt_alphabet)` pairs to
callable edges.  When there is no direct edge, `convert` does a **BFS** over the
graph and composes a chain of edges automatically.

```
GRAPHEMES ──espeak──► IPA
GRAPHEMES ──────────► HANGUL
GRAPHEMES ──pykakasi─► HIRA
HANGUL ──────────────► HIRA   (Jamo decomposition)
IPA ◄──scriptconv──► ARPA / X-SAMPA / RFE / COTOVIA   (phoneme-notation edges)
HIRA ◄──scriptconv──► KANA
```

### Phoneme-notation edges (scriptconv)

The `IPA ↔ ARPA / X-SAMPA / RFE / Cotovía` and `Hiragana ↔ Katakana` edges are
pure symbol transcodes — no language, no phonemization — and are delegated to
[scriptconv](https://github.com/TigreGotico/scriptconv), the single source of
truth for those tables. Because every notation converts to and from IPA, the BFS
chains them automatically, e.g. `X-SAMPA → IPA → ARPA`.

Example multi-hop: `HANGUL → HIRA` uses `HANGUL → HIRA` directly, but if only
`GRAPHEMES → HANGUL` and `HANGUL → HIRA` were registered the BFS would compose
them automatically.

When no path is found, `convert` returns the input unchanged and logs a debug
message — it never raises.

## Phonemization = the graphemes→phoneme converter family

Phonemization is not a special case; it is the family of conversion edges with
`src = Alphabet.GRAPHEMES` and `tgt` = a phoneme alphabet (IPA, ARPA, …).

The `phoneme_type` parameter passed to `convert` selects which backend to use
within that family.  This mirrors exactly what `VoiceConfig.phoneme_type` has
always done: it chooses the backend (espeak, gruut, misaki, …) **and** the
tokenisation recipe that goes with it.

```python
# These two calls are equivalent:
voice.phonemize("hello")                       # old path
convert("hello", "en", Alphabet.GRAPHEMES,
        voice.config.alphabet,
        phoneme_type=voice.config.phoneme_type) # new unified path
```

## CJK extras

Script conversion edges for Japanese and Chinese require optional dependencies:

```
pip install phoonnx[cjk]
# installs: pykakasi==2.3.0, spacy-pkuseg
```

These are also included in `phoonnx[all]`.

## Synthesis dispatch

`TTSVoice.synthesize` chooses how to reach the model's alphabet
(`VoiceConfig.alphabet`) from the caller's input (`SynthesisConfig.alphabet`,
default graphemes):

- **`src == tgt`** — grapheme/text-token models tokenise via the adapter;
  already-phonemic input in the model's own alphabet passes straight through
  (no re-phonemization).
- **grapheme input** — phonemized by the model's own phonemizer. Language-aware
  phonemizers (e.g. Shami) provide per-phoneme language IDs, which are carried
  through here; grapheme→phoneme is phonemization, so it does **not** go through
  the conversion graph.
- **already-phonemic input in a different alphabet** — transcoded to the model's
  alphabet through the conversion graph (the scriptconv notation edges).

The rule of thumb is the same one that separates the two layers: anything that
needs the language is phonemization and stays in phoonnx; the `lang`-free
phoneme↔phoneme hops are scriptconv edges.
