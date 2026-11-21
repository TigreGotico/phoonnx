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

## Conversion is scriptconv's job

phoonnx does not implement alphabet/phoneme conversion — that is
[`scriptconv`](https://github.com/TigreGotico/scriptconv)'s conversion graph.
`scriptconv.graph.DEFAULT_GRAPH` carries the notation (IPA↔ARPA↔X-SAMPA↔…) and
script (Hangul, kana, cangjie) edges; `scriptconv.phonemizers.register` adds the
`text → ipa` phonemization edge (per-language default, `override=` to force a
specific engine); `scriptconv.diacritics.register` adds diacritization.

```python
from scriptconv.graph import DEFAULT_GRAPH
from scriptconv import phonemizers
from scriptconv.phonemizers.enums import Phonemizer

g = DEFAULT_GRAPH.extend(phonemizers.register)

# graphemes → IPA (phonemization via espeak)
ipa = g.convert("hello world", "text", "ipa",
                lang="en", override=Phonemizer.ESPEAK)

# already-phonemic transcode (IPA → ARPABET) — no phonemizer needed
arpa = DEFAULT_GRAPH.convert("h ə l oʊ", "ipa", "arpa")
```

`TTSVoice` builds this route internally: grapheme input goes through the
voice's phonemizer, already-phonemic input is transcoded via
`DEFAULT_GRAPH.convert`.

### Return type

`convert` returns a plain `str` for every edge, the phonemization edge included.
`TTSVoice.phonemize()` returns `PhonemizedChunks` (`list[list[str]]`): one list of
symbols for each sentence, which is the shape the tokenizer reads.

## Conversion graph and routing

`DEFAULT_GRAPH` is a `scriptconv.graph.ConversionGraph`. Its nodes are
representations, such as `ipa`, `arpa`, `hira` and `hanzi`. Its edges are
registered transforms from one node to another.

`convert(text, src, dst, **context)` finds the cheapest path of edges and runs each edge in turn. A lossy edge costs
more than a lossless one, so the router prefers lossless paths. `route(src, dst)`
returns the path and does not run it.

These are the edges of `DEFAULT_GRAPH.extend(phonemizers.register)`:

```
text, text-diacritized ──phonemizer──► IPA          (phonemizers.register)
IPA ◄──► ARPA / X-SAMPA / Kirshenbaum / Lexique / RFE / Cotovía
HALABI ──► IPA
arabic ◄──► buckwalter
japanese ──► hira / kana                            (optional dictionary)
hira ◄──► kana
hangul ──► jamo
hanzi ──► pinyin / bopomofo                         (optional dictionary)
hanzi ──► cangjie
```

An edge that needs an optional dictionary raises `ImportError` with an install
hint when the dictionary is not installed.

### Phoneme-notation edges (scriptconv)

The notation edges, such as `IPA ↔ ARPA`, and the `Hiragana ↔ Katakana` edges are
pure symbol transcodes — no language, no phonemization — and are delegated to
[scriptconv](https://github.com/TigreGotico/scriptconv), the single source of
truth for those tables. Every notation converts to and from IPA, so the router
chains two of them through IPA:

```python
print([(edge.src, edge.dst) for edge in DEFAULT_GRAPH.route("x-sampa", "arpa")])

try:
    DEFAULT_GRAPH.convert("h ə l oʊ", "ipa", "hanzi")
except ValueError as error:
    print(error)
```

```text
[('x-sampa', 'ipa'), ('ipa', 'arpa')]
no conversion path 'ipa' -> 'hanzi'; reachable from 'ipa': arpa, cotovia, kirshenbaum, lexique, rfe, x-sampa
```

When no path is found, `convert` and `route` raise `ValueError`. The message
names the nodes that the source reaches.

## Phonemization = the graphemes→phoneme converter family

Phonemization is not a special case; it is the family of conversion edges from
`text` to a phoneme alphabet.

The `override=` context of `convert` selects the backend of the `text → ipa`
edge. Without it, the default phonemizer of the language runs. A voice does the
same with `VoiceConfig.phoneme_type`: it chooses the backend (espeak, gruut,
misaki, …) **and** the tokenisation recipe that goes with it.

A voice phonemizes through its `TextFrontend`, `TTSVoice.frontend`. The two calls
below give the same symbols:

```python
from phoonnx.config import VoiceConfig
from phoonnx.voice import TTSVoice

config = VoiceConfig.from_dict({
    "phoonnx_version": "1.0",
    "lang_code": "en-GB",
    "phoneme_type": "espeak",
    "alphabet": "ipa",
    "phoneme_id_map": {"_": [0], "^": [1], "$": [2]},
})
voice = TTSVoice(session=None, config=config)
print(voice.phonemize("hello world"))
print(voice.frontend.phonemize("hello world"))
```

```text
[['h', 'ə', 'l', 'ˈ', 'ə', 'ʊ', ' ', 'w', 'ˈ', 'ɜ', 'ː', 'l', 'd']]
[['h', 'ə', 'l', 'ˈ', 'ə', 'ʊ', ' ', 'w', 'ˈ', 'ɜ', 'ː', 'l', 'd']]
```

This example needs the `espeak` extra only. When the `espeak-ng` program is not on
`PATH`, scriptconv uses espyak, a Python port of espeak-ng from the same extra, and
the output is the same.

## CJK extras

Script conversion edges for Japanese and Chinese require optional dependencies:

```
pip install phoonnx[cjk]
# installs: pykakasi>=2.3.0, spacy-pkuseg
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
