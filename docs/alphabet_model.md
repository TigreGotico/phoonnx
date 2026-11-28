# Alphabet conversion model

<!-- agent-hint
covers: how VoiceConfig.alphabet, VoiceConfig.phoneme_type and SynthesisConfig.input_alphabet select the conversion of input text, the scriptconv conversion graph, and the respelling of r-coloured vowels
source: phoonnx/voice.py:TTSVoice._iter_phonemes, phoonnx/voice.py:_respell_rhotics, phoonnx/config.py:SynthesisConfig, scriptconv.graph:DEFAULT_GRAPH, pyproject.toml:cjk
generated: no
verify: .venv/bin/python -m doctest -o ELLIPSIS -o NORMALIZE_WHITESPACE docs/alphabet_model.md && .venv/bin/python -c "import re,sys,tomllib; p=open('docs/alphabet_model.md').read(); cjk=tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies']['cjk']; doc=re.search(chr(10)+'# installs: (.*)',p).group(1).split(', '); print(doc,cjk); sys.exit(0 if doc==cjk else 1)"
see: phonemizers.md, architecture.md, configuration.md
-->

This page explains how phoonnx turns the text of a call into the symbols that a model reads. It is for voice authors and developers who give phonemes, not plain text, to a voice.

## The three alphabet fields

Three fields decide the conversion:

| Field | Tells | Example |
|---|---|---|
| `VoiceConfig.alphabet` | The symbols that the model reads | `Alphabet.IPA` |
| `VoiceConfig.phoneme_type` | The phonemizer that makes those symbols from text, and the tokenizer rules that go with it | `PhonemeType.ESPEAK` |
| `SynthesisConfig.input_alphabet` | The symbols of the text that the caller gives | `None`, which is plain text |

`SynthesisConfig.input_alphabet` is `None` by default, and phoonnx reads `None` as `Alphabet.GRAPHEMES`, plain text. Set it only when you give text that is already phonemic, such as an IPA or ARPABET string.

## How a call reaches the model alphabet

`TTSVoice.prepare` and `synthesize` select one of two paths from `SynthesisConfig.input_alphabet`:

- Plain text goes through `TTSVoice.frontend`, the scriptconv `TextFrontend` of the voice. The frontend normalizes the text, splits it into sentences and runs the phonemizer of `phoneme_type`. A phonemizer that gives side streams, such as the language ids of Shami, gives them here.
- Phonemic text goes through the scriptconv conversion graph, from `SynthesisConfig.input_alphabet` to `VoiceConfig.alphabet`. When the two are equal, the text stays as it is. When the graph has no path, phoonnx keeps the text as written and logs a debug message. phoonnx then splits the text into sentences. ARPABET splits on spaces, and each other alphabet splits into characters.

This example gives ARPABET to an IPA voice. The graph converts `B ER D` to `bɝd`:

```python
>>> from phoonnx.config import Alphabet, SynthesisConfig, VoiceConfig
>>> from phoonnx.voice import TTSVoice
>>> symbols = {"_": [0], "^": [1], "$": [2], " ": [3], "b": [4], "d": [5], "ɜ": [6], "ɹ": [7], "ː": [8]}
>>> def arpabet_to_voice(phoneme_type):
...     config = VoiceConfig.from_dict({"phoonnx_version": "1.0", "lang_code": "en-US",
...                                     "phoneme_type": phoneme_type, "alphabet": "ipa",
...                                     "phoneme_id_map": symbols})
...     voice = TTSVoice(session=None, config=config)
...     return [u.symbols for u in voice.prepare("B ER D", SynthesisConfig(input_alphabet=Alphabet.ARPA))]
>>> arpabet_to_voice("gruut")
[('b', 'ɜ', 'ɹ', 'd')]
>>> arpabet_to_voice("espeak")
[('b', 'ɜ', 'ː', 'd')]

```

The vocabulary of this voice has no `ɝ`. ARPABET text on an IPA voice can name an r-coloured vowel that the vocabulary does not have. The voice then uses the first spelling in this list that its vocabulary has: `ɝ` as `ɜɹ`, `ɚ` or `əɹ`, and `ɚ` as `əɹ`. For a voice with `phoneme_type: espeak`, `ɝ` becomes `ɜː` first, because espeak-ng writes the stressed r-coloured vowel that way. A voice that has the symbol keeps it. When no spelling fits, the tokenizer drops the symbol and logs a warning. phoonnx respells only ARPABET input on an IPA voice.

## The scriptconv conversion graph

phoonnx does not convert between alphabets itself. The conversion graph of [scriptconv](https://github.com/TigreGotico/scriptconv) does it. `scriptconv.graph.DEFAULT_GRAPH` is a `ConversionGraph`. Its nodes are notations and scripts, such as `ipa`, `arpa`, `hira` and `hanzi`. Its edges are the transforms from one node to another. `scriptconv.phonemizers.register` adds the phonemizer edges from `text` to `ipa`:

```python
>>> from scriptconv.graph import DEFAULT_GRAPH
>>> from scriptconv import phonemizers
>>> from scriptconv.phonemizers.enums import Phonemizer
>>> graph = DEFAULT_GRAPH.extend(phonemizers.register)
>>> graph.convert("hello world", "text", "ipa", lang="en", override=Phonemizer.ESPEAK)
'həlˈəʊ wˈɜːld'
>>> DEFAULT_GRAPH.convert("h ə l oʊ", "ipa", "arpa")
'HH AX L OW'

```

The `override` argument selects the phonemizer of the `text` to `ipa` edge. Without it, the default phonemizer of the language runs. A voice selects its phonemizer with `VoiceConfig.phoneme_type` in the same way. The espeak example needs the `espeak` extra. When the `espeak-ng` program is not on `PATH`, scriptconv uses espyak, a Python port of espeak-ng from the same extra, and the output is the same.

`convert(text, src, dst, **context)` finds the path of edges with the lowest cost and runs each edge in turn. An edge that loses information costs more, so the graph prefers lossless paths. `route(src, dst)` gives the path and does not run it. Each notation converts to and from IPA, so the graph joins two notations through IPA:

```python
>>> [(edge.src, edge.dst) for edge in DEFAULT_GRAPH.route("x-sampa", "arpa")]
[('x-sampa', 'ipa'), ('ipa', 'arpa')]
>>> DEFAULT_GRAPH.convert("h ə l oʊ", "ipa", "hanzi")
Traceback (most recent call last):
...
ValueError: no conversion path 'ipa' -> 'hanzi'; reachable from 'ipa': arpa, cotovia, kirshenbaum, lexique, rfe, x-sampa

```

When there is no path, `convert` and `route` raise `ValueError`, and the message names the nodes that the source reaches. `convert` gives a `str` for each edge, the phonemizer edges included. `TTSVoice.phonemize` gives one list of symbols for each sentence, which is the shape that the tokenizer reads.

These are the edges of the extended graph:

```python
>>> sorted({(edge.src, edge.dst) for edge in graph.edges})
[('arabic', 'buckwalter'), ('arpa', 'ipa'), ('buckwalter', 'arabic'), ('cotovia', 'ipa'), ('halabi', 'ipa'), ('hangul', 'jamo'), ('hanzi', 'bopomofo'), ('hanzi', 'cangjie'), ('hanzi', 'pinyin'), ('hira', 'kana'), ('ipa', 'arpa'), ('ipa', 'cotovia'), ('ipa', 'kirshenbaum'), ('ipa', 'lexique'), ('ipa', 'rfe'), ('ipa', 'x-sampa'), ('japanese', 'hira'), ('japanese', 'kana'), ('kana', 'hira'), ('kirshenbaum', 'ipa'), ('lexique', 'ipa'), ('misaki', 'ipa'), ('rfe', 'ipa'), ('text', 'ipa'), ('text-diacritized', 'ipa'), ('x-sampa', 'ipa')]

```

An edge that needs an optional dictionary raises `ImportError` with an install hint when the dictionary is not installed. The Japanese and Chinese script edges need the `cjk` extra, which the `all` extra also installs:

```
pip install phoonnx[cjk]
# installs: pykakasi>=2.3.0, spacy-pkuseg
```

---

[← Phonemizers](phonemizers.md) · [Home](index.md) · [OVOS plugin →](ovos_plugin.md)
