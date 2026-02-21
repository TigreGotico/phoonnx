# Phonemizers

phoonnx supports a large number of phonemizer backends, allowing it to work with voices trained on different phoneme representations. The phonemizer is selected via the `PhonemeType` enum and configured through `Alphabet`.

## PhonemeType Reference

| Value | Description | Languages |
|-------|-------------|-----------|
| `graphemes` | Raw text characters (no phonemization) | Any |
| `unicode` | Unicode codepoints | Any |
| `espeak` | eSpeak-ng (IPA or ARPA output) | 100+ languages |
| `gruut` | Gruut phonemizer | European languages |
| `goruut` | GoRuut phonemizer | European languages |
| `epitran` | Epitran G2P | Many languages |
| `byt5` | ByT5 neural G2P | Multilingual |
| `charsiu` | CharSiu (ByT5 variant with special whitespace handling) | Multilingual |
| `transphone` | Transphone | Multilingual |
| `misaki` | Misaki (for Kokoro-style models) | en, ja |
| `deepphonemizer` | DeepPhonemizer | English |
| `openphonemizer` | OpenPhonemizer | English |
| `g2pen` | g2p-en | English |
| `tugaphone` | TugaPhone | Portuguese |
| `g2pfa` | Persian G2P | Farsi/Persian |
| `openjtalk` | Open JTalk | Japanese |
| `cutlet` | Cutlet | Japanese |
| `pykakasi` | PyKakasi | Japanese |
| `cotovia` | Cotovia (requires system binary) | Galician |
| `phonikud` | Phonikud | Hebrew |
| `mantoq` | Mantoq | Arabic |
| `viphoneme` | VIPhoneme | Vietnamese |
| `g2pk` | G2PK | Korean |
| `kog2p` | KoG2P-K | Korean |
| `g2pc` | G2pC | Chinese |
| `g2pm` | G2pM | Chinese |
| `pypinyin` | PyPinyin | Chinese (Mandarin) |
| `xpinyin` | XPinyin | Chinese (Mandarin) |
| `jieba` | Jieba (word segmentation, not a true phonemizer) | Chinese |
| `mwl_phonemizer` | Mirandese phonemizer | Mirandese |

## Alphabet Reference

The `Alphabet` enum controls the output representation of the phonemizer:

| Value | Description |
|-------|-------------|
| `ipa` | International Phonetic Alphabet |
| `arpa` | ARPAbet (English) |
| `sampa` | SAMPA phoneme set |
| `x-sampa` | X-SAMPA |
| `rfe` | RFE Phonetic Alphabet |
| `unicode` | Unicode characters |
| `hangul` | Korean Hangul |
| `kana` | Japanese Katakana |
| `hira` | Japanese Hiragana |
| `hepburn` | Hepburn romanization (Japanese) |
| `kunrei` | Kunrei romanization (Japanese) |
| `nihon` | Nihon romanization (Japanese) |
| `pinyin` | Pinyin (Chinese) |
| `hanzi` | Chinese characters |
| `eraab` | ERAAB (Persian) |
| `cotovia` | Cotovia phoneme set (Galician) |
| `buckwalter` | Buckwalter transliteration (Arabic) |

## Selecting a Phonemizer

The phonemizer is typically declared in the model's `model.json` config and loaded automatically. You can override it at load time:

```python
from phoonnx.voice import TTSVoice

voice = TTSVoice.load(
    "model.onnx",
    "model.json",
    phoneme_type_str="espeak",
    alphabet_str="ipa",
)
```

Or instantiate one directly:

```python
from phoonnx.config import get_phonemizer, PhonemeType, Alphabet

phonemizer = get_phonemizer(PhonemeType.ESPEAK, alphabet=Alphabet.IPA)
phonemes = phonemizer.phonemize("Hello world", lang="en-US")
```

## Special Language Notes

### Arabic / Hebrew

Models trained with Arabic or Hebrew may need diacritics added before phonemization. Enable this via `SynthesisConfig`:

```python
from phoonnx.config import SynthesisConfig
syn_config = SynthesisConfig(add_diacritics=True)
```

Or set it in the voice config JSON: `"add_diacritics": true`. For Arabic, it is enabled automatically when `lang_code` starts with `"ar"`.

### Galician (Cotovia)

The `cotovia` phonemizer requires the `cotovia` binary to be installed. phoonnx searches for it in `PATH`, a bundled binary, and `/usr/bin/cotovia`. The alphabet can be either native Cotovia or IPA.

### Chinese

Multiple backends are available. `pypinyin` and `xpinyin` produce Pinyin output. `g2pc` and `g2pm` produce IPA. `jieba` is a word segmenter and not a true phonemizer on its own.

## BasePhonemizer

All phonemizers inherit from `BasePhonemizer` which provides:

- `phonemize(text, lang)` — returns `PhonemizedChunks` (list of sentences, each a list of phoneme strings)
- `phonemize_string(text, lang)` — returns raw phoneme string for a single chunk
- `add_diacritics(text, lang)` — adds vowel diacritics (Arabic/Hebrew)
- `chunk_text(text)` — splits text into sentence chunks with punctuation
