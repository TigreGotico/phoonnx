# Phonemizers

This page is the phonemizer backend catalog for advanced users and voice authors. After
reading it you will know every `PhonemeType`, the `Alphabet` values they emit, and how a voice
selects one. The phonemizer is selected via the `PhonemeType` enum and configured through
`Alphabet`; both live in `phoonnx.config`.

## PhonemeType Reference

Every value below is a member of `PhonemeType`. Language-specific backends require the
matching [install extra](installation.md#language-extras).

| Value | Description | Languages |
|-------|-------------|-----------|
| `graphemes` | Raw text characters (no phonemization) | Any |
| `unicode` | Unicode codepoints | Any |
| `espeak` | eSpeak (IPA output) | 100+ languages |
| `gruut` | Gruut phonemizer | Multiple |
| `goruut` | GoRuut / pygoruut phonemizer | Multiple |
| `epitran` | Epitran G2P | Many languages |
| `byt5` | ByT5 neural G2P | Multilingual |
| `charsiu` | CharSiu (ByT5 variant with special whitespace handling) | Multilingual |
| `transphone` | Transphone | Multilingual |
| `misaki` | Misaki back-compat dispatcher across the misaki languages | Multiple |
| `misaki_en` | Misaki English | English |
| `misaki_ja` | Misaki Japanese | Japanese |
| `misaki_zh` | Misaki Chinese (IPA or bopomofo per alphabet) | Chinese |
| `misaki_ko` | Misaki Korean | Korean |
| `misaki_vi` | Misaki Vietnamese | Vietnamese |
| `deepphonemizer` | DeepPhonemizer | English |
| `openphonemizer` | OpenPhonemizer | English |
| `g2pen` | g2p-en | English |
| `tugaphone` | TugaPhone | Portuguese |
| `g2pfa` | Persian G2P | Persian |
| `openjtalk` | Open JTalk | Japanese |
| `cutlet` | Cutlet | Japanese |
| `pykakasi` | PyKakasi | Japanese |
| `cotovia` | Cotovia (see [Galician](galician.md)) | Galician |
| `ahotts` | AhoTTS G2P (variant via `phonemizer_model`) | Basque |
| `phonikud` | Phonikud | Hebrew |
| `mantoq` | Mantoq | Arabic |
| `viphoneme` | VIPhoneme | Vietnamese |
| `g2pk` | G2PK | Korean |
| `kog2p` | KoG2P | Korean |
| `g2pc` | G2pC | Chinese |
| `g2pm` | G2pM | Chinese |
| `pypinyin` | PyPinyin | Chinese |
| `xpinyin` | XPinyin | Chinese |
| `jieba` | Jieba (word segmentation, not a true phonemizer) | Chinese |
| `mwl_phonemizer` | Mirandese phonemizer | Mirandese |
| `vosk` | vosk-tts phoneme inventory (use with `Alphabet.VOSK`) | Russian |
| `shami` | Levantine Arabic / English code-switching front-end (emits per-phoneme language IDs) | ar-LB / en |
| `arbtok` | Dialect-aware Arabic on undiacritized text (o2i lattice; register via `phonemizer_model`) | Arabic |
| `euskaphone` | Dialect-aware Basque (o2i lattice) | Basque |
| `barranquenho` | Barranquenho contact variety (o2i lattice) | Barranquenho |
| `orthography2ipa` | Multilingual data-driven IPA (o2i lattice) | Multilingual |

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
| `bopomofo` | Zhuyin / Bopomofo (Chinese; misaki representation) |
| `hanzi` | Chinese characters |
| `eraab` | ERAAB (Persian) |
| `cotovia` | Cotovia phoneme set (Galician) |
| `buckwalter` | Buckwalter transliteration (Arabic) |
| `graphemes` | Raw text characters |
| `vosk` | vosk-tts phoneme inventory (Russian) |
| `cangjie` | Cangjie input-method decomposition (Chinese) |

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

`get_phonemizer(phoneme_type, alphabet=Alphabet.IPA, model=None)` takes an optional third
argument, `model`, which is the voice's `phonemizer_model`. It selects a variant for the
backends that support one, for example: the AhoTTS engine variant (`classic` / `modern` /
`northern`), the Cotovia notation (`stress` for the stress-marked HiTZ Galician model), the
arbtok register (`iʿrab` for the full case-ending register), or the model id/path for neural
G2P (ByT5, CharSiu, DeepPhonemizer).

Beyond `phonemize()` (grouped by sentence), every backend exposes `phonemize_to_list(text,
lang)` for a flat phoneme list and `phonemize_string(text, lang)` for a single chunk.

## Special Language Notes

### Arabic / Hebrew

Models trained with Arabic or Hebrew may need diacritics added before phonemization. Enable this via `SynthesisConfig`:

```python
from phoonnx.config import SynthesisConfig
syn_config = SynthesisConfig(add_diacritics=True)
```

Or set it in the voice config JSON: `"add_diacritics": true`. For Arabic, it is enabled automatically when `lang_code` starts with `"ar"`.

Arabic diacritization uses [`text2tashkeel`](https://pypi.org/project/text2tashkeel/), installed
by the [`ar` extra](installation.md#language-extras). The diacritizer model defaults to
`rawi-ensemble` (which also restores hamza and the dagger alef); override it with the
`diacritizer_model` field on the voice config or per call on `SynthesisConfig`. Requesting
Arabic diacritics without `text2tashkeel` installed raises a clear `ImportError`. Hebrew uses
Phonikud.

### Galician (Cotovia)

The `cotovia` phonemizer requires the `cotovia` binary to be installed. phoonnx searches for it in `PATH`, a bundled binary, and `/usr/bin/cotovia`. The alphabet can be either native Cotovia or IPA.

### Chinese

Multiple backends are available. `pypinyin` and `xpinyin` produce Pinyin output. `g2pc` and `g2pm` produce IPA. `jieba` is a word segmenter and not a true phonemizer on its own.

## BasePhonemizer

All phonemizers inherit from `BasePhonemizer` which provides:

- `phonemize(text, lang)` — returns `PhonemizedChunks` (list of sentences, each a list of phoneme strings)
- `phonemize_string(text, lang)` — returns raw phoneme string for a single chunk
- `chunk_text(text)` — splits text into sentence chunks with punctuation

Diacritization (Arabic tashkeel, Hebrew niqqud, and the other scripts scriptconv
supports) is not a phonemizer method — it is a separate step exposed as
`scriptconv.diacritics.diacritize(text, lang)`, which phoonnx calls before
phonemizing when a voice enables `add_diacritics`. scriptconv owns the
diacritizer backends and auto-provisions the Hebrew phonikud model.

## G2P for OVOS

This phonemizer layer was extracted into [scriptconv](https://github.com/TigreGotico/scriptconv);
phoonnx now depends on it and keeps thin compatibility shims for the names
above. OVOS users who want grapheme-to-phoneme conversion on its own, without
the rest of phoonnx's TTS pipeline, should use
[ovos-scriptconv-g2p-plugin](https://github.com/TigreGotico/ovos-scriptconv-g2p-plugin),
an OPM `opm.g2p` plugin built directly on scriptconv — see its
[docs/ovos.md](https://github.com/TigreGotico/scriptconv/blob/dev/docs/ovos.md)
for install and configuration.

## Attribution

phoonnx wraps and, where licensing allows, bundles domain-specific G2P work, including:

- [cotovia](https://github.com/TigreGotico/cotovia-mirror) — Galician phonemization (bundled binaries)
- [mantoq](https://github.com/mush42/mantoq) — Arabic phonemization; [text2tashkeel](https://pypi.org/project/text2tashkeel/) — Arabic diacritization
- [hams-levantine-tts](https://github.com/Al-aminI/hams-levantine-tts) — the Shami (Levantine Arabic / English) front-end
- [KoG2P](https://github.com/scarletcho/KoG2P) and [hangul_to_ipa](https://github.com/stannam/hangul_to_ipa) — Korean phonemization and Hangul→IPA
- [arpa2ipa](https://github.com/chorusai/arpa2ipa) — ARPAbet→IPA conversion
- Chinese number verbalization from [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)

Multilingual and neural backends include eSpeak, [gruut](https://github.com/rhasspy/gruut),
[epitran](https://github.com/dmort27/epitran), [misaki](https://github.com/hexgrad/misaki),
[transphone](https://github.com/xinjli/transphone),
[Charsiu](https://github.com/lingjzhu/CharsiuG2P) and OpenVoiceOS ByT5 G2P models.
