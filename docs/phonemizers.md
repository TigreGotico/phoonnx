# Phonemizers

<!-- agent-hint
covers: every PhonemeType and Alphabet value, and how a voice selects and runs its phonemizer
source: phoonnx/config.py:PhonemeType, phoonnx/config.py:Alphabet (re-exported from scriptconv.phonemizers), scriptconv.frontend:TextFrontend, scriptconv.diacritics, phoonnx/config.py:VoiceConfig.add_diacritics
generated: no
verify: python -I -c "from phoonnx.config import PhonemeType, Alphabet; print(len(PhonemeType), len(Alphabet))" (prints 54 23), plus the frontend examples on the page
see: galician.md, architecture.md, installation.md
-->

This page lists every phonemizer backend and output alphabet, and shows how a voice selects one, for voice authors and advanced users.

The `PhonemeType` enum selects the phonemizer, and the `Alphabet` enum sets its output.
Both live in `phoonnx.config`.

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
| `mwl_phonemizer` | Mirandese, through scriptconv on the orthography2ipa specs `mwl`, `mwl-x-sendim` and `mwl-x-ifanes` | Mirandese |
| `vosk` | vosk-tts phoneme inventory (use with `Alphabet.VOSK`) | Russian |
| `shami` | Levantine Arabic / English code-switching front-end (emits per-phoneme language IDs) | apc / en |
| `dizione` | American English: CMUdict lexicon, homograph rules and its own text rules; emits the tiny-voca tokens with a stress value each | en-US |
| `arbtok` | Dialect-aware Arabic on undiacritized text (o2i lattice; register via `phonemizer_model`) | Arabic |
| `euskaphone` | Dialect-aware Basque (o2i lattice) | Basque |
| `barranquenho` | Barranquenho contact variety, through scriptconv on the orthography2ipa spec `ext-PT-x-barranco` | Barranquenho |
| `orthography2ipa` | Multilingual data-driven IPA (o2i lattice) | Multilingual |
| `phonetisaurus` | Phonetisaurus WFST G2P. The model sets the language and the symbols | Any, per model |
| `g2pw` | g2pW polyphone model: pinyin initials, finals and tones | Chinese |
| `f5tts_char` | The text as written, one token per character (Habibi-TTS, SILMA TTS) | Any |
| `f5tts_pinyin` | F5-TTS `convert_char_to_pinyin`: hanzi as pinyin syllables, other characters as written | Chinese |
| `halabi` | Halabi Arabic-Phonetiser, with no diacritizer: the input must have its vowels | Arabic |
| `iqra` | Halabi Arabic-Phonetiser output in the IqraEval `phoneme_ref` convention | Arabic |
| `africa_g2p` | africa-g2p rules: native-orthography units or IPA | 400+ African languages |
| `ghana_g2p` | africa-g2p rules, patched for Ghanaian languages: native units or IPA | 42 Ghanaian languages |
| `matoub_kab` | The Kabyle rule set that the Matoub-82M voice was trained on | Kabyle |
| `leokani` | Leokani: lexicon, ONNX letter-to-sound and context models; emits the Kokoro (misaki) codes | en-US |
| `kolokwa` | kolokwa-g2p model | Liberian English (lir) |

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
| `halabi` | Halabi Arabic-Phonetiser phoneme notation (Arabic) |
| `africa_g2p` | africa-g2p native-orthography phoneme units (African languages) |

## Selecting a Phonemizer

The voice config declares the phonemizer, and phoonnx loads it with the voice. You can override it at load time:

```python
from phoonnx.voice import TTSVoice

voice = TTSVoice.load(
    "model.onnx",
    "model.json",
    phoneme_type_str="espeak",
    alphabet_str="ipa",
)
```

The voice reads text through `voice.frontend`, a scriptconv `TextFrontend` over the voice's
phonemizer. It gives the phonemes the voice synthesizes, one list per sentence:

```python
print(voice.frontend.phonemize("Olá mundo. Como estás?"))
```

With the `OpenVoiceOS/phoonnx_pt-PT_miro_unicode` voice, which uses the `unicode`
phonemizer, the output is:

```text
[['O', 'l', 'a', '́', ' ', 'm', 'u', 'n', 'd', 'o', '.'], ['C', 'o', 'm', 'o', ' ', 'e', 's', 't', 'a', '́', 's', '?']]
```

To phonemize without a voice, build a `TextFrontend` directly:

```python
from scriptconv.frontend import FrontendConfig, TextFrontend
from scriptconv.phonemizers import Phonemizer

frontend = TextFrontend(FrontendConfig("en-US", Phonemizer.ESPEAK))
print(frontend.phonemize("Hello world"))
```

The eSpeak backend needs the `espeak` extra. The output is:

```text
[['h', 'ə', 'l', 'ˈ', 'o', 'ʊ', ' ', 'w', 'ˈ', 'ɜ', 'ː', 'l', 'd']]
```

A voice's `phonemizer_model` selects a variant for the backends that support one, for
example: the AhoTTS engine variant (`classic` / `modern` / `northern`), the Cotovia notation
(`stress` for the stress-marked HiTZ Galician model), the arbtok register (`iʿrab` for the
full case-ending register), or the model id/path for neural G2P (ByT5, CharSiu,
DeepPhonemizer). `FrontendConfig.model` takes the same value.

`phoonnx.config.get_phonemizer` and `phoonnx.config.get_conversion` are deprecated. They
warn, and the warning names the release that removes them.

Besides `phonemize()`, which groups by sentence, every backend has `phonemize_to_list(text,
lang)` for a flat phoneme list and `phonemize_string(text, lang)` for a single chunk.
`frontend.backend` gives the backend of a `TextFrontend`:

```python
print(frontend.backend.phonemize_string("Hello world", "en-US"))
```

```text
həlˈoʊ wˈɜːld
```

## Special Language Notes

### Diacritics, stress and sense marks

The diacritize stage adds marks before phonemization. It restores vowels for Arabic and Hebrew, marks stress for Russian, Ukrainian and other stress languages, and marks homograph readings for Portuguese. A voice turns it on in its config JSON, under `inference`:

```json
"inference": {"add_diacritics": true}
```

A key at the top level of the config is not read. When the config has no key, the stage is on for Arabic language tags (`ar` and the Arabic varieties written in Arabic script, such as `apc` and `arz`) and off for every other language. A call can override the voice with `SynthesisConfig`:

```python
from phoonnx.config import SynthesisConfig
syn_config = SynthesisConfig(add_diacritics=True)
```

Arabic diacritization uses [`text2tashkeel`](https://pypi.org/project/text2tashkeel/), installed
by the [`ar` extra](installation.md#language-extras). The diacritizer model defaults to
`rawi-ensemble` (which also restores hamza and the dagger alef); override it with the
`diacritizer_model` field on the voice config or per call on `SynthesisConfig`. Requesting
Arabic diacritics without `text2tashkeel` installed raises an `ImportError` that names the package. Hebrew uses
Phonikud.

### Galician (Cotovia)

The `cotovia` phonemizer runs [pycotovia](https://github.com/TigreGotico/pycotovia), a pure-Python port of the Cotovia G2P engine, through scriptconv. Install it with the scriptconv `gl` extra. The alphabet can be either native Cotovia or IPA. See [galician.md](galician.md).

### Chinese

Several backends cover Chinese. `pypinyin` and `xpinyin` produce Pinyin output. `g2pc` and `g2pm` produce IPA. `jieba` is a word segmenter and not a true phonemizer on its own.

## BasePhonemizer

All phonemizers inherit from `BasePhonemizer` which provides:

- `phonemize(text, lang)` — returns `PhonemizedChunks` (list of sentences, each a list of phoneme strings)
- `phonemize_string(text, lang)` — returns raw phoneme string for a single chunk
- `chunk_text(text)` — splits text into sentence chunks with punctuation

Diacritization (Arabic tashkeel, Hebrew niqqud, and the other scripts scriptconv
supports) is not a phonemizer method. It is a separate step,
`scriptconv.diacritics.diacritize(text, lang)`, which phoonnx calls before
phonemizing when a voice enables `add_diacritics`. scriptconv owns the
diacritizer backends and auto-provisions the Hebrew phonikud model.

## G2P for OVOS

The phonemizers on this page come from [scriptconv](https://github.com/TigreGotico/scriptconv).
phoonnx re-exports `PhonemeType` and `Alphabet` from it. For grapheme-to-phoneme
conversion in OpenVoiceOS without the TTS pipeline, use
[ovos-scriptconv-g2p-plugin](https://github.com/TigreGotico/ovos-scriptconv-g2p-plugin),
an OPM `opm.g2p` plugin built on scriptconv. scriptconv's
[docs/ovos.md](https://github.com/TigreGotico/scriptconv/blob/dev/docs/ovos.md)
gives the install and configuration steps.

## Attribution

phoonnx wraps and, where licensing allows, bundles domain-specific G2P work, including:

- [cotovia](https://github.com/TigreGotico/cotovia-mirror) — Galician phonemization, through its pure-Python port [pycotovia](https://github.com/TigreGotico/pycotovia)
- [mantoq](https://github.com/mush42/mantoq) — Arabic phonemization; [text2tashkeel](https://pypi.org/project/text2tashkeel/) — Arabic diacritization
- [hams-levantine-tts](https://github.com/Al-aminI/hams-levantine-tts) — the Shami (Levantine Arabic / English) front-end
- [KoG2P](https://github.com/scarletcho/KoG2P) and [hangul_to_ipa](https://github.com/stannam/hangul_to_ipa) — Korean phonemization and Hangul→IPA
- [arpa2ipa](https://github.com/chorusai/arpa2ipa) — ARPAbet→IPA conversion
- Chinese number verbalization from [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)

Multilingual and neural backends include eSpeak, [gruut](https://github.com/rhasspy/gruut),
[epitran](https://github.com/dmort27/epitran), [misaki](https://github.com/hexgrad/misaki),
[transphone](https://github.com/xinjli/transphone),
[Charsiu](https://github.com/lingjzhu/CharsiuG2P) and OpenVoiceOS ByT5 G2P models.

---

[Docs index](index.md) · Previous: [Configuration reference](configuration.md) · Next: [Alphabet conversion model](alphabet_model.md)
