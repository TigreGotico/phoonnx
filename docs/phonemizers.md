# Phonemizers

<!-- agent-hint
covers: every PhonemeType and Alphabet value, and how a voice selects and runs its phonemizer
source: phoonnx/config.py:PhonemeType, phoonnx/config.py:Alphabet, phoonnx/compat.py:build_voice_config, phoonnx/voice.py:TTSVoice.frontend, tests/test_voice_config_to_dict.py
generated: no
verify: .venv/bin/python -m doctest -o ELLIPSIS -o NORMALIZE_WHITESPACE docs/phonemizers.md && .venv/bin/python -W ignore -c "import re,sys; from phoonnx.config import PhonemeType,Alphabet; p=open('docs/phonemizers.md').read(); n=chr(10); q=chr(96); rows=lambda a,b: set(re.findall('^[|] '+q+'([a-z0-9_-]+)'+q+' [|]', p.split(n+a,1)[1].split(n+b,1)[0], re.M)); ok=[rows('## PhonemeType Ref','## Alphabet Ref')=={x.value for x in PhonemeType}, rows('## Alphabet Ref','## Selecting')=={x.value for x in Alphabet}]; print(ok); sys.exit(0 if all(ok) else 1)" && .venv/bin/python -m pytest -q tests/test_voice_config_to_dict.py
see: galician.md, architecture.md, installation.md
-->

This page shows how a voice selects and runs its phonemizer, and lists each phonemizer backend and output alphabet. It is for voice authors and advanced users.

The `PhonemeType` enum selects the phonemizer, and the `Alphabet` enum sets its output. phoonnx re-exports both from scriptconv in `phoonnx.config`.

## PhonemeType Reference

Each value below is a member of `PhonemeType`. A backend for one language needs the matching [install extra](installation.md#language-extras).

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
| `nemo_ipa` | NVIDIA NeMo IPA G2P, as the Magpie `english_phoneme` sub-tokenizer reads it | American English |
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
| `dizione` | American English: CMUdict lexicon, homograph rules and its own text rules. Gives the tiny-voca tokens, each with a stress value | en-US |
| `arbtok` | Dialect-aware Arabic on undiacritized text, on the o2i lattice. `phonemizer_model` sets the register | Arabic |
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
| `leokani` | Leokani: lexicon, ONNX letter-to-sound and context models. Gives the Kokoro (misaki) codes | en-US |
| `kolokwa` | kolokwa-g2p model | Liberian English (lir) |
| `espeak_lt_accent` | eSpeak IPA with the three Lithuanian pitch accents | Lithuanian |
| `hebrew_ipa` | IPA with stress, from text that carries niqqud | Hebrew |
| `openjtalk_prosody` | Open JTalk IPA with accent rise, fall and phrase marks | Japanese |
| `tltk` | TLTK IPA with a tone digit after each syllable | Thai |

## Alphabet Reference

Each value below is a member of `Alphabet`, the symbol set of the phonemizer output:

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
| `bopomofo` | Zhuyin / Bopomofo (Chinese), as misaki writes it |
| `hanzi` | Chinese characters |
| `eraab` | ERAAB (Persian) |
| `cotovia` | Cotovia phoneme set (Galician) |
| `buckwalter` | Buckwalter transliteration (Arabic) |
| `graphemes` | Raw text characters |
| `vosk` | vosk-tts phoneme inventory (Russian) |
| `cangjie` | Cangjie input-method decomposition (Chinese) |
| `halabi` | Halabi Arabic-Phonetiser phoneme notation (Arabic) |
| `africa_g2p` | africa-g2p native-orthography phoneme units (African languages) |
| `misaki` | misaki and Kokoro phoneme codes: IPA, with `A`, `I`, `O`, `Q`, `W` and `Y` for diphthongs (English) |

## Selecting a Phonemizer

The `phoneme_type` and `alphabet` keys of the voice config select the phonemizer, and phoonnx loads it with the voice. The voice reads text through `voice.frontend`, a scriptconv `TextFrontend` over that phonemizer. `phonemize` gives the phonemes that the voice synthesizes, one list per sentence. This example builds a Portuguese voice config with the `unicode` phonemizer. The frontend needs no ONNX session:

```python
>>> from phoonnx.config import VoiceConfig
>>> from phoonnx.voice import TTSVoice
>>> config = VoiceConfig.from_dict({
...     "phoonnx_version": "1.0", "lang_code": "pt-PT", "phoneme_type": "unicode",
...     "alphabet": "unicode", "keep_punctuation": True, "phoneme_id_map": {"_": [0]}})
>>> voice = TTSVoice(session=None, config=config)
>>> voice.frontend.phonemize("Olá mundo. Como estás?")
[['O', 'l', 'a', '́', ' ', 'm', 'u', 'n', 'd', 'o', '.'], ['C', 'o', 'm', 'o', ' ', 'e', 's', 't', 'a', '́', 's', '?']]

```

The `unicode` phonemizer splits each accented letter into the letter and its combining mark. With `keep_punctuation` set to `False`, the default, the `.` and the `?` are not in the output.

To use another phonemizer with a voice, give it to the load:

```python
from phoonnx.voice import TTSVoice

voice = TTSVoice.load("model.onnx", "config.json", phoneme_type_str="espeak", alphabet_str="ipa")
```

The phonemizer must give the symbols of the vocabulary of the voice, or the model reads wrong ids.

To phonemize without a voice, build a `TextFrontend`. The eSpeak backend needs the `espeak` extra:

```python
>>> from scriptconv.frontend import FrontendConfig, TextFrontend
>>> from scriptconv.phonemizers import Phonemizer
>>> frontend = TextFrontend(FrontendConfig("en-US", Phonemizer.ESPEAK))
>>> frontend.phonemize("Hello world")
[['h', 'ə', 'l', 'ˈ', 'o', 'ʊ', ' ', 'w', 'ˈ', 'ɜ', 'ː', 'l', 'd']]

```

The `phonemizer_model` key of a voice selects a variant, for the backends that have variants. `FrontendConfig.model` takes the same value. Examples:

| Backend | `phonemizer_model` | Variant |
|---|---|---|
| `ahotts` | `classic`, `modern`, `northern` | The AhoTTS engine version or dialect. `modern` is the default |
| `cotovia` | `stress` | The stress-marked notation of the HiTZ Galician model |
| `arbtok` | `iʿrab` | The register with full case endings |
| `byt5`, `charsiu`, `deepphonemizer` | a model id or path | The neural G2P model |

`phoonnx.config.get_phonemizer` and `phoonnx.config.get_conversion` are deprecated. They warn, and the warning names the release that removes them.

`frontend.backend` gives the backend of a `TextFrontend`. `phonemize_string(text, lang)` gives the phonemes of one chunk as a string:

```python
>>> frontend.backend.phonemize_string("Hello world", "en-US")
'həlˈoʊ wˈɜːld'

```

## Special Language Notes

### Diacritics, stress and sense marks

The diacritize stage adds marks before phonemization. It restores vowels for Arabic and Hebrew, marks stress for Russian, Ukrainian and the other languages of `stressonnx`, and marks homograph readings for European Portuguese. A voice turns it on in its config JSON, under `inference`:

```json
"inference": {"add_diacritics": true}
```

phoonnx does not read the key at the top level of the config. When the config has no `inference` key, the stage is on for Arabic language tags and off for each other language. The Arabic tags are `ar` and the Arabic varieties in Arabic script, such as `apc` and `arz`:

```python
>>> def diacritics(lang, **keys):
...     return VoiceConfig.from_dict({"phoonnx_version": "1.0", "lang_code": lang, "phoneme_type": "unicode",
...                                   "alphabet": "unicode", "phoneme_id_map": {"_": [0]}, **keys}).add_diacritics
>>> [diacritics(lang) for lang in ("ar", "apc", "arz", "he", "pt-PT")]
[True, True, True, False, False]
>>> diacritics("ar", add_diacritics=False), diacritics("ar", inference={"add_diacritics": False})
(True, False)

```

A call can override the voice with `SynthesisConfig`:

```python
from phoonnx.config import SynthesisConfig
from scriptconv.frontend import Stages
syn_config = SynthesisConfig(stages=Stages(diacritize=True))
```

Arabic diacritization uses [`text2tashkeel`](https://pypi.org/project/text2tashkeel/), which the [`ar` extra](installation.md#language-extras) installs. The default diacritizer model is `rawi-ensemble`. To use another model, set the `diacritizer_model` field of the voice config, or `stages.diacritizer_model` in the `SynthesisConfig` of one call. Arabic diacritics without `text2tashkeel` raise an `ImportError` that names the package. Hebrew uses Phonikud.

### Galician (Cotovia)

The `cotovia` phonemizer runs [pycotovia](https://github.com/TigreGotico/pycotovia), a Python port of the Cotovia G2P engine, through scriptconv. The scriptconv `gl` extra installs it. The alphabet can be native Cotovia or IPA. Read [Galician](galician.md).

### Chinese

Many backends read Chinese. `pypinyin` and `xpinyin` give Pinyin. `g2pc` and `g2pm` give IPA. `jieba` is a word segmenter, not a phonemizer.

## BasePhonemizer

Each phonemizer is a subclass of the scriptconv `BasePhonemizer`, which has these methods:

| Method | Gives |
|---|---|
| `phonemize(text, lang)` | One list of phonemes for each sentence |
| `phonemize_to_list(text, lang)` | One flat list of phonemes |
| `phonemize_string(text, lang)` | The phonemes of one chunk, as a string |
| `phonemize_segments(text, lang)` | The segments, with the side streams of the backend, such as stress |
| `chunk_text(text)` | The sentence chunks of the text, with their punctuation |

Diacritization is not a phonemizer method. It is a separate step, `scriptconv.diacritics.diacritize(text, lang)`, which runs before the phonemizer when a voice sets `add_diacritics` or a call sets `stages.diacritize`. scriptconv owns the diacritizer backends. It downloads the Hebrew phonikud model on first use.

## G2P for OVOS

The phonemizers on this page come from [scriptconv](https://github.com/TigreGotico/scriptconv). For grapheme-to-phoneme conversion in OpenVoiceOS without speech synthesis, use [ovos-scriptconv-g2p-plugin](https://github.com/TigreGotico/ovos-scriptconv-g2p-plugin), an OPM `opm.g2p` plugin on scriptconv. The [docs/ovos.md](https://github.com/TigreGotico/scriptconv/blob/dev/docs/ovos.md) page of scriptconv gives the install and configuration steps.

## Attribution

phoonnx uses G2P work from these projects, through scriptconv:

- [cotovia](https://github.com/TigreGotico/cotovia-mirror): Galician phonemization, through its Python port [pycotovia](https://github.com/TigreGotico/pycotovia)
- [mantoq](https://github.com/mush42/mantoq): Arabic phonemization
- [text2tashkeel](https://pypi.org/project/text2tashkeel/): Arabic diacritization
- [hams-levantine-tts](https://github.com/Al-aminI/hams-levantine-tts): the Shami (Levantine Arabic / English) front-end
- [KoG2P](https://github.com/scarletcho/KoG2P) and [hangul_to_ipa](https://github.com/stannam/hangul_to_ipa): Korean phonemization and Hangul to IPA
- [arpa2ipa](https://github.com/chorusai/arpa2ipa): ARPAbet to IPA conversion

Multilingual and neural backends include eSpeak, [gruut](https://github.com/rhasspy/gruut),
[epitran](https://github.com/dmort27/epitran), [misaki](https://github.com/hexgrad/misaki),
[transphone](https://github.com/xinjli/transphone),
[Charsiu](https://github.com/lingjzhu/CharsiuG2P) and OpenVoiceOS ByT5 G2P models.

---

[Docs index](index.md) · Previous: [Configuration reference](configuration.md) · Next: [Alphabet conversion model](alphabet_model.md)
