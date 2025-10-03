[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TigreGotico/phoonnx)

# Phoonnx

A Python library for multilingual phonemization and Text-to-Speech (TTS) using ONNX models.

## Introduction

`phoonnx` is a comprehensive toolkit for performing high-quality, efficient TTS inference using ONNX-compatible models.
It provides a flexible framework for text normalization, phonemization, and speech synthesis, with built-in support for
multiple languages and phonemic alphabets. The library is also designed to work with models trained using
`phoonnx_train`, including utilities for dataset preprocessing and exporting models to the ONNX format.

## Features

- **Efficient Inference:** Leverages `onnxruntime` for fast and efficient TTS synthesis.
- **Multilingual Support:** Supports a wide range of languages and phonemic alphabets, including IPA, ARPA, Hangul (
  Korean), and Pinyin (Chinese).
- **Multiple Phonemizers:** Integrates with various phonemizers like eSpeak, Gruut, and Epitran to convert text to
  phonemes.
- **Advanced Text Normalization:** Includes robust utilities for expanding contractions and pronouncing numbers and
  dates.
- **Dataset Preprocessing:** Provides a command-line tool to prepare LJSpeech-style datasets for training.
- **Model Export:** A script is included to convert trained models into the ONNX format, ready for deployment.

## Installation

As `phoonnx` is available on PyPI, you can install it using pip.

```bash
pip install phoonnx
```

## Usage

### Synthesizing Speech

The main component for inference is the `TTSVoice` class. You can load a model and synthesize speech from text as
follows:

```python
import wave

from phoonnx.config import VoiceConfig, SynthesisConfig
from phoonnx.voice import TTSVoice

# Load a pre-trained ONNX model and its configuration
voice = TTSVoice.load("model.onnx", "config.json")

# Configure the synthesis parameters (optional)
synthesis_config = SynthesisConfig(
    noise_scale=0.667,
    length_scale=1.0,
    noise_w_scale=0.8,
    enable_phonetic_spellings=True, # apply pronunciation fixes, see "locale" folder in this repo
    add_diacritics=False  # for arabic and hebrew
)

# Synthesize audio from text
text = "Hello, this is a test of the phoonnx library."
slug = f"phoonnx_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
with wave.open(f"{slug}.wav", "wb") as wav_file:
    voice.synthesize_wav(text, wav_file, synthesis_config)

```

### Training

See the dedicated [training.md](/TRAINING.md)

## Supported Phonemizers

`phoonnx` leverages several external Grapheme-to-Phoneme (G2P) and text-processing libraries to provide flexible and
high-quality phonemization across many languages.

You should prefer phonemizers trained on full sentences vs individual words if available

The core phonemizer classes are summarized in the table below, listing the supported languages, the source library they
wrap, and the output alphabets they can generate.

***

| Language(s)         | Phonemizer Class       | Source/Library                                                                                                     | Output Alphabets             | 
|:--------------------|:-----------------------|:-------------------------------------------------------------------------------------------------------------------|:-----------------------------|
| **Multilingual**    | `ByT5Phonemizer`       | [OpenVoiceOS ByT5](https://huggingface.co/collections/OpenVoiceOS/g2p-models-6886a8d612825c3fe65befa0) ONNX Models | IPA                          | High-quality, model-based G2P for an extensive list of languages.                                                                           |
| **Multilingual**    | `CharsiuPhonemizer`    | [Charsiu](https://github.com/lingjzhu/CharsiuG2P) ByT5 ONNX Model                                                  | IPA                          | Very extensive multilingual support, including many regional dialects and variants (e.g., `eng-uk`, `spa-me`, `zho-s`).                     |
| **Multilingual**    | `EspeakPhonemizer`     | `espeak-ng` command-line tool                                                                                      | IPA                          | Broad language coverage, relying on the widely-used `espeak-ng` engine.                                                                     |
| **Multilingual**    | `GruutPhonemizer`      | [gruut](https://github.com/rhasspy/gruut)                                                                          | IPA                          | A tokenizer, text cleaner, and IPA phonemizer for several human languages that supports SSML.                                               |
| **Multilingual**    | `MisakiPhonemizer`     | [misaki](https://github.com/hexgrad/misaki)                                                                        | IPA                          | Misaki is a G2P engine designed for Kokoro models.                                                                                          |
| **Multilingual**    | `TransphonePhonemizer` | [transphone](https://github.com/xinjli/transphone)                                                                 | IPA                          | It provides approximated phoneme tokenizers and G2P model for 7546 languages registered in the Glottolog database.                          |
| **Multilingual**    | `EpitranPhonemizer`    | [epitran](https://github.com/dmort27/epitran)                                                                      | IPA                          | A tool for transcribing orthographic text as IPA                                                                                            |
| **Arabic (ar)**     | `MantoqPhonemizer`     | [mantoq](https://github.com/mush42/mantoq)                                                                         | BUCKWALTER, IPA              | Translates unvoweled Arabic to phonemes, with optional conversion to IPA.                                                                   |
| **Chinese (zh)**    | `JiebaPhonemizer`      | [jieba](https://github.com/fxsjy/jieba)                                                                            | HANZI                        | Segments Chinese text into words with spaces; useful for pre-processing.                                                                    |
| **Chinese (zh)**    | `G2pMPhonemizer`       | [g2pC](https://github.com/Kyubyong/g2pC)                                                                           | IPA, Pinyin                  | CRF-based Grapheme-to-Phoneme converter                                                                                                     |
| **Chinese (zh)**    | `G2pMPhonemizer`       | [g2pm](https://github.com/kakaobrain/g2pm)                                                                         | IPA, Pinyin                  | A Neural Grapheme-to-Phoneme Conversion Package for Mandarin Chinese                                                                        |
| **Chinese (zh)**    | `XpinyinPhonemizer`    | [xpinyin](https://github.com/lxneng/xpinyin)                                                                       | IPA, Pinyin                  | basic pinyin generator with optional tone marks                                                                                             |
| **Chinese (zh)**    | `PypinyinPhonemizer`   | [pypinyin](https://github.com/rainforest32/pypinyin)                                                               | IPA, Pinyin                  | comprehensive and accurate pinyin library                                                                                                   |
| **English (en)**    | `G2PEnPhonemizer`      | [g2pE](https://github.com/Kyubyong/g2p)                                                                            | IPA                          | A deep learning seq2seq framework based on TensorFlow                                                                                       |
| **English (en)**    | `OpenPhonemizer`       | [OpenPhonemizer](https://github.com/NeuralVox/OpenPhonemizer)                                                      | IPA                          | IPA Phonemizer powered by deep learning. This Phonemizer attempts to replicate the espeak Phonemizer while remaining permissively-licensed. |
| **English (en)**    | `DeepPhonemizer`       | [DeepPhonemizer](https://github.com/spring-media/DeepPhonemizer)                                                   | IPA / ARPA                   | Uses pre-trained deep learning models for English.                                                                                          |
| **Galician (gl)**   | `CotoviaPhonemizer`    | [cotovia](https://github.com/TigreGotico/cotovia-mirror)                                                           | IPA, Native Cotovia Phonemes | Relies on the `cotovia`executable for Galician phonemization.                                                                               |
| **Hebrew (he)**     | `PhonikudPhonemizer`   | [phonikud](https://github.com/thewh1teagle/phonikud)                                                               | IPA                          | Converts Hebrew text to IPA phonemes.                                                                                                       |
| **Japanese (ja)**   | `OpenJTaklPhonemizer`  | [pyopenjtalk](https://github.com/r9y9/pyopenjtalk)                                                                 | HEPBURN, KANA                | High-quality Japanese G2P.                                                                                                                  |
| **Japanese (ja)**   | `CutletPhonemizer`     | [cutlet](https://github.com/polm/cutlet)                                                                           | HEPBURN, KUNREI, NIHON       | Provides various Romanization standards.                                                                                                    |
| **Japanese (ja)**   | `PyKakasiPhonemizer`   | [pykakasi](https://codeberg.org/miurahr/pykakasi)                                                                  | HEPBURN, KANA, HIRA          | Romanization and Kana conversion.                                                                                                           |
| **Korean (ko)**     | `G2PKPhonemizer`       | [g2pK](https://github.com/Kyubyong/g2pK)                                                                           | IPA, HANGUL                  | Provides G2P for Korean, with optional IPA conversion.                                                                                      |
| **Korean (ko)**     | `KoG2PPhonemizer`      | [KoG2P](https://github.com/scarletcho/KoG2P)                                                                       | IPA, HANGUL                  | Provides G2P for Korean, with optional IPA conversion.                                                                                      |
| **Persian (fa)**    | `PersianPhonemizer`    | [persian_phonemizer](https://github.com/de-mh/persian_phonemizer)                                                  | ERAAB, IPA                   | Supports both standard IPA and the native ERAAB (diacritical) representations.                                                              |
| **Vietnamese (vi)** | `VIPhonemePhonemizer`  | [Viphoneme](https://github.com/v-nhandt21/Viphoneme)                                                               | IPA                          | Uses the `viphoneme` library for Vietnamese G2P.                                                                                            |

### Credits

Phoonnx is built in the shoulders of giants

- [jaywalnut310/vits](https://github.com/jaywalnut310/vits) - the original VITS implementation, the back-bone architecture of phoonnx models
- [MycroftAI/mimic3](https://github.com/MycroftAI/mimic3) and [rhasspy/piper](https://github.com/rhasspy/piper) - for inspiration and reference implementation of a phonemizer for pre-processing inputs

Individual languages greatly benefit from domain-specific knowledge, for convenience phoonnx also bundles code from

- [uvigo/cotovia](https://github.com/TigreGotico/cotovia-mirror) for galician phonemization (pre-compiled binaries bundled)
- [mush42/mantoq](https://github.com/mush42/mantoq) for arabic phonemization
- [mush42/libtashkeel](https://github.com/mush42/libtashkeel) for arabic diacritics
- [scarletcho/KoG2P](https://github.com/scarletcho/KoG2P) for korean phonemization
- [stannam/hangul_to_ipa](https://github.com/stannam/hangul_to_ipa) a converter from Hangul to IPA
- [chorusai/arpa2ipa](https://github.com/chorusai/arpa2ipa) a converter from Arpabet to IPA
- [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech/blob/8097a56be811a540f4f62a95a9094296c374351a/paddlespeech/t2s/frontend/zh_normalization/) for chinese number verbalization

