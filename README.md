[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TigreGotico/phoonnx)

# Phoonnx

A Python library for multilingual phonemization and Text-to-Speech (TTS) using ONNX models.

## Introduction

`phoonnx` is a comprehensive toolkit for performing high-quality, efficient TTS inference using ONNX-compatible models. It provides a flexible framework for text normalization, phonemization, and speech synthesis, with built-in support for multiple languages and phonemic alphabets. The library is also designed to work with models trained using `phoonnx_train`, including utilities for dataset preprocessing and exporting models to the ONNX format.

## Features

  - **Efficient Inference:** Leverages `onnxruntime` for fast and efficient TTS synthesis.
  - **Multilingual Support:** Supports a wide range of languages and phonemic alphabets, including IPA, ARPA, Hangul (Korean), and Pinyin (Chinese).
  - **Multiple Phonemizers:** Integrates with various phonemizers like eSpeak, Gruut, and Epitran to convert text to phonemes.
  - **Advanced Text Normalization:** Includes robust utilities for expanding contractions and pronouncing numbers and dates.
  - **Dataset Preprocessing:** Provides a command-line tool to prepare LJSpeech-style datasets for training.
  - **Model Export:** A script is included to convert trained models into the ONNX format, ready for deployment.

## Installation

As `phoonnx` is available on PyPI, you can install it using pip.

```bash
pip install phoonnx
```

## Usage

### Synthesizing Speech

The main component for inference is the `TTSVoice` class. You can load a model and synthesize speech from text as follows:

```python
from phoonnx.config import VoiceConfig, SynthesisConfig
from phoonnx.voice import TTSVoice

# Load a pre-trained ONNX model and its configuration
# Assume 'model.onnx' and 'config.json' are available
voice = TTSVoice.load("model.onnx", "config.json")

# Configure the synthesis parameters (optional)
synthesis_config = SynthesisConfig(
    noise_scale=0.667,
    length_scale=1.0,
    noise_w_scale=0.8
)

# Synthesize audio from text
text = "Hello, this is a test of the phoonnx library."
audio_chunk = voice.synthesize(text, synthesis_config=synthesis_config)

# Save the audio to a WAV file
audio_chunk.write_wav("output.wav")
```

### Training

See the dedicated [training.md](/TRAINING.md)

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

phoonnx can also optionally use the following external phonemizers if explicitly installed:

- [Kyubyong/g2pK](https://github.com/Kyubyong/g2pK) for Korean phonemizer
- TODO list all of them
