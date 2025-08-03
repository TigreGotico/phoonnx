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

### Preprocessing Datasets

Use the `preprocess.py` script to prepare your audio and text data for training:

```bash
python phoonnx_train/preprocess.py --dataset-dir /path/to/my/dataset --output-dir /path/to/output
```

### Exporting Models

After training, you can export a PyTorch Lightning checkpoint (`.ckpt`) to an ONNX model:

```bash
python phoonnx_train/export_onnx.py /path/to/my/model.ckpt output.onnx
```

This script will convert the model to an ONNX file with an `opset_version` of 15.
