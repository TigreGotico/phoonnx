# Installation

## Requirements

- Python 3.10+
- `onnxruntime` (CPU) or `onnxruntime-gpu` (CUDA)

## Basic Install

```bash
pip install phoonnx
```

This installs the core inference runtime with the default espeak-ng phonemizer backend.

## Optional Extras

phoonnx uses optional dependencies for specific phonemizer backends and training. Install only what you need.

### Training

```bash
pip install phoonnx[train]
```

Includes PyTorch Lightning, Cython (for monotonic alignment), and all training utilities.

### Language-Specific Phonemizers

Many phonemizer backends have optional dependencies:

```bash
# English (g2p-en)
pip install phoonnx[en]

# Japanese
pip install phoonnx[ja]

# Chinese
pip install phoonnx[zh]

# Korean
pip install phoonnx[ko]

# Arabic (mantoq)
pip install phoonnx[ar]
```

### OVOS Plugin

```bash
pip install phoonnx[ovos]
```

## System Dependencies

### espeak-ng

Many voices use the `espeak` phonemizer backend. Install espeak-ng from your system package manager:

```bash
# Debian/Ubuntu
sudo apt-get install espeak-ng

# macOS
brew install espeak

# Arch Linux
sudo pacman -S espeak-ng
```

### Cotovia (Galician)

For Galician (`gl`) voices using the `cotovia` phonemizer, install the Cotovia TTS binary separately. phoonnx will attempt to locate it in `PATH`, common system paths, or the bundled binary location.

## CUDA Inference

To use GPU inference, install the GPU-enabled ONNX runtime:

```bash
pip install onnxruntime-gpu
```

Then pass `use_cuda=True` when loading a voice:

```python
voice = TTSVoice.load("model.onnx", "model.json", use_cuda=True)
```

## Installing from Source

```bash
git clone https://github.com/TigreGotico/phoonnx
cd phoonnx
pip install -e .
# or with training extras:
pip install -e .[train]
```
