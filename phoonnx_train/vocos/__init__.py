"""Vocos vocoder training for phoonnx.

Trains a Vocos (Siuzdak, 2023 — "Vocos: Closing the gap between
time-domain and Fourier-based neural vocoders for high-quality audio
synthesis") mel→waveform vocoder that pairs with phoonnx mel-emitting
acoustic models (Matcha-TTS and friends).

The package is split so everything that does not need torch stays
importable without it:

- :mod:`phoonnx_train.vocos.data` — torch-free: mel configuration
  constants, audio file discovery, crop math, warm-start source parsing.
- :mod:`phoonnx_train.vocos.dataset` — torch ``Dataset`` of random
  fixed-length waveform crops.
- :mod:`phoonnx_train.vocos.models` — GAN discriminators and losses.
- :mod:`phoonnx_train.vocos.lightning` — the LightningModule.

This ``__init__`` intentionally imports nothing heavy.
"""
