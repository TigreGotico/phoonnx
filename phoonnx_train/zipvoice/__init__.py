"""Vendored ZipVoice (Apache-2.0, https://github.com/k2-fsa/ZipVoice).

ZipVoice (Zhu et al., "ZipVoice: Fast and high-quality zero-shot
text-to-speech with flow matching", arXiv:2506.13053) is a flow-matching
TTS with a Zipformer text-audio backbone and in-context (infilling)
zero-shot cloning. The upstream training code is a
``lhotse``/``piper_phonemize`` based recipe distributed only via git clone
(the ``zipvoice`` PyPI package is an unrelated empty placeholder), so the
model implementation is vendored here as self-contained modules with
imports made package-relative and the lhotse plumbing removed:

- :mod:`.model` — the ZipVoice model (TTSZipformer text encoder +
  flow-matching decoder)
- :mod:`.zipformer` / :mod:`.scaling` / :mod:`.solver` — the Zipformer
  stack (Yao et al., arXiv:2310.11230) and Euler ODE solver
- :mod:`.optim` / :mod:`.lr_scheduler` — ScaledAdam + the Eden schedule
- :mod:`.scaling_converter` — scaled-module → export-friendly conversion
  for ONNX export
- :mod:`.feature` — Vocos log-mel features (24 kHz / 100 bins / hop 256)
- :mod:`.common` — model-side helpers (``condition_time_mask``,
  ``pad_labels``, ``get_tokens_index``, ...)
- :mod:`.lightning` — the pytorch_lightning training loop used by
  ``phoonnx_train/engines/zipvoice.py``
- :mod:`.cfm` / :mod:`.dataset` — standalone, architecture-agnostic CFM
  objective and in-context pair construction, reusable against any
  backbone

This package deliberately imports nothing at the top level: the engine
registry must stay importable in torch-free environments, and every module
here needs torch.
"""
