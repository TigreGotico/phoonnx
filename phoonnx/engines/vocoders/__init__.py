"""
Vocoder registry.

Mel→waveform vocoders register themselves here so acoustic-model adapters
(Matcha-TTS, OptiSpeech, …) can pair with any compatible vocoder without
hard-coding its output layout.  Mirrors :mod:`phoonnx.engines`.

Usage::

    from phoonnx.engines.vocoders import build_vocoder

    vocoder = build_vocoder(
        model_path="alvocat-vocos-22khz.onnx",
        vocoder_type="vocos",            # or None to auto-detect
        config={"n_fft": 1024, "hop_length": 256},
    )
    audio = vocoder.mel_to_audio(mel, denoise=True)
"""
import logging
from typing import Any, Dict, List, Optional, Type

import onnxruntime

from phoonnx.engines.vocoders.base import BaseVocoder, load_onnx_session

LOG = logging.getLogger(__name__)

_REGISTRY: Dict[str, Type[BaseVocoder]] = {}
_DETECT_ORDER: List[str] = []
_PRIORITIES: Dict[str, int] = {}


def register_vocoder(
    name: str,
    vocoder_cls: Type[BaseVocoder],
    *,
    detect_priority: int = 100,
) -> None:
    """Register a vocoder class under *name* (lower priority = checked first)."""
    _REGISTRY[name] = vocoder_cls
    _PRIORITIES[name] = detect_priority
    _DETECT_ORDER.clear()
    _DETECT_ORDER.extend(sorted(_REGISTRY, key=lambda n: _PRIORITIES.get(n, 100)))
    LOG.debug("Registered vocoder %r (priority %d)", name, detect_priority)


def get_vocoder_cls(name: str) -> Type[BaseVocoder]:
    """Return the vocoder class registered under *name* (``KeyError`` if unknown)."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown vocoder {name!r}. Registered vocoders: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_vocoders() -> List[str]:
    """Return the names of all registered vocoders."""
    return list(_REGISTRY)


def build_vocoder(
    model_path: Optional[str] = None,
    vocoder_type: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    session: Optional[onnxruntime.InferenceSession] = None,
) -> BaseVocoder:
    """
    Construct a vocoder.

    If *vocoder_type* (or ``config['vocoder_type']``) names a registered
    vocoder, it is used directly.  Otherwise the ONNX model is probed and
    the first matching vocoder wins, falling back to Vocos.
    """
    config = config or {}
    vtype = vocoder_type or config.get("vocoder_type")
    if vtype and vtype in _REGISTRY:
        return _REGISTRY[vtype](model_path=model_path, config=config, session=session)
    if vtype:
        LOG.warning("Unknown vocoder_type %r — falling back to auto-detection", vtype)

    # Auto-detection needs the session to inspect the output layout.
    if session is None and model_path:
        session = load_onnx_session(model_path)

    for name in _DETECT_ORDER:
        try:
            if _REGISTRY[name].detect(session=session, config=config):
                LOG.debug("Auto-detected vocoder: %s", name)
                return _REGISTRY[name](model_path=model_path, config=config, session=session)
        except Exception:
            LOG.debug("detect() failed for vocoder %s, skipping", name, exc_info=True)

    LOG.warning("No vocoder matched — falling back to Vocos")
    return _REGISTRY["vocos"](model_path=model_path, config=config, session=session)


def _register_builtins() -> None:
    from phoonnx.engines.vocoders.vocos import VocosVocoder
    from phoonnx.engines.vocoders.raw import (
        RawWaveformVocoder,
        WavenextVocoder,
        HiFiGANVocoder,
    )

    # Explicit-type aliases (selected when vocoder_type is set).
    register_vocoder("wavenext", WavenextVocoder, detect_priority=30)
    register_vocoder("hifigan", HiFiGANVocoder, detect_priority=31)
    # Generic detection: Vocos (3 outputs) before raw (1 output).
    register_vocoder("vocos", VocosVocoder, detect_priority=40)
    register_vocoder("raw", RawWaveformVocoder, detect_priority=50)


_register_builtins()
