"""phoonnx: multilingual text to speech with ONNX models.

The names below resolve on first access, so ``import phoonnx`` does not load
the voice runtime and its engines.
"""
from importlib import import_module

_HOMES = {
    "TTSVoice": "phoonnx.voice",
    "AudioChunk": "phoonnx.voice",
    "VoiceConfig": "phoonnx.config",
    "SynthesisConfig": "phoonnx.config",
    "TTSModelManager": "phoonnx.model_manager",
    "Stages": "scriptconv.frontend",
    "FrontendConfig": "scriptconv.frontend",
}

__all__ = list(_HOMES)


def __getattr__(name: str):
    if name not in _HOMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_HOMES[name]), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
