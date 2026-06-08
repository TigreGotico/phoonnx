"""Speaker-encoder registry — mirrors phoonnx.engines.vocoders.

Cloning-capable adapters (e.g. YourTTS) condition on a d-vector produced from
reference audio. Encoders register by name so a voice names its encoder type.
"""
from typing import Any, Dict, Optional, Type

from phoonnx.util import LOG
from phoonnx.engines.speaker_encoders.base import BaseSpeakerEncoder

_REGISTRY: Dict[str, Type[BaseSpeakerEncoder]] = {}


def register_speaker_encoder(name: str, cls: Type[BaseSpeakerEncoder]) -> None:
    _REGISTRY[name] = cls


def get_speaker_encoder(name: str) -> Type[BaseSpeakerEncoder]:
    return _REGISTRY[name]


def list_speaker_encoders():
    return sorted(_REGISTRY)


def build_speaker_encoder(model_path: str, encoder_type: Optional[str] = None,
                          config: Optional[Dict[str, Any]] = None) -> BaseSpeakerEncoder:
    config = config or {}
    etype = encoder_type or config.get("speaker_encoder_type") or "coqui_resnet"
    if etype not in _REGISTRY:
        LOG.warning("Unknown speaker_encoder_type %r — using coqui_resnet", etype)
        etype = "coqui_resnet"
    return _REGISTRY[etype](model_path, config)


def _register_builtins() -> None:
    from phoonnx.engines.speaker_encoders.coqui_resnet import CoquiResNetSpeakerEncoder
    from phoonnx.engines.speaker_encoders.styletts2_style import StyleTTS2StyleEncoder
    register_speaker_encoder("coqui_resnet", CoquiResNetSpeakerEncoder)
    register_speaker_encoder("styletts2_style", StyleTTS2StyleEncoder)


_register_builtins()
