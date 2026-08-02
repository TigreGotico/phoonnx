"""
Inference engine registry.

New architectures register themselves here so that TTSVoice can
auto-select the right ONNX adapter at load time.

Usage::

    from phoonnx.engines import get_adapter, detect_engine

    # Auto-detect from config + session
    adapter = detect_engine(config=cfg, session=sess)

    # Or look up by name
    adapter = get_adapter("vits")
"""
import logging
from typing import Any, Dict, List, Optional, Type

import onnxruntime

from phoonnx.engines.base import BaseOnnxAdapter

LOG = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

_REGISTRY: Dict[str, Type[BaseOnnxAdapter]] = {}
_DETECT_ORDER: List[str] = []
_PRIORITIES: Dict[str, int] = {}


def register_engine(
    name: str,
    adapter_cls: Type[BaseOnnxAdapter],
    *,
    detect_priority: int = 100,
) -> None:
    """
    Register an inference adapter class under *name*.

    Parameters
    ----------
    name : str
        Short identifier (e.g. ``"vits"``, ``"optispeech"``).
    adapter_cls : Type[BaseOnnxAdapter]
        The adapter class (not an instance).
    detect_priority : int
        Lower = checked first during auto-detection.  Default 100.
    """
    _REGISTRY[name] = adapter_cls
    _PRIORITIES[name] = detect_priority
    _DETECT_ORDER.clear()
    _DETECT_ORDER.extend(
        sorted(_REGISTRY, key=lambda n: _PRIORITIES.get(n, 100))
    )
    LOG.debug("Registered inference engine %r (priority %d)", name, detect_priority)


def get_adapter(name: str) -> BaseOnnxAdapter:
    """
    Return a *new instance* of the adapter registered under *name*.

    Raises ``KeyError`` if the name is unknown.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown inference engine {name!r}. "
            f"Registered engines: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]()


def detect_engine(
    config: Optional[Dict[str, Any]] = None,
    session: Optional[onnxruntime.InferenceSession] = None,
) -> BaseOnnxAdapter:
    """
    Probe registered adapters and return the first one whose
    ``detect()`` returns ``True``.

    Falls back to the VITS adapter if nothing matches (since it is the
    most common engine in the phoonnx ecosystem).
    """
    # A voice that names its engine is authoritative. The heuristic probes below
    # exist to identify a voice that does *not* name one, and must never override
    # a config that does: several adapters share incidental traits (a vocoder
    # path, a language-id input), so a lower-priority adapter can otherwise claim
    # a voice that explicitly asked for another engine.
    named = (config or {}).get("engine")
    named = getattr(named, "value", named)
    if isinstance(named, str) and named.strip().lower() in _REGISTRY:
        name = named.strip().lower()
        LOG.debug("Using the engine named by the voice config: %s", name)
        return _REGISTRY[name]()

    for name in _DETECT_ORDER:
        cls = _REGISTRY[name]
        try:
            if cls.detect(config=config, session=session):
                LOG.debug("Auto-detected inference engine: %s", name)
                return cls()
        except Exception:
            LOG.debug("detect() failed for %s, skipping", name, exc_info=True)

    # Fallback
    if "vits" in _REGISTRY:
        LOG.warning("No engine matched — falling back to VITS adapter")
        return _REGISTRY["vits"]()

    raise RuntimeError(
        "Could not detect ONNX engine and no fallback available. "
        f"Registered engines: {list(_REGISTRY)}"
    )


def list_engines() -> List[str]:
    """Return the names of all registered inference engines."""
    return list(_REGISTRY)


# ------------------------------------------------------------------
# Built-in registrations
# ------------------------------------------------------------------

def _register_builtins() -> None:
    from phoonnx.engines.vits import VitsAdapter
    from phoonnx.engines.vits_streaming import VitsStreamingAdapter
    from phoonnx.engines.shami import ShamiAdapter
    from phoonnx.engines.matcha import MatchaAdapter
    from phoonnx.engines.optispeech import OptiSpeechAdapter
    from phoonnx.engines.glowtts import GlowTTSAdapter
    from phoonnx.engines.mixertts import MixerTTSAdapter
    from phoonnx.engines.fastpitch import FastPitchAdapter
    from phoonnx.engines.styletts2 import StyleTTS2Adapter
    from phoonnx.engines.yourtts import YourTTSAdapter
    from phoonnx.engines.zipvoice import ZipVoiceAdapter
    from phoonnx.engines.f5tts import F5TTSAdapter
    from phoonnx.engines.chatterbox import ChatterboxAdapter
    from phoonnx.engines.supertonic import SuperTonicAdapter
    from phoonnx.engines.neutts import NeuTTSAdapter

    # OptiSpeech shares VITS-like x/x_lengths/scales inputs with Matcha, but has
    # a distinctive metadata + wav/durations output signature — check it first.
    register_engine("optispeech", OptiSpeechAdapter, detect_priority=35)
    register_engine("shami", ShamiAdapter, detect_priority=45)
    # Streaming VITS must be probed *before* the plain VITS adapter: it only
    # matches split models (streaming:true + decoder_path), so it never steals a
    # normal single-graph voice, but a normal VITS voice must not steal a split one.
    register_engine("vits_streaming", VitsStreamingAdapter, detect_priority=48)
    register_engine("vits", VitsAdapter, detect_priority=50)
    register_engine("matcha", MatchaAdapter, detect_priority=40)
    register_engine("glowtts", GlowTTSAdapter, detect_priority=42)
    register_engine("mixertts", MixerTTSAdapter, detect_priority=36)
    register_engine("fastpitch", FastPitchAdapter, detect_priority=34)
    register_engine("styletts2", StyleTTS2Adapter, detect_priority=33)
    register_engine("yourtts", YourTTSAdapter, detect_priority=32)
    register_engine("zipvoice", ZipVoiceAdapter, detect_priority=31)
    register_engine("f5tts", F5TTSAdapter, detect_priority=30)
    register_engine("chatterbox", ChatterboxAdapter, detect_priority=29)
    register_engine("supertonic", SuperTonicAdapter, detect_priority=28)
    register_engine("neutts", NeuTTSAdapter, detect_priority=27)


_register_builtins()
