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
    from phoonnx.engines.matcha import MatchaAdapter
    from phoonnx.engines.optispeech import OptiSpeechAdapter
    from phoonnx.engines.glowtts import GlowTTSAdapter
    from phoonnx.engines.mixertts import MixerTTSAdapter

    # OptiSpeech shares VITS-like x/x_lengths/scales inputs with Matcha, but has
    # a distinctive metadata + wav/durations output signature — check it first.
    register_engine("optispeech", OptiSpeechAdapter, detect_priority=35)
    register_engine("vits", VitsAdapter, detect_priority=50)
    register_engine("matcha", MatchaAdapter, detect_priority=40)
    register_engine("glowtts", GlowTTSAdapter, detect_priority=42)
    register_engine("mixertts", MixerTTSAdapter, detect_priority=36)


_register_builtins()
