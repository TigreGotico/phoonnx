"""
Training engine registry.

New architectures register here so that the shared CLI tools
(``train.py``, ``export_onnx.py``, ``preprocess.py``) can work with
any architecture.

Usage::

    from phoonnx_train.engines import get_engine, list_engines

    engine = get_engine("vits")
    model = engine.create_model(config, dataset_paths)
"""
import logging
from typing import Any, Dict, List, Type

from phoonnx_train.engines.base import BaseTrainingEngine

LOG = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

_REGISTRY: Dict[str, Type[BaseTrainingEngine]] = {}


def register_engine(
    name: str,
    engine_cls: Type[BaseTrainingEngine],
) -> None:
    """
    Register a training engine class under *name*.

    Once registered, the engine can be selected via the ``--engine``
    CLI flag in ``train.py`` and ``export_onnx.py``.

    Parameters
    ----------
    name : str
        Short identifier (e.g. ``"vits"``, ``"optispeech"``).
    engine_cls : Type[BaseTrainingEngine]
        The engine class (not an instance).
    """
    _REGISTRY[name] = engine_cls
    LOG.debug("Registered training engine %r", name)


def get_engine(name: str) -> BaseTrainingEngine:
    """
    Return a *new instance* of the engine registered under *name*.

    Parameters
    ----------
    name : str
        Engine identifier previously passed to ``register_engine``.

    Returns
    -------
    BaseTrainingEngine
        A fresh engine instance.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown training engine {name!r}. "
            f"Registered engines: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]()


def list_engines() -> List[str]:
    """
    Return the names of all registered training engines.

    Useful for populating CLI help text or validating user input.
    """
    return list(_REGISTRY)


# ------------------------------------------------------------------
# Built-in registrations
# ------------------------------------------------------------------

def _register_builtins() -> None:
    from phoonnx_train.engines.vits import VitsTrainingEngine
    from phoonnx_train.engines.disentangled_vits import DisentangledVitsTrainingEngine

    register_engine("vits", VitsTrainingEngine)
    register_engine("disentangled-vits", DisentangledVitsTrainingEngine)


_register_builtins()
