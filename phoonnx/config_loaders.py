"""Deprecated name of :mod:`phoonnx.compat`, which holds the config loaders."""
from typing import Any


def __getattr__(name: str) -> Any:
    from phoonnx import compat
    from phoonnx.config import _warn_moved
    value = getattr(compat, name)
    _warn_moved(f"phoonnx.config_loaders.{name}", f"phoonnx.compat.{name}")
    return value
