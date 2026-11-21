"""torch version compatibility helpers for phoonnx_train."""
import inspect
from contextlib import contextmanager
from typing import Any, Dict

import torch


@contextmanager
def trusting_torch_load():
    """torch>=2.6 defaults torch.load(weights_only=True), which rejects
    pickled Lightning checkpoints. Loading your own checkpoint is trusted —
    force weights_only=False for the duration of the context."""
    orig = torch.load

    def _load(*a: Any, **k: Any):
        k["weights_only"] = False
        return orig(*a, **k)

    torch.load = _load
    try:
        yield
    finally:
        torch.load = orig


def compiler_disable(fn):
    """torch<2.4 has no torch.compiler.disable. Wrap a function so it is
    excluded from graph capture on new torch, and a no-op on old torch.

    Applied ONLY when torch.compile is actually enabled for a run — see
    ``disable_compile_on_transforms``. It must never be applied at import
    time: the wrapper carries dynamo eval-frame hooks that torch.jit.trace
    (used by ONNX export) rejects with "Detected that you are using FX to
    torch.jit.trace a dynamo-optimized function". Export paths must always
    see the raw functions."""
    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    if disable is not None:
        return disable(fn)
    return fn


def disable_compile_on_transforms() -> None:
    """Wrap the VITS spline transforms with ``compiler_disable`` in place, so
    that a torch.compile'd graph excludes their data-dependent control flow.

    Call this ONLY when a training run has torch.compile enabled and it
    succeeded — never at import time, or ONNX export's torch.jit.trace of the
    flow module would hit the wrapped functions and raise."""
    from phoonnx_train.vits import modules, transforms

    for name in (
        "piecewise_rational_quadratic_transform",
        "unconstrained_rational_quadratic_spline",
        "rational_quadratic_spline",
    ):
        setattr(transforms, name, compiler_disable(getattr(transforms, name)))

    # modules.py binds ``piecewise_rational_quadratic_transform`` by value at
    # import (``from .transforms import ...``); re-bind its copy too so the
    # call site actually used by the flow forward is disabled under compile.
    if hasattr(modules, "piecewise_rational_quadratic_transform"):
        modules.piecewise_rational_quadratic_transform = compiler_disable(
            modules.piecewise_rational_quadratic_transform
        )


def onnx_export_kwargs() -> Dict[str, Any]:
    """torch>=2.5 defaults torch.onnx.export to the dynamo exporter, which
    cannot trace VITS's data-dependent control flow (and needs the optional
    onnxscript package); force the TorchScript exporter (dynamo=False)
    whenever the running torch accepts the kwarg. torch 2.1-2.4's
    onnx.export does not accept `dynamo` at all, so it must be omitted
    there entirely — probing the signature asks torch itself instead of
    hardcoding a version boundary."""
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        return {"dynamo": False}
    return {}
