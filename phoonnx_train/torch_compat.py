"""torch version compatibility helpers for phoonnx_train."""
from contextlib import contextmanager
from typing import Any, Dict

import torch
from packaging.version import parse as _parse_version


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
    """torch<2.4 has no torch.compiler.disable. Use this decorator instead
    so functions that torch.compile cannot trace (e.g. data-dependent
    control flow in the VITS spline transforms) are excluded from graph
    capture on new torch, and are a no-op on old torch."""
    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    if disable is not None:
        return disable(fn)
    return fn


def onnx_export_kwargs() -> Dict[str, Any]:
    """torch>=2.5 defaults torch.onnx.export to the dynamo exporter, which
    cannot trace VITS's data-dependent control flow (and needs the optional
    onnxscript package); force the TorchScript exporter (dynamo=False) on
    those versions. torch 2.1-2.4's onnx.export does not accept a `dynamo`
    kwarg at all, so it must be omitted there entirely."""
    if _parse_version(torch.__version__.split("+")[0]) >= _parse_version("2.5"):
        return {"dynamo": False}
    return {}
