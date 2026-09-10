"""Import weights from the released ONNX graphs into the PyTorch modules.

This is the fine-tune-from-``supertonic-3`` path. It reads every initializer
(constant tensor) out of an ONNX file, resolving the anonymous initializers that
``torch.onnx.export`` emits for some linear layers by tracing which node
consumes them (the node name keeps the original module path). Tensors are then
copied into a target module's ``state_dict`` by an explicit
released-name -> local-name map, tolerating the usual conv-1x1<->linear squeeze
and linear-weight transpose. A :class:`PortReport` records what loaded, what was
missing, and any shape mismatches; nothing is copied silently under a wrong shape.

Because the released graphs do not contain the encoder-side networks (speech
encoder, the two style encoders), those always stay at their fresh init and must
be trained; the report lists them as expected-missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch


def load_onnx_initializers(path: str) -> Dict[str, np.ndarray]:
    """Return ``{qualified_name: array}`` for every initializer in ``path``.

    Named initializers are used as-is. Anonymous ones (``onnx::...`` / ``/...``)
    are attributed to the consuming node's module path, with the input index
    disambiguating weight (``.weight``) from bias (the 3rd input of Conv/Gemm).
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(path)
    inits = {i.name: i for i in model.graph.initializer}
    consumer: Dict[str, Tuple[str, int]] = {}
    for node in model.graph.node:
        for idx, inp in enumerate(node.input):
            if inp in inits and (inp.startswith("onnx::") or inp.startswith("/")):
                consumer.setdefault(inp, (node.name, idx))

    out: Dict[str, np.ndarray] = {}
    for name, init in inits.items():
        arr = numpy_helper.to_array(init)
        if name.startswith("onnx::") or name.startswith("/"):
            entry = consumer.get(name)
            if entry is None:
                continue
            node_name, idx = entry
            suffix = "bias" if idx >= 2 else "weight"
            qualified = node_name.strip("/").rsplit("/", 1)[0].replace("/", ".") + f".{suffix}"
        else:
            qualified = name
        out[qualified] = arr
    return out


@dataclass
class PortReport:
    loaded: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    mismatched: List[Tuple[str, tuple, tuple]] = field(default_factory=list)

    def summary(self) -> str:
        return f"loaded={len(self.loaded)} missing={len(self.missing)} mismatched={len(self.mismatched)}"


def _shape_aware_copy(dst: torch.Tensor, src_np: np.ndarray) -> bool:
    src = torch.from_numpy(np.ascontiguousarray(src_np)).float()
    if src.dim() == dst.dim() + 1 and src.shape[-1] == 1:  # conv-1x1 -> linear
        src = src.squeeze(-1)
    with torch.no_grad():
        if src.shape == dst.shape:
            dst.copy_(src)
        elif src.dim() == 2 and src.t().shape == dst.shape:
            dst.copy_(src.t())
        elif src.numel() == dst.numel():
            dst.copy_(src.reshape(dst.shape))
        else:
            return False
    return True


def assign_by_map(module: torch.nn.Module, arrays: Dict[str, np.ndarray],
                  mapping: Dict[str, str], report: PortReport) -> PortReport:
    """Copy ``arrays[onnx_name]`` into ``module``'s parameter ``local_name`` for
    every ``local_name -> onnx_name`` in ``mapping``."""
    params = dict(module.named_parameters())
    params.update(module.named_buffers())
    for local_name, onnx_name in mapping.items():
        if local_name not in params:
            report.missing.append(f"{local_name} (no such parameter)")
            continue
        if onnx_name not in arrays:
            report.missing.append(f"{local_name} <- {onnx_name}")
            continue
        if _shape_aware_copy(params[local_name], arrays[onnx_name]):
            report.loaded.append(local_name)
        else:
            report.mismatched.append((local_name, tuple(arrays[onnx_name].shape),
                                      tuple(params[local_name].shape)))
    return report


def import_onnx_weights(onnx_dir: str, *, autoencoder=None, text_to_latent=None,
                        duration_predictor=None,
                        mappings: Dict[str, Dict[str, str]] | None = None) -> Dict[str, PortReport]:
    """Best-effort port from ``<onnx_dir>/{vocoder,text_encoder,vector_estimator,
    duration_predictor}.onnx`` into whichever modules are supplied.

    ``mappings`` provides, per graph, a ``local_name -> onnx_name`` dict. When a
    mapping is omitted the graph is loaded but nothing is assigned (the report
    lists it as fully missing), so a caller can inspect the available names first.
    """
    import os

    mappings = mappings or {}
    reports: Dict[str, PortReport] = {}
    graphs = {
        "vocoder": autoencoder,
        "text_encoder": text_to_latent,
        "vector_estimator": text_to_latent,
        "duration_predictor": duration_predictor,
    }
    for graph, module in graphs.items():
        path = os.path.join(onnx_dir, f"{graph}.onnx")
        if module is None or not os.path.exists(path):
            continue
        arrays = load_onnx_initializers(path)
        report = PortReport()
        assign_by_map(module, arrays, mappings.get(graph, {}), report)
        reports[graph] = report
    return reports
