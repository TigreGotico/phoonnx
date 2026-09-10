"""Pure ``onnx``-package graph surgery for the phoneme-alignment output.

This module only ever imports the lightweight ``onnx`` graph-manipulation
package (never ``onnxruntime`` for inference, never ``torch``), so it is
safe to import both from the training-side export CLI
(``phoonnx_train/export_onnx.py``, which depends on this package) and from
runtime code (``phoonnx.alignment``, which must never depend on ``torch`` or
``phoonnx_train``).

The core operation: promote the per-phoneme duration tensor (found via the
``Ceil`` node every VITS-family duration predictor ends with, rounding
predicted log-durations to an integer frame count) to a named ONNX graph
output. Standard exports don't do this — it is optional and either baked in
at export time (``--add-phoneme-alignment``), applied offline afterwards
(``phoonnx-voices add-alignment``), or applied on demand at load time when a
model without it is asked for alignments (see ``phoonnx.alignment``).
"""
from typing import Optional, Set


def find_duration_tensor(model, tensor_name: str = "autodetect") -> Optional[str]:
    """Return the name of the tensor to expose as the alignment output.

    ``tensor_name="autodetect"`` looks for a unique ``Ceil`` node output — the
    rounded per-phoneme duration in every VITS-family export phoonnx has
    seen. Returns ``None`` (never raises) when no such tensor exists or more
    than one candidate is found (ambiguous); callers log and degrade to "no
    alignment available".
    """
    if tensor_name != "autodetect":
        return tensor_name

    ceil_tensor_names: Set[str] = set()
    for node in model.graph.node:
        if node.op_type != "Ceil":
            continue
        ceil_tensor_names.update(node.output)

    if not ceil_tensor_names or len(ceil_tensor_names) > 1:
        return None
    return next(iter(ceil_tensor_names))


def add_phoneme_alignment_output(model, tensor_name: str = "autodetect") -> Optional[str]:
    """Mutate ``model`` (an ``onnx.ModelProto``) in place, promoting the
    duration tensor to a graph output.

    Returns the promoted tensor's name on success — including when it was
    already an output, a no-op success — or ``None`` when the tensor could
    not be located (no/ambiguous ``Ceil`` node). Callers own loading/saving
    the model and any higher-level logging; this function never raises for
    the "not found" case and only imports the pure ``onnx`` package.
    """
    import onnx

    name = find_duration_tensor(model, tensor_name)
    if name is None:
        return None

    if any(output.name == name for output in model.graph.output):
        return name  # already exposed - nothing to do

    value_info = onnx.helper.ValueInfoProto()
    value_info.name = name
    model.graph.output.append(value_info)
    return name
