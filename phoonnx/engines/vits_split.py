"""Split a monolithic VITS/Piper ONNX graph into an encoder/decoder pair.

A standard Piper/VITS export is a **single** graph: phoneme IDs -> waveform. The
:class:`~phoonnx.engines.vits_streaming.VitsStreamingAdapter` needs two graphs so
it can decode a sentence in chunks. Re-exporting from the PyTorch checkpoint is
the usual way to get them, but it is risky: a version mismatch in the export
pipeline can silently corrupt the duration predictor (see the ``ryan+RT`` case
study in ``docs/streaming.md``).

This module avoids re-export entirely. VITS has one natural cut point -- the
input to the HiFiGAN waveform decoder, which is ``(z * y_mask)`` of shape
``[B, 192, T]``. Everything before it (text encoder, duration predictor, flow,
length regulation, mask multiply) is the *encoder*; the HiFiGAN generator is the
*decoder*. Cutting the ONNX graph there with :func:`onnx.utils.extract_model`
produces two subgraphs that are **the same ops as the original** -- so the split
is lossless by construction and cannot introduce prosody drift.

Because the cut tensor is already masked, the decoder subgraph needs only that
single latent input (plus the speaker id on multi-speaker models, which
``extract_model`` pulls in automatically). The encoder subgraph keeps the
model's original inputs and emits the cut tensor as its sole output.
"""
import os
from pathlib import Path
from typing import List, Optional, Tuple

from ovos_utils.log import LOG

# 192 == VITS ``inter_channels``: the channel count of the latent z that the
# HiFiGAN decoder consumes. The flow's WN layers also carry 192-channel convs,
# so the decoder entry is disambiguated by name (below), not by channel count
# alone.
_INTER_CHANNELS = 192
# Name fragments that mark the HiFiGAN generator entry conv across the piper /
# coqui / vits export lineages seen in the wild.
_DECODER_NAME_HINTS = ("waveform_decoder", "/dec/", "dec.", "conv_pre", "generator")


def _find_cut_tensor(model) -> str:
    """Return the name of the ``(z * y_mask)`` tensor that feeds the decoder.

    Strategy: among all ``Conv`` nodes whose weight has ``in_channels == 192``
    (candidate ``conv_pre`` of the HiFiGAN decoder *and* the flow WN layers),
    keep the ones that are **not** inside the flow (``/flow/``) and whose name
    looks like the waveform decoder. The data input of that conv is the cut.
    """
    init = {i.name: tuple(i.dims) for i in model.graph.initializer}
    candidates: List[Tuple[str, str]] = []  # (node_name, data_input)
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        weight = init.get(node.input[1])
        if not weight or len(weight) != 3 or weight[1] != _INTER_CHANNELS:
            continue
        if "/flow/" in node.name or ".flow." in node.name:
            continue  # flow coupling layers, not the decoder
        candidates.append((node.name, node.input[0]))

    if not candidates:
        raise ValueError(
            "Could not locate the VITS decoder entry (no non-flow Conv with "
            f"{_INTER_CHANNELS} input channels). This does not look like a "
            "splittable VITS/Piper model.")

    # Prefer a candidate whose name matches a known decoder hint; otherwise the
    # single remaining non-flow 192-in conv is the decoder entry.
    for name, data_input in candidates:
        if any(hint in name for hint in _DECODER_NAME_HINTS):
            return data_input
    if len(candidates) == 1:
        return candidates[0][1]
    raise ValueError(
        "Ambiguous VITS decoder entry; candidates: "
        f"{[c[0] for c in candidates]}. Cannot auto-split safely.")


def split_paths(model_path: str) -> Tuple[Path, Path]:
    """Return the ``(encoder, decoder)`` paths this splitter writes for a model
    (siblings of the model, ``<stem>.encoder.onnx`` / ``<stem>.decoder.onnx``)."""
    p = Path(model_path)
    stem = p.name[:-len(".onnx")] if p.name.endswith(".onnx") else p.stem
    return p.with_name(f"{stem}.encoder.onnx"), p.with_name(f"{stem}.decoder.onnx")


def ensure_split_vits(model_path: str, force: bool = False) -> Tuple[str, str]:
    """Split ``model_path`` into an encoder/decoder pair, caching the result.

    Idempotent: if both split files already sit next to the model they are
    reused (this is a one-time cost, not a per-load one). Returns the
    ``(encoder_path, decoder_path)`` as strings.

    Raises ``ValueError`` if the model is not a splittable single-graph VITS.
    """
    encoder_path, decoder_path = split_paths(model_path)
    if not force and encoder_path.is_file() and decoder_path.is_file():
        return str(encoder_path), str(decoder_path)

    import onnx  # local import: onnx (the full package) is only needed to split

    model = onnx.load(model_path)
    output_name = model.graph.output[0].name
    cut = _find_cut_tensor(model)
    LOG.info(f"Splitting VITS model {os.path.basename(model_path)} at '{cut}' "
             f"-> {encoder_path.name} + {decoder_path.name}")

    # Shape inference populates intermediate value_info so extract_model can wire
    # the boundary tensor; the strict checker is skipped because a mid-graph cut
    # legitimately leaves some dynamic dims without a static shape.
    inferred = onnx.shape_inference.infer_shapes(model)
    tmp = str(Path(model_path).with_suffix(".inferred.onnx"))
    onnx.save(inferred, tmp)
    try:
        input_names = [i.name for i in model.graph.input]
        onnx.utils.extract_model(tmp, str(encoder_path), input_names, [cut],
                                 check_model=False)
        onnx.utils.extract_model(tmp, str(decoder_path), [cut], [output_name],
                                 check_model=False)
    finally:
        if os.path.isfile(tmp):
            os.remove(tmp)
    return str(encoder_path), str(decoder_path)
