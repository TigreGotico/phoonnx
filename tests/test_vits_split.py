"""Tests for the load-time VITS ONNX splitter (encoder/decoder surgery).

These build a *minimal* synthetic graph shaped like the relevant part of a VITS
export -- a pre-decoder tensor feeding the HiFiGAN ``conv_pre`` (a Conv with 192
input channels) -- so the splitter's cut-point finder and ``extract_model``
wiring are exercised deterministically, with no model download. The correctness
bar is that encoder->decoder reproduces the monolithic graph's output exactly,
which is the whole point of splitting instead of re-exporting.
"""
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from phoonnx.engines.vits_split import _find_cut_tensor, ensure_split_vits, split_paths


def _decoder_conv_model(pre_conv_name="/waveform_decoder/conv_pre/Conv"):
    """A tiny graph: input[1,192,T] -> Relu -> (cut) -> conv_pre(192->4) ->
    conv_post(4->1) -> output[1,1,T]. The cut tensor is the Relu output that
    feeds the 192-input conv, exactly as in a real VITS model."""
    rng = np.random.default_rng(0)
    w_pre = numpy_helper.from_array(
        rng.standard_normal((4, 192, 1)).astype(np.float32), name="w_pre")
    w_post = numpy_helper.from_array(
        rng.standard_normal((1, 4, 1)).astype(np.float32), name="w_post")
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 192, "T"])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, "T"])
    nodes = [
        helper.make_node("Relu", ["input"], ["pre"], name="/pre/Relu"),
        helper.make_node("Conv", ["pre", "w_pre"], ["h"], name=pre_conv_name),
        helper.make_node("Conv", ["h", "w_post"], ["output"],
                         name="/waveform_decoder/conv_post/Conv"),
    ]
    graph = helper.make_graph(nodes, "tiny_vits", [inp], [out], [w_pre, w_post])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def test_find_cut_tensor_locates_decoder_entry():
    model = _decoder_conv_model()
    assert _find_cut_tensor(model) == "pre"


def test_find_cut_tensor_ignores_flow_convs():
    """A 192-in conv inside the flow must not be mistaken for the decoder."""
    model = _decoder_conv_model(pre_conv_name="/flow/flows.0/enc/Conv")
    # only a flow conv carries 192 inputs -> no decoder entry -> not splittable
    with pytest.raises(ValueError):
        _find_cut_tensor(model)


def test_find_cut_tensor_rejects_non_vits():
    """A graph with no 192-input conv is not a splittable VITS."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, "T"])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, "T"])
    node = helper.make_node("Relu", ["input"], ["output"], name="/Relu")
    graph = helper.make_graph([node], "not_vits", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    with pytest.raises(ValueError):
        _find_cut_tensor(model)


def test_split_is_lossless_and_cached(tmp_path):
    model_path = tmp_path / "voice.onnx"
    onnx.save(_decoder_conv_model(), str(model_path))

    enc_path, dec_path = ensure_split_vits(str(model_path))
    # paths are deterministic siblings of the model
    exp_enc, exp_dec = split_paths(str(model_path))
    assert (enc_path, dec_path) == (str(exp_enc), str(exp_dec))
    assert exp_enc.is_file() and exp_dec.is_file()

    full = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    enc = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])

    # decoder takes exactly the single cut tensor; encoder emits it
    assert [i.name for i in dec.get_inputs()] == ["pre"]
    assert [o.name for o in enc.get_outputs()] == ["pre"]

    x = np.random.default_rng(1).standard_normal((1, 192, 40)).astype(np.float32)
    ref = full.run(None, {"input": x})[0]
    cut = enc.run(None, {"input": x})[0]
    got = dec.run(None, {"pre": cut})[0]
    assert np.array_equal(got, ref)  # split == monolithic, bit-for-bit


def test_ensure_split_reuses_cache(tmp_path):
    model_path = tmp_path / "voice.onnx"
    onnx.save(_decoder_conv_model(), str(model_path))
    enc1, dec1 = ensure_split_vits(str(model_path))
    # delete the source model: a genuine cache hit must not need it anymore
    model_path.unlink()
    enc2, dec2 = ensure_split_vits(str(model_path))
    assert (enc1, dec1) == (enc2, dec2)
