"""Unit tests for the streaming (split encoder/decoder) VITS adapter and the
load-time auto-splitter.

The headline correctness property is *discard-and-stitch losslessness*: decoding
a latent ``z`` in overlapping chunks, discarding the context margins and
concatenating the clean middles, must reproduce a one-shot decode of the same
``z`` exactly. Real convolutional edge-contamination is a model property; what
the adapter code is responsible for is the slice/hop/concatenate bookkeeping and
the interface adaptation (named ``z``/``y_mask`` for pre-split "+RT" voices vs. a
single arbitrarily-named latent tensor for auto-split models).
"""
import numpy as np
import pytest

from phoonnx.engines import get_adapter, detect_engine
from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.vits_streaming import VitsStreamingAdapter, _DEFAULT_STREAMING


class _IO:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape


class FakeEncoder:
    """Encoder stub emitting named ``z``/``y_mask`` with z[0,0,t] == t, so the
    decoded audio is trivially predictable and any misplaced sample is caught."""

    def __init__(self, T, input_names=("input", "input_lengths", "scales", "sid"),
                 single_output=False):
        self.T = T
        self.single_output = single_output
        self._inputs = [_IO(n) for n in input_names]
        self.last_feed = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        if self.single_output:
            return [_IO("/Mul_7_output_0", [1, 192, None])]
        return [_IO("z", [1, 192, None]), _IO("y_mask", [1, 1, None])]

    def run(self, names, feed):
        self.last_feed = feed
        z = np.zeros((1, 192, self.T), dtype=np.float32)
        z[0, 0, :] = np.arange(self.T, dtype=np.float32)
        if self.single_output:
            return [z]
        y_mask = np.ones((1, 1, self.T), dtype=np.float32)
        return [z, y_mask]


class FakeDecoder:
    """Pointwise decoder: each latent frame becomes ``hop`` identical samples of
    its z[0,0] value. Being pointwise (no receptive field), a decoded slice equals
    the matching region of a full decode, so a correct discard-and-stitch loop
    reconstructs the one-shot audio bit-for-bit. Declares its inputs so the
    adapter's name/shape matching is exercised."""

    def __init__(self, hop, rank3=False, single_input=False):
        self.hop = hop
        self.rank3 = rank3
        self.single_input = single_input
        self.calls = 0
        self.seen_inputs = None

    def get_inputs(self):
        if self.single_input:
            return [_IO("/Mul_7_output_0", [1, 192, None])]
        return [_IO("z", [1, 192, None]), _IO("y_mask", [1, 1, None])]

    def run(self, names, feed):
        self.calls += 1
        self.seen_inputs = set(feed)
        z = feed["/Mul_7_output_0"] if self.single_input else feed["z"]
        samples = np.repeat(z[0, 0, :], self.hop).astype(np.float32)
        if self.rank3:
            return [samples.reshape(1, 1, -1)]
        return [samples]


def _request(n=6, speaker_id=None):
    ids = np.arange(n, dtype=np.int64)[None, :]
    return AdapterSynthesisRequest(
        phoneme_ids=ids,
        phoneme_lengths=np.array([n], dtype=np.int64),
        speaker_id=speaker_id,
        params={},
    )


def _adapter(hop, T, rank3=False, streaming=None, set_hop=True,
             single_output=False, single_input=False):
    a = VitsStreamingAdapter(decoder_session=FakeDecoder(hop, rank3=rank3,
                                                         single_input=single_input))
    if streaming:
        a.configure_from_params({"streaming": streaming})
    if set_hop:
        a._hop_length = hop
    return a, FakeEncoder(T, single_output=single_output)


# --------------------------------------------------------------------------
# Registration & strict detection
# --------------------------------------------------------------------------

def test_registered_as_vits_streaming_engine():
    assert isinstance(get_adapter("vits_streaming"), VitsStreamingAdapter)


def test_detect_requires_both_flag_and_decoder():
    assert VitsStreamingAdapter.detect(
        config={"streaming": True, "engine_params": {"decoder_path": "d.onnx"}}) is True
    # streaming flag but no decoder graph in engine_params -> the auto-splitter
    # (in TTSVoice.load) injects decoder_path *before* detection runs, so at
    # detect() time a streamable voice always carries decoder_path.
    assert VitsStreamingAdapter.detect(
        config={"streaming": True, "engine_params": {}}) is False
    assert VitsStreamingAdapter.detect(
        config={"engine_params": {"decoder_path": "d.onnx"}}) is False
    assert VitsStreamingAdapter.detect(config={}) is False
    assert VitsStreamingAdapter.detect(config=None) is False


def test_detect_does_not_hijack_plain_vits():
    plain = {"engine": "vits", "inference": {"noise_scale": 0.667}}
    assert VitsStreamingAdapter.detect(config=plain) is False


# --------------------------------------------------------------------------
# Parameter handling
# --------------------------------------------------------------------------

def test_streaming_params_default_and_merge():
    a = VitsStreamingAdapter()
    assert a._streaming == _DEFAULT_STREAMING
    a.configure_from_params({"streaming": {"context_margin": 24}})
    assert a._streaming["context_margin"] == 24
    assert a._streaming["first_chunk"] == _DEFAULT_STREAMING["first_chunk"]
    assert a._streaming["fallback_frames"] == _DEFAULT_STREAMING["fallback_frames"]


def test_build_feed_dict_filters_and_includes_sid():
    a = VitsStreamingAdapter()
    enc = FakeEncoder(T=4)
    feed = a.build_feed_dict(_request(speaker_id=3), enc)
    assert feed["scales"] == pytest.approx([0.667, 1.0, 0.8], abs=1e-4)
    assert feed["sid"].tolist() == [3]
    enc2 = FakeEncoder(T=4, input_names=("input", "input_lengths", "scales"))
    assert "sid" not in a.build_feed_dict(_request(speaker_id=3), enc2)


# --------------------------------------------------------------------------
# Synthesis: fallback vs. chunked, and the losslessness invariant
# --------------------------------------------------------------------------

def test_short_sentence_falls_back_to_one_shot():
    hop, T = 256, 100  # T <= default fallback_frames (128)
    a, enc = _adapter(hop, T)
    res = a.synthesize(_request(), enc)
    assert a.decoder.calls == 1  # exactly one decode, no chunking
    assert np.array_equal(res.audio, np.repeat(np.arange(T), hop).astype(np.float32))


def test_long_sentence_streams_in_multiple_chunks():
    hop, T = 256, 700
    a, enc = _adapter(hop, T, streaming={"first_chunk": 8, "next_chunk": 160,
                                         "context_margin": 32, "fallback_frames": 128})
    res = a.synthesize(_request(), enc)
    assert a.decoder.calls > 1
    ref = np.repeat(np.arange(T), hop).astype(np.float32)
    assert res.audio.shape == ref.shape
    assert np.array_equal(res.audio, ref)


@pytest.mark.parametrize("T", [129, 200, 321, 700, 1000])
@pytest.mark.parametrize("margin", [16, 32, 64])
def test_stitch_is_lossless_across_lengths_and_margins(T, margin):
    hop = 256
    a, enc = _adapter(hop, T, streaming={**_DEFAULT_STREAMING, "context_margin": margin})
    res = a.synthesize(_request(), enc)
    ref = np.repeat(np.arange(T), hop).astype(np.float32)
    assert res.audio.shape == ref.shape, "length drift in stitching"
    assert np.abs(res.audio - ref).max() == 0.0


def test_stitch_lossless_with_rank3_decoder_output():
    hop, T = 256, 400
    a, enc = _adapter(hop, T, rank3=True)
    res = a.synthesize(_request(), enc)
    ref = np.repeat(np.arange(T), hop).astype(np.float32)
    assert np.array_equal(res.audio, ref)


def test_stitch_lossless_with_autosplit_single_tensor_interface():
    """Auto-split encoders emit one arbitrarily-named latent (no separate mask)
    and the decoder takes a single input; the adapter must match by shape."""
    hop, T = 256, 500
    a, enc = _adapter(hop, T, single_output=True, single_input=True)
    res = a.synthesize(_request(), enc)
    assert a.decoder.seen_inputs == {"/Mul_7_output_0"}  # no y_mask fed
    ref = np.repeat(np.arange(T), hop).astype(np.float32)
    assert np.array_equal(res.audio, ref)


def test_hop_measured_when_not_configured():
    hop, T = 256, 500
    a, enc = _adapter(hop, T, set_hop=False)
    assert a._hop_length is None
    res = a.synthesize(_request(), enc)
    assert a._hop_length == hop
    ref = np.repeat(np.arange(T), hop).astype(np.float32)
    assert np.array_equal(res.audio, ref)


def test_synthesize_without_decoder_raises():
    a = VitsStreamingAdapter()
    with pytest.raises(RuntimeError):
        a.synthesize(_request(), FakeEncoder(T=300))


def test_stream_yields_each_window_and_joins_to_synthesize():
    hop, T = 256, 500
    a, enc = _adapter(hop, T)
    parts = list(a.synthesize_stream(_request(), enc))
    assert len(parts) > 1
    whole = a.synthesize(_request(), enc).audio
    assert np.array_equal(np.concatenate([p.audio for p in parts]), whole)


def test_stream_yields_the_first_window_before_the_last_is_decoded():
    hop, T = 256, 500
    a, enc = _adapter(hop, T)
    stream = a.synthesize_stream(_request(), enc)
    next(stream)
    calls_at_first_part = a.decoder.calls
    list(stream)
    assert calls_at_first_part == 1
    assert a.decoder.calls > calls_at_first_part


def test_stream_short_sentence_is_one_part():
    a, enc = _adapter(256, _DEFAULT_STREAMING["fallback_frames"])
    parts = list(a.synthesize_stream(_request(), enc))
    assert len(parts) == 1
    assert a.decoder.calls == 1


def test_decoder_runs_on_the_voice_providers(monkeypatch):
    seen = {}

    class Session:
        def __init__(self, path, sess_options=None, providers=None):
            seen.update(path=path, providers=providers, threads=sess_options.intra_op_num_threads)

        def get_providers(self):
            return seen["providers"]

    monkeypatch.setattr("onnxruntime.InferenceSession", Session)
    monkeypatch.setattr("phoonnx.providers.available_providers",
                        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.delenv("PHOONNX_ORT_CACHE_DIR", raising=False)
    adapter = VitsStreamingAdapter()
    adapter.configure_from_params({"decoder_path": "d.onnx", "intra_op_num_threads": 3,
                                   "providers": ["CPUExecutionProvider"]})
    assert seen == {"path": "d.onnx", "providers": ["CPUExecutionProvider"], "threads": 3}


def test_decoder_keeps_its_threads_when_loaded_from_the_ort_cache(tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    path = tmp_path / "decoder.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)
    monkeypatch.setenv("PHOONNX_ORT_CACHE_DIR", str(tmp_path / "cache"))
    params = {"decoder_path": str(path), "intra_op_num_threads": 3, "providers": ["CPUExecutionProvider"]}
    for _ in range(2):  # the first load writes the cached graph, the second reads it
        adapter = VitsStreamingAdapter()
        adapter.configure_from_params(params)
        options = adapter.decoder.get_session_options()
        assert (options.intra_op_num_threads, options.inter_op_num_threads) == (3, 1)


def test_a_relative_decoder_path_is_read_from_the_config_folder(tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    import json
    from onnx import TensorProto, helper
    from phoonnx.tokenizer import DEFAULT_IPA_PHONEME_ID_MAP
    from phoonnx.voice import TTSVoice
    voice_dir, elsewhere = tmp_path / "voice", tmp_path / "elsewhere"
    voice_dir.mkdir()
    elsewhere.mkdir()
    for name in ("model.onnx", "decoder.onnx"):
        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])], "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), voice_dir / name)
    config = {"streaming": True, "engine_params": {"decoder_path": "decoder.onnx"},
              "phoneme_type": "espeak", "alphabet": "ipa", "lang_code": "en-US",
              "audio": {"sample_rate": 22050}, "phoneme_id_map": DEFAULT_IPA_PHONEME_ID_MAP}
    (voice_dir / "model.onnx.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.chdir(elsewhere)
    voice = TTSVoice.load(str(voice_dir / "model.onnx"))
    assert isinstance(voice.adapter, VitsStreamingAdapter)
    assert voice.adapter.decoder is not None
    assert voice.config.engine_params["decoder_path"] == str(voice_dir / "decoder.onnx")
