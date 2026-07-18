"""Unit tests for the streaming (split encoder/decoder) VITS adapter.

The headline correctness property is *discard-and-stitch losslessness*: decoding
a latent ``z`` in overlapping chunks, discarding the context margins and
concatenating the clean middles, must reproduce a one-shot decode of the same
``z`` exactly. Real convolutional edge-contamination is a model property; what
the adapter code is responsible for is the slice/hop/concatenate bookkeeping,
which these tests pin down with a deterministic pointwise fake decoder.
"""
import numpy as np
import pytest

from phoonnx.engines import get_adapter, detect_engine
from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.vits_streaming import VitsStreamingAdapter, _DEFAULT_STREAMING


class _Named:
    def __init__(self, name):
        self.name = name


class FakeEncoder:
    """Encoder stub: emits z[0, 0, t] == t so the decoded audio is trivially
    predictable and any misplaced sample is detectable."""

    def __init__(self, T, input_names=("input", "input_lengths", "scales", "sid")):
        self.T = T
        self._inputs = [_Named(n) for n in input_names]
        self.last_feed = None

    def get_inputs(self):
        return self._inputs

    def run(self, names, feed):
        self.last_feed = feed
        z = np.zeros((1, 192, self.T), dtype=np.float32)
        z[0, 0, :] = np.arange(self.T, dtype=np.float32)
        y_mask = np.ones((1, 1, self.T), dtype=np.float32)
        return [z, y_mask]


class FakeDecoder:
    """Pointwise decoder: each latent frame becomes ``hop`` identical samples of
    its z[0,0] value. Being pointwise (no receptive field), a decoded slice equals
    the matching region of a full decode -- so a correct discard-and-stitch loop
    reconstructs the one-shot audio bit-for-bit. Counts calls to prove chunking."""

    def __init__(self, hop, rank3=False):
        self.hop = hop
        self.rank3 = rank3
        self.calls = 0

    def run(self, names, feed):
        self.calls += 1
        z = feed["z"]
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


def _adapter(hop, T, rank3=False, streaming=None, set_hop=True):
    a = VitsStreamingAdapter(decoder_session=FakeDecoder(hop, rank3=rank3))
    if streaming:
        a.configure_from_params({"streaming": streaming})
    if set_hop:
        a._hop_length = hop
    return a, FakeEncoder(T)


# --------------------------------------------------------------------------
# Registration & strict detection
# --------------------------------------------------------------------------

def test_registered_as_vits_streaming_engine():
    assert isinstance(get_adapter("vits_streaming"), VitsStreamingAdapter)


def test_detect_requires_both_flag_and_decoder():
    # both present -> match
    assert VitsStreamingAdapter.detect(
        config={"streaming": True, "engine_params": {"decoder_path": "d.onnx"}}) is True
    # streaming flag but no decoder graph -> a single-graph voice, not streamable
    assert VitsStreamingAdapter.detect(
        config={"streaming": True, "engine_params": {}}) is False
    # decoder path but no streaming flag -> not claimed
    assert VitsStreamingAdapter.detect(
        config={"engine_params": {"decoder_path": "d.onnx"}}) is False
    # nothing -> not claimed
    assert VitsStreamingAdapter.detect(config={}) is False
    assert VitsStreamingAdapter.detect(config=None) is False


def test_detect_does_not_hijack_plain_vits():
    """A normal single-graph VITS config must fall through to the plain adapter."""
    plain = {"engine": "vits", "inference": {"noise_scale": 0.667}}
    assert VitsStreamingAdapter.detect(config=plain) is False


# --------------------------------------------------------------------------
# Parameter handling
# --------------------------------------------------------------------------

def test_streaming_params_default_and_merge():
    a = VitsStreamingAdapter()
    assert a._streaming == _DEFAULT_STREAMING
    # user overrides merge over defaults, untouched keys keep their default
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
    # a single-speaker encoder (no sid input) drops it
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
    assert a.decoder.calls > 1  # actually chunked
    ref = np.repeat(np.arange(T), hop).astype(np.float32)
    assert res.audio.shape == ref.shape
    assert np.array_equal(res.audio, ref)  # bit-identical to one-shot


@pytest.mark.parametrize("T", [129, 200, 321, 700, 1000])
@pytest.mark.parametrize("margin", [16, 32, 64])
def test_stitch_is_lossless_across_lengths_and_margins(T, margin):
    """Regardless of sentence length or context margin, the stitched output must
    equal a one-shot decode of the same z, with the exact same length."""
    hop = 256
    a, enc = _adapter(hop, T, streaming={**_DEFAULT_STREAMING, "context_margin": margin})
    res = a.synthesize(_request(), enc)
    ref = np.repeat(np.arange(T), hop).astype(np.float32)
    assert res.audio.shape == ref.shape, "length drift in stitching"
    assert np.abs(res.audio - ref).max() == 0.0


def test_stitch_lossless_with_rank3_decoder_output():
    """Decoders that emit [B, 1, N] instead of [N] must stitch identically."""
    hop, T = 256, 400
    a, enc = _adapter(hop, T, rank3=True)
    res = a.synthesize(_request(), enc)
    ref = np.repeat(np.arange(T), hop).astype(np.float32)
    assert np.array_equal(res.audio, ref)


def test_hop_measured_when_not_configured():
    """With no configured hop_length, the adapter measures it from the first
    full-context decode and still stitches losslessly."""
    hop, T = 256, 500
    a, enc = _adapter(hop, T, set_hop=False)  # _hop_length stays None
    assert a._hop_length is None
    res = a.synthesize(_request(), enc)
    assert a._hop_length == hop  # measured from the first chunk
    ref = np.repeat(np.arange(T), hop).astype(np.float32)
    assert np.array_equal(res.audio, ref)


def test_synthesize_without_decoder_raises():
    a = VitsStreamingAdapter()  # no decoder graph
    with pytest.raises(RuntimeError):
        a.synthesize(_request(), FakeEncoder(T=300))
