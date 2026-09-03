import unittest
"""Tests for the pluggable multi-engine inference framework.

The framework refactored TTSVoice's synthesis path from inline ONNX I/O into
engine adapters. These tests pin the VITS adapter's behaviour so the refactor
is provably equivalent to the previous inline logic for every ONNX input
convention (piper / MMS / HuggingFace), plus the registry and detection.
"""
import numpy as np
import pytest

from phoonnx.engines import (
    register_engine, get_adapter, detect_engine, list_engines,
)
from phoonnx.engines.base import (
    AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter,
)
from phoonnx.engines.vits import VitsAdapter


class _Named:
    def __init__(self, name): self.name = name


class DummySession:
    def __init__(self, input_names, run_fn=None):
        self._inputs = [_Named(n) for n in input_names]
        self._run_fn = run_fn

    def get_inputs(self): return self._inputs
    def get_outputs(self): return [_Named("output")]
    def run(self, _none, feed): return self._run_fn(feed)


def _request(n=5, spk=0, lang=0, **params):
    ids = np.arange(1, n + 1, dtype=np.int64)[None, :]
    return AdapterSynthesisRequest(
        phoneme_ids=ids, phoneme_lengths=np.array([n], dtype=np.int64),
        speaker_id=spk, language_id=lang, params=params,
    )


# ---------------------------------------------------------------- registry
def test_registry_has_vits():
    assert "vits" in list_engines()
    assert isinstance(get_adapter("vits"), VitsAdapter)


def test_get_adapter_unknown_raises():
    with pytest.raises(KeyError):
        get_adapter("does-not-exist")


def test_register_and_detect_priority():
    import phoonnx.engines as eng
    # snapshot the global registry so this test doesn't leak into others
    snap = (dict(eng._REGISTRY), dict(eng._PRIORITIES), list(eng._DETECT_ORDER))
    try:
        class Dummy(VitsAdapter):
            @staticmethod
            def detect(config=None, session=None): return True
        register_engine("dummy_high", Dummy, detect_priority=1)
        # lower priority number is checked first
        assert isinstance(detect_engine(config={"x": 1}), Dummy)
    finally:
        eng._REGISTRY.clear(); eng._REGISTRY.update(snap[0])
        eng._PRIORITIES.clear(); eng._PRIORITIES.update(snap[1])
        eng._DETECT_ORDER[:] = snap[2]


def test_detect_falls_back_to_vits():
    # nothing matches -> VITS fallback
    adapter = detect_engine(config={"totally": "unknown"})
    assert isinstance(adapter, VitsAdapter)


# ---------------------------------------------- VitsAdapter.build_feed_dict
def test_feed_piper_inputs():
    """piper/phoonnx export: input / input_lengths / scales / sid."""
    sess = DummySession(["input", "input_lengths", "scales", "sid"])
    feed = VitsAdapter().build_feed_dict(_request(spk=3), sess)
    assert set(feed) == {"input", "input_lengths", "scales", "sid"}
    assert feed["input"].tolist() == [[1, 2, 3, 4, 5]]
    assert feed["input_lengths"].tolist() == [5]
    assert feed["scales"] == pytest.approx([0.667, 1.0, 0.8], abs=1e-4)  # noise, length, noise_w
    assert feed["sid"].tolist() == [3]


def test_feed_mms_inputs():
    """MMS export: x / x_length — and the piper aliases are filtered out."""
    sess = DummySession(["x", "x_length", "scales"])
    feed = VitsAdapter().build_feed_dict(_request(), sess)
    assert set(feed) == {"x", "x_length", "scales"}
    assert "input" not in feed and "input_ids" not in feed


def test_feed_hf_inputs():
    """HuggingFace export: input_ids / attention_mask."""
    sess = DummySession(["input_ids", "attention_mask"])
    feed = VitsAdapter().build_feed_dict(_request(), sess)
    assert set(feed) == {"input_ids", "attention_mask"}
    assert feed["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]


def test_feed_separate_scalar_scales():
    """exports with split scale inputs instead of a packed `scales`."""
    sess = DummySession(["input", "input_lengths", "noise_scale",
                         "length_scale", "noise_scale_w"])
    feed = VitsAdapter().build_feed_dict(
        _request(noise_scale=0.5, length_scale=1.2, noise_w_scale=0.9), sess)
    assert feed["noise_scale"].tolist() == pytest.approx([0.5])
    assert feed["length_scale"].tolist() == pytest.approx([1.2])
    assert feed["noise_scale_w"].tolist() == pytest.approx([0.9])
    assert "scales" not in feed


def test_feed_langid_included_when_input_present():
    sess = DummySession(["input", "input_lengths", "scales", "langid"])
    feed = VitsAdapter().build_feed_dict(_request(lang=2), sess)
    assert feed["langid"].tolist() == [2]


def test_params_override_defaults():
    sess = DummySession(["scales"])
    feed = VitsAdapter().build_feed_dict(
        _request(noise_scale=0.1, length_scale=2.0, noise_w_scale=0.3), sess)
    assert feed["scales"] == pytest.approx([0.1, 2.0, 0.3], abs=1e-4)


def test_no_regression_vs_inline_logic():
    """Golden: the adapter feed equals what the previous inline code built."""
    ids = np.array([[1, 2, 3]], dtype=np.int64)
    lens = np.array([3], dtype=np.int64)
    ns, ls, nw = 0.667, 1.0, 0.8
    # what the old inline path produced for a piper model (then filtered):
    expected = {
        "input": ids, "input_lengths": lens,
        "scales": np.array([ns, ls, nw], dtype=np.float32),
        "sid": np.array([0], dtype=np.int64),
    }
    sess = DummySession(["input", "input_lengths", "scales", "sid"])
    feed = VitsAdapter().build_feed_dict(_request(n=3), sess)
    assert set(feed) == set(expected)
    for k in expected:
        assert np.array_equal(feed[k], expected[k]), k


# ----------------------------------------------- parse_outputs / detect
def test_parse_outputs_squeeze():
    res = VitsAdapter().parse_outputs([np.zeros((1, 1, 2048), np.float32)], _request())
    assert isinstance(res, AdapterSynthesisResult)
    assert res.audio.shape == (2048,)


@pytest.mark.parametrize("cfg", [
    {"engine": "piper"}, {"engine": "coqui"},
    {"phoneme_id_map": {"a": [1]}, "phoneme_type": "espeak"},
    {"phoonnx_version": "1.0"}, {"piper_version": "1.0"},
    {"characters": {"pad": "_"}},
    {"phonemizer": "espeak", "phonemes": {}},
])
def test_detect_config_signatures(cfg):
    assert VitsAdapter.detect(config=cfg) is True


def test_detect_session_scales():
    assert VitsAdapter.detect(session=DummySession(["input", "scales"])) is True
    assert VitsAdapter.detect(session=DummySession(["mels"])) is False


def test_default_params_and_parse_config():
    a = VitsAdapter()
    assert a.default_params() == {"noise_scale": 0.667, "length_scale": 1.0, "noise_w_scale": 0.8}
    p = a.parse_config({"inference": {"noise_scale": 0.4, "length_scale": 1.5, "noise_w": 0.7}})
    assert p == {"noise_scale": 0.4, "length_scale": 1.5, "noise_w_scale": 0.7}


def test_full_synth_delegation():
    """End-to-end: build_feed_dict -> session.run -> parse_outputs."""
    captured = {}
    def run_fn(feed):
        captured.update(feed)
        return [np.full((1, 1, 100), 0.25, dtype=np.float32)]
    sess = DummySession(["input", "input_lengths", "scales", "sid"], run_fn=run_fn)
    adapter = VitsAdapter()
    req = _request(n=4)
    out = sess.run(None, adapter.build_feed_dict(req, sess))
    audio = adapter.parse_outputs(out, req).audio
    assert captured["input"].tolist() == [[1, 2, 3, 4]]
    assert audio.shape == (100,) and float(audio[0]) == pytest.approx(0.25)


class TestExplicitEngineWins(unittest.TestCase):
    """A voice that names its engine is authoritative. Heuristic probes exist for
    voices that do NOT name one and must never override a config that does —
    otherwise a lower-priority adapter claims voices belonging to another engine
    (Matcha's vocoder_path probe runs before GlowTTS and VITS and matched any
    voice bundling a vocoder)."""

    def test_named_engine_beats_a_lower_priority_heuristic(self):
        from phoonnx.engines import detect_engine
        from phoonnx.engines.glowtts import GlowTTSAdapter
        from phoonnx.engines.vits import VitsAdapter
        cfg = {"engine": "glowtts", "engine_params": {"vocoder_path": "/x/vocoder.onnx"}}
        self.assertIsInstance(detect_engine(cfg, None), GlowTTSAdapter)
        cfg = {"engine": "vits", "engine_params": {"vocoder_path": "/x/vocoder.onnx"}}
        self.assertIsInstance(detect_engine(cfg, None), VitsAdapter)

    def test_named_engine_is_still_honoured_for_its_own_adapter(self):
        from phoonnx.engines import detect_engine
        from phoonnx.engines.matcha import MatchaAdapter
        cfg = {"engine": "matcha", "engine_params": {"vocoder_path": "/x/vocoder.onnx"}}
        self.assertIsInstance(detect_engine(cfg, None), MatchaAdapter)

    def test_unnamed_engine_still_uses_heuristics(self):
        from phoonnx.engines import detect_engine
        from phoonnx.engines.matcha import MatchaAdapter
        cfg = {"engine_params": {"vocoder_path": "/x/vocoder.onnx"}}
        self.assertIsInstance(detect_engine(cfg, None), MatchaAdapter)

    def test_engine_that_is_not_an_adapter_falls_through(self):
        # "coqui"/"piper" name a training framework, not an adapter
        from phoonnx.engines import detect_engine
        from phoonnx.engines.vits import VitsAdapter
        self.assertIsInstance(detect_engine({"engine": "coqui"}, None), VitsAdapter)
