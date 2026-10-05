"""Tests for the ito engine adapter (Lokutor's distilled English TTS).

Unit tests cover the adapter contract with a dummy session (no weights
needed). The integration tests run only when the ito package, the checkpoint
and an exported voice bundle are present on the machine — the weights live
on a gated HuggingFace repository, so CI runs the unit tests alone and the
export is exercised by `scripts/conversion/ito/export_ito_onnx.py
--check-parity`.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from phoonnx.engines import get_adapter, list_engines
from phoonnx.engines.base import AdapterSynthesisRequest, BaseOnnxAdapter
from phoonnx.engines.ito import ItoAdapter

EXPORTED_VOICE = Path("/home/miro/tmp/ito-out/ito_female")


class _Named:
    def __init__(self, name):
        self.name = name


class DummySession:
    def __init__(self, input_names):
        self._inputs = [_Named(n) for n in input_names]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return [_Named("output")]


def _request(n=5, **params):
    ids = np.arange(1, n + 1, dtype=np.int64)[None, :]
    return AdapterSynthesisRequest(
        phoneme_ids=ids, phoneme_lengths=np.array([n], dtype=np.int64),
        speaker_id=0, language_id=0, params=params,
    )


def test_registered():
    assert "ito" in list_engines()
    assert isinstance(get_adapter("ito"), ItoAdapter)


def test_detect():
    assert ItoAdapter.detect({"engine": "ito"})
    assert not ItoAdapter.detect({"engine": "styletts2"})
    assert not ItoAdapter.detect({})


def test_default_params():
    assert ItoAdapter().default_params() == {"speed": 1.0}


def test_param_labels():
    assert ItoAdapter().param_labels() == {"speed": "Speed"}


def test_feed_dict_pads_and_filters():
    ad = ItoAdapter()
    feed = ad.build_feed_dict(_request(), DummySession(["input", "speed"]))
    # the ito front end prepends the pad token (id 0) to every utterance
    assert feed["input"].tolist() == [[0, 1, 2, 3, 4, 5]]
    assert feed["speed"] == np.float32(1.0)


def test_feed_dict_alias_names():
    ad = ItoAdapter()
    feed = ad.build_feed_dict(_request(), DummySession(["input_ids", "speed"]))
    assert "input_ids" in feed
    assert "input" not in feed and "tokens" not in feed


def test_length_scale_is_the_rate_knob():
    ad = ItoAdapter()
    feed = ad.build_feed_dict(_request(length_scale=1.5),
                              DummySession(["input", "speed"]))
    assert feed["speed"] == np.float32(1.5)
    # the direct alias still works when the standard knob is absent
    feed = ad.build_feed_dict(_request(speed=2.0), DummySession(["input", "speed"]))
    assert feed["speed"] == np.float32(2.0)


def test_token_budget_guard():
    ad = ItoAdapter()
    big = np.ones((1, 400), dtype=np.int64)
    req = AdapterSynthesisRequest(
        phoneme_ids=big, phoneme_lengths=np.array([400], dtype=np.int64), params={})
    with pytest.raises(ValueError, match="split the text"):
        ad.build_feed_dict(req, DummySession(["input", "speed"]))
    # just under the budget (pad included) still passes
    ok = np.ones((1, 399), dtype=np.int64)
    req = AdapterSynthesisRequest(
        phoneme_ids=ok, phoneme_lengths=np.array([399], dtype=np.int64), params={})
    feed = ad.build_feed_dict(req, DummySession(["input", "speed"]))
    assert feed["input"].shape == (1, 400)


def test_configure_uses_vocabulary_pad():
    ad = ItoAdapter()
    ad.configure(MagicMock(tokenizer=MagicMock(pad_id=7),
                           engine_params={"max_tokens": 10}))
    assert ad._pad_id == 7
    assert ad._max_tokens == 10
    feed = ad.build_feed_dict(_request(2), DummySession(["input", "speed"]))
    assert feed["input"].tolist() == [[7, 1, 2]]


def test_parse_outputs():
    ad = ItoAdapter()
    wav = np.zeros((1, 9000), dtype=np.float32)
    dur = np.full((1, 6), 4, dtype=np.int64)
    res = ad.parse_outputs([wav, dur], _request(5), ["wav", "durations"])
    assert res.audio.shape == (9000,)
    assert res.extras["phoneme_id_samples"].tolist() == [4] * 6


def test_parse_outputs_no_durations():
    ad = ItoAdapter()
    wav = np.zeros((1, 9000), dtype=np.float32)
    res = ad.parse_outputs([wav], _request(5), ["wav"])
    assert "phoneme_id_samples" not in res.extras


# ---------------------------------------------------------------- integration
pytest.importorskip("ito")
requires_voice = pytest.mark.skipif(
    not (EXPORTED_VOICE / "model.onnx").is_file()
    or not (EXPORTED_VOICE / "config.json").is_file(),
    reason="exported ito voice bundle not present (gated weights)")


@requires_voice
def test_engine_wiring():
    from phoonnx.config import Engine
    from phoonnx.voice import TTSVoice

    voice = TTSVoice.load(str(EXPORTED_VOICE / "model.onnx"),
                          str(EXPORTED_VOICE / "config.json"))
    assert voice.config.engine == Engine.ITO
    assert isinstance(voice.adapter, ItoAdapter)
    assert voice.config.sample_rate == 24000


@requires_voice
def test_token_ids_match_ito_reference():
    from ito.text import text_to_ids
    from phoonnx.voice import TTSVoice

    voice = TTSVoice.load(str(EXPORTED_VOICE / "model.onnx"),
                          str(EXPORTED_VOICE / "config.json"))
    for text in ("Hello there!", "The package arrives on Friday.",
                 "Could you grab some quinoa on your way home?"):
        ids = []
        for chunk in voice.phonemize(text):
            ids.extend(voice.phonemes_to_ids(list(chunk)))
        # ito's reference emits [pad, *ids]; the adapter adds the pad
        assert ids == text_to_ids(text)[1:], text


@requires_voice
def test_synthesize_matches_ito_reference_length():
    import torch
    from phoonnx.voice import TTSVoice

    voice = TTSVoice.load(str(EXPORTED_VOICE / "model.onnx"),
                          str(EXPORTED_VOICE / "config.json"))
    chunks = list(voice.synthesize("The package arrives on Friday."))
    audio = np.concatenate([c.audio_float_array for c in chunks])
    # reference: 227 frames * 300 hop = 68100 samples
    assert len(audio) == 68100

    # and the model's own duration output agrees with the reference
    from ito.text import text_to_ids
    sess = voice.session
    ids = np.array([text_to_ids("The package arrives on Friday.")],
                   dtype=np.int64)
    _, dur, _, _ = sess.run(None, {"input": ids,
                                   "speed": np.array([1.0], dtype=np.float32)})
    assert int(dur.sum()) == 227
