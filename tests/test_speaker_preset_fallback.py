"""A call that names no speaker or language gets the voice's own value.

``SynthesisConfig`` leaves ``speaker_id`` and ``lang_id`` as None when the call
does not set them. The voice passes None on to the adapter, so an adapter with a
per-voice preset (Magpie's ``engine_options.speaker_id``) can apply it, and an
adapter with no preset feeds 0.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import phoonnx
from phoonnx.config import SynthesisConfig, VoiceConfig
from phoonnx.engines import get_adapter
from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.magpie import MagpieAdapter
from phoonnx.engines.vits_streaming import VitsStreamingAdapter
from phoonnx.model_manager import TTSModelInfo
from phoonnx.voice import TTSVoice

MAGPIE_INDEX = json.loads(
    (Path(phoonnx.__file__).parent / "voice_index" / "magpie.json").read_text(encoding="utf-8"))


class _Input:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape


class _Session:
    """Declares the named inputs, records each feed and returns a short waveform."""

    def __init__(self, input_names):
        self._inputs = [_Input(name) for name in input_names]
        self.feeds = []

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return [_Input("output")]

    def run(self, _names, feed):
        self.feeds.append(feed)
        return [np.zeros((1, 1, 16), np.float32)]


def _magpie_voice(voice_id):
    """Build the voice the way ``TTSModelInfo.load`` does, with stub graphs.

    The index entry's ``engine_options`` reach the voice config through
    ``TTSModelInfo.engine_params``, as on a real load. The graph downloads are
    stubbed out, and ``generate_codes`` records the speaker index it receives.
    """
    info = TTSModelInfo(**MAGPIE_INDEX[voice_id])
    info.download_aux_models = lambda: {}
    config = VoiceConfig.from_dict(
        {"engine": "magpie", "phoneme_type": "unicode", "alphabet": "graphemes",
         "lang_code": info.lang, "audio": {"sample_rate": 22050}},
        engine_params=info.engine_params())

    adapter = MagpieAdapter()
    for graph in ("encoder", "cross_kv", "local", "audio_embed", "lt_embed"):
        setattr(adapter, graph, MagicMock())
    adapter.codec = MagicMock()
    adapter.codec.run.return_value = [np.zeros((1, 16), np.float32)]
    num_speakers = 1 + max(e["engine_options"]["speaker_id"] for e in MAGPIE_INDEX.values())
    adapter.context_embeddings = np.zeros((num_speakers, 4), np.float32)
    adapter.speakers_seen = []

    def generate_codes(token_ids, speaker, params):
        adapter.speakers_seen.append(speaker)
        return np.zeros((2, 4), np.int64)

    adapter.generate_codes = generate_codes
    return TTSVoice(session=MagicMock(), config=config, adapter=adapter)


@pytest.mark.parametrize("voice_id", sorted(MAGPIE_INDEX))
def test_magpie_voice_speaks_as_its_index_speaker(voice_id):
    voice = _magpie_voice(voice_id)
    voice.phoneme_ids_to_audio([5, 6, 7])
    assert voice.adapter.speakers_seen == [MAGPIE_INDEX[voice_id]["engine_options"]["speaker_id"]]


def test_magpie_jason_speaks_as_jason_with_no_speaker_in_the_call():
    expected = MAGPIE_INDEX["magpie/Jason/ar"]["engine_options"]["speaker_id"]
    assert expected != 0
    voice = _magpie_voice("magpie/Jason/ar")
    voice.phoneme_ids_to_audio([5, 6, 7], SynthesisConfig())
    assert voice.adapter.speakers_seen == [expected]


@pytest.mark.parametrize("speaker", [0, 3])
def test_magpie_speaker_in_the_call_wins_over_the_index_speaker(speaker):
    assert speaker != MAGPIE_INDEX["magpie/Jason/ar"]["engine_options"]["speaker_id"]
    voice = _magpie_voice("magpie/Jason/ar")
    voice.phoneme_ids_to_audio([5, 6, 7], SynthesisConfig(speaker=speaker))
    assert voice.adapter.speakers_seen == [speaker]


def _vits_voice(session):
    config = VoiceConfig.from_dict({
        "engine": "piper", "phoneme_type": "unicode", "alphabet": "unicode", "lang_code": "en",
        "audio": {"sample_rate": 22050}, "num_symbols": 4, "num_speakers": 4,
        "phoneme_id_map": {"_": [0], "a": [1], "b": [2], "c": [3]},
    })
    return TTSVoice(session=session, config=config)


def test_voice_with_no_preset_feeds_speaker_0_and_language_0():
    session = _Session(["input", "input_lengths", "scales", "sid", "langid"])
    voice = _vits_voice(session)
    voice.phoneme_ids_to_audio([1, 2, 3])
    assert session.feeds[-1]["sid"].tolist() == [0]
    assert session.feeds[-1]["langid"].tolist() == [0]


def test_voice_with_no_preset_feeds_the_speaker_and_language_of_the_call():
    session = _Session(["input", "input_lengths", "scales", "sid", "langid"])
    voice = _vits_voice(session)
    voice.phoneme_ids_to_audio([1, 2, 3], SynthesisConfig(speaker=2, language=1))
    assert session.feeds[-1]["sid"].tolist() == [2]
    assert session.feeds[-1]["langid"].tolist() == [1]


# Each engine with no speaker preset and the inputs that carry speaker and language.
NO_PRESET_INPUTS = [
    ("vits", ["input", "input_lengths", "scales"], {"sid": [0], "langid": [0]}),
    ("vits_streaming", ["input", "input_lengths", "scales"], {"sid": [0]}),
    ("glowtts", ["input", "input_lengths", "scales"], {"sid": [0]}),
    ("optispeech", ["x", "x_lengths", "scales"], {"sids": [0], "lids": [0]}),
    ("matcha", ["x", "x_lengths", "scales"], {"spks": [0]}),
    ("mixertts", ["token_ids", "pace"], {"speaker": [0]}),
    ("shami", ["phoneme_ids", "phoneme_lengths", "language_ids"], {"speaker_id": [0]}),
    ("tinyvoca", ["tokens", "stress", "speed"], {"voice": [0]}),
]


@pytest.mark.parametrize("engine,inputs,expected", NO_PRESET_INPUTS,
                         ids=[row[0] for row in NO_PRESET_INPUTS])
def test_adapter_with_no_preset_feeds_0_when_the_request_has_none(engine, inputs, expected):
    session = _Session(inputs + list(expected))
    request = AdapterSynthesisRequest(phoneme_ids=np.array([[1, 2, 3]], np.int64),
                                      phoneme_lengths=np.array([3], np.int64))
    assert request.speaker_id is None and request.language_id is None
    feed = get_adapter(engine).build_feed_dict(request, session)
    assert {name: feed[name].tolist() for name in expected} == expected


def test_streaming_decoder_gets_speaker_0_when_the_request_has_none():
    decoder = _Session(["z", "sid"])
    adapter = VitsStreamingAdapter(decoder_session=decoder)
    request = AdapterSynthesisRequest(phoneme_ids=np.array([[1, 2, 3]], np.int64),
                                      phoneme_lengths=np.array([3], np.int64))
    adapter.synthesize(request, _Session(["input", "input_lengths", "scales", "sid"]))
    assert [feed["sid"].tolist() for feed in decoder.feeds] == [[0]]
