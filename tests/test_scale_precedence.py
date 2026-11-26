"""Scale precedence: the call, then the voice's ``config.json``, then the engine's ``default_params()``.

The engine call is recorded, so no model runs. The expected values are the
engine's own ``default_params()``, the numbers in the config and the call, and
hand arithmetic for the SSML rate.
"""
import numpy as np
import pytest

from phoonnx.config import SynthesisConfig, VoiceConfig
from phoonnx.engines import get_adapter, list_engines
from phoonnx.engines.base import AdapterSynthesisResult
from phoonnx.engines.f5tts import F5TTSAdapter
from phoonnx.engines.fastpitch import FastPitchAdapter
from phoonnx.engines.mixertts import MixerTTSAdapter
from phoonnx.engines.optispeech import OptiSpeechAdapter
from phoonnx.engines.styletts2 import StyleTTS2Adapter
from phoonnx.engines.supertonic import SuperTonicAdapter
from phoonnx.engines.vits import VitsAdapter
from phoonnx.engines.zipvoice import ZipVoiceAdapter
from phoonnx.voice import AudioChunk, TTSVoice

CONFIG = {"phoonnx_version": "1.0", "engine": "phoonnx", "phoneme_type": "unicode",
          "alphabet": "unicode", "lang_code": "en-US", "audio": {"sample_rate": 16000},
          "num_symbols": 8, "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a": [3]}}


class _Engine(VitsAdapter):
    """A VITS engine whose defaults differ from 0.667, 1.0 and 0.8."""

    def default_params(self):
        return {"noise_scale": 0.3, "length_scale": 1.25, "noise_w_scale": 0.5}


def _config(inference, engine_params=None):
    return VoiceConfig.from_dict(dict(CONFIG, inference=inference,
                                      **({"engine_params": engine_params} if engine_params else {})))


def _sent_params(adapter, inference, call=None, engine_params=None):
    """The params the engine receives for one id sequence."""
    voice = TTSVoice.__new__(TTSVoice)  # no session or phonemizer: the engine call is recorded
    voice.config, voice.adapter, voice.session = _config(inference, engine_params), adapter, None
    sent = []

    def record(request, session):
        sent.append(request.params)
        return AdapterSynthesisResult(audio=np.zeros(1, dtype=np.float32))

    adapter.synthesize = record
    voice._infer([3], call or SynthesisConfig())
    return sent[0]


def test_a_config_reads_only_the_scales_it_sets():
    config = _config({})
    assert (config.noise_scale, config.length_scale, config.noise_w_scale) == (None, None, None)
    config = _config({"length_scale": 1.1, "noise_w": 0.6})
    assert (config.noise_scale, config.length_scale, config.noise_w_scale) == (None, 1.1, 0.6)


def test_a_config_without_scales_sends_the_engine_defaults():
    assert _sent_params(_Engine(), {}) == {"noise_scale": 0.3, "length_scale": 1.25, "noise_w_scale": 0.5}


@pytest.mark.parametrize("name", list_engines())
def test_every_engine_gets_its_own_defaults_untouched(name):
    assert _sent_params(get_adapter(name), {}) == get_adapter(name).default_params()


@pytest.mark.parametrize("inference, sent", [
    ({}, {"noise_scale": 0.3, "length_scale": 1.25, "noise_w_scale": 0.5}),
    ({"noise_scale": 0.4, "length_scale": 1.1, "noise_w": 0.6},
     {"noise_scale": 0.4, "length_scale": 1.1, "noise_w_scale": 0.6}),
])
def test_the_native_config_writes_only_the_scales_it_read(inference, sent):
    written = _config(inference).to_native_dict()["inference"]
    assert {k: written[k] for k in ("noise_scale", "length_scale", "noise_w") if k in written} == inference
    assert _sent_params(_Engine(), written) == sent


def test_the_scales_a_config_sets_win_over_the_engine():
    assert _sent_params(_Engine(), {"length_scale": 1.1}) == {
        "noise_scale": 0.3, "length_scale": 1.1, "noise_w_scale": 0.5}


def test_a_call_scale_wins_over_the_config_and_the_engine():
    params = _sent_params(_Engine(), {"noise_scale": 0.4, "length_scale": 1.1},
                          SynthesisConfig(length_scale=0.9))
    assert params == {"noise_scale": 0.4, "length_scale": 0.9, "noise_w_scale": 0.5}


# The engines with a speed key of their own, and that key for length_scale 2.0:
# MixerTTS, FastPitch, StyleTTS2, F5-TTS and ZipVoice read a rate, so twice as slow is 1 / 2;
# SuperTonic reads a rate whose default is 1.05, so twice as slow is 1.05 / 2; and
# OptiSpeech reads a duration scale, so twice as slow is 2.
OWN_SPEED_KEY = [
    (MixerTTSAdapter, "pace", 0.5),
    (FastPitchAdapter, "pace", 0.5),
    (StyleTTS2Adapter, "speed", 0.5),
    (SuperTonicAdapter, "speed", 1.05 / 2),
    (F5TTSAdapter, "speed", 0.5),
    (ZipVoiceAdapter, "speed", 0.5),
    (OptiSpeechAdapter, "d_factor", 2.0),
]


@pytest.mark.parametrize("adapter, key, expected", OWN_SPEED_KEY)
def test_a_call_length_scale_reaches_the_engines_own_speed_key(adapter, key, expected):
    assert _sent_params(adapter(), {}, SynthesisConfig(length_scale=2.0))[key] == expected


@pytest.mark.parametrize("adapter, key, expected", OWN_SPEED_KEY)
def test_a_config_length_scale_reaches_the_engines_own_speed_key(adapter, key, expected):
    assert _sent_params(adapter(), {"length_scale": 2.0})[key] == expected


@pytest.mark.parametrize("adapter, key, expected", OWN_SPEED_KEY)
def test_a_speed_key_the_call_names_wins_over_length_scale(adapter, key, expected):
    call = SynthesisConfig(length_scale=2.0, extra={key: 0.7})
    assert _sent_params(adapter(), {}, call)[key] == 0.7


@pytest.mark.parametrize("adapter, key, expected", OWN_SPEED_KEY)
def test_no_length_scale_leaves_the_engines_own_speed_default(adapter, key, expected):
    assert _sent_params(adapter(), {})[key] == adapter().default_params()[key]


# A voice config that names the engine's speed key: the published F5-TTS voices write
# "speed": 1.0 in their engine_params. 0.8 is a pace the voice's author chose.
def _scaled(adapter, key, expected, by=0.8):
    """*by* changed as the length scale 2.0 changes the engine's speed at length scale 1."""
    return by * expected / adapter().length_scale_params(1.0)[key]


@pytest.mark.parametrize("adapter, key, expected", OWN_SPEED_KEY)
def test_a_call_length_scale_scales_a_speed_the_config_names(adapter, key, expected):
    sent = _sent_params(adapter(), {}, SynthesisConfig(length_scale=2.0), {key: 0.8})
    assert sent[key] == pytest.approx(_scaled(adapter, key, expected))


@pytest.mark.parametrize("adapter, key, expected", OWN_SPEED_KEY)
def test_a_config_length_scale_leaves_a_speed_the_config_names(adapter, key, expected):
    assert _sent_params(adapter(), {"length_scale": 2.0}, None, {key: 0.8})[key] == 0.8


@pytest.mark.parametrize("adapter, key, expected", OWN_SPEED_KEY)
def test_a_call_repeating_the_voices_length_scale_leaves_a_speed_the_config_names(adapter, key, expected):
    # the OVOS plugin passes the voice's own length_scale on every call
    sent = _sent_params(adapter(), {"length_scale": 1.3}, SynthesisConfig(length_scale=1.3), {key: 0.8})
    assert sent[key] == pytest.approx(0.8)


@pytest.mark.parametrize("adapter, key, expected", OWN_SPEED_KEY)
def test_a_speed_the_call_names_wins_over_a_speed_the_config_names(adapter, key, expected):
    call = SynthesisConfig(length_scale=2.0, extra={key: 0.7})
    assert _sent_params(adapter(), {}, call, {key: 0.8})[key] == 0.7


@pytest.mark.parametrize("adapter, key, expected", OWN_SPEED_KEY)
def test_a_length_scale_in_the_calls_extra_scales_a_speed_the_config_names(adapter, key, expected):
    call = SynthesisConfig(extra={"length_scale": 2.0})
    assert _sent_params(adapter(), {}, call, {key: 0.8})[key] == pytest.approx(_scaled(adapter, key, expected))


@pytest.mark.parametrize("adapter, key, expected", OWN_SPEED_KEY)
def test_a_length_scale_in_engine_params_is_the_voices_own(adapter, key, expected):
    # a call repeating it changes nothing; a call doubling it scales the speed as 2.0 does 1.0
    named = {"length_scale": 1.5, key: 0.8}
    assert _sent_params(adapter(), {}, SynthesisConfig(length_scale=1.5), named)[key] == pytest.approx(0.8)
    doubled = _sent_params(adapter(), {}, SynthesisConfig(length_scale=3.0), named)[key]
    engine = adapter()
    ratio = engine.length_scale_params(3.0)[key] / engine.length_scale_params(1.5)[key]
    assert doubled == pytest.approx(0.8 * ratio)


@pytest.mark.parametrize("adapter, key, expected", OWN_SPEED_KEY)
def test_a_speed_the_config_sets_to_none_is_left_alone(adapter, key, expected):
    assert _sent_params(adapter(), {}, SynthesisConfig(length_scale=2.0), {key: None})[key] is None


@pytest.mark.parametrize("adapter, key, expected", OWN_SPEED_KEY)
def test_ssml_rate_scales_a_speed_the_config_names(adapter, key, expected):
    sent = []
    engine = adapter()
    engine.synthesize = lambda request, session: (
        sent.append(request.params) or AdapterSynthesisResult(audio=np.zeros(1, dtype=np.float32)))

    class _Voice(TTSVoice):
        frontend = None  # the document holds no say-as

        def __init__(self):  # no model: the engine call is recorded
            self.config, self.adapter, self.session = _config({}, {key: 0.8}), engine, None

        def synthesize(self, text, syn_config=None, include_alignments=False):
            self._infer([3], syn_config or SynthesisConfig())
            yield AudioChunk(sample_rate=16000, sample_width=2, sample_channels=1,
                             audio_float_array=np.full(160, 0.1, dtype=np.float32))

    list(_Voice().synthesize_ssml('<speak><prosody rate="x-slow">slow</prosody> plain</speak>'))
    slow = engine.length_scale_params(1 / 0.6)[key] / engine.length_scale_params(1.0)[key]
    assert [p[key] for p in sent] == [pytest.approx(0.8 * slow), 0.8], (
        "x-slow changes the voice's own speed as it changes the engine's default; "
        "the words outside it keep the voice's speed")


@pytest.mark.parametrize("adapter, key, expected", OWN_SPEED_KEY)
def test_ssml_rate_is_relative_to_a_length_scale_in_engine_params(adapter, key, expected):
    # engine_params length_scale 2.0 and speed 0.8, x-slow: the call's length scale is
    # 2.0 / 0.6, so the speed changes as from 2.0 to 2.0 / 0.6, not as from 1.0 to 1.0 / 0.6
    sent = []
    engine = adapter()
    engine.synthesize = lambda request, session: (
        sent.append(request.params) or AdapterSynthesisResult(audio=np.zeros(1, dtype=np.float32)))

    class _Voice(TTSVoice):
        frontend = None  # the document holds no say-as

        def __init__(self):  # no model: the engine call is recorded
            self.config = _config({}, {"length_scale": 2.0, key: 0.8})
            self.adapter, self.session = engine, None

        def synthesize(self, text, syn_config=None, include_alignments=False):
            self._infer([3], syn_config or SynthesisConfig())
            yield AudioChunk(sample_rate=16000, sample_width=2, sample_channels=1,
                             audio_float_array=np.full(160, 0.1, dtype=np.float32))

    list(_Voice().synthesize_ssml('<speak><prosody rate="x-slow">slow</prosody> plain</speak>'))
    assert sent[0]["length_scale"] == pytest.approx(2.0 / 0.6)
    slow = engine.length_scale_params(2.0 / 0.6)[key] / engine.length_scale_params(2.0)[key]
    assert [p[key] for p in sent] == [pytest.approx(0.8 * slow), 0.8]


class _RecordingVoice(TTSVoice):
    """A voice whose synthesize() records its settings instead of running a model."""

    frontend = None  # no phonemizer: the document holds no say-as

    def __init__(self, inference):  # no model: skip TTSVoice.__init__
        self.config, self.adapter, self.calls = _config(inference), _Engine(), []

    def synthesize(self, text, syn_config=None, include_alignments=False):
        self.calls.append(syn_config)
        yield AudioChunk(sample_rate=16000, sample_width=2, sample_channels=1,
                         audio_float_array=np.full(160, 0.1, dtype=np.float32))


@pytest.mark.parametrize("inference, length_scale", [
    ({}, 1.25 / 1.25),  # the engine's length_scale over rate="fast"
    ({"length_scale": 1.1}, 1.1 / 1.25),  # the config's length_scale over rate="fast"
])
def test_ssml_rate_divides_the_resolved_length_scale(inference, length_scale):
    voice = _RecordingVoice(inference)
    list(voice.synthesize_ssml('<speak><prosody rate="fast">quick</prosody></speak>'))
    assert voice.calls[0].length_scale == pytest.approx(length_scale)
