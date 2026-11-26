"""SSML pitch and range made by the engine itself, where it has the controls.

``<prosody pitch>`` used to be a shift of the finished audio on every engine, and ``range`` was
dropped. An engine with pitch inputs of its own now gets them (tiny-voca's ``pitch_shift`` and
``pitch_range``, Mixer-TTS and FastPitch's ``pitch_mul`` and ``pitch_add``). The engine calls are
recorded, so no model runs.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from phoonnx import audio_fx
from phoonnx.config import AudioPost, SynthesisConfig, VoiceConfig
from phoonnx.engines.base import AdapterSynthesisResult
from phoonnx.engines.mixertts import MixerTTSAdapter
from phoonnx.engines.tinyvoca import TinyVocaAdapter
from phoonnx.engines.vits import VitsAdapter
from phoonnx.ssml import parse_ssml
from phoonnx.voice import AudioChunk, TTSVoice

SEMITONE = math.log(2) / 12


class _Session:
    """A stand-in graph with these input names and this metadata."""

    def __init__(self, inputs, meta=None):
        self._inputs, self._meta = inputs, meta or {}

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in self._inputs]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self._meta)


MIXER = _Session(["token_ids", "pace", "speaker", "emotion", "pitch_mul", "pitch_add"])
FASTPITCH_WITHOUT_PITCH = _Session(["token_ids", "pace"])  # coqui/en-ljspeech-fast_pitch
TINY = _Session(["tokens", "stress", "voice", "speed", "pitch_shift", "pitch_range"])


# -- the reader ----------------------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    ("x-low", 0.25), ("low", 0.5), ("medium", 1.0), ("default", 1.0), ("high", 1.5), ("x-high", 2.0),
    ("+50%", 1.5), ("-50%", 0.5), ("150%", 1.5), ("-100%", 0.0), ("+500%", 2.0),
    ("+2st", 1.0), ("+30Hz", 1.0), ("wide", 1.0),
])
def test_prosody_range(value, expected):
    (piece,) = parse_ssml(f'<speak><prosody range="{value}">words</prosody></speak>')
    assert piece.pitch_range == pytest.approx(expected)


@pytest.mark.parametrize("inner, expected", [("50%", 1.0), ("low", 1.0), ("x-low", 0.5), ("high", 2.0)])
def test_nested_ranges_multiply_and_end_with_their_element(inner, expected):
    pieces = parse_ssml(f'<speak><prosody range="x-high">a <prosody range="{inner}">b</prosody></prosody> c</speak>')
    assert [(p.text, p.pitch_range) for p in pieces] == [("a", 2.0), ("b", expected), ("c", 1.0)]


def test_nested_vocal_tracts_add_up_and_stay_within_12_semitones():
    (one,) = parse_ssml('<speak><amazon:effect vocal-tract-length="-20%">x</amazon:effect></speak>')
    (two,) = parse_ssml('<speak><amazon:effect vocal-tract-length="-20%"><amazon:effect '
                        'vocal-tract-length="-20%">x</amazon:effect></amazon:effect></speak>')
    assert two.tract == pytest.approx(2 * one.tract) and two.pitch == pytest.approx(2 * one.pitch)
    (deep,) = parse_ssml('<speak>' + '<amazon:effect vocal-tract-length="-50%">' * 4 + 'x'
                         + '</amazon:effect>' * 4 + '</speak>')
    assert deep.tract == 12.0 and deep.pitch == 12.0


def test_the_vocal_tract_part_of_the_pitch_is_kept_apart():
    (piece,) = parse_ssml('<speak><amazon:effect vocal-tract-length="-20%">'
                          '<prosody pitch="+2st">words</prosody></amazon:effect></speak>')
    assert piece.tract == pytest.approx(piece.pitch - 2.0)
    assert 3.0 < piece.tract < 4.0
    (plain,) = parse_ssml('<speak><prosody pitch="+2st">words</prosody></speak>')
    assert (plain.pitch, plain.tract) == (2.0, 0.0)


# -- the engines' hooks ----------------------------------------------------------------------

def test_tinyvoca_shifts_its_log_f0_by_ln2_over_12_a_semitone():
    adapter = TinyVocaAdapter()
    assert adapter.pitch_shift_params(4.0, adapter.default_params(), TINY) == {"pitch_shift": pytest.approx(4 * SEMITONE)}
    # a pitch the voice already sets is shifted from where it is
    assert adapter.pitch_shift_params(-2.0, {"pitch_shift": 0.1}, TINY) == {"pitch_shift": pytest.approx(0.1 - 2 * SEMITONE)}
    assert adapter.pitch_range_params(2.0, {"pitch_range": 0.8}, TINY) == {"pitch_range": pytest.approx(1.6)}


def test_tinyvoca_without_the_inputs_leaves_it_to_the_audio():
    adapter, graph = TinyVocaAdapter(), _Session(["tokens", "stress", "voice", "speed"])
    assert adapter.pitch_shift_params(4.0, {}, graph) == {} and adapter.pitch_range_params(2.0, {}, graph) == {}


def _denormalised(z, mul, add, mean, std):
    """The frequencies a Mixer-TTS graph makes from z-scored pitch *z* with these controls."""
    return mean + std * (z * mul + add)


@pytest.mark.parametrize("semitones", [4.0, -4.0, 12.0])
@pytest.mark.parametrize("mul, add", [(1.0, 0.0), (1.3, 0.2)])
def test_a_mixer_shift_multiplies_every_frequency_by_2_to_the_semitones_over_12(semitones, mul, add):
    mean, std = 213.1, 53.2
    adapter = MixerTTSAdapter()
    keys = adapter.pitch_shift_params(semitones, {"pitch_mul": mul, "pitch_add": add,
                                                  "pitch_mean": mean, "pitch_std": std}, MIXER)
    z = np.linspace(-2, 2, 9)
    before = _denormalised(z, mul, add, mean, std)
    after = _denormalised(z, keys["pitch_mul"], keys["pitch_add"], mean, std)
    assert after == pytest.approx(before * 2 ** (semitones / 12))


def test_a_mixer_range_scales_the_movement_and_keeps_the_voices_mean():
    adapter = MixerTTSAdapter()
    keys = adapter.pitch_range_params(2.0, {"pitch_mul": 1.2, "pitch_add": 0.5}, MIXER)
    assert keys == {"pitch_mul": pytest.approx(2.4)}, "pitch_add, where the voice puts its mean, is kept"
    z = np.linspace(-2, 2, 9)  # symmetric around the corpus mean
    before = _denormalised(z, 1.2, 0.5, 213.1, 53.2)
    after = _denormalised(z, keys["pitch_mul"], 0.5, 213.1, 53.2)
    assert after.mean() == pytest.approx(before.mean())
    assert after.std() == pytest.approx(2 * before.std())


def test_a_multi_speaker_mixer_voice_gets_no_range():
    adapter = MixerTTSAdapter()
    adapter.configure(SimpleNamespace(num_speakers=4, engine_params={}))
    assert adapter.pitch_range_params(2.0, adapter.default_params(), MIXER) == {}
    adapter.configure(SimpleNamespace(num_speakers=1, engine_params={}))
    assert adapter.pitch_range_params(2.0, adapter.default_params(), MIXER) == {"pitch_mul": 2.0}


def test_a_mixer_shift_reads_the_statistics_from_the_graph_when_the_voice_has_none():
    adapter = MixerTTSAdapter()
    graph = _Session(MIXER._inputs, {"pitch_mean": "200.0", "pitch_std": "50.0"})
    keys = adapter.pitch_shift_params(12.0, adapter.default_params(), graph)
    assert keys == {"pitch_mul": pytest.approx(2.0), "pitch_add": pytest.approx(200.0 / 50.0)}


@pytest.mark.parametrize("params, meta", [
    ({}, {}),                                          # nobody states them: nipponjo's exports
    ({}, {"pitch_mean": "0.0", "pitch_std": "1.0"}),   # what the export writes without statistics
    ({"pitch_mean": -5, "pitch_std": 50}, {}),
    ({"pitch_mean": "x", "pitch_std": 50}, {}),
])
def test_a_mixer_shift_without_the_speakers_statistics_is_left_to_the_audio(params, meta):
    adapter = MixerTTSAdapter()
    assert adapter.pitch_shift_params(4.0, {**adapter.default_params(), **params}, _Session(MIXER._inputs, meta)) == {}


def test_a_graph_without_pitch_inputs_gets_neither():
    adapter = MixerTTSAdapter()
    params = {**adapter.default_params(), "pitch_mean": 213.1, "pitch_std": 53.2}
    assert adapter.pitch_shift_params(4.0, params, FASTPITCH_WITHOUT_PITCH) == {}
    assert adapter.pitch_range_params(2.0, params, FASTPITCH_WITHOUT_PITCH) == {}


def test_an_engine_without_pitch_controls_has_no_hooks():
    adapter = VitsAdapter()
    assert adapter.pitch_shift_params(4.0, {}, None) == {} and adapter.pitch_range_params(2.0, {}, None) == {}


# -- routing in synthesize_ssml --------------------------------------------------------------

CONFIG = {"phoonnx_version": "1.0", "engine": "phoonnx", "phoneme_type": "unicode",
          "alphabet": "unicode", "lang_code": "en-US", "audio": {"sample_rate": 16000},
          "num_symbols": 8, "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a": [3]}}


def _voice(adapter, session, engine_params=None):
    sent = []
    adapter.synthesize = lambda request, s: (
        sent.append(dict(request.params)) or AdapterSynthesisResult(audio=np.full(1600, 0.1, np.float32)))

    class _Voice(TTSVoice):
        frontend = None  # the documents hold no say-as

        def __init__(self):  # no model: the engine call is recorded
            config = dict(CONFIG, **({"engine_params": engine_params} if engine_params else {}))
            self.config, self.adapter, self.session = VoiceConfig.from_dict(config), adapter, session

        def synthesize(self, text, syn_config=None, include_alignments=False):
            audio, _ = self._infer([3], syn_config or SynthesisConfig())
            yield AudioChunk(sample_rate=16000, sample_width=2, sample_channels=1,
                             audio_float_array=np.asarray(audio, np.float32))

    return _Voice(), sent


@pytest.fixture
def shifts(monkeypatch):
    """The semitones of every shift made on the finished audio."""
    made = []
    real = audio_fx.pitch_shift
    monkeypatch.setattr(audio_fx, "pitch_shift", lambda x, rate, st: made.append(st) or real(x, rate, st))
    return made


RAW = SynthesisConfig(audio=AudioPost(normalize=False))
STATS = {"pitch_mean": 213.1, "pitch_std": 53.2}


def test_a_mixer_voice_with_statistics_makes_the_shift_itself(shifts):
    voice, sent = _voice(MixerTTSAdapter(), MIXER, STATS)
    list(voice.synthesize_ssml('<speak><prosody pitch="+4st">up</prosody> plain</speak>', RAW))
    k = 2 ** (4 / 12)
    assert sent[0]["pitch_mul"] == pytest.approx(k)
    assert sent[0]["pitch_add"] == pytest.approx(213.1 * (k - 1) / 53.2)
    assert (sent[1]["pitch_mul"], sent[1]["pitch_add"]) == (1.0, 0.0), "the words outside keep the voice's pitch"
    assert shifts == [], "a shift the engine made is not made again on the audio"


def test_a_mixer_voice_without_statistics_shifts_the_audio_as_before(shifts):
    voice, sent = _voice(MixerTTSAdapter(), MIXER)
    list(voice.synthesize_ssml('<speak><prosody pitch="+4st">up</prosody></speak>', RAW))
    assert (sent[0]["pitch_mul"], sent[0]["pitch_add"]) == (1.0, 0.0)
    assert shifts == [4.0]


def test_range_reaches_an_engine_that_has_it_and_is_dropped_elsewhere(shifts):
    voice, sent = _voice(TinyVocaAdapter(), TINY)
    list(voice.synthesize_ssml('<speak><prosody range="x-high">wide</prosody></speak>', RAW))
    assert sent[0]["pitch_range"] == pytest.approx(2.0)
    voice, sent = _voice(VitsAdapter(), None)
    list(voice.synthesize_ssml('<speak><prosody range="x-high">wide</prosody></speak>', RAW))
    assert not {"pitch_range", "pitch_mul"} & set(sent[0])
    assert shifts == []


def test_range_and_shift_together_compose_in_that_order(shifts):
    voice, sent = _voice(MixerTTSAdapter(), MIXER, STATS)
    list(voice.synthesize_ssml('<speak><prosody range="x-high" pitch="+12st">both</prosody></speak>', RAW))
    z = np.linspace(-2, 2, 9)
    plain = _denormalised(z, 1.0, 0.0, 213.1, 53.2)
    made = _denormalised(z, sent[0]["pitch_mul"], sent[0]["pitch_add"], 213.1, 53.2)
    assert made == pytest.approx(2 * (213.1 + 2 * (plain - 213.1)))
    assert shifts == []


def test_the_vocal_tract_stays_on_the_audio_and_the_prosody_pitch_goes_to_the_engine(shifts):
    voice, sent = _voice(TinyVocaAdapter(), TINY)
    list(voice.synthesize_ssml('<speak><amazon:effect vocal-tract-length="-20%">'
                               '<prosody pitch="+2st">words</prosody></amazon:effect></speak>', RAW))
    (piece,) = parse_ssml('<speak><amazon:effect vocal-tract-length="-20%">'
                          '<prosody pitch="+2st">words</prosody></amazon:effect></speak>')
    assert sent[0]["pitch_shift"] == pytest.approx(2 * SEMITONE)
    assert shifts == [pytest.approx(piece.tract)]
    shifts.clear()
    list(voice.synthesize_ssml('<speak><amazon:effect vocal-tract-length="-20%">words</amazon:effect></speak>', RAW))
    assert sent[-1]["pitch_shift"] == 0.0 and shifts == [pytest.approx(piece.tract)]


def test_the_engine_is_never_asked_for_more_than_12_semitones(shifts):
    voice, sent = _voice(TinyVocaAdapter(), TINY)
    list(voice.synthesize_ssml('<speak><amazon:effect vocal-tract-length="-50%">'
                               '<prosody pitch="-24st">x</prosody></amazon:effect></speak>', RAW))
    assert sent[0]["pitch_shift"] == pytest.approx(-12 * SEMITONE)


def test_a_pitch_the_call_sets_is_shifted_from_where_it_is(shifts):
    voice, sent = _voice(TinyVocaAdapter(), TINY, {"pitch_shift": 0.05})
    call = SynthesisConfig(extra={"pitch_shift": 0.2}, audio=AudioPost(normalize=False))
    list(voice.synthesize_ssml('<speak><prosody pitch="-3st">down</prosody> plain</speak>', call))
    assert sent[0]["pitch_shift"] == pytest.approx(0.2 - 3 * SEMITONE)
    assert sent[1]["pitch_shift"] == pytest.approx(0.2)
