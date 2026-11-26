"""``AudioPost.stretch_pace``: an engine that cannot change its pace follows the length scale by a time stretch.

The engines are recorded fakes, so no model runs. A fake codec engine makes the same
half second of a 220 Hz tone for every call, whatever the length scale, as Orpheus or
Llasa do; a fake VITS engine makes a length that follows the length scale itself.
"""
import numpy as np
import pytest

from phoonnx.config import AudioPost, SynthesisConfig, VoiceConfig
from phoonnx.engines import get_adapter, list_engines
from phoonnx.engines.base import AdapterSynthesisResult
from phoonnx.engines.vits import VitsAdapter
from phoonnx.voice import AudioChunk, TTSVoice, Utterance

RATE = 16000
SECONDS = 0.5
CONFIG = {"phoonnx_version": "1.0", "engine": "phoonnx", "phoneme_type": "unicode",
          "alphabet": "unicode", "lang_code": "en-US", "audio": {"sample_rate": RATE},
          "num_symbols": 8, "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a": [3]}}
ON = AudioPost(normalize=False, stretch_pace=True)
OFF = AudioPost(normalize=False)


def _tone(seconds=SECONDS):
    t = np.arange(int(RATE * seconds)) / RATE
    return (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class _Codec(VitsAdapter):
    """Makes the same audio whatever the length scale, and no pace key of its own."""

    PACE_CONTROL = False

    def default_params(self):
        return {"temperature": 0.7}

    def synthesize(self, request, session):
        return AdapterSynthesisResult(audio=_tone())

    def synthesize_stream(self, request, session):
        audio = _tone()
        yield AdapterSynthesisResult(audio=audio[: audio.size // 2])
        yield AdapterSynthesisResult(audio=audio[audio.size // 2:])


class _Vits(VitsAdapter):
    """Follows the length scale itself."""

    def synthesize(self, request, session):
        return AdapterSynthesisResult(audio=_tone(SECONDS * request.params["length_scale"]))


class _Voice(TTSVoice):
    frontend = None  # no phonemizer: the SSML documents here hold no say-as


def _voice(adapter, inference=None, engine_params=None):
    voice = _Voice.__new__(_Voice)  # no session or phonemizer: the engine call is faked
    config = dict(CONFIG, **({"inference": inference} if inference else {}),
                  **({"engine_params": engine_params} if engine_params else {}))
    voice.config, voice.adapter, voice.session = VoiceConfig.from_dict(config), adapter, None
    voice.model_path = None
    return voice


def _seconds(voice, config):
    chunk = voice.render(Utterance(text="a", symbols=("a",), ids=(3,)), config)
    return chunk.audio_float_array.size / chunk.sample_rate


def _pitch(audio):
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
    return np.fft.rfftfreq(audio.size, 1 / RATE)[int(np.argmax(spectrum))]


def test_off_by_default_a_codec_engine_ignores_the_length_scale():
    assert AudioPost().stretch_pace is False
    assert _seconds(_voice(_Codec()), SynthesisConfig(length_scale=2.0, audio=OFF)) == SECONDS


@pytest.mark.parametrize("length_scale", [0.6, 0.8, 1.25, 1.6])
def test_on_a_codec_engine_follows_the_length_scale(length_scale):
    seconds = _seconds(_voice(_Codec()), SynthesisConfig(length_scale=length_scale, audio=ON))
    assert seconds == pytest.approx(SECONDS * length_scale, abs=1 / RATE)


def test_the_stretch_keeps_the_pitch():
    chunk = _voice(_Codec()).render(Utterance(text="a", symbols=("a",), ids=(3,)),
                                    SynthesisConfig(length_scale=1.6, audio=ON))
    assert _pitch(chunk.audio_float_array) == pytest.approx(220, abs=5)


def test_an_engine_with_pace_control_is_not_stretched_twice():
    seconds = _seconds(_voice(_Vits()), SynthesisConfig(length_scale=1.6, audio=ON))
    assert seconds == pytest.approx(SECONDS * 1.6, abs=1 / RATE)


def test_no_call_length_scale_no_stretch():
    assert _seconds(_voice(_Codec()), SynthesisConfig(audio=ON)) == SECONDS


def test_the_stretch_is_relative_to_the_voices_own_length_scale():
    # a call that names the voice's own length_scale changes nothing
    voice = _voice(_Codec(), inference={"length_scale": 1.3})
    assert _seconds(voice, SynthesisConfig(length_scale=1.3, audio=ON)) == SECONDS
    assert _seconds(voice, SynthesisConfig(length_scale=2.6, audio=ON)) == pytest.approx(SECONDS * 2, abs=1 / RATE)
    voice = _voice(_Codec(), engine_params={"length_scale": 0.8})
    assert _seconds(voice, SynthesisConfig(length_scale=0.8, audio=ON)) == SECONDS


def test_a_length_scale_in_extra_is_the_calls():
    config = SynthesisConfig(length_scale=1.0, extra={"length_scale": 1.5}, audio=ON)
    assert _seconds(_voice(_Codec()), config) == pytest.approx(SECONDS * 1.5, abs=1 / RATE)


@pytest.mark.parametrize("length_scale, factor", [(5.0, 2.0), (0.1, 0.5)])
def test_the_stretch_stays_within_half_and_twice(length_scale, factor):
    seconds = _seconds(_voice(_Codec()), SynthesisConfig(length_scale=length_scale, audio=ON))
    assert seconds == pytest.approx(SECONDS * factor, abs=1 / RATE)


@pytest.mark.parametrize("length_scale", ["fast", None, 0.0, float("nan")])
def test_a_length_scale_that_is_not_a_positive_number_does_not_stretch(length_scale):
    config = SynthesisConfig(extra={"length_scale": length_scale}, audio=ON)
    assert _seconds(_voice(_Codec()), config) == SECONDS


def test_phoneme_samples_follow_the_stretched_audio():
    voice = _voice(_Codec())
    voice._infer = lambda ids, cfg, alignments=False, streams=None, text=None: (_tone(), np.array([3000, 5000]))
    chunk = voice.render(Utterance(text="a", symbols=("a",), ids=(3,)),
                         SynthesisConfig(length_scale=1.5, audio=ON))
    assert int(np.sum(chunk.phoneme_id_samples)) == chunk.audio_float_array.size
    assert list(chunk.phoneme_id_samples) == [4500, 7500]


def test_stream_yields_a_stretched_sentence_as_one_chunk():
    voice = _voice(_Codec())
    voice.prepare = lambda text, config: iter([Utterance(text="a", symbols=("a",), ids=(3,))])
    plain = list(voice.stream("a", SynthesisConfig(audio=ON)))
    slow = list(voice.stream("a", SynthesisConfig(length_scale=1.5, audio=ON)))
    assert len(plain) == 2, "without a stretch the engine's parts stream as they come"
    assert len(slow) == 1
    assert slow[0].audio_float_array.size / RATE == pytest.approx(SECONDS * 1.5, abs=1 / RATE)


def test_ssml_rate_reaches_a_codec_engine_as_a_stretch():
    voice = _voice(_Codec())

    def synthesize(text, syn_config=None, include_alignments=False):
        yield voice.render(Utterance(text=text, symbols=("a",), ids=(3,)), syn_config)

    voice.synthesize = synthesize
    chunks = list(voice.synthesize_ssml('<speak><prosody rate="x-slow">slow</prosody> plain</speak>',
                                        SynthesisConfig(audio=ON)))
    spoken = [c.audio_float_array.size / c.sample_rate for c in chunks
              if np.abs(c.audio_float_array).max() > 0.01]
    assert spoken == [pytest.approx(SECONDS / 0.6, abs=0.01), pytest.approx(SECONDS, abs=0.01)]


class _CodecWithDefault(_Codec):
    """A codec engine whose defaults carry a length scale (Shami's carry 1.0)."""

    def default_params(self):
        return {"length_scale": 1.25}


def test_the_engines_default_length_scale_is_the_voices_own():
    voice = _voice(_CodecWithDefault())
    assert _seconds(voice, SynthesisConfig(length_scale=1.25, audio=ON)) == SECONDS
    assert _seconds(voice, SynthesisConfig(length_scale=2.5, audio=ON)) == pytest.approx(SECONDS * 2, abs=1 / RATE)


def test_stream_keeps_streaming_when_the_call_names_the_voices_own_length_scale():
    voice = _voice(_Codec(), inference={"length_scale": 1.3})
    voice.prepare = lambda text, config: iter([Utterance(text="a", symbols=("a",), ids=(3,))])
    assert len(list(voice.stream("a", SynthesisConfig(length_scale=1.3, audio=ON)))) == 2


@pytest.mark.parametrize("shape, stretched", [((1, 8000), True), ((2, 4000), False)])
def test_only_one_channel_is_stretched(shape, stretched):
    voice = _voice(_Codec())
    voice._infer = lambda ids, cfg, alignments=False, streams=None, text=None: (
        np.resize(_tone(), shape).astype(np.float32), None)
    config = SynthesisConfig(length_scale=1.5, audio=ON)
    size = voice.render(Utterance(text="a", symbols=("a",), ids=(3,)), config).audio_float_array.size
    assert size == (int(8000 * 1.5) if stretched else 8000)


@pytest.mark.parametrize("inference, engine_params", [
    (None, {"length_scale": 1.2}),
    ({"length_scale": 1.3}, {"length_scale": 0.8}),
    ({"length_scale": 1.3}, None),
])
def test_ssml_rate_stretches_by_the_rate_whatever_the_voices_own_length_scale(inference, engine_params):
    voice = _voice(_Codec(), inference=inference, engine_params=engine_params)

    def synthesize(text, syn_config=None, include_alignments=False):
        yield voice.render(Utterance(text=text, symbols=("a",), ids=(3,)), syn_config)

    voice.synthesize = synthesize
    chunks = list(voice.synthesize_ssml('<speak><prosody rate="x-slow">slow</prosody> plain</speak>',
                                        SynthesisConfig(audio=ON)))
    spoken = [c.audio_float_array.size / c.sample_rate for c in chunks
              if np.abs(c.audio_float_array).max() > 0.01]
    assert spoken == [pytest.approx(SECONDS / 0.6, abs=0.01), pytest.approx(SECONDS, abs=0.01)]


NO_PACE = {"arktts", "chatterbox", "indic_parler", "llasa", "magpie", "mosstts", "neutts",
           "orpheus", "outetts", "pockettts", "qwen3tts", "sparktts", "shami"}


def test_the_engines_without_pace_control():
    without = {name for name in list_engines() if not get_adapter(name).PACE_CONTROL}
    classes = {type(get_adapter(name)).__module__.rsplit(".", 1)[-1] for name in without}
    assert classes == NO_PACE, "a new engine that cannot change its pace sets PACE_CONTROL = False"
    for name in without:
        adapter = get_adapter(name)
        assert type(adapter).length_scale_params is VitsAdapter.length_scale_params, (
            f"{name} has a speed key of its own, so it has pace control")
