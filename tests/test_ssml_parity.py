"""SSML beyond pauses: pitch, marks, effects, breaths, X-SAMPA, and their audio.

tests/test_ssml_say_as.py holds ``<say-as>``.
"""
from __future__ import annotations

import numpy as np
import pytest

from phoonnx import audio_fx
from phoonnx.ssml import parse_ssml
from unittest.mock import MagicMock

from phoonnx.voice import AudioChunk, TTSVoice
from phoonnx.xsampa import xsampa_to_ipa

SR = 22050


@pytest.mark.parametrize("value,expected", [
    ("+2st", 2.0), ("-3st", -3.0), ("high", 3.0), ("x-low", -6.0), ("+100%", 12.0), ("-50%", -12.0),
    ("200Hz", 0.0), ("+40st", 12.0), ("nonsense", 0.0),
])
def test_prosody_pitch_in_semitones(value, expected):
    (piece,) = parse_ssml(f'<speak><prosody pitch="{value}">up</prosody></speak>')
    assert piece.pitch == pytest.approx(expected)


def test_pitch_nests_and_resets():
    got = parse_ssml('<speak><prosody pitch="+2st">a <prosody pitch="+1st">b</prosody></prosody> c</speak>')
    assert [(p.text, p.pitch) for p in got] == [("a", 2.0), ("b", 3.0), ("c", 0.0)]


def test_marks_ride_on_the_next_words_and_trailing_marks_stand_alone():
    got = parse_ssml('<speak>one <mark name="a"/>two <mark name="b"/>three <mark name="end"/></speak>')
    assert [(p.text, p.marks) for p in got] == [("one", ()), ("two", ("a",)), ("three", ("b",)), ("", ("end",))]


def test_amazon_effects_and_breaths():
    got = parse_ssml(
        '<speak><amazon:effect name="whispered">a</amazon:effect> '
        '<amazon:effect name="drc">b</amazon:effect> '
        '<amazon:effect phonation="soft" vocal-tract-length="+20%">c</amazon:effect> '
        '<amazon:auto-breaths frequency="high" duration="long">d</amazon:auto-breaths>'
        '<amazon:breath duration="short"/></speak>')
    assert got[0].effects == ("whisper",) and got[1].effects == ("drc",)
    assert got[2].effects == ("soft",) and got[2].pitch < 0  # a longer vocal tract is lower
    assert got[3].breaths == (1.0, 500, 2)
    assert got[4].sound == ("breath", 250, 1.0)


def test_expletive_is_a_bleep_and_not_the_word():
    got = parse_ssml('<speak>oh <say-as interpret-as="expletive">darn</say-as> well</speak>')
    assert [(p.text, p.sound) for p in got] == [("oh", None), ("", ("beep", 350, 1.0)), ("well", None)]


def test_xsampa_phoneme_becomes_ipa():
    assert xsampa_to_ipa('"tS{ts') == "ˈtʃæts"
    (piece,) = parse_ssml('<speak><phoneme alphabet="x-sampa" ph="s@&quot;lEkS@n">x</phoneme></speak>')
    assert piece.text == "[[ səˈlɛkʃən ]]"


def _f0(audio):
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(audio.size)))
    freqs = np.fft.rfftfreq(audio.size, 1 / SR)
    band = (freqs > 80) & (freqs < 500)
    return freqs[band][np.argmax(spectrum[band])]


def _voiced(seconds=1.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * (np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 400 * t)
                   + 0.25 * np.sin(2 * np.pi * 600 * t))).astype(np.float32)


@pytest.mark.parametrize("semitones", [3, -3, 6])
def test_pitch_shift_moves_the_pitch_and_keeps_the_length(semitones):
    x = _voiced()
    y = audio_fx.pitch_shift(x, SR, semitones)
    assert len(y) == len(x)
    assert _f0(y) == pytest.approx(200 * 2 ** (semitones / 12), rel=0.03)
    assert np.sqrt(np.mean(y ** 2)) == pytest.approx(np.sqrt(np.mean(x ** 2)), rel=0.1)


def test_whisper_is_quiet_and_has_nothing_below_the_voice():
    """A whisper is quieter than speech (it was as loud, and growled) and has no rumble."""
    x = _voiced()
    y = audio_fx.whisper(x, SR)
    assert len(y) == len(x) and not np.allclose(x, y)
    assert 20 * np.log10(np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(x ** 2))) == pytest.approx(-8.0, abs=0.5)
    quieter = audio_fx.whisper(x, SR, level_db=-14.0)
    assert np.sqrt(np.mean(quieter ** 2)) < 0.6 * np.sqrt(np.mean(y ** 2))
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    assert spectrum[freqs < 150].sum() < 0.01 * spectrum.sum()


def _periodicity(audio):
    """Strength of the best periodic peak between 70 and 400 Hz (1.0 is a pure tone)."""
    frame = audio[: int(SR * 0.04)] - np.mean(audio[: int(SR * 0.04)])
    ac = np.correlate(frame, frame, "full")[len(frame) - 1:]
    return float(np.max(ac[int(SR / 400): int(SR / 70)]) / ac[0])


def test_whisper_has_no_pitch_left():
    x = _voiced()
    assert _periodicity(x) > 0.8
    assert _periodicity(audio_fx.whisper(x, SR)) < 0.5


def test_soft_is_quieter_and_compress_narrows_the_range():
    x = _voiced()
    assert np.sqrt(np.mean(audio_fx.soft(x, SR) ** 2)) < 0.7 * np.sqrt(np.mean(x ** 2))
    mixed = np.concatenate([x[:SR // 2] * 0.05, x[:SR // 2]])
    out = audio_fx.compress(mixed, SR)
    rms = lambda a: np.sqrt(np.mean(a ** 2))  # noqa: E731
    assert rms(out[SR // 2:]) / rms(out[:SR // 2]) < rms(mixed[SR // 2:]) / rms(mixed[:SR // 2])


def test_a_breath_is_quiet_has_the_asked_length_and_is_never_the_same_twice():
    first, second = audio_fx.breath(SR, 350), audio_fx.breath(SR, 350)
    assert len(first) == int(SR * 0.35) and not np.allclose(first, second)
    assert np.allclose(audio_fx.breath(SR, 350, seed=3), audio_fx.breath(SR, 350, seed=3))
    level = 20 * np.log10(np.sqrt(np.mean(first ** 2)))
    assert level == pytest.approx(-34.0, abs=0.5) and np.max(np.abs(first)) < 0.15
    assert np.max(np.abs(audio_fx.breath(SR, 500, volume=1e9))) < 0.3          # volume is capped
    # It swells and fades: the middle is louder than either end.
    third = len(first) // 3
    assert np.sqrt(np.mean(first[third:2 * third] ** 2)) > 2 * np.sqrt(np.mean(first[:third // 4] ** 2))
    assert len(audio_fx.beep(SR, 200)) == int(SR * 0.2) and np.max(np.abs(audio_fx.beep(SR, 200))) <= 0.3


class _Config:
    sample_rate = SR
    length_scale = 1.0
    lang_code = "en"
    lang_id_map = None


class _Voice(TTSVoice):
    frontend = None  # no phonemizer: these documents hold no say-as but an expletive
    adapter = MagicMock()


def _voice():
    voice = _Voice.__new__(_Voice)
    voice.config = _Config()
    seen = []

    def synthesize(text, syn_config=None, include_alignments=False):
        seen.append(text)
        for _ in range(3):  # three "sentences" per call
            yield AudioChunk(sample_rate=SR, sample_width=2, sample_channels=1, audio_float_array=_voiced(0.2))

    voice.synthesize = synthesize
    return voice, seen


def test_synthesis_reports_marks_in_time_and_adds_sounds():
    voice, seen = _voice()
    marks = []
    chunks = list(voice.synthesize_ssml(
        '<speak>one <break time="200ms"/><mark name="m"/>two <say-as interpret-as="expletive">x</say-as></speak>',
        on_mark=lambda name, at: marks.append((name, round(at, 3)))))
    assert seen == ["one", "two"]
    # first piece: 3 x 0.2 s; then 0.2 s of silence; the mark falls where "two" begins
    assert marks == [("m", pytest.approx(0.8, abs=1e-3))]
    assert len(chunks[-1].audio_float_array) == int(SR * 0.35)  # the bleep


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 24000, 44100, 48000])
def test_a_breath_of_any_duration_has_exactly_that_many_samples(sample_rate):
    """The slow wobble is smoothed with a window the signal may be shorter than (CodeRabbit on #553)."""
    for duration_ms in (0, 1, 5, 20, 47, 60, 83, 100, 200, 350):
        length = max(1, int(sample_rate * duration_ms / 1000))
        out = audio_fx.breath(sample_rate, duration_ms, seed=3)
        assert out.shape == (length,), (sample_rate, duration_ms)
        assert out.dtype == np.float32 and np.all(np.isfinite(out))
        assert float(np.max(np.abs(out))) <= 1.0


def test_synthesis_applies_pitch_and_breaths_between_sentences():
    voice, _ = _voice()
    chunks = list(voice.synthesize_ssml(
        '<speak><amazon:auto-breaths frequency="x-high"><prosody pitch="+6st">hello</prosody></amazon:auto-breaths></speak>'))
    # 3 sentences, a breath before the 2nd and 3rd: speech, breath, speech, breath, speech
    assert [len(c.audio_float_array) for c in chunks].count(int(SR * 0.35)) == 2
    speech = chunks[0].audio_float_array
    assert _f0(speech) == pytest.approx(200 * 2 ** 0.5, rel=0.04)


# Expected values: the X-SAMPA chart, https://en.wikipedia.org/wiki/X-SAMPA
@pytest.mark.parametrize("xsampa,ipa", [
    ("r\\`", "ɻ"), ("r`", "ɽ"), ("L\\", "ʟ"), ("M\\", "ɰ"), ("H\\", "ʜ"), ("s\\", "ɕ"),
    ("z\\", "ʑ"), ("x\\", "ɧ"), ("v\\", "ʋ"), ("I\\", "ᵻ"), ("U\\", "ᵿ"), ("n_=", "n\u0329"),
    ('"x\\A:l', "ˈɧɑːl"),
])
def test_xsampa_symbols_follow_the_chart(xsampa, ipa):
    assert xsampa_to_ipa(xsampa) == ipa


def test_a_sound_before_the_first_words_is_made_at_the_rate_of_the_speech():
    """Voice conversion or super-resolution can move the speech off the model's own rate."""
    import io
    import wave

    rate = 44100  # the model's own rate is SR (22050)
    voice = _Voice.__new__(_Voice)
    voice.config = _Config()

    def synthesize(text, syn_config=None, include_alignments=False):
        yield AudioChunk(sample_rate=rate, sample_width=2, sample_channels=1,
                         audio_float_array=np.full(rate // 10, 0.1, dtype=np.float32))

    voice.synthesize = synthesize
    ssml = ('<speak><break time="100ms"/><amazon:breath duration="short"/><mark name="m"/>'
            '<say-as interpret-as="expletive">x</say-as>hello</speak>')
    marks = []
    chunks = list(voice.synthesize_ssml(ssml, on_mark=lambda name, at: marks.append((name, at))))
    # 100 ms pause, 250 ms breath, 350 ms bleep, then the words
    assert [c.sample_rate for c in chunks] == [rate] * 4
    assert [len(c.audio_float_array) for c in chunks] == [rate * ms // 1000 for ms in (100, 250, 350, 100)]
    assert marks == [("m", pytest.approx(0.35))]

    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        voice.synthesize_ssml_wav(ssml, wav)
    out.seek(0)
    with wave.open(out, "rb") as wav:
        assert wav.getframerate() == rate
        assert wav.getnframes() == sum(len(c.audio_float_array) for c in chunks)


def test_sounds_with_no_words_after_them_are_made_at_the_model_rate():
    voice, _ = _voice()
    chunks = list(voice.synthesize_ssml('<speak><amazon:breath duration="short"/></speak>'))
    assert [(c.sample_rate, len(c.audio_float_array)) for c in chunks] == [(SR, SR // 4)]
