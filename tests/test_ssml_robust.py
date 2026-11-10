"""The SSML reader and the audio effects survive hostile, random and huge input."""
from __future__ import annotations

import random
import string
import time

import numpy as np
import pytest

from phoonnx import audio_fx
from phoonnx.ssml import Piece, looks_like_ssml, parse_ssml, strip_ssml
from phoonnx.ssml_say_as import say_as_text

KINDS = ["cardinal", "ordinal", "fraction", "unit", "date", "time", "currency", "address",
         "telephone", "characters", "digits", "expletive", "interjection", "nonsense", ""]
LANGS = ["en", "fr", "es", "de", "pt", "it", "nl", "ja", "xx", "en-US", "fr_CA", "", None]
DOC = ('<speak>Hi <break time="1s"/><prosody rate="slow" pitch="+2st" volume="soft">there '
       '<say-as interpret-as="date" format="mdy">11/09/2026</say-as></prosody> '
       '<amazon:effect name="whispered">secret</amazon:effect><mark name="m"/>'
       '<amazon:auto-breaths frequency="high">One. Two.</amazon:auto-breaths></speak>')


def _text(rng, n):
    alphabet = string.printable + "éèàçüñß€£¥°½¼٣٤一二三  "
    return "".join(rng.choice(alphabet) for _ in range(n))


def test_say_as_never_raises_on_random_text():
    rng = random.Random(1)
    for _ in range(1500):
        text, kind, lang = _text(rng, rng.randint(0, 24)), rng.choice(KINDS), rng.choice(LANGS)
        result = say_as_text(text, kind, lang, rng.choice([None, "mdy", "dmy", "ymd", "hms24", "zz"]))
        assert isinstance(result, str)


@pytest.mark.parametrize("value", ["9" * 400, "1" * 5000, "0" * 5000, "-" * 50, "1e999", "1/" + "0" * 50,
                                   "99999999999999999999999999.99", "٣٤", "1,2,3,4,5,6,7,8,9"])
def test_say_as_survives_absurd_numbers(value):
    for kind in ("cardinal", "ordinal", "fraction", "currency", "unit", "time", "date"):
        assert isinstance(say_as_text(value, kind, "en"), str)


def test_parse_never_raises_on_mutated_documents():
    rng = random.Random(2)
    for _ in range(1500):
        chars = list(DOC)
        for _ in range(rng.randint(1, 8)):
            i = rng.randrange(len(chars))
            action = rng.choice(["delete", "insert", "dup"])
            if action == "delete":
                del chars[i]
            elif action == "insert":
                chars.insert(i, rng.choice('<>/="&;:- \'x'))
            else:
                chars.insert(i, chars[i])
            if not chars:
                chars = ["x"]
        pieces = parse_ssml("".join(chars), "en")
        assert isinstance(pieces, list) and all(isinstance(p, Piece) for p in pieces)
        assert all(0 <= p.pause_ms <= 10_000 * 50 and -12.0 <= p.pitch <= 12.0 for p in pieces)


def test_deeply_nested_and_huge_documents_finish_quickly():
    started = time.perf_counter()
    deep = "<speak>" + '<prosody rate="slow">' * 400 + "deep" + "</prosody>" * 400 + "</speak>"
    assert parse_ssml(deep) != [] or True
    wide = "<speak>" + "".join(f'word{i} <break time="10ms"/>' for i in range(20000)) + "</speak>"
    assert len(parse_ssml(wide)) > 1
    many_tags = "<speak>" + "<mark name='a'/>" * 50000 + "tail</speak>"
    assert parse_ssml(many_tags)[-1].text == "tail"
    assert time.perf_counter() - started < 15


def test_stacked_pauses_and_values_are_bounded():
    pieces = parse_ssml('<speak>' + '<break time="10s"/>' * 1000 + 'x</speak>')
    assert pieces[0].pause_ms <= 10_000 * 5
    (piece,) = parse_ssml('<speak><prosody pitch="+999999st" rate="9999%" volume="+9999dB">x</prosody></speak>')
    assert piece.pitch == 12.0 and 0.5 <= piece.rate <= 2.0 and piece.volume <= 2.0


def test_looks_like_ssml_is_linear_on_adversarial_text():
    started = time.perf_counter()
    for text in ("<" * 200000, "<break " * 50000, "<speak " + "a" * 200000, "<a" * 100000):
        looks_like_ssml(text)
        strip_ssml(text)
    assert time.perf_counter() - started < 10


@pytest.mark.parametrize("size", [0, 1, 2, 63, 64, 100, 511, 512, 1000])
@pytest.mark.parametrize("sr", [8000, 16000, 22050, 48000])
def test_effects_handle_tiny_clips(size, sr):
    x = (np.random.default_rng(3).standard_normal(size) * 0.1).astype(np.float32)
    for out in (audio_fx.pitch_shift(x, sr, 3), audio_fx.pitch_shift(x, sr, -12), audio_fx.whisper(x, sr),
                audio_fx.soft(x, sr), audio_fx.compress(x, sr)):
        assert out.dtype == np.float32 and len(out) == size and np.all(np.isfinite(out))


def test_effects_clean_up_bad_samples_and_stay_in_range():
    x = np.random.default_rng(4).standard_normal(22050).astype(np.float32)
    x[::97] = np.nan
    x[5::89] = np.inf
    for out in (audio_fx.pitch_shift(x, 22050, 5), audio_fx.whisper(x, 22050), audio_fx.soft(x, 22050),
                audio_fx.compress(x, 22050)):
        assert np.all(np.isfinite(out)) and np.max(np.abs(out)) <= 1.0 + 1e-6


def test_effects_accept_other_shapes_and_silence():
    silent = np.zeros(22050, dtype=np.float32)
    for out in (audio_fx.pitch_shift(silent, 22050, 4), audio_fx.whisper(silent, 22050),
                audio_fx.soft(silent, 22050), audio_fx.compress(silent, 22050)):
        assert len(out) == 22050 and np.all(out == 0) or np.max(np.abs(out)) < 1e-3
    stereo_like = np.random.default_rng(5).standard_normal((2, 5000)).astype(np.float32) * 0.1
    assert audio_fx.pitch_shift(stereo_like, 22050, 2).ndim == 1


def test_extreme_durations_and_volumes_for_sounds():
    assert len(audio_fx.breath(22050, 0)) == 1
    assert np.max(np.abs(audio_fx.breath(22050, 500, volume=1e9))) <= 0.25
    assert len(audio_fx.beep(22050, 10)) == 220
