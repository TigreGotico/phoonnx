"""SSML read into pieces a voice can speak, and synthesized in order."""
import io
import wave
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from phoonnx.config import SynthesisConfig
from phoonnx.ssml import SUPPORTED_TAGS, Piece, looks_like_ssml, parse_ssml, strip_ssml
from phoonnx.voice import AudioChunk, TTSVoice

# -- reading the markup -------------------------------------------------------------------


def test_a_break_before_the_first_words_is_kept():
    got = parse_ssml('<speak><break time="750ms"/>Once upon a time there lived a man.</speak>', "en-US")
    assert [(p.text, p.pause_ms) for p in got] == [("Once upon a time there lived a man.", 750)]
    got = parse_ssml('<speak>but on one condition only <break time="200ms"/> that you give me the child.</speak>')
    assert [(p.text, p.pause_ms) for p in got] == [
        ("but on one condition only", 0), ("that you give me the child.", 200)]


def test_an_opening_paragraph_adds_no_pause_but_a_break_inside_it_does():
    assert parse_ssml("<speak><p>Il était une fois.</p></speak>", "fr-FR")[0].pause_ms == 0
    assert parse_ssml('<speak><p><break time="600ms"/>Il était une fois.</p></speak>')[0].pause_ms == 600


def test_breaks_add_up_and_are_bounded():
    got = parse_ssml('<speak>a<break time="200ms"/><break time="300ms"/>b</speak>')
    assert got[1].pause_ms == 500
    got = parse_ssml('<speak>a<break time="99999s"/>b</speak>')
    assert got[1].pause_ms == 10_000
    got = parse_ssml('<speak>a<break strength="strong"/>b</speak>')
    assert got[1].pause_ms == 750


@pytest.mark.parametrize("markup, rate, volume", [
    ('<prosody rate="slow">x</prosody>', 0.8, 1.0),
    ('<prosody rate="+20%">x</prosody>', 1.2, 1.0),
    ('<prosody rate="150%">x</prosody>', 1.5, 1.0),
    ('<prosody volume="soft">x</prosody>', 1.0, 0.65),
    ('<prosody volume="-6dB">x</prosody>', 1.0, 0.501),
    ('<emphasis level="strong">x</emphasis>', 0.9, 1.15),
])
def test_prosody_and_emphasis_become_pace_and_level(markup, rate, volume):
    (piece,) = parse_ssml(f"<speak>{markup}</speak>")
    assert piece.rate == pytest.approx(rate, abs=0.01)
    assert piece.volume == pytest.approx(volume, abs=0.01)


def test_a_word_in_another_language_gets_that_language():
    got = parse_ssml('<speak>I say <lang xml:lang="fr-FR">bonjour</lang> often.</speak>', "en-US")
    assert [(p.text, p.lang) for p in got] == [("I say", "en-US"), ("bonjour", "fr-FR"), ("often.", "en-US")]


def test_say_as_sub_and_phoneme_become_words_the_voice_can_read():
    (piece,) = parse_ssml('<speak><say-as interpret-as="characters">XK7</say-as></speak>')
    assert piece.text == "X K 7"
    (piece,) = parse_ssml('<speak><say-as interpret-as="telephone">555-0199</say-as></speak>')
    assert piece.text == "5 5 5 0 1 9 9"
    (piece,) = parse_ssml('<speak><sub alias="World Wide Web">WWW</sub></speak>')
    assert piece.text == "World Wide Web"
    (piece,) = parse_ssml('<speak><phoneme alphabet="ipa" ph="həˈloʊ">hello</phoneme></speak>')
    assert piece.text == "[[ həˈloʊ ]]"


def test_what_a_voice_cannot_do_is_read_as_words_never_as_tags():
    (piece,) = parse_ssml('<speak><amazon:effect name="whispered">Secret</amazon:effect> words</speak>')
    assert piece.text == "Secret words"


def test_silent_text_is_not_said():
    assert parse_ssml('<speak><prosody volume="silent">hush</prosody></speak>') == []
    assert parse_ssml("<speak>  </speak>") == []
    assert parse_ssml("") == []


@pytest.mark.parametrize("broken", ["<speak>unclosed <break time='1s'/> words", "<speak><p>a</speak>",
                                    "<speak>a & b</speak>", "<speak><break time='1s'>x</speak>"])
def test_markup_that_does_not_parse_costs_its_pauses_not_the_reply(broken):
    got = parse_ssml(broken)
    assert got and all(piece.pause_ms == 0 for piece in got)
    assert "<" not in "".join(piece.text for piece in got)


@pytest.mark.parametrize("markup", [
    '<speak><break time="1.2.3s"/>x</speak>', '<speak><prosody rate="..%">x</prosody></speak>',
    '<speak><prosody volume="99999999999999999999dB">x</prosody></speak>',
])
def test_a_number_that_is_not_one_costs_its_markup_not_the_reply(markup):
    got = parse_ssml(markup)
    assert got and got[0].text == "x"


@pytest.mark.parametrize("text, expected", [
    ("<speak>Hello</speak>", True), ('He said <break time="1s"/> hi', True),
    ("a paragraph goes between <p> and </p>", False), ("2 < 3 and 5 > 4", False), ("", False), (None, False),
])
def test_only_real_markup_counts_as_ssml(text, expected):
    assert looks_like_ssml(text) is expected


def test_the_plain_text_is_what_a_screen_shows():
    assert strip_ssml('<speak>Hello <break time="1s"/> world <emphasis>now</emphasis>.</speak>') == "Hello world now."
    assert strip_ssml(None) == ""


def test_tags_are_stripped_in_linear_time():
    import time
    start = time.perf_counter()
    strip_ssml("<" * 200_000)
    assert time.perf_counter() - start < 2.0


def test_the_supported_tags_include_the_ones_the_reader_understands():
    assert {"speak", "break", "prosody", "emphasis", "say-as", "phoneme", "lang", "sub"} <= set(SUPPORTED_TAGS)


# -- synthesizing -----------------------------------------------------------------------------


class RecordingVoice(TTSVoice):
    """A voice whose synthesize() records the text and settings instead of running a model."""

    def __init__(self, lang_id_map=None):  # no model: skip TTSVoice.__init__
        self.config = SimpleNamespace(sample_rate=22050, length_scale=1.0, lang_code="en-US",
                                      lang_id_map=lang_id_map or {})
        self.calls = []

    def synthesize(self, text, syn_config=None, include_alignments=False):
        self.calls.append((text, syn_config))
        audio = np.full(2205, 0.1, dtype=np.float32)
        yield AudioChunk(sample_rate=22050, sample_width=2, sample_channels=1, audio_float_array=audio)


def test_pauses_become_silence_and_speech_keeps_its_order():
    voice = RecordingVoice()
    chunks = list(voice.synthesize_ssml('<speak>One.<break time="500ms"/>Two.</speak>'))
    assert [c.audio_float_array.max() > 0 for c in chunks] == [True, False, True]
    assert len(chunks[1].audio_float_array) == 11025
    assert [t for t, _ in voice.calls] == ["One.", "Two."]


def test_rate_divides_the_length_scale_and_volume_multiplies_the_samples_setting():
    voice = RecordingVoice()
    list(voice.synthesize_ssml('<speak><prosody rate="fast" volume="soft">quick</prosody></speak>',
                               SynthesisConfig(length_scale=1.2, volume=0.8)))
    (text, config), = voice.calls
    assert config.length_scale == pytest.approx(1.2 / 1.25)
    assert config.volume == pytest.approx(0.8 * 0.65)


def test_a_piece_at_the_ordinary_pace_leaves_the_configuration_alone():
    voice = RecordingVoice()
    base = SynthesisConfig(length_scale=1.1)
    list(voice.synthesize_ssml("<speak>plain words</speak>", base))
    assert voice.calls[0][1] is base


def test_a_multilingual_voice_gets_the_language_id_and_a_monolingual_one_does_not():
    multi = RecordingVoice({"en": 0, "fr": 1})
    list(multi.synthesize_ssml('<speak>Say <lang xml:lang="fr-FR">bonjour</lang>.</speak>'))
    # the English piece gets English's id, the French word French's; the lone "." is dropped
    assert [(t, c.lang_id) for t, c in multi.calls] == [("Say", 0), ("bonjour", 1)]
    mono = RecordingVoice()
    list(mono.synthesize_ssml('<speak>Say <lang xml:lang="fr-FR">bonjour</lang>.</speak>'))
    assert all(getattr(c, "lang_id", None) is None for _, c in mono.calls)


def test_the_wav_has_one_format_and_the_pauses_in_it():
    voice = RecordingVoice()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        voice.synthesize_ssml_wav('<speak>a<break time="1s"/>b</speak>', out)
    with wave.open(io.BytesIO(buffer.getvalue())) as handle:
        assert handle.getframerate() == 22050 and handle.getnchannels() == 1
        assert handle.getnframes() == 2205 * 2 + 22050


def test_plain_text_given_to_the_ssml_path_is_still_spoken():
    voice = RecordingVoice()
    list(voice.synthesize_ssml("just words, no markup"))
    assert [t for t, _ in voice.calls] == ["just words, no markup"]
