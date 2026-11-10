"""SSML edge cases: odd XML, hostile and huge input, and what streaming does with the audio."""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from phoonnx import audio_fx
from phoonnx.config import SynthesisConfig
from phoonnx.ssml import looks_like_ssml, parse_ssml, strip_ssml
from phoonnx.voice import AudioChunk, TTSVoice

SR = 22050


def texts(ssml: str, lang: str = "en") -> list[str]:
    return [piece.text for piece in parse_ssml(ssml, lang) if piece.text]


# -- XML -------------------------------------------------------------------------------------


@pytest.mark.parametrize("ssml,expected", [
    ("<speak>Tom &amp; Jerry &lt;3 &#65;&#x42;</speak>", "Tom & Jerry <3 AB"),
    ("<speak>a&nbsp;b caf&eacute; &hellip;</speak>", "a b café …"),
    ("<speak><![CDATA[1 < 2 && 3 > 2]]></speak>", "1 < 2 && 3 > 2"),
    ("<speak>AT&T and R&D</speak>", "AT&T and R&D"),
    ("<speak>1 < 2 and 3 > 2</speak>", "1 < 2 and 3 > 2"),
    ("<speak>a &bogus; b</speak>", "a &bogus; b"),
    ("<speak>a &#0; b</speak>", "a b"),
    ("<speak>a\x00b\x0bc</speak>", "abc"),
    ("<speak>hello <!-- a comment with <speak> in it --> world</speak>", "hello world"),
    ('<?xml version="1.0" encoding="UTF-8"?><speak>declared</speak>', "declared"),
    ("<speak>one</speak> trailing words", "one trailing words"),
    ("<speak>one</speak><speak>two</speak>", "one two"),
    ('<speak><sub alias="a > b">x</sub></speak>', "a > b"),
])
def test_what_a_person_writes_and_xml_refuses_is_still_read_with_its_markup(ssml, expected):
    assert " ".join(texts(ssml)) == expected


def test_a_bare_ampersand_costs_nothing_but_the_ampersand():
    pieces = parse_ssml("<speak>AT&T<break time='1s'/>R&D</speak>")
    assert [(p.text, p.pause_ms) for p in pieces] == [("AT&T", 0), ("R&D", 1000)]


def test_a_doctype_with_entities_is_dropped_not_expanded():
    bomb = ('<!DOCTYPE speak [<!ENTITY a "' + "x" * 1000 + '"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;&a;&a;">'
            '<!ENTITY c "&b;&b;&b;&b;&b;&b;&b;&b;&b;&b;">]><speak>&c;</speak>')
    started = time.perf_counter()
    assert sum(len(t) for t in texts(bomb)) < 100
    assert time.perf_counter() - started < 1


@pytest.mark.parametrize("ssml", [
    '<speak xmlns:amazon="urn:x"><amazon:effect name="whispered">a</amazon:effect></speak>',   # declared
    '<speak><amazon:effect name="whispered">a</amazon:effect></speak>',                         # undeclared
    '<speak><AMAZON:EFFECT name="whispered">a</AMAZON:EFFECT></speak>',                         # upper case
    '<SPEAK><Amazon:Effect name="whispered">a</Amazon:Effect></SPEAK>',
])
def test_a_namespace_prefix_works_declared_or_not(ssml):
    (piece,) = parse_ssml(ssml)
    assert piece.effects == ("whisper",)


def test_a_prefixed_attribute_does_not_cost_the_document_its_markup():
    pieces = parse_ssml('<speak>a<break time="1s"/><prosody amazon:max-duration="2s" rate="fast">b</prosody></speak>')
    assert [(p.text, p.pause_ms, p.rate) for p in pieces] == [("a", 0, 1.0), ("b", 1000, 1.25)]


@pytest.mark.parametrize("ssml", [
    "<speak>a <foo>b</foo> c</speak>",
    "<speak>a <foo:bar x='1'>b</foo:bar> c <baz/></speak>",
    "<speak>a <Foo-Bar>b</Foo-Bar> c</speak>",
])
def test_an_unknown_tag_is_never_read_aloud_and_its_text_stays(ssml):
    (text,) = texts(ssml)
    assert text == "a b c" and "<" not in text and "foo" not in text.lower()


@pytest.mark.parametrize("broken", [
    "<speak>unclosed <break time='1s'/> words", "<speak><p>a</speak>", "<speak><b>x</i></speak>",
    "<speak><break time='1s'>x</speak>", "<speak a=1>x</speak>", "<speak a='1' a='2'>x</speak>",
    "<speak>" + "<s>" * 500 + "x", "</speak>x<speak>", "<speak attr=>x</speak>",
])
def test_malformed_markup_is_read_as_its_text_and_never_raises(broken):
    got = parse_ssml(broken)
    assert got and all(piece.pause_ms == 0 for piece in got)
    assert "<speak" not in "".join(piece.text for piece in got)


def test_a_text_that_loses_its_tags_loses_nothing_else():
    assert strip_ssml("<speak>1 < 2 <break/> done <![CDATA[a < b]]></speak>") == "1 < 2 done a < b"


@pytest.mark.parametrize("ssml", ["", "   ", "\n\t ", "<speak></speak>", "<speak> </speak>", "<speak/>",
                                  "<speak><break time='1s'/></speak>", "<speak><p></p></speak>",
                                  "<speak><prosody rate='fast'/></speak>", "<speak><![CDATA[]]></speak>", None])
def test_nothing_to_say_gives_no_pieces(ssml):
    assert parse_ssml(ssml) == []


def test_emoji_and_right_to_left_scripts_survive_whole():
    assert texts("<speak>😀 hello 👍🏽 🇫🇷</speak>") == ["😀 hello 👍🏽 🇫🇷"]
    assert texts("<speak>مرحبا <break time='1s'/> שלום עולם</speak>") == ["مرحبا", "שלום עולם"]
    (piece,) = parse_ssml('<speak><lang xml:lang="ar-SA">مرحبا بالعالم</lang></speak>')
    assert (piece.text, piece.lang) == ("مرحبا بالعالم", "ar-SA")
    assert texts("<speak>a&#x1F600;b</speak>") == ["a😀b"]


def test_a_document_that_only_looks_like_html_is_still_words():
    assert not looks_like_ssml("a paragraph goes between <p> and </p>")
    assert looks_like_ssml('<ssml:speak xmlns:ssml="x">a</ssml:speak>')
    assert looks_like_ssml("<amazon:effect name='drc'>a</amazon:effect>")
    assert looks_like_ssml("<alexa:name type='first'/>")


# -- bounded work ----------------------------------------------------------------------------


@pytest.mark.parametrize("name,ssml", [
    ("five megabytes of words", "<speak>" + "word " * 1_000_000 + "</speak>"),
    ("a hundred thousand breaks", "<speak>" + 'a<break time="10s"/>' * 100_000 + "</speak>"),
    ("a hundred thousand marks", "<speak>" + '<mark name="m"/>a ' * 100_000 + "</speak>"),
    ("5000 nested elements", "<speak>" + "<emphasis>" * 5000 + "x" + "</emphasis>" * 5000 + "</speak>"),
    ("a hundred thousand '<'", "<speak>" + "<" * 100_000 + "</speak>"),
    ("a hundred thousand '&'", "<speak>" + "&" * 100_000 + "</speak>"),
    ("open CDATA", "<speak>" + "<![CDATA[" * 100_000 + "</speak>"),
    ("open comments", "<speak>" + "<!--" * 100_000 + "</speak>"),
    ("open doctypes", "<speak>" + "<!DOCTYPE x [" * 100_000 + "</speak>"),
    ("open attribute quotes", "<speak>" + '<a "' * 100_000 + "</speak>"),
    ("open processing instructions", "<speak>" + "<?" * 100_000 + "</speak>"),
    ("unclosed tags", "<speak>" + "<s>" * 100_000 + "</speak>"),
    ("a megabyte attribute", "<speak><break time='" + "9" * 1_000_000 + "s'/>x</speak>"),
    ("a hundred thousand breaths", "<speak>" + "<amazon:breath/>" * 100_000 + "</speak>"),
    ("a hundred thousand entities", "<speak>" + "&amp;&nbsp;&#x41;&bogus;" * 100_000 + "</speak>"),
    ("mismatched tags", "<speak>" + "<b>x</i>" * 100_000 + "</speak>"),
])
def test_hostile_input_is_read_in_bounded_time(name, ssml):
    started = time.perf_counter()
    pieces = parse_ssml(ssml, "en")
    assert time.perf_counter() - started < 5, name
    assert sum(p.pause_ms for p in pieces) <= 120_000
    assert sum(p.sound[1] for p in pieces if p.sound) + sum(p.pause_ms for p in pieces) <= 120_000


def test_the_silence_of_one_document_has_a_budget_and_the_marks_survive_it():
    ssml = "<speak>" + 'a<break time="10s"/><mark name="m"/>' * 30 + "</speak>"
    pieces = parse_ssml(ssml)
    assert sum(p.pause_ms for p in pieces) == 120_000
    assert sum(len(p.marks) for p in pieces) == 30  # a mark is an event: the budget never drops one
    assert all(at <= p.pause_ms for p in pieces for at in p.mark_pauses)


def test_nesting_past_the_limit_is_read_as_text_and_nesting_within_it_is_not():
    deep = "<speak>" + "<emphasis>" * 150 + "x" + "</emphasis>" * 150 + "</speak>"
    assert texts(deep) == ["x"] and parse_ssml(deep)[0].rate == 1.0
    ok = "<speak>" + "<prosody rate='fast'>" * 20 + "x" + "</prosody>" * 20 + "</speak>"
    assert parse_ssml(ok)[0].rate == 2.0  # 1.25 ** 20, clamped to the top of the range


@pytest.mark.parametrize("markup,attribute,low,high", [
    ('<prosody rate="1000%">x</prosody>', "rate", 0.5, 2.0), ('<prosody rate="0%">x</prosody>', "rate", 0.5, 2.0),
    ('<prosody rate="-500%">x</prosody>', "rate", 0.5, 2.0), ('<prosody rate="99999">x</prosody>', "rate", 0.5, 2.0),
    ('<prosody pitch="+1000st">x</prosody>', "pitch", -12, 12), ('<prosody pitch="-1000st">x</prosody>', "pitch", -12, 12),
    ('<prosody pitch="+99999%">x</prosody>', "pitch", -12, 12),
    ('<prosody volume="+1000dB">x</prosody>', "volume", 0, 2), ('<prosody volume="-1000dB">x</prosody>', "volume", 0, 2),
    ('<prosody volume="99999">x</prosody>', "volume", 0, 2),
    ('<amazon:effect vocal-tract-length="+9999%">x</amazon:effect>', "pitch", -12, 12),
    ('<amazon:effect vocal-tract-length="-9999%">x</amazon:effect>', "pitch", -12, 12),
])
def test_rate_pitch_and_volume_are_clamped(markup, attribute, low, high):
    (piece,) = parse_ssml(f"<speak>{markup}</speak>")
    assert low <= getattr(piece, attribute) <= high


# -- speech, in time ---------------------------------------------------------------------------


def burst(seconds: float, rate: int = SR, hz: float = 180.0, amplitude: float = 0.4) -> np.ndarray:
    """A voiced burst that starts and stops at zero, as a model's sentence does."""
    n = int(rate * seconds)
    t = np.arange(n) / rate
    wave = np.sin(2 * np.pi * hz * t) + 0.5 * np.sin(2 * np.pi * 2 * hz * t) + 0.25 * np.sin(2 * np.pi * 3 * hz * t)
    return (amplitude * wave / 1.75 * np.hanning(n)).astype(np.float32)


class FakeVoice(TTSVoice):
    """A voice with no model: each synthesize() call yields `sentences` bursts of `seconds`."""

    def __init__(self, rate: int = SR, seconds: float = 1.0, sentences: int = 1, edge: bool = False):
        self.config = SimpleNamespace(sample_rate=rate, length_scale=1.0, lang_code="en", lang_id_map={})
        self.rate, self.seconds, self.sentences, self.edge = rate, seconds, sentences, edge
        self.calls: list[str] = []

    def synthesize(self, text, syn_config=None, include_alignments=False):
        self.calls.append(text)
        for _ in range(self.sentences):
            audio = burst(self.seconds, self.rate)
            if self.edge:  # a chunk that begins and ends on a loud sample
                n = int(self.rate * self.seconds)
                audio = (0.3 + 0.1 * np.sin(2 * np.pi * 180 * np.arange(n) / self.rate)).astype(np.float32)
            yield AudioChunk(sample_rate=self.rate, sample_width=2, sample_channels=1, audio_float_array=audio)


def run(voice: FakeVoice, ssml: str, **kwargs):
    marks: list[tuple[str, float]] = []
    chunks = list(voice.synthesize_ssml(ssml, on_mark=lambda name, at: marks.append((name, at)), **kwargs))
    return chunks, marks


def stream(chunks) -> np.ndarray:
    return np.concatenate([c.audio_float_array for c in chunks]) if chunks else np.zeros(0, np.float32)


def test_a_mark_before_a_break_is_heard_before_the_silence_and_one_after_it_after():
    voice = FakeVoice(seconds=0.5)
    chunks, marks = run(voice, '<speak>one<mark name="a"/><break time="1s"/><mark name="b"/>'
                               '<break time="500ms"/><mark name="c"/>two<mark name="d"/></speak>')
    assert marks == [("a", pytest.approx(0.5)), ("b", pytest.approx(1.5)), ("c", pytest.approx(2.0)),
                     ("d", pytest.approx(2.5))]
    assert [round(len(c.audio_float_array) / SR, 3) for c in chunks] == [0.5, 1.0, 0.5, 0.5]


def test_marks_come_in_the_order_written_at_the_offset_of_the_audio_before_them():
    voice = FakeVoice(seconds=0.25)
    chunks, marks = run(voice, '<speak><mark name="start"/>a <mark name="m1"/>b <break time="300ms"/>'
                               '<mark name="m2"/>c<amazon:breath duration="short"/><mark name="m3"/>d</speak>')
    names = [name for name, _ in marks]
    assert names == ["start", "m1", "m2", "m3"]
    offsets = [at for _, at in marks]
    assert offsets == sorted(offsets) and offsets[0] == 0
    # a (0.25) | b (0.25) | 300 ms | c (0.25) | breath (0.25) | d
    assert offsets == pytest.approx([0.0, 0.25, 0.25 + 0.25 + 0.3, 0.25 + 0.25 + 0.3 + 0.25 + 0.25], abs=1e-3)


def test_a_mark_at_the_very_end_lands_after_a_trailing_break():
    chunks, marks = run(FakeVoice(seconds=0.5), '<speak>one<break time="2s"/><mark name="end"/></speak>')
    assert marks == [("end", pytest.approx(2.5))]
    assert len(stream(chunks)) == int(SR * 0.5) + 2 * SR


def test_marks_before_any_speech_fire_at_zero_and_a_document_of_marks_still_fires_them():
    chunks, marks = run(FakeVoice(), '<speak><mark name="a"/><mark name="b"/></speak>')
    assert marks == [("a", 0.0), ("b", 0.0)] and chunks == []


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 44100, 48000])
@pytest.mark.parametrize("ms", [1, 7, 33, 100, 333, 1234, 2501, 9999])
def test_a_pause_is_exact_to_the_sample(rate, ms):
    voice = FakeVoice(rate=rate, seconds=0.1)
    chunks, _ = run(voice, f'<speak>a<break time="{ms}ms"/>b<amazon:breath duration="short"/></speak>')
    assert [len(c.audio_float_array) for c in chunks] == [rate // 10, rate * ms // 1000, rate // 10, rate * 250 // 1000]
    assert all(c.sample_rate == rate for c in chunks)


def test_a_mark_inside_a_pause_splits_it_without_gaining_or_losing_a_sample():
    for rate in (22050, 44100):
        chunks, marks = run(FakeVoice(rate=rate, seconds=0.1), '<speak>a<break time="333ms"/><mark name="m"/>'
                                                              '<break time="333ms"/>b</speak>')
        assert sum(len(c.audio_float_array) for c in chunks) == rate // 10 * 2 + rate * 666 // 1000
        assert marks == [("m", pytest.approx(0.1 + 0.333, abs=1 / rate))]


@pytest.mark.parametrize("ssml", [
    "<speak>one. <prosody pitch='+4st'>two.</prosody> <prosody pitch='-4st'>three.</prosody></speak>",
    "<speak>one. <amazon:effect name='whispered'>two.</amazon:effect> three.</speak>",
    "<speak>one. <amazon:effect phonation='soft'>two.</amazon:effect> three.</speak>",
    "<speak>one. <amazon:effect name='drc'>two.</amazon:effect> three.</speak>",
    "<speak><amazon:auto-breaths frequency='x-high'>one. two. three.</amazon:auto-breaths></speak>",
    "<speak>one. <say-as interpret-as='expletive'>x</say-as> two. <break time='200ms'/> three.</speak>",
    "<speak><prosody duration='1500ms'>one. two.</prosody> <prosody amazon:max-duration='400ms'>three.</prosody></speak>",
    "<speak><prosody rate='fast' volume='loud'>one.</prosody><break time='1ms'/>two.</speak>",
])
@pytest.mark.parametrize("edge", [False, True])
def test_there_is_no_click_where_speech_meets_silence_or_a_sound(ssml, edge):
    voice = FakeVoice(seconds=0.4, sentences=2, edge=edge)
    chunks, _ = run(voice, ssml)
    audio = stream(chunks)
    joins = np.cumsum([len(c.audio_float_array) for c in chunks])[:-1]
    step = np.abs(np.diff(audio))
    # A voice at 180 Hz and this level moves by at most ~0.04 between two samples; a click is a jump.
    for join in joins:
        around = step[max(0, join - 2): join + 1]
        assert around.max() < 0.05, (int(join), float(around.max()))


def test_a_chunk_that_stops_on_a_loud_sample_is_faded_and_a_quiet_one_is_left_alone():
    loud = np.full(SR // 10, 0.3, dtype=np.float32)
    faded = audio_fx.declick(loud, SR)
    assert abs(faded[0]) < 0.01 and abs(faded[-1]) < 0.01 and faded[SR // 20] == pytest.approx(0.3)
    assert len(faded) == len(loud) and np.all(loud == 0.3)  # the caller's array is not touched
    quiet = burst(0.1)
    assert audio_fx.declick(quiet, SR) is not None and np.array_equal(audio_fx.declick(quiet, SR), quiet)
    assert audio_fx.declick(np.zeros(0, np.float32), SR).size == 0
    assert audio_fx.declick(np.array([0.9], np.float32), SR).size == 1


def test_speech_is_not_synthesized_ahead_of_the_listener():
    voice = FakeVoice(seconds=0.2)
    stream_ = voice.synthesize_ssml("<speak>one. <break time='1s'/> two. <break time='1s'/> three.</speak>")
    next(stream_)
    assert voice.calls == ["one."]
    next(stream_)  # the pause
    assert voice.calls == ["one."]
    next(stream_)
    assert voice.calls == ["one.", "two."]


def test_the_same_markup_and_seed_give_the_same_samples():
    ssml = ("<speak><amazon:auto-breaths frequency='x-high'>one. two. three.</amazon:auto-breaths>"
            "<amazon:breath/><amazon:effect name='whispered'>quiet words.</amazon:effect></speak>")

    def samples(seed):
        return stream(run(FakeVoice(seconds=0.3, sentences=3), ssml, seed=seed)[0])

    assert np.array_equal(samples(5), samples(5))
    assert not np.array_equal(samples(5), samples(6))


def test_the_same_markup_without_noise_is_the_same_every_time():
    ssml = "<speak><prosody pitch='+3st'>one.</prosody><prosody amazon:max-duration='500ms'>two.</prosody></speak>"
    assert np.array_equal(stream(run(FakeVoice(), ssml)[0]), stream(run(FakeVoice(), ssml)[0]))


@pytest.mark.parametrize("ssml,seconds", [
    ('<speak><prosody duration="2s">words</prosody></speak>', 2.0),
    ('<speak><prosody duration="600ms">words</prosody></speak>', 0.6),
    ('<speak><prosody amazon:max-duration="700ms">words</prosody></speak>', 0.7),
    ('<speak><prosody duration="1500ms" amazon:max-duration="1s">words</prosody></speak>', 1.0),
    ('<speak><prosody amazon:max-duration="5s">words</prosody></speak>', 1.0),   # a limit is not a target
    ('<speak><prosody duration="30s">words</prosody></speak>', 2.0),             # stretching stops at 2x
    ('<speak><prosody duration="10ms">words</prosody></speak>', 0.5),            # and squeezing at 0.5x
])
def test_duration_stretches_the_speech_to_the_time_asked_within_a_factor_of_two(ssml, seconds):
    chunks, _ = run(FakeVoice(seconds=1.0), ssml)
    assert len(stream(chunks)) / SR == pytest.approx(seconds, rel=0.03)


def test_a_duration_is_shared_by_the_pieces_in_it_and_leaves_its_pauses_alone():
    ssml = '<speak><prosody duration="4s">aaaa<break time="1s"/>bbbb</prosody></speak>'
    chunks, _ = run(FakeVoice(seconds=1.0), ssml)
    assert [round(len(c.audio_float_array) / SR, 2) for c in chunks] == [1.5, 1.0, 1.5]


def test_an_inner_duration_wins_over_the_outer_one():
    pieces = parse_ssml('<speak><prosody duration="10s"><prosody duration="1s">ab</prosody>cdcd</prosody></speak>')
    assert [(p.text, p.duration_ms) for p in pieces] == [("ab", 1000), ("cdcd", 9000)]


def test_a_pitch_shifted_and_stretched_piece_keeps_its_effects_order_and_length():
    chunks, _ = run(FakeVoice(seconds=1.0), '<speak><prosody pitch="+3st" duration="1500ms">words</prosody></speak>')
    assert len(stream(chunks)) / SR == pytest.approx(1.5, rel=0.03)


def test_a_piece_with_no_speech_still_pays_its_pause_and_marks():
    class Silent(FakeVoice):
        def synthesize(self, text, syn_config=None, include_alignments=False):
            return iter(())

    chunks, marks = run(Silent(), '<speak>a<break time="1s"/><mark name="m"/>b</speak>')
    assert marks == [("m", 0.0)] or [name for name, _ in marks] == ["m"]
    assert sum(len(c.audio_float_array) for c in chunks) == SR


def test_the_wav_writer_passes_the_seed_on():
    import io
    import wave

    ssml = "<speak><amazon:breath/>a</speak>"
    outputs = []
    for _ in range(2):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out:
            FakeVoice(seconds=0.1).synthesize_ssml_wav(ssml, out, seed=3)
        outputs.append(buffer.getvalue())
    assert outputs[0] == outputs[1]


def test_the_length_scale_for_a_rate_is_never_outside_the_clamp():
    voice = FakeVoice(seconds=0.1)
    configs: list[float] = []
    original = voice.synthesize

    def spy(text, syn_config=None, include_alignments=False):
        configs.append(syn_config.length_scale)
        return original(text, syn_config, include_alignments)

    voice.synthesize = spy
    run(voice, '<speak><prosody rate="99999%">a</prosody><prosody rate="0.0001%">b</prosody></speak>',
        syn_config=SynthesisConfig(length_scale=1.0))
    assert configs == [pytest.approx(0.5), pytest.approx(2.0)]
