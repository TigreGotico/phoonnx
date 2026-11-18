"""SSML ``<say-as>`` reads through the voice's text frontend, and so through text2spoken.

Governing text, ``concepts/text2spoken-design.md`` section 17.8: "phoonnx keeps the markup and
the audio, and holds no reading rule." ``TextFrontend.say_as`` "calls the same hook" as the rest
of the text, and an ``expletive`` is a bleep that "does not call text2spoken".

Each expected reading is text2spoken called directly with ``say_as=SayAs(...)``, or a row of the
design's own table (17.9) with its cited source.
"""
import json
import types
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import text2spoken
from scriptconv.frontend import FrontendConfig, TextFrontend
from scriptconv.phonemizers.dizione import DizionePhonemizer
from text2spoken import SayAs

import phoonnx.voice
from phoonnx.config import Alphabet, PhonemeType, SynthesisConfig, VoiceConfig
from phoonnx.ssml import parse_ssml
from phoonnx.voice import TTSVoice


def _voice(lang="en-US"):
    config = VoiceConfig(num_symbols=1, num_speakers=1, num_langs=1, sample_rate=22050,
                         lang_code=lang, phoneme_type=PhonemeType.UNICODE, alphabet=Alphabet.UNICODE,
                         phonemizer_model=None, add_diacritics=False, keep_punctuation=False)
    voice = TTSVoice(config=config, session=MagicMock(), adapter=MagicMock())
    voice.config.tokenizer = types.SimpleNamespace(tokenize=lambda phonemes: list(range(len(phonemes))))
    return voice


def _say_as(markup, lang="en-US"):
    """The text of the one piece that ``<speak>markup</speak>`` gives."""
    (piece,) = parse_ssml(f"<speak>{markup}</speak>", lang, _voice().frontend)
    return piece.text


# Three inputs or more for each value that phoonnx read before, in en-US and pt-PT.
ROWS = [
    *[("cardinal", None, "en-US", t) for t in ("21", "1,234", "-0.5")],
    *[("cardinal", None, "pt-PT", t) for t in ("21", "1.234", "-0,5")],
    *[("number", None, "en-US", t) for t in ("42", "3.14", "1000")],
    *[("number", None, "pt-PT", t) for t in ("42", "3,14", "1000")],
    *[("ordinal", None, lang, t) for lang in ("en-US", "pt-PT") for t in ("3", "21", "123")],
    *[("digits", None, lang, t) for lang in ("en-US", "pt-PT") for t in ("2024", "007", "12a3")],
    *[("telephone", None, "en-US", t) for t in ("555-0199", "+1 212 398 1900", "3981900")],
    *[("telephone", None, "pt-PT", t) for t in ("912 345 678", "+351 212 345 678", "112")],
    *[(kind, None, lang, t) for kind in ("characters", "spell-out", "letters", "verbatim")
      for lang in ("en-US", "pt-PT") for t in ("XK7", "abc", "b2")],
    *[("fraction", None, lang, t) for lang in ("en-US", "pt-PT") for t in ("1/2", "3/4", "3/2")],
    *[("unit", None, "en-US", t) for t in ("5 km", "10 kg", "3.5 m")],
    *[("unit", None, "pt-PT", t) for t in ("5 km", "10 kg", "3,5 m")],
    ("date", None, "en-US", "2026-11-09"), ("date", "mdy", "en-US", "09/21/2001"),
    ("date", "dmy", "en-US", "01/02/1960"), ("date", "y", "en-US", "2024"),
    ("date", None, "pt-PT", "2026-11-09"), ("date", "dmy", "pt-PT", "21/09/2001"),
    ("date", None, "pt-PT", "21/09/2001"),
    ("time", "hms24", "en-US", "14:30"), ("time", None, "en-US", "9:21 PM"), ("time", "hms24", "en-US", "00:01"),
    ("time", "hms24", "pt-PT", "14:30"), ("time", None, "pt-PT", "9:05"), ("time", "hms24", "pt-PT", "00:01"),
    *[("currency", None, "en-US", t) for t in ("$12.50", "12.50 USD", "€3")],
    *[("currency", None, "pt-PT", t) for t in ("12,50 €", "€3", "1,00 EUR")],
    *[("address", None, "en-US", t) for t in ("221B Baker St.", "1600 Pennsylvania Ave", "10 Downing St")],
    *[("address", None, "pt-PT", t) for t in ("Rua Augusta 221", "Av. da Liberdade 100", "R. do Ouro 3")],
    *[("interjection", None, "en-US", t) for t in ("wow", "ouch", "hello")],
    *[("interjection", None, "pt-PT", t) for t in ("olá", "ai", "uau")],
]


@pytest.mark.parametrize("interpret_as, fmt, lang, text", ROWS)
def test_a_say_as_reads_as_text2spoken_reads_it(interpret_as, fmt, lang, text):
    fmt_attr = f' format="{fmt}"' if fmt else ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = text2spoken.normalize(text, lang, say_as=SayAs(interpret_as=interpret_as, format=fmt))
        got = _say_as(f'<say-as interpret-as="{interpret_as}"{fmt_attr}>{text}</say-as>', lang)
    assert got == expected


@pytest.mark.parametrize("markup, expected", [
    # text2spoken-design.md 17.9, row SA01: W3C Note 3.6, "One hundred twenty third"
    ('<say-as interpret-as="ordinal">123</say-as>', "one hundred twenty-third"),
    # row SA04: Note 3.5, the leading '-'
    ('<say-as interpret-as="cardinal">-0.5</say-as>', "minus zero point five"),
    # row SA06: Note 3.1, "February first, 1960"
    ('<say-as interpret-as="date" format="dmy">01/02/1960</say-as>', "February first nineteen sixty"),
    # row SA13: Note 3.3, digit by digit
    ('<say-as interpret-as="telephone">3981900</say-as>', "three nine eight one nine zero zero"),
    # row SA14: Note 3.4, "double-you, three, c"
    ('<say-as interpret-as="characters">W3C</say-as>', "double-u three see"),
])
def test_cited_rows_read_through_a_voice(markup, expected):
    assert _say_as(markup) == expected


def test_the_language_in_force_reads_the_value():
    """``<lang>`` around a say-as on an en-US voice reads the value in that language."""
    got = _say_as('<lang xml:lang="pt-PT"><say-as interpret-as="ordinal">21</say-as></lang>')
    assert got == text2spoken.normalize("21", "pt-PT", say_as=SayAs(interpret_as="ordinal"))
    assert got != _say_as('<say-as interpret-as="ordinal">21</say-as>')


def test_the_say_as_goes_to_the_voice_hook_with_its_attributes():
    voice = _voice()
    calls = []

    def hook(text, lang, say_as=None):
        calls.append((text, lang, say_as))
        return "WORDS"

    voice.phonemizer.normalizer = hook
    (piece,) = parse_ssml('<speak>It is <say-as interpret-as="date" format="mdy" detail="1">'
                          '09/21/2001</say-as>.</speak>', "en-US", voice.frontend)
    assert calls == [("09/21/2001", "en-US", SayAs(interpret_as="date", format="mdy", detail="1"))]
    assert piece.text == "It is WORDS."


def test_synthesize_ssml_passes_the_voice_frontend(monkeypatch):
    voice = _voice()
    seen = []
    monkeypatch.setattr(phoonnx.voice, "parse_ssml", lambda ssml, lang, frontend: seen.append(frontend) or [])
    list(voice.synthesize_ssml("<speak>hi</speak>"))
    assert seen == [voice.frontend]


def test_an_unsupported_value_gives_the_plain_reading_and_a_warning():
    """SSML 1.1 section 3.1.9: an unsupported value renders "as if no interpret-as value were specified"."""
    with pytest.warns(Warning, match="no-such-value"):
        got = _say_as('<say-as interpret-as="no-such-value">12 cats</say-as>')
    assert got == text2spoken.normalize("12 cats", "en-US")
    assert got == "twelve cats"


def test_a_text_with_no_such_value_gives_the_plain_reading():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = _say_as('<say-as interpret-as="date">soon</say-as>')
    assert got == "soon"


def test_an_expletive_is_a_bleep_and_does_not_call_the_hook():
    voice = _voice()
    calls = []
    voice.phonemizer.normalizer = lambda text, lang, say_as=None: calls.append(text) or text
    got = parse_ssml('<speak>oh <say-as interpret-as="expletive">darn</say-as> well</speak>', "en-US",
                     voice.frontend)
    assert [(p.text, p.sound) for p in got] == [("oh", None), ("", ("beep", 350, 1.0)), ("well", None)]
    assert calls == []


def test_with_no_frontend_a_say_as_reads_as_written(caplog):
    (piece,) = parse_ssml('<speak><say-as interpret-as="ordinal">3</say-as></speak>', "en-US")
    assert piece.text == "3"
    assert "needs a text frontend" in caplog.text


TINYVOCA = Path(__file__).parent / "fixtures" / "tinyvoca_config.json"
PLAIN = ["I have 3 cats.", "It is 2026-11-09, at 9:21 PM.", "Dr. Patel paid $12.99 for the book",
         "Call 555-0199 before the 21st."]


def _tinyvoca():
    """The dizione fixture voice: its backend owns the ``normalize`` stage."""
    adapter = MagicMock()
    adapter.default_params.return_value = {}
    adapter.synthesize.return_value = types.SimpleNamespace(audio=np.zeros(4, np.float32), extras={})
    config = VoiceConfig.from_dict(json.loads(TINYVOCA.read_text()))
    return TTSVoice(session=MagicMock(), config=config, adapter=adapter)


def _sent_ids(voice):
    ids = [i for call in voice.adapter.synthesize.call_args_list for i in call.args[0].phoneme_ids[0].tolist()]
    voice.adapter.synthesize.reset_mock()
    return ids


@pytest.mark.parametrize("interpret_as, text", [("ordinal", "3"), ("date", "2026-11-09"), ("characters", "XK7")])
def test_a_backend_that_owns_normalize_reads_a_say_as_through_the_hook(interpret_as, text):
    """text2spoken-design.md 17.8: for a backend that owns ``normalize``, "The hook still runs for the say-as"."""
    voice = _tinyvoca()
    assert "normalize" in voice.frontend.backend.owned_stages
    expected = text2spoken.normalize(text, "en-US", say_as=SayAs(interpret_as=interpret_as))
    assert expected != text
    (piece,) = parse_ssml(f'<speak><say-as interpret-as="{interpret_as}">{text}</say-as></speak>', "en-US",
                          voice.frontend)
    assert piece.text == expected
    list(voice.synthesize_ssml(f'<speak><say-as interpret-as="{interpret_as}">{text}</say-as></speak>',
                               SynthesisConfig(normalize_audio=False)))
    got = _sent_ids(voice)
    list(voice.synthesize(expected, SynthesisConfig(normalize_audio=False)))
    assert got == _sent_ids(voice)


@pytest.mark.parametrize("text", PLAIN)
def test_a_backend_that_owns_normalize_reads_plain_text_without_the_hook(text):
    """The hook reads only the say-as: dizione still reads the plain text with its own rules."""
    voice = _tinyvoca()
    own = TextFrontend(FrontendConfig(lang="en-US", normalizer=None, keep_punctuation=True),
                       backend=DizionePhonemizer(tokens=True))
    assert voice.frontend.phonemize(text) == own.phonemize(text)
    list(voice.synthesize(text, SynthesisConfig(normalize_audio=False)))
    assert _sent_ids(voice) == list(DizionePhonemizer().tokens(text).ids)


def test_a_very_long_letter_run_reads_in_bounded_memory():
    """A 20,000-letter run through the dizione voice finishes; it used to exhaust memory."""
    voice = _tinyvoca()
    list(voice.synthesize("abcdefghij" * 2000, SynthesisConfig(normalize_audio=False)))
    assert _sent_ids(voice)
