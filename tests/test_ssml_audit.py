"""An audit of phoonnx's SSML against W3C SSML 1.1 and the Amazon Polly / Alexa extensions.

``MATRIX`` is the support table of ``docs/ssml.md``: one row per element or attribute, with the
status it has (``supported``, ``degraded`` -- read without its effect, never as tags -- or
``unsupported`` -- ignored on purpose) and a check of what the reader makes of it in English and in
French. ``test_the_docs_list_every_row`` keeps the page and the table the same.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from phoonnx.ssml import Piece, parse_ssml

SUPPORTED, DEGRADED, UNSUPPORTED = "supported", "degraded", "unsupported"
WORDS = {"en": "Hello world", "fr": "Bonjour le monde"}


def spoken(pieces: list[Piece]) -> str:
    """What is said: the text of each piece, joined with ' | '."""
    return " | ".join(piece.text for piece in pieces)


def pauses(pieces: list[Piece]) -> list[int]:
    return [piece.pause_ms for piece in pieces]


@dataclass(frozen=True)
class Row:
    """One element or attribute: how it is written, its status, and what reading it gives."""

    feature: str
    status: str
    ssml: str  # {t} is the words of the language
    check: Callable[[list[Piece], str, str], bool]  # (pieces, words, language) -> as expected


def only(text: str) -> Callable:
    """The reading is exactly `text`: `{t}` in it is the words of the language."""
    return lambda pieces, t, lang: spoken(pieces) == text.replace("{t}", t)


def by_language(**texts: str) -> Callable:
    return lambda pieces, t, lang: spoken(pieces) == texts[lang]


def field(name: str, value) -> Callable:
    return lambda pieces, t, lang: bool(pieces) and getattr(pieces[0], name) == value and spoken(pieces) == t


def say_as(kind: str, value: str, en: str, fr: str, status: str = SUPPORTED, extra: str = "") -> Row:
    return Row(f"say-as interpret-as={kind}" + (f" {extra}" if extra else ""), status,
               f'<speak><say-as interpret-as="{kind}"{" " + extra if extra else ""}>{value}</say-as></speak>',
               by_language(en=en, fr=fr))


MATRIX: list[Row] = [
    # -- structure ---------------------------------------------------------------------------
    Row("speak", SUPPORTED, "<speak>{t}</speak>", only("{t}")),
    Row("speak xml:lang", SUPPORTED, '<speak xml:lang="de-DE">{t}</speak>',
        lambda ps, t, lang: ps[0].lang == "de-DE" and spoken(ps) == t),
    Row("speak with the SSML namespace declared", SUPPORTED,
        '<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis">{t}<break time="1s"/>x</speak>',
        lambda ps, t, lang: spoken(ps) == f"{t} | x" and pauses(ps) == [0, 1000]),
    Row("a prefix declared as the SSML namespace", SUPPORTED,
        '<ssml:speak xmlns:ssml="http://www.w3.org/2001/10/synthesis">{t}<ssml:break time="1s"/>x</ssml:speak>',
        lambda ps, t, lang: spoken(ps) == f"{t} | x" and pauses(ps) == [0, 1000]),
    Row("an undeclared ssml prefix", SUPPORTED, '<speak>{t}<ssml:break time="1s"/>x</speak>',
        lambda ps, t, lang: spoken(ps) == f"{t} | x" and pauses(ps) == [0, 1000]),
    Row("lang xml:lang", SUPPORTED, '<speak>{t} <lang xml:lang="de-DE">Haus</lang></speak>',
        lambda ps, t, lang: [(p.text, p.lang) for p in ps] == [(t, lang), ("Haus", "de-DE")]),
    Row("xml:lang on p, s and w", SUPPORTED, '<speak><p xml:lang="es-ES">hola</p><s xml:lang="it-IT">ciao</s></speak>',
        lambda ps, t, lang: [(p.text, p.lang) for p in ps] == [("hola", "es-ES"), ("ciao", "it-IT")]),
    Row("p", SUPPORTED, "<speak><p>{t}</p><p>{t}</p></speak>",
        lambda ps, t, lang: pauses(ps) == [0, 750]),
    Row("s", SUPPORTED, "<speak><s>{t}</s><s>{t}</s></speak>",
        lambda ps, t, lang: pauses(ps) == [0, 150]),
    Row("w, token", SUPPORTED, "<speak><w>{t}</w> <token>x</token></speak>", only("{t} x")),
    Row("w role", DEGRADED, '<speak><w role="amazon:VB">{t}</w></speak>', only("{t}")),
    # -- say-as ------------------------------------------------------------------------------
    say_as("cardinal", "1234", "one thousand, two hundred and thirty four", "mille deux cent trente-quatre"),
    say_as("number", "12", "twelve", "douze"),
    say_as("ordinal", "3", "third", "troisième"),
    say_as("digits", "123", "1 2 3", "1 2 3"),
    say_as("fraction", "3/4", "three quarters", "trois quarts"),
    say_as("unit", "5km", "five kilometers", "cinq kilomètres"),
    say_as("date", "2026-11-09", "monday, november ninth, twenty twenty six",
           "lundi neuf novembre deux-mille-vingt-six"),
    say_as("date", "1/2/2026", "sunday, february first, twenty twenty six",
           "dimanche premier février deux-mille-vingt-six", extra='format="dmy"'),
    say_as("time", "14:30", "half past two P M", "quatorze heures trente"),
    say_as("time", "7:30 pm", "half past seven P M", "sept heures et demie du soir", extra='format="hms12"'),
    say_as("time", "9 pm", "nine P M", "neuf heures du soir", extra='detail="1"'),
    say_as("date", "9 novembre 2026", "9 novembre 2026", "lundi neuf novembre deux-mille-vingt-six", extra='detail="written"'),
    say_as("currency", "$12.50", "twelve dollars and fifty cents", "douze dollars et cinquante cents"),
    say_as("address", "12 Rue de la Paix", "twelve Rue de la Paix", "douze Rue de la Paix"),
    say_as("telephone", "555-0100", "5 5 5 0 1 0 0", "5 5 5 0 1 0 0"),
    say_as("characters", "XK7", "X K 7", "X K 7"),
    say_as("spell-out", "XK7", "X K 7", "X K 7"),
    say_as("verbatim", "XK7", "X K 7", "X K 7"),
    say_as("interjection", "wow", "wow", "wow"),
    say_as("an unknown type", "abc", "abc", "abc", DEGRADED),
    Row("say-as detail", DEGRADED, '<speak><say-as interpret-as="digits" detail="2">12</say-as></speak>',
        only("1 2")),
    Row("say-as expletive", SUPPORTED, "<speak>{t} <say-as interpret-as=\"expletive\">damn</say-as> {t}</speak>",
        lambda ps, t, lang: [(p.text, p.sound) for p in ps] == [(t, None), ("", ("beep", 350, 1.0)), (t, None)]),
    # -- pronunciation -----------------------------------------------------------------------
    Row("phoneme ipa", SUPPORTED, '<speak><phoneme alphabet="ipa" ph="bɔ̃ʒuʁ">bonjour</phoneme></speak>',
        only("[[ bɔ̃ʒuʁ ]]")),
    Row("phoneme x-sampa", SUPPORTED, """<speak><phoneme alphabet="x-sampa" ph='h@"l@U'>hello</phoneme></speak>""",
        only("[[ həˈləʊ ]]")),
    Row("phoneme another alphabet", DEGRADED,
        '<speak><phoneme alphabet="x-amazon-pinyin" ph="ni3">你</phoneme></speak>', only("你")),
    Row("phoneme without ph", DEGRADED, "<speak><phoneme>{t}</phoneme></speak>", only("{t}")),
    Row("sub alias", SUPPORTED, '<speak><sub alias="World Wide Web">WWW</sub></speak>', only("World Wide Web")),
    Row("sub with an empty alias", SUPPORTED, '<speak>a <sub alias="">gone</sub> b</speak>', only("a b")),
    Row("lookup", DEGRADED, '<speak><lookup ref="l">{t}</lookup></speak>', only("{t}")),
    Row("lexicon", UNSUPPORTED, '<speak><lexicon uri="http://x/l.pls" xml:id="l"/>{t}</speak>', only("{t}")),
    # -- breaks and emphasis -----------------------------------------------------------------
    Row("break time", SUPPORTED, '<speak>a<break time="1.5s"/>b<break time="200ms"/>c</speak>',
        lambda ps, t, lang: pauses(ps) == [0, 1500, 200]),
    Row("break strength", SUPPORTED, '<speak>a<break strength="strong"/>b<break strength="x-weak"/>c</speak>',
        lambda ps, t, lang: pauses(ps) == [0, 750, 100]),
    Row("break strength none", SUPPORTED, '<speak>a<break strength="none"/>b</speak>',
        lambda ps, t, lang: spoken(ps) == "a b" or pauses(ps) == [0, 0]),
    Row("break without attributes", SUPPORTED, "<speak>a<break/>b</speak>",
        lambda ps, t, lang: pauses(ps) == [0, 500]),
    Row("break time without a unit", DEGRADED, '<speak>a<break time="5"/>b</speak>',
        lambda ps, t, lang: pauses(ps) == [0, 500]),
    Row("break over ten seconds", DEGRADED, '<speak>a<break time="60s"/>b</speak>',
        lambda ps, t, lang: pauses(ps) == [0, 10_000]),
    Row("emphasis level", SUPPORTED, '<speak><emphasis level="strong">{t}</emphasis></speak>',
        lambda ps, t, lang: (ps[0].rate, ps[0].volume) == (0.9, 1.15)),
    Row("emphasis level reduced", SUPPORTED, '<speak><emphasis level="reduced">{t}</emphasis></speak>',
        lambda ps, t, lang: (ps[0].rate, ps[0].volume) == (1.05, 0.85)),
    # -- prosody -----------------------------------------------------------------------------
    Row("prosody rate keyword", SUPPORTED, '<speak><prosody rate="slow">{t}</prosody></speak>', field("rate", 0.8)),
    Row("prosody rate percent", SUPPORTED, '<speak><prosody rate="150%">{t}</prosody></speak>', field("rate", 1.5)),
    Row("prosody rate relative", SUPPORTED, '<speak><prosody rate="-20%">{t}</prosody></speak>', field("rate", 0.8)),
    Row("prosody rate number", SUPPORTED, '<speak><prosody rate="1.5">{t}</prosody></speak>', field("rate", 1.5)),
    Row("prosody pitch semitones", SUPPORTED, '<speak><prosody pitch="+3st">{t}</prosody></speak>', field("pitch", 3.0)),
    Row("prosody pitch percent", SUPPORTED, '<speak><prosody pitch="+100%">{t}</prosody></speak>', field("pitch", 12.0)),
    Row("prosody pitch keyword", SUPPORTED, '<speak><prosody pitch="high">{t}</prosody></speak>', field("pitch", 3.0)),
    Row("prosody pitch in Hz", UNSUPPORTED, '<speak><prosody pitch="+30Hz">{t}</prosody></speak>', field("pitch", 0.0)),
    Row("prosody contour", UNSUPPORTED, '<speak><prosody contour="(0%,+20%) (100%,-20%)">{t}</prosody></speak>',
        only("{t}")),
    Row("prosody range", UNSUPPORTED, '<speak><prosody range="x-high">{t}</prosody></speak>', only("{t}")),
    Row("prosody volume keyword", SUPPORTED, '<speak><prosody volume="soft">{t}</prosody></speak>', field("volume", 0.65)),
    Row("prosody volume dB", SUPPORTED, '<speak><prosody volume="-6dB">{t}</prosody></speak>',
        lambda ps, t, lang: abs(ps[0].volume - 10 ** (-6 / 20)) < 1e-9),
    Row("prosody volume number", SUPPORTED, '<speak><prosody volume="50">{t}</prosody></speak>', field("volume", 0.5)),
    Row("prosody duration", SUPPORTED, '<speak><prosody duration="3s">{t}</prosody></speak>',
        lambda ps, t, lang: ps[0].duration_ms == 3000),
    Row("prosody amazon:max-duration", SUPPORTED, '<speak><prosody amazon:max-duration="2s">{t}</prosody></speak>',
        lambda ps, t, lang: ps[0].max_duration_ms == 2000 and spoken(ps) == t),
    Row("nested prosody", SUPPORTED, '<speak><prosody rate="fast" pitch="+2st"><prosody rate="fast">{t}</prosody></prosody></speak>',
        lambda ps, t, lang: (ps[0].rate, ps[0].pitch) == (1.5625, 2.0)),
    # -- voice -------------------------------------------------------------------------------
    Row("voice languages", SUPPORTED, '<speak><voice languages="de-DE en-US">{t}</voice></speak>',
        lambda ps, t, lang: ps[0].lang == "de-DE"),
    Row("voice name, gender, age, variant, required, ordering", DEGRADED,
        '<speak><voice name="Celine" gender="female" age="30" variant="2" required="gender" ordering="gender">{t}</voice></speak>',
        only("{t}")),
    # -- audio, marks, descriptions ----------------------------------------------------------
    Row("audio src with fallback text", DEGRADED, '<speak><audio src="https://x/y.mp3">{t}</audio>after</speak>',
        only("{t} after")),
    Row("audio src without fallback", DEGRADED, '<speak><audio src="https://x/y.mp3"/>{t}</speak>', only("{t}")),
    Row("mark", SUPPORTED, '<speak>{t}<mark name="m"/> x</speak>',
        lambda ps, t, lang: [p.marks for p in ps] == [(), ("m",)]),
    Row("desc", DEGRADED, '<speak><desc>a picture of a cat</desc>{t}</speak>', only("{t}")),
    Row("metadata", DEGRADED, '<speak><metadata><rdf:RDF xmlns:rdf="u">secret</rdf:RDF></metadata>{t}</speak>', only("{t}")),
    Row("meta", DEGRADED, '<speak><meta name="seeAlso" content="x"/>{t}</speak>', only("{t}")),
    Row("seq, par, media", DEGRADED,
        "<speak><seq><media><speak>{t}</speak></media><par><media>one</media><media>two</media></par></seq></speak>",
        lambda ps, t, lang: spoken(ps) == f"{t} | one | two"),
    # -- Amazon Polly and Alexa --------------------------------------------------------------
    Row("amazon:effect name=whispered", SUPPORTED, '<speak><amazon:effect name="whispered">{t}</amazon:effect></speak>',
        field("effects", ("whisper",))),
    Row("amazon:effect name=drc", SUPPORTED, '<speak><amazon:effect name="drc">{t}</amazon:effect></speak>',
        field("effects", ("drc",))),
    Row("amazon:effect phonation=soft", SUPPORTED, '<speak><amazon:effect phonation="soft">{t}</amazon:effect></speak>',
        field("effects", ("soft",))),
    Row("amazon:effect name=phonation value=soft", SUPPORTED,
        '<speak><amazon:effect name="phonation" value="soft">{t}</amazon:effect></speak>',
        field("effects", ("soft",))),
    Row("amazon:effect vocal-tract-length", SUPPORTED,
        '<speak><amazon:effect vocal-tract-length="-20%">{t}</amazon:effect></speak>',
        lambda ps, t, lang: 3.0 < ps[0].pitch < 4.0),
    Row("amazon:auto-breaths", SUPPORTED,
        '<speak><amazon:auto-breaths volume="soft" frequency="high" duration="short">{t}</amazon:auto-breaths></speak>',
        lambda ps, t, lang: ps[0].breaths == (0.65, 250, 2)),
    Row("amazon:breath", SUPPORTED, '<speak>{t}<amazon:breath duration="long" volume="x-soft"/>x</speak>',
        lambda ps, t, lang: [p.sound for p in ps] == [None, ("breath", 500, 0.4), None]),
    Row("amazon:domain", DEGRADED, '<speak><amazon:domain name="news">{t}</amazon:domain></speak>', only("{t}")),
    Row("amazon:emotion", DEGRADED,
        '<speak><amazon:emotion name="excited" intensity="high">{t}</amazon:emotion></speak>', only("{t}")),
    Row("alexa:name", DEGRADED, '<speak>{t} <alexa:name type="first" personId="amzn1.ask.person.X"/></speak>',
        only("{t}")),
    Row("amazon prefix declared by the document", SUPPORTED,
        '<speak xmlns:amazon="https://amazon.com/ssml/2014"><amazon:effect name="whispered">{t}</amazon:effect></speak>',
        field("effects", ("whisper",))),
    Row("unknown element", DEGRADED, '<speak><bogus attr="1">{t}</bogus><unknown/> end</speak>', only("{t} end")),
]


@pytest.mark.parametrize("lang", ["en", "fr"])
@pytest.mark.parametrize("row", MATRIX, ids=lambda row: row.feature)
def test_the_reader_does_what_the_table_says(row, lang):
    words = WORDS[lang]
    pieces = parse_ssml(row.ssml.replace("{t}", words), lang)
    assert row.check(pieces, words, lang), (row.feature, pieces)


def test_the_statuses_are_the_three_the_docs_use():
    assert {row.status for row in MATRIX} <= {SUPPORTED, DEGRADED, UNSUPPORTED}
    assert len({row.feature for row in MATRIX}) == len(MATRIX), "a feature is listed twice"


def test_the_docs_list_every_row():
    page = (Path(__file__).parent.parent / "docs" / "ssml.md").read_text(encoding="utf-8")
    missing = [row.feature for row in MATRIX if f"| {row.feature} | {row.status} |" not in page]
    assert not missing, f"docs/ssml.md lacks (or has another status for): {missing}"
