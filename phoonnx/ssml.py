"""SSML for phoonnx: markup read into pieces a voice can speak one after another.

Supported: pauses (``<break>``, ``<p>``, ``<s>``); pace, level and pitch (``<prosody rate volume
pitch>``, ``<emphasis>``); another language (``<lang>``); how to read a run of text
(``<say-as interpret-as=...>`` for cardinal, ordinal, digits, fraction, unit, date, time, currency,
address, telephone, characters, expletive), ``<sub>`` and phonemes (``<phoneme>`` in IPA or X-SAMPA,
which phoonnx reads as ``[[ ... ]]``); ``<mark>`` events; and Amazon's extensions
(``<amazon:effect name="whispered|drc">``, ``phonation="soft"``, ``vocal-tract-length``,
``<amazon:auto-breaths>``, ``<amazon:breath>``). Each piece is some text, the language to say it
in, how fast (a divisor of the voice's length scale), how loud (a multiplier on the samples), its
pitch shift in semitones, the effects to apply, and how long a silence comes before it. Pitch and
effects are applied to the finished samples (see :mod:`phoonnx.audio_fx`). Anything unknown is
read as plain text, never as tags.

A document that does not parse is read as its text with the tags taken out: a typo in the
markup costs its pauses, not the reply. Values are bounded (a minute of silence is a hang, a
run of digits is not a number), and markup is only recognised inside a ``<speak>``, because
``<p>`` and ``<s>`` are also HTML and an answer about HTML is words.
"""
from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET
import math
from dataclasses import dataclass, replace

from phoonnx.ssml_say_as import say_as_text
from phoonnx.xsampa import xsampa_to_ipa

LOG = logging.getLogger("phoonnx.ssml")

# What a <break strength> is worth, and what a paragraph or sentence adds.
_STRENGTH_MS = {"none": 0, "x-weak": 100, "weak": 250, "medium": 500,
                "strong": 750, "x-strong": 1250}
_PARAGRAPH_MS = 750
_SENTENCE_MS = 150
# A skill asking for a minute of silence is asking for a hang.
_MAX_BREAK_MS = 10_000
_RATES = {"x-slow": 0.6, "slow": 0.8, "medium": 1.0, "default": 1.0, "fast": 1.25, "x-fast": 1.5}
_VOLUMES = {"silent": 0.0, "x-soft": 0.4, "soft": 0.65, "medium": 1.0, "default": 1.0,
            "loud": 1.3, "x-loud": 1.6}
# (rate, volume) multipliers: emphasis is slower and a little louder.
_EMPHASIS = {"strong": (0.9, 1.15), "moderate": (0.95, 1.1),
             "reduced": (1.05, 0.85), "none": (1.0, 1.0)}
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
_PITCHES = {"x-low": -6.0, "low": -3.0, "medium": 0.0, "default": 0.0, "high": 3.0, "x-high": 6.0}
_MAX_SEMITONES = 12.0
# `<amazon:auto-breaths frequency=...>`: a breath after every n-th sentence of the piece.
_BREATH_EVERY = {"x-low": 5, "low": 4, "medium": 3, "default": 3, "high": 2, "x-high": 1}
_BREATH_MS = {"x-short": 150, "short": 250, "medium": 350, "default": 350, "long": 500, "x-long": 700}
# Tags only SSML has. <p>, <s>, <sub>, <audio> and <mark> are HTML's too, and
# an answer about HTML ("a paragraph goes between <p> and </p>") is words:
# those count as markup only inside a <speak>, which this finds.
# A complete tag: '<' straight onto the name (so "1 < w and w > 0" is not one), then
# optional attributes and the closing '>'.
_TAG = re.compile(r"</?(?:speak|break|prosody|emphasis|say-as|phoneme|lang|voice|w|"
                  r"amazon:[a-z-]+)(?:\s[^<>]*)?/?>", re.IGNORECASE)
# A number as float() reads it: "1.5", "2.", ".5". Not "1.2.3" or ".", which
# a looser [\d.]+ admitted and float() then refused, losing the reply.
_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)"
# `<amazon:effect>` and friends: a namespace prefix nobody declares.
_PREFIXED = re.compile(r"<(/?)\s*([A-Za-z][\w.-]*):([\w.-]+)")


#: The tags :func:`parse_ssml` reads. Everything else in a document is read as its text.
SUPPORTED_TAGS = ("speak", "break", "prosody", "emphasis", "say-as", "phoneme", "lang", "voice",
                  "p", "s", "sub", "w", "mark", "audio", "amazon:effect", "amazon:auto-breaths",
                  "amazon:breath", "amazon:domain")


@dataclass(frozen=True)
class Piece:
    """Some text to say, how to say it, and the silence before it.

    `pitch` is a shift in semitones; `effects` are names from :mod:`phoonnx.audio_fx`
    (``"whisper"``, ``"soft"``, ``"drc"``); `breaths` is ``(volume, duration_ms, every)``: a
    breath after every `every`-th sentence; `marks` are ``<mark name>`` events at the start of the
    piece; `sound` is ``("beep"|"breath", duration_ms, volume)`` for a piece that is only a
    sound (``text`` is then empty).
    """

    text: str
    lang: str | None = None
    rate: float = 1.0
    volume: float = 1.0
    pause_ms: int = 0
    pitch: float = 0.0
    effects: tuple = ()
    breaths: tuple | None = None
    marks: tuple = ()
    sound: tuple | None = None


def looks_like_ssml(text: str | None) -> bool:
    """Whether `text` carries SSML markup rather than just a '<' in a sentence."""
    return bool(text) and bool(_TAG.search(text))


def _without_tags(text: str) -> str:
    """`text` with every tag -- a '<', then up to the next '>' -- a space.

    A walk, not the pattern <[^>]*>: from every '<' the pattern ran to the end
    looking for a '>' that was not there, which on text with many '<' and no
    '>' is quadratic.
    """
    out: list[str] = []
    start = 0
    while True:
        opening = text.find("<", start)
        if opening < 0:
            break
        closing = text.find(">", opening + 1)
        if closing < 0:
            break
        out.append(text[start:opening])
        out.append(" ")
        start = closing + 1
    out.append(text[start:])
    return "".join(out)


def strip_ssml(text: str | None) -> str:
    """The words of an SSML string, for a screen, a log or a speaker that cannot read markup."""
    if not text:
        return ""
    words = " ".join(html.unescape(_without_tags(text)).split())
    # A tag removed before a full stop or comma leaves a space no one wrote.
    return re.sub(r"\s+([.,;)])", r"\1", words)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _duration_ms(value: str | None) -> int | None:
    match = re.fullmatch(rf"\s*({_NUMBER})\s*(ms|s)?\s*", value or "")
    if not match:
        return None
    number = float(match.group(1)) * (1000 if match.group(2) == "s" else 1)
    # Bounded before int(): a run of digits reads as infinity.
    return int(min(number, _MAX_BREAK_MS))


def _rate(value: str | None, current: float) -> float:
    if not value:
        return current
    value = value.strip().lower()
    if value in _RATES:
        return current * _RATES[value]
    match = re.fullmatch(rf"([+-]?{_NUMBER})%", value)
    if match:
        number = float(match.group(1))
        # "+20%" is relative, "120%" absolute, as the spec has it.
        factor = 1 + number / 100 if value[0] in "+-" else number / 100
        return _clamp(current * factor, 0.5, 2.0)
    return current


def _volume(value: str | None, current: float) -> float:
    if not value:
        return current
    value = value.strip().lower()
    if value in _VOLUMES:
        return current * _VOLUMES[value]
    match = re.fullmatch(rf"([+-]?{_NUMBER})\s*db", value)
    if match:
        # Past +-96 dB (all 16 bits) nothing changes, and 10 ** 5000 overflows.
        decibels = _clamp(float(match.group(1)), -96.0, 96.0)
        return _clamp(current * 10 ** (decibels / 20), 0.0, 2.0)
    return current


def _pitch(value: str | None, current: float) -> float:
    """Semitones after applying a ``pitch`` value: ``+2st``, ``-10%``, ``high``. Hz is ignored."""
    if not value:
        return current
    value = value.strip().lower()
    if value in _PITCHES:
        return _clamp(current + _PITCHES[value], -_MAX_SEMITONES, _MAX_SEMITONES)
    match = re.fullmatch(rf"([+-]?{_NUMBER})\s*st", value)
    if match:
        return _clamp(current + float(match.group(1)), -_MAX_SEMITONES, _MAX_SEMITONES)
    match = re.fullmatch(rf"([+-]?{_NUMBER})%", value)
    if match:
        number = float(match.group(1))
        factor = 1 + number / 100 if value[0] in "+-" else number / 100
        if factor > 0:
            return _clamp(current + 12.0 * math.log2(factor), -_MAX_SEMITONES, _MAX_SEMITONES)
    return current


def _tract_semitones(value: str | None) -> float:
    """A longer vocal tract (``+10%``) sounds lower; the shift that mimics it, in semitones."""
    if not value:
        return 0.0
    match = re.fullmatch(rf"\s*([+-]?{_NUMBER})\s*%\s*", value.strip())
    if not match:
        return 0.0
    factor = 1 + float(match.group(1)) / 100
    return -12.0 * math.log2(factor) if factor > 0.1 else 0.0


class _Reader:
    """Walks the tree once, gathering text into pieces."""

    def __init__(self, lang: str | None) -> None:
        self.pieces: list[Piece] = []
        self.text: list[str] = []
        self.style = Piece("", lang)
        self.pause = 0
        # A <break> written before the first words: kept, unlike the edge of
        # an opening <p> or <s>, which has nothing before it to pause from.
        self.lead_break = 0
        # <mark> events wait here for the next piece to carry them.
        self.marks: list[str] = []

    def sound(self, kind: str, ms: int, volume: float = 1.0) -> None:
        """A piece that is only a sound: a breath, or a bleep over an expletive."""
        self.flush()
        self.pieces.append(Piece("", self.style.lang, pause_ms=self.pause, marks=tuple(self.marks),
                                 sound=(kind, ms, volume)))
        self.pause, self.marks = 0, []

    def flush(self) -> None:
        text = " ".join(" ".join(self.text).split())
        # A tag between a word and its full stop leaves a space nobody wrote.
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        self.text = []
        if not text or self.style.volume <= 0:
            return
        if not any(ch.isalnum() for ch in text):
            # A lone "." left behind by a language switch: nothing to say, and
            # a synthesiser asked to say it anyway answers with a click.
            return
        self.pieces.append(replace(self.style, text=text, pause_ms=self.pause, marks=tuple(self.marks)))
        self.pause, self.marks = 0, []

    def silence(self, ms: int, *, boundary: bool = False) -> None:
        """Pause before whatever comes next. A <break> adds to what is pending;
        a paragraph or sentence edge only makes sure there is at least `ms`,
        so the end of one paragraph and the start of the next pause once."""
        self.flush()
        ms = max(0, ms)
        self.pause = min(_MAX_BREAK_MS, max(self.pause, ms) if boundary else self.pause + ms)
        if not self.pieces and not boundary:
            self.lead_break = min(_MAX_BREAK_MS, self.lead_break + ms)

    def styled(self, element, style: Piece) -> None:
        """Read `element`'s children in `style`, then carry on as before."""
        self.flush()
        before, self.style = self.style, style
        self.read(element)
        self.flush()
        self.style = before

    def read(self, element) -> None:
        if element.text:
            self.text.append(element.text)
        for child in element:
            self.visit(child)
            if child.tail:
                self.text.append(child.tail)

    def visit(self, element) -> None:
        tag = element.tag.rsplit("}", 1)[-1].lower()
        if tag == "break":
            ms = _duration_ms(element.get("time"))
            if ms is None:
                ms = _STRENGTH_MS.get(element.get("strength", "medium"), 500)
            self.silence(ms)
        elif tag in ("p", "s"):
            edge = _PARAGRAPH_MS if tag == "p" else _SENTENCE_MS
            self.silence(edge, boundary=True)
            self.read(element)
            self.silence(edge, boundary=True)
        elif tag == "prosody":
            self.styled(element, replace(
                self.style,
                rate=_clamp(_rate(element.get("rate"), self.style.rate), 0.5, 2.0),
                volume=_clamp(_volume(element.get("volume"), self.style.volume), 0.0, 2.0),
                pitch=_pitch(element.get("pitch"), self.style.pitch),
            ))
        elif tag == "emphasis":
            rate, volume = _EMPHASIS.get(element.get("level", "moderate"), _EMPHASIS["moderate"])
            self.styled(element, replace(
                self.style, rate=_clamp(self.style.rate * rate, 0.5, 2.0),
                volume=_clamp(self.style.volume * volume, 0.0, 2.0),
            ))
        elif tag in ("lang", "voice"):
            lang = element.get(_XML_LANG) or element.get("lang")
            self.styled(element, replace(self.style, lang=lang) if lang else self.style)
        elif tag == "say-as":
            kind = (element.get("interpret-as") or "").lower()
            written = "".join(element.itertext())
            if kind == "expletive":
                self.sound("beep", 350)
            else:
                self.text.append(say_as_text(written, kind, self.style.lang,
                                             element.get("format"), element.get("detail")))
        elif tag == "sub":
            self.text.append(element.get("alias") or "".join(element.itertext()))
        elif tag == "phoneme":
            # phoonnx reads [[ ... ]] as phonemes in the voice's own (espeak, IPA) alphabet;
            # an alphabet it cannot convert is read as the written word.
            ph = element.get("ph")
            alphabet = (element.get("alphabet") or "ipa").lower()
            written = "".join(element.itertext())
            if ph and alphabet == "ipa":
                self.text.append(f"[[ {ph} ]]")
            elif ph and alphabet in ("x-sampa", "xsampa", "sampa"):
                self.text.append(f"[[ {xsampa_to_ipa(ph)} ]]")
            else:
                self.text.append(written)
        elif tag == "mark":
            # A <mark> splits the piece so the event lands where the author put it.
            self.flush()
            name = element.get("name")
            if name:
                self.marks.append(name)
            self.read(element)
        elif tag == "amazon-effect":
            effects = self.style.effects
            name = (element.get("name") or "").lower()
            if name in ("whispered", "whisper") and "whisper" not in effects:
                effects += ("whisper",)
            if name == "drc" and "drc" not in effects:
                effects += ("drc",)
            if (element.get("phonation") or "").lower() == "soft" and "soft" not in effects:
                effects += ("soft",)
            shift = _tract_semitones(element.get("vocal-tract-length"))
            self.styled(element, replace(
                self.style, effects=effects,
                pitch=_clamp(self.style.pitch + shift, -_MAX_SEMITONES, _MAX_SEMITONES)))
        elif tag == "amazon-auto-breaths":
            volume = _volume(element.get("volume"), 1.0)
            ms = _BREATH_MS.get((element.get("duration") or "medium").lower(), 350)
            every = _BREATH_EVERY.get((element.get("frequency") or "medium").lower(), 3)
            off = (element.get("frequency") or "").lower() == "off"
            self.styled(element, replace(self.style, breaths=None if off else (volume, ms, every)))
        elif tag == "amazon-breath":
            ms = _BREATH_MS.get((element.get("duration") or "medium").lower(), 350)
            self.sound("breath", ms, _volume(element.get("volume"), 1.0))
        elif tag == "audio":
            # An <audio src> is not fetched. Its fallback text, if any, is read.
            self.read(element)
        else:
            # speak, mark, w, token, amazon:* and anything unknown: the words.
            self.read(element)


def parse_ssml(ssml: str, lang: str | None = None) -> list[Piece]:
    """Read `ssml` into pieces to synthesise one after another.

    Silence after the last piece is dropped, and so is the pause an opening
    <p> or <s> implies. A <break> before the first words is kept: a story
    read a sentence at a time puts the pause between two paragraphs at the
    start of the second one's sentence.
    """
    source = (ssml or "").strip()
    if not source:
        return []
    if not re.match(r"<\s*speak\b", source, re.IGNORECASE):
        source = f"<speak>{source}</speak>"
    source = _PREFIXED.sub(lambda m: f"<{m.group(1)}{m.group(2)}-{m.group(3)}"
                           if m.group(2).lower() != "xml" else m.group(0), source)
    try:
        root = ET.fromstring(source)
    except ET.ParseError as error:
        LOG.info("ssml: not well-formed (%s); reading the text without it", error)
        text = strip_ssml(ssml)
        return [Piece(text, lang)] if text else []
    reader = _Reader(lang)
    try:
        reader.read(root)
        reader.flush()
        if reader.marks:  # marks after the last words: an event with nothing to say
            reader.pieces.append(Piece("", lang, marks=tuple(reader.marks)))
    except Exception as error:  # noqa: BLE001 - a value nothing above foresaw
        LOG.info("ssml: could not be read (%s); reading the text without it",
                 type(error).__name__)
        text = strip_ssml(ssml)
        return [Piece(text, lang)] if text else []
    if reader.pieces:
        reader.pieces[0] = replace(reader.pieces[0], pause_ms=reader.lead_break)
    return reader.pieces
