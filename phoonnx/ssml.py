"""SSML for phoonnx: markup read into pieces a voice can speak one after another.

A VITS voice cannot whisper, change pitch or show emotion, so the markup is read for what it
*can* do: a pause (``<break>``, ``<p>``, ``<s>``), a pace and a level (``<prosody rate/volume>``,
``<emphasis>``), a word in another language (``<lang>``), a date, a spelled-out word or a
substitution read as words (``<say-as>``, ``<sub>``) and phonemes (``<phoneme ph>``, which
phoonnx reads as ``[[ ... ]]``). Each piece is some text, the language to say it in, how fast
(a divisor of the voice's length scale), how loud (a multiplier on the samples) and how long a
silence comes before it. Everything a voice cannot do is read as plain text, never as tags.

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
from dataclasses import dataclass, replace

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
_SPELLED = {"characters", "spell-out", "letters", "verbatim"}
_DIGITS = {"digits", "telephone"}
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
                  "p", "s", "sub", "w", "mark", "audio")


@dataclass(frozen=True)
class Piece:
    """Some text to say, how to say it, and the silence before it."""

    text: str
    lang: str | None = None
    rate: float = 1.0
    volume: float = 1.0
    pause_ms: int = 0


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

    def flush(self) -> None:
        text = " ".join(" ".join(self.text).split())
        self.text = []
        if not text or self.style.volume <= 0:
            return
        if not any(ch.isalnum() for ch in text):
            # A lone "." left behind by a language switch: nothing to say, and
            # a synthesiser asked to say it anyway answers with a click.
            return
        self.pieces.append(replace(self.style, text=text, pause_ms=self.pause))
        self.pause = 0

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
            words = "".join(element.itertext())
            kind = (element.get("interpret-as") or "").lower()
            if kind in _SPELLED:
                words = " ".join(ch for ch in words if not ch.isspace())
            elif kind in _DIGITS:
                words = " ".join(ch for ch in words if ch.isdigit())
            self.text.append(words)
        elif tag == "sub":
            self.text.append(element.get("alias") or "".join(element.itertext()))
        elif tag == "phoneme":
            # phoonnx reads [[ ... ]] as phonemes in the voice's own (espeak, IPA)
            # alphabet; anything else is read as the written word.
            ph = element.get("ph")
            alphabet = (element.get("alphabet") or "ipa").lower()
            written = "".join(element.itertext())
            self.text.append(f"[[ {ph} ]]" if ph and alphabet == "ipa" else written)
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
    except Exception as error:  # noqa: BLE001 - a value nothing above foresaw
        LOG.info("ssml: could not be read (%s); reading the text without it",
                 type(error).__name__)
        text = strip_ssml(ssml)
        return [Piece(text, lang)] if text else []
    if reader.pieces:
        reader.pieces[0] = replace(reader.pieces[0], pause_ms=reader.lead_break)
    return reader.pieces
