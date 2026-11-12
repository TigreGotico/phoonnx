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
import html.entities
import logging
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from xml.parsers import expat

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
# ... and so is a thousand breaks of ten seconds: the silence and the sounds of one document
# (pauses, breaths, bleeps) share this budget; what does not fit is dropped, marks are kept.
_MAX_NONSPEECH_MS = 120_000
# `<prosody duration>`: the time asked for a stretch of speech, not a pause.
_MAX_DURATION_MS = 600_000
# Elements nested deeper than this are not a document, they are an attack on the recursion limit.
_MAX_DEPTH = 100
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
_TAG = re.compile(r"</?(?:ssml:)?(?:speak|break|prosody|emphasis|say-as|phoneme|lang|voice|w|"
                  r"(?:amazon|alexa):[a-z-]+)(?:\s[^<>]*)?/?>", re.IGNORECASE)
# A number as float() reads it: "1.5", "2.", ".5". Not "1.2.3" or ".", which
# a looser [\d.]+ admitted and float() then refused, losing the reply.
_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)"
_SSML_NAMESPACE = "http://www.w3.org/2001/10/synthesis"
# Elements whose content is about the document, not for the listener.
_SILENT_TAGS = frozenset({"desc", "metadata", "meta", "lexicon"})
# Characters XML 1.0 cannot carry (control characters, lone surrogates, U+FFFE/U+FFFF).
_ILLEGAL_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
_ENTITY = re.compile(r"&(#[0-9]+|#[xX][0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]*);")
_XML_ENTITIES = frozenset({"amp", "lt", "gt", "quot", "apos"})
# A start or end tag as a person writes it: a name, then attributes whose quoted values may hold '>'.
_MARKUP = re.compile(r"""</?[A-Za-z_][^<>"']*(?:(?:"[^"]*"|'[^']*')[^<>"']*)*>""")
_NEXT_SPECIAL = re.compile(r"[<&]")


#: The tags :func:`parse_ssml` reads. Everything else in a document is read as its text.
SUPPORTED_TAGS = ("speak", "break", "prosody", "emphasis", "say-as", "phoneme", "lang", "voice",
                  "p", "s", "sub", "w", "token", "lookup", "mark", "audio", "seq", "par", "media",
                  "desc", "metadata", "meta", "lexicon", "amazon:effect", "amazon:auto-breaths",
                  "amazon:breath", "amazon:domain", "amazon:emotion", "alexa:name")


@dataclass(frozen=True)
class Piece:
    """Some text to say, how to say it, and the silence before it.

    `pitch` is a shift in semitones; `effects` are names from :mod:`phoonnx.audio_fx`
    (``"whisper"``, ``"soft"``, ``"drc"``); `breaths` is ``(volume, duration_ms, every)``: a
    breath after every `every`-th sentence; `marks` are ``<mark name>`` events at the start of the
    piece, and `mark_pauses` holds, for each, how many milliseconds of the piece's `pause_ms` come
    before it (a mark written ahead of a ``<break>`` is heard before the silence, one written after it
    is heard after); `sound` is ``("beep"|"breath", duration_ms, volume)`` for a piece that is only a
    sound (``text`` is then empty); `duration_ms` is the time ``<prosody duration>`` asks for this
    text and `max_duration_ms` the most ``amazon:max-duration`` allows (speech is stretched or
    squeezed to fit, within a factor of two).
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
    mark_pauses: tuple = ()
    duration_ms: int | None = None
    max_duration_ms: int | None = None


def looks_like_ssml(text: str | None) -> bool:
    """Whether `text` carries SSML markup rather than just a '<' in a sentence."""
    return bool(text) and bool(_TAG.search(text))


def _without_tags(text: str) -> str:
    """`text` with every tag -- a '<' straight onto a name, '/', '!' or '?', then up to the next
    '>' -- a space. A '<' before a space or a digit ("1 < 2") is a word, and a CDATA section
    keeps its content.

    A walk, not the pattern <[^>]*>: from every '<' the pattern ran to the end
    looking for a '>' that was not there, which on text with many '<' and no
    '>' is quadratic.
    """
    out: list[str] = []
    start = 0
    search = 0
    while True:
        opening = text.find("<", search)
        if opening < 0:
            break
        if text.startswith("<![CDATA[", opening):
            end = text.find("]]>", opening + 9)
            if end >= 0:
                out.append(text[start:opening])
                out.append(html.escape(text[opening + 9:end], quote=False))  # unescaped again by the caller
                start = search = end + 3
                continue
        if not (text[opening + 1: opening + 2].isalpha() or text[opening + 1: opening + 2] in ("/", "!", "?")):
            search = opening + 1
            continue
        closing = text.find(">", opening + 1)
        if closing < 0:
            break
        out.append(text[start:opening])
        out.append(" ")
        start = search = closing + 1
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


def _duration_ms(value: str | None, cap: int = _MAX_BREAK_MS) -> int | None:
    """A time value (``500ms``, ``1.5s``; the unit is required) in milliseconds, at most `cap`."""
    match = re.fullmatch(rf"\s*({_NUMBER})\s*(ms|s)\s*", (value or "").lower())
    if not match:
        return None
    number = float(match.group(1)) * (1000 if match.group(2) == "s" else 1)
    # Bounded before int(): a run of digits reads as infinity.
    return int(min(number, cap))


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
    match = re.fullmatch(_NUMBER, value)
    if match:  # a bare number is a multiplier on the default rate (SSML 1.1)
        return _clamp(current * float(value), 0.5, 2.0)
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
    match = re.fullmatch(rf"([+-]?{_NUMBER})%?", value)
    if match:
        # "+20%" and "-10" are relative, "50%" and "50" a share of the full level (0 to 100).
        number = float(match.group(1))
        factor = 1 + number / 100 if value[0] in "+-" else number / 100
        return _clamp(current * factor, 0.0, 2.0)
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


class _TooDeep(ValueError):
    """Elements nested past :data:`_MAX_DEPTH`."""


def _entity(match: re.Match) -> str:
    """What to write for the entity in `match` so that an XML parser accepts it (or "&amp;" if it was no entity)."""
    body = match.group(1)
    if body is None:  # a bare '&': "AT&T"
        return "&amp;"
    if body[0] == "#":
        number = int(body[2:], 16) if body[1] in "xX" else int(body[1:])
        legal = number in (9, 10, 13) or 0x20 <= number <= 0xD7FF or 0xE000 <= number <= 0xFFFD \
            or 0x10000 <= number <= 0x10FFFF
        return f"&#{number};" if legal else ""
    if body in _XML_ENTITIES:
        return f"&{body};"
    named = html.entities.html5.get(f"{body};")  # &nbsp; &eacute; ...: HTML's, not XML's
    if named is not None:
        return html.escape(named, quote=False)
    return f"&amp;{body};"


_ENTITY_OR_AMP = re.compile(r"&(?:(#[0-9]+|#[xX][0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]*);)?")


def _repair(source: str) -> str:
    """`source` with what a person (or a language model) writes and XML refuses made legal.

    A bare ``&`` ("AT&T") and a bare ``<`` ("1 < 2"), HTML's named entities (``&nbsp;``), control
    characters, an XML declaration, comments and a ``<!DOCTYPE>`` (whose entities could be made to
    expand into a megabyte). Linear: every search for a closer that was once missing is not repeated.
    """
    source = _ILLEGAL_XML.sub("", source)
    missing: dict[str, int] = {}  # closer -> position after which there is none

    def find(closer: str, start: int) -> int:
        if closer in missing and start >= missing[closer]:
            return -1
        found = source.find(closer, start)
        if found < 0:
            missing[closer] = start
        return found

    def text(chunk: str) -> str:
        return _ENTITY_OR_AMP.sub(_entity, chunk).replace("]]>", "]]&gt;")

    out: list[str] = []
    pos = 0
    while True:
        k = source.find("<", pos)
        if k < 0:
            out.append(text(source[pos:]))
            break
        out.append(text(source[pos:k]))
        pos = k + 1
        if source.startswith("<![CDATA[", k):
            end = find("]]>", k + 9)
            if end >= 0:
                out.append(source[k: end + 3])
                pos = end + 3
                continue
        elif source.startswith("<!--", k):
            end = find("-->", k + 4)
            if end >= 0:
                pos = end + 3
                continue
        elif source.startswith("<?", k):
            end = find("?>", k + 2)
            if end >= 0:
                pos = end + 2
                continue
        elif source.startswith("<!", k):
            end = find(">", k + 2)
            if end >= 0 and source.find("[", k, end) >= 0:  # an internal subset ends in "]>"
                end = find("]>", k + 2)
                end = end + 1 if end >= 0 else -1
            if end >= 0:
                pos = end + 1
                continue
        else:
            tag = _MARKUP.match(source, k)
            if tag:
                out.append("<" + text(tag.group()[1:]).replace("<", "&lt;"))
                pos = tag.end()
                continue
        out.append("&lt;")  # a '<' that opens nothing
    return "".join(out)


def _tag_name(name: str, ssml_prefixes: set[str]) -> str:
    """The lower-case name of an element: ``ssml:break`` (the prefix declared as SSML's) is ``break``."""
    name = name.lower()
    prefix, colon, local = name.partition(":")
    return local if colon and prefix in ssml_prefixes else name


def _build(source: str) -> ET.Element:
    """The tree of `source`, read without namespaces: ``amazon:effect`` is a name whether or not
    anything declared ``amazon``, and ``amazon:max-duration`` an attribute like any other."""
    builder = ET.TreeBuilder()
    parser = expat.ParserCreate()
    parser.buffer_text = True
    # "ssml:" is the prefix everyone uses for the SSML namespace, declared or not.
    prefixes: set[str] = {"ssml"}
    names: list[str] = []

    def start(name: str, attrs: dict) -> None:
        if len(names) >= _MAX_DEPTH:
            raise _TooDeep(name)
        for key, value in attrs.items():
            if key.startswith("xmlns:") and value == _SSML_NAMESPACE:
                prefixes.add(key[6:])
        names.append(_tag_name(name, prefixes))
        builder.start(names[-1], attrs)

    def end(_name: str) -> None:
        builder.end(names.pop())

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = builder.data
    parser.Parse(source, True)
    return builder.close()


def _language(element: ET.Element) -> str | None:
    """The language an element says its content is in: ``xml:lang`` anywhere, ``lang`` on <lang>,
    the first of ``languages`` on <voice>."""
    value = element.get("xml:lang")
    if not value and element.tag == "lang":
        value = element.get("lang")
    if not value and element.tag == "voice":
        value = (element.get("languages") or "").split(" ")[0]
    return value.strip() or None if value else None


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
        # <mark> events wait here, with the pending pause that came before each, for the next piece.
        self.marks: list[tuple[str, int]] = []

    def _carried(self) -> dict:
        """The pause and marks pending, as the fields of the piece that takes them."""
        fields = {"pause_ms": self.pause, "marks": tuple(name for name, _ in self.marks),
                  "mark_pauses": tuple(min(at, self.pause) for _, at in self.marks)}
        self.pause, self.marks = 0, []
        return fields

    def sound(self, kind: str, ms: int, volume: float = 1.0) -> None:
        """A piece that is only a sound: a breath, or a bleep over an expletive."""
        self.flush()
        self.pieces.append(Piece("", self.style.lang, sound=(kind, ms, volume), **self._carried()))

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
        self.pieces.append(replace(self.style, text=text, **self._carried()))

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

    def fit(self, start: int, duration_ms: int | None, longest_ms: int | None) -> None:
        """Ask the text pieces from `start` on to take `duration_ms` between them (and at most
        `longest_ms`), the time of the pauses and sounds among them taken out, shared by length."""
        inside = self.pieces[start:]
        spoken = [i for i, piece in enumerate(inside, start) if piece.text]
        if not spoken:
            return
        away = sum(piece.pause_ms for piece in inside[1:]) + sum(
            piece.sound[1] for piece in inside if piece.sound)
        for limit, key in ((duration_ms, "duration_ms"), (longest_ms, "max_duration_ms")):
            if not limit:  # none, or 0: nothing was asked
                continue
            open_ = [i for i in spoken if getattr(self.pieces[i], key) is None]  # the inner one wins
            is_open = set(open_)  # a list here made `not in` quadratic in the number of pieces
            taken = sum(getattr(self.pieces[i], key) or 0 for i in spoken if i not in is_open)
            budget = limit - away - taken
            weight = sum(len(self.pieces[i].text) for i in open_)
            if not open_ or not weight:
                continue
            for i in open_:  # pauses longer than the time asked leave the least: the speech is squeezed
                share = int(max(budget, 0) * len(self.pieces[i].text) / weight)
                self.pieces[i] = replace(self.pieces[i], **{key: max(1, share)})

    def visit(self, element) -> None:
        lang = _language(element)
        if not lang or lang == self.style.lang:
            self.visit_tag(element)
            return
        self.flush()
        before, self.style = self.style, replace(self.style, lang=lang)
        try:
            self.visit_tag(element)
            self.flush()
        finally:
            self.style = before

    def visit_tag(self, element) -> None:
        tag = element.tag
        if tag in _SILENT_TAGS:
            return  # about the document, not for the listener
        if tag == "break":
            ms = _duration_ms(element.get("time"))
            if ms is None:
                ms = _STRENGTH_MS.get((element.get("strength") or "medium").strip().lower(), 500)
            self.silence(ms)
        elif tag in ("p", "s", "media"):
            edge = _PARAGRAPH_MS if tag == "p" else _SENTENCE_MS
            self.silence(edge, boundary=True)
            self.read(element)
            self.silence(edge, boundary=True)
        elif tag == "prosody":
            style = replace(
                self.style,
                rate=_clamp(_rate(element.get("rate"), self.style.rate), 0.5, 2.0),
                volume=_clamp(_volume(element.get("volume"), self.style.volume), 0.0, 2.0),
                pitch=_pitch(element.get("pitch"), self.style.pitch),
            )
            duration = _duration_ms(element.get("duration"), _MAX_DURATION_MS)
            longest = _duration_ms(element.get("amazon:max-duration"), _MAX_DURATION_MS)
            self.flush()
            start = len(self.pieces)
            self.styled(element, style)
            if duration is not None or longest is not None:
                self.fit(start, duration, longest)
        elif tag == "emphasis":
            rate, volume = _EMPHASIS.get((element.get("level") or "moderate").strip().lower(),
                                         _EMPHASIS["moderate"])
            self.styled(element, replace(
                self.style, rate=_clamp(self.style.rate * rate, 0.5, 2.0),
                volume=_clamp(self.style.volume * volume, 0.0, 2.0),
            ))
        elif tag in ("lang", "voice"):
            self.styled(element, self.style)  # the language, if it said one, is read by visit()
        elif tag == "say-as":
            kind = (element.get("interpret-as") or "").lower()
            written = "".join(element.itertext())
            if kind == "expletive":
                self.sound("beep", 350)
            else:
                self.text.append(say_as_text(written, kind, self.style.lang,
                                             element.get("format"), element.get("detail")))
        elif tag == "sub":
            alias = element.get("alias")
            self.text.append(alias if alias is not None else "".join(element.itertext()))
        elif tag == "phoneme":
            # phoonnx reads [[ ... ]] as phonemes in the voice's own (espeak, IPA) alphabet;
            # an alphabet it cannot convert is read as the written word.
            ph = re.sub(r"[\[\]]", "", element.get("ph") or "").strip()
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
                self.marks.append((name, self.pause))
            self.read(element)
        elif tag == "amazon:effect":
            effects = self.style.effects
            name = (element.get("name") or "").lower()
            if name in ("whispered", "whisper") and "whisper" not in effects:
                effects += ("whisper",)
            if name == "drc" and "drc" not in effects:
                effects += ("drc",)
            phonation = element.get("phonation") or (element.get("value") if name == "phonation" else "")
            if (phonation or "").lower() == "soft" and "soft" not in effects:
                effects += ("soft",)
            tract = element.get("vocal-tract-length") or (
                element.get("value") if name == "vocal-tract-length" else None)
            shift = _tract_semitones(tract)
            self.styled(element, replace(
                self.style, effects=effects,
                pitch=_clamp(self.style.pitch + shift, -_MAX_SEMITONES, _MAX_SEMITONES)))
        elif tag == "amazon:auto-breaths":
            volume = _volume(element.get("volume"), 1.0)
            ms = _BREATH_MS.get((element.get("duration") or "medium").lower(), 350)
            every = _BREATH_EVERY.get((element.get("frequency") or "medium").lower(), 3)
            off = (element.get("frequency") or "").lower() == "off"
            self.styled(element, replace(self.style, breaths=None if off else (volume, ms, every)))
        elif tag == "amazon:breath":
            ms = _BREATH_MS.get((element.get("duration") or "medium").lower(), 350)
            self.sound("breath", ms, _volume(element.get("volume"), 1.0))
        else:
            # speak, audio (an <audio src> is not fetched: its fallback text is read), w, token,
            # lookup, seq, par, amazon:domain, amazon:emotion and anything unknown: the words.
            self.read(element)


def _within_budget(pieces: list[Piece]) -> list[Piece]:
    """`pieces` with the silence and sounds that do not fit :data:`_MAX_NONSPEECH_MS` cut (marks stay)."""
    left = _MAX_NONSPEECH_MS
    kept: list[Piece] = []
    for piece in pieces:
        pause = min(piece.pause_ms, left)
        left -= pause
        sound = piece.sound
        if sound is not None:
            if sound[1] > left:
                sound = None
            else:
                left -= sound[1]
        if pause != piece.pause_ms or sound is not piece.sound:
            piece = replace(piece, pause_ms=pause, sound=sound,
                            mark_pauses=tuple(min(at, pause) for at in piece.mark_pauses))
        if piece.text or piece.sound is not None or piece.marks:
            kept.append(piece)
    return kept


def _plain(ssml: str, lang: str | None) -> list[Piece]:
    text = strip_ssml(ssml)
    return [Piece(text, lang)] if text else []


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
    try:
        # Always inside a <speak> of its own: a nested <speak> is only its words, and a document
        # with two roots or text after the root still reads.
        root = _build(f"<speak>{_repair(source)}</speak>")
    except (expat.ExpatError, ET.ParseError, ValueError) as error:
        LOG.info("ssml: not well-formed (%s); reading the text without it", error)
        return _plain(ssml, lang)
    reader = _Reader(lang)
    try:
        reader.visit(root)
        reader.flush()
        if reader.marks:  # marks after the last words: an event with nothing to say
            reader.pieces.append(Piece("", lang, **reader._carried()))
    except Exception as error:  # noqa: BLE001 - a value nothing above foresaw
        LOG.info("ssml: could not be read (%s); reading the text without it",
                 type(error).__name__)
        return _plain(ssml, lang)
    if reader.pieces:
        first = reader.pieces[0]
        lead = reader.lead_break
        reader.pieces[0] = replace(first, pause_ms=lead,
                                   mark_pauses=tuple(min(at, lead) for at in first.mark_pauses))
    return _within_budget(reader.pieces)
