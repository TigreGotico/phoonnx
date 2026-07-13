"""Language segmentation for code-switched text.

Ported from hams_tts.text.codeswitch (Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


def _is_arabic(c: str) -> bool:
    o = ord(c)
    return (
        0x0600 <= o <= 0x06FF
        or 0x0750 <= o <= 0x077F
        or 0x08A0 <= o <= 0x08FF
        or 0xFB50 <= o <= 0xFDFF
        or 0xFE70 <= o <= 0xFEFF
    )


def _is_latin(c: str) -> bool:
    o = ord(c)
    return (
        0x0041 <= o <= 0x005A
        or 0x0061 <= o <= 0x007A
        or 0x00C0 <= o <= 0x024F
    )


def _char_lang(c: str) -> Optional[str]:
    if _is_arabic(c):
        return "ar"
    if _is_latin(c):
        return "en"
    return None


@dataclass
class Span:
    text: str
    lang: str  # "ar" | "en"


def segment(text: str, default_lang: str = "en") -> List[Span]:
    """Return merged, language-tagged spans covering ``text`` in order."""
    if not text:
        return []

    raw: List[Tuple[str, Optional[str]]] = []
    cur_chars: List[str] = []
    cur_lang: Optional[str] = _char_lang(text[0])
    for c in text:
        cl = _char_lang(c)
        if cl == cur_lang:
            cur_chars.append(c)
        else:
            raw.append(("".join(cur_chars), cur_lang))
            cur_chars = [c]
            cur_lang = cl
    raw.append(("".join(cur_chars), cur_lang))

    resolved: List[Optional[str]] = [lang for _, lang in raw]
    last_seen: Optional[str] = None
    for i, lang in enumerate(resolved):
        if lang is not None:
            last_seen = lang
        else:
            resolved[i] = last_seen
    next_seen: Optional[str] = None
    for i in range(len(resolved) - 1, -1, -1):
        if raw[i][1] is not None:
            next_seen = raw[i][1]
        elif resolved[i] is None:
            resolved[i] = next_seen
    resolved = [r if r is not None else default_lang for r in resolved]

    spans: List[Span] = []
    for (txt, _), lang in zip(raw, resolved):
        if spans and spans[-1].lang == lang:
            spans[-1] = Span(spans[-1].text + txt, lang)
        else:
            spans.append(Span(txt, lang))

    out: List[Span] = []
    for s in spans:
        t = s.text.strip()
        if t:
            out.append(Span(t, s.lang))
    return out
