"""English grapheme-to-phoneme, producing IPA from the shared inventory.

Ported from hams_tts.text.english_g2p (Apache-2.0).
"""

from __future__ import annotations

from typing import List

from . import espeak
from .phoneme_inventory import fold_to_inventory, tokenize_ipa

_REWRITE = [
    ("ɡ", "ɡ"),
    ("g", "ɡ"),
    ("ɹ", "ɹ"),
    ("ʔ", "ʔ"),
    ("ɐ", "ə"),
    ("ɚ", "ə"),
    ("ɝ", "ɜː"),
    ("ɫ", "ɫ"),
    ("oʊ", "o͡ʊ"), ("aʊ", "a͡ʊ"), ("aɪ", "a͡ɪ"), ("eɪ", "e͡ɪ"), ("ɔɪ", "ɔ͡ɪ"),
    ("dʒ", "d͡ʒ"), ("tʃ", "t͡ʃ"),
]


def _rewrite(ipa: str) -> str:
    for a, b in _REWRITE:
        ipa = ipa.replace(a, b)
    return ipa


def english_g2p(text: str) -> str:
    """Convert English text to an IPA string drawn from the shared inventory."""
    if espeak.available():
        ipa = espeak.phonemize(text, voice="en-us")
        ipa = _rewrite(ipa)
        return "".join(
            s if s == " " else fold_to_inventory(s)
            for s in _retokenize(ipa)
        )
    return _fallback(text)


def _retokenize(ipa: str) -> List[str]:
    out: List[str] = []
    for word in ipa.split(" "):
        toks = tokenize_ipa(word).symbols
        if out:
            out.append(" ")
        out.extend(toks)
    return out


_FALLBACK_MAP = {
    "a": "æ", "b": "b", "c": "k", "d": "d", "e": "ɛ", "f": "f", "g": "ɡ",
    "h": "h", "i": "ɪ", "j": "d͡ʒ", "k": "k", "l": "l", "m": "m", "n": "n",
    "o": "ɒ", "p": "p", "q": "k", "r": "ɹ", "s": "s", "t": "t", "u": "ʌ",
    "v": "v", "w": "w", "x": "ks", "y": "j", "z": "z",
}


def _fallback(text: str) -> str:
    words = []
    for word in text.lower().split():
        words.append("".join(_FALLBACK_MAP.get(c, "") for c in word))
    return " ".join(w for w in words if w)
