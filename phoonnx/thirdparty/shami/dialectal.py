"""Dialectal de-desinentialisation for Levantine Arabic.

Ported from hams_tts.text.dialectal (Apache-2.0).
"""

from __future__ import annotations

FATHA, KASRA, DAMMA = "َ", "ِ", "ُ"
SUKUN, SHADDA = "ْ", "ّ"
TANWIN_F, TANWIN_K, TANWIN_D = "ً", "ٍ", "ٌ"
TEH_MARBUTA = "ة"

_MARKS = {FATHA, KASRA, DAMMA, SUKUN, SHADDA, TANWIN_F, TANWIN_K, TANWIN_D}
_BARE_SHORT = {FATHA, KASRA, DAMMA}
_DROP_TANWIN = {TANWIN_K, TANWIN_D}


def _strip_word(w: str) -> str:
    if not w:
        return w
    i = len(w) - 1
    while i >= 0 and w[i] in _MARKS:
        i -= 1
    if i < 0:
        return w
    letter, marks, prefix = w[i], set(w[i + 1:]), w[:i]

    has_shadda = SHADDA in marks
    has_sukun = SUKUN in marks
    keep_case = ""

    if marks & _DROP_TANWIN or (marks & _BARE_SHORT):
        pass
    elif TANWIN_F in marks:
        keep_case = "" if letter == TEH_MARBUTA else TANWIN_F

    rebuilt = letter + (SHADDA if has_shadda else "") + keep_case + (SUKUN if has_sukun else "")
    return prefix + rebuilt


def strip_case_endings(text: str) -> str:
    """Remove MSA word-final case/mood markers so the Levantine G2P sees pausal forms."""
    return " ".join(_strip_word(w) for w in text.split())
