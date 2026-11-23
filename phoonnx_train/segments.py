"""Phoneme segments: each base symbol with the marks that modify it.

A segment phoneme map gives one id to ``ɐ̃``, ``tʲ``, ``aː`` or ``d͡ʒ`` instead of
one id to each code point, so that an input feature table can describe whole
segments. Every segment is a substring of the phonemizer output, so the
runtime tokenizer, which matches the longest key first, finds the same ids.
"""
import unicodedata
from typing import List

#: Modifier letters that modify the symbol before them: aspiration, breathy
#: voice, palatalization, labialization, labio-palatalization, velarization,
#: pharyngealization, glottalization, long, half-long, rhotic hook, ejective,
#: nasal and lateral release, and the spacing raised, lowered, advanced and
#: retracted marks.
ATTACHING = frozenset("ʰʱʲʷᶣˠˤˀːˑ˞ʼⁿˡ˔˕˖˗")
#: Tie bars, which join the symbols before and after them into one segment.
TIES = frozenset("͜͡")
#: The three tone notations with levels: Chao tone letters, the left-stem tone
#: letters U+A712 to U+A716, and the tone numerals ¹ to ⁵. A run in one notation
#: is one segment, the contour, so ``˥˩`` and ``⁵¹`` are one id each. A change of
#: notation starts a segment: ``˨˩˦꜔꜒`` is the IPA sandhi notation, an underlying
#: 214 and then a surface 35. The neutral-tone ⁰ has no level and stands alone.
TONE_NOTATIONS = (frozenset("˥˦˧˨˩"), frozenset("\ua712\ua713\ua714\ua715\ua716"), frozenset("¹²³⁴⁵"))


def segment_phonemes(text: str) -> List[str]:
    """Cut *text* into segments that join back into it unchanged.

    A combining mark, an attaching modifier letter, and the symbol after a tie
    bar join the segment before them. A run of tone letters in one notation is
    one segment.
    Every other character, the stress marks, spaces and punctuation included,
    starts a segment of its own.
    """
    segments: List[str] = []
    after_tie = False
    for char in text:
        joins = segments and (
            after_tie
            or unicodedata.category(char) == "Mn"
            or char in ATTACHING
            or any(char in notation and segments[-1][-1] in notation for notation in TONE_NOTATIONS))
        if joins:
            segments[-1] += char
        else:
            segments.append(char)
        after_tie = char in TIES
    return segments
