"""Phoneme segments: each base symbol with the marks that modify it.

A segment phoneme map gives one id to ``ɐ̃``, ``tʲ``, ``aː`` or ``d͡ʒ`` instead of
one id to each code point, so that an input feature table can describe whole
segments. Every segment is a substring of the phonemizer output, so the
runtime tokenizer, which matches the longest key first, finds the same ids.
"""
from typing import List

#: Modifier letters that modify the symbol before them: aspiration, breathy
#: voice, palatalization, labialization, labio-palatalization, velarization,
#: pharyngealization, glottalization, long, half-long, rhotic hook, ejective,
#: nasal and lateral release, and the spacing raised, lowered, advanced and
#: retracted marks.
ATTACHING = frozenset("ʰʱʲʷᶣˠˤˀːˑ˞ʼⁿˡ˔˕˖˗")
#: Tie bars, which join the symbols before and after them into one segment.
TIES = frozenset("͜͡")
def segment_phonemes(text: str) -> List[str]:
    """Cut *text* into segments that join back into it unchanged.

    phonematcher's ``segment_spans`` decides each segment, so the map and the
    feature table read the text the way phonematcher's distance does: ``aⁿda``
    is ``a``, ``ⁿd``, ``a``, and ``tⁿ`` keeps its nasal release. A tie bar
    between two spans joins them, so ``a͡i`` stays one segment. Every other
    character between spans, the stress marks, spaces and punctuation
    included, is a segment of its own.
    """
    try:
        from phonematcher import segment_spans
    except ImportError as e:
        raise ImportError("--segment-map needs phonematcher: install phoonnx[train-features]") from e
    segments: List[str] = []
    pos = 0
    for start, end in segment_spans(text):
        gap = text[pos:start]
        if pos and gap and set(gap) <= TIES:
            segments[-1] += gap + text[start:end]
        else:
            segments.extend(gap)
            segments.append(text[start:end])
        pos = end
    segments.extend(text[pos:])
    return segments
