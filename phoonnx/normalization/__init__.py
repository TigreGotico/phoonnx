"""Text normalization: rewriting written text into the words a voice should speak.

``normalize(text, lang)`` is the entry point. It runs the raw text through the
per-concern passes in this package — dates and times, hyphen-joined
word/digit pairs, units and ordinal indicators, then a word-by-word pass that
expands contractions, titles, fractions and bare numbers — so that nothing
reaches the phonemizer as digits or symbols.

The language-specific lookup tables live in :mod:`phoonnx.normalization.tables`,
apart from the code that applies them, so a new language is a data change.
"""

from phoonnx.normalization.datetimes import (
    _normalize_dates_and_times,
    pronounce_date,
    pronounce_time,
)
from phoonnx.normalization.numbers import (
    _get_number_separators,
    _normalize_number_word,
    is_fraction,
)
from phoonnx.normalization.tables import CONTRACTIONS, TITLES, UNITS
from phoonnx.normalization.text import (
    _normalize_word,
    _normalize_word_hyphen_digit,
    normalize,
)
from phoonnx.normalization.units import _normalize_units
