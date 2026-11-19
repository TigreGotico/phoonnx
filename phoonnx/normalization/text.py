"""Word-level helpers. text2spoken has no equivalent of them."""
import re

from phoonnx.normalization.numbers import _normalize_number_word
from phoonnx.normalization.tables import CONTRACTIONS, TITLES

def _normalize_word_hyphen_digit(text: str) -> str:
    """
    Helper function to normalize words attached to digits with a hyphen,
    such as 'sub-23' -> 'sub 23'.

    A hyphen (or en-dash/em-dash) directly between two digit runs, such as
    '3-2' or '1139-1185', is left untouched: that is a score or a range,
    not a word glued to a digit, and it is up to the phonemizer to read it
    as such.
    """
    # Regex to find a word (\w+) followed by a hyphen/en-dash/em-dash and a
    # digit (\d+).
    pattern = re.compile(r"(\w+)[-–—](\d+)")

    def _replace(match: "re.Match") -> str:
        word_part = match.group(1)
        if word_part.isdigit():
            # digit-dash-digit is a range or score (e.g. '3-2', '1139-1185')
            return match.group(0)
        return f"{word_part} {match.group(2)}"

    text = pattern.sub(_replace, text)
    return text

def _normalize_word(word: str, full_lang: str, rbnf_engine, expand_contractions: bool = True) -> str:
    """
    Helper function to normalize a single word.
    """
    lang_code = full_lang.split("-")[0]
    word = word.replace("’", "'")

    if expand_contractions and word in CONTRACTIONS.get(lang_code, {}):
        return CONTRACTIONS[lang_code][word]

    if word in TITLES.get(lang_code, {}):
        return TITLES[lang_code][word]

    # Delegate number parsing to the new helper function
    normalized_number = _normalize_number_word(word, full_lang, rbnf_engine)
    if normalized_number != word:
        return normalized_number

    return word
