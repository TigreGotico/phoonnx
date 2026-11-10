"""Spoken forms for numbers: separators, fractions and digit strings."""
import string

from ovos_number_parser import pronounce_number, pronounce_fraction
from ovos_number_parser.util import is_numeric
from unicode_rbnf import FormatPurpose

from phoonnx.log import LOG

def _get_number_separators(full_lang: str) -> tuple[str, str]:
    """
    Determines decimal and thousands separators based on language.
    Defaults to '.' decimal and ',' thousands for most languages.
    Special cases:
    - 'pt', 'es', 'fr', 'de': ',' decimal and '.' thousands.
    """
    lang_code = full_lang.split("-")[0]
    decimal_separator = '.'
    thousands_separator = ','
    if lang_code in ["pt", "es", "fr", "de"]:
        decimal_separator = ','
        thousands_separator = '.'
    return decimal_separator, thousands_separator

def is_fraction(word: str) -> bool:
    """Checks if a word is a fraction like '3/3'."""
    if "/" in word:
        parts = word.split("/")
        if len(parts) == 2:
            n1, n2 = parts
            return n1.isdigit() and n2.isdigit()
    return False

def _normalize_number_word(word: str, full_lang: str, rbnf_engine) -> str:
    """
    Helper function to normalize a single word that is a number, handling
    decimal and thousands separators based on locale.
    """
    cleaned_word = word.rstrip(string.punctuation)

    # Handle fractions like '3/3'
    if is_fraction(cleaned_word):
        try:
            return pronounce_fraction(cleaned_word, full_lang) + word[len(cleaned_word):]
        except Exception as e:
            LOG.error(f"ovos-number-parser failed to pronounce fraction: {word} - ({e})")
            return word

    # Handle numbers with locale-specific separators
    decimal_separator, thousands_separator = _get_number_separators(full_lang)
    temp_cleaned_word = cleaned_word

    # Check if the word contains a thousands separator followed by digits and a decimal separator
    # This is a specific check for formats like '123.456,78'
    has_thousands_and_decimal = (
            thousands_separator in temp_cleaned_word and
            decimal_separator in temp_cleaned_word and
            temp_cleaned_word.index(thousands_separator) < temp_cleaned_word.index(decimal_separator)
    )

    if has_thousands_and_decimal:
        temp_cleaned_word = temp_cleaned_word.replace(thousands_separator, "")
        temp_cleaned_word = temp_cleaned_word.replace(decimal_separator, ".")
    elif decimal_separator in temp_cleaned_word and is_numeric(temp_cleaned_word.replace(decimal_separator, ".", 1)):
        # Handle cases like '1,2' -> '1.2'
        temp_cleaned_word = temp_cleaned_word.replace(decimal_separator, ".")
    elif thousands_separator in temp_cleaned_word and is_numeric(temp_cleaned_word.replace(thousands_separator, "", 1)):
        # Handle cases like '1.234' -> '1234'
        temp_cleaned_word = temp_cleaned_word.replace(thousands_separator, "")

    # Check if the word is a valid number after processing
    if is_numeric(temp_cleaned_word):
        try:
            num = float(temp_cleaned_word) if "." in temp_cleaned_word else int(temp_cleaned_word)
            return pronounce_number(num, lang=full_lang) + word[len(cleaned_word):]
        except Exception as e:
            LOG.error(f"ovos-number-parser failed to pronounce number: {word} - ({e})")
            return word

    elif rbnf_engine and cleaned_word.isdigit():
        try:
            pronounced_number = rbnf_engine.format_number(cleaned_word, FormatPurpose.CARDINAL).text
            return pronounced_number + word[len(cleaned_word):]
        except Exception as e:
            LOG.error(f"unicode-rbnf failed to pronounce number: {word} - ({e})")
            return word

    return word
