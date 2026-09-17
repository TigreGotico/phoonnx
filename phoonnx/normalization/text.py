"""The text normalization pipeline: raw written text to speakable words."""
import re

from unicode_rbnf import RbnfEngine

from phoonnx.log import LOG
from phoonnx.normalization.datetimes import _normalize_dates_and_times
from phoonnx.normalization.numbers import _normalize_number_word
from phoonnx.normalization.tables import CONTRACTIONS, TITLES
from phoonnx.normalization.units import _normalize_units

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

def _normalize_word(word: str, full_lang: str, rbnf_engine) -> str:
    """
    Helper function to normalize a single word.
    """
    lang_code = full_lang.split("-")[0]
    word = word.replace("’", "'")

    if word in CONTRACTIONS.get(lang_code, {}):
        return CONTRACTIONS[lang_code][word]

    if word in TITLES.get(lang_code, {}):
        return TITLES[lang_code][word]

    # Delegate number parsing to the new helper function
    normalized_number = _normalize_number_word(word, full_lang, rbnf_engine)
    if normalized_number != word:
        return normalized_number

    return word

def normalize(text: str, lang: str) -> str:
    """
    Normalize a text string for speech by expanding contractions and titles and converting dates, times, numbers, units, and fractions into spoken forms.
    
    Parameters:
        text (str): Input text to normalize.
        lang (str): Locale language tag (e.g., "en", "en-US") that determines language-specific rules.
    
    Returns:
        str: The normalized text with expanded contractions and titles and with dates, times, numbers, units, and fractions converted to their spoken equivalents.
    
    Notes:
        Date and time normalization is attempted for the locale and is silently skipped if unsupported. Numeric formatting may use a locale RBNF engine when available; otherwise a fallback is used.
    """
    full_lang = lang
    lang_code = full_lang.split("-")[0]
    dialog = text

    # Step 1: Handle dates and times with ovos-date-parser
    date_format = "MDY" if full_lang.lower() == "en-us" else "DMY"
    try:
        dialog = _normalize_dates_and_times(dialog, full_lang, date_format)
    except:  # throws exception on unsupported langs
        pass

    # Step 2: Normalize words with hyphens and digits
    dialog = _normalize_word_hyphen_digit(dialog)

    # Step 3: Expand units attached to numbers
    dialog = _normalize_units(dialog, full_lang)

    # Step 4: Normalize word-by-word
    words = dialog.split()
    rbnf_engine = None
    try:
        rbnf_engine = RbnfEngine.for_language(lang_code)
    except (ValueError, KeyError) as e:
        LOG.debug(f"RBNF engine not available for language '{lang_code}': {e}")

    normalized_words = [_normalize_word(word, full_lang, rbnf_engine) for word in words]
    dialog = " ".join(normalized_words)

    return dialog
if __name__ == "__main__":
    # --- Example usage for demonstration purposes ---

    # General normalization examples
    print("General English example: " + normalize('I\'m Dr. Prof. 3/3 0.5% of 12345€, 5ft, and 10kg', 'en'))
    print(
        f"Word Salad Portuguese (Dr. Prof. 3/3 0,5% de 12345€, 5m, e 10kg): {normalize('Dr. Prof. 3/3 0,5% de 12345€, 5m, e 10kg', 'pt')}")
    print(
        f"Word Salad Portuguese (Dr. Prof. 3/3 0.5% de 12345€, 5m, e 10kg): {normalize('Dr. Prof. 3/3 0.5% de 12345€, 5m, e 10kg', 'pt')}")

    # Portuguese examples with comma decimal separator
    print("\n--- Portuguese Decimal Separator Examples ---")
    print(
        f"Original: 'A coima aplicada é de 1,2 milhões de euros.' Normalized: '{normalize('A coima aplicada é de 1,2 milhões de euros.', 'pt')}'")
    print(
        f"Original: 'Agora, tem 1,88 metros e muito para contar.' Normalized: '{normalize('Agora, tem 1,88 metros e muito para contar.', 'pt')}'")
    print(
        f"Original: 'Ainda temos 1,7 milhões de pobres!' Normalized: '{normalize('Ainda temos 1,7 milhões de pobres!', 'pt')}'")
    print(f"Original: 'O lucro foi de 123.456,78€.' Normalized: '{normalize('O lucro foi de 123.456,78€.', 'pt')}'")
    print(f"Normalized: '{normalize('O lucro foi de 123.456,78€.', 'pt-PT')}'")

    # English dates and times
    print("\n--- English Date & Time Examples ---")
    print(f"English date (MDY format): {normalize('The date is 08/03/2025', 'en-US')}")
    print(f"English ambiguous date (MDY assumed): {normalize('The report is due 15/05/2025', 'en-US')}")
    print(f"English date with dashes: {normalize('The event is on 11-04-2025', 'en-US')}")
    print(f"English AM/PM time: {normalize('The meeting is at 10am', 'en-US')}")
    print(f"English military time: {normalize('The party is at 19h30', 'en-US')}")
    print(f"English month name: {normalize('The report is due 15 May 2025', 'en-US')}")

    # Portuguese dates and times
    print("\n--- Portuguese Date & Time Examples ---")
    print(f"Portuguese date (A data é 03/08/2025): {normalize('A data é 03/08/2025', 'pt')}")
    print(
        f"Portuguese ambiguous date (O relatório é para 15/05/2025): {normalize('O relatório é para 15/05/2025', 'pt')}")
    print(
        f"Portuguese date with dashes (O evento é no dia 25-10-2024): {normalize('O evento é no dia 25-10-2024', 'pt')}")
    print(f"Portuguese military time (O encontro é às 14h30): {normalize('O encontro é às 14h30', 'pt')}")

    # Other examples
    print(f"\n--- Other Examples ---")
    print(f"English fraction: {normalize('The fraction is 1/2', 'en')}")
    print(f"English plural fraction: {normalize('There are 3/4 of a cup', 'en')}")
    print(f"Spanish example with units: {normalize('The temperature is 25ºC', 'es')}")
    print(f"Portuguese with punctuation: {normalize('12345€, 5m e 10kg', 'pt')}")
    print(
        f"Portuguese word-digit: {normalize('Esta temporada leva oito jogos ao serviço da equipa sub-23 leonina.', 'pt')}")
