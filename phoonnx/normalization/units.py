"""Expansion of units and ordinal indicators attached to numbers."""
import re

from ovos_number_parser import pronounce_number

from phoonnx.log import LOG
from phoonnx.normalization.numbers import _get_number_separators
from phoonnx.normalization.tables import UNITS

def _normalize_units(text: str, full_lang: str) -> str:
    """
    Helper function to normalize units attached to numbers.
    This function handles symbolic and alphanumeric units separately
    to avoid issues with word boundaries.
    """
    # "º" (U+00BA, masculine ordinal indicator, e.g. "1º andar") looks like
    # "°" (U+00B0, degree sign) but is not the same character. Only treat it
    # as a degree sign when it is actually used as a temperature unit
    # (e.g. "20ºC" / "20ºc"), so ordinals are not corrupted into degrees.
    # Case-insensitive to match the (IGNORECASE) unit regex below.
    text = re.sub(r"º(?=\s?[CFK]\b)", "°", text, flags=re.IGNORECASE)

    # Any remaining "º" is a genuine ordinal indicator attached to a digit
    # (e.g. "1º andar", "20º"). normalize() must never leave raw digits in
    # the output, so expand these as ordinal numbers, falling back to a
    # plain cardinal (still dropping the "º") if ordinal pronunciation is
    # unavailable for the language.
    def _replace_ordinal_indicator(match: "re.Match") -> str:
        number = int(match.group(1))
        try:
            return pronounce_number(number, full_lang, ordinals=True)
        except Exception as e:
            LOG.error(f"Failed to pronounce ordinal number: {number}º - ({e})")
            return pronounce_number(number, full_lang)

    text = re.sub(r"(\d+)º", _replace_ordinal_indicator, text)

    lang_code = full_lang.split("-")[0]
    if lang_code in UNITS:
        # Determine number separators for the language
        decimal_separator, thousands_separator = _get_number_separators(full_lang)

        # Separate units into symbolic and alphanumeric
        symbolic_units = {k: v for k, v in UNITS[lang_code].items() if not k.isalnum()}
        alphanumeric_units = {k: v for k, v in UNITS[lang_code].items() if k.isalnum()}
        # The unit regex ignores case, so "ml" must find the key "mL".
        folded_units = {k.casefold(): v for k, v in alphanumeric_units.items()}

        # A number is one of: digits grouped in threes by the thousands separator
        # ("1,500" and "12,345.6" in English, "1.500,5" in Portuguese); digits
        # with a "." decimal part, which comma-decimal languages also meet
        # ("1.5 km"); digits with the language's decimal part. A number never
        # starts right after a digit and a separator, so "2,5 km" in English
        # stays as written instead of reading "2," and "5 km" apart.
        thousands, decimal = re.escape(thousands_separator), re.escape(decimal_separator)
        grouped = rf"\d{{1,3}}(?:{thousands}\d{{3}})+(?:{decimal}\d+)?"
        number_pattern_str = (rf"(?<!\d)(?<!\d[{thousands}{decimal}])"
                              rf"({grouped}|\d+\.\d+(?![.\d])|\d+(?:{decimal}\d+)?)")

        def number_value(number: str):
            if re.fullmatch(grouped, number):
                number = number.replace(thousands_separator, "")
            number = number.replace(decimal_separator, ".")
            return float(number) if "." in number else int(number)

        # Create regex pattern for symbolic units and replace them first
        sorted_symbolic = sorted(symbolic_units.keys(), key=len, reverse=True)
        symbolic_pattern_str = "|".join(re.escape(unit) for unit in sorted_symbolic)
        if symbolic_pattern_str:
            symbolic_pattern = re.compile(number_pattern_str + r"\s*(" + symbolic_pattern_str + r")", re.IGNORECASE)

            def replace_symbolic(match):
                number = match.group(1)
                unit_symbol = match.group(2)
                # The regex is IGNORECASE (e.g. "°c" matches "°C"), so the
                # matched text may not share the dict key's exact case.
                unit_word = (symbolic_units.get(unit_symbol)
                             or symbolic_units.get(unit_symbol.upper())
                             or symbolic_units.get(unit_symbol.lower()))
                try:
                    return f"{pronounce_number(number_value(number), full_lang)} {unit_word}"
                except Exception as e:
                    LOG.error(f"Failed to pronounce number with unit: {number}{unit_symbol} - ({e})")
                    return match.group(0)

            text = symbolic_pattern.sub(replace_symbolic, text)

        # Create regex pattern for alphanumeric units and replace them next
        sorted_alphanumeric = sorted(alphanumeric_units.keys(), key=len, reverse=True)
        alphanumeric_pattern_str = "|".join(re.escape(unit) for unit in sorted_alphanumeric)
        if alphanumeric_pattern_str:
            alphanumeric_pattern = re.compile(number_pattern_str + r"\s*(" + alphanumeric_pattern_str + r")\b",
                                              re.IGNORECASE)

            def replace_alphanumeric(match):
                number = match.group(1)
                unit_symbol = match.group(2)
                # None for a letter that matches only without case, such as "İ".
                unit_word = folded_units.get(unit_symbol.casefold())
                if unit_word is None:
                    return match.group(0)
                try:
                    return f"{pronounce_number(number_value(number), full_lang)} {unit_word}"
                except Exception as e:
                    LOG.error(f"Failed to pronounce number with unit: {number}{unit_symbol} - ({e})")
                    return match.group(0)

            text = alphanumeric_pattern.sub(replace_alphanumeric, text)
    return text
