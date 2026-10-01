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

        # Create regex pattern for symbolic units and replace them first
        sorted_symbolic = sorted(symbolic_units.keys(), key=len, reverse=True)
        symbolic_pattern_str = "|".join(re.escape(unit) for unit in sorted_symbolic)
        if symbolic_pattern_str:
            # Pattern to match numbers with optional thousands and decimal separators
            number_pattern_str = rf"(\d+[{re.escape(thousands_separator)}]?\d*[{re.escape(decimal_separator)}]?\d*)"
            symbolic_pattern = re.compile(number_pattern_str + r"\s*(" + symbolic_pattern_str + r")", re.IGNORECASE)

            def replace_symbolic(match):
                number = match.group(1)
                # Remove thousands separator and replace decimal separator for parsing
                if thousands_separator in number and decimal_separator in number:
                    number = number.replace(thousands_separator, "").replace(decimal_separator, ".")
                elif decimal_separator != "." and decimal_separator in number:
                    number = number.replace(decimal_separator, ".")
                unit_symbol = match.group(2)
                # The regex is IGNORECASE (e.g. "°c" matches "°C"), so the
                # matched text may not share the dict key's exact case.
                unit_word = (symbolic_units.get(unit_symbol)
                             or symbolic_units.get(unit_symbol.upper())
                             or symbolic_units.get(unit_symbol.lower()))
                try:
                    return f"{pronounce_number(float(number) if '.' in number else int(number), full_lang)} {unit_word}"
                except Exception as e:
                    LOG.error(f"Failed to pronounce number with unit: {number}{unit_symbol} - ({e})")
                    return match.group(0)

            text = symbolic_pattern.sub(replace_symbolic, text)

        # Create regex pattern for alphanumeric units and replace them next
        sorted_alphanumeric = sorted(alphanumeric_units.keys(), key=len, reverse=True)
        alphanumeric_pattern_str = "|".join(re.escape(unit) for unit in sorted_alphanumeric)
        if alphanumeric_pattern_str:
            number_pattern_str = rf"(\d+[{re.escape(thousands_separator)}]?\d*[{re.escape(decimal_separator)}]?\d*)"
            alphanumeric_pattern = re.compile(number_pattern_str + r"\s*(" + alphanumeric_pattern_str + r")\b",
                                              re.IGNORECASE)

            def replace_alphanumeric(match):
                number = match.group(1)
                # Remove thousands separator and replace decimal separator for parsing
                if thousands_separator in number and decimal_separator in number:
                    number = number.replace(thousands_separator, "").replace(decimal_separator, ".")
                elif decimal_separator != "." and decimal_separator in number:
                    number = number.replace(decimal_separator, ".")
                unit_symbol = match.group(2)
                unit_word = alphanumeric_units[unit_symbol]
                return f"{pronounce_number(float(number) if '.' in number else int(number), full_lang)} {unit_word}"

            text = alphanumeric_pattern.sub(replace_alphanumeric, text)
    return text
