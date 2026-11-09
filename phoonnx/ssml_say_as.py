"""``<say-as interpret-as=...>``: how a run of text should be read aloud.

:func:`say_as_text` turns the written form into the words a voice should say, in the language
of the text, using the same number and date parsers as :mod:`phoonnx.normalization`. It handles
the interpretations Amazon Polly offers (``cardinal``, ``ordinal``, ``digits``, ``fraction``,
``unit``, ``date``, ``time``, ``telephone``, ``address``, ``characters``/``spell-out``,
``currency``, ``interjection``, ``expletive``) and returns the text unchanged for anything it
does not know, or cannot read: a skill's typo costs its formatting, never the reply.
"""
from __future__ import annotations

import datetime
import functools
import re
from typing import Optional

from phoonnx.log import LOG

SPELLED = {"characters", "spell-out", "letters", "verbatim"}
DIGITS = {"digits", "telephone"}

# Words for a money amount: (major singular, major plural, minor singular, minor plural).
_CURRENCY_WORDS = {
    "en": {"USD": ("dollar", "dollars", "cent", "cents"), "EUR": ("euro", "euros", "cent", "cents"),
           "GBP": ("pound", "pounds", "penny", "pence"), "CAD": ("dollar", "dollars", "cent", "cents"),
           "JPY": ("yen", "yen", "sen", "sen"), "CHF": ("franc", "francs", "centime", "centimes")},
    "fr": {"USD": ("dollar", "dollars", "cent", "cents"), "EUR": ("euro", "euros", "centime", "centimes"),
           "GBP": ("livre", "livres", "penny", "pence"), "CAD": ("dollar", "dollars", "cent", "cents"),
           "JPY": ("yen", "yens", "sen", "sen"), "CHF": ("franc", "francs", "centime", "centimes")},
    "es": {"USD": ("dólar", "dólares", "centavo", "centavos"), "EUR": ("euro", "euros", "céntimo", "céntimos"),
           "GBP": ("libra", "libras", "penique", "peniques")},
    "de": {"USD": ("Dollar", "Dollar", "Cent", "Cent"), "EUR": ("Euro", "Euro", "Cent", "Cent"),
           "GBP": ("Pfund", "Pfund", "Penny", "Pence"), "CHF": ("Franken", "Franken", "Rappen", "Rappen")},
    "pt": {"USD": ("dólar", "dólares", "centavo", "centavos"), "EUR": ("euro", "euros", "cêntimo", "cêntimos"),
           "BRL": ("real", "reais", "centavo", "centavos")},
    "it": {"USD": ("dollaro", "dollari", "centesimo", "centesimi"), "EUR": ("euro", "euro", "centesimo", "centesimi")},
}
_AND = {"en": "and", "fr": "et", "es": "con", "de": "und", "pt": "e", "it": "e"}
_SYMBOLS = {"$": "USD", "US$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "C$": "CAD", "CA$": "CAD",
            "R$": "BRL", "CHF": "CHF"}

_ADDRESS_ABBREVIATIONS = {
    "en": {"st": "Street", "ave": "Avenue", "blvd": "Boulevard", "rd": "Road", "dr": "Drive",
           "ln": "Lane", "ct": "Court", "pl": "Place", "apt": "Apartment", "ste": "Suite",
           "n": "North", "s": "South", "e": "East", "w": "West"},
    "fr": {"av": "avenue", "bd": "boulevard", "bld": "boulevard", "r": "rue", "pl": "place",
           "ch": "chemin", "all": "allée", "imp": "impasse", "app": "appartement", "bte": "boîte"},
}


def _language(lang: Optional[str]) -> str:
    return (lang or "en").replace("_", "-")


def _number(value: str, lang: str, ordinal: bool = False) -> str:
    """The words for a number written with the thousands and decimal marks of `lang`."""
    from ovos_number_parser import pronounce_number

    text = value.strip()
    decimal = "," if lang.split("-")[0] in ("fr", "es", "de", "pt", "it", "nl", "ru", "pl") else "."
    thousands = "." if decimal == "," else ","
    text = re.sub(r"[\s  ]", "", text)
    if thousands in text and decimal in text and text.index(thousands) < text.index(decimal):
        text = text.replace(thousands, "")
    elif text.count(thousands) >= 1 and decimal not in text and re.fullmatch(
            rf"\d{{1,3}}(?:\{thousands}\d{{3}})+", text):
        text = text.replace(thousands, "")
    text = text.replace(decimal, ".")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        raise ValueError(f"not a number: {value!r}")
    number = float(text) if "." in text else int(text)
    if ordinal:
        number = int(number)
        try:
            from ovos_number_parser import pronounce_ordinal

            return pronounce_ordinal(number, lang)
        except Exception:  # noqa: BLE001 - not every language has one: fall back to the generic form
            return pronounce_number(number, lang=lang, ordinals=True)
    said = pronounce_number(number, lang=lang)
    if isinstance(number, float) and re.search(r"\d", said):
        # Some languages leave a decimal above 999 as digits: say the two halves ourselves.
        whole, _, fraction = text.partition(".")
        point = _DECIMAL_WORD.get(lang.split("-")[0], "point")
        digits = " ".join(pronounce_number(int(d), lang=lang) for d in fraction)
        said = f"{pronounce_number(int(whole), lang=lang)} {point} {digits}"
    return said


_DECIMAL_WORD = {"en": "point", "fr": "virgule", "es": "coma", "pt": "vírgula", "de": "Komma",
                 "it": "virgola", "nl": "komma"}


def _ordinal(text: str, lang: str) -> str:
    digits = re.sub(r"[^\d]", "", text)
    return _number(digits, lang, ordinal=True)


def _fraction(text: str, lang: str) -> str:
    from ovos_number_parser import pronounce_fraction

    text = text.strip()
    mixed = re.fullmatch(r"(\d+)\s+(\d+)\s*/\s*(\d+)", text)
    if mixed:
        whole, num, den = mixed.groups()
        join = _AND.get(lang.split("-")[0], "and")
        return f"{_number(whole, lang)} {join} {pronounce_fraction(f'{num}/{den}', lang)}"
    simple = re.fullmatch(r"(\d+)\s*/\s*(\d+)", text)
    if simple:
        return pronounce_fraction(f"{simple.group(1)}/{simple.group(2)}", lang)
    return _number(text, lang)


def _unit(text: str, lang: str) -> str:
    from phoonnx.normalization.units import _normalize_units

    spaced = re.sub(r"(?<=\d)(?=[^\d\s.,/-])", " ", text.strip())
    said = _normalize_units(spaced, lang)
    # Whatever the unit table did not turn into words, the number still can be.
    return re.sub(r"\d+(?:[.,]\d+)?", lambda m: _number(m.group(0), lang), said)


_DATE_ORDERS = {"ymd", "ydm", "mdy", "myd", "dmy", "dym", "ym", "my", "md", "dm", "y", "m", "d"}


def _date(text: str, lang: str, fmt: Optional[str]) -> str:
    from ovos_date_parser import nice_date

    parts = [p for p in re.split(r"[-/.\s,]+", text.strip()) if p]
    numeric = all(p.isdigit() for p in parts)
    order = (fmt or "").lower()
    if not numeric and not order:
        return text
    if numeric and not order:
        if len(parts) == 3 and len(parts[0]) == 4:
            order = "ymd"
        elif len(parts) == 3:
            order = "mdy" if lang.split("-")[0] == "en" and not lang.endswith(("GB", "AU", "NZ")) else "dmy"
        elif len(parts) == 2:
            order = "my" if len(parts[1]) == 4 else "md" if lang.split("-")[0] == "en" else "dm"
        else:
            order = "y"
    if order not in _DATE_ORDERS or len(order) != len(parts):
        raise ValueError(f"cannot read {text!r} as {order!r}")
    fields = dict(zip(order, parts))
    year = int(fields["y"]) if "y" in fields else None
    month = int(fields["m"]) if "m" in fields else None
    day = int(fields["d"]) if "d" in fields else None
    if year is not None and year < 100:
        year += 2000 if year < 70 else 1900
    if year and month and day:
        return nice_date(datetime.date(year, month, day), lang)
    spoken = []
    day_words = _number(str(day), lang, ordinal=lang.startswith("en")) if day is not None else None
    month_words = _month_name(month, lang) if month is not None else None
    month_first = lang.split("-")[0] == "en" and not lang.endswith(("GB", "AU", "NZ", "IE", "IN"))
    spoken.extend(w for w in ((month_words, day_words) if month_first else (day_words, month_words)) if w)
    if year is not None:
        spoken.append(_number(str(year), lang))
    return " ".join(spoken)


@functools.lru_cache(maxsize=64)
def _month_names(lang: str) -> dict:
    """Month number -> name, found from what the date parser says for each month.

    The parser only speaks whole dates, so each month's name is the word that all of its sample
    dates share and no other month's samples do (weekday, day and year words differ or are shared
    by every month).
    """
    from ovos_date_parser import nice_date

    samples = ((2001, 1), (2002, 2), (2013, 5), (2024, 17))
    shared = {}
    for month in range(1, 13):
        sets = []
        for year, day in samples:
            words = nice_date(datetime.date(year, month, day), lang).lower().replace(",", " ").split()
            sets.append(words)
        keep = [w for w in sets[0] if all(w in other for other in sets[1:])]
        shared[month] = keep
    names = {}
    for month, words in shared.items():
        others = {w for m, ws in shared.items() if m != month for w in ws}
        unique = [w for w in words if w not in others]
        names[month] = " ".join(unique) if unique else " ".join(words)
    return names


def _month_name(month: int, lang: str) -> str:
    """A month's name in `lang` (empty string if the date parser cannot tell)."""
    return _month_names(lang).get(month, "")


def _time(text: str, lang: str, fmt: Optional[str]) -> str:
    from ovos_date_parser import nice_time

    raw = text.strip().lower().replace(".", ":", 1) if re.fullmatch(r"\d{1,2}\.\d{2}", text.strip()) else text.strip().lower()
    m = re.fullmatch(r"(\d{1,2})\s*[:h]\s*(\d{2})?(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?", raw)
    if not m:
        raise ValueError(f"not a time: {text!r}")
    hour, minute = int(m.group(1)), int(m.group(2) or 0)
    suffix = (m.group(4) or "").replace(".", "")
    if suffix == "pm" and hour < 12:
        hour += 12
    if suffix == "am" and hour == 12:
        hour = 0
    use_24 = bool(fmt and "24" in fmt) or (not suffix and lang.split("-")[0] != "en")
    return nice_time(datetime.time(hour, minute), lang, speech=True, use_24hour=use_24, use_ampm=not use_24)


def _currency(text: str, lang: str) -> str:
    raw = text.strip()
    code = None
    m = re.match(r"^(US\$|CA\$|C\$|R\$|CHF|[$€£¥])\s*([\d.,\s]+)$", raw)
    if m:
        code, amount = _SYMBOLS[m.group(1)], m.group(2)
    else:
        m = re.match(r"^([\d.,\s]+?)\s*(US\$|CA\$|C\$|R\$|CHF|[$€£¥]|[A-Z]{3})$", raw)
        if m:
            amount, symbol = m.group(1), m.group(2)
            code = _SYMBOLS.get(symbol, symbol)
        else:
            m = re.match(r"^([A-Z]{3})\s*([\d.,\s]+)$", raw)
            if m:
                code, amount = m.group(1), m.group(2)
    if code is None:
        raise ValueError(f"not an amount: {text!r}")
    short = lang.split("-")[0]
    words = _CURRENCY_WORDS.get(short, {}).get(code)
    amount = amount.strip()
    decimal = "," if short in ("fr", "es", "de", "pt", "it") else "."
    major_text, _, minor_text = amount.rpartition(decimal) if decimal in amount and re.search(
        rf"\{decimal}\d{{1,2}}$", amount) else (amount, "", "")
    major = int(re.sub(r"[^\d]", "", major_text) or 0)
    minor = int((minor_text + "0")[:2]) if minor_text else 0
    if words is None:
        return f"{_number(str(major), lang)}{' ' + _number(str(minor), lang) if minor else ''} {code}".strip()
    one, many, minor_one, minor_many = words
    out = f"{_number(str(major), lang)} {one if major == 1 else many}"
    if minor:
        out += f" {_AND.get(short, 'and')} {_number(str(minor), lang)} {minor_one if minor == 1 else minor_many}"
    return out


def _address(text: str, lang: str) -> str:
    table = _ADDRESS_ABBREVIATIONS.get(lang.split("-")[0], {})

    def word(match: re.Match) -> str:
        token = match.group(0)
        return table.get(token.rstrip(".").lower(), token) if token.endswith(".") or token.lower() in table else token

    out = re.sub(r"[A-Za-z]+\.?", word, re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text))
    return re.sub(r"\d+", lambda m: _house_number(m.group(0), lang), out)


def _house_number(digits: str, lang: str) -> str:
    """A street number the way it is said: 221 -> two twenty-one, 1600 -> sixteen hundred."""
    if len(digits) == 3:
        return f"{_number(digits[0], lang)} {_number(digits[1:], lang)}" if digits[1] != "0" else _number(digits, lang)
    if len(digits) == 4:
        if digits.endswith("00"):
            return f"{_number(digits[:2], lang)} {'hundred' if lang.startswith('en') else '100'}"
        return f"{_number(digits[:2], lang)} {_number(digits[2:], lang)}"
    if len(digits) <= 2:
        return _number(digits, lang)
    return " ".join(_number(d, lang) for d in digits)


def say_as_text(text: str, interpret_as: str, lang: Optional[str] = None,
                fmt: Optional[str] = None, detail: Optional[str] = None) -> str:
    """The words for `text` read as `interpret_as` in `lang`.

    :param text: The written form, e.g. ``"21"``, ``"1/2"``, ``"2026-11-09"``, ``"$12.50"``.
    :param interpret_as: A ``<say-as>`` interpretation (case-insensitive).
    :param lang: Language of the text (default English).
    :param fmt: The ``format`` attribute, e.g. ``"mdy"`` for a date or ``"hms24"`` for a time.
    :param detail: The ``detail`` attribute (accepted, not yet used).
    :return: Words to synthesize. `text` itself if the interpretation is unknown or the text
        cannot be read that way.
    """
    kind = (interpret_as or "").strip().lower()
    lang = _language(lang)
    try:
        if kind in SPELLED:
            return " ".join(ch for ch in text if not ch.isspace())
        if kind in DIGITS:
            return " ".join(ch for ch in text if ch.isdigit())
        if kind in ("cardinal", "number"):
            return _number(text, lang)
        if kind == "ordinal":
            return _ordinal(text, lang)
        if kind == "fraction":
            return _fraction(text, lang)
        if kind == "unit":
            return _unit(text, lang)
        if kind == "date":
            return _date(text, lang, fmt)
        if kind == "time":
            return _time(text, lang, fmt)
        if kind == "currency":
            return _currency(text, lang)
        if kind == "address":
            return _address(text, lang)
        if kind == "expletive":
            return ""
        # interjection and anything unknown: the written words, as they are.
    except Exception as error:  # noqa: BLE001 - a typo costs the formatting, not the reply
        LOG.info("say-as %s: could not read %r (%s); reading it as written", kind, text, error)
    return text
