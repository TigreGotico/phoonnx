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
           "BRL": ("real", "reais", "centavo", "centavos"), "GBP": ("libra", "libras", "péni", "pence")},
    "it": {"USD": ("dollaro", "dollari", "centesimo", "centesimi"), "EUR": ("euro", "euro", "centesimo", "centesimi"),
           "GBP": ("sterlina", "sterline", "penny", "penny")},
    "nl": {"USD": ("dollar", "dollar", "cent", "cent"), "EUR": ("euro", "euro", "cent", "cent"),
           "GBP": ("pond", "pond", "penny", "pence")},
}
_AND = {"en": "and", "fr": "et", "es": "con", "de": "und", "pt": "e", "it": "e", "nl": "en"}
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


# The number parser takes ages on absurd values (a 5000-digit "number" stalled it for minutes):
# past this many digits a value is read digit by digit instead.
_MAX_DIGITS = 15
_MAX_TEXT = 200  # longest run of text read as a value


def _digit_words(digits: str, lang: str) -> str:
    from ovos_number_parser import pronounce_number

    return " ".join(pronounce_number(int(d), lang=lang) for d in digits if d.isdigit())


def _plain_amount(value: str, lang: str) -> str:
    """`value` as digits with '.' for the decimal mark, whichever separators it was written with.

    The last separator is the decimal mark when it is followed by one or two digits (or four or
    more), or when the other separator also appears; three digits after a lone separator are
    thousands in the usual convention of `lang` ("1,234" in English, "1.234" in German) and
    decimals otherwise. Raises ValueError for anything that is not a number.
    """
    text = re.sub(r"[\s\u00a0\u202f']", "", value.strip())
    if not re.fullmatch(r"[+-]?\d[\d.,]*", text):
        raise ValueError(f"not a number: {value!r}")
    marks = [(i, c) for i, c in enumerate(text) if c in ".,"]
    if not marks:
        return text
    index, mark = marks[-1]
    after = text[index + 1:]
    kinds = {c for _, c in marks}
    if len(marks) > 1 and len(kinds) == 1:
        decimal = None                       # "1,234,567": only thousands marks
    elif len(kinds) > 1:
        decimal = mark                       # "1.234,5": the last one is the decimal mark
    elif len(after) == 3 and not text.lstrip("+-").startswith("0" + mark):
        usual = "," if lang.split("-")[0] in ("en", "ja", "zh", "ko", "he", "th") else "."
        decimal = None if mark == usual else mark
    else:
        decimal = mark
    if decimal is None:
        return re.sub(r"[.,]", "", text)
    return f"{re.sub(r'[.,]', '', text[:index])}.{after}"


def _number(value: str, lang: str, ordinal: bool = False) -> str:
    """The words for a number written with the thousands and decimal marks of `lang`."""
    from ovos_number_parser import pronounce_number

    text = _plain_amount(value, lang)
    whole_digits = text.lstrip("+-").partition(".")[0]
    if len(whole_digits) > _MAX_DIGITS or len(text) > 4 * _MAX_DIGITS:
        return _digit_words(text[:60], lang)
    number = float(text) if "." in text else int(text)
    if ordinal:
        return _ordinal_words(int(number), lang)
    if isinstance(number, float):
        whole, _, fraction = text.lstrip("+-").partition(".")
        short = lang.split("-")[0]
        point = _DECIMAL_WORD.get(short) or _parser_decimal_word(short)
        if point is None:
            # The parser's own decimal reading; a language it has no words for raises, and the
            # value is then read as written.
            said = pronounce_number(abs(number), lang=lang, places=len(fraction))
        else:
            # Say the two halves ourselves: the parser drops a zero whole part in German, reads
            # the decimals as one number in Portuguese, leaves them as digits above 999 in French,
            # and truncates a float in the Nordic languages (44,29 is read 44,28).
            digits = " ".join(pronounce_number(int(d), lang=lang) for d in fraction)
            said = f"{pronounce_number(int(whole or 0), lang=lang)} {point} {digits}"
        if text.startswith("-"):
            # The sign is said apart, in the parser's own word for minus: int("-0") has no sign,
            # and the parser drops the fraction of a negative value in some languages.
            one = pronounce_number(1, lang=lang)
            said = f"{pronounce_number(-1, lang=lang).removesuffix(one).strip()} {said}"
        return said
    said = pronounce_number(number, lang=lang)
    return said


@functools.lru_cache(maxsize=64)
def _parser_decimal_word(lang: str) -> Optional[str]:
    """The word the number parser says between the halves of 1.5, when its reading has that shape."""
    from ovos_number_parser import pronounce_number

    try:
        said = pronounce_number(1.5, lang=lang, places=1)
        one, five = pronounce_number(1, lang=lang), pronounce_number(5, lang=lang)
    except (NotImplementedError, ValueError):
        return None
    if said.startswith(one) and said.endswith(five):
        return said[len(one):len(said) - len(five)].strip() or None
    return None


_DECIMAL_WORD = {"en": "point", "fr": "virgule", "es": "coma", "pt": "vírgula", "de": "Komma",
                 "it": "virgola", "nl": "komma"}


def _masculine(words: str, lang: str) -> str:
    """The parser's ordinals come out feminine in Spanish, Italian and French: make them masculine."""
    short = lang.split("-")[0]
    if short in ("es", "it"):
        return " ".join(w[:-1] + "o" if w.endswith("a") else w for w in words.split())
    if short == "fr" and words.endswith("première"):
        return words[: -len("première")] + "premier"
    return words


def _ordinal_words(number: int, lang: str) -> str:
    from ovos_number_parser import pronounce_number, pronounce_ordinal

    if len(str(number)) > _MAX_DIGITS:
        return _digit_words(str(number), lang)
    try:
        words = pronounce_ordinal(number, lang)
    except Exception:  # noqa: BLE001 - not every language has one: fall back to the generic form
        words = pronounce_number(number, lang=lang, ordinals=True)
    return _masculine(words, lang)


def _ordinal(text: str, lang: str) -> str:
    digits = re.sub(r"[^\d]", "", text)[:60]
    return _ordinal_words(int(digits), lang)


def _fraction(text: str, lang: str) -> str:
    from ovos_number_parser import pronounce_fraction

    text = text.strip()
    if any(len(run) > 6 for run in re.findall(r"\d+", text)):
        raise ValueError("fraction too large to say")
    mixed = re.fullmatch(r"(\d+)\s+(\d+)\s*/\s*(\d+)", text)
    if mixed:
        whole, num, den = mixed.groups()
        join = _AND.get(lang.split("-")[0], "and")
        return f"{_number(whole, lang)} {join} {pronounce_fraction(f'{num}/{den}', lang)}"
    simple = re.fullmatch(r"(\d+)\s*/\s*(\d+)", text)
    if simple:
        return pronounce_fraction(f"{simple.group(1)}/{simple.group(2)}", lang)
    return _number(text, lang)


_PER = {  # "/h", "/s", "/min" -> "per hour" ...
    "en": ("per", "hour", "second", "minute"), "fr": ("par", "heure", "seconde", "minute"),
    "es": ("por", "hora", "segundo", "minuto"), "de": ("pro", "Stunde", "Sekunde", "Minute"),
    "pt": ("por", "hora", "segundo", "minuto"), "it": ("a", "ora", "secondo", "minuto"),
    "nl": ("per", "uur", "seconde", "minuut"),
}
# Units the text normalizer leaves as symbols in these languages: (singular, plural).
_SYMBOL_UNITS = {
    "it": {"km": ("chilometro", "chilometri"), "m": ("metro", "metri"), "cm": ("centimetro", "centimetri"),
           "mm": ("millimetro", "millimetri"), "kg": ("chilogrammo", "chilogrammi"), "g": ("grammo", "grammi"),
           "l": ("litro", "litri"), "ml": ("millilitro", "millilitri"), "°C": ("grado Celsius", "gradi Celsius"),
           "%": ("per cento", "per cento")},
    "nl": {"km": ("kilometer", "kilometer"), "m": ("meter", "meter"), "cm": ("centimeter", "centimeter"),
           "mm": ("millimeter", "millimeter"), "kg": ("kilogram", "kilogram"), "g": ("gram", "gram"),
           "l": ("liter", "liter"), "ml": ("milliliter", "milliliter"), "°C": ("graad Celsius", "graden Celsius"),
           "%": ("procent", "procent")},
}


def _unit(text: str, lang: str) -> str:
    from phoonnx.normalization.units import _normalize_units

    if re.search(r"\d{%d,}" % (_MAX_DIGITS + 1), text):
        raise ValueError("number too long to say as a quantity")
    short = lang.split("-")[0]
    spaced = re.sub(r"(?<=\d)(?=[^\d\s.,/-])", " ", text.strip())
    # The quantity is read here, not by the unit normalizer, which reads "1,5 L" as "eins Komma
    # fünf null null Liter" in German and "1 200 km" as "un deux cents": it is given a 1 or a 2
    # (for the singular or plural of the unit) and the real number put back in its place.
    spaced = re.sub(r"(?<=\d )l\b", "L", spaced)  # the normalizer knows litres as "L" only
    lead = re.match(r"([+-]?\d(?:[\d.,\u00a0\u202f ]*\d)?)(?=\s*[^\d\s.,])", spaced)
    stand_in = real = None
    if lead:
        try:
            amount = float(_plain_amount(lead.group(1), lang))
        except ValueError:
            amount = None
        if amount is not None:
            stand_in = "1" if amount == 1 else "2"
            real = _number(lead.group(1), lang)
            spaced = stand_in + spaced[lead.end():]
    said = _normalize_units(spaced, lang)
    table = _SYMBOL_UNITS.get(short, {})
    if table:  # the normalizer left these as symbols: say them
        def symbol(match: re.Match) -> str:
            one, many = table[match.group(2)]
            return f"{match.group(1)} {one if match.group(1).strip() in ('1', '1.0') else many}"

        said = re.sub(r"(\d+(?:[.,]\d+)?)\s*(°C|km|cm|mm|ml|kg|m|g|l|%)(?![\w°])", symbol, said)
    per = _PER.get(short, _PER["en"])
    for suffix, word in (("h", per[1]), ("s", per[2]), ("min", per[3])):
        glue = "'" if short == "it" and per[0] == "a" else " "
        said = re.sub(rf"/{suffix}\b", f" {per[0]}{glue}{word}" if glue == " " else f" all'{word}", said)
    # Whatever the unit table did not turn into words, the number still can be.
    said = re.sub(r"\d+(?:[.,]\d+)?", lambda m: _number(m.group(0), lang), said)
    if stand_in is not None:
        placeholder = _number(stand_in, lang)
        if said.startswith(placeholder):
            rest = said[len(placeholder):]
            if stand_in == "1" and short in ("en", "fr", "es", "pt"):
                rest = _singular(rest)
            if stand_in == "1" and real == placeholder:
                # "uno kilómetro", "eins Kilogramm": before a noun it is "un" and "ein".
                real = {"es": "un", "it": "un", "de": "ein"}.get(short, real)
            said = real + rest
    return said


def _singular(rest: str) -> str:
    """" kilograms per hour" -> " kilogram per hour": the normalizer pluralises the unit after a 1."""
    found = re.match(r"(\s+)([^\s]{4,})", rest)
    if not found or not found.group(2).endswith("s") or found.group(2).endswith(("ss", "ius")):
        return rest
    word = found.group(2)
    word = word[:-2] if word.endswith(("ches", "shes")) else word[:-1]
    return found.group(1) + word + rest[found.end():]


_DATE_ORDERS = {"ymd", "ydm", "mdy", "myd", "dmy", "dym", "ym", "my", "md", "dm", "y", "m", "d"}


def _written_date(parts: list, lang: str) -> Optional[tuple]:
    """``(order, parts)`` for a date with its month written out ("1er mai 2026"), else None.

    The month becomes its number so the date is read like a numeric one; a day may carry an
    ordinal suffix ("1er", "1st", "2e", "3º").
    """
    months = {word: number for number, name in _month_names(lang).items() for word in name.split()}
    if lang.split("-")[0] == "en":
        months.update({w: i + 1 for i, w in enumerate(
            "january february march april may june july august september october november december".split())})
    order, numbers = "", []
    for part in parts:
        if part.lower() in ("de", "of", "del", "the", "van"):  # "9 de noviembre de 2026", "1st of May"
            continue
        bare = re.sub(r"(?<=\d)(?:st|nd|rd|th|er|re|e|ème|º|°)$", "", part.lower())
        if part.lower() in months and "m" not in order:
            order, numbers = order + "m", numbers + [str(months[part.lower()])]
        elif bare.isdigit() and len(bare) == 4 and "y" not in order:
            order, numbers = order + "y", numbers + [bare]
        elif bare.isdigit() and len(bare) <= 2 and "d" not in order:
            order, numbers = order + "d", numbers + [bare]
        else:
            return None
    return (order, numbers) if "m" in order and order in _DATE_ORDERS else None


def _date(text: str, lang: str, fmt: Optional[str]) -> str:
    from ovos_date_parser import nice_date

    parts = [p for p in re.split(r"[-/.\s,]+", text.strip()) if p]
    # A weekday is said only when the text has one ("lun. 10/11/2026"): the date alone is just a date.
    weekday = next((day for day in (_weekday_of(p, lang) for p in parts) if day is not None), None)
    parts = [p for p in parts if _weekday_of(p, lang) is None]
    numeric = all(p.isdigit() for p in parts)
    order = (fmt or "").lower()
    if not numeric:
        written = _written_date(parts, lang)  # "9 novembre 2026", "November 9, 2026"
        if written is None:
            if not order:
                return text
        else:
            order, parts = written
            numeric = True
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
        date = datetime.date(year, month, day)
        said_weekday = _weekday_names(lang).get(weekday) if weekday is not None else None
        if lang.split("-")[0] in _COMPOSED_DATES:
            said = _composed_date(date, lang)
        else:
            said = nice_date(date, lang)
            first = said.split(maxsplit=1)
            if first and first[0].strip(",").lower() == _weekday_names(lang).get(date.weekday()):
                said = first[1] if len(first) > 1 else ""   # the parser says a weekday nobody wrote
        if said_weekday and lang.split("-")[0] == "fr":
            said = said.removeprefix("le ")  # "lundi dix novembre", not "lundi le dix novembre"
        return f"{said_weekday} {said}" if said_weekday else said
    spoken = []
    day_words = _number(str(day), lang, ordinal=lang.startswith("en")) if day is not None else None
    month_words = _month_name(month, lang) if month is not None else None
    month_first = lang.split("-")[0] == "en" and not lang.endswith(("GB", "AU", "NZ", "IE", "IN"))
    spoken.extend(w for w in ((month_words, day_words) if month_first else (day_words, month_words)) if w)
    if year is not None:
        spoken.append(_year_words(year, lang))
    return " ".join(spoken)


# Languages whose date formatter reads the year badly ("zwei tausend sechs und zwanzig"): build the
# date from the number and month words, which are right. {d} day, {m} month, {y} year.
_COMPOSED_DATES = {"en": "the {d} of {m}, {y}", "fr": "le {d} {m} {y}", "de": "{d} {m} {y}", "nl": "{d} {m} {y}", "es": "{d} de {m} de {y}",
                   "pt": "{d} de {m} de {y}", "it": "{d} {m} {y}"}


def _composed_date(date: datetime.date, lang: str) -> str:
    short = lang.split("-")[0]
    first = date.day == 1
    if short in ("de", "en"):
        day = _ordinal_words(date.day, lang)
    elif short in ("it", "es", "pt", "fr") and first:
        day = _ordinal_words(1, lang)
    else:
        day = _number(str(date.day), lang)
    return _COMPOSED_DATES[short].format(d=day, m=_month_name(date.month, lang),
                                         y=_year_words(date.year, lang))


def _year_words(year: int, lang: str) -> str:
    """A year as a person says it: "twenty twenty six" in English, the number elsewhere."""
    if lang.split("-")[0] == "en" and (1100 <= year <= 1999 or 2010 <= year <= 2099):
        high, low = divmod(year, 100)
        return f"{_number(str(high), lang)} " + ("hundred" if not low else f"oh {_number(str(low), lang)}"
                                                 if low < 10 else _number(str(low), lang))
    if lang.split("-")[0] == "en" and 2000 <= year <= 2009:
        return "two thousand" + (f" {_number(str(year - 2000), lang)}" if year > 2000 else "")
    return _number(str(year), lang)


@functools.lru_cache(maxsize=64)
def _weekday_names(lang: str) -> dict:
    """Weekday number (Monday 0) -> name, the first word of what the date parser says for that day."""
    from ovos_date_parser import nice_date

    names = {}
    for offset in range(7):
        date = datetime.date(2026, 11, 9 + offset)
        words = nice_date(date, lang).lower().split()
        names[offset] = words[0].strip(",") if words else ""
    return names


def _weekday_of(token: str, lang: str) -> Optional[int]:
    """The weekday number a written token names ("lundi", "lun.", "Mon"), else None."""
    word = token.lower().strip(".")
    if len(word) < 3 or word.isdigit():
        return None
    for number, name in _weekday_names(lang).items():
        if name and (word == name or name.startswith(word)):
            return number
    return None


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


_TIME = re.compile(r"(\d{1,2})(?:\s*[:h]\s*(\d{2})?(?::(\d{2}))?)?\s*(?:(?:uhr|uur|horas?|ore|heures?)\b)?"
                   r"\s*(a\.?\s?m\.?|p\.?\s?m\.?)?")
# A bare number is not a time: one of these must say so ("9 pm", "14 h", "14:30", "14 Uhr").
_TIME_MARK = re.compile(r"[:h]|uhr|uur|hora|ore|heure|[ap]\.?\s?m")


def _time(text: str, lang: str, fmt: Optional[str]) -> str:
    from ovos_date_parser import nice_time

    # "14.30" is a time in most of Europe; the dot goes before the pattern sees it.
    raw = re.sub(r"^(\d{1,2})\.(\d{2})\b", r"\1:\2", text.strip().lower())
    m = _TIME.fullmatch(raw)
    if not m or not _TIME_MARK.search(raw):
        raise ValueError(f"not a time: {text!r}")
    hour, minute = int(m.group(1)), int(m.group(2) or 0)
    suffix = re.sub(r"[.\s]", "", m.group(4) or "")
    if suffix and not 1 <= hour <= 12:
        raise ValueError(f"not a 12-hour time: {text!r}")
    if suffix == "pm" and hour < 12:
        hour += 12
    if suffix == "am" and hour == 12:
        hour = 0
    short = lang.split("-")[0]
    use_24 = bool(fmt and "24" in fmt) or (not suffix and short != "en")
    said = nice_time(datetime.time(hour, minute), lang, speech=True, use_24hour=use_24, use_ampm=not use_24)
    if short == "en":
        # "p.m." ends in a full stop of its own: before the stop that ends the sentence it read
        # "p.m.." and pronounced the pause twice. Say the letters.
        said = re.sub(r"\b([ap])\.m\.?", lambda found: f"{found.group(1).upper()} M", said)
    elif short == "fr":
        # "heure" is feminine, so its half is "demie"; "midi" and "minuit" are masculine ("midi et demi").
        said = re.sub(r"\b(heures?) et demi\b", r"\1 et demie", said)
    return said


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
    plain = _plain_amount(amount, lang)
    whole, _, cents = plain.partition(".")
    major = int(whole or 0)
    minor = int((cents + "0")[:2]) if cents else 0
    if words is None:
        # No words for the currency: the code, with "and" so that 1200 and 50 are not 1250.
        out = f"{_number(str(major), lang)} {code}"
        return out + (f" {_AND.get(short, 'and')} {_number(str(minor), lang)}" if minor else "")
    one, many, minor_one, minor_many = words
    major_words = _number(str(major), lang)
    if major == 1 and one == "sterlina":
        major_words = "una"
    if one in ("libra",) and short in ("es", "pt"):
        major_words = _feminine(major_words, short)
    out = f"{major_words} {one if major == 1 else many}"
    if minor:
        out += f" {_AND.get(short, 'and')} {_number(str(minor), lang)} {minor_one if minor == 1 else minor_many}"
    return out


def _feminine(words: str, short: str) -> str:
    """Spanish and Portuguese numbers before a feminine noun: "doscientas libras", "una libra"."""
    last = {"es": {"uno": "una", "veintiuno": "veintiuna"}, "pt": {"um": "uma", "dois": "duas"}}[short]
    parts = words.split()
    if parts:
        parts[-1] = last.get(parts[-1], parts[-1])
    return re.sub(r"(ient|ent)os\b", r"\1as", " ".join(parts))


def _address(text: str, lang: str) -> str:
    table = _ADDRESS_ABBREVIATIONS.get(lang.split("-")[0], {})

    def word(match: re.Match) -> str:
        token = match.group(0)
        return table.get(token.rstrip(".").lower(), token) if token.endswith(".") or token.lower() in table else token

    out = re.sub(r"[A-Za-z]+\.?", word, re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text))
    return re.sub(r"\d+", lambda m: _house_number(m.group(0), lang), out)


def _house_number(digits: str, lang: str) -> str:
    """A street number the way it is said: in English 221 -> two twenty-one and 1600 -> sixteen
    hundred; other languages say the whole number."""
    if not lang.startswith("en"):
        return _number(digits, lang)
    if len(digits) == 3:
        return f"{_number(digits[0], lang)} {_number(digits[1:], lang)}" if digits[1] != "0" else _number(digits, lang)
    if len(digits) == 4:
        if digits.endswith("00"):
            return f"{_number(digits[:2], lang)} hundred"
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
    if len(text) > _MAX_TEXT and kind not in SPELLED | DIGITS:
        return text  # not a value, whatever it is called: say it as written
    text = text if isinstance(text, str) else str(text)
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
        LOG.info("say-as %s: could not read %.60r (%s); reading it as written", kind, text, error)
    return text
