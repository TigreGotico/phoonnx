"""Text normalisation for the Levantine/English code-switching front-end.

Ported from hams_tts.text.normalize (Apache-2.0).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

try:
    from num2words import num2words as _num2words
    _HAVE_NUM2WORDS = True
except Exception:
    _HAVE_NUM2WORDS = False

TATWEEL = "ـ"
ZWJ = "‍"
ZWNJ = "‌"
ARABIC_DIATRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")

_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})

_PUNCT_MAP = {
    "،": ",", "؍": ",", "٫": ".", "٬": ",",
    "؛": ";", "؟": "?", "٪": "%", "−": "-",
    "“": '"', "”": '"', "„": '"', "‟": '"', "’": "'", "‘": "'",
    "–": "—", "―": "—", "‒": "—",
    " ": " ", " ": " ", "\t": " ",
}
_PUNCT_TABLE = {ord(k): v for k, v in _PUNCT_MAP.items()}

_LEV_ONES = {
    0: "صِفِر", 1: "واحَد", 2: "تْنين", 3: "تْلاتة", 4: "أَربَعة",
    5: "خَمسة", 6: "سِتّة", 7: "سَبعة", 8: "تْمانية", 9: "تِسعة",
}
_LEV_TEENS = {
    10: "عَشَرة", 11: "إِحْداعَش", 12: "إِتْناعَش", 13: "تْلَتّاعَش",
    14: "أَربَعتاعَش", 15: "خَمستاعَش", 16: "سِتّاعَش", 17: "سَبعتاعَش",
    18: "تْمانتاعَش", 19: "تِسعتاعَش",
}
_LEV_TENS = {
    20: "عِشرين", 30: "تْلاتين", 40: "أَربعين", 50: "خَمسين",
    60: "سِتّين", 70: "سَبعين", 80: "تْمانين", 90: "تِسعين",
}
_AR_HUNDREDS = {
    1: "مية", 2: "ميتين", 3: "تْلاتمية", 4: "أَربَعمية", 5: "خَمسمية",
    6: "سِتّمية", 7: "سَبعمية", 8: "تْمانمية", 9: "تِسعمية",
}


def arabic_int_to_words(n: int) -> str:
    if n < 0:
        return "ناقِص " + arabic_int_to_words(-n)
    if n < 10:
        return _LEV_ONES[n]
    if n < 20:
        return _LEV_TEENS[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        if ones == 0:
            return _LEV_TENS[tens * 10]
        return f"{_LEV_ONES[ones]} وْ{_LEV_TENS[tens * 10]}"
    if n < 1000:
        hund, rem = divmod(n, 100)
        head = _AR_HUNDREDS[hund]
        if rem == 0:
            return head
        return f"{head} وْ{arabic_int_to_words(rem)}"
    if n < 1_000_000:
        thou, rem = divmod(n, 1000)
        if thou == 1:
            head = "أَلف"
        elif thou == 2:
            head = "أَلفين"
        elif thou <= 10:
            head = f"{arabic_int_to_words(thou)} تْلاف"
        else:
            head = f"{arabic_int_to_words(thou)} أَلف"
        if rem == 0:
            return head
        return f"{head} وْ{arabic_int_to_words(rem)}"
    mill, rem = divmod(n, 1_000_000)
    if mill == 1:
        head = "مِليون"
    elif mill == 2:
        head = "مِليونين"
    else:
        head = f"{arabic_int_to_words(mill)} مَلايين"
    if rem == 0:
        return head
    return f"{head} وْ{arabic_int_to_words(rem)}"


def english_int_to_words(n: int) -> str:
    if _HAVE_NUM2WORDS:
        return _num2words(n)
    return _english_fallback(n)


_EN_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_EN_SCALES = [(1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand"), (100, "hundred")]


def _english_fallback(n: int) -> str:
    if n < 0:
        return "minus " + _english_fallback(-n)
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _EN_TENS[t] + (("-" + _EN_ONES[o]) if o else "")
    for value, name in _EN_SCALES:
        if n >= value:
            head, rem = divmod(n, value)
            out = f"{_english_fallback(head)} {name}"
            if rem:
                joiner = " and " if value == 100 or rem < 100 else " "
                out += joiner + _english_fallback(rem)
            return out
    return _EN_ONES[n]


_CURRENCY = {
    "$": ("dollar", "dollars", "دولار"),
    "£": ("pound", "pounds", "جْنيه"),
    "€": ("euro", "euros", "يورو"),
    "₪": ("shekel", "shekels", "شيكِل"),
    "د.إ": ("dirham", "dirhams", "دِرهَم"),
    "ر.س": ("riyal", "riyals", "ريال"),
    "ل.ل": ("lira", "lira", "ليرة"),
    "ل.س": ("lira", "lira", "ليرة"),
    "ج.م": ("pound", "pounds", "جْنيه"),
}


def _verbalise_currency(symbol: str, amount: str, lang: str) -> str:
    sg, pl, ar = _CURRENCY.get(symbol, ("", "", ""))
    if "." in amount:
        whole, frac = amount.split(".", 1)
    else:
        whole, frac = amount, ""
    whole_i = int(whole) if whole.isdigit() else 0
    if lang == "ar":
        words = arabic_int_to_words(whole_i) + " " + ar
        if frac and int(frac):
            words += " وْ" + arabic_int_to_words(int(frac)) + " قِرش"
        return words
    unit = sg if whole_i == 1 else pl
    words = f"{english_int_to_words(whole_i)} {unit}"
    if frac and int(frac):
        words += f" and {english_int_to_words(int(frac))} cents"
    return words


def normalize_unicode(text: str) -> str:
    """NFC + cosmetic cleanup that is safe for TTS."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(TATWEEL, "").replace(ZWJ, "").replace(ZWNJ, "")
    text = text.translate(_PUNCT_TABLE)
    text = text.translate(_DIGIT_MAP)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_diacritics(text: str) -> str:
    """Remove Arabic tashkeel."""
    return ARABIC_DIATRITICS.sub("", text)


_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_CURRENCY_RE = re.compile(
    r"([$£€₪]|د\.إ|ر\.س|ل\.ل|ل\.س|ج\.م)\s*(\d+(?:[.,]\d+)?)"
    r"|(\d+(?:[.,]\d+)?)\s*([$£€₪])"
)
_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def verbalize(text: str, lang: str) -> str:
    """Expand numbers / currency / percentages / times into spoken words for ``lang``."""
    is_ar = lang == "ar"

    def _int_words(n: int) -> str:
        return arabic_int_to_words(n) if is_ar else english_int_to_words(n)

    def _cur(m: re.Match) -> str:
        if m.group(1):
            return " " + _verbalise_currency(m.group(1), m.group(2).replace(",", "."), lang) + " "
        return " " + _verbalise_currency(m.group(4), m.group(3).replace(",", "."), lang) + " "

    text = _CURRENCY_RE.sub(_cur, text)

    def _pct(m: re.Match) -> str:
        val = m.group(1)
        num = _float_words(val, is_ar)
        tail = "بالمية" if is_ar else "percent"
        return f" {num} {tail} "

    text = _PERCENT_RE.sub(_pct, text)

    def _time(m: re.Match) -> str:
        h, mn = int(m.group(1)), int(m.group(2))
        if is_ar:
            base = "السّاعة " + _int_words(h)
            return base if mn == 0 else f"{base} وْ{_int_words(mn)} دْقيقة"
        if mn == 0:
            return f" {_int_words(h)} o'clock "
        return f" {_int_words(h)} {_int_words(mn)} "

    text = _TIME_RE.sub(_time, text)

    def _num(m: re.Match) -> str:
        return " " + _float_words(m.group(0).replace(",", "."), is_ar) + " "

    text = _NUMBER_RE.sub(_num, text)

    return re.sub(r"\s+", " ", text).strip()


def _float_words(token: str, is_ar: bool) -> str:
    neg = token.startswith("-")
    token = token.lstrip("-")
    if "." in token:
        whole, frac = token.split(".", 1)
    else:
        whole, frac = token, ""
    whole_i = int(whole) if whole else 0
    words = arabic_int_to_words(whole_i) if is_ar else english_int_to_words(whole_i)
    if frac:
        point = "فاصِل" if is_ar else "point"
        digit_words = " ".join(
            (arabic_int_to_words(int(d)) if is_ar else english_int_to_words(int(d)))
            for d in frac
        )
        words = f"{words} {point} {digit_words}"
    if neg:
        words = ("ناقِص " if is_ar else "minus ") + words
    return words


def normalize(text: str, lang: Optional[str] = None) -> str:
    """Convenience: unicode-normalise then (if ``lang`` given) verbalise."""
    text = normalize_unicode(text)
    if lang is not None:
        text = verbalize(text, lang)
    return text
