"""Spoken forms for the date and time patterns that survive into raw text."""
import datetime
import re
from datetime import date

from ovos_date_parser import nice_time, nice_date

from phoonnx.log import LOG

def pronounce_date(date_obj: date, full_lang: str) -> str:
    """
    Pronounces a date object using ovos-date-parser.
    """
    return nice_date(date_obj, full_lang)


def pronounce_time(time_string: str, full_lang: str) -> str:
    """
    Pronounces a time string using ovos-date-parser.
    Handles military time like "15h01" and converts it to a
    datetime.time object before passing it to nice_time.
    """
    try:
        hours, mins = time_string.split("h")
        time_obj = datetime.time(int(hours), int(mins))
        # Use nice_time from ovos-date-parser
        return nice_time(time_obj, full_lang, speech=True, use_24hour=True, use_ampm=False)
    except Exception as e:
        LOG.warning(f"Failed to parse time string '{time_string}': {e}")
        return time_string.replace("h", " ")


def _normalize_dates_and_times(text: str, full_lang: str, date_format: str = "DMY") -> str:
    """
    Helper function to normalize dates and times using regular expressions.
    This prepares the strings for pronunciation.
    """
    lang_code = full_lang.split("-")[0]
    # Pre-process with regex to handle English am/pm times.
    # Anchored to a preceding digit so ordinary words containing "am"/"pm"
    # (e.g. "I am happy", "spam") are never touched.
    if lang_code == "en":
        text = re.sub(
            r"(?i)\b(\d+)\s*([ap])\.?m\.?\b",
            lambda m: f"{m.group(1)} {m.group(2).upper()} M",
            text,
        )

    # Normalize times like "15h01" to words
    time_pattern = re.compile(r"(\d{1,2})h(\d{2})", re.IGNORECASE)

    def replace_time(match):
        time_str = match.group(0)
        return pronounce_time(time_str, full_lang)

    text = time_pattern.sub(replace_time, text)

    # Find dates like "DD/MM/YYYY" or "YYYY/MM/DD"
    date_pattern = re.compile(r"(\d{1,4})[/-](\d{1,2})[/-](\d{1,4})")

    # Expand every date found in the text, not just the first one.
    result_parts = []
    pos = 0
    for match in date_pattern.finditer(text):
        # Get the three parts of the date string
        part1_str, part2_str, part3_str = match.groups()
        p1, p2, p3 = int(part1_str), int(part2_str), int(part3_str)

        # Initialize month, day, and year
        month, day, year = None, None, None

        # Determine year first based on length (4 digits)
        if len(part1_str) == 4:
            year, rest_parts = p1, [p2, p3]
        elif len(part3_str) == 4:
            year, rest_parts = p3, [p1, p2]
        else:
            # If no 4-digit year, it's ambiguous, assume a 2-digit year.
            # We'll assume the last part is the year based on common patterns.
            year = p3
            # Expand 2-digit year to 4-digit year
            if year < 100:
                # Assume years 00-29 are 2000-2029, 30-99 are 1930-1999
                year = 2000 + year if year < 30 else 1900 + year
            rest_parts = [p1, p2]

        # From the remaining parts, try to determine day and month
        if day is None and any(p > 12 and len(str(p)) == 2 for p in rest_parts):
            # If a two-digit number is > 12, it's a day
            day_candidate = next((p for p in rest_parts if p > 12), None)
            if day_candidate:
                day = day_candidate
                rest_parts.remove(day_candidate)
                month = rest_parts[0]

        # Fallback to date_format if day/month are still ambiguous
        if day is None or month is None:
            if date_format.lower() == "mdy":
                month, day = rest_parts[0], rest_parts[1]
            else:  # default to DD/MM/YY
                day, month = rest_parts[0], rest_parts[1]

        try:
            date_obj = date(year, month, day)
            replacement = pronounce_date(date_obj, full_lang)
        except (ValueError, IndexError) as e:
            LOG.warning(f"Could not parse date from '{match.group(0)}': {e}")
            replacement = match.group(0)

        result_parts.append(text[pos:match.start()])
        result_parts.append(replacement)
        pos = match.end()

    result_parts.append(text[pos:])
    text = "".join(result_parts)

    return text
