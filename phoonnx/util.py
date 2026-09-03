"""Language-tag identity helpers.

The text normalization pipeline lives in :mod:`phoonnx.normalization`; the
names it used to export are re-exported here so existing imports keep working.
"""
import re
import unicodedata
from typing import List, Tuple, Union

from langcodes import Language, standardize_tag, tag_distance
from scriptconv.scripts import normalize_script_tag

from phoonnx.log import LOG
from phoonnx.normalization import (
    CONTRACTIONS,
    TITLES,
    UNITS,
    _get_number_separators,
    _normalize_dates_and_times,
    _normalize_number_word,
    _normalize_units,
    _normalize_word,
    _normalize_word_hyphen_digit,
    is_fraction,
    normalize,
    pronounce_date,
    pronounce_time,
)

def _private_use(names: List[str]) -> List[str]:
    """ASCII-fold dialect names into valid BCP-47 private-use subtags (1-8 alnum)."""
    tags = []
    for name in names:
        ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
        ascii_name = re.sub(r"[^a-z0-9]", "", ascii_name.lower())
        tags += [ascii_name[i:i + 8] for i in range(0, len(ascii_name), 8)]
    return [t for t in tags if t]


# Languages official/major in more than one country: the region is a real choice the
# voice must declare, so we never guess one for these (a "pt" voice is as likely BR
# as PT). For everything else a bare code has a single unambiguous CLDR region.
_MULTI_REGION = frozenset({
    "en", "es", "pt", "fr", "ar", "de", "nl", "it", "ru", "zh", "sw", "ms",
    "hi", "bn", "ta", "ur", "pa", "ne", "fa", "ku", "ps", "az",
    "ha", "ff", "yo", "om", "ln", "ti", "aa", "ee", "ny", "so", "wo", "kr",
    "qu", "ay", "gn", "ca", "eu", "sq", "sv", "sr", "bs", "la", "os",
})


def normalize_lang(lang: str) -> str:
    if lang == "tgl" or lang == "tl":
        # HACK: langcodes erroneously (debatable) changes this one to "fil"
        return "tl"
    # MMS tags scripts ("crk-script_syllabics") and dialects ("cak-central-dialect")
    # with unregistered words. Map scripts to ISO-15924 subtags; keep dialects as a
    # BCP-47 private-use subtag (cak-central-dialect -> cak-x-central) so the
    # distinction survives instead of being erased.
    parts = re.split(r"[-_]", lang.lower())
    if len(parts) > 1:
        script = next((normalize_script_tag(p) for p in parts[1:] if normalize_script_tag(p)), None)
        if script:
            lang = f"{parts[0]}-{script}"
        elif "dialect" in parts[1:]:
            tags = _private_use([p for p in parts[1:] if p != "dialect"])
            lang = f"{parts[0]}-x-" + "-".join(tags) if tags else parts[0]
    try:
        lang = standardize_tag(lang)
    except Exception:
        return lang
    # Add the CLDR region for bare 639-1 codes that map to a single country (cy ->
    # cy-GB). Skip multi-region languages, whose region must come from the model.
    if re.fullmatch(r"[a-z]{2}", lang) and lang not in _MULTI_REGION:
        try:
            territory = Language.get(lang).maximize().territory
            if territory and len(territory) == 2:  # ISO-3166 alpha-2, not "419"
                lang = f"{lang}-{territory}"
        except Exception:
            pass
    return lang


def match_lang(target_lang: str, valid_langs: Union[str, List[str]]) -> Tuple[str, int]:
    """
    Validates and returns the closest supported language code.

    Args:
        target_lang (str): The language code to validate.

    Returns:
        str: The validated language code.

    Raises:
        ValueError: If the language code is unsupported.
    """
    if isinstance(valid_langs, str):
        valid_langs = [valid_langs]
    if target_lang in valid_langs:
        return target_lang, 0
    best_lang = "und"
    best_distance = 10000000
    for l in valid_langs:
        try:
            distance: int = tag_distance(l, target_lang)
        except:
            try:
                l = f"{l.split('-')[0]}-{l.split('-')[1]}"
                distance: int = tag_distance(l, target_lang)
            except:
                try:
                    distance: int = tag_distance(l.split('-')[0], target_lang)
                except:
                    continue
        if distance < best_distance:
            best_lang, best_distance = l, distance

    # If the score is low (meaning a good match), return the language
    if best_distance <= 10:
        return best_lang, best_distance
    return "und", 10000
