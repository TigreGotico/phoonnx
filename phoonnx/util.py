"""Language-tag identity helpers.

Text normalization comes from :mod:`text2spoken`. The normalization names of
:mod:`phoonnx.normalization` stay readable here, deprecated, until that package
is removed.
"""
import re
import unicodedata
from typing import List, Tuple, Union

from linguatag import lang_distance, standardize_lang, with_default_region
from scriptconv.scripts import normalize_script_tag

from phoonnx.log import LOG
from phoonnx import normalization as _normalization

def _private_use(names: List[str]) -> List[str]:
    """ASCII-fold dialect names into valid BCP-47 private-use subtags (1-8 alnum)."""
    tags = []
    for name in names:
        ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
        ascii_name = re.sub(r"[^a-z0-9]", "", ascii_name.lower())
        tags += [ascii_name[i:i + 8] for i in range(0, len(ascii_name), 8)]
    return [t for t in tags if t]


# linguatag measures `ar` with any region, and each individual Arabic language
# (`arz`, `apc`, ...) in Arabic script, at 5 or less from `ar`. A tag in another
# language or in another script is further away.
_ARABIC_DISTANCE = 5


def is_arabic(lang: str) -> bool:
    """Whether ``lang`` names Arabic or an individual Arabic language, in Arabic script."""
    return lang_distance(lang, "ar") <= _ARABIC_DISTANCE


def normalize_lang(lang: str, default_region: bool = True) -> str:
    """``lang`` in standard form, with its default region when ``default_region``.

    Lookup and matching use the default region. A voice's phonemizer takes
    ``default_region=False``: it gets the language its config names, because a
    trained voice must keep the phonemes it was trained on.
    """
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
    # The default region is the place where the language first appeared and is
    # still spoken (en-GB, pt-PT). A language with no default region stays bare.
    return with_default_region(lang) if default_region else standardize_lang(lang)


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
        distance = lang_distance(l, target_lang)
        if distance < best_distance:
            best_lang, best_distance = l, distance

    # If the score is low (meaning a good match), return the language
    if best_distance <= 10:
        return best_lang, best_distance
    return "und", 10000


def __getattr__(name: str):
    if name in _normalization.DEPRECATED_NAMES:
        return getattr(_normalization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
