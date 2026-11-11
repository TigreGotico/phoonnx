"""Deprecated: text normalization comes from :mod:`text2spoken`.

``normalize`` calls :func:`text2spoken.normalize`. text2spoken has no equivalent
of the helpers and tables below, so they keep their local code in the submodules
of this package until the package is removed. Each name warns when it is read.
"""
import warnings
from importlib import import_module

import text2spoken

from phoonnx.version import VERSION_MAJOR

REMOVAL_VERSION = f"{VERSION_MAJOR + 1}.0.0"

# The name, and the submodule that holds its local code.
_LOCAL_NAMES = {
    "CONTRACTIONS": "tables",
    "TITLES": "tables",
    "UNITS": "tables",
    "_get_number_separators": "numbers",
    "_normalize_number_word": "numbers",
    "is_fraction": "numbers",
    "_normalize_dates_and_times": "datetimes",
    "pronounce_date": "datetimes",
    "pronounce_time": "datetimes",
    "_normalize_units": "units",
    "_normalize_word": "text",
    "_normalize_word_hyphen_digit": "text",
}

DEPRECATED_NAMES = frozenset({"normalize", *_LOCAL_NAMES})


def _normalize(text: str, lang: str, expand_contractions: bool = True) -> str:
    """:func:`text2spoken.normalize`. text2spoken leaves contractions as written, so
    ``expand_contractions`` changes nothing."""
    return text2spoken.normalize(text, lang)


def __getattr__(name: str):
    if name not in DEPRECATED_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(f"phoonnx.normalization.{name} is deprecated and is removed in phoonnx "
                  f"{REMOVAL_VERSION}. Use text2spoken.normalize.", DeprecationWarning, stacklevel=2)
    if name == "normalize":
        return _normalize
    return getattr(import_module(f"{__name__}.{_LOCAL_NAMES[name]}"), name)
