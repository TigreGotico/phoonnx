"""Thin wrapper around the ``espeak-ng`` CLI for IPA phonemisation.

Ported from hams_tts.text.espeak (Apache-2.0).  Kept inside thirdparty/shami so the
vendored front-end is self-contained even on machines where the broader phoonnx
espeak phonemizer is unused.
"""

from __future__ import annotations

import functools
import re
import shutil
import subprocess

_ESPEAK = shutil.which("espeak-ng") or shutil.which("espeak")

_PAREN = re.compile(r"\([^)]*\)")
_ZW = dict.fromkeys(map(ord, "‍‌​﻿"), None)


def available() -> bool:
    return _ESPEAK is not None


def binary() -> str | None:
    return _ESPEAK


@functools.lru_cache(maxsize=8192)
def phonemize(text: str, voice: str = "en-us") -> str:
    """Return a space-delimited IPA string (words separated by single spaces)."""
    if not _ESPEAK:
        raise RuntimeError("espeak-ng not found on PATH")
    if not text.strip():
        return ""
    proc = subprocess.run(
        [_ESPEAK, "-v", voice, "--ipa=1", "-q", "--", text],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    raw = proc.stdout or ""
    raw = _PAREN.sub(" ", raw).translate(_ZW)
    words = []
    for w in raw.split():
        phon = w.replace("_", "").strip()
        if phon:
            words.append(phon)
    return " ".join(words)
