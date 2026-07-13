"""Arabic automatic diacritisation (tashkeel).

Ported from hams_tts.text.diacritize (Apache-2.0).
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from .normalize import ARABIC_DIATRITICS

_ARABIC_LETTER = re.compile(r"[ء-ي]")


def diacritic_ratio(text: str) -> float:
    letters = _ARABIC_LETTER.findall(text)
    if not letters:
        return 1.0
    marks = ARABIC_DIATRITICS.findall(text)
    return len(marks) / max(1, len(letters))


def is_already_diacritized(text: str, threshold: float = 0.35) -> bool:
    return diacritic_ratio(text) >= threshold


def _camel_backend() -> Optional[Callable[[str], str]]:
    try:
        from camel_tools.disambig.mle import MLEDisambiguator

        mle = MLEDisambiguator.pretrained()

        def _run(text: str) -> str:
            out_tokens = []
            for word in text.split():
                disambig = mle.disambiguate([word])
                if disambig and disambig[0].analyses:
                    diac = disambig[0].analyses[0].analysis.get("diac", word)
                    out_tokens.append(diac)
                else:
                    out_tokens.append(word)
            return " ".join(out_tokens)

        return _run
    except Exception:
        return None


def _catt_backend(model_id: str = "facebook/mms-tts-ara") -> Optional[Callable[[str], str]]:
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        catt_id = "MagedSaeed/catt-encoder-only"
        AutoTokenizer.from_pretrained(catt_id)
        AutoModelForTokenClassification.from_pretrained(catt_id)

        def _run(text: str) -> str:
            raise NotImplementedError("CATT decode runs on the GPU server")

        return _run
    except Exception:
        return None


def _passthrough_backend() -> Callable[[str], str]:
    return lambda text: text


_FACTORIES = {
    "camel": _camel_backend,
    "catt": _catt_backend,
    "passthrough": lambda: _passthrough_backend(),
}


def get_diacritizer(backend: str = "auto") -> Optional[Callable[[str], str]]:
    if backend == "passthrough":
        return _passthrough_backend()
    if backend == "auto":
        for name in ("camel", "catt"):
            fn = _FACTORIES[name]()
            if fn is not None:
                return fn
        return None
    factory = _FACTORIES.get(backend)
    if factory is None:
        raise ValueError(f"unknown diacritiser backend: {backend!r}")
    return factory()
