"""The unified text front-end: ``str -> (phoneme_ids, language_ids)``.

Ported from hams_tts.text.frontend (Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .codeswitch import Span, segment
from .diacritize import get_diacritizer, is_already_diacritized
from .english_g2p import english_g2p
from .levantine_g2p import arabic_fallback_ipa, levantine_g2p
from .normalize import normalize_unicode, verbalize
from .phoneme_inventory import (
    BOS,
    EOS,
    PUNCT,
    SYL_SEP,
    SYMBOL_TO_ID,
    WORD_SEP,
    Lang,
    tokenize_ipa,
)

_PUNCT_SET = set(PUNCT)


@dataclass
class Utterance:
    """Everything downstream stages needs, plus debug breadcrumbs."""

    text: str
    normalized: str
    ipa: str
    symbols: List[str]
    phoneme_ids: List[int]
    language_ids: List[int]
    spans: List[Span] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.phoneme_ids)

    def pretty(self) -> str:
        rows = [f"text       : {self.text}", f"normalized : {self.normalized}", f"ipa        : {self.ipa}"]
        rows.append("tokens     : " + " ".join(self.symbols))
        rows.append("lang_ids   : " + " ".join(Lang(lid).name[0] if lid in (1, 2) else "·" for lid in self.language_ids))
        return "\n".join(rows)


class TextFrontend:
    def __init__(
        self,
        diacritizer_backend: str = "auto",
        default_lang: str = "en",
        back_emphatics: bool = True,
        epenthesize: bool = True,
        dialectal_case_drop: bool = True,
    ) -> None:
        self.default_lang = default_lang
        self.back_emphatics = back_emphatics
        self.epenthesize = epenthesize
        self.dialectal_case_drop = dialectal_case_drop
        self._diacritizer = get_diacritizer(diacritizer_backend)
        self.diacritizer_backend = diacritizer_backend
        self.has_diacritizer = self._diacritizer is not None

    def _arabic_to_ipa(self, text: str) -> str:
        if is_already_diacritized(text):
            diac = text
        elif self._diacritizer is not None:
            diac = self._diacritizer(text)
        else:
            return arabic_fallback_ipa(text, self.back_emphatics)
        if self.dialectal_case_drop:
            from .dialectal import strip_case_endings
            diac = strip_case_endings(diac)
        return levantine_g2p(diac, self.back_emphatics, self.epenthesize)

    def _span_to_ipa(self, span: Span) -> str:
        verbalized = verbalize(span.text, span.lang)
        if span.lang == "ar":
            return self._arabic_to_ipa(verbalized)
        return english_g2p(verbalized)

    def process(self, text: str) -> Utterance:
        normalized = normalize_unicode(text)
        spans = segment(normalized, default_lang=self.default_lang)

        symbols: List[str] = [BOS]
        language_ids: List[int] = [int(Lang.PAD)]
        ipa_parts: List[str] = []

        for idx, span in enumerate(spans):
            if idx > 0:
                symbols.append(WORD_SEP)
                language_ids.append(int(Lang.NEUTRAL))
            ipa = self._span_to_ipa(span)
            ipa_parts.append(ipa)
            span_lang = Lang.AR if span.lang == "ar" else Lang.EN
            for s in tokenize_ipa(ipa).symbols:
                symbols.append(s)
                if s in _PUNCT_SET or s in (WORD_SEP, SYL_SEP):
                    language_ids.append(int(Lang.NEUTRAL))
                else:
                    language_ids.append(int(span_lang))

        symbols.append(EOS)
        language_ids.append(int(Lang.PAD))

        phoneme_ids = [SYMBOL_TO_ID[s] for s in symbols]
        ipa = "  ".join(ipa_parts)
        assert len(phoneme_ids) == len(language_ids) == len(symbols)
        return Utterance(
            text=text,
            normalized=normalized,
            ipa=ipa,
            symbols=symbols,
            phoneme_ids=phoneme_ids,
            language_ids=language_ids,
            spans=spans,
        )

    __call__ = process


_DEFAULT: Optional[TextFrontend] = None


def get_default_frontend() -> TextFrontend:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = TextFrontend()
    return _DEFAULT
