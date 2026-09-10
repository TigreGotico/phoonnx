"""Character-level tokenizer for SuperTonic.

The released model treats text as a sequence of Unicode characters: input is
NFKD-normalised, wrapped in a ``<lang>...</lang>`` tag, and each character is
looked up in an indexer that maps its Unicode code point to a small integer id.
Training only needs a bijection from the characters that actually occur in the
corpus to contiguous ids, so the indexer is built lazily from the data rather
than shipping the full 65536-entry table; :meth:`CharTokenizer.to_indexer_list`
expands it back to that table shape at export time for the inference engine.

Tokenizer keys are code points and are used verbatim — NFKD applies to input
*text*, never to the stored key set.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List

AVAILABLE_LANGS = [
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr",
    "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk",
    "sl", "sv", "tr", "uk", "vi", "na",
]

CODEPOINT_RANGE = 65536  # uint16 space, matching the released unicode_indexer.json

_TERMINAL = re.compile(r"[.!?;:,'\"')\]}…。」』】〉》›»]$")


def normalize_text(text: str, lang: str) -> str:
    """NFKD-normalise, collapse whitespace, ensure a terminal punctuation mark,
    and wrap the result in a ``<lang>...</lang>`` tag."""
    if lang not in AVAILABLE_LANGS:
        raise ValueError(f"unsupported language {lang!r}; expected one of {AVAILABLE_LANGS}")
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and not _TERMINAL.search(text):
        text += "."
    return f"<{lang}>{text}</{lang}>"


@dataclass
class CharTokenizer:
    """Maps characters to contiguous ids. Id ``0`` is reserved for padding /
    unknown characters."""

    char2id: Dict[str, int] = field(default_factory=dict)
    PAD_ID: int = 0

    @classmethod
    def build_from_texts(cls, texts: List[str], langs: List[str]) -> "CharTokenizer":
        chars = set()
        for text, lang in zip(texts, langs):
            chars.update(normalize_text(text, lang))
        return cls(char2id={c: i + 1 for i, c in enumerate(sorted(chars))})

    def extend_with_texts(self, texts: List[str], langs: List[str]) -> "CharTokenizer":
        """New tokenizer keeping every existing id (so a pretrained embedding
        stays row-aligned) and appending ids for previously unseen characters."""
        chars = set()
        for text, lang in zip(texts, langs):
            chars.update(normalize_text(text, lang))
        new = sorted(chars - self.char2id.keys())
        nxt = max(self.char2id.values(), default=0) + 1
        merged = dict(self.char2id)
        for c in new:
            merged[c] = nxt
            nxt += 1
        return CharTokenizer(char2id=merged)

    def encode(self, text: str, lang: str) -> List[int]:
        return [self.char2id.get(c, self.PAD_ID) for c in normalize_text(text, lang)]

    @property
    def vocab_size(self) -> int:
        return (max(self.char2id.values(), default=0)) + 1

    def to_dict(self) -> Dict[str, int]:
        return dict(self.char2id)

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "CharTokenizer":
        return cls(char2id=dict(d))

    def to_indexer_list(self, size: int = CODEPOINT_RANGE) -> List[int]:
        """Expand to a ``size``-long list mapping ``codepoint -> id`` (the shape
        the inference engine's ``unicode_indexer.json`` uses). Code points not in
        the vocab map to :attr:`PAD_ID`."""
        table = [self.PAD_ID] * size
        for char, idx in self.char2id.items():
            cp = ord(char)
            if cp < size:
                table[cp] = idx
        return table
