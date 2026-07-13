"""Vendored Shami (Levantine Arabic / English code-switching) text front-end.

Ported from hams_tts.text (Apache-2.0) so phoonnx can drive ShamiVITS ONNX models
without requiring the hams-tts package at runtime.
"""

from .frontend import TextFrontend, Utterance, get_default_frontend
from .phoneme_inventory import (
    Lang,
    SYMBOL_TO_ID,
    ID_TO_SYMBOL,
    VOCAB_SIZE,
    encode,
    decode,
    tokenize_ipa,
)

__all__ = [
    "TextFrontend",
    "Utterance",
    "get_default_frontend",
    "Lang",
    "SYMBOL_TO_ID",
    "ID_TO_SYMBOL",
    "VOCAB_SIZE",
    "encode",
    "decode",
    "tokenize_ipa",
]
