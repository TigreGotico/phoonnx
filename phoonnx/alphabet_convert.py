"""Alphabet-keyed grapheme→script conversion step.

Provides a reusable mapping from :class:`~phoonnx.config.Alphabet` values to
text-transform callables, plus :func:`convert_to_alphabet` which any engine can
invoke before tokenization to normalise input text into the script representation
the model was trained on.

The underlying transforms live in :mod:`phoonnx.lang_preprocess`; this module
only re-exports them via an :data:`ALPHABET_CONVERTERS` dict so callers are
decoupled from the language-specific details.

.. note::
   ``ChatterboxMTLTokenizer`` has its own ``_script_transform`` dispatch keyed on
   ISO language codes.  That dispatch is intentionally left in place for now to
   avoid double-converting multilingual Chatterbox text.

   # TODO: migrate ChatterboxMTLTokenizer to convert_to_alphabet
"""
from typing import Callable

from phoonnx.config import Alphabet
from phoonnx.lang_preprocess import (
    hangul_to_jamo,
    japanese_to_hiragana,
    chinese_to_cangjie,
)

# Alphabets that require a script-conversion step before tokenization.
# Keys are Alphabet enum values; values are pure-text callables (str → str).
# Alphabets absent from this mapping (IPA, UNICODE, ARPA, …) are pass-through.
ALPHABET_CONVERTERS: dict[Alphabet, Callable[[str], str]] = {
    Alphabet.HANGUL: hangul_to_jamo,
    Alphabet.HIRA: japanese_to_hiragana,
    Alphabet.CANGJIE: chinese_to_cangjie,
}


def convert_to_alphabet(text: str, alphabet: Alphabet) -> str:
    """Convert *text* to the script representation required by *alphabet*.

    Returns *text* unchanged when no converter is registered for *alphabet*
    (e.g. :attr:`~phoonnx.config.Alphabet.UNICODE`, :attr:`~phoonnx.config.Alphabet.IPA`).

    Parameters
    ----------
    text:
        Raw grapheme input text.
    alphabet:
        The target alphabet/script declared by the voice's
        :class:`~phoonnx.config.VoiceConfig`.

    Returns
    -------
    str
        Script-converted text, or *text* if no conversion is needed.
    """
    converter = ALPHABET_CONVERTERS.get(alphabet)
    if converter is None:
        return text
    return converter(text)
