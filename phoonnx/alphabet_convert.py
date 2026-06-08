"""Generic pairwise alphabet-conversion util.

Provides :func:`convert` — a ``(src_alphabet, dst_alphabet)`` keyed dispatch — plus
the original :func:`convert_to_alphabet` shim for back-compat.

The underlying transforms live in :mod:`phoonnx.lang_preprocess`; this module only
re-exports them via :data:`ALPHABET_CONVERTERS` so callers are decoupled from the
language-specific details.

Engine seam
-----------
Engines and adapters that need a script-conversion step should call
:func:`convert` directly with the ``(src, dst)`` pair they require.
``TTSVoice.synthesize`` provides a *default* hook that calls
``convert(text, Alphabet.UNICODE, self.config.alphabet)`` as a best-effort
pass-through when the engine has not opted into a custom pairing.  Engines that
need a different source alphabet (e.g. a romanisation step from ARPA) should
perform their own :func:`convert` call in ``encode_text`` and rely on the fact
that the default hook is a no-op when ``src == dst`` or no converter is
registered.

.. note::
   ``ChatterboxMTLTokenizer`` has its own ``_script_transform`` dispatch keyed on
   ISO language codes.  That dispatch is intentionally left in place to avoid
   double-converting multilingual Chatterbox text.
"""
import logging
from typing import Callable

from phoonnx.config import Alphabet
from phoonnx.lang_preprocess import (
    hangul_to_jamo,
    japanese_to_hiragana,
    chinese_to_cangjie,
)

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry: keyed by (src_alphabet, dst_alphabet) pairs.
# Values are pure text callables (str → str).
# ---------------------------------------------------------------------------

ALPHABET_CONVERTERS: dict[tuple[Alphabet, Alphabet], Callable[[str], str]] = {
    (Alphabet.UNICODE, Alphabet.HANGUL): hangul_to_jamo,
    (Alphabet.UNICODE, Alphabet.HIRA): japanese_to_hiragana,
    (Alphabet.UNICODE, Alphabet.CANGJIE): chinese_to_cangjie,
}


def convert(text: str, src: Alphabet, dst: Alphabet) -> str:
    """Convert *text* from *src* alphabet representation to *dst*.

    Returns *text* unchanged when:

    * ``src == dst`` (identity — no conversion needed), or
    * no converter is registered for the ``(src, dst)`` pair (graceful
      identity with a debug-level log).

    Parameters
    ----------
    text:
        Input text in the *src* alphabet's representation.
    src:
        Source :class:`~phoonnx.config.Alphabet`.
    dst:
        Target :class:`~phoonnx.config.Alphabet`.

    Returns
    -------
    str
        Converted text, or *text* unmodified when conversion is not needed or
        not available.
    """
    if src == dst:
        return text
    converter = ALPHABET_CONVERTERS.get((src, dst))
    if converter is None:
        LOG.debug(
            "alphabet_convert: no converter registered for (%s, %s); returning text unchanged",
            src,
            dst,
        )
        return text
    return converter(text)


# ---------------------------------------------------------------------------
# Back-compat shim — callers passing only a dst alphabet continue to work.
# ---------------------------------------------------------------------------

def convert_to_alphabet(text: str, alphabet: Alphabet) -> str:
    """Convert *text* to the script representation required by *alphabet*.

    Back-compat wrapper around :func:`convert` that assumes
    :attr:`~phoonnx.config.Alphabet.UNICODE` as the source.

    Parameters
    ----------
    text:
        Raw grapheme / Unicode input text.
    alphabet:
        Target alphabet declared by the voice's
        :class:`~phoonnx.config.VoiceConfig`.

    Returns
    -------
    str
        Script-converted text, or *text* if no conversion is needed.
    """
    return convert(text, Alphabet.UNICODE, alphabet)
