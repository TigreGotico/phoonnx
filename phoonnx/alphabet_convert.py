"""Generic pairwise alphabet-conversion util for phoonnx.

Two alphabets live in phoonnx's synthesis pipeline:

* **VoiceConfig.alphabet** — the *model-expected* alphabet: the script or phoneme
  representation the ONNX model was trained to consume.  Set by the model's
  ``config.json``; callers must not change it at synthesis time.

* **SynthesisConfig.alphabet** — the *user-input* alphabet: the representation the
  caller provides to :meth:`~phoonnx.voice.TTSVoice.synthesize`.  ``None`` means
  "same as the model's alphabet" (no conversion needed).

The bridge between them is :func:`convert`::

    convert(text, src=syn_config.alphabet or voice.config.alphabet,
                  dst=voice.config.alphabet)

Registered converters
---------------------
The :data:`ALPHABET_CONVERTERS` dict maps ``(src, dst)`` pairs to callables:

+-----------------------------+---------------------------------------------+
| Pair                        | Transform                                   |
+=============================+=============================================+
| ``(UNICODE, HANGUL)``       | Hangul syllables → conjoining Jamo (NFD)    |
+-----------------------------+---------------------------------------------+
| ``(UNICODE, HIRA)``         | Kanji → hiragana via *pykakasi* (opt. dep)  |
+-----------------------------+---------------------------------------------+
| ``(UNICODE, CANGJIE)``      | Hanzi → Cangjie tokens via *spacy-pkuseg*   |
|                             | + HF ``Cangjie5_TC.json`` (opt. dep)        |
+-----------------------------+---------------------------------------------+

Missing optional deps degrade to identity with a warning; the engine never
raises due to a missing converter.

.. note::
   Phonemizers (IPA/ARPA etc.) are *not* routed through this module — they run in
   :meth:`~phoonnx.voice.TTSVoice.phonemize` after this step and already emit the
   model's alphabet.  :func:`convert` only bridges *script/grapheme* alphabets
   (cases where the model consumes modified text, not phoneme IDs).
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
# Registry: (src_alphabet, dst_alphabet) -> Callable[[str], str]
# ---------------------------------------------------------------------------

ALPHABET_CONVERTERS: dict[tuple[Alphabet, Alphabet], Callable[[str], str]] = {
    (Alphabet.UNICODE, Alphabet.HANGUL): hangul_to_jamo,
    (Alphabet.UNICODE, Alphabet.HIRA): japanese_to_hiragana,
    (Alphabet.UNICODE, Alphabet.CANGJIE): chinese_to_cangjie,
}


def convert(text: str, src: Alphabet, dst: Alphabet) -> str:
    """Convert *text* from the *src* alphabet representation to *dst*.

    Returns *text* unchanged when:

    * ``src == dst`` (identity — no conversion needed), or
    * the ``(src, dst)`` pair is not in :data:`ALPHABET_CONVERTERS` (graceful
      identity; the missing pair is logged at DEBUG level).

    Parameters
    ----------
    text:
        Input text in the *src* alphabet's representation.
    src:
        Source :class:`~phoonnx.config.Alphabet` — the caller's representation.
        Pass ``SynthesisConfig.alphabet`` (falling back to ``VoiceConfig.alphabet``
        when the former is ``None``).
    dst:
        Target :class:`~phoonnx.config.Alphabet` — what the model expects.
        Pass ``VoiceConfig.alphabet``.

    Returns
    -------
    str
        Converted text, or *text* unchanged when no conversion is needed or
        available.
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
