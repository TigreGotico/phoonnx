"""
orthography2ipa-backed phonemizer for phoonnx.

Wraps orthography2ipa.G2P to provide a BasePhonemizer-compatible interface
for 387+ language codes.  Output is always IPA (Alphabet.IPA).
"""
from typing import Dict

from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.config import Alphabet


class Orthography2IPAPhonemizer(BasePhonemizer):
    """
    Data-driven IPA phonemizer backed by orthography2ipa.

    Supports every language code returned by ``orthography2ipa.available_codes()``
    (387+ codes as of 0.2.0a1), plus any BCP-47 tag that
    ``orthography2ipa.resolve`` can map to a supported code.  Per-language
    quality varies; orthography2ipa provides rule-based G2P that is strongest
    for languages with regular orthographies.

    The G2P engine for each resolved code is loaded lazily on first use and
    cached for the lifetime of this instance.
    """

    def __init__(self):
        super().__init__(alphabet=Alphabet.IPA)
        self._cache: Dict[str, object] = {}

    def _engine(self, resolved_lang: str):
        """Return (and lazily initialise) the G2P engine for *resolved_lang*."""
        if resolved_lang not in self._cache:
            import orthography2ipa
            self._cache[resolved_lang] = orthography2ipa.G2P(resolved_lang)
        return self._cache[resolved_lang]

    @classmethod
    def supported_langs(cls):
        """Return the list of language codes supported by orthography2ipa."""
        import orthography2ipa
        return orthography2ipa.available_codes()

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        """
        Resolve *target_lang* to the canonical code used by orthography2ipa.

        Uses ``orthography2ipa.resolve`` which applies BCP-47 closest-match
        logic internally.  Raises ``ValueError`` if no match is found.
        """
        import orthography2ipa
        try:
            resolved = orthography2ipa.resolve(target_lang)
            # Verify the resolved code actually has a G2P engine (probe it)
            orthography2ipa.G2P(resolved)
            return resolved
        except (KeyError, Exception) as exc:
            raise ValueError(
                f"orthography2ipa: unsupported language {target_lang!r}"
            ) from exc

    def phonemize_string(self, text: str, lang: str) -> str:
        resolved = self.get_lang(lang)
        return self._engine(resolved).transcribe(text)
