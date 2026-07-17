"""orthography2ipa-backed phonemizer for phoonnx.

Wraps ``orthography2ipa.G2P`` to expose the shared grapheme->IPA lattice as a
``BasePhonemizer``.  A single instance covers the whole orthography2ipa family
(hundreds of language specs: the Arabic lects, Basque, Galician, Mirandese,
Portuguese, the East-Slavic set, ...): one backend, many languages, always IPA.

Per-language quality varies -- the lattice is strongest for regular
orthographies.  For undiacritized Arabic use ``ArbtokPhonemizer`` (which adds a
diacritizer on top of this same lattice); for dialect-aware Basque use
``EuskaphonePhonemizer``.

``orthography2ipa`` is imported lazily so importing ``phoonnx`` never requires
it; install with ``pip install orthography2ipa`` (or ``pip install phoonnx[o2i]``).
"""
from typing import Dict, List

from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.config import Alphabet


class Orthography2IPAPhonemizer(BasePhonemizer):
    """
    Data-driven multilingual IPA phonemizer backed by orthography2ipa.

    Accepts any language code that ``orthography2ipa.resolve`` can map to a
    supported spec (BCP-47 closest-match logic lives in orthography2ipa itself,
    so it is not reinvented here).  The per-language G2P engine is created
    lazily on first use and cached for the lifetime of the instance.
    """

    def __init__(self):
        super().__init__(alphabet=Alphabet.IPA)
        self._cache: Dict[str, object] = {}

    def _engine(self, resolved_lang: str):
        """Return, lazily creating and caching, the G2P engine for *resolved_lang*."""
        if resolved_lang not in self._cache:
            import orthography2ipa
            self._cache[resolved_lang] = orthography2ipa.G2P(resolved_lang)
        return self._cache[resolved_lang]

    @classmethod
    def supported_langs(cls) -> List[str]:
        """Return every language code orthography2ipa ships a spec for."""
        import orthography2ipa
        return orthography2ipa.available_codes()

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        """
        Resolve *target_lang* to the canonical orthography2ipa spec code.

        Delegates BCP-47 resolution to ``orthography2ipa.resolve``.

        Raises:
            ValueError: If orthography2ipa has no spec for *target_lang*.
        """
        try:
            import orthography2ipa
        except ImportError as e:
            raise ImportError(
                "orthography2ipa is required for the Orthography2IPA phonemizer. "
                "Install it with 'pip install orthography2ipa' "
                "(or 'pip install phoonnx[o2i]')."
            ) from e
        # ``resolve`` echoes an unknown tag back rather than raising, so probe
        # the resolved code by constructing its G2P engine (raises if no spec).
        resolved = orthography2ipa.resolve(target_lang)
        try:
            orthography2ipa.G2P(resolved)
        except Exception as exc:
            raise ValueError(
                f"orthography2ipa: unsupported language {target_lang!r}"
            ) from exc
        return resolved

    def phonemize_string(self, text: str, lang: str) -> str:
        resolved = self.get_lang(lang)
        return self._engine(resolved).transcribe(text)


if __name__ == "__main__":
    pho = Orthography2IPAPhonemizer()
    for lang, text in [
        ("ar", "ذهب الطالب إلى المكتبة لقراءة كتاب"),
        ("eu", "Kaixo, mundua."),
        ("gl", "Boas, o mundo."),
        ("pt", "Olá, quem são vocês?"),
        ("mwl", "Buonos dies, mundo."),
    ]:
        print(f"{lang}: {pho.phonemize_string(text, lang)!r}")
