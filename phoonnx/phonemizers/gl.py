from typing import Optional

from pycotovia import Phonemizer as _PycotoviaPhonemizer, cotovia_to_ipa as _cotovia_to_ipa

from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.config import Alphabet


class CotoviaPhonemizer(BasePhonemizer):
    """
    Galician phonemizer backed by pycotovia — a pure-Python port of the Cotovia
    G2P engine that has verified parity with the original C binary.

    Output alphabets:
    - ``Alphabet.COTOVIA`` — raw Cotovia phoneme notation (e.g. ``"Este e uN ..."``)
    - ``Alphabet.IPA`` — IPA string produced by pycotovia's ``cotovia_to_ipa``

    Voices trained on Cotovia-alphabet output continue to receive the same
    notation strings as before because pycotovia is binary-parity-tested
    (see pycotovia/docs/parity.md).
    """

    def __init__(self, alphabet: Alphabet = Alphabet.IPA):
        self._engine: Optional[_PycotoviaPhonemizer] = None
        super().__init__(alphabet)

    @property
    def engine(self) -> _PycotoviaPhonemizer:
        if self._engine is None:
            self._engine = _PycotoviaPhonemizer()
        return self._engine

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        return cls.match_lang(target_lang, ["gl-ES"])

    def phonemize_string(self, text: str, lang: str) -> str:
        self.get_lang(lang)
        cotovia_output = self.engine.phonemize(text)
        if self.alphabet == Alphabet.IPA:
            return _cotovia_to_ipa(cotovia_output)
        return cotovia_output
