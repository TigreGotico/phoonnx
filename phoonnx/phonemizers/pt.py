import re

from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.config import Alphabet


class TugaphonePhonemizer(BasePhonemizer):

    def __init__(self):
        from tugaphone import TugaPhonemizer
        self.tuga = TugaPhonemizer()
        super().__init__(Alphabet.IPA)

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        """
        Validates and returns the closest supported language code.

        Args:
            target_lang (str): The language code to validate.

        Returns:
            str: The validated language code.

        Raises:
            ValueError: If the language code is unsupported.
        """
        # this check is here only to throw an exception if invalid language is provided
        return cls.match_lang(target_lang, ["pt-PT", "pt-BR", "pt-AO", "pt-MZ", "pt-TL"])

    def phonemize_string(self, text: str, lang: str) -> str:
        lang = self.get_lang(lang)
        # TODO - support regional dialects
        return self.tuga.phonemize_sentence(text, lang)



class BarranquenhoPhonemizer(BasePhonemizer):
    """
    Barranquenho phonemizer backed by ``g2p_barranquenho``, a rule-based
    engine built on the orthography2ipa lattice for the Portuguese/Spanish
    contact variety spoken in Barrancos.  Output is always IPA.

    ``g2p_barranquenho.phonemize`` is word-level (a single word -> a list of IPA
    phoneme strings), so the input is lowercased, split into words, each word is
    phonemized and its phonemes joined, and the words are rejoined with spaces.
    ``g2p_barranquenho`` is imported lazily so importing ``phoonnx`` does not
    hard-require it.
    """

    _LANGS = ["ext-PT-x-barrancos"]

    def __init__(self, alphabet: Alphabet = Alphabet.IPA):
        if alphabet != Alphabet.IPA:
            raise ValueError("g2p_barranquenho only outputs IPA")
        self._phonemize = None
        super().__init__(alphabet)

    @property
    def phonemize_fn(self):
        """Lazily import and cache ``g2p_barranquenho.phonemize``."""
        if self._phonemize is None:
            try:
                from g2p_barranquenho import phonemize
            except ImportError as e:
                raise ImportError(
                    "g2p_barranquenho is required for the Barranquenho phonemizer. "
                    "Install it with 'pip install g2p_barranquenho' "
                    "(or 'pip install phoonnx[pt]')."
                ) from e
            self._phonemize = phonemize
        return self._phonemize

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        """
        Validates and returns the Barranquenho language code.

        Raises:
            ValueError: If the language code is unsupported.
        """
        return cls.match_lang(target_lang, cls._LANGS)

    def phonemize_string(self, text: str, lang: str = "ext-PT-x-barrancos") -> str:
        self.get_lang(lang)
        words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        return " ".join("".join(self.phonemize_fn(w)) for w in words if w)


if __name__ == "__main__":

    pho = TugaphonePhonemizer()

    sentences = [
        "O gato dorme.",
        "Tu falas português muito bem.",
        "O comboio chegou à estação.",
        "A menina comeu o pão todo.",
        "Vou pôr a manteiga no frigorífico.",
        "Ele está a trabalhar no escritório.",
        "Choveu muito ontem à noite.",
        "A rapariga comprou um telemóvel novo.",
        "Vamos tomar um pequeno-almoço.",
        "O carro ficou sem gasolina."
    ]

    for s in sentences:
        print(s)
        for code in ["pt-PT", "pt-BR", "pt-AO", "pt-MZ", "pt-TL"]:
            print(f"{code} → {pho.phonemize_string(s, code)}")
        print("######")

    print("\n--- Barranquenho ---")
    barr = BarranquenhoPhonemizer()
    for s in ["Boca de la casa.", "Cantá manhán aquí."]:
        print(f"{s} → {barr.phonemize_string(s, 'ext-PT-x-barrancos')}")
