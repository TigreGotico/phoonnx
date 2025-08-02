from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.config import Alphabet

class OpenJTaklPhonemizer(BasePhonemizer):

    def __init__(self, alphabet=Alphabet.IPA):
        assert alphabet in [Alphabet.HEPBURN, Alphabet.KANA]
        import pyopenjtalk
        self.g2p = pyopenjtalk.g2p
        super().__init__(alphabet)

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
        return cls.match_lang(target_lang, ["ja"])

    def phonemize_string(self, text: str, lang: str = "ja") -> str:
        """
        """
        lang = self.get_lang(lang)
        return self.g2p(text, kana=self.alphabet == Alphabet.KANA)


class CutletPhonemizer(BasePhonemizer):

    def __init__(self, alphabet=Alphabet.HEPBURN, use_foreign_spelling = False):
        assert alphabet in [Alphabet.HEPBURN, Alphabet.KUNREI, Alphabet.NIHON]
        # If `use_foreign_spelling` is true, output will use the foreign spelling
        #         provided in a UniDic lemma when available. For example, "カツ" will
        #         become "cutlet" instead of "katsu".
        import cutlet
        self.g2p = cutlet.Cutlet(alphabet)
        self.g2p.use_foreign_spelling = use_foreign_spelling
        super().__init__(alphabet)

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
        return cls.match_lang(target_lang, ["ja"])

    def phonemize_string(self, text: str, lang: str = "ja") -> str:
        """
        """
        lang = self.get_lang(lang)
        return self.g2p.romaji(text)

class PyKakasiPhonemizer(BasePhonemizer):

    def __init__(self, alphabet=Alphabet.HEPBURN):
        assert alphabet in [Alphabet.HEPBURN, Alphabet.KANA, Alphabet.HIRA]
        # kana, hira, hepburn
        import pykakasi
        self.g2p = pykakasi.kakasi()
        super().__init__(alphabet)

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
        return cls.match_lang(target_lang, ["ja"])

    def phonemize_string(self, text: str, lang: str = "ja") -> str:
        """
        """
        lang = self.get_lang(lang)
        return " ".join([
            a[self.alphabet] for a in
            self.g2p.convert(text)
        ])


if __name__ == "__main__":
    text = "こんにちは"
    text = "カツカレーは美味しい"
    lang = "ja"

    pho = OpenJTaklPhonemizer(kana=True)
    pho = CutletPhonemizer()
    pho = PyKakasiPhonemizer(Alphabet.KANA)
    from phoonnx.phonemizers.mul import MisakiPhonemizer
    #pho = MisakiPhonemizer()
    print(f"\n--- Getting phonemes for '{text}' ---")
    phonemes_cotovia = pho.phonemize(text, lang)
    print(f"  Phonemes: {phonemes_cotovia}")
