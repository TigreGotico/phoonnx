from phoonnx.phonemizers.base import BasePhonemizer, Alphabet
from mwl_phonemizer import CRFOrthoCorrector


class MirandesePhonemizer(BasePhonemizer):
    _LANGS = ["mwl"]

    def __init__(self):
        super().__init__(Alphabet.IPA)
        self.pho = CRFOrthoCorrector()

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
        return cls.match_lang(target_lang, cls._LANGS)

    def phonemize_string(self, text: str, lang: str) -> str:
        # Validate language is supported
        lang = self.get_lang(lang)
        return self.pho.phonemize_sentence(text)


if __name__ == "__main__":
    pho = MirandesePhonemizer()
    print(pho.phonemize_string("ls", "mwl"))