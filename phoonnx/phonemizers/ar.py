from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.thirdparty.mantoq import g2p as mantoq


class MantoqPhonemizer(BasePhonemizer):

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
        return cls.match_lang(target_lang, ["ar"])

    def phonemize_string(self, text: str, lang: str = "ar") -> str:
        """
        """
        lang = self.get_lang(lang)
        normalized_text, phonemes = mantoq(text)
        return " ".join(phonemes).replace("_+_", " ")


if __name__ == "__main__":
    text = "مرحبا بالعالم"
    # gets normalized to
    # مَرْحَبًا بِالْعالَم

    pho = MantoqPhonemizer()
    print(pho.phonemize(text, "ar"))
    # [('m a r H a b a n aa   b i l E aa l a m', '.', True)]
