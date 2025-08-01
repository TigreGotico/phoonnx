from phoonnx.phonemizers.base import BasePhonemizer



class PersianPhonemizer(BasePhonemizer):

    def __init__(self, ipa=True):
        from persian_phonemizer import Phonemizer
        output_format = "IPA" if ipa else 'eraab'
        self.g2p = Phonemizer(output_format)

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
        return cls.match_lang(target_lang, ["fa"])

    def phonemize_string(self, text: str, lang: str = "fa") -> str:
        """
        """
        lang = self.get_lang(lang)
        return self.g2p.phonemize(text)


if __name__ == "__main__":
    text = "دوچرخه جدید علی گم شد."

    pho = PersianPhonemizer()
    lang = "fa"

    print(f"\n--- Getting phonemes for '{text}' ---")
    phonemes_cotovia = pho.phonemize(text, lang)
    print(f"  Phonemes: {phonemes_cotovia}")
