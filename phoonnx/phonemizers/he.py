from phoonnx.config import Alphabet
from phoonnx.phonemizers.base import BasePhonemizer


class PhonikudPhonemizer(BasePhonemizer):

    def __init__(self):
        from phonikud import phonemize
        self.g2p = phonemize
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
        return cls.match_lang(target_lang, ["he"])

    def phonemize_string(self, text: str, lang: str = "he") -> str:
        """
        """
        lang = self.get_lang(lang)
        return self.g2p(text)


if __name__ == "__main__":
    # text = "מתכת יקרה"
    text = 'שָׁלוֹם עוֹלָם'

    pho = PhonikudPhonemizer()
    lang = "he"

    print(f"\n--- Getting phonemes for '{text}' ---")
    # text = pho.add_diacritics(text, lang)
    phonemes = pho.phonemize(text, lang)
    print(f"  Phonemes: {phonemes}")
    # --- Getting phonemes for 'שָׁלוֹם עוֹלָם' ---
    #   Phonemes: [('ʃalˈom ʔolˈam', '.', True)]
