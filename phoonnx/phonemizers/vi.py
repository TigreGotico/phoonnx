from phoonnx.phonemizers.base import BasePhonemizer


class VIPhonemePhonemizer(BasePhonemizer):

    def __init__(self):
        from viphoneme import vi2IPA
        self.g2p = vi2IPA

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
        return cls.match_lang(target_lang, ["vi"])

    def phonemize_string(self, text: str, lang: str = "vi") -> str:
        """
        """
        lang = self.get_lang(lang)
        return self.g2p(text)


if __name__ == "__main__":
    text = "Được viết vào 6/4/2020, có thể xử lí những trường hợp chứa English"

    pho = VIPhonemePhonemizer()
    lang = "vi"

    print(f"\n--- Getting phonemes for '{text}' ---")
    phonemes_cotovia = pho.phonemize(text, lang)
    print(f"  Phonemes: {phonemes_cotovia}")
