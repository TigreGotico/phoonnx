

from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.thirdparty.hangul2ipa import hangul2ipa
from phoonnx.config import Alphabet


class G2PKPhonemizer(BasePhonemizer):

    def __init__(self, descriptive=True, group_vowels=True, to_syl=True,
                 alphabet=Alphabet.IPA):
        assert alphabet in [Alphabet.IPA, Alphabet.HANGUL]
        from g2pk import G2p
        self.g2p = G2p()
        self.descriptive = descriptive
        self.group_vowels = group_vowels
        self.to_syl = to_syl
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
        return cls.match_lang(target_lang, ["ko"])

    def phonemize_string(self, text: str, lang: str = "ko") -> str:
        """
        """
        lang = self.get_lang(lang)
        p = self.g2p(text, descriptive=self.descriptive,
                     group_vowels=self.group_vowels,
                     to_syl=self.to_syl)
        if self.alphabet == Alphabet.IPA:
            return hangul2ipa(p)
        return p


class KoG2PPhonemizer(BasePhonemizer):
    """https://github.com/scarletcho/KoG2P"""
    def __init__(self,   alphabet=Alphabet.IPA):
        assert alphabet in [Alphabet.IPA, Alphabet.HANGUL]
        from phoonnx.thirdparty.kog2p import runKoG2P
        self.g2p = runKoG2P
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
        return cls.match_lang(target_lang, ["ko"])

    def phonemize_string(self, text: str, lang: str = "ko") -> str:
        """
        """
        lang = self.get_lang(lang)
        p = self.g2p(text)
        if self.alphabet == Alphabet.IPA:
            return hangul2ipa(p)
        return p


if __name__ == "__main__":

    pho = G2PKPhonemizer(ipa=False)
    pho2 = KoG2PPhonemizer(ipa=False)
    lang = "ko"

    text = "터미널에서 원하는 문자열을 함께 입력해 사용할 수 있습니다."
    print(f"\n--- Getting phonemes for '{text}' ---")
    phonemes_cotovia = pho.phonemize(text, lang)
    print(f"  G2PK Phonemes: {phonemes_cotovia}")

    phonemes_cotovia = pho2.phonemize(text, lang)
    print(f"  KoG2P Phonemes: {phonemes_cotovia}")

