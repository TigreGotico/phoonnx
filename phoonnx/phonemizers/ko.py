from g2pk import G2p

from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.thirdparty.hangul2ipa import hangul2ipa


class G2PKPhonemizer(BasePhonemizer):

    def __init__(self, descriptive=True, group_vowels=True, to_syl=True, ipa=True):
        self.g2p = G2p()
        self.descriptive = descriptive
        self.group_vowels = group_vowels
        self.to_syl = to_syl
        self.ipa = ipa

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
        if self.ipa:
            return hangul2ipa(p)
        return p


class KoG2PPhonemizer(BasePhonemizer):
    """https://github.com/scarletcho/KoG2P"""
    def __init__(self, ipa=True):
        from phoonnx.thirdparty.kog2p import runKoG2P
        self.g2p = runKoG2P
        self.ipa = ipa

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
        if self.ipa:
            return hangul2ipa(p)
        return p


if __name__ == "__main__":

    pho = G2PKPhonemizer()
    pho2 = KoG2PPhonemizer()
    lang = "ko"

    text = "터미널에서 원하는 문자열을 함께 입력해 사용할 수 있습니다."
    print(f"\n--- Getting phonemes for '{text}' ---")
    phonemes_cotovia = pho.phonemize(text, lang)
    print(f"  G2PK Phonemes: {phonemes_cotovia}")

    phonemes_cotovia = pho2.phonemize(text, lang)
    print(f"  KoG2P Phonemes: {phonemes_cotovia}")

