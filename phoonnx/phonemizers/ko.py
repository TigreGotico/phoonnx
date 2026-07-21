"""Compatibility shim — implementation lives in scriptconv.phonemizers.

Import paths and class identities are preserved for existing voice
configs and callers; construction through ``phoonnx.config.get_phonemizer``
injects phoonnx's text normalizer so behavior is unchanged.
"""
from scriptconv.phonemizers.ko import G2PKPhonemizer  # noqa: F401
from scriptconv.phonemizers.base import BasePhonemizer
from scriptconv.phonemizers.enums import Alphabet


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
