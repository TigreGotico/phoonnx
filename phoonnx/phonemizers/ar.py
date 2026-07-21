"""Compatibility shim — implementation lives in scriptconv.phonemizers.

Import paths and class identities are preserved for existing voice
configs and callers; construction through ``phoonnx.config.get_phonemizer``
injects phoonnx's text normalizer so behavior is unchanged.
"""
from scriptconv.phonemizers.ar import ArbtokPhonemizer  # noqa: F401
from scriptconv.phonemizers.base import BasePhonemizer
from scriptconv.phonemizers.enums import Alphabet
from phoonnx.thirdparty.bw2ipa import mantoq_to_ipa
from phoonnx.thirdparty.mantoq import g2p as mantoq


class MantoqPhonemizer(BasePhonemizer):

    def __init__(self, alphabet=Alphabet.BUCKWALTER):
        if alphabet not in [Alphabet.IPA, Alphabet.BUCKWALTER]:
            raise ValueError("unsupported alphabet")
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
        return cls.match_lang(target_lang, ["ar"])

    def phonemize_string(self, text: str, lang: str = "ar") -> str:
        """
        Phonemizes an Arabic string using the Mantoq G2P tool.
        If the alphabet is set to IPA, it then converts the result using bw2ipa.
        """
        lang = self.get_lang(lang)
        # The mantoq function returns a tuple of (normalized_text, phonemes)
        normalized_text, phonemes = mantoq(text)

        # The phonemes are a list of characters, we join them into a string
        # and replace the word separator token with a space.
        phonemes = "".join(phonemes).replace("_+_", " ")

        if self.alphabet == Alphabet.IPA:
            # If the alphabet is IPA, we use the bw2ipa function to translate
            # the Buckwalter-like phonemes into IPA.
            return mantoq_to_ipa(phonemes)

        # Otherwise, we return the phonemes in the default Mantoq alphabet.
        return phonemes
