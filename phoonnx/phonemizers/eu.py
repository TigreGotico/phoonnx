from typing import List, Optional

from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.config import Alphabet


# Collapse multi-char IPA tokens emitted by AhoTTS into the single-char
# symbols used by the StyleTTS2-eu / VITS-eu phoneme_id_map. Stressed vowels
# carry a leading "'" in the AhoTTS output and become uppercase ASCII letters;
# the affricates / aspirated stops collapse to single ASCII placeholders.
MULTICHAR = {
    "tʃ": "C",
    "ts": "V",
    "tʂ": "P",
    "'i": "I",
    "'e": "E",
    "'a": "A",
    "'o": "O",
    "'u": "U",
    "pʰ": "H",
    "kʰ": "K",
    "tʰ": "T",
}


class AhoTTSPhonemizer(BasePhonemizer):
    """
    A phonemizer for Basque (eu) backed by the ``pyahotts`` package, which wraps
    the AhoTTS engine.

    This is the **V1** AhoTTS engine, a close approximation (~96%) of the
    StyleTTS2-eu V3 phonemizer. It powers the ``hitz-eu-styletts2`` and
    ``hitz-eu_*`` mirror voices, whose configs set ``phoneme_type: ahotts``.

    ``pyahotts`` is imported lazily so that importing ``phoonnx`` does not
    hard-require it; a clear error is raised on first use if it is missing.
    """

    def __init__(self, alphabet: Alphabet = Alphabet.IPA):
        """
        Initialize the AhoTTS phonemizer.

        Args:
            alphabet (Alphabet): Accepted for signature parity with the other
                phonemizers; AhoTTS always produces IPA-derived tokens.
        """
        self._ahotts = None
        super().__init__(alphabet)

    @property
    def ahotts(self):
        """Lazily create and cache the underlying ``pyahotts.AhoTTS`` engine."""
        if self._ahotts is None:
            try:
                from pyahotts import AhoTTS
            except ImportError as e:
                raise ImportError(
                    "pyahotts is required for the AhoTTS Basque phonemizer. "
                    "Install it with 'pip install pyahotts>=0.1.0a1' "
                    "(or 'pip install phoonnx[eu]')."
                ) from e
            self._ahotts = AhoTTS()
        return self._ahotts

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        """
        Validate and return the closest supported language code.

        Args:
            target_lang (str): The language code to validate.

        Returns:
            str: The validated language code.

        Raises:
            ValueError: If the language code is unsupported.
        """
        return cls.match_lang(target_lang, ["eu-ES"])

    def phonemize_string(self, text: str, lang: str) -> str:
        """
        Convert Basque text into a StyleTTS2-eu tokenizable phoneme string.

        Calls ``pyahotts.AhoTTS.get_phonemes`` to obtain per-word lists of IPA
        phone tokens, collapses multi-char IPA tokens to their single-char
        StyleTTS2 symbols (see :data:`MULTICHAR`), joins each word's phones with
        no separator, and joins words with a single space.

        Parameters:
            text (str): The input text to phonemize.
            lang (str): The language code (validated; AhoTTS only supports eu).

        Returns:
            str: The collapsed phoneme string (one char per StyleTTS2 symbol).
        """
        self.get_lang(lang)
        words: List[List[str]] = self.ahotts.get_phonemes(text, lang="eu", ipa=True)
        return " ".join(
            "".join(MULTICHAR.get(phone, phone) for phone in word)
            for word in words
        )


if __name__ == "__main__":
    eu = AhoTTSPhonemizer()
    for sentence in ["Kaixo, mundua.", "Euskara hizkuntza zaharra eta ederra da."]:
        print(f"\n--- Getting phonemes for '{sentence}' (AhoTTS) ---")
        print(f"  AhoTTS Phonemes: {eu.phonemize_string(sentence, 'eu')}")
