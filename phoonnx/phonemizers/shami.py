"""Shami (Levantine Arabic / English code-switching) phonemizer.

Wraps the vendored :mod:`phoonnx.thirdparty.shami` text front-end so ShamiVITS ONNX
models can be driven directly from text.  The front-end emits:

* a shared-IPA phoneme stream, and
* a *parallel* per-phoneme language-ID stream (AR / EN / NEUTRAL / PAD)

which the :class:`ShamiAdapter` feeds to the exported ONNX model.
"""

from typing import List, Tuple

from quebra_frases import sentence_tokenize

from phoonnx.config import Alphabet
from phoonnx.phonemizers.base import BasePhonemizer, PhonemizedChunks
from phoonnx.thirdparty.shami import TextFrontend


class ShamiPhonemizer(BasePhonemizer):
    """Phonemizer for Levantine Arabic + English code-switching (ShamiVITS)."""

    def __init__(self, alphabet: Alphabet = Alphabet.IPA,
                 diacritizer_backend: str = "auto"):
        if alphabet != Alphabet.IPA:
            raise ValueError("ShamiPhonemizer only supports IPA")
        super().__init__(alphabet)
        self.frontend = TextFrontend(diacritizer_backend=diacritizer_backend)

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        return cls.match_lang(target_lang, ["ar", "en"])

    def phonemize_string(self, text: str, lang: str) -> str:
        """Return the raw IPA string for ``text``.

        This is required by :class:`BasePhonemizer` but is not the preferred
        entry-point for Shami inference because it discards language IDs.
        """
        utterance = self.frontend.process(text)
        return utterance.ipa

    def phonemize(self, text: str, lang: str) -> PhonemizedChunks:
        """Return sentence-level phoneme lists."""
        phonemes, _ = self.phonemize_with_language_ids(text, lang)
        return phonemes

    def phonemize_with_language_ids(
        self, text: str, lang: str
    ) -> Tuple[PhonemizedChunks, List[List[int]]]:
        """Return sentence-level phoneme lists and matching per-phoneme language IDs.

        The returned language IDs are integers from :class:`phoonnx.thirdparty.shami.Lang`:
        ``PAD=0``, ``AR=1``, ``EN=2``, ``NEUTRAL=3``.
        """
        if not text:
            return [], []

        all_phonemes: PhonemizedChunks = []
        all_lang_ids: List[List[int]] = []

        for sentence in sentence_tokenize(text):
            if not sentence.strip():
                continue
            utterance = self.frontend.process(sentence)
            all_phonemes.append(utterance.symbols)
            all_lang_ids.append(utterance.language_ids)

        return all_phonemes, all_lang_ids
