from typing import Any, Optional

from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.config import Alphabet


class CotoviaPhonemizer(BasePhonemizer):
    """
    Galician phonemizer backed by pycotovia — a pure-Python port of the Cotovia
    G2P engine that has verified parity with the original C binary.

    Output alphabets:
    - ``Alphabet.COTOVIA`` — raw Cotovia phoneme notation (e.g. ``"Este e uN ..."``)
    - ``Alphabet.IPA`` — IPA string produced by pycotovia's ``cotovia_to_ipa``

    Voices trained on Cotovia-alphabet output continue to receive the same
    notation strings as before because pycotovia is binary-parity-tested
    (see pycotovia/docs/parity.md).

    ``pycotovia`` is an optional dependency (the ``gl`` extra); it is imported
    lazily so that ``import phoonnx.phonemizers`` works without it installed.
    """

    def __init__(self, alphabet: Alphabet = Alphabet.IPA,
                 model: Optional[str] = None):
        """
        Args:
            alphabet (Alphabet): ``COTOVIA`` (raw notation) or ``IPA``.
            model (Optional[str]): phonemizer variant, from the voice's
                ``phonemizer_model``.  ``"stress"`` emits the cotovia notation
                with the stressed vowel marked by a trailing ``^`` (e.g.
                ``"o^la"``) and multi-char phonemes preserved -- matching the
                AhoTTS Cotovia front-end the HiTZ gl VITS voices were trained on,
                whose tokenizer gives stressed vowels their own ids.  The default
                (``None``) is the original stressless transcription used by the
                proxectonos gl voices.
        """
        self._engine: Optional[Any] = None
        self.model = (model or "").lower() or None
        super().__init__(alphabet)

    @property
    def engine(self) -> Any:
        if self._engine is None:
            try:
                from pycotovia import Phonemizer as _PycotoviaPhonemizer
            except ImportError as e:
                raise ImportError(
                    "pycotovia is required for the Galician (Cotovia) phonemizer. "
                    "Install it with 'pip install pycotovia' or 'pip install phoonnx[gl]'."
                ) from e
            self._engine = _PycotoviaPhonemizer()
        return self._engine

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        return cls.match_lang(target_lang, ["gl-ES"])

    def phonemize_string(self, text: str, lang: str) -> str:
        self.get_lang(lang)
        if self.model == "stress":
            # tra=2 -> stressed vowel followed by '^'.  The trailing space the
            # engine appends per word is kept (word separator); the tokenizer
            # folds 'V^' and multi-char phonemes (tS, rr, ...) back into single
            # tokens via the voice's phoneme_id_map compound keys.
            return self.engine.phonemize(text, tra=2)
        cotovia_output = self.engine.phonemize(text)
        if self.alphabet == Alphabet.IPA:
            try:
                from pycotovia import cotovia_to_ipa as _cotovia_to_ipa
            except ImportError as e:
                raise ImportError(
                    "pycotovia is required for the Galician (Cotovia) phonemizer. "
                    "Install it with 'pip install pycotovia' or 'pip install phoonnx[gl]'."
                ) from e
            return _cotovia_to_ipa(cotovia_output)
        return cotovia_output
