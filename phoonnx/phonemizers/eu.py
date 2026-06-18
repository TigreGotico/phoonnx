from typing import Optional

from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.config import Alphabet


# AhoTTS engine variant, selected per-voice via ``phonemizer_model`` in the
# voice config.  Each maps to an ``ahotts_g2p.phonemize`` call.
ENGINES = {
    "classic": {"version": "classic"},    # original AhoTTS engine (HiTZ VITS voices)
    "modern": {"version": "modern"},      # StyleTTS-era build (HiTZ/StyleTTS2-eu)
    "northern": {"dialect": "northern"},  # Northern (Iparralde / Iparrahotsa) dialect
}
DEFAULT_ENGINE = "modern"


class AhoTTSPhonemizer(BasePhonemizer):
    """
    AhoTTS phonemizer backed by ``ahotts-g2p``, a pure-Python port of the
    AhoTTS engine (no C build, no runtime dependencies).

    Covers the two AhoTTS-frontend languages: Basque (``eu``) and Spanish
    (``es``).  The target language is taken from the per-call ``lang`` argument
    (so the same phonemizer instance serves both); ``ahotts_g2p.phonemize``
    dispatches on it.

    The engine variant is chosen per-voice with ``phonemizer_model`` in the
    voice config (passed to the constructor as ``engine``):

      * ``classic``  -- the original AhoTTS engine; the HiTZ VITS voices (eu+es).
      * ``modern``   -- the StyleTTS-era build; HiTZ/StyleTTS2-eu.  The default.
      * ``northern`` -- the Northern (Iparralde / Iparrahotsa) Basque dialect:
        pronounced /h/, French vowels (ü -> /y/), uvular /ʁ/, a remapped
        sibilant system.  Basque only.

    ``ahotts_g2p.phonemize`` already returns the collapsed single-char training
    string, so no further token folding is needed.  It is imported lazily so
    importing ``phoonnx`` does not hard-require it.
    """

    SUPPORTED_LANGS = ["eu-ES", "es-ES"]

    def __init__(self, engine: Optional[str] = None,
                 alphabet: Alphabet = Alphabet.IPA):
        """
        Args:
            engine (Optional[str]): AhoTTS variant -- ``"classic"``, ``"modern"``
                or ``"northern"`` (from the voice's ``phonemizer_model``).
                Defaults to ``"modern"``.
            alphabet (Alphabet): Accepted for signature parity; AhoTTS always
                produces IPA-derived single-char tokens.
        """
        engine = (engine or DEFAULT_ENGINE).lower()
        if engine not in ENGINES:
            raise ValueError(
                f"unknown AhoTTS engine {engine!r} "
                f"(supported: {', '.join(ENGINES)})"
            )
        self.engine = engine
        self._phonemize = None
        super().__init__(alphabet)

    @property
    def phonemize_fn(self):
        """Lazily import and cache ``ahotts_g2p.phonemize``."""
        if self._phonemize is None:
            try:
                from ahotts_g2p import phonemize
            except ImportError as e:
                raise ImportError(
                    "ahotts-g2p is required for the AhoTTS Basque phonemizer. "
                    "Install it with 'pip install ahotts-g2p' "
                    "(or 'pip install phoonnx[eu]')."
                ) from e
            self._phonemize = phonemize
        return self._phonemize

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        """
        Validate and return the closest supported language code (eu or es).

        Raises:
            ValueError: If the language code is unsupported.
        """
        return cls.match_lang(target_lang, cls.SUPPORTED_LANGS)

    def phonemize_string(self, text: str, lang: str) -> str:
        """
        Convert Basque or Spanish text into the AhoTTS single-char IPA training
        string.

        Args:
            text (str): The input text to phonemize.
            lang (str): The language code (``eu`` or ``es``).  The ``northern``
                dialect engine is Basque-only; combining it with ``es`` raises.

        Returns:
            str: Space-separated single-char phoneme tokens.
        """
        g2p_lang = self.get_lang(lang).split("-")[0]
        kwargs = dict(ENGINES[self.engine])
        if g2p_lang != "eu" and "dialect" in kwargs:
            raise ValueError(
                f"the {self.engine!r} AhoTTS engine (dialect) is Basque-only; "
                f"it cannot phonemize lang={lang!r}"
            )
        return self.phonemize_fn(text, lang=g2p_lang, **kwargs)


if __name__ == "__main__":
    for eng in ("classic", "modern", "northern"):
        eu = AhoTTSPhonemizer(eng)
        print(f"\n--- AhoTTS '{eng}' (eu) ---")
        for sentence in ["Kaixo, mundua.", "Euskara hizkuntza zaharra eta ederra da."]:
            print(f"  {sentence!r}: {eu.phonemize_string(sentence, 'eu')}")
    for eng in ("classic", "modern"):
        es = AhoTTSPhonemizer(eng)
        print(f"\n--- AhoTTS '{eng}' (es) ---")
        for sentence in ["Hola mundo.", "El sol brilla sobre la montaña."]:
            print(f"  {sentence!r}: {es.phonemize_string(sentence, 'es')}")
