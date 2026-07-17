from phoonnx.phonemizers.base import BasePhonemizer, Alphabet


# phoonnx lang -> mwl_phonemizer dialect spec code. Mirandese is dialect-aware:
# the central variety is the default, with Sendinese and the Raiano (Ifanês)
# border variety selectable per call.
DIALECTS = {
    "mwl": "mwl",                      # central Mirandese (the standard)
    "mwl-x-sendim": "mwl-x-sendim",    # Sendinese
    "mwl-x-ifanes": "mwl-x-ifanes",    # Raiano / Ifanês (border variety)
}


class MirandesePhonemizer(BasePhonemizer):
    """
    Mirandese phonemizer backed by ``mwl_phonemizer``, an orthography2ipa-lattice
    engine for the three Mirandese varieties.  Output is always IPA.

    The variety is taken from the per-call ``lang`` argument (central ``mwl`` by
    default), so one instance serves all three.  ``mwl_phonemizer`` is imported
    lazily so importing ``phoonnx`` does not hard-require it.
    """

    _LANGS = list(DIALECTS)

    def __init__(self):
        super().__init__(Alphabet.IPA)
        self._phonemize = None

    @property
    def phonemize_fn(self):
        """Lazily import and cache ``mwl_phonemizer.phonemize``."""
        if self._phonemize is None:
            try:
                from mwl_phonemizer import phonemize
            except ImportError as e:
                raise ImportError(
                    "mwl_phonemizer is required for the Mirandese phonemizer. "
                    "Install it with 'pip install mwl_phonemizer' "
                    "(or 'pip install phoonnx[mwl]')."
                ) from e
            self._phonemize = phonemize
        return self._phonemize

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        """
        Validates and returns the closest supported Mirandese variety code.

        Raises:
            ValueError: If the language code is unsupported.
        """
        return cls.match_lang(target_lang, cls._LANGS)

    def phonemize_string(self, text: str, lang: str = "mwl") -> str:
        dialect = DIALECTS[self.get_lang(lang)]
        return self.phonemize_fn(text, dialect=dialect)


if __name__ == "__main__":
    pho = MirandesePhonemizer()
    for sentence in ["Buonos dies, mundo.", "L mirandés ye ua lhéngua."]:
        print(sentence)
        for code in MirandesePhonemizer._LANGS:
            print(f"  {code} -> {pho.phonemize_string(sentence, code)}")
