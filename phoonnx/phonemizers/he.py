import os.path

import requests

from phoonnx.phonemizers.base import BasePhonemizer


class PhonikudPhonemizer(BasePhonemizer):
    dl_url = "https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx"

    def __init__(self, model: str = None, diacritics=True):
        from phonikud_onnx import Phonikud
        from phonikud import phonemize
        self.g2p = phonemize
        self.diacritics = diacritics
        if model is None:
            base_path = os.path.expanduser("~/.local/share/phonikud")
            fname = self.dl_url.split("/")[-1]
            model = f"{base_path}/{fname}"
            if not os.path.isfile(model):
                os.makedirs(base_path, exist_ok=True)
                # TODO - streaming download
                data = requests.get(self.dl_url).content
                with open(model, "wb") as f:
                    f.write(data)
        self.phonikud = Phonikud(model) if diacritics else None

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
        return cls.match_lang(target_lang, ["he"])

    def phonemize_string(self, text: str, lang: str = "he") -> str:
        """
        """
        lang = self.get_lang(lang)
        if self.diacritics:
            text = self.phonikud.add_diacritics(text)
        return self.g2p(text)


if __name__ == "__main__":
    #text = "מתכת יקרה"
    text = 'שָׁלוֹם עוֹלָם'

    pho = PhonikudPhonemizer(diacritics=False)
    lang = "he"

    print(f"\n--- Getting phonemes for '{text}' ---")
    phonemes = pho.phonemize(text, lang)
    print(f"  Phonemes: {phonemes}")
    # --- Getting phonemes for 'שָׁלוֹם עוֹלָם' ---
    #   Phonemes: [('ʃalˈom ʔolˈam', '.', True)]