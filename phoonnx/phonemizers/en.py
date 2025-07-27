import os

import requests

from phoonnx.thirdparty.arpa2ipa import arpa_to_ipa_lookup
from phoonnx.phonemizers.base import BasePhonemizer


class DeepPhonemizer(BasePhonemizer):
    """
    https://github.com/spring-media/DeepPhonemizer
    """
    MODELS = {
        "latin_ipa_forward.pt": "https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/latin_ipa_forward.pt",
        "en_us_cmudict_ipa_forward.pt": "https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt",
        "en_us_cmudict_forward.pt": "https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_forward.pt"
    }

    def __init__(self, model="latin_ipa_forward.pt"):
        import dp
        from dp.phonemizer import Phonemizer
        import torch
        # needed for latest torch version
        torch.serialization.add_safe_globals([dp.preprocessing.text.Preprocessor])
        torch.serialization.add_safe_globals([dp.preprocessing.text.LanguageTokenizer])
        torch.serialization.add_safe_globals([dp.preprocessing.text.SequenceTokenizer])

        if not os.path.isfile(model):
            if model in self.MODELS:
                url = self.MODELS[model]
                cache_dir = os.path.expanduser("~/.local/share/deepphonemizer")
                os.makedirs(cache_dir, exist_ok=True)
                model_path = os.path.join(cache_dir, model)
                if not os.path.isfile(model_path):
                    print(f"Downloading {model} from {url}...")
                    with requests.get(url, stream=True) as r:
                        r.raise_for_status()
                        with open(model_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    print(f"Saved model to {model_path}")
                model = model_path
            else:
                raise ValueError("invalid model")

        self.phonemizer = Phonemizer.from_checkpoint(model)

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
        return cls.match_lang(target_lang, ['de', 'en_us'])

    def phonemize_string(self, text: str, lang: str) -> str:
        """
        Normalizes input text by applying a series of transformations
        and returns it as a sequence of graphemes.

        Parameters:
            text (str): Input text to be converted to graphemes.
            lang (str): The language code (ignored for grapheme phonemization,
                        but required by BasePhonemizer).

        Returns:
            str: A normalized string of graphemes.
        """
        lang = self.get_lang(lang)
        return self.phonemizer(text, lang)


class OpenPhonemizer(BasePhonemizer):
    """
    https://github.com/NeuralVox/OpenPhonemizer
    """

    def __init__(self):
        from openphonemizer import OpenPhonemizer
        import torch
        # needed for latest torch version
        import dp
        torch.serialization.add_safe_globals([dp.preprocessing.text.Preprocessor])
        torch.serialization.add_safe_globals([dp.preprocessing.text.LanguageTokenizer])
        torch.serialization.add_safe_globals([dp.preprocessing.text.SequenceTokenizer])

        self.phonemizer = OpenPhonemizer()

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
        return cls.match_lang(target_lang, ["en"])

    def phonemize_string(self, text: str, lang: str) -> str:
        """
        Normalizes input text by applying a series of transformations
        and returns it as a sequence of graphemes.

        Parameters:
            text (str): Input text to be converted to graphemes.
            lang (str): The language code (ignored for grapheme phonemization,
                        but required by BasePhonemizer).

        Returns:
            str: A normalized string of graphemes.
        """
        lang = self.get_lang(lang)
        return self.phonemizer(text)


class G2P_EN(BasePhonemizer):
    """
    https://github.com/Kyubyong/g2p
    """

    def __init__(self):
        import nltk
        nltk.download('averaged_perceptron_tagger_eng')
        nltk.download('cmudict')
        from g2p_en import G2p
        self.g2p = G2p()

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
        return cls.match_lang(target_lang, ["en"])

    def phonemize_string(self, text: str, lang: str) -> str:
        """
        Normalizes input text by applying a series of transformations
        and returns it as a sequence of graphemes.

        Parameters:
            text (str): Input text to be converted to graphemes.
            lang (str): The language code (ignored for grapheme phonemization,
                        but required by BasePhonemizer).

        Returns:
            str: A normalized string of graphemes.
        """
        lang = self.get_lang(lang)
        # NOTE: this model returns ARPA not IPA, need to map phonemes
        return "".join([arpa_to_ipa_lookup.get(pho, pho) for pho in self.g2p(text)])



if __name__ == "__main__":
    # for comparison
    from phoonnx.phonemizers.mul import (ByT5Phonemizer, EspeakPhonemizer, GruutPhonemizer,
                                         EpitranPhonemizer, CharsiuPhonemizer)
    byt5 = ByT5Phonemizer()
    espeak = EspeakPhonemizer()
    gruut = GruutPhonemizer()
    epitr = EpitranPhonemizer()
    charsiu = CharsiuPhonemizer()
    openphon = OpenPhonemizer()
    g2pen = G2P_EN()
    dp = DeepPhonemizer()

    lang = "en-gb"

    print("\n--- Getting phonemes for 'Hello, world. How are you?' ---")
    text1 = "Hello, world. How are you?"
    phonemes1 = espeak.phonemize(text1, lang)
    phonemes1b = gruut.phonemize(text1, lang)
    phonemes1c = byt5.phonemize(text1, lang)
    phonemes1d = epitr.phonemize(text1, lang)
    phonemes1e = charsiu.phonemize(text1, lang)
    phonemes1f = openphon.phonemize(text1, lang)
    phonemes1g = g2pen.phonemize(text1, lang)
    phonemes1h = dp.phonemize(text1, lang)
    print(f" Espeak         Phonemes: {phonemes1}")
    print(f" Gruut          Phonemes: {phonemes1b}")
    print(f" byt5           Phonemes: {phonemes1c}")
    print(f" Epitran        Phonemes: {phonemes1d}")
    print(f" Charsiu        Phonemes: {phonemes1e}")
    print(f" OpenPhonemizer Phonemes: {phonemes1f}")
    print(f" DeepPhonemizer Phonemes: {phonemes1h}")
    print(f" G2P_en         Phonemes: {phonemes1g}")

    print("\n--- Getting phonemes for 'This is a test: a quick one; and done!' ---")
    text2 = "This is a test: a quick one; and done!"
    phonemes2 = espeak.phonemize(text2, lang)
    phonemes2b = gruut.phonemize(text2, lang)
    phonemes2c = byt5.phonemize(text2, lang)
    phonemes2d = epitr.phonemize(text2, lang)
    phonemes2e = charsiu.phonemize(text2, lang)
    print(f"  Espeak Phonemes: {phonemes2}")
    print(f"  Gruut Phonemes: {phonemes2b}")
    print(f"   byt5  Phonemes: {phonemes2c}")
    print(f" Epitran Phonemes: {phonemes2d}")
    print(f" Charsiu Phonemes: {phonemes2e}")

    print("\n--- Getting phonemes for 'Just a phrase without punctuation' ---")
    text3 = "Just a phrase without punctuation"
    phonemes3 = espeak.phonemize(text3, lang)
    phonemes3b = gruut.phonemize(text3, lang)
    phonemes3c = byt5.phonemize(text3, lang)
    phonemes3d = epitr.phonemize(text3, lang)
    phonemes3e = charsiu.phonemize(text3, lang)
    print(f"  Espeak Phonemes: {phonemes3}")
    print(f"  Gruut Phonemes: {phonemes3b}")
    print(f"   byt5  Phonemes: {phonemes3c}")
    print(f" Epitran Phonemes: {phonemes3d}")
    print(f" Charsiu Phonemes: {phonemes3e}")

