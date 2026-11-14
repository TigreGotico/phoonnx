"""The add_diacritics default follows the language a tag names, not its spelling.

Arabic (``ar``, with any region) and the individual Arabic languages of the
ISO 639-3 ``ara`` macrolanguage (``apc`` Levantine, ``arz`` Egyptian), in
Arabic script, default to diacritics. Languages whose code only starts with
"ar" do not: ``arl`` is Arabela (Peru), ``arn`` Mapudungun (Chile), ``arg``
Aragonese (Spain). Neither does Arabic written in Latin script (``ar-Latn``,
or ``acy`` Cypriot Maronite Arabic, which CLDR writes in Latin script).
"""
import json
import os
import tempfile
import unittest

from phoonnx.config import Alphabet, PhonemeType, VoiceConfig

ARABIC = ("ar", "ar-EG", "apc", "arz")
NOT_ARABIC = ("arl", "arn", "arg", "ar-Latn", "acy", "en", "he")


def _post_init_default(lang_code):
    return VoiceConfig(num_symbols=1, num_speakers=1, num_langs=1, sample_rate=16000,
                       lang_code=lang_code, phoneme_type=PhonemeType.ESPEAK,
                       alphabet=Alphabet.IPA, phonemizer_model=None).add_diacritics


def _piper_default(lang_code):
    cfg = {"piper_version": "1.0.0", "phoneme_type": "espeak",
           "phoneme_id_map": {"a": [0], "b": [1]}, "espeak": {"voice": "en-us"}}
    return VoiceConfig.from_dict(cfg, lang_code=lang_code).add_diacritics


class TestPostInitDefault(unittest.TestCase):
    def test_arabic_in_arabic_script_defaults_on(self):
        for lang in ARABIC:
            with self.subTest(lang=lang):
                self.assertIs(_post_init_default(lang), True)

    def test_other_languages_default_off(self):
        for lang in NOT_ARABIC:
            with self.subTest(lang=lang):
                self.assertIs(_post_init_default(lang), False)


class TestPiperDefault(unittest.TestCase):
    def test_arabic_in_arabic_script_defaults_on(self):
        for lang in ARABIC:
            with self.subTest(lang=lang):
                self.assertIs(_piper_default(lang), True)

    def test_other_languages_default_off(self):
        for lang in NOT_ARABIC:
            with self.subTest(lang=lang):
                self.assertIs(_piper_default(lang), False)


class TestChatterboxDefault(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.bpe_path = os.path.join(cls.tmpdir.name, "tokenizer.json")
        with open(cls.bpe_path, "w", encoding="utf-8") as f:
            json.dump({"version": "1.0", "added_tokens": [], "normalizer": None,
                       "pre_tokenizer": None, "post_processor": None, "decoder": None,
                       "model": {"type": "BPE", "unk_token": "[UNK]",
                                 "vocab": {"[UNK]": 0, "a": 1}, "merges": []}}, f)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _default(self, lang_code):
        return VoiceConfig.from_dict({"engine": "chatterbox"}, bpe_tokenizer_json=self.bpe_path,
                                     lang_code=lang_code).add_diacritics

    def test_arabic_and_hebrew_default_on(self):
        for lang in ARABIC + ("he", "he-IL"):
            with self.subTest(lang=lang):
                self.assertIs(self._default(lang), True)

    def test_other_languages_default_off(self):
        for lang in ("arl", "arn", "arg", "ar-Latn", "acy", "en"):
            with self.subTest(lang=lang):
                self.assertIs(self._default(lang), False)


if __name__ == "__main__":
    unittest.main()
