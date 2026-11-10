"""Language matching follows the default-region rule.

A bare tag means the region where the language first appeared and is still
spoken: `en` is British English and `pt` is European Portuguese.
"""
import os
import tempfile
import unittest
from unittest.mock import patch

from phoonnx.model_manager import TTSModelInfo, TTSModelManager
from phoonnx.util import match_lang
from phoonnx.voice import PhoneticSpellings


class TestMatchLangDefaultRegion(unittest.TestCase):
    def test_bare_en_picks_british_english(self):
        self.assertEqual(match_lang("en", ["en-US", "en-GB"])[0], "en-GB")

    def test_bare_pt_picks_european_portuguese(self):
        self.assertEqual(match_lang("pt", ["pt-BR", "pt-PT"])[0], "pt-PT")

    def test_regional_request_keeps_its_region(self):
        self.assertEqual(match_lang("en-US", ["en-GB", "en-US"]), ("en-US", 0))
        self.assertEqual(match_lang("pt-BR", ["pt-PT", "pt-BR"]), ("pt-BR", 0))


class TestMatchLangUnparseableEntry(unittest.TestCase):
    """An entry that is not a valid tag is measured as its longest valid
    prefix, and the entry itself is returned."""

    def test_voice_id_with_region(self):
        self.assertEqual(match_lang("zh-HK", ["en-US", "zh-HK-HiuMaanNeural"]),
                         ("zh-HK-HiuMaanNeural", 0))

    def test_voice_id_without_region(self):
        self.assertEqual(match_lang("de", ["en-US", "de-Thorsten1234"]),
                         ("de-Thorsten1234", 0))


class TestGetLangVoicesDefaultRegion(unittest.TestCase):
    def setUp(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        expand = patch("phoonnx.model_manager.os.path.expanduser", return_value=tmpdir.name)
        expand.start()
        self.addCleanup(expand.stop)
        self.manager = TTSModelManager(cache_path=os.path.join(tmpdir.name, "cache.json"))

    def _voices(self, *langs):
        voices = [TTSModelInfo(voice_id=f"v/{lang}", lang=lang,
                               model_url=f"https://example.com/{lang}.onnx")
                  for lang in langs]
        self.manager.voices = {v.voice_id: v for v in voices}

    def test_bare_en_lists_british_voice_first(self):
        self._voices("en-US", "en-GB")
        self.assertEqual([v.lang for v in self.manager.get_lang_voices("en")],
                         ["en-GB", "en-US"])

    def test_bare_pt_lists_european_voice_first(self):
        self._voices("pt-BR", "pt-PT")
        self.assertEqual([v.lang for v in self.manager.get_lang_voices("pt")],
                         ["pt-PT", "pt-BR"])


class TestPhoneticSpellingsDefaultRegion(unittest.TestCase):
    def _locale(self, *langs):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        for lang in langs:
            os.makedirs(os.path.join(tmpdir.name, lang))
            with open(os.path.join(tmpdir.name, lang, "phonetic_spellings.txt"), "w") as f:
                f.write(f"word:{lang}\n")
        return tmpdir.name

    def test_bare_en_reads_british_spellings(self):
        spellings = PhoneticSpellings.from_lang("en", self._locale("en-US", "en-GB"))
        self.assertEqual(spellings.replacements, {"word": "en-GB"})

    def test_bare_pt_reads_european_spellings(self):
        spellings = PhoneticSpellings.from_lang("pt", self._locale("pt-BR", "pt-PT"))
        self.assertEqual(spellings.replacements, {"word": "pt-PT"})

    def test_unknown_language_raises(self):
        with self.assertRaises(FileNotFoundError):
            PhoneticSpellings.from_lang("zz-ZZ", self._locale("en-GB", "pt-PT"))


if __name__ == "__main__":
    unittest.main()
