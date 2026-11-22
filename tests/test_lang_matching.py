"""Language matching follows the default-region rule.

A bare tag means the region where the language first appeared and is still
spoken: `en` is British English and `pt` is European Portuguese.
"""
import os
import tempfile
import unittest
from unittest.mock import patch

from phoonnx.model_manager import TTSModelInfo, TTSModelManager
from linguatag import default_region

from phoonnx.util import match_lang, normalize_lang
from phoonnx.voice import PhoneticSpellings


class TestMatchLangDefaultRegion(unittest.TestCase):
    def test_bare_en_picks_british_english(self):
        self.assertEqual(match_lang("en", ["en-US", "en-GB"])[0], "en-GB")

    def test_bare_pt_picks_european_portuguese(self):
        self.assertEqual(match_lang("pt", ["pt-BR", "pt-PT"])[0], "pt-PT")

    def test_regional_request_keeps_its_region(self):
        self.assertEqual(match_lang("en-US", ["en-GB", "en-US"]), ("en-US", 0))
        self.assertEqual(match_lang("pt-BR", ["pt-PT", "pt-BR"]), ("pt-BR", 0))


class TestNormalizeLangDefaultRegion(unittest.TestCase):
    """normalize_lang gives a bare tag the region linguatag defaults it to."""

    def test_rule_examples(self):
        self.assertEqual(normalize_lang("en"), "en-GB")
        self.assertEqual(normalize_lang("pt"), "pt-PT")

    def test_languages_of_several_countries_get_their_default_region(self):
        for lang in ("es", "fr", "de", "ar", "zh", "sw", "hi", "ur"):
            with self.subTest(lang=lang):
                self.assertEqual(normalize_lang(lang), f"{lang}-{default_region(lang)}")

    def test_language_with_no_default_region_stays_bare(self):
        # Tags with no country: uncoded, no linguistic content, private use.
        for lang in ("mis", "zxx", "qaa"):
            with self.subTest(lang=lang):
                self.assertIsNone(default_region(lang))
                self.assertEqual(normalize_lang(lang), lang)

    def test_three_letter_and_script_tags_get_their_default_region(self):
        for tag, expected in (("arz", f"arz-{default_region('arz')}"),
                              ("yue", f"yue-{default_region('yue')}"),
                              ("zh-Hant", f"zh-Hant-{default_region('zh-Hant')}"),
                              ("crk-script_syllabics", f"crk-Cans-{default_region('crk-Cans')}")):
            with self.subTest(tag=tag):
                self.assertEqual(normalize_lang(tag), expected)


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



class TestGetLangVoicesDialects(TestGetLangVoicesDefaultRegion):
    """A voice in a dialect of the requested language and region serves the request.

    text2spoken reads colloquial numbers only for a dialect tag such as ``arz``
    (Egyptian Arabic), so a dialect voice carries that tag. linguatag gives
    ``arz-EG`` asking for ``ar-EG`` the distance 10 and the other way round 80.
    """

    def _ids(self, lang):
        return [v.lang for v in self.manager.get_lang_voices(lang)]

    def test_a_regional_request_finds_the_dialect_voice_of_its_region(self):
        self._voices("arz-EG", "ary-MA", "ar-EG", "ar")
        self.assertEqual(self._ids("ar-EG"), ["ar-EG", "ar", "arz-EG"])

    def test_each_habibi_region_finds_its_dialect(self):
        pairs = {"ar-DZ": "arq-DZ", "ar-EG": "arz-EG", "ar-IQ": "acm-IQ",
                 "ar-MA": "ary-MA", "ar-SA": "ars-SA", "ar-AE": "afb-AE"}
        self._voices(*pairs.values())
        for request, dialect in pairs.items():
            with self.subTest(request=request):
                self.assertEqual(self._ids(request), [dialect])

    def test_standard_arabic_does_not_take_a_dialect_voice(self):
        self._voices("ary-MA", "arz-EG", "ar")
        self.assertEqual(self._ids("ar"), ["ar"])

    def test_a_dialect_request_does_not_take_a_voice_of_the_macrolanguage(self):
        self._voices("ar-EG")
        self.assertEqual(self._ids("arz-EG"), [])

    def test_another_language_stays_out(self):
        self._voices("he-IL", "fa-IR")
        self.assertEqual(self._ids("ar-EG"), [])

    def test_the_rule_holds_for_arabic_only(self):
        # linguatag also measures lb asking for de, and vro asking for et, at 10
        self._voices("lb-LU", "vro-EE")
        self.assertEqual((self._ids("de"), self._ids("et")), ([], []))

    def test_a_member_language_request_stays_as_strict_as_before(self):
        # vro asking for et is 10, at the threshold, and the lookup keeps under 10
        self._voices("et-EE")
        self.assertEqual(self._ids("vro"), [])


def test_the_habibi_dialect_voices_answer_their_regional_request(tmp_path):
    manager = TTSModelManager(cache_path=str(tmp_path / "cache.json"))
    manager.merge_default_voices()
    expected = {"ar-DZ": "habibi/ar-alg", "ar-EG": "habibi/ar-egy", "ar-IQ": "habibi/ar-irq",
                "ar-MA": "habibi/ar-mar", "ar-SA": "habibi/ar-sau", "ar-AE": "habibi/ar-uae"}
    for request, voice_id in expected.items():
        ids = [v.voice_id for v in manager.get_lang_voices(request)]
        assert voice_id in ids, request
        assert [i for i in ids if i in expected.values()] == [voice_id], request

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
