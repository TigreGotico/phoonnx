import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.util import _normalize_word


class TestRemovePunctuation(unittest.TestCase):
    """Adversarial tests for BasePhonemizer.remove_punctuation."""

    def test_arabic_punctuation_removed(self):
        text = "مرحبا، كيف حالك؟"
        result = BasePhonemizer.remove_punctuation(text)
        self.assertNotIn("،", result)
        self.assertNotIn("؟", result)

    def test_curly_quotes_removed(self):
        text = "she said “hello” to me"
        result = BasePhonemizer.remove_punctuation(text)
        self.assertNotIn("“", result)
        self.assertNotIn("”", result)

    def test_intraword_apostrophe_preserved(self):
        text = "don't stop"
        result = BasePhonemizer.remove_punctuation(text)
        self.assertIn("don't", result)

    def test_intraword_curly_apostrophe_preserved(self):
        text = "don’t stop"
        result = BasePhonemizer.remove_punctuation(text)
        self.assertIn("don’t", result)

    def test_intraword_hyphen_preserved(self):
        text = "a well-known fact."
        result = BasePhonemizer.remove_punctuation(text)
        self.assertIn("well-known", result)
        self.assertNotIn(".", result)

    def test_leading_trailing_hyphen_stripped(self):
        # a hyphen not sandwiched between letters is still punctuation
        text = "- dash at start"
        result = BasePhonemizer.remove_punctuation(text)
        self.assertFalse(result.startswith("-"))

    def test_ascii_punctuation_still_removed(self):
        text = "hello, world!"
        result = BasePhonemizer.remove_punctuation(text)
        self.assertNotIn(",", result)
        self.assertNotIn("!", result)


class TestContractionCurlyApostrophe(unittest.TestCase):
    """The CONTRACTIONS lookup keys use straight apostrophes; curly variants must
    be normalized before the lookup or they silently miss."""

    def test_curly_apostrophe_contraction_expands(self):
        rbnf_engine = MagicMock()
        result = _normalize_word("don’t", "en-US", rbnf_engine)
        self.assertEqual(result, "do not")

    def test_straight_apostrophe_contraction_still_expands(self):
        rbnf_engine = MagicMock()
        result = _normalize_word("don't", "en-US", rbnf_engine)
        self.assertEqual(result, "do not")


class TestGruutEmptySentence(unittest.TestCase):
    """An empty gruut sentence result must not IndexError on sent_phonemes[-1]."""

    def test_empty_sentence_phonemes_skipped(self):
        from phoonnx.phonemizers.mul import GruutPhonemizer

        fake_sentence = MagicMock()
        fake_sentence.__iter__ = lambda self: iter([])
        fake_sentence.__bool__ = lambda self: False  # empty sentence -> empty sent_phonemes
        fake_sentence.text = ""

        phonemizer = GruutPhonemizer.__new__(GruutPhonemizer)
        phonemizer.get_lang = lambda lang: lang

        with patch("gruut.sentences", return_value=[fake_sentence]):
            # must not raise IndexError; empty result is simply skipped
            result = list(phonemizer._text_to_phonemes("...", "en"))
        self.assertEqual(result, [])


class TestConverterCachePoisoning(unittest.TestCase):
    """A transient converter-construction failure must not poison the module-level
    cache, and a runtime failure during conversion must degrade gracefully."""

    def test_kakasi_construction_failure_not_cached(self):
        import phoonnx.lang_preprocess as lp

        lp._kakasi = None
        fake_pykakasi = MagicMock()
        fake_pykakasi.kakasi.side_effect = RuntimeError("boom")
        with patch.dict("sys.modules", {"pykakasi": fake_pykakasi}):
            result = lp.japanese_to_hiragana("こんにちは")
        self.assertEqual(result, "こんにちは")
        self.assertIsNone(lp._kakasi)  # not poisoned, retry allowed next call

    def test_kakasi_convert_runtime_error_degrades_gracefully(self):
        import phoonnx.lang_preprocess as lp

        fake_kakasi = MagicMock()
        fake_kakasi.convert.side_effect = RuntimeError("boom")
        lp._kakasi = fake_kakasi
        result = lp.japanese_to_hiragana("こんにちは")
        self.assertEqual(result, "こんにちは")
        lp._kakasi = None  # reset global for other tests

    def test_cangjie_construction_failure_not_cached(self):
        import phoonnx.lang_preprocess as lp

        lp._cangjie = None
        with patch.object(lp, "ChineseCangjieConverter", side_effect=RuntimeError("boom")):
            result = lp.chinese_to_cangjie("你好")
        self.assertEqual(result, "你好")
        self.assertIsNone(lp._cangjie)

    def test_cangjie_call_runtime_error_degrades_gracefully(self):
        import phoonnx.lang_preprocess as lp

        fake_converter = MagicMock(side_effect=RuntimeError("boom"))
        lp._cangjie = fake_converter
        result = lp.chinese_to_cangjie("你好")
        self.assertEqual(result, "你好")
        lp._cangjie = None  # reset global for other tests


class TestPyarabicNoSyntaxWarning(unittest.TestCase):
    def test_import_raises_no_syntax_warning(self):
        proc = subprocess.run(
            [sys.executable, "-W", "error::SyntaxWarning", "-c",
             "import phoonnx.thirdparty.mantoq.pyarabic.trans"],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
