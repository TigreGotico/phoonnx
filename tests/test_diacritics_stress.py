"""Tests for add_diacritics routing — Slavic stress (ru/uk/be) and existing ar/he backends."""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from phoonnx.phonemizers.base import BasePhonemizer, GraphemePhonemizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stub_stressonnx(*, has_model_kwarg: bool = True):
    """Return a minimal stressonnx stub module."""
    mod = types.ModuleType("stressonnx")
    if has_model_kwarg:
        def stress(text, lang, model=None):
            return f"[stressed:{lang}:{model}]{text}"
    else:
        def stress(text, lang):
            return f"[stressed:{lang}]{text}"
    mod.stress = stress
    return mod


class TestAddDiacriticsRouting(unittest.TestCase):
    """add_diacritics dispatches to the correct per-language backend."""

    def setUp(self):
        self.phonemizer = GraphemePhonemizer()

    # -- Russian / Ukrainian / Belarusian ------------------------------------

    def test_ru_routes_to_stressonnx(self):
        stub = _make_stub_stressonnx()
        with patch.dict(sys.modules, {"stressonnx": stub}):
            result = self.phonemizer.add_diacritics("замок", "ru")
        self.assertEqual(result, "[stressed:ru:None]замок")

    def test_uk_routes_to_stressonnx(self):
        stub = _make_stub_stressonnx()
        with patch.dict(sys.modules, {"stressonnx": stub}):
            result = self.phonemizer.add_diacritics("замок", "uk")
        self.assertIn("[stressed:uk:", result)

    def test_be_routes_to_stressonnx(self):
        stub = _make_stub_stressonnx()
        with patch.dict(sys.modules, {"stressonnx": stub}):
            result = self.phonemizer.add_diacritics("замак", "be")
        self.assertIn("[stressed:be:", result)

    def test_ru_region_tag(self):
        """ru-RU (BCP-47 region suffix) still routes to stress backend."""
        stub = _make_stub_stressonnx()
        with patch.dict(sys.modules, {"stressonnx": stub}):
            result = self.phonemizer.add_diacritics("замок", "ru-RU")
        self.assertIn("[stressed:ru-RU:", result)

    def test_model_kwarg_forwarded(self):
        """When a model name is given it is passed to stressonnx.stress."""
        stub = _make_stub_stressonnx(has_model_kwarg=True)
        with patch.dict(sys.modules, {"stressonnx": stub}):
            result = self.phonemizer.add_diacritics("замок", "ru", model="silero")
        self.assertIn("silero", result)

    def test_model_kwarg_omitted_when_backend_lacks_it(self):
        """If stressonnx.stress has no model param, it is not forwarded (no TypeError)."""
        stub = _make_stub_stressonnx(has_model_kwarg=False)
        with patch.dict(sys.modules, {"stressonnx": stub}):
            # should not raise even though model= was provided
            result = self.phonemizer.add_diacritics("замок", "ru", model="silero")
        self.assertIn("[stressed:ru]", result)

    def test_missing_stressonnx_returns_text_unchanged(self):
        """If stressonnx is not installed the original text is returned (best-effort)."""
        # Remove stressonnx from sys.modules to simulate ImportError
        with patch.dict(sys.modules, {"stressonnx": None}):
            result = self.phonemizer.add_diacritics("замок", "ru")
        self.assertEqual(result, "замок")

    def test_stressonnx_exception_returns_text_unchanged(self):
        """If stressonnx.stress raises, the original text is returned."""
        stub = types.ModuleType("stressonnx")
        stub.stress = MagicMock(side_effect=RuntimeError("model load failed"))
        with patch.dict(sys.modules, {"stressonnx": stub}):
            result = self.phonemizer.add_diacritics("замок", "ru")
        self.assertEqual(result, "замок")

    # -- Arabic / Hebrew — existing backends still work ---------------------

    def test_ar_routes_to_tashkeel(self):
        mock_tashkeel = MagicMock()
        mock_tashkeel.diacritize.return_value = "مَرْحَبًا"
        self.phonemizer._tashkeel = mock_tashkeel
        result = self.phonemizer.add_diacritics("مرحبا", "ar")
        mock_tashkeel.diacritize.assert_called_once()
        self.assertEqual(result, "مَرْحَبًا")

    def test_he_routes_to_phonikud(self):
        mock_phonikud = MagicMock()
        mock_phonikud.diacritize.return_value = "שָׁלוֹם"
        self.phonemizer._phonikud = mock_phonikud
        result = self.phonemizer.add_diacritics("שלום", "he")
        mock_phonikud.diacritize.assert_called_once()
        self.assertEqual(result, "שָׁלוֹם")

    def test_other_lang_passthrough(self):
        """Languages with no diacritization backend return text unchanged."""
        result = self.phonemizer.add_diacritics("hello", "en")
        self.assertEqual(result, "hello")

        result = self.phonemizer.add_diacritics("hola", "es")
        self.assertEqual(result, "hola")


class TestVoiceConfigAutoEnable(unittest.TestCase):
    """VoiceConfig.__post_init__ auto-enables add_diacritics for ru/uk/be."""

    def _make_config(self, lang_code):
        from phoonnx.config import VoiceConfig, PhonemeType, Alphabet, Engine
        from phoonnx.tokenizer import TTSTokenizer, Vocabulary
        tok = TTSTokenizer(Vocabulary(char2idx={"a": 0, "_": 1}),
                           add_blank_char=False, add_blank_word=False,
                           use_eos_bos=False, blank_at_end=False, blank_at_start=False)
        return VoiceConfig(
            num_symbols=2, num_speakers=1, num_langs=1,
            sample_rate=22050, lang_code=lang_code,
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.UNICODE,
            phonemizer_model=None,
            tokenizer=tok,
        )

    def test_ru_auto_enables(self):
        cfg = self._make_config("ru")
        self.assertTrue(cfg.add_diacritics)

    def test_uk_auto_enables(self):
        cfg = self._make_config("uk")
        self.assertTrue(cfg.add_diacritics)

    def test_be_auto_enables(self):
        cfg = self._make_config("be")
        self.assertTrue(cfg.add_diacritics)

    def test_ar_still_auto_enables(self):
        cfg = self._make_config("ar")
        self.assertTrue(cfg.add_diacritics)

    def test_en_does_not_auto_enable(self):
        cfg = self._make_config("en")
        self.assertFalse(cfg.add_diacritics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
