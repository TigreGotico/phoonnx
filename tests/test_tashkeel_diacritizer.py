"""Tests for the text2tashkeel-backed Arabic diacritizer wiring.

text2tashkeel is only pulled in by the ``[ar]`` extra, which CI does not
install for these unit tests, so the actual ``Diacritizer`` is mocked out via
``sys.modules`` rather than imported for real (see test_arbtok_phonemizer.py
for the "real dependency" style tests that exercise arbtok instead).
"""
import sys
import types
import unittest
from unittest.mock import MagicMock

from phoonnx.config import SynthesisConfig, VoiceConfig
from phoonnx.phonemizers.base import GraphemePhonemizer


def _install_fake_text2tashkeel():
    """Install a fake ``text2tashkeel`` module and return the mock Diacritizer class."""
    fake_module = types.ModuleType("text2tashkeel")
    diacritizer_cls = MagicMock()
    diacritizer_cls.side_effect = lambda model: MagicMock(
        diacritize=lambda text: f"{text}+{model}")
    fake_module.Diacritizer = diacritizer_cls
    sys.modules["text2tashkeel"] = fake_module
    return diacritizer_cls


def _remove_fake_text2tashkeel():
    sys.modules.pop("text2tashkeel", None)


class TestAddDiacriticsRouting(unittest.TestCase):
    def setUp(self):
        self.diacritizer_cls = _install_fake_text2tashkeel()

    def tearDown(self):
        _remove_fake_text2tashkeel()

    def test_arabic_routes_to_tashkeel_with_requested_model(self):
        p = GraphemePhonemizer()
        out = p.add_diacritics("مرحبا", "ar", model="rawi-ensemble")
        self.assertEqual(out, "مرحبا+rawi-ensemble")
        self.diacritizer_cls.assert_called_once_with("rawi-ensemble")

    def test_default_model_used_when_none_requested(self):
        p = GraphemePhonemizer(diacritizer_model="some-default")
        out = p.add_diacritics("مرحبا", "ar")
        self.assertEqual(out, "مرحبا+some-default")
        self.diacritizer_cls.assert_called_once_with("some-default")

    def test_non_arabic_hebrew_lang_passes_through_unmodified(self):
        p = GraphemePhonemizer()
        out = p.add_diacritics("hello", "en")
        self.assertEqual(out, "hello")
        self.diacritizer_cls.assert_not_called()

    def test_caches_diacritizer_per_model_name(self):
        p = GraphemePhonemizer()
        p.add_diacritics("a", "ar", model="model-a")
        p.add_diacritics("b", "ar", model="model-a")
        p.add_diacritics("c", "ar", model="model-b")
        # one instantiation per distinct model name, reused across calls
        self.assertEqual(self.diacritizer_cls.call_count, 2)
        self.diacritizer_cls.assert_any_call("model-a")
        self.diacritizer_cls.assert_any_call("model-b")
        self.assertIn("model-a", p._tashkeel)
        self.assertIn("model-b", p._tashkeel)


class TestMissingText2Tashkeel(unittest.TestCase):
    def test_missing_dependency_raises_loud_import_error(self):
        # ensure it really looks missing, regardless of what is actually installed
        sys.modules["text2tashkeel"] = None
        try:
            p = GraphemePhonemizer()
            with self.assertRaises(ImportError) as ctx:
                p.add_diacritics("مرحبا", "ar")
            self.assertIn("phoonnx[ar]", str(ctx.exception))
            self.assertIn("text2tashkeel", str(ctx.exception))
        finally:
            sys.modules.pop("text2tashkeel", None)


class TestSynthesisConfigDiacritizerModelDefaults(unittest.TestCase):
    def test_default_is_none(self):
        self.assertIsNone(SynthesisConfig().diacritizer_model)

    def test_effective_model_falls_back_to_voice_config(self):
        voice_config_model = "rawi-ensemble"
        syn_config = SynthesisConfig()
        effective = syn_config.diacritizer_model or voice_config_model
        self.assertEqual(effective, "rawi-ensemble")

    def test_explicit_syn_config_model_overrides_voice_config(self):
        voice_config_model = "rawi-ensemble"
        syn_config = SynthesisConfig(diacritizer_model="other-model")
        effective = syn_config.diacritizer_model or voice_config_model
        self.assertEqual(effective, "other-model")


class TestVoiceConfigDiacritizerModelRoundtrip(unittest.TestCase):
    def _voice_config(self, diacritizer_model="rawi-ensemble"):
        # a native phoonnx config (carries phoonnx_version) so from_dict takes the
        # is_phoonnx() path, where diacritizer_model round-trips through "inference"
        cfg = {
            "phoonnx_version": "1.0",
            "phoneme_type": "espeak",
            "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a": [3], " ": [4]},
            "num_symbols": 5, "num_speakers": 1, "audio": {"sample_rate": 22050},
            "inference": {
                "noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8,
                "add_diacritics": True, "diacritizer_model": diacritizer_model,
            },
        }
        return VoiceConfig.from_dict(cfg, lang_code="ar")

    def test_roundtrip_preserves_diacritizer_model(self):
        vc = self._voice_config(diacritizer_model="rawi-ensemble")
        self.assertEqual(vc.diacritizer_model, "rawi-ensemble")
        native = vc.to_native_dict()
        self.assertEqual(native["inference"]["diacritizer_model"], "rawi-ensemble")
        vc2 = VoiceConfig.from_dict(dict(native), lang_code="ar")
        self.assertEqual(vc2.diacritizer_model, "rawi-ensemble")

    def test_roundtrip_preserves_non_default_diacritizer_model(self):
        vc = self._voice_config(diacritizer_model="some-other-model")
        native = vc.to_native_dict()
        vc2 = VoiceConfig.from_dict(dict(native), lang_code="ar")
        self.assertEqual(vc2.diacritizer_model, "some-other-model")


if __name__ == "__main__":
    unittest.main()
