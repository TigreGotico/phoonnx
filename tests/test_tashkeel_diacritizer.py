"""Tests for the Arabic diacritizer-model config wiring.

phoonnx owns no diacritizer implementation: diacritization is delegated to
``scriptconv.diacritics.diacritize`` (see test_scriptconv_integration.py). What
phoonnx keeps is the *config* — SynthesisConfig / VoiceConfig carry the
``diacritizer_model`` selection and must round-trip it faithfully.
"""
import unittest

from phoonnx.config import SynthesisConfig, VoiceConfig


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
