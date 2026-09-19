"""Voices we publish and recommend must be resolvable through the catalog.

A voice can exist on HuggingFace, be named by ovos-config's per-locale
recommendations, and still be unreachable: the plugin resolves a voice id
against the shipped index, and an id the index does not carry raises
``Unknown voice`` at construction rather than falling back to another voice.

These ids are pinned rather than counted, so an index rewrite that drops one
fails here instead of reaching a user as a language that cannot speak.
"""
import unittest

from phoonnx.model_manager import TTSModelManager

# Named by ovos-config recommends/offline_male and offline_female.
RECOMMENDED = [
    "OpenVoiceOS/phoonnx_fy-NL_miro_unicode",
    "OpenVoiceOS/phoonnx_fy-NL_dii_unicode",
    "OpenVoiceOS/phoonnx_es-CO_miro_espeak",
    "OpenVoiceOS/phoonnx_es-CO_dii_espeak",
    "OpenVoiceOS/phoonnx_an_miro_unicode",
    "OpenVoiceOS/phoonnx_an_dii_unicode",
    "OpenVoiceOS/phoonnx_ast_miro_unicode",
    "OpenVoiceOS/phoonnx_ast_dii_unicode",
    "OpenVoiceOS/phoonnx_oc_miro_unicode",
    "OpenVoiceOS/phoonnx_oc_dii_unicode",
]


class TestRecommendedVoicesResolve(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.manager = TTSModelManager()
        cls.manager.merge_default_voices()

    def test_every_recommended_voice_is_in_the_catalog(self):
        missing = [v for v in RECOMMENDED if self.manager.get_voice(v) is None]
        self.assertEqual([], missing)

    def test_each_one_reports_the_language_it_is_recommended_for(self):
        for voice_id, expected in [(v, v.split("phoonnx_")[1].split("_")[0])
                                   for v in RECOMMENDED]:
            with self.subTest(voice_id):
                info = self.manager.get_voice(voice_id)
                self.assertIsNotNone(info, f"{voice_id} is not in the catalog")
                self.assertTrue(
                    (info.lang or "").lower().startswith(expected.lower()),
                    f"{voice_id} reports lang {info.lang!r}, not {expected!r}")


if __name__ == "__main__":
    unittest.main()
