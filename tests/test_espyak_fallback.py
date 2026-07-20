"""EspeakPhonemizer falls back to the pure-Python espyak G2P when the
espeak-ng binary is unavailable, and can be forced to prefer it."""
import unittest
from unittest.mock import patch

from phoonnx.phonemizers.mul import EspeakPhonemizer, EspeakError


class TestEspyakFallback(unittest.TestCase):
    def setUp(self):
        EspeakPhonemizer._binary_available = None

    def tearDown(self):
        EspeakPhonemizer._binary_available = None

    def test_prefer_espyak_skips_binary(self):
        pho = EspeakPhonemizer(prefer_espyak=True)
        with patch.object(EspeakPhonemizer, "_run_espeak_command") as run_cmd:
            result = pho.phonemize_string("hello world", "en")
            run_cmd.assert_not_called()
        self.assertIn("ə", result)

    def test_fallback_when_binary_missing(self):
        with patch("shutil.which", return_value=None):
            pho = EspeakPhonemizer()
            result = pho.phonemize_string("hallo wereld", "nl")
        self.assertTrue(result.strip())

    def test_binary_used_when_available(self):
        EspeakPhonemizer._binary_available = True
        pho = EspeakPhonemizer()
        with patch.object(EspeakPhonemizer, "_run_espeak_command",
                          return_value="mocked") as run_cmd:
            self.assertEqual(pho.phonemize_string("hello", "en"), "mocked")
            run_cmd.assert_called_once()

    def test_error_when_neither_available(self):
        with patch("shutil.which", return_value=None), \
                patch.object(EspeakPhonemizer, "_espyak_phonemize",
                             side_effect=ImportError("no espyak")):
            pho = EspeakPhonemizer()
            with self.assertRaises(EspeakError):
                pho.phonemize_string("hello", "en")

    def test_g2p_instances_cached_per_lang(self):
        pho = EspeakPhonemizer(prefer_espyak=True)
        pho.phonemize_string("one", "en")
        pho.phonemize_string("two", "en")
        self.assertEqual(len(pho._espyak_g2p), 1)

    def test_matches_binary_output_shape(self):
        # dialect resolution goes through get_lang (en-gb -> en-gb-x-rp)
        pho = EspeakPhonemizer(prefer_espyak=True)
        result = pho.phonemize_string("water", "en-gb")
        self.assertTrue(result.strip())


if __name__ == "__main__":
    unittest.main()
