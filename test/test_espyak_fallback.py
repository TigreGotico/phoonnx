"""EspeakPhonemizer prefers the espeak-ng shared library, falls back to the
espeak-ng subprocess, and finally to the pure-Python espyak G2P. It can also
be forced to prefer espyak."""
import unittest
from unittest.mock import patch

from phoonnx.phonemizers.mul import EspeakPhonemizer, EspeakError


class TestEspyakFallback(unittest.TestCase):
    def setUp(self):
        EspeakPhonemizer.reset_availability_cache()

    def tearDown(self):
        EspeakPhonemizer.reset_availability_cache()

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

    def test_library_used_when_available(self):
        """The shared library is preferred over spawning a subprocess."""
        EspeakPhonemizer._binary_available = True
        EspeakPhonemizer._library_available = True
        pho = EspeakPhonemizer()
        with patch.object(EspeakPhonemizer, "_library_phonemize",
                          return_value="from-library") as lib, \
                patch.object(EspeakPhonemizer, "_run_espeak_command") as run_cmd:
            self.assertEqual(pho.phonemize_string("hello", "en"), "from-library")
            lib.assert_called_once()
            run_cmd.assert_not_called()

    def test_binary_used_when_library_unavailable(self):
        """Without the library, the subprocess is used as before."""
        EspeakPhonemizer._binary_available = True
        EspeakPhonemizer._library_available = False
        pho = EspeakPhonemizer()
        with patch.object(EspeakPhonemizer, "_run_espeak_command",
                          return_value="mocked") as run_cmd:
            self.assertEqual(pho.phonemize_string("hello", "en"), "mocked")
            run_cmd.assert_called_once()

    def test_binary_used_when_library_opted_out(self):
        """use_library=False keeps the original subprocess behaviour."""
        EspeakPhonemizer._binary_available = True
        EspeakPhonemizer._library_available = True
        pho = EspeakPhonemizer(use_library=False)
        with patch.object(EspeakPhonemizer, "_run_espeak_command",
                          return_value="mocked") as run_cmd:
            self.assertEqual(pho.phonemize_string("hello", "en"), "mocked")
            run_cmd.assert_called_once()

    def test_library_failure_falls_back_to_binary(self):
        """A language the library cannot load still gets phonemized."""
        EspeakPhonemizer._binary_available = True
        EspeakPhonemizer._library_available = True
        pho = EspeakPhonemizer()
        with patch.object(EspeakPhonemizer, "_library_phonemize",
                          side_effect=RuntimeError("failed to load voice")), \
                patch.object(EspeakPhonemizer, "_run_espeak_command",
                             return_value="mocked") as run_cmd:
            self.assertEqual(pho.phonemize_string("hello", "en"), "mocked")
            run_cmd.assert_called_once()
            # the failure is remembered, so the library is not retried
            self.assertIn(pho.get_lang("en"), pho._library_failed)

    def test_error_when_nothing_available(self):
        with patch("shutil.which", return_value=None), \
                patch.object(EspeakPhonemizer, "_has_library", return_value=False), \
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

    def test_backend_instances_cached_per_lang(self):
        """Backends are expensive to build, so they are reused per language."""
        if not EspeakPhonemizer._has_library():
            self.skipTest("espeak-ng shared library not available")
        pho = EspeakPhonemizer()
        pho.phonemize_string("one", "en")
        pho.phonemize_string("two", "en")
        self.assertEqual(len(pho._library_backends), 1)

    def test_matches_binary_output_shape(self):
        # dialect resolution goes through get_lang (en-gb -> en-gb-x-rp)
        pho = EspeakPhonemizer(prefer_espyak=True)
        result = pho.phonemize_string("water", "en-gb")
        self.assertTrue(result.strip())


if __name__ == "__main__":
    unittest.main()
