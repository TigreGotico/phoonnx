"""Test suite for the Shami (Levantine Arabic) phonemizer and adapter."""
import unittest
from unittest.mock import patch

import numpy as np

from phoonnx.config import Alphabet
from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.phonemizers.shami import ShamiPhonemizer
from phoonnx.engines.shami import ShamiAdapter
from phoonnx.engines.base import AdapterSynthesisRequest


class TestShamiPhonemizer(unittest.TestCase):
    """Unit tests for ShamiPhonemizer (Levantine Arabic / English code-switching)."""

    def setUp(self):
        self.phonemizer = ShamiPhonemizer()

    def test_init(self):
        self.assertIsInstance(self.phonemizer, BasePhonemizer)
        self.assertEqual(self.phonemizer.alphabet, Alphabet.IPA)

    def test_get_lang_valid(self):
        for code in ["ar", "AR", "ar-LB", "ar_LB", "ar-SY", "en", "en-US", "ara"]:
            with self.subTest(code=code):
                self.assertIn(self.phonemizer.get_lang(code), {"ar", "en"})

    def test_get_lang_invalid(self):
        for code in ["fr", "de", "es", "zh", "", " "]:
            with self.subTest(code=code):
                with self.assertRaises(ValueError):
                    self.phonemizer.get_lang(code)

    def test_phonemize_with_language_ids(self):
        """phonemize_with_language_ids returns IPA chunks and per-token language IDs."""
        phonemes, lang_ids = self.phonemizer.phonemize_with_language_ids("hello", "en")
        self.assertIsInstance(phonemes, list)
        self.assertIsInstance(lang_ids, list)
        self.assertEqual(len(lang_ids), len(phonemes))
        for chunk_ids in lang_ids:
            self.assertIsInstance(chunk_ids, list)

    def test_phonemize_string_arabic(self):
        """phonemize_string returns a non-empty IPA string for Arabic."""
        result = self.phonemizer.phonemize_string("مرحبا", "ar")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_phonemize_string_english(self):
        """phonemize_string returns a non-empty IPA string for English."""
        result = self.phonemizer.phonemize_string("hello", "en")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_phonemize_mixed_language_ids(self):
        """Mixed Arabic/English text yields distinct language IDs."""
        phonemes, lang_ids = self.phonemizer.phonemize_with_language_ids(
            "مرحبا hello", "ar-LB"
        )
        self.assertGreater(len(phonemes), 0)
        all_ids = [lid for chunk in lang_ids for lid in chunk]
        self.assertTrue(set(all_ids).issubset({0, 1, 2, 3}))
        self.assertGreater(all_ids.count(1), 0)
        self.assertGreater(all_ids.count(2), 0)

    def test_default_language_parameter(self):
        """Default language should be ar-LB."""
        result = self.phonemizer.phonemize_string("مرحبا", "ar-LB")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_empty_text(self):
        phonemes, lang_ids = self.phonemizer.phonemize_with_language_ids("", "ar")
        self.assertEqual(phonemes, [])
        self.assertEqual(lang_ids, [])


class TestShamiAdapter(unittest.TestCase):
    """Unit tests for ShamiAdapter ONNX inference wrapper."""

    def test_detect_by_config(self):
        self.assertTrue(ShamiAdapter.detect(config={"engine": "shami"}))
        self.assertTrue(ShamiAdapter.detect(config={"engine": "hams"}))
        self.assertFalse(ShamiAdapter.detect(config={"engine": "piper"}))

    def test_default_params(self):
        adapter = ShamiAdapter()
        params = adapter.default_params()
        self.assertIn("noise_scale", params)
        self.assertIn("length_scale", params)
        self.assertIn("noise_w_scale", params)

    @patch("phoonnx.engines.shami.onnxruntime.InferenceSession")
    def test_synthesize_request_shape(self, mock_session_cls):
        mock_session = mock_session_cls.return_value
        mock_session.get_inputs.return_value = [
            type("Input", (), {"name": "phoneme_ids"})(),
            type("Input", (), {"name": "phoneme_lengths"})(),
            type("Input", (), {"name": "language_ids"})(),
        ]
        mock_session.run.return_value = [np.zeros((1, 100), dtype=np.float32)]

        adapter = ShamiAdapter()
        adapter.session = mock_session
        req = AdapterSynthesisRequest(
            phoneme_ids=np.array([[1, 2, 3]], dtype=np.int64),
            phoneme_lengths=np.array([3], dtype=np.int64),
            language_ids=np.array([[1, 1, 1]], dtype=np.int64),
        )
        result = adapter.synthesize(req, mock_session)
        self.assertIsInstance(result.audio, np.ndarray)
        mock_session.run.assert_called_once()
        args, _ = mock_session.run.call_args
        self.assertIsNone(args[0])
        feed = args[1]
        self.assertIn("language_ids", feed)
        np.testing.assert_array_equal(feed["language_ids"], req.language_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=True)
