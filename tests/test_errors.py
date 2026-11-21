import multiprocessing
import pickle
import unittest
from concurrent.futures import ProcessPoolExecutor

import phoonnx.errors as errors
from phoonnx._bpe import UnsupportedTokenizer
from phoonnx._sentencepiece import UnsupportedSentencePieceModel
from phoonnx.config import UnsupportedVoiceLanguage
from phoonnx.voice_cache import VoiceExceedsMemoryBudget
from scriptconv.phonemizers.enums import Phonemizer


class TestErrorTaxonomy(unittest.TestCase):
    """errors.py collects the typed failures without forking them."""

    def test_collected_errors_are_the_classes_the_code_raises(self):
        self.assertIs(errors.UnsupportedVoiceLanguage, UnsupportedVoiceLanguage)
        self.assertIs(errors.VoiceExceedsMemoryBudget, VoiceExceedsMemoryBudget)
        self.assertIs(errors.UnsupportedTokenizer, UnsupportedTokenizer)
        self.assertIs(errors.UnsupportedSentencePieceModel, UnsupportedSentencePieceModel)

    def test_all_lists_every_collected_error(self):
        self.assertEqual(
            set(errors.__all__),
            {"PhoonnxError", "VoiceLoadError", "FrontendMismatchError",
             "UnsupportedConfigError", "ExportNotSupported", "PrecisionUnavailable",
             "UnsupportedVoiceLanguage", "VoiceExceedsMemoryBudget",
             "UnsupportedTokenizer", "UnsupportedSentencePieceModel", "NonFiniteAudioError",
             "UnknownVoice"},
        )

    def test_resource_refusal_is_distinguishable_from_a_config_defect(self):
        self.assertTrue(issubclass(errors.VoiceExceedsMemoryBudget, RuntimeError))
        self.assertTrue(issubclass(errors.UnsupportedVoiceLanguage, ValueError))
        self.assertFalse(issubclass(errors.UnsupportedVoiceLanguage,
                                    errors.VoiceExceedsMemoryBudget))

    def test_catching_via_errors_catches_what_the_raiser_raises(self):
        with self.assertRaises(errors.VoiceExceedsMemoryBudget):
            raise VoiceExceedsMemoryBudget("too big")



def _raise_precision_unavailable():
    raise errors.PrecisionUnavailable("v", "int8")


class TestErrorsPickle(unittest.TestCase):
    """A typed error survives pickling, so it crosses from a worker process to its parent."""

    def assertRoundTrips(self, error, message, **fields):
        restored = pickle.loads(pickle.dumps(error))
        self.assertIs(type(restored), type(error))
        self.assertEqual(str(restored), message)
        for name, value in fields.items():
            self.assertEqual(getattr(restored, name), value)

    def test_precision_unavailable(self):
        self.assertRoundTrips(errors.PrecisionUnavailable("v", "int8"),
                              "voice 'v' has no int8 files; precision 'auto' or 'full' loads it",
                              voice_id="v", precision="int8")

    def test_unsupported_voice_language(self):
        self.assertRoundTrips(errors.UnsupportedVoiceLanguage("v", "xx", Phonemizer.ESPEAK),
                              "voice 'v': no phonemizer backend supports lang 'xx' for phoneme_type 'espeak'",
                              voice="v", lang_code="xx", phoneme_type=Phonemizer.ESPEAK)

    def test_non_finite_audio_keeps_its_message(self):
        self.assertRoundTrips(errors.NonFiniteAudioError("m.onnx"),
                              "model 'm.onnx' returned non-finite audio samples (NaN or inf)",
                              model_path="m.onnx")

    def test_a_worker_process_error_reaches_the_parent(self):
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
            with self.assertRaises(errors.PrecisionUnavailable) as raised:
                pool.submit(_raise_precision_unavailable).result()
        self.assertEqual(raised.exception.precision, "int8")


if __name__ == "__main__":
    unittest.main()
