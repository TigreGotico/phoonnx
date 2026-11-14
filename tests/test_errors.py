import unittest

import phoonnx.errors as errors
from phoonnx._bpe import UnsupportedTokenizer
from phoonnx._sentencepiece import UnsupportedSentencePieceModel
from phoonnx.config import UnsupportedVoiceLanguage
from phoonnx.voice_cache import VoiceExceedsMemoryBudget


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
            {"UnsupportedVoiceLanguage", "VoiceExceedsMemoryBudget",
             "UnsupportedTokenizer", "UnsupportedSentencePieceModel"},
        )

    def test_resource_refusal_is_distinguishable_from_a_config_defect(self):
        self.assertTrue(issubclass(errors.VoiceExceedsMemoryBudget, RuntimeError))
        self.assertTrue(issubclass(errors.UnsupportedVoiceLanguage, ValueError))
        self.assertFalse(issubclass(errors.UnsupportedVoiceLanguage,
                                    errors.VoiceExceedsMemoryBudget))

    def test_catching_via_errors_catches_what_the_raiser_raises(self):
        with self.assertRaises(errors.VoiceExceedsMemoryBudget):
            raise VoiceExceedsMemoryBudget("too big")


if __name__ == "__main__":
    unittest.main()
