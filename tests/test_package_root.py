"""The package root exports and the PhoonnxError hierarchy.

``import phoonnx`` must stay cheap: the training package and every submodule
import the package, so the root resolves its names on first access instead of
importing :mod:`phoonnx.voice` up front.
"""
import subprocess
import sys
import unittest

import phoonnx.errors as errors
from phoonnx._bpe import UnsupportedTokenizer
from phoonnx._sentencepiece import UnsupportedSentencePieceModel
from phoonnx.config import PhonemeType, UnsupportedVoiceLanguage
from phoonnx.model_manager import PrecisionUnavailable
from phoonnx.voice_cache import VoiceExceedsMemoryBudget


class TestPackageRoot(unittest.TestCase):

    def test_root_names_are_the_home_module_objects(self):
        import scriptconv.frontend

        import phoonnx.config
        import phoonnx.model_manager
        import phoonnx.voice

        homes = {
            "TTSVoice": phoonnx.voice,
            "AudioChunk": phoonnx.voice,
            "Utterance": phoonnx.voice,
            "VoiceConfig": phoonnx.config,
            "SynthesisConfig": phoonnx.config,
            "TTSModelManager": phoonnx.model_manager,
            "Stages": scriptconv.frontend,
            "FrontendConfig": scriptconv.frontend,
            "Chunk": scriptconv.frontend,
        }
        self.assertEqual(set(phoonnx.__all__), set(homes))
        for name, home in homes.items():
            with self.subTest(name=name):
                self.assertIs(getattr(phoonnx, name), getattr(home, name))
        self.assertFalse(hasattr(phoonnx, "NoSuchName"))

    def test_from_import_of_the_root_names(self):
        from phoonnx import SynthesisConfig, TTSVoice
        from phoonnx.config import SynthesisConfig as home_config
        from phoonnx.voice import TTSVoice as home_voice

        self.assertIs(TTSVoice, home_voice)
        self.assertIs(SynthesisConfig, home_config)

    def test_root_import_defers_the_voice_module(self):
        code = (
            "import sys, phoonnx\n"
            "assert 'phoonnx.voice' not in sys.modules, 'eager import'\n"
            "from phoonnx import TTSVoice\n"
            "assert sys.modules['phoonnx.voice'].TTSVoice is TTSVoice\n"
            "import phoonnx.errors\n"
        )
        result = subprocess.run([sys.executable, "-c", code],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_errors_module_does_not_import_the_voice_module(self):
        code = (
            "import sys, phoonnx.errors\n"
            "assert 'phoonnx.voice' not in sys.modules, 'eager import'\n"
            "assert issubclass(phoonnx.errors.UnsupportedTokenizer,"
            " phoonnx.errors.PhoonnxError)\n"
        )
        result = subprocess.run([sys.executable, "-c", code],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


class TestErrorHierarchy(unittest.TestCase):

    def old_errors(self):
        """Each error that predates the hierarchy, an instance of it, and its builtin base."""
        return [
            (UnsupportedTokenizer, UnsupportedTokenizer("Split pattern"), Exception),
            (UnsupportedSentencePieceModel,
             UnsupportedSentencePieceModel("protobuf wire type 7"), Exception),
            (UnsupportedVoiceLanguage,
             UnsupportedVoiceLanguage("voice", "xx", PhonemeType.ESPEAK), ValueError),
            (VoiceExceedsMemoryBudget, VoiceExceedsMemoryBudget("too big"),
             RuntimeError),
            (PrecisionUnavailable, PrecisionUnavailable("demo/voice", "int8"), ValueError),
        ]

    def test_each_old_error_is_caught_by_phoonnx_error_and_its_old_base(self):
        for cls, instance, base in self.old_errors():
            with self.subTest(error=cls.__name__):
                self.assertIs(getattr(errors, cls.__name__), cls)
                with self.assertRaises(errors.PhoonnxError) as caught:
                    raise instance
                self.assertIs(caught.exception, instance)
                with self.assertRaises(base) as caught:
                    raise instance
                self.assertIs(caught.exception, instance)

    def test_unsupported_voice_language_keeps_its_fields(self):
        error = UnsupportedVoiceLanguage("voice", "xx", PhonemeType.ESPEAK)
        self.assertIsInstance(error, errors.PhoonnxError)
        self.assertEqual((error.voice, error.lang_code, error.phoneme_type),
                         ("voice", "xx", PhonemeType.ESPEAK))
        self.assertEqual(
            str(error),
            "voice 'voice': no phonemizer backend supports lang 'xx' for "
            "phoneme_type 'espeak'")

    def test_a_missing_precision_is_a_load_error(self):
        error = PrecisionUnavailable("demo/voice", "int8")
        self.assertIsInstance(error, errors.VoiceLoadError)
        self.assertEqual((error.voice_id, error.precision), ("demo/voice", "int8"))
        self.assertEqual(str(error), "voice 'demo/voice' has no int8 files; "
                                     "precision 'auto' or 'full' loads it")

    def test_new_error_tree(self):
        parents = {
            "PhoonnxError": Exception,
            "VoiceLoadError": errors.PhoonnxError,
            "FrontendMismatchError": errors.VoiceLoadError,
            "UnsupportedConfigError": errors.VoiceLoadError,
            "ExportNotSupported": errors.PhoonnxError,
        }
        for name, parent in parents.items():
            with self.subTest(error=name):
                self.assertEqual(getattr(errors, name).__bases__, (parent,))


if __name__ == "__main__":
    unittest.main()
