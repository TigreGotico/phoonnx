"""phoonnx takes text normalization from text2spoken.

The names of phoonnx.normalization, and the same names read from phoonnx.util, are
deprecated: each one warns, and ``normalize`` returns what text2spoken returns.
phoonnx's own call sites call text2spoken directly, so they emit no warning.
"""
import json
import subprocess
import sys
import unittest
import warnings
from importlib import import_module
from pathlib import Path
from unittest.mock import MagicMock

import text2spoken

import phoonnx.normalization as normalization
import phoonnx.util as util
from phoonnx.version import VERSION_MAJOR

# Each helper text2spoken has no equivalent of, and the submodule that keeps its local code.
LOCAL_HELPERS = {
    "CONTRACTIONS": "tables", "TITLES": "tables", "UNITS": "tables",
    "_get_number_separators": "numbers", "_normalize_number_word": "numbers", "is_fraction": "numbers",
    "_normalize_dates_and_times": "datetimes", "pronounce_date": "datetimes", "pronounce_time": "datetimes",
    "_normalize_units": "units",
    "_normalize_word": "text", "_normalize_word_hyphen_digit": "text",
}

SENTENCES = [
    ("Dr Smith paid £3.50 for 2 tickets on 14 March 2024.", "en-GB"),
    ("I don't know, it's 12 o'clock", "en-US"),
    ("O jogo é 3-2 e custou 1,5 euros.", "pt-PT"),
]

# phoonnx would expand both contractions; text2spoken leaves them as written.
CONTRACTED = "I don't know, it's 12 o'clock"


def shim_warnings(caught) -> list:
    return [str(w.message) for w in caught if "phoonnx.normalization" in str(w.message)]


class TestShimNamesWarn(unittest.TestCase):

    def test_normalize_warns_and_returns_what_text2spoken_returns(self):
        for module in (normalization, util):
            for text, lang in SENTENCES:
                with self.subTest(module=module.__name__, text=text):
                    with self.assertWarns(DeprecationWarning) as caught:
                        shim = module.normalize
                    self.assertIn(f"phoonnx {VERSION_MAJOR + 1}.0.0", str(caught.warning))
                    expected = text2spoken.normalize(text, lang)
                    self.assertEqual(shim(text, lang), expected)
                    self.assertEqual(shim(text, lang, expand_contractions=False), expected)

    def test_each_local_helper_warns_and_is_the_local_code(self):
        for module in (normalization, util):
            for name, submodule in LOCAL_HELPERS.items():
                with self.subTest(module=module.__name__, name=name):
                    with self.assertWarns(DeprecationWarning):
                        value = getattr(module, name)
                    local = import_module(f"phoonnx.normalization.{submodule}")
                    self.assertIs(value, getattr(local, name))


class TestCallSitesUseText2spoken(unittest.TestCase):

    def test_every_phonemizer_gets_text2spoken_normalize_without_a_warning(self):
        from phoonnx.config import PhonemeType, get_phonemizer
        for phoneme_type in (PhonemeType.ESPEAK, PhonemeType.GRAPHEMES, PhonemeType.GRUUT):
            with self.subTest(phoneme_type=phoneme_type):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    phonemizer = get_phonemizer(phoneme_type)
                    phonemizer.normalizer(CONTRACTED, "en-US")
                self.assertIs(phonemizer.normalizer, text2spoken.normalize)
                self.assertEqual(shim_warnings(caught), [])

    def test_scorer_phonemizes_the_text2spoken_text_without_a_warning(self):
        from phoonnx_train.evaluation.scorer import text_to_ids
        phonemizer = MagicMock()
        phonemizer.phonemize_to_list.return_value = [["a"]]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            text_to_ids(CONTRACTED, phonemizer, MagicMock(), "en-US")
        phonemizer.phonemize_to_list.assert_called_once_with(text2spoken.normalize(CONTRACTED, "en-US"), "en-US")
        self.assertEqual(shim_warnings(caught), [])

    def test_importing_phoonnx_emits_no_shim_warning(self):
        """Run in a fresh interpreter, so that every module import happens inside the check."""
        code = (
            "import json, warnings\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    warnings.simplefilter('always')\n"
            "    import text2spoken\n"
            "    import phoonnx.util, phoonnx.config, phoonnx.ssml_say_as\n"
            "    import phoonnx_train.preprocess, phoonnx_train.evaluation.scorer\n"
            "    phoonnx.ssml_say_as.say_as_text('5 km', 'unit', 'en')\n"
            "print(json.dumps({\n"
            "    'warnings': [str(w.message) for w in caught if 'phoonnx.normalization' in str(w.message)],\n"
            "    'preprocess_uses_text2spoken': phoonnx_train.preprocess.normalize is text2spoken.normalize}))\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                                cwd=Path(__file__).resolve().parents[1])
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(report, {"warnings": [], "preprocess_uses_text2spoken": True})


if __name__ == "__main__":
    unittest.main()
