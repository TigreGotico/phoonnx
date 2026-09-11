import unittest

import phoonnx.normalization as normalization
import phoonnx.util as util
from phoonnx.normalization import datetimes, numbers, tables, text
from phoonnx.normalization import units as units_module

# Every name phoonnx.util exported before the normalization pipeline moved into
# its own package. Importing any of them from phoonnx.util must keep working.
MOVED_NAMES = [
    "CONTRACTIONS", "TITLES", "UNITS",
    "normalize", "is_fraction", "pronounce_date", "pronounce_time",
    "_get_number_separators", "_normalize_number_word",
    "_normalize_dates_and_times", "_normalize_word_hyphen_digit",
    "_normalize_units", "_normalize_word",
]


class TestUtilReExports(unittest.TestCase):

    def test_every_moved_name_is_still_importable_from_util(self):
        for name in MOVED_NAMES:
            with self.subTest(name=name):
                self.assertIs(getattr(util, name), getattr(normalization, name))

    def test_log_is_still_reachable_through_util(self):
        from phoonnx.log import LOG
        self.assertIs(util.LOG, LOG)

    def test_lang_identity_helpers_stay_in_util(self):
        self.assertEqual(util.normalize_lang("cy"), "cy-GB")
        self.assertEqual(util.match_lang("pt-BR", ["pt-PT", "en-US"]), ("pt-PT", 5))


class TestPackageLayout(unittest.TestCase):
    """Tables are data, apart from the code that applies them."""

    def test_tables_module_holds_only_data(self):
        self.assertEqual(
            {"CONTRACTIONS", "TITLES", "UNITS"},
            {n for n in vars(tables) if not n.startswith("__")},
        )

    def test_each_concern_owns_its_transformations(self):
        self.assertTrue(hasattr(numbers, "_normalize_number_word"))
        self.assertTrue(hasattr(datetimes, "_normalize_dates_and_times"))
        self.assertTrue(hasattr(units_module, "_normalize_units"))
        self.assertTrue(hasattr(text, "normalize"))


class TestPipelineStillNormalizes(unittest.TestCase):
    """A spot check on the whole pipeline, independent of the unit tests."""

    def test_english_sentence_leaves_no_digits_or_symbols(self):
        out = normalization.normalize("Dr. Smith owes me 12.5% of 3/4 at 25°C", "en")
        self.assertFalse(any(c.isdigit() for c in out), out)
        self.assertIn("Doctor", out)
        self.assertIn("per cent", out)
        self.assertIn("degrees celsius", out)

    def test_portuguese_uses_the_portuguese_tables(self):
        out = normalization.normalize("Sr. Silva pagou 10kg", "pt")
        self.assertIn("Senhor", out)
        self.assertIn("quilogramas", out)

    def test_digit_hyphen_digit_score_is_left_intact_pt(self):
        # A score is a range, not a word glued to a digit: the phonemizer,
        # not this pipeline, is responsible for reading "3-2" aloud.
        out = normalization.normalize("O jogo é 3-2.", "pt-PT")
        self.assertEqual(out, "O jogo é 3-2.")

    def test_digit_hyphen_digit_score_is_left_intact_en(self):
        out = normalization.normalize("The score is 3-2.", "en-US")
        self.assertEqual(out, "The score is 3-2.")

    def test_digit_hyphen_digit_page_range_is_left_intact(self):
        out = normalization.normalize("páginas 1139-1185", "pt-PT")
        self.assertEqual(out, "páginas 1139-1185")

    def test_digit_en_dash_digit_range_is_left_intact(self):
        out = normalization.normalize("páginas 1139–1185", "pt-PT")
        self.assertEqual(out, "páginas 1139–1185")

    def test_word_hyphen_digit_still_normalizes(self):
        out = normalization.normalize("sub-23", "en-US")
        self.assertNotIn("-", out)
        self.assertIn("twenty", out.lower())


if __name__ == "__main__":
    unittest.main()
