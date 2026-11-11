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
        self.assertTrue(hasattr(text, "_normalize_word"))


if __name__ == "__main__":
    unittest.main()
