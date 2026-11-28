import unittest
from unittest import mock

from ovos_number_parser import pronounce_number

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


class TestUnitCase(unittest.TestCase):
    """The unit regex ignores case, so a unit in any case reads like its table key."""

    def test_no_two_units_differ_only_in_case(self):
        for lang, table in units_module.UNITS.items():
            keys = [k for k in table if k.isalnum()]
            with self.subTest(lang=lang):
                self.assertEqual(len(keys), len({k.casefold() for k in keys}))

    def test_unit_in_any_case_reads_like_the_table_key(self):
        for lang, table in units_module.UNITS.items():
            full_lang = util.normalize_lang(lang)
            for key in (k for k in table if k.isalnum()):
                expected = units_module._normalize_units(f"5 {key}", full_lang)
                for variant in (key.lower(), key.upper()):
                    with self.subTest(lang=lang, unit=variant):
                        self.assertEqual(units_module._normalize_units(f"5 {variant}", full_lang), expected)

    def test_portuguese_millilitre_in_lower_case(self):
        out = util.normalize("Junte 50 ml de azeite.", "pt-PT")
        self.assertIn(units_module.UNITS["pt"]["mL"], out)
        self.assertNotIn(" ml ", out)

    def test_unit_with_a_letter_that_lower_does_not_fold(self):
        # casefold maps the long s to "s"; lower keeps it.
        self.assertEqual(units_module._normalize_units("5 \u017f", "en-GB"),
                         units_module._normalize_units("5 s", "en-GB"))


class TestUnitNumbers(unittest.TestCase):
    """The unit pass reads the number before a unit by the language's separators."""

    def spoken(self, value, lang, unit):
        return f"{pronounce_number(value, lang)} {units_module.UNITS[lang.split('-')[0]][unit]}"

    def test_number_before_a_unit_reads_its_value(self):
        cases = [("1,500 km", "en-US", 1500, "km"), ("1,500,000 km", "en-US", 1500000, "km"),
                 ("12,345.6 km", "en-US", 12345.6, "km"), ("2.5 km", "en-US", 2.5, "km"),
                 ("1,500%", "en-US", 1500, "%"),
                 ("1.500 km", "pt-PT", 1500, "km"), ("1.500,5 km", "pt-PT", 1500.5, "km"),
                 ("1,5 km", "pt-PT", 1.5, "km"), ("1.5 km", "pt-PT", 1.5, "km"),
                 ("1.25 kg", "pt-PT", 1.25, "kg"), ("0.25%", "pt-PT", 0.25, "%"),
                 ("36.6°C", "pt-PT", 36.6, "°C"), ("0.5 km", "es-ES", 0.5, "km")]
        for text, lang, value, unit in cases:
            with self.subTest(text=text, lang=lang):
                self.assertEqual(units_module._normalize_units(text, lang), self.spoken(value, lang, unit))

    def test_number_after_a_letter_and_a_full_stop_is_read(self):
        self.assertEqual(units_module._normalize_units("approx.5 km", "en-US"),
                         "approx." + self.spoken(5, "en-US", "km"))

    def test_number_that_continues_another_number_is_left_as_written(self):
        for text in ("2,5 km", "1234,567 km"):
            with self.subTest(text=text):
                self.assertEqual(units_module._normalize_units(text, "en-US"), text)

    def test_unit_that_matches_only_without_case_is_left_as_written(self):
        # "İ" matches "i" in an IGNORECASE regex, but casefolds to "i̇".
        self.assertEqual(units_module._normalize_units("5 MİN", "pt-PT"), "5 MİN")

    def test_number_that_cannot_be_spoken_is_left_as_written(self):
        with mock.patch.object(units_module, "pronounce_number", side_effect=ValueError):
            self.assertEqual(units_module._normalize_units("5 km", "en-US"), "5 km")

if __name__ == "__main__":
    unittest.main()
