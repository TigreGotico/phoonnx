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
    """A number before a unit reads the same with or without its thousands separator."""

    def test_grouped_number_reads_like_the_plain_number(self):
        cases = [("1,500 km", "1500 km", "en-US"), ("12,345.6 km", "12345.6 km", "en-US"),
                 ("1,500,000 km", "1500000 km", "en-US"), ("1,500%", "1500%", "en-US"),
                 ("1.500 km", "1500 km", "pt-PT"), ("1.500,5 km", "1500,5 km", "pt-PT")]
        for grouped, plain, lang in cases:
            with self.subTest(text=grouped, lang=lang):
                self.assertEqual(units_module._normalize_units(grouped, lang),
                                 units_module._normalize_units(plain, lang))

    def test_number_that_is_not_grouped_in_threes_is_left_as_written(self):
        for text, lang in [("2,5 km", "en-US"), ("1234,567 km", "en-US"), ("1.5 km", "pt-PT")]:
            with self.subTest(text=text, lang=lang):
                self.assertEqual(units_module._normalize_units(text, lang), text)

    def test_unit_that_matches_only_without_case_is_left_as_written(self):
        # "İ" matches "i" in an IGNORECASE regex, but casefolds to "i̇".
        self.assertEqual(units_module._normalize_units("5 MİN", "pt-PT"), "5 MİN")

    def test_no_crash_in_full_normalisation(self):
        for text, lang in [("1,500 km", "en-US"), ("2,5 km", "en-US"), ("1.5 km", "pt-PT")]:
            with self.subTest(text=text, lang=lang):
                self.assertNotRegex(util.normalize(text, lang), r"\d")

if __name__ == "__main__":
    unittest.main()
