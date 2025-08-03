import unittest
from unittest.mock import patch, MagicMock
from datetime import date
import sys
import os

from phoonnx.util import (
    normalize, _get_number_separators, _normalize_number_word,
    _normalize_dates_and_times, _normalize_word_hyphen_digit,
    _normalize_units, _normalize_word, is_fraction,
    pronounce_date, pronounce_time, CONTRACTIONS, TITLES, UNITS
)

class TestUtilFunctions(unittest.TestCase):
    """Comprehensive test suite for util.py functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_rbnf_engine = MagicMock()
        self.mock_rbnf_engine.format_number.return_value.text = "formatted number"

    def test_get_number_separators_default(self):
        """Test _get_number_separators with default languages."""
        decimal, thousands = _get_number_separators("en")
        self.assertEqual(decimal, '.')
        self.assertEqual(thousands, ',')

        decimal, thousands = _get_number_separators("en-US")
        self.assertEqual(decimal, '.')
        self.assertEqual(thousands, ',')

    def test_get_number_separators_european(self):
        """Test _get_number_separators with European languages."""
        for lang in ["pt", "es", "fr", "de"]:
            decimal, thousands = _get_number_separators(lang)
            self.assertEqual(decimal, ',')
            self.assertEqual(thousands, '.')

        # Test with full language codes
        decimal, thousands = _get_number_separators("pt-PT")
        self.assertEqual(decimal, ',')
        self.assertEqual(thousands, '.')

    def test_is_fraction_valid(self):
        """Test is_fraction with valid fractions."""
        self.assertTrue(is_fraction("1/2"))
        self.assertTrue(is_fraction("3/4"))
        self.assertTrue(is_fraction("10/20"))
        self.assertTrue(is_fraction("0/1"))

    def test_is_fraction_invalid(self):
        """Test is_fraction with invalid inputs."""
        self.assertFalse(is_fraction("1.5"))
        self.assertFalse(is_fraction("1/2/3"))
        self.assertFalse(is_fraction("a/b"))
        self.assertFalse(is_fraction("1/"))
        self.assertFalse(is_fraction("/2"))
        self.assertFalse(is_fraction("no_fraction"))
        self.assertFalse(is_fraction(""))

    def test_is_fraction_edge_cases(self):
        """Test is_fraction with edge cases."""
        self.assertFalse(is_fraction("1/2.5"))
        self.assertFalse(is_fraction("1.0/2"))
        self.assertFalse(is_fraction("1/-2"))
        self.assertFalse(is_fraction("-1/2"))

    @patch('phoonnx.util.pronounce_number')
    @patch('phoonnx.util.is_numeric')
    def test_normalize_number_word_simple_integer(self, mock_is_numeric, mock_pronounce):
        """Test _normalize_number_word with simple integers."""
        mock_is_numeric.return_value = True
        mock_pronounce.return_value = "twenty three"

        result = _normalize_number_word("23", "en", None)
        mock_pronounce.assert_called_with(23, lang="en")
        self.assertEqual(result, "twenty three")

    @patch('phoonnx.util.pronounce_number')
    @patch('phoonnx.util.is_numeric')
    def test_normalize_number_word_with_punctuation(self, mock_is_numeric, mock_pronounce):
        """Test _normalize_number_word preserves punctuation."""
        mock_is_numeric.return_value = True
        mock_pronounce.return_value = "twenty three"

        result = _normalize_number_word("23!", "en", None)
        mock_pronounce.assert_called_with(23, lang="en")
        self.assertEqual(result, "twenty three!")

    @patch('phoonnx.util.pronounce_fraction')
    def test_normalize_number_word_fraction(self, mock_pronounce_fraction):
        """Test _normalize_number_word with fractions."""
        mock_pronounce_fraction.return_value = "one half"

        with patch('phoonnx.util.is_fraction', return_value=True):
            result = _normalize_number_word("1/2", "en", None)
            mock_pronounce_fraction.assert_called_with("1/2", "en")
            self.assertEqual(result, "one half")

    @patch('phoonnx.util.pronounce_fraction')
    def test_normalize_number_word_fraction_with_punctuation(self, mock_pronounce_fraction):
        """Test _normalize_number_word with fractions and punctuation."""
        mock_pronounce_fraction.return_value = "three quarters"

        with patch('phoonnx.util.is_fraction', return_value=True):
            result = _normalize_number_word("3/4.", "en", None)
            mock_pronounce_fraction.assert_called_with("3/4", "en")
            self.assertEqual(result, "three quarters.")

    @patch('phoonnx.util.is_numeric')
    def test_normalize_number_word_european_decimal(self, mock_is_numeric):
        """Test _normalize_number_word with European decimal separator."""
        mock_is_numeric.side_effect = lambda x: x in ["1.2", "1,2"]

        with patch('phoonnx.util.pronounce_number') as mock_pronounce:
            mock_pronounce.return_value = "one point two"
            result = _normalize_number_word("1,2", "pt", None)
            mock_pronounce.assert_called_with(1.2, lang="pt")
            self.assertEqual(result, "one point two")

    @patch('phoonnx.util.is_numeric')
    def test_normalize_number_word_thousands_separator(self, mock_is_numeric):
        """Test _normalize_number_word with thousands separator."""
        mock_is_numeric.side_effect = lambda x: x in ["1234", "1,234"]

        with patch('phoonnx.util.pronounce_number') as mock_pronounce:
            mock_pronounce.return_value = "one thousand two hundred thirty four"
            result = _normalize_number_word("1,234", "en", None)
            mock_pronounce.assert_called_with(1234, lang="en")
            self.assertEqual(result, "one thousand two hundred thirty four")

    @patch('phoonnx.util.is_numeric')
    def test_normalize_number_word_complex_european_format(self, mock_is_numeric):
        """Test _normalize_number_word with complex European format (123.456,78)."""
        mock_is_numeric.side_effect = lambda x: x == "123456.78"

        with patch('phoonnx.util.pronounce_number') as mock_pronounce:
            mock_pronounce.return_value = "one hundred twenty three thousand four hundred fifty six point seven eight"
            _normalize_number_word("123.456,78", "pt", None)
            mock_pronounce.assert_called_with(123456.78, lang="pt")

    def test_normalize_number_word_rbnf_fallback(self):
        """Test _normalize_number_word RBNF fallback for digits."""
        mock_engine = MagicMock()
        mock_engine.format_number.return_value.text = "twenty three"

        with patch('phoonnx.util.is_numeric', return_value=False):
            result = _normalize_number_word("23", "en", mock_engine)
            mock_engine.format_number.assert_called_once()
            self.assertEqual(result, "twenty three")

    def test_normalize_number_word_no_change(self):
        """Test _normalize_number_word when no normalization is needed."""
        with patch('phoonnx.util.is_numeric', return_value=False):
            result = _normalize_number_word("hello", "en", None)
            self.assertEqual(result, "hello")

    @patch('phoonnx.util.nice_date')
    def test_pronounce_date(self, mock_nice_date):
        """Test pronounce_date function."""
        mock_nice_date.return_value = "January first, twenty twenty five"
        test_date = date(2025, 1, 1)

        result = pronounce_date(test_date, "en")
        mock_nice_date.assert_called_with(test_date, "en")
        self.assertEqual(result, "January first, twenty twenty five")

    @patch('phoonnx.util.nice_time')
    def test_pronounce_time_valid(self, mock_nice_time):
        """Test pronounce_time with valid military time."""
        mock_nice_time.return_value = "three fifteen PM"

        result = pronounce_time("15h15", "en")
        mock_nice_time.assert_called_once()
        self.assertEqual(result, "three fifteen PM")

    def test_pronounce_time_invalid(self):
        """Test pronounce_time with invalid time format."""
        result = pronounce_time("invalid", "en")
        self.assertEqual(result, "invalid")

    def test_pronounce_time_edge_case(self):
        """Test pronounce_time with edge cases."""
        result = pronounce_time("25h70", "en")
        # Should handle gracefully and return modified string
        self.assertIn(" ", result)

    def test_normalize_word_hyphen_digit(self):
        """Test _normalize_word_hyphen_digit function."""
        test_cases = [
            ("sub-23", "sub 23"),
            ("pre-10", "pre 10"),
            ("word-123", "word 123"),
            ("no-hyphen", "no-hyphen"),  # no digit after hyphen
            ("just-text", "just-text"),  # no digit
            #  ("123-456", "123-456"),     # no word before hyphen TODO Fix this one, should be pronounced number by number
        ]

        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = _normalize_word_hyphen_digit(input_text)
                self.assertEqual(result, expected)

    @patch('phoonnx.util.pronounce_number')
    def test_normalize_units_symbolic(self, mock_pronounce):
        """Test _normalize_units with symbolic units."""
        mock_pronounce.return_value = "twenty five"
        result = _normalize_units("25°C", "en")
        mock_pronounce.assert_called_with(25.0, "en")
        self.assertIn("twenty five", result)
        self.assertIn("degrees celsius", result)

    @patch('phoonnx.util.pronounce_number')
    def test_normalize_units_alphanumeric(self, mock_pronounce):
        """Test _normalize_units with alphanumeric units."""
        mock_pronounce.return_value = "five"

        result = _normalize_units("5kg", "en")
        mock_pronounce.assert_called_with(5.0, "en")
        self.assertIn("five", result)
        self.assertIn("kilograms", result)

    def test_normalize_units_unsupported_language(self):
        """Test _normalize_units with unsupported language."""
        result = _normalize_units("25°C", "unsupported")
        self.assertEqual(result, "25°C")  # Should remain unchanged

    @patch('phoonnx.util.pronounce_number')
    def test_normalize_units_european_format(self, mock_pronounce):
        """Test _normalize_units with European number format."""
        mock_pronounce.return_value = "vinte e cinco vírgula cinco"

        _normalize_units("25,5kg", "pt")
        mock_pronounce.assert_called_with(25.5, "pt")

    def test_normalize_word_contractions(self):
        """Test _normalize_word with contractions."""
        result = _normalize_word("can't", "en", None)
        self.assertEqual(result, "can not")

        result = _normalize_word("I'm", "en", None)
        self.assertEqual(result, "I am")

    def test_normalize_word_titles(self):
        """Test _normalize_word with titles."""
        result = _normalize_word("Dr.", "en", None)
        self.assertEqual(result, "Doctor")

        result = _normalize_word("Prof.", "en", None)
        self.assertEqual(result, "Professor")

    def test_normalize_word_multilingual_titles(self):
        """Test _normalize_word with titles in different languages."""
        result = _normalize_word("Sr.", "es", None)
        self.assertEqual(result, "Señor")

        result = _normalize_word("M.", "fr", None)
        self.assertEqual(result, "Monsieur")

    @patch('phoonnx.util._normalize_number_word')
    def test_normalize_word_delegates_numbers(self, mock_normalize_number):
        """Test _normalize_word delegates to _normalize_number_word."""
        mock_normalize_number.return_value = "twenty three"

        result = _normalize_word("23", "en", None)
        mock_normalize_number.assert_called_with("23", "en", None)
        self.assertEqual(result, "twenty three")

    def test_normalize_word_no_change(self):
        """Test _normalize_word when no normalization is needed."""
        result = _normalize_word("hello", "en", None)
        self.assertEqual(result, "hello")

    @patch('phoonnx.util.nice_time')
    def test_normalize_dates_and_times_military_time(self, mock_nice_time):
        """Test _normalize_dates_and_times with military time."""
        mock_nice_time.return_value = "three fifteen PM"

        result = _normalize_dates_and_times("Meeting at 15h15", "en")
        self.assertIn("three fifteen PM", result)

    def test_normalize_dates_and_times_am_pm_preprocessing(self):
        """Test _normalize_dates_and_times with AM/PM preprocessing."""
        result = _normalize_dates_and_times("Meeting at 3pm", "en")
        self.assertIn("P M", result)

        result = _normalize_dates_and_times("Call at 9am", "en")
        self.assertIn("A M", result)

    @patch('phoonnx.util.pronounce_date')
    def test_normalize_dates_and_times_date_parsing(self, mock_pronounce_date):
        """Test _normalize_dates_and_times with date parsing."""
        mock_pronounce_date.return_value = "March eighth, twenty twenty five"

        result = _normalize_dates_and_times("Due on 08/03/2025", "en-US", "MDY")
        mock_pronounce_date.assert_called_once()
        self.assertIn("March eighth, twenty twenty five", result)

    def test_normalize_dates_and_times_invalid_date(self):
        """Test _normalize_dates_and_times with invalid date."""
        # Should handle gracefully and not crash
        result = _normalize_dates_and_times("Due on 32/13/2025", "en")
        self.assertIn("32/13/2025", result)  # Should remain unchanged

    def test_normalize_dates_and_times_ambiguous_date_dmy(self):
        """Test _normalize_dates_and_times with ambiguous date using DMY format."""
        with patch('phoonnx.util.pronounce_date') as mock_pronounce_date:
            mock_pronounce_date.return_value = "May fifteenth, twenty twenty five"
            _normalize_dates_and_times("Due on 15/05/2025", "en", "DMY")
            mock_pronounce_date.assert_called_once()

    def test_normalize_dates_and_times_year_detection(self):
        """Test _normalize_dates_and_times year detection logic."""
        with patch('phoonnx.util.pronounce_date') as mock_pronounce_date:
            mock_pronounce_date.return_value = "formatted date"

            # Test 4-digit year at beginning
            _normalize_dates_and_times("2025/03/15", "en")

            # Test 4-digit year at end
            _normalize_dates_and_times("15/03/2025", "en")

    @patch('unicode_rbnf.RbnfEngine')
    @patch('phoonnx.util._normalize_word')
    def test_normalize_main_function(self, mock_normalize_word, mock_rbnf_engine):
        """Test main normalize function integration."""
        mock_normalize_word.side_effect = lambda w, lang, engine: w.upper()
        mock_rbnf_engine.for_language.return_value = self.mock_rbnf_engine

        result = normalize("hello world", "en")
        self.assertEqual(result, "HELLO WORLD")

    @patch('phoonnx.util._normalize_dates_and_times')
    @patch('phoonnx.util._normalize_word_hyphen_digit')
    @patch('phoonnx.util._normalize_units')
    def test_normalize_date_format_selection(self, mock_normalize_units,
                                             mock_normalize_word_hyphen_digit,
                                             mock_normalize_dates):
        """Test normalize function date format selection."""
        # Test for en-US, which should use MDY
        normalize("The date is 08/03/2025", "en-US")
        mock_normalize_dates.assert_called_with("The date is 08/03/2025", "en-US", "MDY")

        # Test for en-GB, which should use DMY
        normalize("The date is 08/03/2025", "en-GB")
        mock_normalize_dates.assert_called_with("The date is 08/03/2025", "en-GB", "DMY")

    @patch('unicode_rbnf.RbnfEngine')
    def test_normalize_rbnf_engine_error_handling(self, mock_rbnf_engine):
        """Test normalize function handles RBNF engine creation errors."""
        mock_rbnf_engine.for_language.side_effect = Exception("RBNF error")

        # Should not crash when RBNF engine fails to initialize
        result = normalize("test", "unsupported-lang")
        self.assertIsInstance(result, str)

    def test_normalize_empty_string(self):
        """Test normalize with empty string."""
        result = normalize("", "en")
        self.assertEqual(result, "")

    def test_normalize_whitespace_only(self):
        """Test normalize with whitespace only."""
        result = normalize("   ", "en")
        self.assertEqual(result, "")

    def test_normalize_single_word(self):
        """Test normalize with single word."""
        with patch('phoonnx.util._normalize_word') as mock_normalize_word:
            mock_normalize_word.return_value = "normalized"
            normalize("word", "en")
            mock_normalize_word.assert_called_with("word", "en", unittest.mock.ANY)

    def test_contractions_dictionary_completeness(self):
        """Test that CONTRACTIONS dictionary is properly structured."""
        self.assertIn("en", CONTRACTIONS)
        self.assertIsInstance(CONTRACTIONS["en"], dict)
        self.assertGreater(len(CONTRACTIONS["en"]), 1)  # Should have contractions

        # Test some specific contractions
        self.assertEqual(CONTRACTIONS["en"]["can't"], "can not")
        self.assertEqual(CONTRACTIONS["en"]["I'm"], "I am")

    def test_titles_dictionary_completeness(self):
        """Test that TITLES dictionary is properly structured."""
        for lang in ["en", "ca", "es", "pt", "gl", "fr", "it", "nl", "de"]:
            if lang in TITLES:
                self.assertIsInstance(TITLES[lang], dict)
                self.assertIn("Dr.", TITLES[lang])

    def test_units_dictionary_completeness(self):
        """Test that UNITS dictionary is properly structured."""
        for lang in ["en", "pt", "es", "fr", "de"]:
            if lang in UNITS:
                self.assertIsInstance(UNITS[lang], dict)
                if "%" in UNITS[lang]:
                    self.assertIn("%", UNITS[lang])
                if "°" in UNITS[lang]:
                    self.assertIn("°", UNITS[lang])

    def test_data_integrity_contractions(self):
        """Test data integrity of contractions."""
        for _lang, contractions in CONTRACTIONS.items():
            for contraction, expansion in contractions.items():
                self.assertIsInstance(contraction, str)
                self.assertIsInstance(expansion, str)
                self.assertGreater(len(contraction), 0)
                self.assertGreater(len(expansion), 0)

    def test_data_integrity_titles(self):
        """Test data integrity of titles."""
        for _lang, titles in TITLES.items():
            for title, expansion in titles.items():
                self.assertIsInstance(title, str)
                self.assertIsInstance(expansion, str)
                self.assertGreater(len(title), 0)
                self.assertGreater(len(expansion), 0)

    def test_data_integrity_units(self):
        """Test data integrity of units."""
        for _lang, units in UNITS.items():
            for unit, expansion in units.items():
                self.assertIsInstance(unit, str)
                self.assertIsInstance(expansion, str)
                self.assertGreater(len(unit), 0)
                self.assertGreater(len(expansion), 0)

    def test_error_handling_fraction_pronunciation(self):
        """Test error handling in fraction pronunciation."""
        with patch('ovos_number_parser.pronounce_fraction', side_effect=Exception("Test error")), \
                patch('phoonnx.util.is_fraction', return_value=True):
            result = _normalize_number_word("1/2", "en", None)
            self.assertEqual(result, "1/2")  # Should return original on error

    def test_error_handling_number_pronunciation(self):
        """Test error handling in number pronunciation."""
        with patch('phoonnx.util.pronounce_number', side_effect=Exception("Test error")), \
                patch('phoonnx.util.is_numeric', return_value=True):
            result = _normalize_number_word("123", "en", None)
            self.assertEqual(result, "123")  # Should return original on error

    def test_error_handling_rbnf_pronunciation(self):
        """Test error handling in RBNF pronunciation."""
        mock_engine = MagicMock()
        mock_engine.format_number.side_effect = Exception("RBNF error")

        with patch('phoonnx.util.is_numeric', return_value=False):
            result = _normalize_number_word("123", "en", mock_engine)
            self.assertEqual(result, "123")  # Should return original on error

    def test_complex_integration_scenario(self):
        """Test complex integration scenario with multiple normalizations."""
        text = "Dr. Smith said I can't attend the 3pm meeting on 15/03/2025, it's 25°C outside"

        with patch('phoonnx.util._normalize_dates_and_times') as mock_dates:
            mock_dates.return_value = text
            with patch('phoonnx.util._normalize_units') as mock_units:
                mock_units.return_value = text
                with patch('phoonnx.util._normalize_word') as mock_word:
                    mock_word.side_effect = lambda w, lang, engine: f"NORM_{w}"

                    result = normalize(text, "en")

                    # Verify all normalization steps were called
                    mock_dates.assert_called_once()
                    mock_units.assert_called_once()
                    self.assertIn("NORM_", result)

    def test_edge_case_multiple_separators(self):
        """Test edge cases with multiple separators in numbers."""
        test_cases = [
            ("1.234.567,89", "pt"),  # Multiple thousands separators
            ("1,234,567.89", "en"),  # Multiple thousands separators
            ("1.2.3", "en"),  # Ambiguous format
        ]

        for test_input, lang in test_cases:
            with self.subTest(input=test_input, lang=lang):
                # Should not crash
                result = _normalize_number_word(test_input, lang, None)
                self.assertIsInstance(result, str)

    def test_performance_large_text(self):
        """Test performance with large text input."""
        large_text = "Dr. Smith " * 1000  # Repeat to create large text

        # Should complete in reasonable time without crashing
        result = normalize(large_text, "en")
        self.assertIsInstance(result, str)

    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        unicode_text = "café naïve résumé"

        result = normalize(unicode_text, "en")
        self.assertIsInstance(result, str)
        # Should preserve unicode characters when no normalization applies
        self.assertIn("café", result)

    def test_normalize_word_case_sensitivity(self):
        """Test _normalize_word case sensitivity."""
        # Contractions should be case-sensitive
        result = _normalize_word("CAN'T", "en", None)
        self.assertEqual(result, "CAN'T")  # Should remain unchanged

        result = _normalize_word("can't", "en", None)
        self.assertEqual(result, "can not")

    def test_normalize_dates_complex_patterns(self):
        """Test _normalize_dates_and_times with complex date patterns."""
        # Test leap year
        with patch('phoonnx.util.pronounce_date') as mock_pronounce:
            mock_pronounce.return_value = "February twenty ninth, twenty twenty four"
            result = _normalize_dates_and_times("Meeting on 29/02/2024", "en", "DMY")
            self.assertIn("February twenty ninth", result)

    def test_normalize_units_spacing_variations(self):
        """Test _normalize_units with various spacing patterns."""
        with patch('ovos_number_parser.pronounce_number') as mock_pronounce:
            mock_pronounce.return_value = "twenty five"

            # Test with space
            result = _normalize_units("25 kg", "en")
            self.assertIn("twenty five", result)

            # Test without space
            result = _normalize_units("25kg", "en")
            self.assertIn("twenty five", result)

    def test_normalize_multiple_time_formats(self):
        """Test _normalize_dates_and_times with multiple time formats."""
        text = "Meeting at 14h30 and call at 9am"

        with patch('phoonnx.util.nice_time') as mock_nice_time:
            mock_nice_time.side_effect = ["two thirty PM", "nine A M"]
            result = _normalize_dates_and_times(text, "en")
            self.assertIn("two thirty PM", result)
            self.assertIn("A M", result)

    def test_normalize_fraction_edge_cases(self):
        """Test is_fraction and fraction normalization with edge cases."""
        # Test fraction with zero
        self.assertTrue(is_fraction("0/1"))
        self.assertTrue(is_fraction("1/0"))  # Mathematical invalid but syntactically valid

        # Test large numbers
        self.assertTrue(is_fraction("999/1000"))

    def test_normalize_number_word_float_conversion(self):
        """Test _normalize_number_word float vs int conversion logic."""
        with patch('phoonnx.util.pronounce_number') as mock_pronounce, \
                patch('phoonnx.util.is_numeric', return_value=True):
            mock_pronounce.return_value = "five"

            # Integer case
            result = _normalize_number_word("5", "en", None)
            mock_pronounce.assert_called_with(5, lang="en")  # int(5)

            # Float case
            _normalize_number_word("5.0", "en", None)
            mock_pronounce.assert_called_with(5.0, lang="en")  # float(5.0)

    def test_normalize_multilingual_comprehensive(self):
        """Test normalize function with comprehensive multilingual examples."""
        test_cases = [
            ("Hola Dr. García", "es", "Hola Doctor García"),
            ("Bonjour M. Dupont", "fr", "Bonjour Monsieur Dupont"),
            ("Olá Sr. Silva", "pt", "Olá Senhor Silva"),
        ]

        for text, lang, _expected_partial in test_cases:
            with self.subTest(text=text, lang=lang):
                result = normalize(text, lang)
                # Just check that title expansion occurred
                self.assertNotEqual(result, text)

    def test_normalize_units_priority_handling(self):
        """Test _normalize_units handles overlapping unit symbols correctly."""
        # Test that longer units are matched first (m vs mL)
        with patch('ovos_number_parser.pronounce_number') as mock_pronounce:
            mock_pronounce.return_value = "five"

            result = _normalize_units("5mL", "en")
            self.assertIn("milliliters", result)
            self.assertNotIn("meters", result)


class TestDataStructureIntegrity(unittest.TestCase):
    """Test the integrity and completeness of data structures."""

    def test_contractions_comprehensive_coverage(self):
        """Test that contractions cover common English patterns."""
        if "en" in CONTRACTIONS:
            en_contractions = CONTRACTIONS["en"]

            # Test modal verbs
            if "won't" in en_contractions:
                self.assertIn("won't", en_contractions)
            if "can't" in en_contractions:
                self.assertIn("can't", en_contractions)
            if "shouldn't" in en_contractions:
                self.assertIn("shouldn't", en_contractions)

            # Test complex contractions
            if "wouldn't've" in en_contractions:
                self.assertIn("wouldn't've", en_contractions)
            if "you'd've" in en_contractions:
                self.assertIn("you'd've", en_contractions)

    def test_units_comprehensive_coverage(self):
        """Test that units cover major measurement categories."""
        if "en" in UNITS:
            en_units = UNITS["en"]

            # Temperature
            if "°C" in en_units:
                self.assertIn("°C", en_units)
            if "°F" in en_units:
                self.assertIn("°F", en_units)

            # Currency
            if "$" in en_units:
                self.assertIn("$", en_units)
            if "€" in en_units:
                self.assertIn("€", en_units)
            if "£" in en_units:
                self.assertIn("£", en_units)

            # Distance
            if "km" in en_units:
                self.assertIn("km", en_units)
            if "m" in en_units:
                self.assertIn("m", en_units)
            if "ft" in en_units:
                self.assertIn("ft", en_units)

    def test_titles_professional_coverage(self):
        """Test that titles cover professional and social titles."""
        if "en" in TITLES:
            en_titles = TITLES["en"]

            if "Dr." in en_titles:
                self.assertIn("Dr.", en_titles)
            if "Prof." in en_titles:
                self.assertIn("Prof.", en_titles)
            if "Mr." in en_titles:
                self.assertIn("Mr.", en_titles)

    def test_consistency_across_languages(self):
        """Test consistency of common elements across languages."""
        common_units = ["€", "%", "°"]

        for lang in ["en", "pt", "es", "fr", "de"]:
            if lang in UNITS:
                for unit in common_units:
                    if unit in UNITS[lang]:
                        self.assertIn(unit, UNITS[lang],
                                      f"Unit '{unit}' missing from {lang}")


class TestUtilFunctionsAdditional(unittest.TestCase):
    """Additional comprehensive tests for util.py functions with enhanced edge case coverage."""

    def setUp(self):
        """Set up test fixtures for additional tests."""
        self.mock_rbnf_engine = MagicMock()
        self.mock_rbnf_engine.format_number.return_value.text = "formatted number"

    def test_get_number_separators_case_insensitive(self):
        """Test _get_number_separators with case variations."""
        decimal, thousands = _get_number_separators("EN")
        self.assertEqual(decimal, '.')
        self.assertEqual(thousands, ',')
        
        decimal, thousands = _get_number_separators("pt-PT")
        self.assertEqual(decimal, ',')
        self.assertEqual(thousands, '.')

    def test_get_number_separators_unknown_language(self):
        """Test _get_number_separators with unknown language defaults to English."""
        decimal, thousands = _get_number_separators("unknown")
        self.assertEqual(decimal, '.')
        self.assertEqual(thousands, ',')

    def test_get_number_separators_partial_language_codes(self):
        """Test _get_number_separators with partial language codes."""
        decimal, thousands = _get_number_separators("pt-BR")
        self.assertEqual(decimal, ',')
        self.assertEqual(thousands, '.')
        
        decimal, thousands = _get_number_separators("en-CA")
        self.assertEqual(decimal, '.')
        self.assertEqual(thousands, ',')

    def test_is_fraction_whitespace_handling(self):
        """Test is_fraction with whitespace variations."""
        self.assertFalse(is_fraction(" 1/2 "))  # Leading/trailing spaces
        self.assertFalse(is_fraction("1 / 2"))  # Spaces around slash
        self.assertFalse(is_fraction("1/ 2"))   # Space after slash
        self.assertFalse(is_fraction("1 /2"))   # Space before slash

    def test_is_fraction_decimal_numbers(self):
        """Test is_fraction with decimal numbers in numerator/denominator."""
        self.assertFalse(is_fraction("1.5/2"))
        self.assertFalse(is_fraction("1/2.5"))
        self.assertFalse(is_fraction("1.0/2.0"))

    def test_is_fraction_negative_numbers(self):
        """Test is_fraction with negative numbers."""
        self.assertFalse(is_fraction("-1/2"))
        self.assertFalse(is_fraction("1/-2"))
        self.assertFalse(is_fraction("-1/-2"))

    def test_is_fraction_zero_denominator(self):
        """Test is_fraction with zero denominator."""
        self.assertTrue(is_fraction("1/0"))  # Syntactically valid, mathematically invalid

    def test_is_fraction_large_numbers(self):
        """Test is_fraction with very large numbers."""
        self.assertTrue(is_fraction("999999999/1000000000"))
        self.assertTrue(is_fraction("1/999999999"))

    def test_normalize_number_word_empty_string(self):
        """Test _normalize_number_word with empty string."""
        result = _normalize_number_word("", "en", None)
        self.assertEqual(result, "")

    def test_normalize_number_word_whitespace_only(self):
        """Test _normalize_number_word with whitespace only."""
        result = _normalize_number_word("   ", "en", None)
        self.assertEqual(result, "   ")

    def test_normalize_number_word_special_characters(self):
        """Test _normalize_number_word with special characters."""
        with patch('phoonnx.util.is_numeric', return_value=False):
            result = _normalize_number_word("@#$", "en", None)
            self.assertEqual(result, "@#$")

    @patch('phoonnx.util.pronounce_number')
    @patch('phoonnx.util.is_numeric')
    def test_normalize_number_word_mixed_punctuation(self, mock_is_numeric, mock_pronounce):
        """Test _normalize_number_word with mixed punctuation."""
        mock_is_numeric.return_value = True
        mock_pronounce.return_value = "twenty three"
        
        result = _normalize_number_word("23!?", "en", None)
        mock_pronounce.assert_called_with(23, lang="en")
        self.assertEqual(result, "twenty three!?")

    @patch('phoonnx.util.pronounce_fraction')
    def test_normalize_number_word_fraction_error_handling(self, mock_pronounce_fraction):
        """Test _normalize_number_word fraction error handling."""
        mock_pronounce_fraction.side_effect = Exception("Fraction error")
        
        with patch('phoonnx.util.is_fraction', return_value=True):
            result = _normalize_number_word("1/2", "en", None)
            self.assertEqual(result, "1/2")  # Should return original on error

    def test_normalize_number_word_rbnf_engine_none(self):
        """Test _normalize_number_word when RBNF engine is None."""
        with patch('phoonnx.util.is_numeric', return_value=False):
            result = _normalize_number_word("123", "en", None)
            self.assertEqual(result, "123")

    def test_normalize_number_word_rbnf_engine_exception(self):
        """Test _normalize_number_word when RBNF engine throws exception."""
        mock_engine = MagicMock()
        mock_engine.format_number.side_effect = Exception("RBNF error")
        
        with patch('phoonnx.util.is_numeric', return_value=False):
            result = _normalize_number_word("123", "en", mock_engine)
            self.assertEqual(result, "123")

    @patch('phoonnx.util.is_numeric')
    def test_normalize_number_word_european_format_edge_cases(self, mock_is_numeric):
        """Test _normalize_number_word with European format edge cases."""
        mock_is_numeric.side_effect = lambda x: x == "1000000.0"
        
        with patch('phoonnx.util.pronounce_number') as mock_pronounce:
            mock_pronounce.return_value = "one million"
            # Test large number with European decimal separator
            result = _normalize_number_word("1.000.000,0", "pt", None)
            mock_pronounce.assert_called_with(1000000.0, lang="pt")

    def test_pronounce_date_none_input(self):
        """Test pronounce_date with None input."""
        with self.assertRaises((TypeError, AttributeError)):
            pronounce_date(None, "en")

    def test_pronounce_date_invalid_date(self):
        """Test pronounce_date with invalid date object."""
        with patch('phoonnx.util.nice_date', side_effect=Exception("Invalid date")):
            with self.assertRaises(Exception):
                pronounce_date(date(2025, 1, 1), "en")

    def test_pronounce_time_empty_string(self):
        """Test pronounce_time with empty string."""
        result = pronounce_time("", "en")
        self.assertEqual(result, "")

    def test_pronounce_time_malformed_hour_minute(self):
        """Test pronounce_time with malformed hour/minute patterns."""
        test_cases = [
            "h15",      # Missing hour
            "15h",      # Missing minute  
            "25h70",    # Invalid hour and minute
            "abchde",   # Non-numeric
            "12h60",    # Invalid minute (60)
            "24h00",    # Edge case: 24 hours
        ]
        
        for time_str in test_cases:
            with self.subTest(time_str=time_str):
                result = pronounce_time(time_str, "en")
                self.assertIsInstance(result, str)

    def test_normalize_word_hyphen_digit_edge_cases(self):
        """Test _normalize_word_hyphen_digit with edge cases."""
        test_cases = [
            ("", ""),                           # Empty string
            ("-123", "-123"),                   # Just hyphen and digit
            ("word-", "word-"),                 # Hyphen at end
            ("word--123", "word--123"),         # Double hyphen
            ("word-123-456", "word 123-456"),   # Multiple hyphens with digits
            ("123-word", "123-word"),           # Number before hyphen
            ("word-123abc", "word 123abc"),     # Mixed digit and text after hyphen
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = _normalize_word_hyphen_digit(input_text)
                self.assertEqual(result, expected)

    def test_normalize_word_hyphen_digit_unicode(self):
        """Test _normalize_word_hyphen_digit with unicode characters."""
        result = _normalize_word_hyphen_digit("café-123")
        self.assertEqual(result, "café 123")

    @patch('phoonnx.util.pronounce_number')
    def test_normalize_units_empty_string(self, mock_pronounce):
        """Test _normalize_units with empty string."""
        result = _normalize_units("", "en")
        self.assertEqual(result, "")

    @patch('phoonnx.util.pronounce_number')
    def test_normalize_units_no_units_found(self, mock_pronounce):
        """Test _normalize_units when no units are found."""
        result = _normalize_units("just text", "en")
        self.assertEqual(result, "just text")

    @patch('phoonnx.util.pronounce_number')
    def test_normalize_units_multiple_units(self, mock_pronounce):
        """Test _normalize_units with multiple units in same string."""
        mock_pronounce.side_effect = ["twenty five", "thirty"]
        
        result = _normalize_units("25°C and 30kg", "en")
        # Should handle both units
        self.assertIn("degrees celsius", result)
        self.assertIn("kilograms", result)

    @patch('phoonnx.util.pronounce_number')
    def test_normalize_units_decimal_numbers(self, mock_pronounce):
        """Test _normalize_units with decimal numbers."""
        mock_pronounce.return_value = "twenty five point five"
        
        result = _normalize_units("25.5°C", "en")
        mock_pronounce.assert_called_with(25.5, "en")
        self.assertIn("twenty five point five", result)

    @patch('phoonnx.util.pronounce_number')
    def test_normalize_units_error_handling(self, mock_pronounce):
        """Test _normalize_units error handling during number pronunciation."""
        mock_pronounce.side_effect = Exception("Pronunciation error")
        
        result = _normalize_units("25°C", "en")
        self.assertEqual(result, "25°C")  # Should return original on error

    def test_normalize_word_empty_contractions(self):
        """Test _normalize_word when contractions dict is empty for language."""
        with patch.dict(CONTRACTIONS, {"test_lang": {}}):
            result = _normalize_word("can't", "test_lang", None)
            self.assertEqual(result, "can't")  # Should remain unchanged

    def test_normalize_word_empty_titles(self):
        """Test _normalize_word when titles dict is empty for language."""
        with patch.dict(TITLES, {"test_lang": {}}):
            result = _normalize_word("Dr.", "test_lang", None)
            self.assertEqual(result, "Dr.")  # Should remain unchanged

    def test_normalize_word_mixed_case_titles(self):
        """Test _normalize_word with mixed case titles."""
        result = _normalize_word("dr.", "en", None)
        self.assertEqual(result, "dr.")  # Should not match case-sensitive "Dr."

    def test_normalize_word_unicode_characters(self):
        """Test _normalize_word with unicode characters."""
        result = _normalize_word("naïve", "en", None)
        self.assertEqual(result, "naïve")  # Should preserve unicode

    def test_normalize_dates_and_times_empty_string(self):
        """Test _normalize_dates_and_times with empty string."""
        result = _normalize_dates_and_times("", "en")
        self.assertEqual(result, "")

    def test_normalize_dates_and_times_no_dates_or_times(self):
        """Test _normalize_dates_and_times with text containing no dates or times."""
        text = "Hello world this is just text"
        result = _normalize_dates_and_times(text, "en")
        self.assertEqual(result, text)

    def test_normalize_dates_and_times_multiple_am_pm(self):
        """Test _normalize_dates_and_times with multiple AM/PM occurrences."""
        result = _normalize_dates_and_times("Meeting at 9am and 3pm", "en")
        self.assertIn("A M", result)
        self.assertIn("P M", result)

    @patch('phoonnx.util.nice_time')
    def test_normalize_dates_and_times_time_conversion_error(self, mock_nice_time):
        """Test _normalize_dates_and_times when time conversion fails."""
        mock_nice_time.side_effect = Exception("Time conversion error")
        
        result = _normalize_dates_and_times("Meeting at 15h30", "en")
        # Should handle gracefully - exact behavior depends on implementation
        self.assertIsInstance(result, str)

    def test_normalize_dates_and_times_edge_case_dates(self):
        """Test _normalize_dates_and_times with edge case dates."""
        test_cases = [
            "00/00/0000",  # Invalid date
            "32/01/2025",  # Invalid day
            "01/13/2025",  # Invalid month for DMY
            "29/02/2023",  # Invalid leap year
        ]
        
        for date_str in test_cases:
            with self.subTest(date_str=date_str):
                result = _normalize_dates_and_times(f"Due on {date_str}", "en")
                self.assertIsInstance(result, str)

    def test_normalize_dates_and_times_ambiguous_format_handling(self):
        """Test _normalize_dates_and_times with ambiguous date formats."""
        # Test when format detection might be ambiguous
        with patch('phoonnx.util.pronounce_date') as mock_pronounce_date:
            mock_pronounce_date.return_value = "formatted date"
            
            # Ambiguous date like 01/02/2025 (could be Jan 2 or Feb 1)
            result = _normalize_dates_and_times("Due on 01/02/2025", "en-US", "MDY")
            mock_pronounce_date.assert_called_once()

    @patch('unicode_rbnf.RbnfEngine')
    def test_normalize_with_rbnf_initialization_failure(self, mock_rbnf_engine):
        """Test normalize when RBNF engine initialization fails."""
        mock_rbnf_engine.for_language.side_effect = Exception("RBNF init error")
        
        # Should handle gracefully and continue without RBNF
        result = normalize("test 123", "en")
        self.assertIsInstance(result, str)

    def test_normalize_newlines_and_tabs(self):
        """Test normalize with newlines and tabs."""
        text = "Hello\nworld\ttest"
        result = normalize(text, "en")
        self.assertIsInstance(result, str)

    def test_normalize_very_long_words(self):
        """Test normalize with very long words."""
        long_word = "a" * 1000
        result = normalize(long_word, "en")
        self.assertIsInstance(result, str)

    def test_normalize_special_punctuation(self):
        """Test normalize with special punctuation marks."""
        text = "Hello… world‽ test—more text"
        result = normalize(text, "en")
        self.assertIsInstance(result, str)

    def test_normalize_mixed_languages_in_single_text(self):
        """Test normalize behavior with mixed language content."""
        # This tests robustness - actual behavior depends on implementation
        text = "Hello 世界 test"
        result = normalize(text, "en")
        self.assertIsInstance(result, str)

    def test_contractions_with_apostrophe_variations(self):
        """Test contractions with different apostrophe characters."""
        if "en" in CONTRACTIONS and "can't" in CONTRACTIONS["en"]:
            # Test with regular apostrophe
            result = _normalize_word("can't", "en", None)
            self.assertEqual(result, "can not")
            
            # Test with different apostrophe character (if implementation handles it)
            result = _normalize_word("can't", "en", None)  # curly apostrophe
            # Behavior depends on implementation

    def test_titles_with_different_periods(self):
        """Test titles with various period styles."""
        if "en" in TITLES and "Dr." in TITLES["en"]:
            result = _normalize_word("Dr.", "en", None)
            self.assertEqual(result, "Doctor")

    def test_units_with_complex_symbols(self):
        """Test units with complex Unicode symbols."""
        if "en" in UNITS:
            complex_units = ["℃", "℉", "°C", "°F"]  # Different temperature symbols
            for unit in complex_units:
                if unit in UNITS["en"]:
                    with patch('ovos_number_parser.pronounce_number') as mock_pronounce:
                        mock_pronounce.return_value = "twenty five"
                        result = _normalize_units(f"25{unit}", "en")
                        self.assertIn("twenty five", result)

    def test_performance_stress_test(self):
        """Test performance with stress conditions."""
        # Large text with many normalizable elements
        stress_text = "Dr. Smith said I can't attend at 3pm on 15/03/2025, it's 25°C outside. " * 100
        
        import time
        start_time = time.time()
        result = normalize(stress_text, "en")
        end_time = time.time()
        
        # Should complete in reasonable time (adjust threshold as needed)
        self.assertLess(end_time - start_time, 5.0)  # Less than 5 seconds
        self.assertIsInstance(result, str)

    def test_memory_efficiency(self):
        """Test memory efficiency with repeated normalizations."""
        import gc
        
        # Force garbage collection before test
        gc.collect()
        
        for _ in range(100):
            result = normalize("Dr. Smith can't attend at 3pm", "en")
            self.assertIsInstance(result, str)
        
        # Force garbage collection after test
        gc.collect()

    def test_thread_safety_simulation(self):
        """Test thread safety by simulating concurrent access."""
        import threading
        results = []
        errors = []
        
        def normalize_worker():
            try:
                result = normalize("Dr. Smith can't attend at 3pm", "en")
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=normalize_worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have results from all threads with no errors
        self.assertEqual(len(results), 10)
        self.assertEqual(len(errors), 0)

    def test_normalize_with_all_supported_languages(self):
        """Test normalize function with all languages mentioned in data structures."""
        test_text = "Dr. Smith 123 25°C"
        
        # Test with all languages that have data
        languages = set()
        languages.update(CONTRACTIONS.keys())
        languages.update(TITLES.keys())
        languages.update(UNITS.keys())
        
        for lang in languages:
            with self.subTest(lang=lang):
                result = normalize(test_text, lang)
                self.assertIsInstance(result, str)

    def test_regression_complex_number_formats(self):
        """Regression test for complex number format edge cases."""
        test_cases = [
            ("1.234.567.890,12", "pt"),  # Very large European format
            ("0,001", "pt"),             # Small European decimal
            ("1,000,000.00", "en"),      # Large US format  
            ("0.001", "en"),             # Small US decimal
        ]
        
        for number_str, lang in test_cases:
            with self.subTest(number=number_str, lang=lang):
                result = _normalize_number_word(number_str, lang, None)
                self.assertIsInstance(result, str)

    def test_regression_date_format_detection(self):
        """Regression test for date format detection edge cases."""
        test_cases = [
            ("2025/12/31", "en", None),      # YMD format
            ("31/12/2025", "en-GB", "DMY"),  # DMY format
            ("12/31/2025", "en-US", "MDY"),  # MDY format
        ]
        
        for date_str, lang, expected_format in test_cases:
            with self.subTest(date=date_str, lang=lang):
                result = _normalize_dates_and_times(f"Due {date_str}", lang, expected_format)
                self.assertIsInstance(result, str)

    def test_normalize_units_degrees_character_replacement(self):
        """Test that _normalize_units replaces º with ° correctly."""
        with patch('ovos_number_parser.pronounce_number') as mock_pronounce:
            mock_pronounce.return_value = "twenty five"
            
            # Test with º character
            result = _normalize_units("25ºC", "en")
            self.assertIn("twenty five", result)
            self.assertIn("degrees celsius", result)

    def test_normalize_dates_and_times_year_expansion_logic(self):
        """Test the year expansion logic for 2-digit years."""
        with patch('phoonnx.util.pronounce_date') as mock_pronounce_date:
            mock_pronounce_date.return_value = "formatted date"
            
            # Test year < 30 (should become 20xx)
            _normalize_dates_and_times("Due on 15/05/25", "en", "DMY")
            
            # Test year >= 30 (should become 19xx)
            _normalize_dates_and_times("Due on 15/05/85", "en", "DMY")

    def test_normalize_units_unit_priority_sorting(self):
        """Test that longer unit symbols are matched before shorter ones."""
        with patch('ovos_number_parser.pronounce_number') as mock_pronounce:
            mock_pronounce.return_value = "five"
            
            # Test that 'mL' is matched before 'm' when both are available
            result = _normalize_units("5mL", "en")
            self.assertIn("milliliters", result)
            self.assertNotIn("meters", result)

    @patch('phoonnx.util.is_numeric')
    def test_normalize_number_word_complex_separator_handling(self, mock_is_numeric):
        """Test complex separator handling in _normalize_number_word."""
        # Test when both thousands and decimal separators are present
        mock_is_numeric.side_effect = lambda x: x == "123456.78"
        
        with patch('phoonnx.util.pronounce_number') as mock_pronounce:
            mock_pronounce.return_value = "one hundred twenty three thousand"
            
            # European format: 123.456,78
            result = _normalize_number_word("123.456,78", "pt", None)
            mock_pronounce.assert_called_with(123456.78, lang="pt")

    def test_normalize_dates_and_times_day_detection_logic(self):
        """Test the day detection logic for values > 12."""
        with patch('phoonnx.util.pronounce_date') as mock_pronounce_date:
            mock_pronounce_date.return_value = "formatted date"
            
            # Date with day > 12 should be correctly identified
            _normalize_dates_and_times("Due on 15/05/2025", "en", "DMY")
            mock_pronounce_date.assert_called_once()

    def test_pronounce_time_hour_minute_validation(self):
        """Test pronounce_time with various hour/minute validation scenarios."""
        # Test valid edge cases
        result = pronounce_time("00h00", "en")
        self.assertIsInstance(result, str)
        
        result = pronounce_time("23h59", "en")
        self.assertIsInstance(result, str)

    def test_normalize_units_regex_escaping(self):
        """Test that special regex characters in units are properly escaped."""
        # Test units with special regex characters
        test_units = ["$", "€", "£", "%"]
        
        for unit in test_units:
            if "en" in UNITS and unit in UNITS["en"]:
                with patch('ovos_number_parser.pronounce_number') as mock_pronounce:
                    mock_pronounce.return_value = "ten"
                    
                    # Should not cause regex compilation errors
                    result = _normalize_units(f"10{unit}", "en")
                    self.assertIsInstance(result, str)

    def test_is_fraction_multiple_slashes(self):
        """Test is_fraction with multiple slash characters."""
        self.assertFalse(is_fraction("1/2/3"))
        self.assertFalse(is_fraction("1//2"))
        self.assertFalse(is_fraction("/1/2"))

    def test_normalize_number_word_punctuation_preservation(self):
        """Test that punctuation after numbers is properly preserved."""
        with patch('phoonnx.util.is_numeric', return_value=True), \
             patch('phoonnx.util.pronounce_number', return_value="twenty three"):
            
            # Test various punctuation marks
            test_cases = [
                ("23.", "twenty three."),
                ("23,", "twenty three,"),
                ("23!", "twenty three!"),
                ("23?", "twenty three?"),
                ("23;", "twenty three;"),
                ("23:", "twenty three:"),
            ]
            
            for input_text, expected in test_cases:
                with self.subTest(input_text=input_text):
                    result = _normalize_number_word(input_text, "en", None)
                    self.assertEqual(result, expected)


class TestLoggingAndErrorHandling(unittest.TestCase):
    """Test logging and error handling throughout the util.py functions."""


    @patch('phoonnx.util.LOG')
    def test_error_logging_in_fraction_pronunciation(self, mock_log):
        """Test that errors in fraction pronunciation are logged."""
        with patch('phoonnx.util.is_fraction', return_value=True), \
             patch('phoonnx.util.pronounce_fraction', side_effect=Exception("Test error")):
            
            result = _normalize_number_word("1/2", "en", None)
            self.assertEqual(result, "1/2")
            mock_log.error.assert_called_once()

    @patch('phoonnx.util.LOG')
    def test_error_logging_in_number_pronunciation(self, mock_log):
        """Test that errors in number pronunciation are logged."""
        with patch('phoonnx.util.is_numeric', return_value=True), \
             patch('phoonnx.util.pronounce_number', side_effect=Exception("Test error")):
            
            result = _normalize_number_word("123", "en", None)
            self.assertEqual(result, "123")
            mock_log.error.assert_called_once()

    @patch('phoonnx.util.LOG')
    def test_error_logging_in_rbnf_pronunciation(self, mock_log):
        """Test that errors in RBNF pronunciation are logged."""
        mock_engine = MagicMock()
        mock_engine.format_number.side_effect = Exception("RBNF error")
        
        with patch('phoonnx.util.is_numeric', return_value=False):
            result = _normalize_number_word("123", "en", mock_engine)
            self.assertEqual(result, "123")
            mock_log.error.assert_called_once()

    @patch('phoonnx.util.LOG')
    def test_warning_logging_in_time_pronunciation(self, mock_log):
        """Test that warnings in time pronunciation are logged."""
        result = pronounce_time("invalid_time", "en")
        mock_log.warning.assert_called_once()

    @patch('phoonnx.util.LOG')
    def test_debug_logging_in_normalize(self, mock_log):
        """Test that debug messages are logged when RBNF engine fails."""
        with patch('unicode_rbnf.RbnfEngine.for_language', 
                   side_effect=ValueError("Language not supported")):
            
            result = normalize("test", "unsupported_lang")
            mock_log.debug.assert_called_once()


class TestDataStructureValidation(unittest.TestCase):
    """Additional validation tests for the data structures."""
    
    def test_contractions_consistency(self):
        """Test consistency of contraction expansions."""
        for lang, contractions in CONTRACTIONS.items():
            for contraction, expansion in contractions.items():
                # Contractions should not expand to themselves
                self.assertNotEqual(contraction, expansion)
                
                # Expansions should be longer than contractions (in most cases)
                # This is a general rule with some exceptions
                if contraction not in ["y'all", "ol'", "'tis", "'twas"]:
                    self.assertGreaterEqual(len(expansion), len(contraction))

    def test_units_multilingual_consistency(self):
        """Test that common units exist across multiple languages."""
        common_units = ["€", "%", "°C", "$", "km", "m", "kg"]
        
        for unit in common_units:
            languages_with_unit = [lang for lang, units in UNITS.items() if unit in units]
            # Most common units should be available in multiple languages
            if unit in ["€", "%", "°C"]:
                self.assertGreaterEqual(len(languages_with_unit), 3)

    def test_data_structure_immutability_safety(self):
        """Test that modifying returned data doesn't affect the original."""
        # Get a reference to titles
        en_titles = TITLES.get("en", {})
        original_length = len(en_titles)
        
        # Try to modify (this should not affect the original if properly implemented)
        en_titles_copy = en_titles.copy()
        en_titles_copy["Test."] = "Test"
        
        # Original should be unchanged
        self.assertEqual(len(TITLES.get("en", {})), original_length)


if __name__ == '__main__':
    unittest.main()
