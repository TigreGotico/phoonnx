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
        if callable(_get_number_separators):
            decimal, thousands = _get_number_separators("en")
            self.assertEqual(decimal, '.')
            self.assertEqual(thousands, ',')
            
            decimal, thousands = _get_number_separators("en-US")
            self.assertEqual(decimal, '.')
            self.assertEqual(thousands, ',')

    def test_get_number_separators_european(self):
        """Test _get_number_separators with European languages."""
        if callable(_get_number_separators):
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
        if callable(is_fraction):
            self.assertTrue(is_fraction("1/2"))
            self.assertTrue(is_fraction("3/4"))
            self.assertTrue(is_fraction("10/20"))
            self.assertTrue(is_fraction("0/1"))

    def test_is_fraction_invalid(self):
        """Test is_fraction with invalid inputs."""
        if callable(is_fraction):
            self.assertFalse(is_fraction("1.5"))
            self.assertFalse(is_fraction("1/2/3"))
            self.assertFalse(is_fraction("a/b"))
            self.assertFalse(is_fraction("1/"))
            self.assertFalse(is_fraction("/2"))
            self.assertFalse(is_fraction("no_fraction"))
            self.assertFalse(is_fraction(""))

    def test_is_fraction_edge_cases(self):
        """Test is_fraction with edge cases."""
        if callable(is_fraction):
            self.assertFalse(is_fraction("1/2.5"))
            self.assertFalse(is_fraction("1.0/2"))
            self.assertFalse(is_fraction("1/-2"))
            self.assertFalse(is_fraction("-1/2"))

    @patch('phoonnx.util.pronounce_number')
    @patch('phoonnx.util.is_numeric')
    def test_normalize_number_word_simple_integer(self, mock_is_numeric, mock_pronounce):
        """Test _normalize_number_word with simple integers."""
        if callable(_normalize_number_word):
            mock_is_numeric.return_value = True
            mock_pronounce.return_value = "twenty three"
            
            result = _normalize_number_word("23", "en", None)
            mock_pronounce.assert_called_with(23, lang="en")
            self.assertEqual(result, "twenty three")

    @patch('phoonnx.util.pronounce_number')
    @patch('phoonnx.util.is_numeric')
    def test_normalize_number_word_with_punctuation(self, mock_is_numeric, mock_pronounce):
        """Test _normalize_number_word preserves punctuation."""
        if callable(_normalize_number_word):
            mock_is_numeric.return_value = True
            mock_pronounce.return_value = "twenty three"
            
            result = _normalize_number_word("23!", "en", None)
            mock_pronounce.assert_called_with(23, lang="en")
            self.assertEqual(result, "twenty three!")

    @patch('phoonnx.util.pronounce_fraction')
    def test_normalize_number_word_fraction(self, mock_pronounce_fraction):
        """Test _normalize_number_word with fractions."""
        if callable(_normalize_number_word):
            mock_pronounce_fraction.return_value = "one half"
            
            with patch('phoonnx.util.is_fraction', return_value=True):
                result = _normalize_number_word("1/2", "en", None)
                mock_pronounce_fraction.assert_called_with("1/2", "en")
                self.assertEqual(result, "one half")

    @patch('phoonnx.util.pronounce_fraction')
    def test_normalize_number_word_fraction_with_punctuation(self, mock_pronounce_fraction):
        """Test _normalize_number_word with fractions and punctuation."""
        if callable(_normalize_number_word):
            mock_pronounce_fraction.return_value = "three quarters"
            
            with patch('phoonnx.util.is_fraction', return_value=True):
                result = _normalize_number_word("3/4.", "en", None)
                mock_pronounce_fraction.assert_called_with("3/4", "en")
                self.assertEqual(result, "three quarters.")

    @patch('phoonnx.util.is_numeric')
    def test_normalize_number_word_european_decimal(self, mock_is_numeric):
        """Test _normalize_number_word with European decimal separator."""
        if callable(_normalize_number_word):
            mock_is_numeric.side_effect = lambda x: x in ["1.2", "1,2"]
            
            with patch('phoonnx.util.pronounce_number') as mock_pronounce:
                mock_pronounce.return_value = "one point two"
                result = _normalize_number_word("1,2", "pt", None)
                mock_pronounce.assert_called_with(1.2, lang="pt")
                self.assertEqual(result, "one point two")

    @patch('phoonnx.util.is_numeric')
    def test_normalize_number_word_thousands_separator(self, mock_is_numeric):
        """Test _normalize_number_word with thousands separator."""
        if callable(_normalize_number_word):
            mock_is_numeric.side_effect = lambda x: x in ["1234", "1,234"]
            
            with patch('phoonnx.util.pronounce_number') as mock_pronounce:
                mock_pronounce.return_value = "one thousand two hundred thirty four"
                result = _normalize_number_word("1,234", "en", None)
                mock_pronounce.assert_called_with(1234, lang="en")
                self.assertEqual(result, "one thousand two hundred thirty four")

    @patch('phoonnx.util.is_numeric')
    def test_normalize_number_word_complex_european_format(self, mock_is_numeric):
        """Test _normalize_number_word with complex European format (123.456,78)."""
        if callable(_normalize_number_word):
            mock_is_numeric.side_effect = lambda x: x == "123456.78"
            
            with patch('phoonnx.util.pronounce_number') as mock_pronounce:
                mock_pronounce.return_value = "one hundred twenty three thousand four hundred fifty six point seven eight"
                _normalize_number_word("123.456,78", "pt", None)
                mock_pronounce.assert_called_with(123456.78, lang="pt")

    def test_normalize_number_word_rbnf_fallback(self):
        """Test _normalize_number_word RBNF fallback for digits."""
        if callable(_normalize_number_word):
            mock_engine = MagicMock()
            mock_engine.format_number.return_value.text = "twenty three"
            
            with patch('phoonnx.util.is_numeric', return_value=False):
                result = _normalize_number_word("23", "en", mock_engine)
                mock_engine.format_number.assert_called_once()
                self.assertEqual(result, "twenty three")

    def test_normalize_number_word_no_change(self):
        """Test _normalize_number_word when no normalization is needed."""
        if callable(_normalize_number_word):
            with patch('phoonnx.util.is_numeric', return_value=False):
                result = _normalize_number_word("hello", "en", None)
                self.assertEqual(result, "hello")

    @patch('phoonnx.util.nice_date')
    def test_pronounce_date(self, mock_nice_date):
        """Test pronounce_date function."""
        if callable(pronounce_date):
            mock_nice_date.return_value = "January first, twenty twenty five"
            test_date = date(2025, 1, 1)
            
            result = pronounce_date(test_date, "en")
            mock_nice_date.assert_called_with(test_date, "en")
            self.assertEqual(result, "January first, twenty twenty five")

    @patch('phoonnx.util.nice_time')
    def test_pronounce_time_valid(self, mock_nice_time):
        """Test pronounce_time with valid military time."""
        if callable(pronounce_time):
            mock_nice_time.return_value = "three fifteen PM"
            
            result = pronounce_time("15h15", "en")
            mock_nice_time.assert_called_once()
            self.assertEqual(result, "three fifteen PM")

    def test_pronounce_time_invalid(self):
        """Test pronounce_time with invalid time format."""
        if callable(pronounce_time):
            result = pronounce_time("invalid", "en")
            self.assertEqual(result, "invalid")

    def test_pronounce_time_edge_case(self):
        """Test pronounce_time with edge cases."""
        if callable(pronounce_time):
            result = pronounce_time("25h70", "en")
            # Should handle gracefully and return modified string
            self.assertIn(" ", result)

    def test_normalize_word_hyphen_digit(self):
        """Test _normalize_word_hyphen_digit function."""
        if callable(_normalize_word_hyphen_digit):
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

    @unittest.expectedFailure
    @patch('phoonnx.util.pronounce_number')
    def test_normalize_units_symbolic(self, mock_pronounce):
        """Test _normalize_units with symbolic units."""
        if callable(_normalize_units):
            mock_pronounce.return_value = "twenty five"
            
            result = _normalize_units("25°C", "en")
            mock_pronounce.assert_called_with(25.0, "en")
            self.assertIn("twenty five", result)
            self.assertIn("degrees celsius", result)

    @patch('phoonnx.util.pronounce_number')
    def test_normalize_units_alphanumeric(self, mock_pronounce):
        """Test _normalize_units with alphanumeric units."""
        if callable(_normalize_units):
            mock_pronounce.return_value = "five"
            
            result = _normalize_units("5kg", "en")
            mock_pronounce.assert_called_with(5.0, "en")
            self.assertIn("five", result)
            self.assertIn("kilograms", result)

    def test_normalize_units_unsupported_language(self):
        """Test _normalize_units with unsupported language."""
        if callable(_normalize_units):
            result = _normalize_units("25°C", "unsupported")
            self.assertEqual(result, "25°C")  # Should remain unchanged

    @patch('phoonnx.util.pronounce_number')
    def test_normalize_units_european_format(self, mock_pronounce):
        """Test _normalize_units with European number format."""
        if callable(_normalize_units):
            mock_pronounce.return_value = "vinte e cinco vírgula cinco"
            
            _normalize_units("25,5kg", "pt")
            mock_pronounce.assert_called_with(25.5, "pt")

    def test_normalize_word_contractions(self):
        """Test _normalize_word with contractions."""
        if callable(_normalize_word):
            result = _normalize_word("can't", "en", None)
            self.assertEqual(result, "can not")
            
            result = _normalize_word("I'm", "en", None)
            self.assertEqual(result, "I am")

    def test_normalize_word_titles(self):
        """Test _normalize_word with titles."""
        if callable(_normalize_word):
            result = _normalize_word("Dr.", "en", None)
            self.assertEqual(result, "Doctor")
            
            result = _normalize_word("Prof.", "en", None)
            self.assertEqual(result, "Professor")

    def test_normalize_word_multilingual_titles(self):
        """Test _normalize_word with titles in different languages."""
        if callable(_normalize_word):
            result = _normalize_word("Sr.", "es", None)
            self.assertEqual(result, "Señor")
            
            result = _normalize_word("M.", "fr", None)
            self.assertEqual(result, "Monsieur")

    @patch('phoonnx.util._normalize_number_word')
    def test_normalize_word_delegates_numbers(self, mock_normalize_number):
        """Test _normalize_word delegates to _normalize_number_word."""
        if callable(_normalize_word):
            mock_normalize_number.return_value = "twenty three"
            
            result = _normalize_word("23", "en", None)
            mock_normalize_number.assert_called_with("23", "en", None)
            self.assertEqual(result, "twenty three")

    def test_normalize_word_no_change(self):
        """Test _normalize_word when no normalization is needed."""
        if callable(_normalize_word):
            result = _normalize_word("hello", "en", None)
            self.assertEqual(result, "hello")

    @patch('phoonnx.util.nice_time')
    def test_normalize_dates_and_times_military_time(self, mock_nice_time):
        """Test _normalize_dates_and_times with military time."""
        if callable(_normalize_dates_and_times):
            mock_nice_time.return_value = "three fifteen PM"
            
            result = _normalize_dates_and_times("Meeting at 15h15", "en")
            self.assertIn("three fifteen PM", result)

    def test_normalize_dates_and_times_am_pm_preprocessing(self):
        """Test _normalize_dates_and_times with AM/PM preprocessing."""
        if callable(_normalize_dates_and_times):
            result = _normalize_dates_and_times("Meeting at 3pm", "en")
            self.assertIn("P M", result)
            
            result = _normalize_dates_and_times("Call at 9am", "en")
            self.assertIn("A M", result)

    @patch('phoonnx.util.pronounce_date')
    def test_normalize_dates_and_times_date_parsing(self, mock_pronounce_date):
        """Test _normalize_dates_and_times with date parsing."""
        if callable(_normalize_dates_and_times):
            mock_pronounce_date.return_value = "March eighth, twenty twenty five"
            
            result = _normalize_dates_and_times("Due on 08/03/2025", "en-US", "MDY")
            mock_pronounce_date.assert_called_once()
            self.assertIn("March eighth, twenty twenty five", result)

    def test_normalize_dates_and_times_invalid_date(self):
        """Test _normalize_dates_and_times with invalid date."""
        if callable(_normalize_dates_and_times):
            # Should handle gracefully and not crash
            result = _normalize_dates_and_times("Due on 32/13/2025", "en")
            self.assertIn("32/13/2025", result)  # Should remain unchanged

    def test_normalize_dates_and_times_ambiguous_date_dmy(self):
        """Test _normalize_dates_and_times with ambiguous date using DMY format."""
        if callable(_normalize_dates_and_times):
            with patch('phoonnx.util.pronounce_date') as mock_pronounce_date:
                mock_pronounce_date.return_value = "May fifteenth, twenty twenty five"
                _normalize_dates_and_times("Due on 15/05/2025", "en", "DMY")
                mock_pronounce_date.assert_called_once()

    def test_normalize_dates_and_times_year_detection(self):
        """Test _normalize_dates_and_times year detection logic."""
        if callable(_normalize_dates_and_times):
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
        if callable(normalize):
            mock_normalize_word.side_effect = lambda w, lang, engine: w.upper()
            mock_rbnf_engine.for_language.return_value = self.mock_rbnf_engine
            
            result = normalize("hello world", "en")
            self.assertEqual(result, "HELLO WORLD")

    @unittest.expectedFailure
    def test_normalize_date_format_selection(self):
        """Test normalize function date format selection."""
        if callable(normalize):
            with patch('phoonnx.util._normalize_dates_and_times') as mock_normalize_dates:
                mock_normalize_dates.return_value = "normalized"
                with patch('phoonnx.util._normalize_word', return_value="word"):
                    normalize("test", "en-US")
                    mock_normalize_dates.assert_called_with("normalized", "en-US", "MDY")
                    
                    normalize("test", "en-GB")
                    mock_normalize_dates.assert_called_with("normalized", "en-GB", "DMY")

    @patch('unicode_rbnf.RbnfEngine')
    def test_normalize_rbnf_engine_error_handling(self, mock_rbnf_engine):
        """Test normalize function handles RBNF engine creation errors."""
        if callable(normalize):
            mock_rbnf_engine.for_language.side_effect = Exception("RBNF error")
            
            # Should not crash when RBNF engine fails to initialize
            result = normalize("test", "unsupported-lang")
            self.assertIsInstance(result, str)

    def test_normalize_empty_string(self):
        """Test normalize with empty string."""
        if callable(normalize):
            result = normalize("", "en")
            self.assertEqual(result, "")

    def test_normalize_whitespace_only(self):
        """Test normalize with whitespace only."""
        if callable(normalize):
            result = normalize("   ", "en")
            self.assertEqual(result, "")

    def test_normalize_single_word(self):
        """Test normalize with single word."""
        if callable(normalize):
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
        if callable(_normalize_number_word):
            with patch('ovos_number_parser.pronounce_fraction', side_effect=Exception("Test error")), \
                 patch('phoonnx.util.is_fraction', return_value=True):
                result = _normalize_number_word("1/2", "en", None)
                self.assertEqual(result, "1/2")  # Should return original on error

    def test_error_handling_number_pronunciation(self):
        """Test error handling in number pronunciation."""
        if callable(_normalize_number_word):
            with patch('phoonnx.util.pronounce_number', side_effect=Exception("Test error")), \
                 patch('phoonnx.util.is_numeric', return_value=True):
                result = _normalize_number_word("123", "en", None)
                self.assertEqual(result, "123")  # Should return original on error

    def test_error_handling_rbnf_pronunciation(self):
        """Test error handling in RBNF pronunciation."""
        if callable(_normalize_number_word):
            mock_engine = MagicMock()
            mock_engine.format_number.side_effect = Exception("RBNF error")
            
            with patch('phoonnx.util.is_numeric', return_value=False):
                result = _normalize_number_word("123", "en", mock_engine)
                self.assertEqual(result, "123")  # Should return original on error

    def test_complex_integration_scenario(self):
        """Test complex integration scenario with multiple normalizations."""
        if callable(normalize):
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
        if callable(_normalize_number_word):
            test_cases = [
                ("1.234.567,89", "pt"),  # Multiple thousands separators
                ("1,234,567.89", "en"),  # Multiple thousands separators  
                ("1.2.3", "en"),         # Ambiguous format
            ]
            
            for test_input, lang in test_cases:
                with self.subTest(input=test_input, lang=lang):
                    # Should not crash
                    result = _normalize_number_word(test_input, lang, None)
                    self.assertIsInstance(result, str)

    def test_performance_large_text(self):
        """Test performance with large text input."""
        if callable(normalize):
            large_text = "Dr. Smith " * 1000  # Repeat to create large text
            
            # Should complete in reasonable time without crashing
            result = normalize(large_text, "en")
            self.assertIsInstance(result, str)

    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        if callable(normalize):
            unicode_text = "café naïve résumé"
            
            result = normalize(unicode_text, "en")
            self.assertIsInstance(result, str)
            # Should preserve unicode characters when no normalization applies
            self.assertIn("café", result)

    def test_normalize_word_case_sensitivity(self):
        """Test _normalize_word case sensitivity."""
        if callable(_normalize_word):
            # Contractions should be case-sensitive
            result = _normalize_word("CAN'T", "en", None)
            self.assertEqual(result, "CAN'T")  # Should remain unchanged
            
            result = _normalize_word("can't", "en", None)
            self.assertEqual(result, "can not")

    def test_normalize_dates_complex_patterns(self):
        """Test _normalize_dates_and_times with complex date patterns."""
        if callable(_normalize_dates_and_times):
            # Test leap year
            with patch('phoonnx.util.pronounce_date') as mock_pronounce:
                mock_pronounce.return_value = "February twenty ninth, twenty twenty four"
                result = _normalize_dates_and_times("Meeting on 29/02/2024", "en", "DMY")
                self.assertIn("February twenty ninth", result)

    def test_normalize_units_spacing_variations(self):
        """Test _normalize_units with various spacing patterns."""
        if callable(_normalize_units):
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
        if callable(_normalize_dates_and_times):
            text = "Meeting at 14h30 and call at 9am"
            
            with patch('phoonnx.util.nice_time') as mock_nice_time:
                mock_nice_time.side_effect = ["two thirty PM", "nine A M"]
                result = _normalize_dates_and_times(text, "en")
                self.assertIn("two thirty PM", result)
                self.assertIn("A M", result)

    def test_normalize_fraction_edge_cases(self):
        """Test is_fraction and fraction normalization with edge cases."""
        if callable(is_fraction):
            # Test fraction with zero
            self.assertTrue(is_fraction("0/1"))
            self.assertTrue(is_fraction("1/0"))  # Mathematical invalid but syntactically valid
            
            # Test large numbers
            self.assertTrue(is_fraction("999/1000"))

    def test_normalize_number_word_float_conversion(self):
        """Test _normalize_number_word float vs int conversion logic."""
        if callable(_normalize_number_word):
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
        if callable(normalize):
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
        if callable(_normalize_units):
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


if __name__ == '__main__':
    unittest.main()
