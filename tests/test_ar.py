"""Test suite for phoonnx Arabic phonemizer (MantoqPhonemizer)."""
import unittest
from unittest.mock import patch
import sys
import os

from phoonnx.phonemizers.ar import MantoqPhonemizer
from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.config import Alphabet


class TestMantoqPhonemizer(unittest.TestCase):
    """Comprehensive unit tests for MantoqPhonemizer class.
    
    Testing Framework: unittest (Python standard library)
    Following existing project patterns from tests/test_util.py
    """

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.phonemizer_mantoq = MantoqPhonemizer()
        self.phonemizer_ipa = MantoqPhonemizer(alphabet=Alphabet.IPA)

    def tearDown(self):
        """Clean up after each test method."""
        pass

    # Constructor and Initialization Tests
    def test_init_default_alphabet(self):
        """Test initialization with default MANTOQ alphabet."""
        phonemizer = MantoqPhonemizer()
        self.assertEqual(phonemizer.alphabet, Alphabet.MANTOQ)
        self.assertIsInstance(phonemizer, BasePhonemizer)

    def test_init_ipa_alphabet(self):
        """Test initialization with IPA alphabet."""
        phonemizer = MantoqPhonemizer(alphabet=Alphabet.IPA)
        self.assertEqual(phonemizer.alphabet, Alphabet.IPA)
        self.assertIsInstance(phonemizer, BasePhonemizer)

    def test_init_unicode_alphabet(self):
        """Test initialization with UNICODE alphabet."""
        phonemizer = MantoqPhonemizer(alphabet=Alphabet.UNICODE)
        self.assertEqual(phonemizer.alphabet, Alphabet.UNICODE)
        self.assertIsInstance(phonemizer, BasePhonemizer)

    def test_init_inheritance_chain(self):
        """Test proper inheritance from BasePhonemizer."""
        self.assertIsInstance(self.phonemizer_mantoq, BasePhonemizer)
        self.assertTrue(hasattr(self.phonemizer_mantoq, 'alphabet'))
        self.assertTrue(hasattr(self.phonemizer_mantoq, 'phonemize_string'))
        # Verify super().__init__ was called properly
        self.assertIsNotNone(self.phonemizer_mantoq.alphabet)

    # Language Validation Tests (get_lang method)
    def test_get_lang_valid_arabic_codes(self):
        """Test get_lang with various valid Arabic language codes."""
        valid_codes = [
            "ar", "AR", "Ar", "aR",  # Case variations
            "ar-SA", "ar-EG", "ar-LB", "ar-JO", "ar-AE",  # Regional variants
            "ar_SA", "ar_EG", "ar_LB", "ar_JO", "ar_AE"   # Underscore variants
        ]
        
        for code in valid_codes:
            with self.subTest(code=code):
                result = MantoqPhonemizer.get_lang(code)
                self.assertEqual(result, "ar", f"Failed to validate Arabic code: {code}")

    def test_get_lang_invalid_languages(self):
        """Test get_lang rejection of non-Arabic language codes."""
        invalid_codes = [
            "en", "fr", "es", "de", "it", "ru", "zh", "ja", "ko", "hi",
            "en-US", "fr-FR", "es-ES", "de-DE", "zh-CN", "ja-JP"
        ]
        
        for code in invalid_codes:
            with self.subTest(code=code):
                with self.assertRaises(ValueError, msg=f"Should reject non-Arabic code: {code}"):
                    MantoqPhonemizer.get_lang(code)

    def test_get_lang_edge_cases(self):
        """Test get_lang with edge case inputs."""
        edge_cases = [
            ("", "empty string"),
            (" ", "whitespace only"),
            ("ar ", "trailing space"),
            (" ar", "leading space"),
            ("  ar  ", "multiple spaces"),
            ("AR-", "incomplete region code"),
            ("_ar", "leading underscore"),
            ("ar_", "trailing underscore"),
            ("arabic", "full language name"),
            ("ara", "ISO 639-2 code")
        ]
        
        for case, description in edge_cases:
            with self.subTest(case=repr(case), desc=description):
                # Most edge cases should raise ValueError
                with self.assertRaises(ValueError, msg=f"Should reject edge case: {description}"):
                    MantoqPhonemizer.get_lang(case)

    def test_get_lang_invalid_types(self):
        """Test get_lang with invalid input types."""
        invalid_inputs = [
            (None, "None value"),
            (123, "integer"),
            (12.34, "float"),
            ([], "empty list"),
            (['ar'], "list with string"),
            ({}, "empty dict"),
            ({'lang': 'ar'}, "dict with lang"),
            (object(), "generic object"),
            (True, "boolean True"),
            (False, "boolean False")
        ]
        
        for invalid_input, description in invalid_inputs:
            with self.subTest(input=invalid_input, desc=description):
                with self.assertRaises((ValueError, TypeError, AttributeError)):
                    MantoqPhonemizer.get_lang(invalid_input)

    def test_get_lang_as_class_method(self):
        """Test get_lang called as class method."""
        result = MantoqPhonemizer.get_lang("ar")
        self.assertEqual(result, "ar")

    def test_get_lang_as_instance_method(self):
        """Test get_lang called as instance method."""
        result = self.phonemizer_mantoq.get_lang("ar")
        self.assertEqual(result, "ar")

    def test_get_lang_consistency_class_vs_instance(self):
        """Test that get_lang returns consistent results when called as class vs instance method."""
        test_codes = ["ar", "AR", "ar-SA", "ar_EG"]
        
        for code in test_codes:
            with self.subTest(code=code):
                class_result = MantoqPhonemizer.get_lang(code)
                instance_result = self.phonemizer_mantoq.get_lang(code)
                self.assertEqual(class_result, instance_result)

    # Basic Phonemization Tests with MANTOQ alphabet
    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_mantoq_basic(self, mock_mantoq):
        """Test basic phonemize_string functionality with MANTOQ alphabet."""
        mock_mantoq.return_value = ("مرحبا", ['m', 'a', 'r', 'h', 'a', 'b', 'a'])
        
        result = self.phonemizer_mantoq.phonemize_string("مرحبا", "ar")
        
        mock_mantoq.assert_called_once_with("مرحبا")
        self.assertEqual(result, "marhaba")
        self.assertIsInstance(result, str)

    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_mantoq_single_character(self, mock_mantoq):
        """Test phonemization of single Arabic character."""
        mock_mantoq.return_value = ("ب", ['b'])
        
        result = self.phonemizer_mantoq.phonemize_string("ب", "ar")
        
        mock_mantoq.assert_called_once_with("ب")
        self.assertEqual(result, "b")

    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_mantoq_with_word_separators(self, mock_mantoq):
        """Test phonemize_string with word separator tokens (_+_)."""
        mock_mantoq.return_value = ("مرحبا بالعالم", ['m', 'a', 'r', '_+_', 'h', 'a', 'b', 'a'])
        
        result = self.phonemizer_mantoq.phonemize_string("مرحبا بالعالم", "ar")
        
        mock_mantoq.assert_called_once_with("مرحبا بالعالم")
        self.assertEqual(result, "mar haba")
        self.assertIn(' ', result)
        self.assertEqual(result.count(' '), 1)

    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_mantoq_multiple_separators(self, mock_mantoq):
        """Test phonemize_string with multiple word separators."""
        mock_mantoq.return_value = ("text", ['a', '_+_', 'b', '_+_', 'c', '_+_', 'd'])
        
        result = self.phonemizer_mantoq.phonemize_string("كلام كثير", "ar")
        
        self.assertEqual(result, "a b c d")
        self.assertEqual(result.count(' '), 3)

    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_mantoq_consecutive_separators(self, mock_mantoq):
        """Test handling of consecutive word separators."""
        mock_mantoq.return_value = ("text", ['word', '_+_', '_+_', 'another'])
        
        result = self.phonemizer_mantoq.phonemize_string("test", "ar")
        
        # Should result in double space
        self.assertEqual(result, "word  another")

    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_mantoq_no_separators(self, mock_mantoq):
        """Test phonemize_string with no word separators."""
        mock_mantoq.return_value = ("كتاب", ['k', 'i', 't', 'a', 'b'])
        
        result = self.phonemizer_mantoq.phonemize_string("كتاب", "ar")
        
        self.assertEqual(result, "kitab")
        self.assertNotIn(' ', result)

    # Phonemization Tests with IPA alphabet
    @patch('phoonnx.thirdparty.bw2ipa.translate')
    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_ipa_basic(self, mock_mantoq, mock_bw2ipa):
        """Test phonemize_string with IPA alphabet."""
        mock_mantoq.return_value = ("مرحبا", ['m', 'a', 'r', 'h', 'a', 'b', 'a'])
        mock_bw2ipa.return_value = "marħaba"
        
        result = self.phonemizer_ipa.phonemize_string("مرحبا", "ar")
        
        mock_mantoq.assert_called_once_with("مرحبا")
        mock_bw2ipa.assert_called_once_with("marhaba")
        self.assertEqual(result, "marħaba")

    @patch('phoonnx.thirdparty.bw2ipa.translate')
    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_ipa_with_separators(self, mock_mantoq, mock_bw2ipa):
        """Test IPA phonemization preserving word separators."""
        mock_mantoq.return_value = ("مرحبا بالعالم", ['m', 'a', 'r', '_+_', 'h', 'a', 'b', 'a'])
        mock_bw2ipa.return_value = "mar ħaba"
        
        result = self.phonemizer_ipa.phonemize_string("مرحبا بالعالم", "ar")
        
        mock_mantoq.assert_called_once_with("مرحبا بالعالم")
        mock_bw2ipa.assert_called_once_with("mar haba")
        self.assertEqual(result, "mar ħaba")

    @patch('phoonnx.thirdparty.bw2ipa.translate')
    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_ipa_complex_phonemes(self, mock_mantoq, mock_bw2ipa):
        """Test IPA phonemization with complex Arabic phonemes."""
        mock_mantoq.return_value = ("الشمس", ['?', 'a', 'l', 'S', 'a', 'm', 's'])
        mock_bw2ipa.return_value = "ʔaʃːams"  # IPA representation with gemination
        
        result = self.phonemizer_ipa.phonemize_string("الشمس", "ar")
        
        mock_bw2ipa.assert_called_once_with("?alSams")
        self.assertEqual(result, "ʔaʃːams")

    # Edge Cases and Error Handling
    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_empty_text(self, mock_mantoq):
        """Test phonemize_string with empty text."""
        mock_mantoq.return_value = ("", [])
        
        result = self.phonemizer_mantoq.phonemize_string("", "ar")
        
        mock_mantoq.assert_called_once_with("")
        self.assertEqual(result, "")

    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_whitespace_only(self, mock_mantoq):
        """Test phonemize_string with whitespace-only text."""
        mock_mantoq.return_value = ("   ", [' ', ' ', ' '])
        
        result = self.phonemizer_mantoq.phonemize_string("   ", "ar")
        
        self.assertEqual(result, "   ")

    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_newlines_and_tabs(self, mock_mantoq):
        """Test phonemize_string with newlines and tabs."""
        mock_mantoq.return_value = ("text\nwith\ttabs", ['t', 'e', 'x', 't', '\n', 'w', 'i', 't', 'h', '\t', 't', 'a', 'b', 's'])
        
        result = self.phonemizer_mantoq.phonemize_string("مرحبا\nبالعالم\tاليوم", "ar")
        
        self.assertIn('\n', result)
        self.assertIn('\t', result)

    def test_phonemize_string_default_language_parameter(self):
        """Test phonemize_string with default language parameter (should be 'ar')."""
        with patch('phoonnx.thirdparty.mantoq.g2p') as mock_mantoq:
            mock_mantoq.return_value = ("text", ['t', 'e', 's', 't'])
            
            result = self.phonemizer_mantoq.phonemize_string("مرحبا")
            
            mock_mantoq.assert_called_once_with("مرحبا")
            self.assertEqual(result, "test")

    def test_phonemize_string_invalid_language(self):
        """Test phonemize_string with invalid language code."""
        invalid_langs = ["en", "fr", "es", "de", "zh", "ja", "invalid"]
        
        for lang in invalid_langs:
            with self.subTest(lang=lang):
                with self.assertRaises(ValueError):
                    self.phonemizer_mantoq.phonemize_string("test", lang)

    # Exception Handling and Error Propagation
    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_mantoq_exception(self, mock_mantoq):
        """Test phonemize_string when mantoq raises an exception."""
        mock_mantoq.side_effect = Exception("Mantoq processing error")
        
        with self.assertRaises(Exception) as context:
            self.phonemizer_mantoq.phonemize_string("مرحبا", "ar")
        
        self.assertIn("Mantoq processing error", str(context.exception))

    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_mantoq_import_error(self, mock_mantoq):
        """Test phonemize_string when mantoq has import issues."""
        mock_mantoq.side_effect = ImportError("mantoq library not found")
        
        with self.assertRaises(ImportError):
            self.phonemizer_mantoq.phonemize_string("مرحبا", "ar")

    @patch('phoonnx.thirdparty.bw2ipa.translate')
    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_bw2ipa_exception(self, mock_mantoq, mock_bw2ipa):
        """Test phonemize_string when bw2ipa raises an exception."""
        mock_mantoq.return_value = ("text", ['t', 'e', 's', 't'])
        mock_bw2ipa.side_effect = Exception("bw2ipa translation error")
        
        with self.assertRaises(Exception) as context:
            self.phonemizer_ipa.phonemize_string("مرحبا", "ar")
        
        self.assertIn("bw2ipa translation error", str(context.exception))

    @patch('phoonnx.thirdparty.bw2ipa.translate')
    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_bw2ipa_import_error(self, mock_mantoq, mock_bw2ipa):
        """Test phonemize_string when bw2ipa has import issues."""
        mock_mantoq.return_value = ("text", ['t', 'e', 's', 't'])
        mock_bw2ipa.side_effect = ImportError("bw2ipa library not found")
        
        with self.assertRaises(ImportError):
            self.phonemizer_ipa.phonemize_string("مرحبا", "ar")

    # Data Format Validation Tests
    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_invalid_mantoq_return_formats(self, mock_mantoq):
        """Test phonemize_string when mantoq returns unexpected data formats."""
        test_cases = [
            (None, "None return value"),
            ("not a tuple", "String instead of tuple"),
            (123, "Integer instead of tuple"),
            ([], "List instead of tuple"),
            (("only_one_element",), "Tuple with one element"),
            ((1, 2, 3), "Tuple with three elements"),
            (("text", None), "Second element is None"),
            (("text", "not a list"), "Second element is not iterable"),
            (("text", 123), "Second element is not iterable"),
            ((None, []), "First element is None"),
            ((123, []), "First element is not string")
        ]
        
        for return_value, description in test_cases:
            with self.subTest(return_value=return_value, desc=description):
                mock_mantoq.return_value = return_value
                with self.assertRaises((TypeError, ValueError, AttributeError)):
                    self.phonemizer_mantoq.phonemize_string("مرحبا", "ar")

    @patch('phoonnx.thirdparty.bw2ipa.translate')
    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_bw2ipa_return_types(self, mock_mantoq, mock_bw2ipa):
        """Test phonemize_string when bw2ipa returns various types."""
        mock_mantoq.return_value = ("text", ['t', 'e', 's', 't'])
        
        test_cases = [
            (None, None),
            ("", ""),
            ("valid_ipa", "valid_ipa"),
            (123, 123)  # Should handle non-string returns
        ]
        
        for return_value, expected in test_cases:
            with self.subTest(return_value=return_value):
                mock_bw2ipa.return_value = return_value
                result = self.phonemizer_ipa.phonemize_string("مرحبا", "ar")
                self.assertEqual(result, expected)

    # Comprehensive Arabic Text Tests
    def test_phonemize_string_arabic_test_cases_from_main(self):
        """Test phonemize_string with the Arabic samples from the main block."""
        arabic_test_cases = [
            ("مرحبا بالعالم", "Hello world greeting"),
            ("ذهب الطالب إلى المكتبة لقراءة كتاب عن تاريخ الأندلس.", "Complex sentence with various elements"),
            ("الشمس", "Sun - tests sun letter assimilation"),
            ("فيل", "Elephant - tests long vowels"),
            ("يوم", "Day - tests glides"),
            ("سور", "Wall - tests long vowels"),
            ("لو", "If - tests glides")
        ]
        
        with patch('phoonnx.thirdparty.mantoq.g2p') as mock_mantoq, \
             patch('phoonnx.thirdparty.bw2ipa.translate') as mock_bw2ipa:
            for arabic_text, description in arabic_test_cases:
                with self.subTest(text=arabic_text, desc=description):
                    # Mock realistic phoneme output
                    mock_phonemes = list(arabic_text.replace(' ', '_+_'))
                    mock_mantoq.return_value = (arabic_text, mock_phonemes)
                    mock_bw2ipa.return_value = f"ipa_{arabic_text}"
                    
                    # Test MANTOQ alphabet
                    result_mantoq = self.phonemizer_mantoq.phonemize_string(arabic_text, "ar")
                    self.assertIsInstance(result_mantoq, str)
                    
                    # Test IPA alphabet
                    result_ipa = self.phonemizer_ipa.phonemize_string(arabic_text, "ar")
                    self.assertIsInstance(result_ipa, str)
                    self.assertTrue(result_ipa.startswith("ipa_"))

    def test_phonemize_string_additional_arabic_samples(self):
        """Test phonemize_string with additional Arabic text samples."""
        additional_samples = [
            ("كتاب", "Book - simple word"),
            ("مدرسة", "School"),
            ("بيت", "House"),
            ("ماء", "Water"),
            ("نار", "Fire"),
            ("هواء", "Air"),
            ("أرض", "Earth"),
            ("قمر", "Moon"),
            ("نجم", "Star"),
            ("بحر", "Sea")
        ]
        
        with patch('phoonnx.thirdparty.mantoq.g2p') as mock_mantoq:
            for arabic_text, description in additional_samples:
                with self.subTest(text=arabic_text, desc=description):
                    mock_phonemes = list(arabic_text)
                    mock_mantoq.return_value = (arabic_text, mock_phonemes)
                    
                    result = self.phonemizer_mantoq.phonemize_string(arabic_text, "ar")
                    self.assertIsInstance(result, str)
                    mock_mantoq.assert_called_with(arabic_text)

    # Special Characters and Mixed Content Tests
    def test_phonemize_string_mixed_content(self):
        """Test phonemize_string with mixed Arabic and other content."""
        mixed_cases = [
            ("مرحبا123", "Arabic with numbers"),
            ("مرحبا!", "Arabic with exclamation"),
            ("مرحبا؟", "Arabic with Arabic question mark"), 
            ("مرحبا?", "Arabic with Latin question mark"),
            ("مرحبا، كيف حالك؟", "Arabic with punctuation"),
            ("مرحبا (test)", "Arabic with parentheses"),
            ("مرحبا@gmail.com", "Arabic with email-like text"),
            ("123 مرحبا 456", "Arabic surrounded by numbers"),
            ("Hello مرحبا World", "Arabic mixed with English"),
            ("مرحبا - Welcome", "Arabic with dash and English")
        ]
        
        with patch('phoonnx.thirdparty.mantoq.g2p') as mock_mantoq:
            for text, description in mixed_cases:
                with self.subTest(text=repr(text), desc=description):
                    mock_mantoq.return_value = ("normalized", ['t', 'e', 's', 't'])
                    
                    result = self.phonemizer_mantoq.phonemize_string(text, "ar")
                    self.assertIsInstance(result, str)
                    mock_mantoq.assert_called_with(text)

    # Performance and Stress Tests
    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_very_long_text(self, mock_mantoq):
        """Test phonemize_string with very long Arabic text."""
        long_text = "مرحبا بالعالم الجميل " * 100  # ~2000 characters
        long_phonemes = ['m', 'a', 'r'] * 500  # Simulate long phoneme output
        
        mock_mantoq.return_value = ("normalized", long_phonemes)
        
        result = self.phonemizer_mantoq.phonemize_string(long_text, "ar")
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 1500)  # 3 chars * 500 repetitions

    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_empty_phonemes_list(self, mock_mantoq):
        """Test phonemize_string when mantoq returns empty phonemes list."""
        mock_mantoq.return_value = ("text", [])
        
        result = self.phonemizer_mantoq.phonemize_string("مرحبا", "ar")
        self.assertEqual(result, "")

    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_single_separator(self, mock_mantoq):
        """Test phonemize_string with only separator tokens."""
        mock_mantoq.return_value = ("text", ['_+_'])
        
        result = self.phonemizer_mantoq.phonemize_string("   ", "ar")
        self.assertEqual(result, " ")

    # Type Validation Tests
    def test_phonemize_string_parameter_type_validation(self):
        """Test phonemize_string with various invalid parameter types."""
        invalid_text_types = [
            (None, "None value"),
            (123, "integer"),
            (12.34, "float"), 
            ([], "empty list"),
            ({}, "empty dict"),
            (object(), "generic object"),
            (True, "boolean True"),
            (False, "boolean False")
        ]
        
        for invalid_text, description in invalid_text_types:
            with self.subTest(text_type=type(invalid_text).__name__, desc=description):
                with self.assertRaises((TypeError, AttributeError, ValueError)):
                    self.phonemizer_mantoq.phonemize_string(invalid_text, "ar")

    # Word Boundary and Separator Preservation Tests
    @patch('phoonnx.thirdparty.mantoq.g2p')
    def test_phonemize_string_word_boundary_edge_cases(self, mock_mantoq):
        """Test edge cases in word boundary handling."""
        test_cases = [
            (["a", "_+_", "b"], "a b", 1, "Basic separator"),
            (["a", "_+_", "b", "_+_", "c"], "a b c", 2, "Multiple separators"),
            (["_+_", "starts", "with", "sep"], " startswith sep", 1, "Leading separator"),
            (["ends", "with", "_+_"], "endswith ", 1, "Trailing separator"),
            (["_+_", "_+_", "multiple", "_+_", "_+_"], "  multiple  ", 4, "Multiple consecutive"),
            (["no", "separators", "here"], "noseparatorshere", 0, "No separators"),
            (["_+_"], " ", 1, "Only separator"),
            ([], "", 0, "Empty phonemes")
        ]
        
        for phonemes, expected, expected_spaces, description in test_cases:
            with self.subTest(phonemes=phonemes, desc=description):
                mock_mantoq.return_value = ("text", phonemes)
                
                result = self.phonemizer_mantoq.phonemize_string("test", "ar")
                self.assertEqual(result, expected, f"Unexpected result for: {description}")
                self.assertEqual(result.count(' '), expected_spaces, f"Wrong space count for: {description}")

    # Alphabet-specific Behavior Tests
    def test_alphabet_specific_processing_paths(self):
        """Test that different alphabet settings trigger different processing paths."""
        with patch('phoonnx.thirdparty.mantoq.g2p') as mock_mantoq, \
             patch('phoonnx.thirdparty.bw2ipa.translate') as mock_bw2ipa:
            mock_mantoq.return_value = ("text", ['t', 'e', 's', 't'])
            
            # Test MANTOQ path (should not call bw2ipa)
            mock_bw2ipa.reset_mock()
            mantoq_result = self.phonemizer_mantoq.phonemize_string("مرحبا", "ar")
            mock_bw2ipa.assert_not_called()
            self.assertEqual(mantoq_result, "test")
            
            # Test IPA path (should call bw2ipa)
            mock_bw2ipa.return_value = "ipa_test"
            ipa_result = self.phonemizer_ipa.phonemize_string("مرحبا", "ar")
            mock_bw2ipa.assert_called_once_with("test")
            self.assertEqual(ipa_result, "ipa_test")
            
            # Results should be different
            self.assertNotEqual(mantoq_result, ipa_result)

    def test_alphabet_attribute_consistency(self):
        """Test that alphabet attribute remains consistent throughout object lifecycle."""
        # Test MANTOQ phonemizer
        self.assertEqual(self.phonemizer_mantoq.alphabet, Alphabet.MANTOQ)
        with patch('phoonnx.thirdparty.mantoq.g2p') as mock_mantoq:
            mock_mantoq.return_value = ("text", [])
            self.phonemizer_mantoq.phonemize_string("test", "ar")
            # Alphabet should remain unchanged after phonemization
            self.assertEqual(self.phonemizer_mantoq.alphabet, Alphabet.MANTOQ)
        
        # Test IPA phonemizer
        self.assertEqual(self.phonemizer_ipa.alphabet, Alphabet.IPA)
        with patch('phoonnx.thirdparty.mantoq.g2p') as mock_mantoq, \
             patch('phoonnx.thirdparty.bw2ipa.translate') as mock_bw2ipa:
            mock_mantoq.return_value = ("text", [])
            mock_bw2ipa.return_value = ""
            self.phonemizer_ipa.phonemize_string("test", "ar")
            # Alphabet should remain unchanged after phonemization
            self.assertEqual(self.phonemizer_ipa.alphabet, Alphabet.IPA)


class TestMantoqPhonemizerIntegration(unittest.TestCase):
    """Integration tests for MantoqPhonemizer.
    
    These tests check the integration between MantoqPhonemizer and its dependencies,
    though they still use mocks to avoid requiring actual external libraries.
    """
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.phonemizer_mantoq = MantoqPhonemizer()
        self.phonemizer_ipa = MantoqPhonemizer(alphabet=Alphabet.IPA)

    def test_full_pipeline_mantoq_alphabet(self):
        """Test complete phonemization pipeline for MANTOQ alphabet."""
        with patch('phoonnx.thirdparty.mantoq.g2p') as mock_mantoq:
            # Simulate realistic mantoq output
            mock_mantoq.return_value = (
                "مرحبا بالعالم", 
                ['m', 'a', 'r', 'H', 'a', 'b', 'a', '_+_', 'b', 'i', 'l', 'E', 'a', 'l', 'a', 'm']
            )
            
            result = self.phonemizer_mantoq.phonemize_string("مرحبا بالعالم", "ar")
            
            # Verify full pipeline execution
            mock_mantoq.assert_called_once_with("مرحبا بالعالم")
            self.assertEqual(result, "marHaba bilEalam")
            self.assertIn(' ', result)

    def test_full_pipeline_ipa_alphabet(self):
        """Test complete phonemization pipeline for IPA alphabet."""
        with patch('phoonnx.thirdparty.mantoq.g2p') as mock_mantoq, \
             patch('phoonnx.thirdparty.bw2ipa.translate') as mock_bw2ipa:
            # Simulate realistic mantoq output
            mock_mantoq.return_value = (
                "مرحبا بالعالم",
                ['m', 'a', 'r', 'H', 'a', 'b', 'a', '_+_', 'b', 'i', 'l', 'E', 'a', 'l', 'a', 'm']
            )
            # Simulate realistic bw2ipa output
            mock_bw2ipa.return_value = "marħaba bilʕaːlam"
            
            result = self.phonemizer_ipa.phonemize_string("مرحبا بالعالم", "ar")
            
            # Verify full pipeline execution
            mock_mantoq.assert_called_once_with("مرحبا بالعالم")
            mock_bw2ipa.assert_called_once_with("marHaba bilEalam")
            self.assertEqual(result, "marħaba bilʕaːlam")

    def test_error_propagation_through_pipeline(self):
        """Test that errors are properly propagated through the processing pipeline."""
        # Test mantoq error propagation
        with patch('phoonnx.thirdparty.mantoq.g2p') as mock_mantoq:
            mock_mantoq.side_effect = RuntimeError("Mantoq processing failed")
            
            with self.assertRaises(RuntimeError) as context:
                self.phonemizer_mantoq.phonemize_string("مرحبا", "ar")
            
            self.assertIn("Mantoq processing failed", str(context.exception))

        # Test bw2ipa error propagation  
        with patch('phoonnx.thirdparty.mantoq.g2p') as mock_mantoq, \
             patch('phoonnx.thirdparty.bw2ipa.translate') as mock_bw2ipa:
            mock_mantoq.return_value = ("text", ['t', 'e', 's', 't'])
            mock_bw2ipa.side_effect = RuntimeError("bw2ipa translation failed")
            
            with self.assertRaises(RuntimeError) as context:
                self.phonemizer_ipa.phonemize_string("مرحبا", "ar")
            
            self.assertIn("bw2ipa translation failed", str(context.exception))

    @unittest.skipIf(True, "Real integration test - enable when testing with actual libraries")
    def test_real_arabic_phonemization_samples(self):
        """Real integration test with actual Arabic text samples.
        
        This test should be enabled when testing with actual mantoq and bw2ipa libraries.
        Currently skipped to avoid dependency on external libraries in unit tests.
        """
        # Test cases from the main block of the original file
        test_cases = [
            "مرحبا بالعالم",
            "ذهب الطالب إلى المكتبة لقراءة كتاب عن تاريخ الأندلس.",
            "الشمس",
            "فيل", 
            "يوم",
            "سور",
            "لو"
        ]
        
        for arabic_text in test_cases:
            with self.subTest(text=arabic_text):
                # Test with MANTOQ alphabet
                result_mantoq = self.phonemizer_mantoq.phonemize_string(arabic_text, "ar")
                self.assertIsInstance(result_mantoq, str)
                self.assertGreater(len(result_mantoq), 0)
                
                # Test with IPA alphabet
                result_ipa = self.phonemizer_ipa.phonemize_string(arabic_text, "ar")
                self.assertIsInstance(result_ipa, str)
                self.assertGreater(len(result_ipa), 0)
                
                # Results should be different (MANTOQ vs IPA)
                self.assertNotEqual(result_mantoq, result_ipa)


if __name__ == '__main__':
    # Create a test suite with comprehensive output
    unittest.main(verbosity=2, buffer=True)
