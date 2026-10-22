"""Arabic number verbalization (ovos-number-parser backed, no pyarabic)."""
import pathlib
import unittest

from scriptconv.phonemizers._vendored.mantoq.num2words import normalize_digits, num2words


class TestNormalizeDigits(unittest.TestCase):
    def test_arabic_indic_digits(self):
        self.assertEqual(normalize_digits("٠١٢٣٤٥٦٧٨٩"), "0123456789")

    def test_extended_arabic_indic_digits(self):
        self.assertEqual(normalize_digits("۰۱۲۳۴۵۶۷۸۹"), "0123456789")

    def test_arabic_decimal_and_thousands_separators(self):
        self.assertEqual(normalize_digits("١٬٢٣٤٫٥"), "1234.5")

    def test_ascii_passthrough(self):
        self.assertEqual(normalize_digits("abc 123"), "abc 123")


class TestNum2Words(unittest.TestCase):
    def test_integer_verbalized_in_arabic(self):
        out = num2words("عندي ٢٣ كتابا")
        self.assertNotRegex(out, r"\d")
        self.assertIn("وعشرون", out)

    def test_percent(self):
        out = num2words("50%")
        self.assertIn("بالمئة", out)
        self.assertNotIn("%", out)

    def test_decimal_number(self):
        out = num2words("0.5")
        self.assertNotRegex(out, r"\d")

    def test_no_numbers_is_identity_modulo_spaces(self):
        self.assertEqual(num2words("مرحبا  بكم"), "مرحبا بكم")

    def test_year_like_number(self):
        out = num2words("1999")
        self.assertNotRegex(out, r"\d")
        self.assertIn("ألف", out)


class TestNoPyarabicAnywhere(unittest.TestCase):
    def test_tree_is_pyarabic_free(self):
        root = pathlib.Path("phoonnx")
        hits = [str(f) for f in root.rglob("*.py")
                if "pyarabic" in f.read_text(errors="ignore")]
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
