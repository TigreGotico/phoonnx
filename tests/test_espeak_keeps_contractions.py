"""An espeak-ng voice is given English contractions as written, not expanded.

phoonnx's normalizer expands every English contraction before the text reaches
the phonemizer ("don't" -> "do not", "it's" -> "it is", 166 of them). espeak-ng
says a contraction the way people do, so the expansion only made sentences sound
like a form letter, and the table is a guess for "he's" and "I'd". Numbers, dates,
units and titles are still normalized, and phonemizers other than espeak keep the
expansion.
"""
import shutil
import unittest

from phoonnx.config import PhonemeType, get_phonemizer
from phoonnx.normalization.text import normalize


class TestNormalizeContractions(unittest.TestCase):

    def test_contractions_expand_by_default(self):
        self.assertEqual(normalize("I don't know what it's called", "en-US"),
                         "I do not know what it is called")

    def test_contractions_can_be_kept(self):
        self.assertEqual(
            normalize("I don't know what it's called", "en-US", expand_contractions=False),
            "I don't know what it's called")

    def test_numbers_and_units_still_change_when_contractions_are_kept(self):
        text = normalize("It's 12 o'clock, don't wait", "en-US", expand_contractions=False)
        self.assertIn("It's", text)
        self.assertIn("don't", text)
        self.assertNotIn("12", text)

    def test_typographic_apostrophes_are_kept_as_a_plain_apostrophe(self):
        self.assertEqual(normalize("Don’t stop", "en-US", expand_contractions=False),
                         "Don't stop")


class TestEspeakPhonemizerKeepsContractions(unittest.TestCase):

    def test_the_espeak_normalizer_keeps_them(self):
        phonemizer = get_phonemizer(PhonemeType.ESPEAK)
        self.assertEqual(phonemizer.normalizer("I don't know", "en-US"), "I don't know")
        self.assertNotIn("12", phonemizer.normalizer("It's 12 o'clock", "en-US"))

    def test_another_phonemizer_still_expands_them(self):
        phonemizer = get_phonemizer(PhonemeType.GRUUT)
        self.assertEqual(phonemizer.normalizer("I don't know", "en-US"), "I do not know")

    @unittest.skipUnless(shutil.which("espeak-ng"), "needs the espeak-ng binary")
    def test_espeak_says_dont_as_one_word(self):
        phonemes = "".join("".join(chunk) for chunk in
                           get_phonemizer(PhonemeType.ESPEAK).phonemize("I don't know", "en-US"))
        # "do not" would be duːnˌɑːt; "don't" is dˈoʊnt
        self.assertIn("dˈoʊnt", phonemes)
        self.assertNotIn("duː", phonemes)


if __name__ == "__main__":
    unittest.main()
