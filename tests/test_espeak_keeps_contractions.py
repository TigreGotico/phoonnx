"""An espeak-ng voice is given English contractions as written, not expanded.

espeak-ng says a contraction the way people do, so expanding it ("don't" -> "do not")
only makes a sentence sound like a form letter. text2spoken leaves contractions as
written for every phonemizer, and still reads numbers, dates and units.
"""
import shutil
import unittest

from phoonnx.config import PhonemeType, get_phonemizer


class TestEspeakPhonemizerKeepsContractions(unittest.TestCase):

    def test_the_espeak_normalizer_keeps_them(self):
        """The espeak phonemizer's normalizer keeps contractions."""
        phonemizer = get_phonemizer(PhonemeType.ESPEAK)
        self.assertEqual(phonemizer.normalizer("I don't know", "en-US"), "I don't know")
        self.assertNotIn("12", phonemizer.normalizer("It's 12 o'clock", "en-US"))

    def test_another_phonemizer_keeps_them_too(self):
        """Any other phonemizer's normalizer keeps them as well."""
        phonemizer = get_phonemizer(PhonemeType.GRUUT)
        self.assertEqual(phonemizer.normalizer("I don't know", "en-US"), "I don't know")

    @unittest.skipUnless(shutil.which("espeak-ng"), "needs the espeak-ng binary")
    def test_espeak_says_dont_as_one_word(self):
        """espeak-ng reads "don't" as one word, which expanding would split."""
        phonemes = "".join("".join(chunk) for chunk in
                           get_phonemizer(PhonemeType.ESPEAK).phonemize("I don't know", "en-US"))
        # "do not" would be duːnˌɑːt; "don't" is dˈoʊnt
        self.assertIn("dˈoʊnt", phonemes)
        self.assertNotIn("duː", phonemes)


if __name__ == "__main__":
    unittest.main()
