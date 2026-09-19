"""Tests for phoneme-map seeding and the untrained-symbol audit."""
import unittest

from phoonnx.tokenizer import (DEFAULT_IPA_PHONEME_ID_MAP, SPECIAL_TOKENS,
                               phoneme_map_seed, untrained_map_symbols)


class TestPhonemeMapSeed(unittest.TestCase):
    def test_default_ipa_seed_includes_full_table(self):
        seed = phoneme_map_seed({"a", "b"}, ipa=True)
        self.assertTrue(set(DEFAULT_IPA_PHONEME_ID_MAP).issubset(seed))
        self.assertIn("a", seed)

    def test_corpus_only_excludes_digits_and_punctuation(self):
        corpus = {"a", "ˈ", "tˤ"}
        seed = phoneme_map_seed(corpus, ipa=True, include_defaults=False)
        self.assertEqual(seed, corpus)
        for sym in "0123456789!?.,;:":
            self.assertNotIn(sym, seed)

    def test_non_ipa_never_adds_defaults(self):
        seed = phoneme_map_seed({"x"}, ipa=False, include_defaults=True)
        self.assertEqual(seed, {"x"})

    def test_empty_corpus(self):
        self.assertEqual(phoneme_map_seed(set(), ipa=False), set())
        self.assertEqual(phoneme_map_seed(set(), ipa=True, include_defaults=False), set())

    def test_input_not_mutated(self):
        corpus = {"a"}
        phoneme_map_seed(corpus, ipa=True)
        self.assertEqual(corpus, {"a"})


class TestUntrainedMapSymbols(unittest.TestCase):
    def test_flags_symbols_missing_from_corpus(self):
        pmap = {"_": 0, "^": 1, "$": 2, " ": 3, "a": 4, "5": 5, "!": 6}
        self.assertEqual(untrained_map_symbols(pmap, {"a"}), ["!", "5"])

    def test_special_tokens_never_flagged(self):
        pmap = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.assertEqual(untrained_map_symbols(pmap, set()), [])

    def test_full_default_map_against_digit_free_corpus(self):
        corpus = {k for k in DEFAULT_IPA_PHONEME_ID_MAP if k.isalpha()}
        unused = untrained_map_symbols(DEFAULT_IPA_PHONEME_ID_MAP, corpus)
        for d in "0123456789":
            self.assertIn(d, unused)
        self.assertNotIn("a", unused)

    def test_clean_map_yields_nothing(self):
        pmap = {"_": 0, "a": 1, "b": 2}
        self.assertEqual(untrained_map_symbols(pmap, {"a", "b"}), [])


if __name__ == "__main__":
    unittest.main()
