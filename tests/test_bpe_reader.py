"""The built-in ``tokenizer.json`` reader must agree with ``tokenizers`` exactly.

A wrong id does not raise — it synthesises the wrong audio. So the reader is
held to bit-identical output against the reference implementation over every
fixture, and over any real voice vocab that happens to be in the local cache.
"""
import glob
import json
import os
import random
import time
import unittest

from phoonnx._bpe import Tokenizer, UnsupportedTokenizer
from phoonnx.tokenizer import load_hf_tokenizer

try:
    from tokenizers import Tokenizer as ReferenceTokenizer
except ImportError:
    ReferenceTokenizer = None

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

TEXTS = [
    "", " ", "\n", "a", "hello world",
    "Hello, World! How are you today?",
    "a  b   c", "text with     five spaces",
    "  leading and trailing  ",
    "new\nline\ttab\r\ncrlf",
    "olá, tudo bem? çãõ ñ àéîõü",
    "The quick brown fox jumps over the lazy dog 12345.",
    "мир 世界 한국어 🌍 emoji test",
    "don't can't I'll we've they're it's",
    "123 4567 0.5 -3 1,000,000 42%",
    "MiXeD CaSe UPPER lower",
    "sub-word hyphen_underscore (parens) [brackets] {braces} <angle>",
    "ΑΒΓ αβγ ⟨angle⟩ ½ ¾ № — – ’ “ ”",
    "الْعَرَبِيَّة مرحبا بالعالم",
    "...!!!???;;;:::",
]

FUZZ_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    " \t\n\r.,!?;:'\"-_()[]{}<>/@#$%^&*+=~`|\\"
    "áéíóúñçãõäöüßΑΒΓαβγмирשלום العربية 世界日本語한국어🌍🎉½№’—"
)


def fixture_files():
    """Vocabularies the reader must handle, and handle identically."""
    return sorted(glob.glob(os.path.join(FIXTURES, "tokenizer_*.json")))


def declined_fixture_files():
    """Vocabularies the reader must refuse, so `tokenizers` gets them instead."""
    return sorted(glob.glob(os.path.join(FIXTURES, "declined_*.json")))


def voice_cache_files():
    """Real voice vocabularies, when this machine has downloaded any."""
    root = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    pattern = os.path.join(root, "phoonnx", "**", "tokenizer.json")
    return sorted(glob.glob(pattern, recursive=True))


class TestFixturesPresent(unittest.TestCase):
    def test_both_pre_tokenizer_families_are_covered(self):
        """A ByteLevel vocab and a Whitespace vocab, the two shapes voices ship."""
        kinds = set()
        for path in fixture_files():
            with open(path, encoding="utf-8") as handle:
                spec = json.load(handle)
            kinds.add(json.dumps(spec.get("pre_tokenizer")).count("ByteLevel") > 0)
        self.assertEqual(kinds, {True, False}, fixture_files())

    def test_a_realistic_vocabulary_is_exercised_in_ci(self):
        """The shipped vocabularies hold 90k-151k tokens. A 600-token toy shares
        none of their merge depth, so CI must carry something the same shape."""
        biggest = 0
        for path in fixture_files():
            with open(path, encoding="utf-8") as handle:
                biggest = max(biggest, len(json.load(handle)["model"]["vocab"]))
        self.assertGreaterEqual(biggest, 10000, fixture_files())

    def test_the_qwen_and_llama_families_are_both_present(self):
        """The two ByteLevel shapes the voices are built on. Llama-family sets
        `ignore_merges`, which changes the ids; Qwen-family does not."""
        flags = set()
        for path in fixture_files():
            with open(path, encoding="utf-8") as handle:
                model = json.load(handle)["model"]
            if len(model["vocab"]) >= 10000:
                flags.add(bool(model.get("ignore_merges")))
        self.assertEqual(flags, {True, False}, fixture_files())


@unittest.skipIf(ReferenceTokenizer is None, "the `tokenizers` package is not installed")
class TestDifferentialAgainstTokenizers(unittest.TestCase):
    """Every id the reader produces must equal the reference id."""

    def _pair(self, path):
        return Tokenizer.from_file(path), ReferenceTokenizer.from_file(path)

    def _compare(self, path):
        mine, reference = self._pair(path)
        for text in TEXTS:
            self.assertEqual(mine.encode(text).ids, reference.encode(text).ids,
                             f"{os.path.basename(path)} disagrees on {text!r}")

    def test_fixtures_encode_identically(self):
        paths = fixture_files()
        self.assertTrue(paths, "no fixtures found")
        for path in paths:
            with self.subTest(fixture=os.path.basename(path)):
                self._compare(path)

    def test_fixtures_survive_random_text(self):
        """Fuzz: the shapes a user actually types are not the only shapes."""
        rng = random.Random(20260811)
        for path in fixture_files():
            mine, reference = self._pair(path)
            for _ in range(400):
                text = "".join(rng.choice(FUZZ_ALPHABET)
                               for _ in range(rng.randint(0, 60)))
                self.assertEqual(mine.encode(text).ids, reference.encode(text).ids,
                                 f"{os.path.basename(path)} disagrees on {text!r}")

    def test_encode_without_special_tokens_matches(self):
        """The engines that read a checkpoint's own BPE pass this flag."""
        for path in fixture_files():
            mine, reference = self._pair(path)
            for text in TEXTS:
                with self.subTest(fixture=os.path.basename(path), text=text):
                    self.assertEqual(
                        mine.encode(text, add_special_tokens=False).ids,
                        reference.encode(text, add_special_tokens=False).ids)

    def test_vocab_size_matches(self):
        """VoiceConfig derives a Chatterbox voice's num_symbols from this."""
        for path in fixture_files():
            mine, reference = self._pair(path)
            with self.subTest(fixture=os.path.basename(path)):
                self.assertEqual(mine.get_vocab_size(), reference.get_vocab_size())
                self.assertEqual(mine.get_vocab_size(False),
                                 reference.get_vocab_size(False))
                self.assertEqual(mine.get_vocab(), reference.get_vocab())

    def test_token_to_id_matches(self):
        for path in fixture_files():
            mine, reference = self._pair(path)
            for token in ("[SPACE]", "[UNK]", "<|endoftext|>", "the", "nope!!"):
                with self.subTest(fixture=os.path.basename(path), token=token):
                    self.assertEqual(mine.token_to_id(token),
                                     reference.token_to_id(token))

    def test_downloaded_voice_vocabs_encode_identically(self):
        paths = voice_cache_files()
        if not paths:
            self.skipTest("no voice vocabularies in the local cache")
        for path in paths:
            with self.subTest(vocab=path):
                try:
                    self._compare(path)
                except UnsupportedTokenizer as unsupported:
                    self.skipTest(f"reader declines this vocab: {unsupported}")


class TestReaderWithoutReference(unittest.TestCase):
    """What the reader must do whether or not `tokenizers` is installed."""

    def test_encode_is_deterministic(self):
        for path in fixture_files():
            tokenizer = Tokenizer.from_file(path)
            first = tokenizer.encode("hello world, hello again").ids
            self.assertEqual(first, tokenizer.encode("hello world, hello again").ids)
            self.assertTrue(first)

    def test_unknown_model_is_refused_loudly(self):
        """Silence is the failure mode that corrupts audio, so refuse instead."""
        spec = {"model": {"type": "WordPiece", "vocab": {}}}
        with self.assertRaises(UnsupportedTokenizer):
            Tokenizer(spec)

    def test_unknown_normalizer_is_refused_loudly(self):
        spec = {"model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
                "normalizer": {"type": "Bert"}}
        with self.assertRaises(UnsupportedTokenizer):
            Tokenizer(spec).encode("a")

    def test_loader_returns_the_pure_python_reader(self):
        for path in fixture_files():
            self.assertIsInstance(load_hf_tokenizer(path), Tokenizer)

    @unittest.skipIf(ReferenceTokenizer is None, "the `tokenizers` package is not installed")
    def test_loader_falls_back_when_the_vocab_is_beyond_the_reader(self):
        """An exotic vocab degrades to `tokenizers` rather than breaking."""
        import phoonnx._bpe as bpe
        from unittest import mock
        path = fixture_files()[0]
        with mock.patch.object(bpe.Tokenizer, "from_file",
                               side_effect=UnsupportedTokenizer("pretend")):
            self.assertIsInstance(load_hf_tokenizer(path), ReferenceTokenizer)


if __name__ == "__main__":
    unittest.main()


class TestVocabulariesThatMustBeDeclined(unittest.TestCase):
    """Accepting a vocabulary the reader cannot reproduce is worse than
    refusing it: the fallback to `tokenizers` only fires on a refusal, so a
    wrong acceptance goes all the way to the audio."""

    def test_the_indic_parler_prompt_vocab_is_declined(self):
        """The prompt tokenizer — the text actually spoken — is BPE with
        `byte_fallback` and `fuse_unk` set. Read without them, a character
        outside the vocabulary becomes `<unk>` where it should become a run of
        `<0xNN>` byte tokens."""
        paths = declined_fixture_files()
        self.assertTrue(paths, "no declined fixtures found")
        for path in paths:
            with self.subTest(fixture=os.path.basename(path)):
                with self.assertRaises(UnsupportedTokenizer):
                    Tokenizer.from_file(path)

    @unittest.skipIf(ReferenceTokenizer is None, "the `tokenizers` package is not installed")
    def test_a_declined_vocab_is_handed_to_tokenizers(self):
        for path in declined_fixture_files():
            with self.subTest(fixture=os.path.basename(path)):
                self.assertIsInstance(load_hf_tokenizer(path), ReferenceTokenizer)

    def _bpe(self, **options):
        spec = {"model": dict({"type": "BPE", "vocab": {"a": 0}, "merges": []},
                              **options)}
        return spec

    def test_each_unimplemented_bpe_option_is_declined(self):
        for options in ({"byte_fallback": True},
                        {"fuse_unk": True},
                        {"dropout": 0.1},
                        {"end_of_word_suffix": "</w>"},
                        {"a_switch_from_a_future_version": True}):
            with self.subTest(options=options):
                with self.assertRaises(UnsupportedTokenizer):
                    Tokenizer(self._bpe(**options))

    def test_the_options_it_does_implement_are_accepted(self):
        Tokenizer(self._bpe(byte_fallback=False, fuse_unk=False, dropout=None,
                            end_of_word_suffix=None, ignore_merges=False,
                            unk_token=None, continuing_subword_prefix=""))

    def test_a_unigram_model_names_the_file_and_the_remedy(self):
        path = os.path.join(FIXTURES, "does_not_matter.json")
        with self.assertRaises(UnsupportedTokenizer) as caught:
            Tokenizer({"model": {"type": "Unigram", "vocab": []}}, source=path)
        message = str(caught.exception)
        self.assertIn(path, message)
        self.assertIn("Unigram", message)
        self.assertIn("phoonnx[indic-parler]", message)

    def test_metaspace_options_that_move_boundaries_are_declined(self):
        for pre in ({"type": "Metaspace", "replacement": "▁", "split": True},
                    {"type": "Metaspace", "replacement": "▁",
                     "prepend_scheme": "first"}):
            with self.subTest(pre_tokenizer=pre):
                spec = self._bpe()
                spec["pre_tokenizer"] = pre
                with self.assertRaises(UnsupportedTokenizer):
                    Tokenizer(spec).encode("a b")


class TestMergeLoopIsNotQuadratic(unittest.TestCase):
    """A pre-token is not always short. The pre-tokenizers these vocabularies
    declare keep a run of letters whole, so a paragraph of Chinese or Japanese —
    written without spaces — arrives at the merge loop as one piece of thousands
    of characters. Rescanning it after every merge is O(n^2), and it blocks the
    request."""

    CHINESE = ("中文测试一二三四五六七"
               "八九十百千万亿天地人和"
               "平世界你好谢谢再见")

    def _bytelevel_fixture(self):
        for path in fixture_files():
            with open(path, encoding="utf-8") as handle:
                spec = json.load(handle)
            if (len(spec["model"]["vocab"]) >= 10000
                    and "ByteLevel" in json.dumps(spec.get("pre_tokenizer"))):
                return path
        self.skipTest("no realistic ByteLevel fixture")

    def _time(self, tokenizer, length):
        text = (self.CHINESE * (length // len(self.CHINESE) + 1))[:length]
        start = time.perf_counter()
        tokenizer.encode(text)
        return time.perf_counter() - start

    def test_cost_grows_about_linearly_with_pre_token_length(self):
        tokenizer = Tokenizer.from_file(self._bytelevel_fixture())
        tokenizer.encode("warm the regex cache up")
        short = self._time(tokenizer, 1000)
        long = self._time(tokenizer, 8000)
        # 8x the text. Linear would be ~8x the time; quadratic would be ~64x.
        # The bound is loose because a shared CI runner is a noisy clock, and
        # still nowhere near what the rescan did (measured 0.43 s -> 27 s here).
        self.assertLess(long, max(short * 20, 0.5),
                        f"1000 chars took {short:.4f}s, 8000 took {long:.4f}s")

    def test_a_long_pre_token_finishes_promptly(self):
        """A 4000-character request is an ordinary size for a TTS server."""
        tokenizer = Tokenizer.from_file(self._bytelevel_fixture())
        tokenizer.encode("warm the regex cache up")
        self.assertLess(self._time(tokenizer, 4000), 2.0)

    @unittest.skipIf(ReferenceTokenizer is None, "the `tokenizers` package is not installed")
    def test_a_long_pre_token_still_agrees_with_the_reference(self):
        """Speed is worthless if it changed an id."""
        path = self._bytelevel_fixture()
        mine, reference = Tokenizer.from_file(path), ReferenceTokenizer.from_file(path)
        for length in (200, 1000, 4000):
            text = (self.CHINESE * (length // len(self.CHINESE) + 1))[:length]
            with self.subTest(length=length):
                self.assertEqual(mine.encode(text).ids, reference.encode(text).ids)
