"""The built-in ``tokenizer.json`` reader must decode exactly like ``tokenizers``.

OmniVoice hands the ids of the text it read back to the engine, and the engine
needs the string again. A decoder that is close but not identical does not
raise — it returns text that no longer matches the audio, so the reader is held
to character-identical output against the reference implementation.
"""
import glob
import json
import os
import random
import unittest

from phoonnx._bpe import Tokenizer, UnsupportedTokenizer

try:
    from tokenizers import Tokenizer as ReferenceTokenizer
except ImportError:
    ReferenceTokenizer = None

from tests.test_bpe_reader import TEXTS, FUZZ_ALPHABET, fixture_files

# Text the reader must survive, on top of the corpus the encoder is held to.
EXTRA_TEXTS = [
    "", " ", "  ", "\t", "\n\n\n",
    "a b", "a  b", "a   b   c    d",
    "   leading spaces", "trailing spaces   ", "   both   ",
    "ASCII only, nothing fancy.",
    "Ação, coração, não — português com acentos",
    "Grüße aus München, Straße, Fußgängerzone",
    "日本語のテキストです。これはテストです。",
    "中文测试一二三四五六七八九十",
    "한국어 문장 테스트입니다",
    "Привет, мир! Это тест.",
    "مرحبا بالعالم",
    "🌍🎉🚀 emoji only",
    "mixed 🌍 emoji 日本語 and ASCII",
    "punctuation!?;:,.'\"()[]{}<>/\\|@#$%^&*+=~`",
    "1234567890 3.14159 -42 1,000,000 50%",
    "tab\tseparated\tvalues",
    "line\nbreaks\r\nand\rreturns",
    "a" * 200,
    "the quick brown fox " * 20,
]

ALL_TEXTS = list(TEXTS) + EXTRA_TEXTS

# Ids that are not the well-behaved output of an encoder.
HOSTILE_IDS = [
    [],
    [0],
    [-1],
    [-999999],
    [10 ** 12],
    [2 ** 63],
    [999999999, 0, 999999999],
    [-1, 0, -2, 1],
]


def omnivoice_vocab_files():
    """Real voice vocabularies from the HuggingFace cache, when present."""
    root = (os.environ.get("HF_HOME")
            or os.path.join(os.environ.get("XDG_CACHE_HOME")
                            or os.path.expanduser("~/.cache"), "huggingface"))
    pattern = os.path.join(root, "hub", "models--*omnivoice*", "snapshots", "*",
                           "tokenizer.json")
    return sorted(glob.glob(pattern))


def simple_bpe(**extra):
    """A tiny BPE vocabulary, so a decoder can be exercised on its own."""
    letters = list("abcdefghijklmnopqrstuvwxyz .,'")
    vocab = {token: index for index, token in enumerate(letters)}
    spec = {
        "version": "1.0",
        "added_tokens": [],
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
    }
    spec.update(extra)
    return spec


class TestDecoderChainAgainstTokenizers(unittest.TestCase):
    """Character-identical output against the reference, or the test fails."""

    def setUp(self):
        if ReferenceTokenizer is None:
            self.skipTest("the `tokenizers` package is not installed")

    def _pair(self, path):
        return Tokenizer.from_file(path), ReferenceTokenizer.from_file(path)

    def _compare_round_trips(self, path):
        mine, reference = self._pair(path)
        checked = 0
        for text in ALL_TEXTS:
            ids = reference.encode(text).ids
            for skip in (True, False):
                self.assertEqual(
                    mine.decode(ids, skip_special_tokens=skip),
                    reference.decode(ids, skip_special_tokens=skip),
                    f"{os.path.basename(path)} disagrees on {text!r} "
                    f"(skip_special_tokens={skip})")
                checked += 1
        return checked

    def test_fixtures_decode_identically(self):
        paths = fixture_files()
        self.assertTrue(paths, "no fixtures found")
        for path in paths:
            with self.subTest(fixture=os.path.basename(path)):
                self.assertGreater(self._compare_round_trips(path), 0)

    def test_hostile_ids_decode_identically(self):
        """Ids the model never produced: unknown, negative, huge, empty."""
        for path in fixture_files():
            mine, reference = self._pair(path)
            size = reference.get_vocab_size()
            cases = HOSTILE_IDS + [[size], [size + 1000], [0, size, 1]]
            for ids in cases:
                with self.subTest(fixture=os.path.basename(path), ids=ids):
                    try:
                        expected = reference.decode(ids)
                    except (OverflowError, TypeError, ValueError):
                        # The reference refuses ids outside u32; the reader is
                        # only held to it where the reference answers.
                        continue
                    self.assertEqual(mine.decode(ids), expected)

    def test_special_tokens_are_kept_or_dropped_identically(self):
        """The added tokens themselves, decoded both ways."""
        for path in fixture_files():
            mine, reference = self._pair(path)
            base = mine.get_vocab(False)
            added = [i for token, i in mine.get_vocab().items()
                     if token not in base]
            ids = sorted(set(
                [i for i in range(min(20, reference.get_vocab_size()))]
                + [reference.get_vocab_size() - 1 - n for n in range(20)
                   if reference.get_vocab_size() - 1 - n >= 0]
                + added))
            for skip in (True, False):
                with self.subTest(fixture=os.path.basename(path), skip=skip):
                    self.assertEqual(mine.decode(ids, skip_special_tokens=skip),
                                     reference.decode(ids, skip_special_tokens=skip))

    def test_fuzzed_text_decodes_identically(self):
        rng = random.Random(20260812)
        for path in fixture_files():
            mine, reference = self._pair(path)
            for _ in range(200):
                text = "".join(rng.choice(FUZZ_ALPHABET)
                               for _ in range(rng.randint(0, 60)))
                ids = reference.encode(text).ids
                self.assertEqual(mine.decode(ids), reference.decode(ids),
                                 f"{os.path.basename(path)} disagrees on {text!r}")

    def test_fuzzed_id_sequences_decode_identically(self):
        """Random ids, not ids that came from real text."""
        rng = random.Random(1234)
        for path in fixture_files():
            mine, reference = self._pair(path)
            size = reference.get_vocab_size()
            for _ in range(200):
                ids = [rng.randrange(0, size + 50)
                       for _ in range(rng.randint(0, 30))]
                self.assertEqual(mine.decode(ids), reference.decode(ids), ids)

    def _compare_spec(self, spec, ids_pool):
        mine = Tokenizer(spec)
        reference = ReferenceTokenizer.from_str(json.dumps(spec))
        for ids in ids_pool:
            for skip in (True, False):
                self.assertEqual(mine.decode(ids, skip_special_tokens=skip),
                                 reference.decode(ids, skip_special_tokens=skip),
                                 f"{spec.get('decoder')} disagrees on {ids}")

    def test_each_supported_decoder_type_matches(self):
        """One synthetic vocabulary per decoder the reader claims to implement."""
        decoders = [
            None,
            {"type": "ByteLevel", "add_prefix_space": True,
             "trim_offsets": True, "use_regex": True},
            {"type": "Metaspace", "replacement": "▁",
             "prepend_scheme": "always", "split": False},
            {"type": "Metaspace", "replacement": "▁",
             "prepend_scheme": "never", "split": False},
            {"type": "WordPiece", "prefix": "##", "cleanup": True},
            {"type": "WordPiece", "prefix": "##", "cleanup": False},
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
            {"type": "Fuse"},
            {"type": "ByteFallback"},
            {"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"},
                 "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
                {"type": "Strip", "content": " ", "start": 1, "stop": 0},
            ]},
        ]
        # A vocabulary carrying every shape a decoder reacts to.
        tokens = ["hello", "▁world", "##ing", " leading", "n't", ".",
                  "<0xE4>", "<0xB8>", "<0xAD>", "<0xFF>", "Ġthe", "A"]
        vocab = {token: index for index, token in enumerate(tokens)}
        pool = [[], [0], [0, 1], [0, 1, 2, 3], [3, 4, 5],
                [6, 7, 8], [9], [6, 7, 8, 0], [10, 0], [11, 0, 1],
                list(range(len(tokens)))]
        for decoder in decoders:
            with self.subTest(decoder=decoder):
                spec = {"version": "1.0", "added_tokens": [],
                        "model": {"type": "BPE", "vocab": vocab, "merges": []},
                        "decoder": decoder}
                self._compare_spec(spec, pool)

    def test_special_token_flag_is_honoured_like_the_reference(self):
        """`special: false` added tokens survive `skip_special_tokens=True`."""
        vocab = {"a": 0, "b": 1}
        spec = {
            "version": "1.0",
            "added_tokens": [
                {"id": 2, "content": "<s>", "single_word": False,
                 "lstrip": False, "rstrip": False, "normalized": False,
                 "special": True},
                {"id": 3, "content": "[plain]", "single_word": False,
                 "lstrip": False, "rstrip": False, "normalized": False,
                 "special": False},
            ],
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
        }
        self._compare_spec(spec, [[0, 2, 1], [2, 3], [3], [2], [0, 3, 1]])

    def test_the_real_omnivoice_vocabulary_decodes_identically(self):
        """The vocabulary the live deployment actually loads."""
        paths = omnivoice_vocab_files()
        if not paths:
            self.skipTest("no omnivoice vocabulary in the local HuggingFace cache")
        for path in paths:
            with self.subTest(vocab=path):
                self.assertGreater(self._compare_round_trips(path), 0)


class TestDecodeWithoutReference(unittest.TestCase):
    """What decoding must do whether or not `tokenizers` is installed."""

    def _bytelevel_fixture(self):
        for path in fixture_files():
            with open(path, encoding="utf-8") as handle:
                spec = json.load(handle)
            if spec.get("decoder", {}) and spec["decoder"].get("type") == "ByteLevel":
                return path
        self.skipTest("no ByteLevel fixture")

    def test_text_survives_a_round_trip(self):
        tokenizer = Tokenizer.from_file(self._bytelevel_fixture())
        for text in ("hello world", "Hello, World! How are you today?",
                     "a  b   c", "the quick brown fox jumps over the lazy dog",
                     "punctuation, and: semicolons; too.",
                     "numbers 123 456 and 7.89"):
            with self.subTest(text=text):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text).ids),
                                 text)

    def test_an_empty_id_list_decodes_to_an_empty_string(self):
        for path in fixture_files():
            with self.subTest(fixture=os.path.basename(path)):
                self.assertEqual(Tokenizer.from_file(path).decode([]), "")

    def test_ids_outside_the_vocabulary_contribute_nothing(self):
        tokenizer = Tokenizer.from_file(self._bytelevel_fixture())
        ids = tokenizer.encode("hello world").ids
        size = tokenizer.get_vocab_size()
        noisy = [size + 5] + ids + [-1, size * 10]
        self.assertEqual(tokenizer.decode(noisy), tokenizer.decode(ids))

    def test_a_huge_id_does_not_raise(self):
        tokenizer = Tokenizer.from_file(self._bytelevel_fixture())
        self.assertEqual(tokenizer.decode([2 ** 70]), "")

    def test_skip_special_tokens_drops_them_and_false_keeps_them(self):
        path = self._bytelevel_fixture()
        tokenizer = Tokenizer.from_file(path)
        with open(path, encoding="utf-8") as handle:
            added = [t for t in json.load(handle)["added_tokens"]
                     if t.get("special")]
        if not added:
            self.skipTest("fixture declares no special tokens")
        special = added[0]
        ids = [special["id"]] + tokenizer.encode("hello world").ids
        self.assertNotIn(special["content"], tokenizer.decode(ids))
        kept = tokenizer.decode(ids, skip_special_tokens=False)
        self.assertIn(special["content"], kept)

    def test_an_unimplemented_decoder_is_refused_while_loading(self):
        """Refusing beats returning text that quietly does not match the audio.

        The refusal has to happen as the file is read, not at the first
        decode. Callers fall back to the ``tokenizers`` package when this
        reader declines a vocabulary, and that fallback wraps loading — a
        decoder that only raised on use would load, encode, and then fail in
        the middle of synthesis on a machine that could have decoded it.
        """
        spec = simple_bpe(decoder={"type": "CTC", "pad_token": "<pad>"})
        with self.assertRaises(UnsupportedTokenizer) as caught:
            Tokenizer(spec)
        message = str(caught.exception)
        self.assertIn("CTC", message)
        self.assertIn("tokenizers", message)

    def test_an_unimplemented_decoder_inside_a_sequence_is_refused(self):
        spec = simple_bpe(decoder={"type": "Sequence", "decoders": [
            {"type": "Fuse"}, {"type": "CTC", "pad_token": "<pad>"}]})
        with self.assertRaises(UnsupportedTokenizer):
            Tokenizer(spec)

    def test_a_regex_replace_decoder_is_refused_rather_than_approximated(self):
        r"""Python's `re` is not the Rust regex crate, and the difference is silent.

        The reference implementation inserts the replacement literally and
        accepts syntax Python does not have, so approximating it with `re`
        expands backreferences that should be literal and mis-reads patterns
        like ``\p{L}``. Wrong text is never noticed once it has been spoken.
        """
        spec = simple_bpe(decoder={"type": "Replace",
                                   "pattern": {"Regex": "([a-z]+)([0-9]+)"},
                                   "content": "\\2\\1"})
        with self.assertRaises(UnsupportedTokenizer) as caught:
            Tokenizer(spec)
        self.assertIn("tokenizers", str(caught.exception))

    def test_a_string_replace_decoder_is_still_accepted(self):
        # The refusal above must not take the pattern form real vocabularies use.
        spec = simple_bpe(decoder={"type": "Replace",
                                   "pattern": {"String": "_"},
                                   "content": " "})
        self.assertIsInstance(Tokenizer(spec), Tokenizer)

    def test_decode_is_deterministic(self):
        tokenizer = Tokenizer.from_file(self._bytelevel_fixture())
        ids = tokenizer.encode("hello world, hello again").ids
        self.assertEqual(tokenizer.decode(ids), tokenizer.decode(ids))


if __name__ == "__main__":
    unittest.main()
