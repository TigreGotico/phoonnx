"""The built-in ``tokenizer.json`` reader must agree with ``tokenizers`` exactly.

A wrong id does not raise — it synthesises the wrong audio. So the reader is
held to bit-identical output against the reference implementation over every
fixture, and over any real voice vocab that happens to be in the local cache.
"""
import glob
import json
import os
import random
import re
import sys
import tempfile
import time
import unicodedata
import unittest

from phoonnx import _bpe
from phoonnx._bpe import Tokenizer, UnsupportedTokenizer
from phoonnx.tokenizer import BPETokenizer, load_hf_tokenizer, load_tokenizers_package

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
        return Tokenizer.from_file(path), load_tokenizers_package(path)

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
            self.assertIsInstance(load_hf_tokenizer(path),
                                  type(load_tokenizers_package(path)))


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
                self.assertIsInstance(load_hf_tokenizer(path),
                                      type(load_tokenizers_package(path)))

    def test_a_declined_vocab_encodes_through_tokenizers(self):
        """The fallback encodes and decodes on tokenizers 0.x and 1.x. Each
        character of this text is outside the vocabulary, so it becomes its
        UTF-8 bytes as ``<0xNN>`` tokens, whose ids start at 3, after ``<s>``
        (id 20001). tokenizers 0.23.3 gives the same ids."""
        path = os.path.join(FIXTURES, "declined_bpe_byte_fallback.json")
        tokenizer = BPETokenizer(path)
        ids = tokenizer.tokenize(list("Hello, World! ¾"))
        self.assertEqual(ids, [20001] + [3 + byte for byte in
                                         "▁Hello,▁World!▁¾".encode()])
        self.assertEqual(tokenizer.decode(ids), "▁Hello,▁World!▁¾")

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


class _Rc3Tokenizer:
    """tokenizers 1.0.0rc3: the Tokenizer itself encodes; there is no compile()."""

    @classmethod
    def from_file(cls, path):
        return cls()


class _Rc4Tokenizer(_Rc3Tokenizer):
    """tokenizers 1.0.0rc4: only what compile() returns encodes."""

    def compile(self):
        return "compiled"


@unittest.skipIf(ReferenceTokenizer is None, "the `tokenizers` package is not installed")
class TestTheLoaderFollowsTheApi(unittest.TestCase):
    """The release candidates of tokenizers 1.0 share the major version and
    differ in API, so a version check picks the wrong path on one of them."""

    def _load(self, stub, version):
        from unittest import mock

        import tokenizers
        with mock.patch.object(tokenizers, "Tokenizer", stub), \
                mock.patch.object(tokenizers, "__version__", version):
            return load_tokenizers_package("tokenizer.json")

    def test_a_tokenizer_without_compile_is_used_as_it_is(self):
        self.assertIsInstance(self._load(_Rc3Tokenizer, "1.0.0-rc.3"), _Rc3Tokenizer)

    def test_a_tokenizer_with_compile_is_compiled(self):
        self.assertEqual(self._load(_Rc4Tokenizer, "0.0.0"), "compiled")

    def test_a_declined_vocab_reaches_the_loader(self):
        """CI resolves a tokenizers 0.x, so this pins the 1.0 wiring there too."""
        from unittest import mock

        import tokenizers
        path = os.path.join(FIXTURES, "declined_bpe_byte_fallback.json")
        with mock.patch.object(tokenizers, "Tokenizer", _Rc4Tokenizer):
            self.assertEqual(load_hf_tokenizer(path), "compiled")


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
        mine, reference = Tokenizer.from_file(path), load_tokenizers_package(path)
        for length in (200, 1000, 4000):
            text = (self.CHINESE * (length // len(self.CHINESE) + 1))[:length]
            with self.subTest(length=length):
                self.assertEqual(mine.encode(text).ids, reference.encode(text).ids)


# Added tokens that overlap, prefix each other, hold regex syntax, or sit
# where a pre-tokenizer would cut them.
OVERLAPPING_ADDED = [
    "<|", "<|a|>", "<|a|>b", "<|speech_1|>", "<|speech_12|>", "<|speech_123|>",
    "[S]", "[SPACE]", "[SPACE]x", "ab", "abc", "b c", "-", "^x", "]", "\\",
    ".*", "世界", "界", "🌍x", " e",
]

ADDED_TEXTS = TEXTS + [
    "<|", "<|a|", "<|a|>", "<|a|>b", "<|a|><|a|>b<|", "<|speech_1234|>",
    "<|speech_12|>3", "[SPACE][SPACE]x[S]", "abcab cabc", "a b c", "世界界",
    "🌍x🌍", "^x^", "\\]-.*", " e e", "<|<|<|a|>", "x<|endoftext|>y",
]

ADDED_ALPHABET = "<|>a_bcspeh123[SPACEx]^\\.*-世界🌍 "


def _regex_split(added, text):
    """The alternation the reader used to cut added tokens out of text."""
    pattern = "(" + "|".join(sorted(map(re.escape, added), key=len, reverse=True)) + ")"
    return re.split(pattern, text)


def _with_added(path, contents, special=()):
    """A fixture spec with *contents* appended as added tokens, numbered as
    `tokenizers` numbers them: a token the model vocab holds keeps its id,
    and a new one takes the next free id."""
    with open(path, encoding="utf-8") as handle:
        spec = json.load(handle)
    vocab = spec["model"]["vocab"]
    next_id = max(list(vocab.values()) + [t["id"] for t in spec["added_tokens"]]) + 1
    for content in contents:
        if content in vocab:
            token_id = vocab[content]
        else:
            token_id, next_id = next_id, next_id + 1
        spec["added_tokens"].append({
            "id": token_id, "content": content, "single_word": False,
            "lstrip": False, "rstrip": False, "normalized": False,
            "special": content in special})
    return spec


class TestAddedTokens(unittest.TestCase):
    """Added tokens are cut out of the text before anything else: leftmost
    first, and the longest where several start at the same place."""

    def _spec(self, contents=OVERLAPPING_ADDED, special=("<|a|>", "[S]")):
        return _with_added(os.path.join(FIXTURES, "tokenizer_bpe_qwen.json"),
                           contents, special)

    def _fuzz(self, count=600):
        rng = random.Random(20261118)
        return ["".join(rng.choice(ADDED_ALPHABET) for _ in range(rng.randint(0, 40)))
                for _ in range(count)]

    def test_the_cut_equals_the_longest_first_alternation(self):
        spec = self._spec()
        mine = Tokenizer(spec)
        added = [t["content"] for t in spec["added_tokens"]]
        for text in ADDED_TEXTS + self._fuzz():
            self.assertEqual(mine._split_added(text), _regex_split(added, text),
                             f"cut disagrees on {text!r}")

    def test_an_empty_added_token_is_ignored(self):
        """`tokenizers` ignores an added token with no content."""
        path = os.path.join(FIXTURES, "tokenizer_bpe_whitespace.json")
        plain = _with_added(path, [])
        expected = [6563, 255, 62, 84, 28, 2, 179, 79, 0, 6561, 6561]
        self.assertEqual(Tokenizer(plain).encode("hello world").ids, expected)
        self.assertEqual(Tokenizer(_with_added(path, [""])).encode("hello world").ids,
                         expected)

    def test_a_vocab_without_added_tokens_keeps_the_text_whole(self):
        spec = self._spec()
        spec["added_tokens"] = []
        mine = Tokenizer(spec)
        for text in ("", "<|a|>", "hello"):
            self.assertEqual(mine._split_added(text), [text])

    @unittest.skipIf(ReferenceTokenizer is None, "the `tokenizers` package is not installed")
    def test_overlapping_added_tokens_encode_as_the_reference_does(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "tokenizer.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self._spec(), handle)
            mine, reference = Tokenizer.from_file(path), load_tokenizers_package(path)
            for text in ADDED_TEXTS + self._fuzz(300):
                for add_special in (True, False):
                    self.assertEqual(
                        mine.encode(text, add_special_tokens=add_special).ids,
                        reference.encode(text, add_special_tokens=add_special).ids,
                        f"disagrees on {text!r}, add_special_tokens={add_special}")
                self.assertEqual(mine.decode(mine.encode(text).ids),
                                 reference.decode(reference.encode(text).ids))

    def test_65536_speech_tokens(self):
        """The NeuTTS and Llasa shape: one added token per codec code. The
        `tokenizers` 1.0 release candidates panic on this many, so the ids are
        checked against the codes they encode."""
        spec = self._spec([f"<|speech_{code}|>" for code in range(65536)])
        first = spec["added_tokens"][-65536]["id"]
        mine = Tokenizer(spec)
        codes = random.Random(7).choices(range(65536), k=200)
        prompt = "".join(f"<|speech_{code}|>" for code in codes)
        self.assertEqual(mine.encode(prompt, add_special_tokens=False).ids,
                         [first + code for code in codes])
        added = [t["content"] for t in spec["added_tokens"]]
        for text in ("user: hello<|speech_1|><|speech_65535|>world<|speech_",
                     "<|speech_655360|><|speech_|><|speech_00|><|speech_6553|>5"):
            self.assertEqual(mine._split_added(text), _regex_split(added, text))


FLAG_TOKENS = ["<mask>", "<MASK>", "[SP]", "a b", " y", "z ", "qx"]
FLAG_ALPHABET = list("<mask>MASKab xyzq[]P-_1\t\u00a0\n")


def _flagged(path, flags, normalizer=None):
    """A fixture spec whose added tokens (`FLAG_TOKENS`) all carry *flags*."""
    spec = _with_added(path, FLAG_TOKENS)
    for token in spec["added_tokens"][-len(FLAG_TOKENS):]:
        token.update(flags)
    if normalizer:
        spec["normalizer"] = normalizer
    return spec


class TestAddedTokenFlags(unittest.TestCase):
    """``lstrip``, ``rstrip``, ``single_word`` and ``normalized`` of an added token."""

    def _tokenizer(self, **flags):
        path = os.path.join(FIXTURES, "tokenizer_bpe_qwen.json")
        return Tokenizer(_flagged(path, flags))

    def _ids(self, tokenizer, text):
        return tokenizer.encode(text, add_special_tokens=False).ids

    def test_lstrip_takes_the_white_space_before_the_token(self):
        mine = self._tokenizer(lstrip=True)
        token = mine.token_to_id("[SP]")
        self.assertEqual(self._ids(mine, "a  \t[SP]b"), self._ids(mine, "a") + [token] + self._ids(mine, "b"))
        self.assertEqual(self._ids(mine, "a [SP]"), self._ids(mine, "a") + [token])

    def test_rstrip_takes_the_white_space_after_the_token(self):
        mine = self._tokenizer(rstrip=True)
        token = mine.token_to_id("[SP]")
        self.assertEqual(self._ids(mine, "a[SP] \n b"), self._ids(mine, "a") + [token] + self._ids(mine, "b"))

    def test_single_word_refuses_a_match_inside_a_word(self):
        mine = self._tokenizer(single_word=True)
        token = mine.token_to_id("[SP]")
        self.assertNotIn(token, self._ids(mine, "a[SP]"))
        self.assertNotIn(token, self._ids(mine, "[SP]a"))
        self.assertEqual(self._ids(mine, "a [SP] b"),
                         self._ids(mine, "a ") + [token] + self._ids(mine, " b"))

    def test_a_token_without_normalized_is_not_matched_after_the_normalizer(self):
        path = os.path.join(FIXTURES, "tokenizer_bpe_qwen.json")
        lowercase = {"type": "Lowercase"}
        raw = Tokenizer(_flagged(path, {"normalized": False}, lowercase))
        normalized = Tokenizer(_flagged(path, {"normalized": True}, lowercase))
        token = raw.token_to_id("<mask>")
        self.assertNotIn(token, self._ids(raw, "<Mask>"))
        self.assertEqual(self._ids(normalized, "<Mask>"), [normalized.token_to_id("<MASK>")])

    @unittest.skipIf(ReferenceTokenizer is None, "the `tokenizers` package is not installed")
    def test_every_flag_encodes_as_the_reference_does(self):
        rng = random.Random(777)
        combos = [{"lstrip": True}, {"rstrip": True}, {"lstrip": True, "rstrip": True},
                  {"single_word": True}, {"normalized": True},
                  {"normalized": True, "single_word": True, "lstrip": True, "rstrip": True}]
        bases = [("tokenizer_bpe_qwen.json", None),
                 ("tokenizer_bpe_qwen.json", {"type": "Lowercase"}),
                 ("tokenizer_bpe_whitespace.json", None)]
        for name, normalizer in bases:
            path = os.path.join(FIXTURES, name)
            for flags in combos:
                with tempfile.TemporaryDirectory() as folder:
                    target = os.path.join(folder, "tokenizer.json")
                    with open(target, "w", encoding="utf-8") as handle:
                        json.dump(_flagged(path, flags, normalizer), handle)
                    mine, reference = Tokenizer.from_file(target), load_tokenizers_package(target)
                    for _ in range(120):
                        pieces = [rng.choice(FLAG_ALPHABET) if rng.random() < .5 else rng.choice(FLAG_TOKENS)
                                  for _ in range(rng.randint(0, 8))]
                        text = "".join(pieces)
                        for add_special in (True, False):
                            self.assertEqual(
                                mine.encode(text, add_special_tokens=add_special).ids,
                                reference.encode(text, add_special_tokens=add_special).ids,
                                f"{name} {normalizer} {flags} disagrees on {text!r}")


class TestUnicodeProperties(unittest.TestCase):
    """`re` has no \\p{..}, so the reader writes a property out as ranges."""

    @classmethod
    def setUpClass(cls):
        cls.everything = "".join(map(chr, range(sys.maxunicode + 1)))
        cls.by_category = {}
        for char in cls.everything:
            cls.by_category.setdefault(unicodedata.category(char), []).append(char)

    def test_each_property_is_its_code_points_written_as_ranges(self):
        """The class text is the property's code points, consecutive ones
        merged into a range: the text a scan of every code point writes."""
        majors = sorted({category[0] for category in self.by_category})
        for prop in majors + sorted(self.by_category):
            ranges = []
            for category, chars in self.by_category.items():
                if category.startswith(prop) if len(prop) == 1 else category == prop:
                    ranges += [[ord(char), ord(char)] for char in chars]
            merged = []
            for low, high in sorted(ranges):
                if merged and merged[-1][1] == low - 1:
                    merged[-1][1] = high
                else:
                    merged.append([low, high])
            expected = "".join(
                re.escape(chr(low)) if low == high
                else f"{re.escape(chr(low))}-{re.escape(chr(high))}"
                for low, high in merged)
            with self.subTest(prop=prop):
                self.assertEqual(_bpe._category_ranges(prop), expected)

    def test_one_table_of_category_runs_serves_every_property(self):
        runs = _bpe._category_runs()
        self.assertIs(_bpe._category_runs(), runs)
        self.assertEqual((runs[0][1], runs[-1][2]), (0, sys.maxunicode))
        for (category, low, high), (following, next_low, _) in zip(runs, runs[1:]):
            self.assertEqual(next_low, high + 1)
            self.assertNotEqual(category, following)
        for category, low, high in runs:
            self.assertEqual(unicodedata.category(chr(low)), category)
            self.assertEqual(unicodedata.category(chr(high)), category)
