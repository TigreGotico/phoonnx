"""The built-in ``tokenizer.model`` reader must agree with ``sentencepiece`` exactly.

A wrong id does not raise — it synthesises the wrong audio. So the reader is
held to identical output against the reference implementation over every
Pocket TTS vocabulary, and over any real voice model in the local cache.

The Viterbi search has real ties on these vocabularies: two segmentations that
score the same. SentencePiece breaks them by the order it builds lattice
nodes, so the reader must build them in the same order. The fuzz tests below
are what proves it.
"""
import glob
import os
import random
import struct
import unittest

from phoonnx._sentencepiece import (SentencePieceProcessor,
                                    UnsupportedSentencePieceModel)

try:
    import sentencepiece as reference_sentencepiece
except ImportError:
    reference_sentencepiece = None

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

TEXTS = [
    "", " ", "  ", "   ", "\n", "\t", "\r\n", "a",
    "hello world", "Hello, World! How are you today?",
    "a  b   c", "text with     five spaces",
    "  leading and trailing  ",
    "new\nline\ttab\r\ncrlf",
    "O rato roeu a roupa do rei de Roma.",
    "L'élève va à l'école, n'est-ce pas ?",
    "¿Qué tal? ¡Muy bien, gracias!",
    "Ich weiß, dass zwei plus zwei vier ist.",
    "Perché no? La città è bellissima!",
    "olá, tudo bem? çãõ ñ àéîõü",
    "The quick brown fox jumps over the lazy dog 12345.",
    "MiXeD CaSe UPPER lower",
    "123 4567 0.5 -3 1,000,000 42%",
    "sub-word hyphen_underscore (parens) [brackets] {braces} <angle>",
    "...!!!???;;;:::",
    "ΑΒΓ αβγ ⟨angle⟩ ½ ¾ № — – ’ “ ”",
    # certainly outside a Latin-script vocabulary: the unknown/byte path
    "мир 世界 한국어 🌍 emoji test",
    "الْعَرَبِيَّة مرحبا بالعالم",
    "你好世界", "😀🎉🇵🇹",
    # the escaped-space character itself, typed by a user
    "▁", "▁▁", "hallo▁welt",
    "a" * 200,
]

# Inputs whose lattice has two equally scoring paths, found by running the
# reader against itself with the tie-break flipped. SentencePiece keeps the
# left-most node — the one that starts earliest, so the longest piece ending
# where the tie is — and these strings are the proof that the reader does the
# same. Flip the comparison in `_viterbi` and they change ids.
TIE_TEXTS = [
    "ü1'\"ob0Ha\"Lñ kQuAc.ßLbbb",
    "ãzñqm5pppliNínöYßbZ6U",
    "ú-SyHAßúKgUa'f\"k9ú úlll",
]

FUZZ_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    " \t\n\r.,!?;:'\"-_()[]{}<>/@#$%^&*+=~`|\\"
    "áéíóúñçãõäöüßΑΒΓαβγмир שלום العربية 世界日本語한국어🌍🎉½№’—▁"
)


def fixture_models():
    """The Pocket TTS vocabularies, one per language."""
    return sorted(glob.glob(os.path.join(FIXTURES, "tokenizer_sp_*.model")))


def declined_models():
    """Models the reader must refuse rather than guess at."""
    return sorted(glob.glob(os.path.join(FIXTURES, "declined_sp_*.model")))


def voice_cache_models():
    """Real voice vocabularies, when this machine has downloaded any."""
    roots = [os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")]
    found = []
    for root in roots:
        found += glob.glob(os.path.join(root, "phoonnx", "**", "tokenizer.model"),
                           recursive=True)
        found += glob.glob(os.path.join(root, "huggingface", "hub",
                                        "*pocket-tts*", "**", "tokenizer.model"),
                           recursive=True)
    return sorted(set(found))


def _varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _tag(number, wire):
    return _varint((number << 3) | wire)


def _bytes_field(number, payload):
    return _tag(number, 2) + _varint(len(payload)) + payload


def _varint_field(number, value):
    return _tag(number, 0) + _varint(value)


def _float_field(number, value):
    return _tag(number, 5) + struct.pack("<f", value)


def build_model(model_type=1, normalizer_name="identity", charsmap=b"",
                remove_extra_whitespaces=False, escape_whitespaces=True,
                treat_whitespace_as_suffix=False, byte_fallback=False,
                byte_pieces=True):
    """Hand-build a ModelProto with one knob turned, to test the refusals.

    Written by hand rather than trained, because the point of each fixture is
    one flag, and a trained model would carry a 200 kB normalizer table with
    it.
    """
    pieces = [_bytes_field(1, _bytes_field(1, b"<unk>") + _float_field(2, 0.0)
                           + _varint_field(3, 2))]
    for text, score in ((b"\xe2\x96\x81hello", -1.0), (b"a", -3.0), (b"b", -4.0)):
        pieces.append(_bytes_field(1, _bytes_field(1, text)
                                   + _float_field(2, score) + _varint_field(3, 1)))
    if byte_pieces:
        for value in range(256):
            name = ("<0x%02X>" % value).encode("utf-8")
            pieces.append(_bytes_field(1, _bytes_field(1, name)
                                       + _float_field(2, 0.0)
                                       + _varint_field(3, 6)))
    trainer = (_varint_field(3, model_type)
               + _varint_field(24, int(treat_whitespace_as_suffix))
               + _varint_field(35, int(byte_fallback)))
    normalizer = (_bytes_field(1, normalizer_name.encode("utf-8"))
                  + _bytes_field(2, charsmap)
                  + _varint_field(3, 1)
                  + _varint_field(4, int(remove_extra_whitespaces))
                  + _varint_field(5, int(escape_whitespaces)))
    return (b"".join(pieces) + _bytes_field(2, trainer)
            + _bytes_field(3, normalizer))


class TestFixturesPresent(unittest.TestCase):
    def test_every_pocket_tts_language_is_covered(self):
        """Six languages ship, and each one has its own vocabulary."""
        self.assertGreaterEqual(len(fixture_models()), 6, fixture_models())

    def test_the_fixtures_are_real_sized_vocabularies(self):
        """The shipped models hold 4000 pieces. A toy vocabulary shares none of
        their lattice depth, so CI must carry something the same shape."""
        for path in fixture_models():
            processor = SentencePieceProcessor(path)
            self.assertGreaterEqual(processor.GetPieceSize(), 4000, path)


class TestModelFacts(unittest.TestCase):
    """What the shipped models declare, read straight out of the file."""

    def test_the_fixtures_use_byte_fallback(self):
        for path in fixture_models():
            processor = SentencePieceProcessor(path)
            self.assertTrue(processor._byte_fallback, path)
            self.assertEqual(len(processor._byte_ids), 256, path)

    def test_the_fixtures_add_a_dummy_prefix(self):
        for path in fixture_models():
            self.assertTrue(SentencePieceProcessor(path)._add_dummy_prefix, path)


class TestRefusals(unittest.TestCase):
    """Anything the reader cannot do exactly, it must decline out loud."""

    def _refuses(self, **kwargs):
        processor = SentencePieceProcessor()
        with self.assertRaises(UnsupportedSentencePieceModel) as caught:
            processor.LoadFromSerializedProto(build_model(**kwargs))
        return str(caught.exception)

    def test_bpe_models_are_declined(self):
        self.assertIn("BPE", self._refuses(model_type=2))

    def test_word_and_char_models_are_declined(self):
        self.assertIn("WORD", self._refuses(model_type=3))
        self.assertIn("CHAR", self._refuses(model_type=4))

    def test_a_trained_bpe_model_is_declined(self):
        """The synthetic fixtures turn one flag; this one came off the trainer."""
        for path in declined_models():
            with self.assertRaises(UnsupportedSentencePieceModel):
                SentencePieceProcessor(path)

    def test_a_non_identity_normalizer_is_declined(self):
        message = self._refuses(normalizer_name="nmt_nfkc",
                                charsmap=b"\x00\x01\x02\x03")
        self.assertIn("nmt_nfkc", message)

    def test_a_compiled_charsmap_is_declined_even_when_named_identity(self):
        self.assertIn("charsmap", self._refuses(charsmap=b"\x00\x01"))

    def test_remove_extra_whitespaces_is_declined(self):
        self.assertIn("remove_extra_whitespaces",
                      self._refuses(remove_extra_whitespaces=True))

    def test_unescaped_whitespace_is_declined(self):
        self.assertIn("escape_whitespaces",
                      self._refuses(escape_whitespaces=False))

    def test_whitespace_as_suffix_is_declined(self):
        self.assertIn("treat_whitespace_as_suffix",
                      self._refuses(treat_whitespace_as_suffix=True))

    def test_byte_fallback_without_byte_pieces_is_declined(self):
        self.assertIn("byte", self._refuses(byte_fallback=True,
                                            byte_pieces=False))

    def test_a_supported_model_loads(self):
        processor = SentencePieceProcessor()
        processor.LoadFromSerializedProto(build_model())
        self.assertEqual(processor.Encode("hello"), [1])


@unittest.skipIf(reference_sentencepiece is None,
                 "the `sentencepiece` package is not installed")
class TestDifferentialAgainstSentencePiece(unittest.TestCase):
    """Every id the reader produces must equal the reference id."""

    def _pair(self, path):
        reference = reference_sentencepiece.SentencePieceProcessor()
        reference.Load(path)
        return SentencePieceProcessor(path), reference

    def _paths(self):
        return fixture_models() + voice_cache_models()

    def test_fixtures_encode_identically(self):
        paths = self._paths()
        self.assertTrue(paths, "no models found")
        for path in paths:
            mine, reference = self._pair(path)
            for text in TEXTS:
                with self.subTest(model=os.path.basename(path), text=text):
                    self.assertEqual(mine.Encode(text), reference.Encode(text))

    def test_fixtures_survive_random_text(self):
        """Fuzz: the shapes a user actually types are not the only shapes.

        This is also the tie hunt. A tie the reader breaks the other way shows
        up here as a different id sequence for the same text.
        """
        rng = random.Random(20260814)
        for path in self._paths():
            mine, reference = self._pair(path)
            for _ in range(600):
                text = "".join(rng.choice(FUZZ_ALPHABET)
                               for _ in range(rng.randint(0, 60)))
                self.assertEqual(mine.Encode(text), reference.Encode(text),
                                 f"{os.path.basename(path)} disagrees "
                                 f"on {text!r}")

    def test_viterbi_ties_break_the_same_way(self):
        """Equal-scoring segmentations must be resolved the reference's way."""
        for path in self._paths():
            mine, reference = self._pair(path)
            for text in TIE_TEXTS:
                with self.subTest(model=os.path.basename(path), text=text):
                    self.assertEqual(mine.Encode(text), reference.Encode(text))

    def test_decode_matches_on_encoded_text(self):
        for path in self._paths():
            mine, reference = self._pair(path)
            for text in TEXTS:
                ids = reference.Encode(text)
                with self.subTest(model=os.path.basename(path), text=text):
                    self.assertEqual(mine.Decode(ids), reference.Decode(ids))

    def test_decode_matches_on_arbitrary_ids(self):
        """Pocket TTS decodes model output, not only what it encoded, so the
        control, unknown and byte pieces all have to come back right."""
        rng = random.Random(20260814)
        for path in self._paths():
            mine, reference = self._pair(path)
            size = mine.GetPieceSize()
            sequences = [[rng.randrange(size) for _ in range(rng.randint(1, 25))]
                         for _ in range(400)]
            # Deliberate shapes: a control token first, a byte token first, an
            # unknown token first, a lone byte of a multi-byte character.
            sequences += [[1, 2, 3], [0], [0, 4], [4, 5, 6], [3, 4],
                          [0xE4 % size]]
            for ids in sequences:
                self.assertEqual(mine.Decode(ids), reference.Decode(ids),
                                 f"{os.path.basename(path)} disagrees "
                                 f"on {ids}")

    def test_round_trip_where_sentencepiece_round_trips(self):
        """Decode(Encode(x)) == x wherever the reference manages it."""
        for path in self._paths():
            mine, reference = self._pair(path)
            for text in TEXTS:
                if reference.Decode(reference.Encode(text)) != text:
                    continue
                with self.subTest(model=os.path.basename(path), text=text):
                    self.assertEqual(mine.Decode(mine.Encode(text)), text)

    def test_pieces_match(self):
        """Same ids means same pieces, but the mapping is worth pinning too."""
        for path in self._paths():
            mine, reference = self._pair(path)
            self.assertEqual(mine.GetPieceSize(), reference.GetPieceSize(), path)
            for index in range(mine.GetPieceSize()):
                self.assertEqual(mine.IdToPiece(index),
                                 reference.IdToPiece(index), path)


if __name__ == "__main__":
    unittest.main()
