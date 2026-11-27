"""The built-in ``tokenizer.model`` reader must agree with ``sentencepiece`` exactly.

A wrong id does not raise — it synthesises the wrong audio. So the reader is
held to identical output against the ids, pieces and text that the
``sentencepiece`` package gave for every model in tests/fixtures/. That
package is not a test dependency: scripts/make_sentencepiece_fixtures.py
trains the small models and writes sentencepiece_reference.json.

The Viterbi search has real ties on the Pocket TTS vocabularies: two
segmentations that score the same. SentencePiece breaks them by the order it
builds lattice nodes, so the reader must build them in the same order. The
reference holds tie strings and random text for that reason.
"""
import glob
import json
import os
import struct
import unittest

from phoonnx._sentencepiece import (SentencePieceProcessor,
                                    UnsupportedSentencePieceModel)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
with open(os.path.join(FIXTURES, "sentencepiece_reference.json"), encoding="utf-8") as _handle:
    REFERENCE = json.load(_handle)["models"]


def fixture_models():
    """The Pocket TTS vocabularies, one per language."""
    return sorted(glob.glob(os.path.join(FIXTURES, "tokenizer_sp_*.model")))


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

    def test_word_and_char_models_are_declined(self):
        self.assertIn("WORD", self._refuses(model_type=3))
        self.assertIn("CHAR", self._refuses(model_type=4))

    def test_a_malformed_charsmap_is_declined(self):
        self.assertIn("charsmap", self._refuses(charsmap=b"\x00\x01"))
        self.assertIn("charsmap", self._refuses(charsmap=b"\x40\x00\x00\x00\x01"))

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


class TestAgainstSentencePiece(unittest.TestCase):
    """Every id, piece and text the reader gives must equal the reference."""

    def test_every_fixture_model_has_a_reference(self):
        models = sorted(os.path.basename(path)
                        for path in glob.glob(os.path.join(FIXTURES, "*sp_*.model")))
        self.assertEqual(models, sorted(REFERENCE))
        for name in ("sp_bpe_nfkc.model", "sp_bpe_identity_no_byte_fallback.model",
                     "sp_unigram_identity.model"):
            self.assertIn(name, REFERENCE)

    def test_encode_matches(self):
        for name, reference in REFERENCE.items():
            processor = SentencePieceProcessor(os.path.join(FIXTURES, name))
            for row in reference["encode"]:
                with self.subTest(model=name, text=row["text"]):
                    self.assertEqual(processor.Encode(row["text"]), row["ids"])

    def test_decode_matches_on_encoded_text(self):
        for name, reference in REFERENCE.items():
            processor = SentencePieceProcessor(os.path.join(FIXTURES, name))
            for row in reference["encode"]:
                with self.subTest(model=name, text=row["text"]):
                    self.assertEqual(processor.Decode(row["ids"]), row["decoded"])

    def test_decode_matches_on_arbitrary_ids(self):
        """Pocket TTS decodes model output, not only what it encoded, so the
        control, unknown and byte pieces all have to come back right."""
        for name, reference in REFERENCE.items():
            processor = SentencePieceProcessor(os.path.join(FIXTURES, name))
            for row in reference["decode"]:
                with self.subTest(model=name, ids=row["ids"]):
                    self.assertEqual(processor.Decode(row["ids"]), row["text"])

    def test_pieces_match(self):
        for name, reference in REFERENCE.items():
            processor = SentencePieceProcessor(os.path.join(FIXTURES, name))
            self.assertEqual([processor.IdToPiece(i) for i in range(processor.GetPieceSize())],
                             reference["pieces"], name)

    def test_nfkc_folds_compatibility_characters(self):
        """A value check that needs no reference: nmt_nfkc reads the full-width
        and ligature forms as the plain letters."""
        processor = SentencePieceProcessor(os.path.join(FIXTURES, "sp_bpe_nfkc.model"))
        self.assertEqual(processor.Encode("ＦＵＬＬ ﬁne"), processor.Encode("FULL fine"))


if __name__ == "__main__":
    unittest.main()
