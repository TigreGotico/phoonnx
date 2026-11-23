"""The segment phoneme map: phoonnx_train.segments and preprocess --segment-map."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from phoonnx.tokenizer import DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_PAD_TOKEN, DEFAULT_BLANK_WORD_TOKEN
from phoonnx_train.segments import segment_phonemes
from tests.test_preprocess_pipeline import _invoke, _jsonl

SPECIALS = {DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BLANK_WORD_TOKEN}
# segments written out by hand: the expected value is not computed by the code under test
TEXT = "ˈtʲaː ɐ̃d͡ʒa, ma˥˩."
SEGMENTS = ["ˈ", "tʲ", "aː", " ", "ɐ̃", "d͡ʒ", "a", ",", " ", "m", "a", "˥˩", "."]
# the stub frontend drops punctuation, and the space is the word-blank special
CLI_TEXT = "ˈtʲaː ɐ̃d͡ʒa ma˥˩"
CLI_SEGMENTS = ["ˈ", "tʲ", "aː", "ɐ̃", "d͡ʒ", "a", "m", "a", "˥˩"]


class TestSegmentPhonemes(unittest.TestCase):
    def test_marks_join_their_base_and_prosody_stands_alone(self):
        self.assertEqual(segment_phonemes(TEXT), SEGMENTS)

    def test_decomposed_nasal_vowel_is_one_segment_as_written(self):
        self.assertEqual(segment_phonemes("ẽj"), ["ẽ", "j"])
        self.assertEqual(segment_phonemes("ẽj"), ["ẽ", "j"])

    def test_tone_diacritic_stays_on_its_vowel(self):
        self.assertEqual(segment_phonemes("má"), ["m", "á"])

    def test_stress_and_syllable_dot_do_not_join(self):
        self.assertEqual(segment_phonemes("ˌa.ˈb"), ["ˌ", "a", ".", "ˈ", "b"])

    def test_tie_bar_joins_three_symbols(self):
        self.assertEqual(segment_phonemes("k͡pʷa"), ["k͡pʷ", "a"])

    def test_a_leading_mark_is_kept(self):
        self.assertEqual(segment_phonemes("̃a"), ["̃", "a"])

    def test_segments_join_back_to_the_input(self):
        for text in (TEXT, "ɐ̃ˈmaɲɐ ˈvɔw ɐ ˈkazɐ.", "tʃʰa˧˥ ʔa˨˩˦"):
            self.assertEqual("".join(segment_phonemes(text)), text)


class TestSegmentMapCli(unittest.TestCase):
    def _run(self, tmp: Path, *extra: str):
        src = tmp / "a.jsonl"
        _jsonl(src, [{"text": CLI_TEXT, "audio": "a.wav"}])
        return _invoke(["-i", str(src), "-o", str(tmp / "out"), "-l", "pt", "--skip-audio",
                        "--phoneme-type", "graphemes", "--alphabet", "ipa", *extra])

    def test_the_map_holds_the_segments_and_the_ids_read_them_back(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            result = self._run(tmp, "--segment-map")
            self.assertEqual(result.exit_code, 0, result.output)
            config = json.loads((tmp / "out" / "config.json").read_text(encoding="utf-8"))
            id_map = config["phoneme_id_map"]
            self.assertEqual(set(id_map) - SPECIALS, set(CLI_SEGMENTS))
            self.assertIs(config["fold_compounds"], True)

            row = json.loads((tmp / "out" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()[0])
            symbol = {i: s for s, i in id_map.items()}
            self.assertEqual([symbol[i] for i in row["phoneme_ids"] if symbol[i] not in SPECIALS], CLI_SEGMENTS)

    def test_without_the_flag_the_map_stays_per_code_point(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            result = self._run(tmp, "--corpus-only-map")
            self.assertEqual(result.exit_code, 0, result.output)
            config = json.loads((tmp / "out" / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(set(config["phoneme_id_map"]) - SPECIALS, set(CLI_TEXT) - SPECIALS)
            self.assertNotIn("fold_compounds", config)

    def test_segment_map_refuses_resume(self):
        with TemporaryDirectory() as tmp:
            result = self._run(Path(tmp), "--segment-map", "--resume")
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("--resume cannot be combined", result.output)


if __name__ == "__main__":
    unittest.main()
