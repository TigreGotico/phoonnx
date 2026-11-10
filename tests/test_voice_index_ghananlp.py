"""The three GhanaNLP community voices, and the fields they cannot lose.

Each of these entries carries a correction. The upstream repositories name a
front end or ship a file layout that does not work, so an entry that loses one
of these fields loads a voice that runs and says the wrong thing, which no
schema check can see.
"""
import json
import unittest
from pathlib import Path

from phoonnx.index_schema import validate_index_file
from phoonnx.model_manager import TTSModelManager

INDEX = "ghananlp.json"


def _entries():
    path = TTSModelManager.voice_index_path() / INDEX
    return json.loads(path.read_text(encoding="utf-8"))


class TestGhanaNlpIndex(unittest.TestCase):

    def test_the_file_is_valid(self):
        path = TTSModelManager.voice_index_path() / INDEX
        self.assertEqual(validate_index_file(path), [])

    def test_the_file_is_in_the_merge_order(self):
        # Not merely present on disk: a file left out of the order is appended
        # alphabetically, so the order is where the intent is recorded.
        self.assertIn(INDEX, TTSModelManager._VOICE_INDEX_ORDER)
        self.assertIn(TTSModelManager.voice_index_path() / INDEX,
                      TTSModelManager.voice_index_files())

    def test_every_voice_is_reachable_by_source(self):
        by_source = TTSModelManager().get_available_voice_ids_by_source()
        self.assertEqual(sorted(by_source["ghananlp"]), sorted(_entries()))

    def test_the_twi_voices_phonemize_through_lingua_franca_nova(self):
        # espeak serves no Twi voice. lfn is the proxy this model family was
        # trained with, and it measured best of the three front ends tried.
        # Dropping this field falls back to what the config names, which is
        # en-us for stable-twi-tts and measured worst by a wide margin.
        for voice_id in ("ghananlpcommunity/stable-twi-tts",
                         "ghananlpcommunity/nano-twi"):
            entry = _entries()[voice_id]
            self.assertEqual(entry["lang"], "tw")
            self.assertEqual(entry["phonemizer_lang"], "lfn")
            self.assertEqual(entry["phoneme_type"], "espeak")

    def test_nano_twi_carries_its_vocoder(self):
        # Matcha is two-stage. Without the vocoder the entry loads a model
        # that returns a mel and never becomes audio.
        entry = _entries()["ghananlpcommunity/nano-twi"]
        self.assertEqual(entry["engine"], "matcha")
        self.assertEqual(entry["vocoder_type"], "vocos")
        self.assertTrue(entry["vocoder_url"].endswith("vocoder.onnx"))

    def test_kokoro_names_one_speaker_style_not_the_packed_file(self):
        # The upstream voices.bin holds all 53 speakers in one file. phoonnx
        # reshapes a style file to [-1, 256] and indexes by token count, so the
        # packed file silently reaches speaker 0 only. style_url must name a
        # single split style.
        entry = _entries()["ghananlpcommunity/poto-tts-kokoro-gh"]
        self.assertEqual(entry["engine"], "styletts2")
        self.assertIn("/voices/", entry["style_url"])
        self.assertTrue(entry["style_url"].endswith(".bin"))
        self.assertNotIn("voices.bin", entry["style_url"])

    def test_kokoro_reads_the_british_phoneme_table(self):
        # Upstream compiles its Ghanaian dictionary against espeak's British
        # table and every speaker it recommends is British. The config on the
        # mirror says en-us; this override is what takes effect.
        entry = _entries()["ghananlpcommunity/poto-tts-kokoro-gh"]
        self.assertEqual(entry["phonemizer_lang"], "en-gb")

    def test_every_url_points_at_our_own_mirror(self):
        # The upstream repositories are the source, never the download: their
        # files are the ones with the faults these entries correct.
        for voice_id, entry in _entries().items():
            for key, value in entry.items():
                if key.endswith("_url") and value:
                    self.assertIn("huggingface.co/OpenVoiceOS/", value,
                                  f"{voice_id}.{key} does not point at our mirror")
