"""The three kasanoma voices, and the fields they cannot lose.

No kasanoma config names the language it speaks: ``language`` is present and
null in all three, and ``espeak.voice`` is ``lfn``. So the language a caller
asks for comes from the index or from nowhere, and ``lang`` here is the only
record of which voice speaks Twi, Chichewa or Makhuwa.

The model files come from one repository per voice under ``Phoonnx``.
Upstream publishes two of the three as a zip on a GitHub release only, which
phoonnx cannot read, and the third from an account that carries no licence
file. Those repositories are the one stable url for all three.
"""
import json
import re
import unittest

from phoonnx.index_schema import validate_index_file
from phoonnx.model_manager import TTSModelManager

INDEX = "piper_community.json"

# voice id -> the language tag, and the repository of the voice
EXPECTED = {
    "piper_community/ghananlpcommunity/kasanoma-twi": ("tw", "vits-tw-kasanoma-espeak"),
    "piper_community/michsethowusu/kasanoma-chichewa": ("ny", "vits-ny-kasanoma-espeak"),
    "piper_community/michsethowusu/kasanoma-makhuwa": ("vmw", "vits-vmw-kasanoma-espeak"),
}


def _entries():
    path = TTSModelManager.voice_index_path() / INDEX
    return json.loads(path.read_text(encoding="utf-8"))


class TestKasanomaIndex(unittest.TestCase):

    def test_the_file_is_valid(self):
        path = TTSModelManager.voice_index_path() / INDEX
        self.assertEqual(validate_index_file(path), [])

    def test_all_three_voices_are_in_the_catalog(self):
        manager = TTSModelManager()
        manager.merge_default_voices()
        missing = [v for v in EXPECTED if manager.get_voice(v) is None]
        self.assertEqual([], missing)

    def test_each_one_names_the_language_its_config_does_not(self):
        # The config says language: null. Lose lang here and the voice speaks
        # into a language nothing asked for.
        entries = _entries()
        for voice_id, (lang, _) in EXPECTED.items():
            with self.subTest(voice_id):
                self.assertEqual(entries[voice_id]["lang"], lang)

    def test_each_one_phonemizes_through_lingua_franca_nova(self):
        # espeak serves none of these three languages. lfn is the front end
        # the models were trained with (espeak.voice in every config).
        entries = _entries()
        for voice_id in EXPECTED:
            with self.subTest(voice_id):
                entry = entries[voice_id]
                self.assertEqual(entry["phonemizer_lang"], "lfn")
                self.assertEqual(entry["phoneme_type"], "espeak")
                self.assertEqual(entry["engine"], "piper")

    def test_each_one_reads_from_its_own_repository(self):
        entries = _entries()
        for voice_id, (_, repo) in EXPECTED.items():
            with self.subTest(voice_id):
                entry = entries[voice_id]
                base = rf"https://huggingface\.co/Phoonnx/{re.escape(repo)}/resolve/[0-9a-f]{{40}}/"
                self.assertRegex(entry["model_url"], f"^{base}model\\.onnx$")
                self.assertRegex(entry["config_url"], f"^{base}model\\.onnx\\.json$")
