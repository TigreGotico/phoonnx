"""The three kasanoma voices, and the fields they cannot lose.

No kasanoma config names the language it speaks: ``language`` is present and
null in all three, and ``espeak.voice`` is ``lfn``. So the language a caller
asks for comes from the index or from nowhere, and ``lang`` here is the only
record of which voice speaks Twi, Chichewa or Makhuwa.

The model files come from ``OpenVoiceOS/phoonnx-vits``, the architecture
repository that holds every VITS voice phoonnx loads, one folder per voice.
Upstream publishes two of the three as a zip on a GitHub release only, which
phoonnx cannot read, and the third from an account that carries no licence
file. That mirror is the one stable url for all three.
"""
import json
import unittest

from phoonnx.index_schema import validate_index_file
from phoonnx.model_manager import TTSModelManager

INDEX = "piper_community.json"

ARCHITECTURE_REPO = "https://huggingface.co/OpenVoiceOS/phoonnx-vits/resolve/main"

# voice id -> the language tag, and the folder inside the architecture repository
EXPECTED = {
    "piper_community/ghananlpcommunity/kasanoma-twi": ("tw", "tw-kasanoma"),
    "piper_community/michsethowusu/kasanoma-chichewa": ("ny", "ny-kasanoma"),
    "piper_community/michsethowusu/kasanoma-makhuwa": ("vmw", "vmw-kasanoma"),
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

    def test_each_one_reads_from_its_folder_in_the_architecture_repository(self):
        # A VITS voice resolves out of OpenVoiceOS/phoonnx-vits and never out
        # of a repository of its own.
        entries = _entries()
        for voice_id, (_, folder) in EXPECTED.items():
            with self.subTest(voice_id):
                entry = entries[voice_id]
                base = f"{ARCHITECTURE_REPO}/{folder}/"
                self.assertEqual(entry["model_url"], base + "model.onnx")
                self.assertEqual(entry["config_url"], base + "model.onnx.json")
