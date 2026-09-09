"""The on-disk voice index survives a manager that cannot read it.

An index that fails to parse loads as an empty registry. Writing that registry
back replaces the file with ``{}``, which loses the catalog and the evidence of
what broke it in the same step, so construction must never write over an index
that is already there.
"""
import json
import os
import tempfile
import unittest

from phoonnx.model_manager import TTSModelManager


class TestExistingIndexIsNotOverwritten(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "voices.json")

    def test_an_unparseable_index_is_left_on_disk(self):
        truncated = '{"a-voice": {"voice_id": "a-voice"}\n'
        with open(self.path, "w") as f:
            f.write(truncated)

        TTSModelManager(cache_path=self.path)

        with open(self.path) as f:
            self.assertEqual(truncated, f.read())

    def test_a_populated_index_survives_construction(self):
        entry = {"a-voice": {"voice_id": "a-voice", "lang": "en-us"}}
        with open(self.path, "w") as f:
            json.dump(entry, f)

        TTSModelManager(cache_path=self.path)

        with open(self.path) as f:
            self.assertEqual(entry, json.load(f))

    def test_a_missing_index_is_created(self):
        TTSModelManager(cache_path=self.path)

        self.assertTrue(os.path.isfile(self.path))
        with open(self.path) as f:
            self.assertEqual({}, json.load(f))


if __name__ == "__main__":
    unittest.main()
