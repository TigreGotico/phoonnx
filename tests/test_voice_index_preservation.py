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

from tempfile import mkdtemp
from unittest.mock import patch

from phoonnx import model_manager
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


class TestUnreadableIndexIsNamedAccurately(unittest.TestCase):
    """The diagnostic fires when the index did not parse, and only then.

    Its whole purpose is to be believed. An operator who sees it on a healthy
    empty index learns to ignore it, and then misses the real one.
    """

    def _construct_and_capture(self, contents):
        path = os.path.join(mkdtemp(), "voices.json")
        with open(path, "w") as handle:
            handle.write(contents)
        complaints = []
        with patch.object(model_manager.LOG, "error", complaints.append):
            model_manager.TTSModelManager(cache_path=path)
        with open(path) as handle:
            survived = handle.read()
        return [c for c in complaints if "could not be read" in c], survived

    def test_an_empty_object_with_a_newline_is_not_reported_as_broken(self):
        errors, survived = self._construct_and_capture("{}\n")
        self.assertEqual([], errors)
        self.assertEqual("{}\n", survived)

    def test_an_empty_file_is_reported_as_broken(self):
        errors, survived = self._construct_and_capture("")
        self.assertEqual(1, len(errors))
        self.assertEqual("", survived)

    def test_truncated_json_is_reported_as_broken(self):
        errors, survived = self._construct_and_capture('{"a": ')
        self.assertEqual(1, len(errors))
        self.assertEqual('{"a": ', survived)


if __name__ == "__main__":
    unittest.main()
