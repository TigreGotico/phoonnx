import os
import tempfile
import unittest

from phoonnx.errors import UnknownVoice
from phoonnx.model_manager import TTSModelManager


class TestUnknownVoiceLookup(unittest.TestCase):
    def setUp(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.manager = TTSModelManager(cache_path=os.path.join(tmpdir.name, "cache.json"))

    def test_index_raises_unknown_voice_naming_the_id_and_the_listing(self):
        with self.assertRaises(UnknownVoice) as ctx:
            self.manager.voices["no/such-voice"]
        self.assertIn("no/such-voice", str(ctx.exception))
        self.assertIn("phoonnx-voices list-voices", str(ctx.exception))

    def test_unknown_voice_is_still_a_key_error(self):
        with self.assertRaises(KeyError):
            self.manager.voices["no/such-voice"]

    def test_replaced_registry_keeps_the_typed_error(self):
        self.manager.voices = {}
        with self.assertRaises(UnknownVoice):
            self.manager.voices["x"]

    def test_get_still_returns_none(self):
        self.assertIsNone(self.manager.voices.get("x"))
        self.assertIsNone(self.manager.get_voice("x"))
