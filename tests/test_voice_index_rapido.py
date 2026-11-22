"""The Rápido voices, and the fields the engine needs to load them.

Rápido runs four graphs. The flow model is the voice's own model, and the
engine reads the other three from engine_params under fixed keys. An entry
whose aux_model_urls miss a key, or name it differently, loads a voice that
raises at the first synthesis.
"""
import json
import unittest

from phoonnx.index_schema import validate_index_file
from phoonnx.model_manager import TTSModelManager

INDEX = "rapido.json"
AUX_KEYS = {"text_encoder_path": "text_encoder.onnx", "aligner_path": "aligner.onnx",
            "decoder_path": "decoder.onnx"}


def _entries():
    path = TTSModelManager.voice_index_path() / INDEX
    return json.loads(path.read_text(encoding="utf-8"))


class TestRapidoIndex(unittest.TestCase):

    def test_the_file_is_valid(self):
        self.assertEqual(validate_index_file(TTSModelManager.voice_index_path() / INDEX), [])

    def test_the_file_is_in_the_merge_order(self):
        self.assertIn(INDEX, TTSModelManager._VOICE_INDEX_ORDER)

    def test_every_voice_is_reachable_by_source(self):
        by_source = TTSModelManager().get_available_voice_ids_by_source()
        self.assertEqual(sorted(by_source["rapido"]), ["edwixx/pt-BR-rapido"])

    def _file_sets(self):
        """Each precision of each voice: the entry and its int8 variant."""
        for voice_id, entry in _entries().items():
            yield voice_id, entry
            for name, variant in (entry.get("variants") or {}).items():
                yield f"{voice_id} ({name})", variant

    def test_every_url_is_in_the_phoonnx_org(self):
        for voice_id, files in self._file_sets():
            urls = [files["model_url"], files["config_url"], *files["aux_model_urls"].values()]
            for url in urls:
                self.assertTrue(url.startswith("https://huggingface.co/Phoonnx/"), (voice_id, url))

    def test_each_precision_names_the_three_graphs_from_its_own_folder(self):
        # The int8 files share the fp32 text encoder, aligner and decoder: only
        # the main graph differs (model_int8.onnx), and both sit in one repo commit.
        for voice_id, files in self._file_sets():
            folder = files["model_url"].rsplit("/", 1)[0]
            self.assertEqual(files["config_url"], f"{folder}/config.json")
            self.assertEqual(files["aux_model_urls"],
                             {key: f"{folder}/{name}" for key, name in AUX_KEYS.items()}, voice_id)
        for voice_id, entry in _entries().items():
            self.assertEqual(entry["engine"], "rapido")
            self.assertEqual(entry["lang"], "pt-BR")

    def test_the_int8_files_are_a_variant_of_the_voice(self):
        variant = _entries()["edwixx/pt-BR-rapido"]["variants"]["int8"]
        self.assertTrue(variant["model_url"].endswith("/model_int8.onnx"))
        # Not measured faster than full yet, so auto keeps the full files.
        self.assertNotIn("auto", variant)


if __name__ == "__main__":
    unittest.main()
