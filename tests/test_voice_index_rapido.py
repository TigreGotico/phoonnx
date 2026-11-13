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
        self.assertEqual(sorted(by_source["rapido"]), ["edwixx/pt-BR-rapido", "edwixx/pt-BR-rapido-int8"])

    def test_each_voice_names_the_three_graphs_from_its_own_folder(self):
        # The int8 voice shares the fp32 text encoder, aligner and decoder by
        # content, but each entry reads them from its own folder, so the two
        # voices never depend on each other's layout.
        for voice_id, entry in _entries().items():
            folder = entry["model_url"].rsplit("/", 1)[0]
            self.assertEqual(entry["engine"], "rapido")
            self.assertEqual(entry["lang"], "pt-BR")
            self.assertEqual(entry["config_url"], f"{folder}/config.json")
            self.assertEqual(entry["aux_model_urls"],
                             {key: f"{folder}/{name}" for key, name in AUX_KEYS.items()}, voice_id)


if __name__ == "__main__":
    unittest.main()
