import json
import unittest
from dataclasses import fields
from typing import Callable, Dict, List, Optional, Set, Tuple
from unittest.mock import patch

from phoonnx import index_schema
from phoonnx.index_schema import (KNOWN_FIELDS, REQUIRED_FIELDS, _describes,
                                  validate_entry, validate_index_file)
from phoonnx.model_manager import TTSModelInfo, TTSModelManager


class TestSchemaMatchesLoader(unittest.TestCase):
    """The schema is read off TTSModelInfo, so the two cannot drift apart."""

    def test_known_fields_are_the_dataclass_fields(self):
        self.assertEqual(KNOWN_FIELDS, {f.name for f in fields(TTSModelInfo)})

    def test_required_fields_are_the_ones_without_defaults(self):
        self.assertEqual(REQUIRED_FIELDS, {"voice_id", "lang", "model_url"})


class TestBundledIndexFiles(unittest.TestCase):
    """Every voice-index file phoonnx ships must satisfy the schema."""

    def test_every_bundled_index_file_validates(self):
        index_files = TTSModelManager.voice_index_files()
        self.assertGreater(len(index_files), 0)
        problems = []
        for path in index_files:
            problems += validate_index_file(path)
        self.assertEqual(problems, [], "\n".join(problems))

    def test_every_bundled_entry_loads_into_the_dataclass(self):
        for path in TTSModelManager.voice_index_files():
            with open(path, encoding="utf-8") as f:
                index = json.load(f)
            for voice_id, entry in index.items():
                with self.subTest(index=path.name, voice=voice_id):
                    TTSModelInfo(**entry)


class TestValidateEntry(unittest.TestCase):

    def _entry(self, **overrides):
        entry = {"voice_id": "org/voice", "lang": "pt-PT",
                 "model_url": "https://example.invalid/model.onnx"}
        entry.update(overrides)
        return entry

    def test_minimal_entry_is_valid(self):
        self.assertEqual(validate_entry("org/voice", self._entry()), [])

    def test_unknown_field_is_rejected(self):
        problems = validate_entry("org/voice", self._entry(vocoder_ulr="x"))
        self.assertEqual(len(problems), 1)
        self.assertIn("unknown field 'vocoder_ulr'", problems[0])

    def test_missing_required_field_is_rejected(self):
        entry = self._entry()
        del entry["model_url"]
        problems = validate_entry("org/voice", entry)
        self.assertEqual(len(problems), 1)
        self.assertIn("missing required field 'model_url'", problems[0])

    def test_wrong_scalar_type_is_rejected(self):
        problems = validate_entry("org/voice", self._entry(requires_reference="yes"))
        self.assertEqual(len(problems), 1)
        self.assertIn("'requires_reference'", problems[0])

    def test_wrong_mapping_type_is_rejected(self):
        problems = validate_entry(
            "org/voice", self._entry(aux_model_urls="https://example.invalid/a.onnx"))
        self.assertEqual(len(problems), 1)
        self.assertIn("'aux_model_urls'", problems[0])

    def test_mapping_with_wrong_value_type_is_rejected(self):
        problems = validate_entry("org/voice", self._entry(vocab_override={"a": "1"}))
        self.assertEqual(len(problems), 1)
        self.assertIn("'vocab_override'", problems[0])

    def test_value_outside_an_enum_is_rejected(self):
        problems = validate_entry("org/voice", self._entry(phoneme_type="esspeak"))
        self.assertEqual(len(problems), 1)
        self.assertIn("'phoneme_type'", problems[0])

    def test_value_inside_an_enum_is_accepted(self):
        self.assertEqual(validate_entry("org/voice", self._entry(phoneme_type="espeak")), [])

    def test_null_is_accepted_for_optional_fields(self):
        self.assertEqual(validate_entry("org/voice", self._entry(vocoder_url=None)), [])

    def test_unparseable_language_tag_is_rejected(self):
        problems = validate_entry("org/voice", self._entry(lang="not a language"))
        self.assertTrue(any("usable language tag" in p for p in problems), problems)

    def test_phonemizer_lang_is_checked_too(self):
        problems = validate_entry("org/voice", self._entry(phonemizer_lang="!!"))
        self.assertTrue(any("usable language tag" in p for p in problems), problems)

    def test_voice_id_must_match_its_key(self):
        problems = validate_entry("org/voice", self._entry(voice_id="org/other"))
        self.assertEqual(len(problems), 1)
        self.assertIn("does not match the key", problems[0])


class TestFutureFieldTypes(unittest.TestCase):
    """Adding a field to the dataclass must be all that a new field needs."""

    def _validate_with_field(self, name, annotation, value):
        entry = {"voice_id": "org/voice", "lang": "pt-PT",
                 "model_url": "https://example.invalid/model.onnx", name: value}
        with patch.dict(index_schema._FIELD_TYPES, {name: annotation}):
            return validate_entry("org/voice", entry)

    def test_a_list_field_validates_its_elements(self):
        self.assertEqual(self._validate_with_field("speaker_tags", List[str], ["a", "b"]), [])
        self.assertEqual(self._validate_with_field("speaker_tags", List[str], []), [])

    def test_a_list_field_rejects_a_wrong_element_type(self):
        problems = self._validate_with_field("speaker_tags", List[str], ["a", 1])
        self.assertEqual(len(problems), 1)
        self.assertIn("'speaker_tags'", problems[0])

    def test_a_list_field_rejects_a_non_sequence(self):
        problems = self._validate_with_field("speaker_tags", List[str], "a")
        self.assertEqual(len(problems), 1)
        self.assertIn("'speaker_tags'", problems[0])

    def test_optional_and_nested_sequences_are_described(self):
        self.assertTrue(_describes(None, Optional[List[str]]))
        self.assertTrue(_describes([{"a": "b"}], List[Dict[str, str]]))
        self.assertTrue(_describes([1, "a"], Tuple[int, str]))
        self.assertFalse(_describes([1], Tuple[int, str]))
        self.assertTrue(_describes(["a"], Set[str]))

    def test_an_annotation_shape_it_cannot_describe_says_so(self):
        with self.assertRaises(TypeError) as caught:
            _describes(None, Callable[[int], str])
        self.assertIn("index_schema cannot describe", str(caught.exception))

    def test_every_dataclass_annotation_in_use_is_describable(self):
        for name, annotation in index_schema._FIELD_TYPES.items():
            with self.subTest(field=name):
                _describes(object(), annotation)


class TestValidateIndexFile(unittest.TestCase):

    def test_a_typo_in_a_scratch_index_is_caught(self):
        import tempfile
        index = {"org/voice": {"voice_id": "org/voice", "lang": "pt-PT",
                               "model_url": "https://example.invalid/model.onnx",
                               "phonemizer_langg": "pt"}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(index, f)
            path = f.name
        problems = validate_index_file(path)
        self.assertEqual(len(problems), 1)
        self.assertIn("unknown field 'phonemizer_langg'", problems[0])


if __name__ == "__main__":
    unittest.main()
