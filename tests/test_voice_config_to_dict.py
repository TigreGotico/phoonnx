"""VoiceConfig.to_dict writes the native config.json: it reads back to the same ids and fits the schema."""
import copy
import json
import re
from pathlib import Path

import jsonschema
import pytest

import phoonnx
from phoonnx.config import Alphabet, Engine, PhonemeType, VoiceConfig
from phoonnx.engines.coqui_config import voice_config_from_coqui
from phoonnx.engines.glowtts_config import voice_config_from_larynx
from phoonnx.engines.mixertts_config import voice_config_from_mixer
from phoonnx.engines.optispeech_config import voice_config_from_optispeech_meta
from phoonnx.version import __version__
from tests.config_fixtures import (CANONICAL, COQUI, PIPER, TRANSFORMERS, VOSK, assert_same_voice,
                                   piper_ids)

SCHEMA = json.loads((Path(phoonnx.__file__).parent / "config.schema.json").read_text(encoding="utf-8"))


def load(config, **companions):
    return VoiceConfig.from_dict(copy.deepcopy(config), **companions)


def read_back(voice, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(voice.to_dict(), ensure_ascii=False), encoding="utf-8")
    return VoiceConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))


def test_piper_ids_equal_the_loader_and_piper_itself(tmp_path):
    old = load(PIPER)
    new = read_back(old, tmp_path)
    sentences = [["a", "b"], ["a", " ", "b", "!"]]
    assert_same_voice(old, new, sentences)
    for sentence in sentences:
        assert new.tokenizer.tokenize(sentence) == piper_ids(PIPER["phoneme_id_map"], sentence)
    assert new.keep_punctuation is True


def test_vosk_ids(tmp_path):
    old = load(VOSK)
    new = read_back(old, tmp_path)
    assert_same_voice(old, new, [["a0", "sch"]])
    # vosk_tts reads ^ 0 p 0 p 0 $
    assert new.tokenizer.tokenize(["a0", "sch"]) == [1, 0, 10, 0, 20, 0, 2]
    assert new.phoneme_type == PhonemeType.VOSK and new.engine == Engine.VOSK


def test_coqui_ids(tmp_path):
    old = load(COQUI)
    new = read_back(old, tmp_path)
    assert_same_voice(old, new, [["a", "b"]])
    # VitsCharacters: pad 0, punctuations, characters, blank last; the blank sits between the symbols
    assert new.tokenizer.tokenize(["a", "b"]) == [5, 3, 5, 4, 5]


def test_transformers_ids(tmp_path):
    old = load({}, **TRANSFORMERS)
    new = read_back(old, tmp_path)
    assert_same_voice(old, new, [["a", "b"], ["b", " ", "a"]])
    # the MMS tokenizer puts its pad token between the characters and at both ends
    assert new.tokenizer.tokenize(["a", "b"]) == [0, 1, 0, 2, 0]


def test_tokens_file_voice_has_no_bos_and_keeps_its_ids(tmp_path):
    tokens = tmp_path / "tokens.txt"
    tokens.write_text("_ 0\n^ 1\n$ 2\na 3\nb 4\n", encoding="utf-8")
    old = load({"pad": "_"}, tokens_txt=str(tokens))
    new = read_back(old, tmp_path)
    # the file names no BOS or EOS token, so the voice adds none, although the vocabulary holds "^" and "$"
    assert old.tokenizer.tokenize(["a", "b"]) == [0, 3, 0, 4, 0]
    assert new.tokenizer.tokenize(["a", "b"]) == [0, 3, 0, 4, 0]


def test_canonical_config_ids(tmp_path):
    old = load(CANONICAL)
    new = read_back(old, tmp_path)
    assert_same_voice(old, new, [["a", "b"]])
    assert new.tokenizer.tokenize(["a", "b"]) == [0, 2, 0, 3, 0]


def test_raw_text_engine_has_an_empty_symbol_table(tmp_path):
    old = load({"engine": "omnivoice"})
    native = old.to_dict()
    assert native["phoneme_id_map"] == {} and native["engine"] == "omnivoice"
    new = read_back(old, tmp_path)
    assert_same_voice(old, new, [["a"]])
    assert new.alphabet == Alphabet.GRAPHEMES == old.alphabet and new.sample_rate == 24000


def test_glowtts_larynx_builder_round_trips(tmp_path):
    voice = voice_config_from_larynx({"audio": {"sample_rate": 22050}, "model": {"num_symbols": 3}},
                                     "0 _\n1 a\n2 b\n", lang_code="en-us")
    new = read_back(voice, tmp_path)
    assert_same_voice(voice, new, [["a", "b"]])
    assert new.tokenizer.tokenize(["a", "b"]) == [0, 1, 0, 2, 0]


def test_coqui_builder_round_trips(tmp_path):
    voice = voice_config_from_coqui(COQUI, lang_code="en-us", engine=Engine.FASTPITCH)
    new = read_back(voice, tmp_path)
    assert_same_voice(voice, new, [["a", "b"]])
    assert new.engine == Engine.FASTPITCH


def test_mixertts_builder_round_trips(tmp_path):
    voice = voice_config_from_mixer(["_pad_", "_+_", "<", "b"], lang_code="ar", phoneme_type=PhonemeType.MANTOQ)
    new = read_back(voice, tmp_path)
    assert_same_voice(voice, new, [["<", "b"]])
    assert new.phoneme_type == PhonemeType.MANTOQ


def test_optispeech_builder_round_trips(tmp_path):
    meta = {"input_symbols": {"_": 0, "^": 1, "$": 2, "a": 3}, "special_symbols": {},
            "text_processor": {"tokenizer": "ipa"}, "languages": ["en-us"], "sample_rate": 22050}
    voice = voice_config_from_optispeech_meta(meta)
    new = read_back(voice, tmp_path)
    assert_same_voice(voice, new, [["a"]])
    assert new.tokenizer.tokenize(["a"]) == voice.tokenizer.tokenize(["a"]) == [1, 0, 3, 0, 2]


def test_native_config_round_trips_byte_equal(tmp_path):
    native = load(PIPER, lang_code="en-us").to_dict()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(native, ensure_ascii=False), encoding="utf-8")
    again = VoiceConfig.from_dict(json.loads(path.read_text(encoding="utf-8"))).to_dict()
    assert json.dumps(again, ensure_ascii=False) == json.dumps(native, ensure_ascii=False)


def test_native_round_trip_keeps_blank_between_and_phonemizer_identity():
    native = load(PIPER).to_dict()
    native["blank_between"] = "tokens"
    native["phonemizer_identity"] = {"name": "espeak", "version": "1.52"}
    again = VoiceConfig.from_dict(copy.deepcopy(native)).to_dict()
    assert json.dumps(again, ensure_ascii=False) == json.dumps(native, ensure_ascii=False)


def test_to_dict_writes_the_running_version_and_no_providers():
    voice = load(PIPER, engine_params={"providers": ["CPUExecutionProvider"], "x": 1})
    native = voice.to_dict()
    assert native["phoonnx_version"] == __version__
    assert native["engine_params"] == {"x": 1}


@pytest.mark.parametrize("config, companions", [
    (PIPER, {}), (VOSK, {}), (COQUI, {}), (CANONICAL, {}), ({"engine": "omnivoice"}, {}), ({}, TRANSFORMERS),
])
def test_to_dict_validates_against_the_schema(config, companions):
    jsonschema.validate(load(config, **companions).to_dict(), SCHEMA)


def test_unset_scales_stay_out_of_the_inference_block():
    voice = load(PIPER)
    voice.noise_scale = voice.noise_w_scale = None
    voice.length_scale = 1.2
    native = voice.to_dict()
    assert native["inference"] == {"length_scale": 1.2, "add_diacritics": voice.add_diacritics,
                                   "diacritizer_model": voice.diacritizer_model}
    jsonschema.validate(native, SCHEMA)


# Phonemizers the schema lists before a scriptconv alpha publishes them. Remove a name
# once the scriptconv floor in pyproject.toml includes it, and give it a row in
# docs/phonemizers.md.
SCRIPTCONV_PENDING: set = {"nemo_ipa"}


def test_schema_enums_list_every_engine_phonemizer_and_alphabet():
    properties = SCHEMA["properties"]
    assert properties["engine"]["enum"] == [e.value for e in Engine]
    # PhonemeType and Alphabet come from scriptconv, so the installed scriptconv may be
    # older than the schema: a value the schema lists must be scriptconv's or pending
    for key, enum, pending in (("phoneme_type", PhonemeType, SCRIPTCONV_PENDING), ("alphabet", Alphabet, set())):
        values = {v.value for v in enum}
        missing = [v for v in values if v not in properties[key]["enum"]]
        assert missing == [], f"{key} enum lacks {missing}"
        unknown = [v for v in properties[key]["enum"] if v not in values | pending]
        assert unknown == [], f"{key} enum lists values scriptconv does not have: {unknown}"


def test_the_phonemizer_docs_have_a_row_for_every_published_phonemizer():
    docs = (Path(phoonnx.__file__).parent.parent / "docs" / "phonemizers.md").read_text(encoding="utf-8")
    # the Alphabet table has rows of the same shape, and some names are in both tables
    table = docs.split("## PhonemeType Reference", 1)[1].split("## Alphabet Reference", 1)[0]
    rows = set(re.findall(r"^\| `([a-z0-9_]+)` \|", table, flags=re.MULTILINE))
    published = set(SCHEMA["properties"]["phoneme_type"]["enum"]) - SCRIPTCONV_PENDING
    assert sorted(published - rows) == []


def test_schema_refuses_a_config_without_a_symbol_table():
    native = load(PIPER).to_dict()
    del native["phoneme_id_map"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(native, SCHEMA)


@pytest.mark.parametrize("casing", ["ignore", "lower", "upper", "casefold"])
def test_schema_takes_every_text_casing(casing):
    native = load(PIPER).to_dict()
    native["text_casing"] = casing
    jsonschema.validate(native, SCHEMA)


def test_schema_refuses_an_unknown_text_casing():
    # get_text_casing reads an unknown name as "ignore", so the voice would keep its case unnoticed
    native = load(PIPER).to_dict()
    native["text_casing"] = "title"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(native, SCHEMA)


def test_frontend_is_built_from_the_native_keys():
    native = load(PIPER, lang_code="en-us").to_dict()
    native["phonemizer_model"] = "model.onnx"
    native["inference"].update(add_diacritics=True, diacritizer_model="rawi")
    frontend = VoiceConfig.from_dict(native).frontend
    assert (frontend.lang, frontend.phonemizer, frontend.alphabet, frontend.model) == (
        "en-US", PhonemeType.ESPEAK, Alphabet.IPA, "model.onnx")
    assert frontend.keep_punctuation is True
    assert (frontend.stages.diacritize, frontend.stages.diacritizer_model) == (True, "rawi")


def test_to_native_dict_keeps_a_shim():
    voice = load(PIPER)
    with pytest.warns(DeprecationWarning, match=r"VoiceConfig\.to_native_dict.*VoiceConfig\.to_dict"):
        assert voice.to_native_dict() == voice.to_dict()


@pytest.mark.parametrize("config, companions", [
    (PIPER, {}), (COQUI, {}), ({}, TRANSFORMERS), ({"engine": "omnivoice"}, {}),
])
def test_native_config_is_a_fixed_point_of_from_dict_and_to_dict(config, companions):
    first = load(config, **companions).to_dict()
    second = VoiceConfig.from_dict(copy.deepcopy(first)).to_dict()
    assert json.dumps(second, ensure_ascii=False) == json.dumps(first, ensure_ascii=False)


def test_tokens_file_native_config_is_a_fixed_point(tmp_path):
    tokens = tmp_path / "tokens.txt"
    tokens.write_text("_ 0\n^ 1\n$ 2\na 3\nb 4\n", encoding="utf-8")
    first = load({"pad": "_"}, tokens_txt=str(tokens)).to_dict()
    assert first["bos"] == "^" and first["use_eos_bos"] is False
    second = VoiceConfig.from_dict(copy.deepcopy(first)).to_dict()
    assert json.dumps(second, ensure_ascii=False) == json.dumps(first, ensure_ascii=False)
