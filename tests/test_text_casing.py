"""A voice trained with a text casing reads its text in that casing at inference.

The id lists are worked out by hand from a small ``phoneme_id_map`` that holds
both cases of ``H``/``h``, so a casing that did not run gives other ids. No
model is loaded: ``prepare`` is the text half of synthesis.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from phoonnx.config import VoiceConfig, get_text_casing
from phoonnx.engines.vits import VitsAdapter
from phoonnx.voice import TTSVoice

ID_MAP = {"_": [0], "^": [1], "$": [2], "h": [3], "e": [4], "l": [5], "o": [6], "H": [7],
          "E": [8], "L": [9], "O": [10]}
CONFIG = {"phoonnx_version": "1.0", "engine": "phoonnx", "phoneme_type": "unicode",
          "alphabet": "unicode", "lang_code": "en-US", "audio": {"sample_rate": 16000},
          "num_symbols": 16, "phoneme_id_map": ID_MAP}
# BOS, then blank-separated symbols, then a blank and EOS
LOWER_HELLO = [1, 0, 3, 0, 4, 0, 5, 0, 5, 0, 6, 0, 2]
CAPITAL_HELLO = [1, 0, 7, 0, 4, 0, 5, 0, 5, 0, 6, 0, 2]


def _voice(**extra) -> TTSVoice:
    config = VoiceConfig.from_dict(dict(CONFIG, **extra))
    return TTSVoice(config=config, session=None, adapter=VitsAdapter())


def _ids(voice: TTSVoice, text: str):
    return [list(u.ids) for u in voice.prepare(text)]


def test_without_the_key_the_text_keeps_its_case():
    voice = _voice()
    assert _ids(voice, "Hello") == [CAPITAL_HELLO]
    assert _ids(voice, "hello") == [LOWER_HELLO]


def test_a_config_without_the_key_applies_no_casing():
    assert _voice().config.text_casing == "ignore"


def test_a_voice_trained_with_lower_reads_hello_as_hello():
    voice = _voice(text_casing="lower")
    assert _ids(voice, "Hello") == [LOWER_HELLO]
    assert _ids(voice, "hello") == [LOWER_HELLO]


def test_a_voice_trained_with_upper_reads_hello_in_capitals():
    voice = _voice(text_casing="upper")
    upper_hello = [1, 0, 7, 0, 8, 0, 9, 0, 9, 0, 10, 0, 2]
    assert _ids(voice, "hello") == [upper_hello]


def test_the_casing_survives_the_native_config_round_trip():
    native = _voice(text_casing="lower").config.to_native_dict()
    assert native["text_casing"] == "lower"
    assert VoiceConfig.from_dict(native).text_casing == "lower"


def test_the_casing_names_map_to_the_str_methods():
    assert [get_text_casing(name)("Straße") for name in ("lower", "upper", "casefold", "ignore")] == \
           ["straße", "STRASSE", "strasse", "Straße"]


def test_an_exported_voice_config_records_the_casing():
    from phoonnx_train.engines.vits import _write_piper_json
    from phoonnx_train.engines.yourtts import _yourtts_voice_json

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "voice.piper.json"
        _write_piper_json(dict(CONFIG, text_casing="lower"), path)
        assert json.loads(path.read_text())["text_casing"] == "lower"
        _write_piper_json(CONFIG, path)
        assert "text_casing" not in json.loads(path.read_text())
    assert _yourtts_voice_json(dict(CONFIG, text_casing="lower"), None)["text_casing"] == "lower"
    assert "text_casing" not in _yourtts_voice_json(CONFIG, None)
