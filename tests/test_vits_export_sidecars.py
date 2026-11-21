"""The configs written beside an exported VITS model load on their own."""
import json

from phoonnx.config import VoiceConfig
from phoonnx_train.engines.vits import write_sidecars

TRAIN_CONFIG = {
    "dataset": "demo",
    "audio": {"sample_rate": 22050, "quality": "medium"},
    "lang_code": "an",
    "inference": {"noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8, "add_diacritics": False},
    "alphabet": "ipa",
    "phoneme_type": "espeak",
    "phonemizer_model": "",
    "phoneme_id_map": {"_": 0, "^": 1, "$": 2, " ": 3, "a": 4, "b": 5},
    "num_symbols": 6,
    "num_speakers": 1,
    "speaker_id_map": {},
    "phoonnx_version": "1.0.0",
}


def test_native_config_is_always_written_and_loads(tmp_path):
    write_sidecars(dict(TRAIN_CONFIG), tmp_path, "voice")
    native = json.loads((tmp_path / "voice.json").read_text(encoding="utf-8"))
    assert native == TRAIN_CONFIG
    assert VoiceConfig.is_phoonnx(native)
    assert VoiceConfig.from_dict(native).lang_code.split("-")[0] == "an"  # phoonnx expands it to a full tag
    assert not (tmp_path / "voice.piper.json").exists()
    assert not (tmp_path / "voice.tokens.txt").exists()


def test_piper_config_has_list_ids_and_the_language(tmp_path):
    write_sidecars(dict(TRAIN_CONFIG), tmp_path, "voice", piper=True)
    piper = json.loads((tmp_path / "voice.piper.json").read_text(encoding="utf-8"))
    assert piper["phoneme_id_map"] == {p: [i] for p, i in TRAIN_CONFIG["phoneme_id_map"].items()}
    assert piper["espeak"] == {"voice": "an"}
    assert piper["language"] == {"code": "an"}
    assert VoiceConfig.is_piper(piper)
    assert VoiceConfig.from_dict(piper).lang_code.split("-")[0] == "an"
    # the native config beside it is untouched by the piper export
    assert json.loads((tmp_path / "voice.json").read_text(encoding="utf-8")) == TRAIN_CONFIG


def test_tokens_are_named_after_the_model(tmp_path):
    write_sidecars(dict(TRAIN_CONFIG), tmp_path, "voice", generate_tokens=True)
    assert (tmp_path / "voice.tokens.txt").read_text(encoding="utf-8").splitlines() == ["_", "^", "$", " ", "a", "b"]
