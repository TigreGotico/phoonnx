"""A voice that cannot load raises VoiceLoadError before any model or audio work."""
import json
from unittest.mock import MagicMock, patch

import pytest

from phoonnx.errors import PhoonnxError, VoiceLoadError
from phoonnx.voice import TTSVoice


@pytest.fixture
def fake_runtime():
    with patch("phoonnx.voice.make_session", return_value=MagicMock()) as session, \
            patch("phoonnx.voice.detect_engine", return_value=MagicMock()):
        yield session


def test_a_voice_with_no_config_raises_before_the_model_loads(tmp_path, fake_runtime):
    model = tmp_path / "model.onnx"
    with pytest.raises(VoiceLoadError, match="model.onnx.json"):
        TTSVoice.load(model)
    fake_runtime.assert_not_called()


def test_a_missing_config_path_named_by_the_caller_raises(tmp_path, fake_runtime):
    with pytest.raises(PhoonnxError, match="voice.json"):
        TTSVoice.load(tmp_path / "model.onnx", tmp_path / "voice.json")


def test_a_failed_load_writes_no_wav(tmp_path, fake_runtime):
    wav = tmp_path / "out.wav"
    with pytest.raises(VoiceLoadError):
        TTSVoice.load(tmp_path / "model.onnx").synthesize_wav("Hello.", str(wav))
    assert not wav.exists()


def test_a_voice_with_its_config_still_loads(tmp_path, fake_runtime):
    config = tmp_path / "model.onnx.json"
    config.write_text(json.dumps({"phoneme_type": "unicode", "alphabet": "unicode",
                                  "audio": {"sample_rate": 22050}, "phoneme_id_map": {"_": [0], "a": [1]}}))
    voice = TTSVoice.load(tmp_path / "model.onnx")
    assert voice.config.sample_rate == 22050
    fake_runtime.assert_called_once()


def _vits_voice(tmp_path, adapter_cls, num_symbols):
    from phoonnx.engines.vits import VitsAdapter
    config = tmp_path / "model.onnx.json"
    config.write_text(json.dumps({"phoneme_type": "unicode", "alphabet": "unicode", "num_symbols": num_symbols,
                                  "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a": [3], "b": [4]}}))
    adapter = MagicMock(spec=adapter_cls or VitsAdapter)
    with patch("phoonnx.voice.detect_engine", return_value=adapter):
        return TTSVoice.load(tmp_path / "model.onnx")


def test_a_vits_vocabulary_id_the_embedding_cannot_hold_raises(tmp_path, fake_runtime):
    with pytest.raises(VoiceLoadError, match="id 4.*num_symbols=4"):
        _vits_voice(tmp_path, None, num_symbols=4)


def test_a_vits_vocabulary_that_fits_loads(tmp_path, fake_runtime):
    assert _vits_voice(tmp_path, None, num_symbols=5).config.num_symbols == 5


def test_engines_whose_graph_ignores_num_symbols_are_not_checked(tmp_path, fake_runtime):
    from phoonnx.engines.styletts2 import StyleTTS2Adapter
    assert _vits_voice(tmp_path, StyleTTS2Adapter, num_symbols=4).config.num_symbols == 4
