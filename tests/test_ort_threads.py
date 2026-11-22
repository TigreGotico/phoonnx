"""The thread count of a voice reaches every ONNX Runtime session it builds.

Each test reads ``intra_op_num_threads`` back from the session that ONNX
Runtime built, so it checks the value the runtime holds, not the argument.
"""
import json
import os
import time
from unittest.mock import patch

import onnx
import onnxruntime
import pytest
from onnx import TensorProto, helper

from phoonnx import alignment, providers
from phoonnx.engines.glowtts import GlowTTSAdapter
from phoonnx.engines.matcha import MatchaAdapter
from phoonnx.engines.mixertts import MixerTTSAdapter
from phoonnx.engines.speaker_encoders.coqui_resnet import CoquiResNetSpeakerEncoder
from phoonnx.engines.speaker_encoders.styletts2_style import StyleTTS2StyleEncoder
from phoonnx.engines.vocoders.vocos import VocosVocoder
from phoonnx.providers import THREADS_ENV_VAR, make_session, resolve_threads
from phoonnx.voice import TTSVoice

CPU = ["CPUExecutionProvider"]


def tiny_model(path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "tiny", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return str(path)


def threads_of(session):
    return session.get_session_options().intra_op_num_threads


@pytest.fixture(autouse=True)
def no_thread_env(monkeypatch):
    monkeypatch.delenv(THREADS_ENV_VAR, raising=False)
    monkeypatch.delenv("PHOONNX_ORT_CACHE_DIR", raising=False)
    monkeypatch.delenv("PHOONNX_SHARE_ONNX_SESSIONS", raising=False)


@pytest.mark.parametrize("argument, env, expected", [
    (None, None, None),
    (3, None, 3),
    (None, "4", 4),
    (3, "4", 3),  # the argument comes first
    (0, "4", 0),  # 0 asks for ONNX Runtime's default, also over the environment
    (None, "0", 0),
    (None, " 2 ", 2),
    ("2", None, 2),  # a plugin config may hold a digit string
])
def test_threads_come_from_the_argument_then_the_environment(monkeypatch, argument, env, expected):
    if env is not None:
        monkeypatch.setenv(THREADS_ENV_VAR, env)
    assert resolve_threads(argument) == expected


@pytest.mark.parametrize("env", ["two", "-1", "1.5"])
def test_a_bad_environment_value_warns_and_keeps_the_default(monkeypatch, env):
    monkeypatch.setenv(THREADS_ENV_VAR, env)
    with patch("phoonnx.providers.LOG") as log:
        assert resolve_threads() is None
    assert THREADS_ENV_VAR in log.warning.call_args.args[0]


@pytest.mark.parametrize("threads", [-1, "two", "1.5", 1.5, True])
def test_an_argument_that_is_not_a_whole_number_raises(threads):
    with pytest.raises(ValueError, match="zero or more"):
        resolve_threads(threads)


def test_the_session_runs_on_the_requested_threads(tmp_path, monkeypatch):
    path = tiny_model(tmp_path / "m.onnx")
    assert threads_of(make_session(path, providers=CPU)) == 0
    assert threads_of(make_session(path, providers=CPU, threads=2)) == 2
    monkeypatch.setenv(THREADS_ENV_VAR, "3")
    assert threads_of(make_session(path, providers=CPU)) == 3


def test_voices_share_a_session_only_at_the_same_thread_count(tmp_path):
    path = tiny_model(tmp_path / "m.onnx")
    two = make_session(path, providers=CPU, threads=2)
    assert make_session(path, providers=CPU, threads=2) is two
    three = make_session(path, providers=CPU, threads=3)
    assert three is not two
    assert (threads_of(two), threads_of(three)) == (2, 3)
    default = make_session(path, providers=CPU)
    assert default is not two and threads_of(default) == 0


def test_explicit_session_options_keep_their_own_threads(tmp_path):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 5
    session = make_session(tiny_model(tmp_path / "m.onnx"), providers=CPU, sess_options=options, threads=2)
    assert threads_of(session) == 5


def test_the_thread_count_is_part_of_the_session_key(tmp_path):
    path = tiny_model(tmp_path / "m.onnx")
    assert providers.session_key(path, CPU, None, 2) != providers.session_key(path, CPU, None, None)


def test_a_loaded_voice_and_its_engine_params_get_the_thread_count(tmp_path):
    path = tiny_model(tmp_path / "voice.onnx")
    (tmp_path / "voice.onnx.json").write_text(json.dumps({"phoneme_type": "unicode", "alphabet": "unicode"}))
    voice = TTSVoice.load(path, providers=CPU, threads=2)
    assert threads_of(voice.session) == 2
    assert voice.config.engine_params["threads"] == 2
    # a runtime setting, like the providers: an exported config does not record it
    assert "threads" not in voice.config.to_dict().get("engine_params", {})


def test_a_voice_with_no_thread_count_leaves_the_engine_params_alone(tmp_path):
    path = tiny_model(tmp_path / "voice.onnx")
    (tmp_path / "voice.onnx.json").write_text(json.dumps({"phoneme_type": "unicode", "alphabet": "unicode"}))
    voice = TTSVoice.load(path, providers=CPU)
    assert threads_of(voice.session) == 0
    assert "threads" not in voice.config.engine_params


def test_the_environment_reaches_a_loaded_voice(tmp_path, monkeypatch):
    monkeypatch.setenv(THREADS_ENV_VAR, "3")
    path = tiny_model(tmp_path / "voice.onnx")
    (tmp_path / "voice.onnx.json").write_text(json.dumps({"phoneme_type": "unicode", "alphabet": "unicode"}))
    voice = TTSVoice.load(path, providers=CPU)
    assert threads_of(voice.session) == 3
    assert voice.config.engine_params["threads"] == 3


def test_auxiliary_graphs_read_the_thread_count_of_the_engine_params(tmp_path):
    params = {"providers": CPU, "threads": 2}
    vocoder = VocosVocoder(model_path=tiny_model(tmp_path / "vocoder.onnx"), config=params)
    encoder = CoquiResNetSpeakerEncoder(tiny_model(tmp_path / "encoder.onnx"), config=params)
    style = StyleTTS2StyleEncoder(tiny_model(tmp_path / "style.onnx"), config=params)
    assert (threads_of(vocoder.session), threads_of(encoder.session), threads_of(style.session)) == (2, 2, 2)


def test_the_alignment_copy_runs_on_the_thread_count_of_the_voice(tmp_path):
    model = tiny_model(tmp_path / "voice.onnx")
    patched = alignment.alignment_model_path(model)
    tiny_model(patched)
    later = time.time() + 5
    os.utime(patched, (later, later))  # newer than the model: reused, not patched again
    assert threads_of(alignment.ensure_alignment_session(model, CPU, 2)) == 2


def test_a_new_alignment_copy_runs_on_the_thread_count_of_the_voice(tmp_path, monkeypatch):
    model = tiny_model(tmp_path / "voice.onnx")
    monkeypatch.setattr(alignment, "export_alignment_model", lambda source, destination: tiny_model(destination))
    assert not alignment.alignment_model_path(model).exists()
    assert threads_of(alignment.ensure_alignment_session(model, CPU, 2)) == 2


def test_threads_zero_keeps_the_default_over_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(THREADS_ENV_VAR, "3")
    path = tiny_model(tmp_path / "voice.onnx")
    (tmp_path / "voice.onnx.json").write_text(json.dumps({"phoneme_type": "unicode", "alphabet": "unicode"}))
    voice = TTSVoice.load(path, providers=CPU, threads=0)
    assert threads_of(voice.session) == 0
    # the auxiliary graphs read 0 too, not the environment
    assert threads_of(make_session(tiny_model(tmp_path / "aux.onnx"), providers=CPU,
                                   threads=voice.config.engine_params["threads"])) == 0


@pytest.mark.parametrize("adapter", [GlowTTSAdapter, MatchaAdapter, MixerTTSAdapter])
def test_a_mel_engine_builds_its_vocoder_on_the_threads_of_the_voice(tmp_path, adapter):
    engine = adapter()
    engine.configure_from_params({"vocoder_path": tiny_model(tmp_path / "vocoder.onnx"), "vocoder_type": "hifigan",
                                  "vocoder_config": {}, "providers": CPU, "threads": 2})
    assert threads_of(engine.vocoder.session) == 2


def test_the_alignment_copy_of_a_threads_zero_voice_keeps_the_default(tmp_path, monkeypatch):
    monkeypatch.setenv(THREADS_ENV_VAR, "3")
    path = tiny_model(tmp_path / "voice.onnx")
    (tmp_path / "voice.onnx.json").write_text(json.dumps({"phoneme_type": "unicode", "alphabet": "unicode"}))
    voice = TTSVoice.load(path, providers=CPU, threads=0)
    monkeypatch.setattr(alignment, "export_alignment_model", lambda source, destination: tiny_model(destination))
    assert threads_of(voice._ensure_alignment_session()) == 0
