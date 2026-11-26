"""A voice warns at load when its phonemizer differs from the one it was trained with.

The contract (``speech-stack-contract.md``, section Voice provenance) says: "A
voice records its phonemizer and warns on a mismatch. It does not refuse to
load." and "A voice with no ``phonemizer_identity`` loads with no warning."

Each voice here loads through ``TTSVoice.load`` from a config file next to a
small real ONNX graph. The running identity comes from the orthography2ipa
backend called directly, not from the voice.
"""
import copy
import json
import time
from dataclasses import asdict
from unittest.mock import patch

import pytest
from onnx import TensorProto, helper, save
from scriptconv.frontend import TextFrontend
from scriptconv.phonemizers import get_phonemizer

from phoonnx.config import PhonemeType
from phoonnx.voice import TTSVoice

LANG = "pt-PT"
CONFIG = {"phoonnx_version": "1.0", "engine": "phoonnx", "phoneme_type": "orthography2ipa",
          "alphabet": "ipa", "lang_code": LANG, "audio": {"sample_rate": 16000},
          "num_symbols": 4, "phoneme_id_map": {"_": [0], "^": [1], "$": [2], " ": [3]}}
OTHER_REVISION = "sha256:" + "0" * 64


def _running_identity(lang=LANG):
    """The identity preprocess records for *lang*: the backend alone, no frontend version."""
    return asdict(get_phonemizer(PhonemeType.ORTHOGRAPHY2IPA).identity(lang))


def _unicode_identity(lang=LANG):
    """The identity a unicode voice records: the codepoint backend alone, no frontend version."""
    return asdict(get_phonemizer(PhonemeType.UNICODE).identity(lang))


def _write_model(path):
    ids = helper.make_tensor_value_info("input", TensorProto.INT64, [1, "T"])
    output = helper.make_tensor_value_info("output", TensorProto.INT64, [1, "T"])
    graph = helper.make_graph([helper.make_node("Identity", ["input"], ["output"])],
                              "tiny", [ids], [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    save(model, str(path))


def _load_warnings(tmp_path, **config):
    """The warnings ``TTSVoice.load`` logs for a voice with *config* over :data:`CONFIG`."""
    model = tmp_path / "pt_PT-provenance.onnx"
    _write_model(model)
    (tmp_path / "pt_PT-provenance.onnx.json").write_text(
        json.dumps({**copy.deepcopy(CONFIG), **config}), encoding="utf-8")
    with patch("phoonnx.voice.LOG") as log:
        voice = TTSVoice.load(model)
        if voice._provenance_thread is not None:
            voice._provenance_thread.join()
    assert isinstance(voice, TTSVoice)
    return [str(call.args[0]) for call in log.warning.call_args_list]


def test_a_matching_block_gives_no_warning(tmp_path):
    assert _load_warnings(tmp_path, phonemizer_identity=_running_identity()) == []


def test_another_revision_warns_and_names_both(tmp_path):
    running = _running_identity()
    trained = dict(running, revision=OTHER_REVISION)
    warnings = _load_warnings(tmp_path, phonemizer_identity=trained)
    assert len(warnings) == 1
    assert "pt_PT-provenance" in warnings[0]
    assert f"revision: trained {OTHER_REVISION!r}, running {running['revision']!r}" in warnings[0]
    assert "lang:" not in warnings[0]


def test_another_version_with_the_same_revision_gives_no_warning(tmp_path):
    trained = dict(_running_identity(), version="0.0.1")
    assert _load_warnings(tmp_path, phonemizer_identity=trained) == []


def test_another_version_with_no_revision_warns(tmp_path):
    running = _running_identity()
    trained = dict(running, version="0.0.1", revision=None)
    warnings = _load_warnings(tmp_path, phonemizer_identity=trained)
    assert len(warnings) == 1
    assert f"revision: trained None, running {running['revision']!r}" in warnings[0]
    assert f"version: trained '0.0.1', running {running['version']!r}" in warnings[0]


def test_another_lang_warns(tmp_path):
    trained = _running_identity("es-ES")
    running = _running_identity()
    warnings = _load_warnings(tmp_path, phonemizer_identity=trained)
    assert len(warnings) == 1
    assert "lang: trained 'es-ES', running 'pt-PT'" in warnings[0]
    # the revision is the version string of the backend, so it does not depend on the language
    assert trained["revision"] == running["revision"]
    assert "revision:" not in warnings[0]


def test_a_recorded_frontend_version_is_compared(tmp_path):
    trained = dict(_running_identity(), frontend=-1)
    warnings = _load_warnings(tmp_path, phonemizer_identity=trained)
    assert len(warnings) == 1
    assert "frontend: trained -1, running " in warnings[0]


def test_a_frontend_block_with_the_backend_revision_warns(tmp_path):
    # A block recorded through a frontend carries the backend's revision, not the
    # frontend's revision, so comparing it with the frontend identity names
    # ``revision``.
    model = tmp_path / "pt_PT-provenance.onnx"
    _write_model(model)
    (tmp_path / "pt_PT-provenance.onnx.json").write_text(
        json.dumps(copy.deepcopy(CONFIG)), encoding="utf-8")
    frontend = asdict(TTSVoice.load(model).frontend.identity())
    backend = asdict(get_phonemizer(PhonemeType.ORTHOGRAPHY2IPA).identity(LANG))
    trained = dict(backend, frontend=frontend["frontend"])
    warnings = _load_warnings(tmp_path, phonemizer_identity=trained)
    assert len(warnings) == 1
    assert f"revision: trained {backend['revision']!r}, running {frontend['revision']!r}" in warnings[0]


@pytest.mark.parametrize("config", [{}, {"phonemizer_identity": None}, {"phonemizer_identity": {}}])
def test_no_block_gives_no_warning(tmp_path, config):
    assert _load_warnings(tmp_path, **config) == []


@pytest.mark.parametrize("block", ["orthography2ipa", ["orthography2ipa"], {"backend": "orthography2ipa"},
                                   dict(backend="orthography2ipa", distribution="orthography2ipa",
                                        version="1", lang=LANG, colour="red")])
def test_a_malformed_block_warns_and_the_voice_loads(tmp_path, block):
    warnings = _load_warnings(tmp_path, phonemizer_identity=block)
    assert len(warnings) == 1
    assert "pt_PT-provenance" in warnings[0]
    assert f"cannot compare phonemizer_identity {block!r} with the running phonemizer" in warnings[0]


@pytest.mark.parametrize("block", ["", [], 0])
def test_a_falsy_non_dict_block_warns_and_the_voice_loads(tmp_path, block):
    warnings = _load_warnings(tmp_path, phonemizer_identity=block)
    assert len(warnings) == 1
    assert "pt_PT-provenance" in warnings[0]
    assert f"cannot compare phonemizer_identity {block!r} with the running phonemizer" in warnings[0]


def test_a_running_phonemizer_that_cannot_name_itself_warns(tmp_path):
    with patch("scriptconv.frontend.TextFrontend.identity", side_effect=RuntimeError("no identity")):
        warnings = _load_warnings(tmp_path, phonemizer_identity=_running_identity())
    assert len(warnings) == 1
    assert "pt_PT-provenance" in warnings[0]
    assert "cannot compare phonemizer_identity" in warnings[0]


def test_a_unicode_voice_with_a_matching_block_gives_no_warning(tmp_path):
    config = {"phoneme_type": "unicode", "alphabet": "unicode",
              "phonemizer_identity": _unicode_identity()}
    assert _load_warnings(tmp_path, **config) == []


def test_a_unicode_voice_with_a_changed_block_warns(tmp_path):
    trained = dict(_unicode_identity(), backend="graphemes")
    config = {"phoneme_type": "unicode", "alphabet": "unicode",
              "phonemizer_identity": trained}
    warnings = _load_warnings(tmp_path, **config)
    assert len(warnings) == 1
    assert "pt_PT-provenance" in warnings[0]
    assert "backend: trained 'graphemes', running 'unicode'" in warnings[0]


def test_a_raw_text_engine_gives_no_warning(tmp_path):
    """Orpheus reads text through its own tokenizer, so no phonemizer makes its input."""
    trained = dict(_running_identity(), revision=OTHER_REVISION, lang="es-ES")
    config = {"engine": "orpheus", "phoneme_type": None, "alphabet": None, "phonemizer_identity": trained}
    assert _load_warnings(tmp_path, **config) == []


def test_load_returns_before_the_check_finishes(tmp_path):
    """A slow ``identity`` does not hold the load: it returns while the check still runs.

    The provenance check runs in a daemon thread, so a 2 s identity must not
    make ``TTSVoice.load`` take 2 s. The warning still appears once the thread
    ends.
    """
    model = tmp_path / "pt_PT-provenance.onnx"
    _write_model(model)
    (tmp_path / "pt_PT-provenance.onnx.json").write_text(
        json.dumps({**copy.deepcopy(CONFIG),
                    "phonemizer_identity": dict(_running_identity(), revision=OTHER_REVISION)}),
        encoding="utf-8")

    original_identity = TextFrontend.identity

    def slow_identity(self):
        time.sleep(2)
        return original_identity(self)

    with patch.object(TextFrontend, "identity", slow_identity), patch("phoonnx.voice.LOG") as log:
        started = time.monotonic()
        voice = TTSVoice.load(model)
        elapsed = time.monotonic() - started
        assert elapsed < 1.0, f"load waited {elapsed:.2f}s for the provenance check"
        assert voice._provenance_thread is not None
        voice._provenance_thread.join()
    warnings = [str(call.args[0]) for call in log.warning.call_args_list]
    assert len(warnings) == 1
    assert "pt_PT-provenance" in warnings[0]
    assert "was trained with another phonemizer" in warnings[0]


def test_no_block_starts_no_thread(tmp_path):
    """A voice with no provenance block starts no background check."""
    model = tmp_path / "pt_PT-provenance.onnx"
    _write_model(model)
    (tmp_path / "pt_PT-provenance.onnx.json").write_text(
        json.dumps(copy.deepcopy(CONFIG)), encoding="utf-8")
    voice = TTSVoice.load(model)
    assert voice._provenance_thread is None
