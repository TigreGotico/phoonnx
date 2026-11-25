"""A Runtime gives a voice load its files and opens its graphs, as phoonnx.js needs in Pyodide.

The voice reads graphemes, so the load needs no phonemizer model. Its ids are
spaced and start high, so an id cannot equal the position of its letter.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from phoonnx import model_manager
from phoonnx.model_manager import TTSModelInfo
from phoonnx.runtime import Runtime, current_runtime, using_runtime
from phoonnx.voice import TTSVoice

HUB = "https://huggingface.co/Phoonnx/demo/resolve/0123456789abcdef0123456789abcdef01234567"
LETTERS = "abcdefghijklmnop "
CONFIG = {"phoonnx_version": "1.0.0", "engine": "vits", "phoneme_type": "graphemes", "alphabet": "unicode",
          "lang_code": "en-US", "audio": {"sample_rate": 22050},
          "phoneme_id_map": {c: [100 + 3 * i] for i, c in enumerate(LETTERS)}}
NO_SESSION = object()


def ids_of(text):
    return [100 + 3 * LETTERS.index(c) for c in text]


def no_network(*args, **kwargs):
    raise AssertionError("the load went to the network")


def no_onnxruntime(*args, **kwargs):
    raise AssertionError("the load opened a graph with onnxruntime")


@pytest.fixture
def files(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(CONFIG), encoding="utf-8")
    # the graph is never opened, so its file need not exist
    return {f"{HUB}/config.json": config, f"{HUB}/model.onnx": tmp_path / "model.onnx"}


def runtime_of(files, opened):
    def open_session(path, providers=None, threads=None):
        opened.append(Path(path))
        return NO_SESSION
    return Runtime(fetch=lambda url: files[url], open_session=open_session)


def test_an_index_voice_loads_from_the_runtime_files_and_sessions(files):
    opened = []
    info = TTSModelInfo(voice_id="demo", lang="en-US", engine="vits",
                        model_url=f"{HUB}/model.onnx", config_url=f"{HUB}/config.json")
    with patch("phoonnx.model_manager._hf_fetch", no_network), \
            patch("phoonnx.model_manager._direct_stream", no_network), \
            patch("phoonnx.providers._build_session", no_onnxruntime), \
            patch("phoonnx.model_manager._sidecar_urls", no_network):
        voice = info.load(runtime=runtime_of(files, opened))
    assert voice.session is NO_SESSION
    assert opened == [files[f"{HUB}/model.onnx"]]
    assert [list(u.ids) for u in voice.prepare("bad")] == [ids_of("bad")]
    assert current_runtime() is None


def test_a_local_voice_opens_its_graph_through_the_runtime(files):
    model = files[f"{HUB}/model.onnx"]
    model.with_suffix(".onnx.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    opened = []
    with patch("phoonnx.providers._build_session", no_onnxruntime):
        voice = TTSVoice.load(str(model), runtime=runtime_of(files, opened))
    assert voice.session is NO_SESSION
    assert opened == [model]
    assert current_runtime() is None


def test_without_open_session_the_graph_lists_its_external_data(files):
    """A runtime that only fetches leaves the graph to onnxruntime, which needs the sidecars beside it."""
    listed = []
    info = TTSModelInfo(voice_id="demo", lang="en-US", engine="vits", model_url=f"{HUB}/model.onnx")
    with patch("phoonnx.model_manager._sidecar_urls", lambda url, path: listed.append(url) or []), \
            using_runtime(Runtime(fetch=lambda url: files[url])):
        assert info._fetch_onnx(f"{HUB}/model.onnx") == files[f"{HUB}/model.onnx"]
    assert listed == [f"{HUB}/model.onnx"]


def test_a_runtime_holds_for_its_block_only():
    outer, inner = Runtime(), Runtime()
    with using_runtime(outer):
        with using_runtime(None):
            assert current_runtime() is outer
        with using_runtime(inner):
            assert current_runtime() is inner
        assert current_runtime() is outer
    assert current_runtime() is None
    assert model_manager.current_runtime is current_runtime
