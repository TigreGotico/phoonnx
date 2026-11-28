"""A voice cached under a repo id that has since been renamed loads from that cache.

The cache built here is the layout the hub writes: blobs, ``refs/main`` and a
snapshot of symlinks into the blobs. The offline cases run the loader in a child
process with ``HF_HUB_OFFLINE=1``, as a user without a network would.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from phoonnx import model_manager as mm

RENAMED = "Phoonnx/shami-apc-shami"
EARLIER = ["Phoonnx/shami-apc", "Phoonnx/phoonnx-shami", "OpenVoiceOS/phoonnx-shami"]
COMMIT = "0" * 40

LOAD = """
import sys
from phoonnx.model_manager import TTSModelInfo
info = TTSModelInfo(voice_id="test/moved", lang="apc", model_url=sys.argv[1])
print(info.download_model().resolve())
"""


def cache_repo(cache: Path, repo_id: str, files: dict) -> Path:
    """Write ``files`` into ``cache`` under ``repo_id`` as the hub does; give the repo folder."""
    root = cache / f"models--{repo_id.replace('/', '--')}"
    (root / "refs").mkdir(parents=True)
    (root / "refs" / "main").write_text(COMMIT)
    (root / "blobs").mkdir()
    for name, data in files.items():
        blob = root / "blobs" / f"{name}-blob"
        blob.write_bytes(data)
        link = root / "snapshots" / COMMIT / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob)
    return root


def load_offline(cache: Path, url: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "HF_HUB_CACHE": str(cache), "HF_HUB_OFFLINE": "1"}
    return subprocess.run([sys.executable, "-c", LOAD, url], env=env, capture_output=True, text=True)


def url(repo_id: str, path: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{path}"


def test_the_table_lists_the_earlier_ids_newest_first():
    assert mm._earlier_ids(RENAMED) == EARLIER
    assert mm._earlier_ids("Phoonnx/vits-en_us-ryan-high-espeak") == ["OpenVoiceOS/vits-en_us-ryan-high-espeak"]


def test_no_index_url_names_an_earlier_id():
    earlier = {old for olds in mm._RENAMED_FROM.values() for old in olds}
    for path in mm.TTSModelManager.voice_index_files():
        for entry in json.loads(path.read_text(encoding="utf-8")).values():
            for value in json.dumps(entry).split('"'):
                match = mm._HF_URL.match(value)
                assert not (match and match["repo"] in earlier), (path.name, value)


@pytest.mark.parametrize("old", EARLIER)
def test_offline_load_reads_the_copy_cached_under_an_earlier_id(tmp_path, old):
    root = cache_repo(tmp_path, old, {"model.onnx": b"graph cached under " + old.encode()})
    result = load_offline(tmp_path, url(RENAMED, "model.onnx"))
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == (root / "blobs" / "model.onnx-blob").resolve()
    assert not (tmp_path / f"models--{RENAMED.replace('/', '--')}").exists()


def test_the_newest_earlier_id_wins_when_the_cache_holds_two(tmp_path):
    cache_repo(tmp_path, EARLIER[2], {"model.onnx": b"oldest"})
    newer = cache_repo(tmp_path, EARLIER[1], {"model.onnx": b"newer"})
    result = load_offline(tmp_path, url(RENAMED, "model.onnx"))
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == (newer / "blobs" / "model.onnx-blob").resolve()


@pytest.mark.parametrize("repo_id", [RENAMED, "Phoonnx/vits-en_us-ryan-high-espeak"])
def test_a_repo_cached_only_under_its_own_id_loads_as_before(tmp_path, repo_id):
    root = cache_repo(tmp_path, repo_id, {"model.onnx": b"graph"})
    result = load_offline(tmp_path, url(repo_id, "model.onnx"))
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == (root / "blobs" / "model.onnx-blob").resolve()
    assert sorted(p.name for p in tmp_path.iterdir()) == [root.name]


def test_online_a_graph_cached_under_the_old_id_is_not_downloaded_again(tmp_path, monkeypatch):
    """An older hub client keeps each blob in the folder of its repo, so a file cached under the
    old id is not found under the new id. One 2 KB graph comes from the Hub into a scratch cache
    and is laid out that way under the old id. Loading it through the new URL must not download it."""
    from huggingface_hub import constants, file_download, hf_hub_download
    old, new, name = "Phoonnx/phoonnx-chatterbox-turbo", "Phoonnx/chatterbox-en_us-turbo-unicode", "embed_tokens.onnx"
    fetched = Path(hf_hub_download(old, name, cache_dir=tmp_path / "scratch"))
    data = fetched.read_bytes()
    cache = tmp_path / "hub"
    root = cache / f"models--{old.replace('/', '--')}"
    blob = root / "blobs" / hashlib.sha256(data).hexdigest()
    blob.parent.mkdir(parents=True)
    blob.write_bytes(data)
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text(fetched.parent.name)
    (root / "snapshots" / fetched.parent.name).mkdir(parents=True)
    (root / "snapshots" / fetched.parent.name / name).symlink_to(blob)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(mm, "HF_HUB_CACHE", str(cache))

    def no_download(*args, **kwargs):
        raise AssertionError("the file was downloaded again")

    with patch.object(file_download, "http_get", no_download), patch.object(file_download, "xet_get", no_download):
        path = mm._resolve(url(new, name))
    assert path.resolve() == blob.resolve()
    assert not (cache / f"models--{new.replace('/', '--')}").exists()
