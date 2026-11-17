"""Offline loads of a pinned voice from a hub cache filled at another commit.

The voice index pins every file to a commit. A device that cached a voice at an older
commit, or that holds only commit snapshots and no ``refs/main`` (the cache a pinned
install writes), must still load the voice offline after the pin moves. Online, the pinned
commit is fetched. These tests build the cache layout the hub client writes (blobs, refs,
snapshot symlinks) and read it through the real ``hf_hub_download``.
"""
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from huggingface_hub import constants
from huggingface_hub.errors import LocalEntryNotFoundError

from phoonnx import model_manager
from phoonnx.model_manager import _cached_path, _hf_fetch

OLD = "1" * 40
MID = "2" * 40
NEW = "3" * 40
REPO = "Phoonnx/a-repo"


def url(path, rev=NEW, repo=REPO):
    return f"https://huggingface.co/{repo}/resolve/{rev}/{path}"


class _Cache(unittest.TestCase):

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self._dir.cleanup)
        self.cache = Path(self._dir.name)
        for patcher in (patch.object(constants, "HF_HUB_CACHE", str(self.cache)),
                        patch.object(model_manager, "HF_HUB_CACHE", str(self.cache)),
                        patch.object(constants, "HF_HUB_OFFLINE", True),
                        patch.dict(model_manager._OFFLINE_SNAPSHOTS, clear=True)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.log = patch.object(model_manager, "LOG").start()
        self.addCleanup(patch.stopall)

    def snapshot(self, sha, files, mtime, repo=REPO, main=False):
        """Write ``files`` (path -> bytes) as the hub does, at commit ``sha``."""
        root = self.cache / f"models--{repo.replace('/', '--')}"
        (root / "blobs").mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            blob = root / "blobs" / hashlib.sha256(data).hexdigest()
            blob.write_bytes(data)
            link = root / "snapshots" / sha / name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(blob)
        os.utime(root / "snapshots" / sha, (mtime, mtime))
        if main:
            (root / "refs").mkdir(exist_ok=True)
            (root / "refs" / "main").write_text(sha)

    def warnings(self):
        return [str(call.args[0]) for call in self.log.warning.call_args_list]


class TestOfflineFallback(_Cache):

    def test_an_older_commit_cache_loads_offline_with_a_warning(self):
        self.snapshot(OLD, {"config.json": b"old"}, mtime=1000, main=True)
        self.assertEqual(_hf_fetch(url("config.json")).read_bytes(), b"old")
        [warning] = self.warnings()
        self.assertIn(REPO, warning)
        self.assertIn(NEW, warning)
        self.assertIn(OLD, warning)

    def test_commit_snapshots_without_refs_load_the_newest_after_a_pin_bump(self):
        self.snapshot(OLD, {"config.json": b"old"}, mtime=1000)
        self.snapshot(MID, {"config.json": b"mid"}, mtime=2000)
        self.assertFalse((self.cache / "models--Phoonnx--a-repo" / "refs").exists())
        self.assertEqual(_hf_fetch(url("config.json")).read_bytes(), b"mid")

    def test_the_snapshot_refs_main_names_wins_over_a_newer_one(self):
        self.snapshot(OLD, {"config.json": b"old"}, mtime=1000, main=True)
        self.snapshot(MID, {"config.json": b"mid"}, mtime=2000)
        self.assertEqual(_hf_fetch(url("config.json")).read_bytes(), b"old")

    def test_a_snapshot_without_the_file_is_passed_over(self):
        self.snapshot(OLD, {"config.json": b"old"}, mtime=1000)
        self.snapshot(MID, {"README.md": b"card"}, mtime=2000, main=True)
        self.assertEqual(_hf_fetch(url("config.json")).read_bytes(), b"old")

    def test_the_pinned_snapshot_is_read_when_it_holds_the_file(self):
        self.snapshot(OLD, {"config.json": b"old"}, mtime=1000, main=True)
        self.snapshot(NEW, {"config.json": b"new"}, mtime=500)
        self.assertEqual(_hf_fetch(url("config.json")).read_bytes(), b"new")
        self.assertEqual(self.warnings(), [])

    def test_no_snapshot_with_the_file_still_raises(self):
        self.snapshot(OLD, {"README.md": b"card"}, mtime=1000, main=True)
        with self.assertRaises(LocalEntryNotFoundError):
            _hf_fetch(url("config.json"))

    def test_an_old_repo_id_cache_falls_back_too(self):
        self.snapshot(OLD, {"config.json": b"old"}, mtime=1000, repo="OpenVoiceOS/a-repo", main=True)
        path = _hf_fetch(url("config.json"))
        self.assertEqual(path.read_bytes(), b"old")
        self.assertIn("models--OpenVoiceOS--a-repo", str(path))

    def test_the_size_question_reads_the_fallback_offline_only(self):
        self.snapshot(OLD, {"model.onnx": b"graph"}, mtime=1000, main=True)
        self.assertEqual(_cached_path(url("model.onnx")).read_bytes(), b"graph")
        with patch.object(constants, "HF_HUB_OFFLINE", False):
            self.assertIsNone(_cached_path(url("model.onnx")))


class TestOneSnapshotPerFolder(_Cache):

    def test_a_graph_and_its_sidecar_come_from_one_snapshot(self):
        self.snapshot(OLD, {"model.onnx": b"graph-old", "model.onnx_data": b"data-old"}, mtime=1000, main=True)
        self.snapshot(MID, {"model.onnx": b"graph-mid", "model.onnx_data": b"data-mid"}, mtime=2000)
        self.assertEqual(_hf_fetch(url("model.onnx")).read_bytes(), b"graph-old")
        self.assertEqual(_hf_fetch(url("model.onnx_data")).read_bytes(), b"data-old")
        self.assertEqual(len(self.warnings()), 1)

    def test_a_sidecar_never_comes_from_another_commit_than_its_graph(self):
        self.snapshot(OLD, {"model.onnx": b"graph-old"}, mtime=1000, main=True)
        self.snapshot(MID, {"model.onnx": b"graph-mid", "model.onnx_data": b"data-mid"}, mtime=2000)
        self.assertEqual(_hf_fetch(url("model.onnx")).read_bytes(), b"graph-old")
        with self.assertRaises(LocalEntryNotFoundError):
            _hf_fetch(url("model.onnx_data"))

    def test_two_voices_of_one_mirror_repo_keep_their_own_snapshots(self):
        self.snapshot(OLD, {"a/model.onnx": b"voice-a"}, mtime=1000)
        self.snapshot(MID, {"b/model.onnx": b"voice-b"}, mtime=2000, main=True)
        self.assertEqual(_hf_fetch(url("b/model.onnx")).read_bytes(), b"voice-b")
        self.assertEqual(_hf_fetch(url("a/model.onnx")).read_bytes(), b"voice-a")


class TestOnline(_Cache):

    def setUp(self):
        super().setUp()
        patcher = patch.object(constants, "HF_HUB_OFFLINE", False)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.snapshot(OLD, {"config.json": b"old"}, mtime=1000, main=True)

    def test_the_pinned_bytes_are_fetched_not_the_cached_ones(self):
        def download(repo_id, filename, revision, **kwargs):
            # what the hub client does online: fetch the pinned commit into its snapshot
            self.snapshot(revision, {filename: b"new"}, mtime=3000, repo=repo_id)
            return str(self.cache / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision / filename)

        with patch.object(model_manager, "hf_hub_download", side_effect=download) as fetch:
            self.assertEqual(_hf_fetch(url("config.json")).read_bytes(), b"new")
        self.assertEqual(fetch.call_args.kwargs["revision"], NEW)
        self.assertEqual(self.warnings(), [])

    def test_a_hub_that_does_not_answer_falls_back(self):
        # hf_hub_download raises LocalEntryNotFoundError on a connection error with nothing cached
        with patch.object(model_manager, "hf_hub_download",
                          side_effect=LocalEntryNotFoundError("connection error")):
            self.assertEqual(_hf_fetch(url("config.json")).read_bytes(), b"old")


if __name__ == "__main__":
    unittest.main()
