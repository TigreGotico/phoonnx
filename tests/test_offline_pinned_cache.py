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
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

from phoonnx import model_manager
from phoonnx.model_manager import TTSModelInfo, _cached_path, _hf_fetch

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
                        patch.dict(model_manager._OFFLINE_SNAPSHOTS, clear=True),
                        patch.dict(model_manager._VOICE_SNAPSHOTS, clear=True)):
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

    def test_refs_main_does_not_outrank_a_newer_snapshot(self):
        self.snapshot(OLD, {"config.json": b"old"}, mtime=1000, main=True)
        self.snapshot(MID, {"config.json": b"mid"}, mtime=2000)
        self.assertEqual(_hf_fetch(url("config.json")).read_bytes(), b"mid")

    def test_a_pin_bump_offline_keeps_the_pinned_bytes_the_device_last_loaded(self):
        # A fetch at a sha writes no ref, so refs/main stays at the last commit an unpinned
        # install fetched. The device then loaded the pin MID, and the index moves on to NEW.
        self.snapshot(OLD, {"config.json": b"before-the-fix"}, mtime=1000, main=True)
        self.snapshot(MID, {"config.json": b"fixed"}, mtime=2000)
        self.assertEqual(_hf_fetch(url("config.json", rev=MID)).read_bytes(), b"fixed")
        self.assertEqual(_hf_fetch(url("config.json", rev=NEW)).read_bytes(), b"fixed")

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

    def test_a_renamed_repo_falls_back_in_the_cache_of_its_earlier_id(self):
        renamed = {REPO: ["Phoonnx/a-repo-before", "OpenVoiceOS/a-repo-first"]}
        with patch.dict(model_manager._RENAMED_FROM, renamed):
            self.snapshot(OLD, {"config.json": b"old"}, mtime=1000, repo="Phoonnx/a-repo-before")
            path = _hf_fetch(url("config.json"))
        self.assertEqual(path.read_bytes(), b"old")
        self.assertIn("models--Phoonnx--a-repo-before", str(path))

    def test_the_size_question_reads_the_fallback_offline_only(self):
        self.snapshot(OLD, {"model.onnx": b"graph"}, mtime=1000, main=True)
        self.assertEqual(_cached_path(url("model.onnx")).read_bytes(), b"graph")
        with patch.object(constants, "HF_HUB_OFFLINE", False):
            self.assertIsNone(_cached_path(url("model.onnx")))

    def test_a_fallback_load_is_measured_with_the_offline_flag_unset(self):
        # A hub that does not answer (connection error) with HF_HUB_OFFLINE unset: the voice
        # loads from the OLD snapshot through _hf_fetch, and the size question then reads that
        # recorded fallback instead of charging the voice 0 bytes.
        self.snapshot(OLD, {"model.onnx": b"x" * 777}, mtime=1000, main=True)
        with patch.object(constants, "HF_HUB_OFFLINE", False), \
                patch.object(model_manager, "hf_hub_download",
                             side_effect=LocalEntryNotFoundError("connection error")):
            loaded = _hf_fetch(url("model.onnx"))
            self.assertEqual(loaded.read_bytes(), b"x" * 777)
            [warning] = self.warnings()
            self.assertIn(REPO, warning)
            self.assertEqual(_cached_path(url("model.onnx")), loaded)
            info = TTSModelInfo(voice_id="v", lang="en-US", model_url=url("model.onnx"))
            self.assertEqual(info.disk_size(), 777)

    def test_no_fallback_taken_still_measures_none_with_the_offline_flag_unset(self):
        # No load has fallen back, so the uncached pinned URL has no recorded snapshot and the
        # size question stays None online, exactly as it did before the fix.
        self.snapshot(OLD, {"model.onnx": b"graph"}, mtime=1000, main=True)
        with patch.object(constants, "HF_HUB_OFFLINE", False):
            self.assertIsNone(_cached_path(url("model.onnx")))


class TestOneSnapshotPerFolder(_Cache):

    def test_a_graph_and_its_sidecar_come_from_one_snapshot(self):
        self.snapshot(OLD, {"model.onnx": b"graph-old", "model.onnx_data": b"data-old"}, mtime=3000)
        self.snapshot(MID, {"model.onnx": b"graph-mid", "model.onnx_data": b"data-mid"}, mtime=2000)
        self.assertEqual(_hf_fetch(url("model.onnx")).read_bytes(), b"graph-old")
        self.assertEqual(_hf_fetch(url("model.onnx_data")).read_bytes(), b"data-old")
        self.assertEqual(len(self.warnings()), 1)

    def test_a_sidecar_never_comes_from_another_commit_than_its_graph(self):
        self.snapshot(OLD, {"model.onnx": b"graph-old"}, mtime=3000)
        self.snapshot(MID, {"model.onnx": b"graph-mid", "model.onnx_data": b"data-mid"}, mtime=2000)
        self.assertEqual(_hf_fetch(url("model.onnx")).read_bytes(), b"graph-old")
        with self.assertRaises(LocalEntryNotFoundError):
            _hf_fetch(url("model.onnx_data"))

    def test_two_voices_of_one_mirror_repo_keep_their_own_snapshots(self):
        self.snapshot(OLD, {"a/model.onnx": b"voice-a"}, mtime=1000)
        self.snapshot(MID, {"b/model.onnx": b"voice-b"}, mtime=2000, main=True)
        self.assertEqual(_hf_fetch(url("b/model.onnx")).read_bytes(), b"voice-b")
        self.assertEqual(_hf_fetch(url("a/model.onnx")).read_bytes(), b"voice-a")

    def test_a_folder_the_pinned_snapshot_holds_in_part_never_falls_back(self):
        # a download cut off after the config of the new pin: the graph must not come from
        # an older commit than the config, so the load fails as it does without pins
        self.snapshot(OLD, {"config.json": b"config-old", "model.onnx": b"graph-old"}, mtime=1000)
        self.snapshot(NEW, {"config.json": b"config-new"}, mtime=2000)
        self.assertEqual(_hf_fetch(url("config.json")).read_bytes(), b"config-new")
        with self.assertRaises(LocalEntryNotFoundError):
            _hf_fetch(url("model.onnx"))
        self.assertIsNone(_cached_path(url("model.onnx")))
        self.assertEqual(self.warnings(), [])

    def test_other_folders_of_a_partly_pinned_repo_still_fall_back(self):
        self.snapshot(OLD, {"config.json": b"root-old", "b/model.onnx": b"voice-b-old"}, mtime=1000)
        self.snapshot(NEW, {"a/model.onnx": b"voice-a-new"}, mtime=2000)
        self.assertEqual(_hf_fetch(url("b/model.onnx")).read_bytes(), b"voice-b-old")
        self.assertEqual(_hf_fetch(url("config.json")).read_bytes(), b"root-old")


class TestOneSnapshotPerVoice(_Cache):
    """A voice whose files sit in two folders of one repo loads from one commit.

    The pockettts layout: the graph is published per language, the voice states under
    ``<lang>/voices/``. Loading one voice must not read its graph from the commit that
    last cached the graph folder and its voice state from an older one.
    """

    GRAPH = "en/model.onnx"
    ALBA = "en/voices/alba.bin"
    ANNA = "en/voices/anna.bin"
    PAULA = "en/voices/paula.bin"

    def cache_two_voices(self):
        """OLD holds the graph and alba, the newer MID the graph and anna. NEW is not cached."""
        self.snapshot(OLD, {self.GRAPH: b"graph-old", self.ALBA: b"alba-old"}, mtime=1000)
        self.snapshot(MID, {self.GRAPH: b"graph-mid", self.ANNA: b"anna-mid"}, mtime=2000)

    def voice(self, state, voice_id="alba"):
        return TTSModelInfo(voice_id=voice_id, lang="en-US",
                            model_url=url(self.GRAPH), style_url=url(state))

    def test_a_voice_loads_its_graph_and_its_state_from_the_same_snapshot(self):
        self.cache_two_voices()
        with model_manager._voice_scope(self.voice(self.ALBA)):
            graph = _hf_fetch(url(self.GRAPH))
            alba = _hf_fetch(url(self.ALBA))
            self.assertEqual(graph.read_bytes(), b"graph-old")
            self.assertEqual(alba.read_bytes(), b"alba-old")
            # the size question reads the files the load reads
            self.assertEqual(_cached_path(url(self.GRAPH)), graph)
            self.assertEqual(_cached_path(url(self.ALBA)), alba)
        [warning] = self.warnings()
        self.assertIn(REPO, warning)
        self.assertIn(NEW, warning)
        self.assertIn(OLD, warning)
        # the choice is the voice load's, not the folder's
        self.assertFalse(model_manager._OFFLINE_SNAPSHOTS)

    def test_a_second_voice_of_the_same_repo_takes_both_files_from_mid(self):
        self.cache_two_voices()
        with model_manager._voice_scope(self.voice(self.ANNA, voice_id="anna")):
            self.assertEqual(_hf_fetch(url(self.GRAPH)).read_bytes(), b"graph-mid")
            self.assertEqual(_hf_fetch(url(self.ANNA)).read_bytes(), b"anna-mid")
        self.assertEqual(len(self.warnings()), 1)

    def test_a_voice_no_snapshot_holds_in_full_does_not_load_mixed(self):
        self.cache_two_voices()
        self.snapshot("4" * 40, {self.PAULA: b"paula-late"}, mtime=3000)
        with model_manager._voice_scope(self.voice(self.PAULA, voice_id="paula")):
            with self.assertRaises(LocalEntryNotFoundError):
                _hf_fetch(url(self.GRAPH))
            with self.assertRaises(LocalEntryNotFoundError):
                _hf_fetch(url(self.PAULA))
            self.assertIsNone(_cached_path(url(self.PAULA)))
        self.assertEqual(self.warnings(), [])

    def test_disk_size_measures_the_files_of_one_snapshot(self):
        # graph-old is 9 bytes, the newer graph-mid 99, alba-old 5: the voice is 14 bytes
        self.snapshot(OLD, {self.GRAPH: b"g" * 9, self.ALBA: b"a" * 5}, mtime=1000)
        self.snapshot(MID, {self.GRAPH: b"g" * 99, self.ANNA: b"n" * 7}, mtime=2000)
        self.assertEqual(self.voice(self.ALBA).disk_size(), 9 + 5)
        self.assertEqual(self.voice(self.ANNA, voice_id="anna").disk_size(), 99 + 7)

    def test_a_voice_load_fallback_is_measured_with_the_offline_flag_unset(self):
        # the hub does not answer and HF_HUB_OFFLINE is unset: the load falls back to OLD for
        # both files, and a later size question reads the same two files
        self.snapshot(OLD, {self.GRAPH: b"g" * 9, self.ALBA: b"a" * 5}, mtime=1000)
        self.snapshot(MID, {self.GRAPH: b"g" * 99, self.ANNA: b"n" * 7}, mtime=2000)
        alba = self.voice(self.ALBA)
        with patch.object(constants, "HF_HUB_OFFLINE", False), \
                patch.object(model_manager, "hf_hub_download",
                             side_effect=LocalEntryNotFoundError("connection error")):
            with model_manager._voice_scope(alba):
                self.assertEqual(_hf_fetch(url(self.GRAPH)).read_bytes(), b"g" * 9)
            self.assertEqual(alba.disk_size(), 9 + 5)

    def test_the_external_data_of_a_graph_follows_the_voice_snapshot(self):
        sidecar = self.GRAPH + "_data"
        self.snapshot(OLD, {self.GRAPH: b"graph-old", sidecar: b"data-old", self.ALBA: b"alba-old"},
                      mtime=1000)
        self.snapshot(MID, {self.GRAPH: b"graph-mid", sidecar: b"data-mid", self.ANNA: b"anna-mid"},
                      mtime=2000)
        with model_manager._voice_scope(self.voice(self.ALBA)):
            self.assertEqual(_hf_fetch(url(self.GRAPH)).read_bytes(), b"graph-old")
            self.assertEqual(_hf_fetch(url(sidecar)).read_bytes(), b"data-old")
            self.assertEqual(_cached_path(url(sidecar)).read_bytes(), b"data-old")

    def test_outside_a_voice_scope_the_folder_rule_still_decides(self):
        # the behaviour before the per-voice rule: the newest snapshot holding the graph,
        # and the only snapshot holding alba
        self.cache_two_voices()
        self.assertEqual(_hf_fetch(url(self.GRAPH)).read_bytes(), b"graph-mid")
        self.assertEqual(_hf_fetch(url(self.ALBA)).read_bytes(), b"alba-old")


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

    def test_a_file_missing_at_the_pin_online_raises_and_never_falls_back(self):
        # a wrong pin must show: the cached snapshot holds the file, and it is not read
        with patch.object(model_manager, "hf_hub_download",
                          side_effect=EntryNotFoundError("404 at the pinned commit")):
            with self.assertRaises(EntryNotFoundError) as raised:
                _hf_fetch(url("config.json"))
        self.assertNotIsInstance(raised.exception, LocalEntryNotFoundError)
        self.assertEqual(self.warnings(), [])

    def test_a_hub_that_does_not_answer_falls_back(self):
        # hf_hub_download raises LocalEntryNotFoundError on a connection error with nothing cached
        with patch.object(model_manager, "hf_hub_download",
                          side_effect=LocalEntryNotFoundError("connection error")):
            self.assertEqual(_hf_fetch(url("config.json")).read_bytes(), b"old")


if __name__ == "__main__":
    unittest.main()
