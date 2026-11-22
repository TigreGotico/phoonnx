"""The voice index names one Hub repo per model, and a voice cached from the shared mirror repo still loads."""
import json
import re
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from huggingface_hub.errors import RemoteEntryNotFoundError, LocalEntryNotFoundError

from phoonnx import model_manager
from phoonnx.model_manager import _HF_URL, _hf_fetch
from tests.test_offline_pinned_cache import _Cache

PACKAGE = Path(model_manager.__file__).parent
SPLIT = json.loads((PACKAGE / "split_repos.json").read_text(encoding="utf-8"))
MMS = "Phoonnx/vits-abi-mms-graphemes"
OUTETTS = "Phoonnx/outetts-multi-outeai-0.6b-unicode"
MIRROR_MMS = SPLIT[MMS]["from"]
FOLDER = SPLIT[MMS]["folder"]
PINNED = "a" * 40


def index_urls():
    urls = set()
    for index_file in (PACKAGE / "voice_index").glob("*.json"):
        urls.update(re.findall(r'"(https://huggingface\.co/[^"]+)"', index_file.read_text(encoding="utf-8")))
    return urls


def index_url(repo, path):
    [url] = {u for u in index_urls() if _HF_URL.match(u)["repo"] == repo and _HF_URL.match(u)["path"] == path}
    return url


class TestCacheFromTheMirror(_Cache):
    """The cache holds the mirror repo at a commit other than the per-model repo's, as a device that loaded
    the voice before the split does."""

    def test_a_voice_cached_from_the_mirror_loads_offline_from_the_mirror_folder(self):
        self.snapshot(PINNED, {f"{FOLDER}/model.onnx": b"graph"}, mtime=1000, repo=MIRROR_MMS)
        path = _hf_fetch(index_url(MMS, "model.onnx"))
        self.assertEqual(path.read_bytes(), b"graph")
        self.assertIn(f"models--{MIRROR_MMS.replace('/', '--')}", str(path))

    def test_the_mirror_commit_in_the_table_is_the_one_asked_for(self):
        self.snapshot(SPLIT[MMS]["rev"], {f"{FOLDER}/model.onnx": b"pinned"}, mtime=1000, repo=MIRROR_MMS)
        self.snapshot(PINNED, {f"{FOLDER}/model.onnx": b"newer"}, mtime=2000, repo=MIRROR_MMS)
        self.assertEqual(_hf_fetch(index_url(MMS, "model.onnx")).read_bytes(), b"pinned")
        self.assertEqual(self.warnings(), [])

    def test_a_voice_cached_from_the_openvoiceos_mirror_loads_offline(self):
        old_mirror = "OpenVoiceOS/" + MIRROR_MMS.split("/", 1)[1]
        self.snapshot(PINNED, {f"{FOLDER}/model.onnx": b"graph"}, mtime=1000, repo=old_mirror)
        path = _hf_fetch(index_url(MMS, "model.onnx"))
        self.assertEqual(path.read_bytes(), b"graph")
        self.assertIn(f"models--{old_mirror.replace('/', '--')}", str(path))

    def test_a_file_from_outside_the_folder_keeps_its_mirror_path(self):
        shared = "speakers/en-female-1-neutral.json"
        self.assertIn(shared, SPLIT[OUTETTS]["kept"])
        mirror = SPLIT[OUTETTS]["from"]
        self.snapshot(PINNED, {shared: b"speaker", f"{SPLIT[OUTETTS]['folder']}/model.onnx": b"graph"},
                      mtime=1000, repo=mirror)
        self.assertEqual(_hf_fetch(index_url(OUTETTS, shared)).read_bytes(), b"speaker")
        self.assertEqual(_hf_fetch(index_url(OUTETTS, "model.onnx")).read_bytes(), b"graph")

    def test_a_voice_load_takes_every_file_through_the_mirror(self):
        files = {f"{FOLDER}/model.onnx": b"graph", f"{FOLDER}/config.json": b"{}"}
        self.snapshot(PINNED, files, mtime=1000, repo=MIRROR_MMS)
        info = model_manager.TTSModelInfo(voice_id="mms/abi", lang="abi",
                                          model_url=index_url(MMS, "model.onnx"),
                                          config_url=index_url(MMS, "config.json"))
        with model_manager._voice_scope(info):
            paths = [_hf_fetch(info.model_url), _hf_fetch(info.config_url)]
        self.assertEqual([p.read_bytes() for p in paths], [b"graph", b"{}"])
        for path in paths:
            self.assertIn(f"models--{MIRROR_MMS.replace('/', '--')}/snapshots/{PINNED}/{FOLDER}/", str(path))

    def test_a_cache_without_the_mirror_does_not_reach_for_it(self):
        self.snapshot(PINNED, {f"{FOLDER}/model.onnx": b"graph"}, mtime=1000, repo="Phoonnx/phoonnx-other")
        with self.assertRaises(LocalEntryNotFoundError):
            _hf_fetch(index_url(MMS, "model.onnx"))


class TestFetchWithoutTheMirrorCache(unittest.TestCase):

    def test_the_fetch_goes_to_the_model_repo_at_its_own_commit(self):
        with patch.object(model_manager, "HF_HUB_CACHE", "/nonexistent-cache"), \
                patch.object(model_manager, "hf_hub_download", return_value="/got") as download:
            _hf_fetch(index_url(MMS, "model.onnx"))
        [call] = download.call_args_list
        match = _HF_URL.match(index_url(MMS, "model.onnx"))
        self.assertEqual((call.kwargs["repo_id"], call.kwargs["revision"], call.kwargs["filename"]),
                         (MMS, match["rev"], "model.onnx"))
        self.assertNotEqual(match["rev"], SPLIT[MMS]["rev"])

    def test_a_404_from_the_model_repo_is_not_hidden(self):
        with patch.object(model_manager, "HF_HUB_CACHE", "/nonexistent-cache"), \
                patch.object(model_manager, "hf_hub_download",
                             side_effect=RemoteEntryNotFoundError("404", response=MagicMock(status_code=404))):
            with self.assertRaises(RemoteEntryNotFoundError):
                _hf_fetch(index_url(MMS, "model.onnx"))


class TestTheIndexNamesTheModelRepos(unittest.TestCase):

    def test_every_table_entry_names_its_mirror_folder_and_commit(self):
        for repo, entry in SPLIT.items():
            with self.subTest(repo=repo):
                self.assertRegex(entry["rev"], r"^[0-9a-f]{40}$")
                self.assertTrue(entry["from"].startswith("Phoonnx/phoonnx-"))
                self.assertTrue(entry["folder"])

    def test_every_url_of_a_model_repo_is_pinned_to_one_commit(self):
        commits = {}
        for url in index_urls():
            match = _HF_URL.match(url)
            if match["repo"] in SPLIT:
                commits.setdefault(match["repo"], set()).add(match["rev"])
        self.assertEqual(set(commits), set(SPLIT))
        for repo, revs in commits.items():
            with self.subTest(repo=repo):
                [rev] = revs
                self.assertRegex(rev, r"^[0-9a-f]{40}$")

    def test_no_url_is_left_on_a_mirror_folder_that_has_a_repo(self):
        folders = {(entry["from"], entry["folder"]) for entry in SPLIT.values()}
        left = [url for url in index_urls()
                if (_HF_URL.match(url)["repo"], _HF_URL.match(url)["path"].split("/", 1)[0]) in folders]
        self.assertEqual(left, [])


if __name__ == "__main__":
    unittest.main()
