import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import onnxruntime
import requests

from huggingface_hub.errors import (EntryNotFoundError, HfHubHTTPError,
                                    LocalEntryNotFoundError)

from phoonnx.model_manager import (TTSModelInfo, TTSModelManager, _direct_dir,
                                   _hf_fetch)
from phoonnx.config import Engine, PhonemeType, Alphabet

HUB = "https://huggingface.co/an-org/a-repo/resolve/main"


def _mock_response(status_code=200, json_data=None, content=b"", text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.text = text
    resp.iter_content.return_value = [content] if content else []
    resp.headers = {}
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("No JSON")
    if status_code >= 400:
        # requests attaches the response to the error; the sidecar handler
        # reads its status to tell "absent" from "failed"
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status_code} error", response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


def _hub_500():
    """The hub client wraps a server error, keeping the response on it."""
    return HfHubHTTPError("500 Server Error", response=MagicMock(status_code=500))


class FakeHub:
    """A stand-in for ``huggingface_hub.hf_hub_download``.

    It copies the real cache layout, which is the whole point of this change:
    content-addressed blobs in one flat directory, and a snapshot directory of
    links named after the files in the repo. onnxruntime resolves an
    external-data reference against the directory the graph itself resolves to,
    so a fixture that puts every file in one plain directory would pass for
    layouts the real cache rejects.
    """

    def __init__(self, root: Path, files=None):
        self.root = root
        self.files = dict(files or {})   # "path/in/repo" -> bytes
        self.downloads = []              # every call, in order
        self.errors = {}                 # "path/in/repo" -> exception to raise

    def __call__(self, repo_id, filename, revision, **kwargs):
        self.downloads.append((repo_id, filename, revision))
        if filename in self.errors:
            raise self.errors[filename]
        if filename not in self.files:
            raise EntryNotFoundError(f"404 for {filename}")
        data = self.files[filename]
        blobs = self.root / repo_id / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        blob = blobs / hashlib.sha256(data).hexdigest()
        if not blob.exists():
            blob.write_bytes(data)
        link = self.root / repo_id / "snapshots" / revision / filename
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            os.symlink(blob, link)
        return str(link)

    def stage(self, url: str, content) -> None:
        """Publish ``content`` at ``url`` on this fake hub."""
        if isinstance(content, (dict, list)):
            content = json.dumps(content).encode("utf-8")
        elif isinstance(content, str):
            content = content.encode("utf-8")
        self.files[url.split("/resolve/")[-1].split("/", 1)[-1]
                   if "/resolve/" in url
                   else url.split("/raw/")[-1].split("/", 1)[-1]] = content

    @property
    def unique_downloads(self):
        return set(self.downloads)


class HubTestCase(unittest.TestCase):
    """Every artifact comes from the hub cache, and nothing is copied out of it.

    A voice directory owned by phoonnx no longer exists, so these tests assert
    on what the hub returns and on what reaches onnxruntime — not on files in a
    directory phoonnx manages.
    """

    def setUp(self):
        self._hubdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._hubdir.cleanup)
        self.hub_root = Path(self._hubdir.name)
        self.hub = FakeHub(self.hub_root)
        patcher = patch("phoonnx.model_manager.hf_hub_download", self.hub)
        self.addCleanup(patcher.stop)
        patcher.start()
        # a non-hub download must not land in the developer's real cache
        self._directdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._directdir.cleanup)
        dpatch = patch("phoonnx.model_manager.HF_HUB_CACHE", self._directdir.name)
        self.addCleanup(dpatch.stop)
        dpatch.start()

    def make_info(self, **kwargs):
        defaults = dict(voice_id="test/voice", lang="en-US",
                        model_url=f"{HUB}/model.onnx")
        defaults.update(kwargs)
        return TTSModelInfo(**defaults)

    def stream(self, bodies):
        """Patch ``requests.get`` to serve ``{url_suffix: response}``."""
        def side_effect(url, timeout=None, stream=None):
            for suffix, resp in bodies.items():
                if url.endswith(suffix):
                    cm = MagicMock()
                    cm.__enter__.return_value = resp
                    return cm
            raise AssertionError(f"unexpected request for {url}")
        return patch("phoonnx.model_manager.requests.get", side_effect=side_effect)


class TestArtifactsComeFromTheHub(HubTestCase):
    def test_the_graph_path_is_the_hub_s_own(self):
        self.hub.stage(f"{HUB}/model.onnx", b"onnxdata")
        info = self.make_info()
        path = info.download_model()
        self.assertEqual(path.read_bytes(), b"onnxdata")
        self.assertTrue(str(path).startswith(str(self.hub_root)),
                        f"{path} is not in the shared cache")

    def test_nothing_is_copied_into_a_phoonnx_directory(self):
        """The voice cache phoonnx used to own is gone. ``voice_path`` answers
        with the directory the hub keeps the graph in."""
        self.hub.stage(f"{HUB}/model.onnx", b"onnxdata")
        info = self.make_info()
        self.assertEqual(info.voice_path, info.download_model().parent)
        self.assertTrue(str(info.voice_path).startswith(str(self.hub_root)))

    def test_two_voices_naming_one_model_share_one_file(self):
        self.hub.stage(f"{HUB}/model.onnx", b"onnxdata")
        url = f"{HUB}/model.onnx"
        a = self.make_info(voice_id="org/first", model_url=url).download_model()
        b = self.make_info(voice_id="org/second", model_url=url).download_model()
        self.assertEqual(a, b, "both voices resolve to the one cached file")
        self.assertEqual(len({os.path.realpath(p) for p in
                              self.hub_root.rglob("model.onnx")}), 1,
                         "one physical copy in the cache")

    def test_raw_urls_are_served_by_the_hub_too(self):
        """``/raw/<rev>/`` serves the same bytes as ``/resolve/<rev>/``, so it
        must not be pushed onto the private download path."""
        self.hub.files["tokens.txt"] = b"a\nb\n"
        info = self.make_info(
            tokens_url="https://huggingface.co/an-org/a-repo/raw/main/tokens.txt")
        with patch("phoonnx.model_manager.requests.get") as mock_get:
            self.assertEqual(info.download_tokens_txt(), "a\nb\n")
        mock_get.assert_not_called()
        self.assertEqual(self.hub.downloads,
                         [("an-org/a-repo", "tokens.txt", "main")])

    def test_json_artifacts_are_read_from_the_cache(self):
        self.hub.stage(f"{HUB}/config.json", {"sample_rate": 22050})
        self.hub.stage(f"{HUB}/vocab.json", {"a": 1})
        self.hub.stage(f"{HUB}/tokenizer_config.json", {"unk_token": "<unk>"})
        info = self.make_info(config_url=f"{HUB}/config.json",
                              vocab_url=f"{HUB}/vocab.json",
                              tokenizer_config_url=f"{HUB}/tokenizer_config.json")
        with patch("phoonnx.model_manager.requests.get") as mock_get:
            self.assertEqual(info.download_config(), {"sample_rate": 22050})
            self.assertEqual(info.download_vocab(), {"a": 1})
            self.assertEqual(info.download_tokenizer_config(),
                             {"unk_token": "<unk>"})
        mock_get.assert_not_called()

    def test_a_vocab_shipped_in_the_index_entry_is_used_as_is(self):
        """``vocab_override`` carries the vocabulary inside the catalog entry;
        there is nothing to download."""
        info = self.make_info(vocab_override={"x": 3})
        self.assertEqual(info.download_vocab(), {"x": 3})
        self.assertEqual(self.hub.downloads, [])

    def test_a_missing_graph_propagates(self):
        info = self.make_info()
        with self.assertRaises(EntryNotFoundError):
            info.download_model()

    def test_hub_failure_on_the_primary_graph_propagates(self):
        self.hub.errors["model.onnx"] = _hub_500()
        with self.assertRaises(HfHubHTTPError):
            self.make_info().download_model()

    def test_aux_and_auxiliary_graphs_resolve(self):
        for name in ("speech_encoder.onnx", "embed_tokens.onnx",
                     "conditional_decoder.onnx", "extra.onnx", "speaker.bin"):
            self.hub.stage(f"{HUB}/{name}", b"payload-" + name.encode())
        self.hub.stage(f"{HUB}/model.onnx", b"onnxdata")
        info = self.make_info(
            aux_model_urls={"extra_path": f"{HUB}/extra.onnx",
                            "speaker_path": f"{HUB}/speaker.bin"},
            speech_encoder_url=f"{HUB}/speech_encoder.onnx",
            embed_tokens_url=f"{HUB}/embed_tokens.onnx",
            conditional_decoder_url=f"{HUB}/conditional_decoder.onnx")
        params = info.engine_params()
        for key in ("extra_path", "speaker_path", "speech_encoder_path",
                    "embed_tokens_path", "conditional_decoder_path"):
            self.assertTrue(str(params[key]).startswith(str(self.hub_root)),
                            f"{key} left the cache: {params[key]}")


class TestNonHubVoices(HubTestCase):
    """Self-hosted and mirrored voices are rare but legitimate. They lose the
    deduplication, not the ability to load."""

    URL = "https://mirror.example.org/v/model.onnx"

    def test_a_non_hub_url_is_downloaded_not_dropped(self):
        with self.stream({"model.onnx_data": _mock_response(404),
                          "model.onnx": _mock_response(200, content=b"selfhosted")}), \
                patch("phoonnx.model_manager.LOG") as mock_log:
            path = self.make_info(model_url=self.URL).download_model()
        self.assertIsNotNone(path, "a non-hub voice must not resolve to None")
        self.assertEqual(path.read_bytes(), b"selfhosted")
        self.assertTrue(mock_log.warning.called,
                        "the private copy must be announced")
        self.assertEqual(self.hub.downloads, [], "the hub was never asked")

    def test_the_graph_and_its_sidecar_share_one_directory(self):
        """The reason a private download still needs a directory of its own:
        onnxruntime resolves the sidecar against the graph's directory."""
        with self.stream({"model.onnx_data": _mock_response(200, content=b"weights"),
                          "model.onnx": _mock_response(200, content=b"graph")}):
            path = self.make_info(model_url=self.URL).download_model()
        self.assertEqual((path.parent / "model.onnx_data").read_bytes(), b"weights")
        self.assertEqual(path.parent, _direct_dir(self.URL))

    def test_files_published_together_land_together(self):
        self.assertEqual(_direct_dir("https://m.example/v/model.onnx"),
                         _direct_dir("https://m.example/v/vocab.json"))
        self.assertNotEqual(_direct_dir("https://m.example/v/model.onnx"),
                            _direct_dir("https://m.example/w/model.onnx"))

    def test_a_second_call_does_not_download_again(self):
        bodies = {"model.onnx_data": _mock_response(404),
                  "model.onnx": _mock_response(200, content=b"selfhosted")}
        info = self.make_info(model_url=self.URL)
        with self.stream(bodies) as mock_get:
            info.download_model()
            first = mock_get.call_count
            info.download_model()
        self.assertEqual(mock_get.call_count, first + 1,
                         "only the sidecar is re-probed, the graph is cached")

    def test_a_compressed_download_is_not_mistaken_for_truncated(self):
        """A gzipped response is complete even though it is shorter on the wire.

        ``Content-Length`` counts the compressed bytes while the stream yields
        the decoded body, so comparing them rejects a download that arrived
        perfectly intact. Transfer compression is the server's choice, not
        ours, so this is a voice that simply refuses to install.
        """
        body = b"a decoded body that is longer than its compressed length"
        resp = _mock_response(200, content=body)
        resp.headers = {"Content-Length": "12", "Content-Encoding": "gzip"}
        with self.stream({"model.onnx_data": _mock_response(404),
                          "model.onnx": resp}):
            path = self.make_info(model_url=self.URL).download_model()
        self.assertEqual(path.read_bytes(), body)

    def test_identity_encoding_still_catches_truncation(self):
        # The exemption above must not become a way to skip the size check.
        resp = _mock_response(200, content=b"partial")
        resp.headers = {"Content-Length": "999", "Content-Encoding": "identity"}
        with self.stream({"model.onnx_data": _mock_response(404),
                          "model.onnx": resp}):
            with self.assertRaises(IOError):
                self.make_info(model_url=self.URL).download_model()

    def test_a_truncated_download_is_not_left_behind(self):
        resp = _mock_response(200, content=b"partial")
        resp.headers = {"Content-Length": "999"}
        with self.stream({"model.onnx_data": _mock_response(404),
                          "model.onnx": resp}):
            with self.assertRaises(IOError):
                self.make_info(model_url=self.URL).download_model()
        self.assertFalse((_direct_dir(self.URL) / "model.onnx").exists())


class TestMetadataTimeout(unittest.TestCase):
    """The metadata lookup must give up long before the body budget.

    ``hf_hub_download`` asks the hub about the file before fetching it. If that
    call inherits the body timeout, a blackholed network stalls the caller for
    the whole budget before the download it blocks has even begun — and the
    request in front of it has usually timed out by then anyway.
    """

    def test_metadata_lookup_does_not_inherit_the_body_timeout(self):
        from phoonnx import model_manager as mm
        seen = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return "/tmp/x"

        with patch.object(mm, "hf_hub_download", side_effect=fake):
            mm._hf_fetch(f"{HUB}/model.onnx",
                         timeout=120)
        self.assertLessEqual(seen["etag_timeout"], 30,
                             "a metadata call must not wait out the body timeout")


class TestSidecarUrlKeepsTheQuery(HubTestCase):
    """A query on the model URL must not swallow the sidecar suffix.

    HuggingFace's own download links carry ``?download=true``, so this is the
    URL form an index entry most easily picks up. Appending ``_data`` to the
    whole string puts the suffix inside the query, which the hub ignores — the
    request returns the graph again, the "no sidecar" branch is never entered,
    and onnxruntime is handed a graph whose weights are missing.
    """

    def test_the_suffix_goes_on_the_path_not_the_query(self):
        from phoonnx.model_manager import _sidecar_url
        self.assertEqual(_sidecar_url(f"{HUB}/model.onnx?download=true"),
                         f"{HUB}/model.onnx_data?download=true")
        self.assertEqual(_sidecar_url(f"{HUB}/model.onnx"),
                         f"{HUB}/model.onnx_data")

    def test_a_query_url_still_fetches_the_real_sidecar(self):
        self.hub.stage(f"{HUB}/model.onnx", b"graph")
        self.hub.stage(f"{HUB}/model.onnx_data", b"weights")
        info = self.make_info(model_url=f"{HUB}/model.onnx?download=true")
        path = info.download_model()
        sidecar = path.parent / "model.onnx_data"
        self.assertTrue(sidecar.is_file(),
                        "the sidecar must land beside the graph")
        self.assertEqual(sidecar.read_bytes(), b"weights",
                         "the sidecar must be the weights, not the graph again")


class TestSidecarErrors(HubTestCase):
    """The external-weights sidecar is the biggest file a voice pulls (2.45 GB
    for the omnivoice backbone). "Absent" must stay distinguishable from
    "failed", or a graph loads with its weights missing and synthesizes
    silence."""

    def setUp(self):
        super().setUp()
        self.hub.stage(f"{HUB}/model.onnx", b"onnxdata")
        self.info = self.make_info()

    def test_absent_sidecar_is_tolerated(self):
        # a single-file graph (piper/vits) simply has no <name>.onnx_data
        path = self.info.download_model()
        self.assertTrue(path.is_file())
        self.assertFalse((path.parent / "model.onnx_data").exists())

    def test_offline_with_nothing_cached_is_tolerated(self):
        self.hub.errors["model.onnx_data"] = LocalEntryNotFoundError("offline")
        self.assertTrue(self.info.download_model().is_file())

    def test_a_full_disk_is_not_mistaken_for_an_absent_sidecar(self):
        self.hub.errors["model.onnx_data"] = OSError(28, "No space left on device")
        with self.assertRaises(OSError):
            self.info.download_model()

    def test_a_permission_error_is_not_mistaken_for_an_absent_sidecar(self):
        self.hub.errors["model.onnx_data"] = PermissionError(13, "Permission denied")
        with self.assertRaises(PermissionError):
            self.info.download_model()

    def test_a_server_error_is_not_mistaken_for_an_absent_sidecar(self):
        self.hub.errors["model.onnx_data"] = _hub_500()
        with self.assertRaises(HfHubHTTPError):
            self.info.download_model()

    def test_a_404_on_a_non_hub_sidecar_is_still_tolerated(self):
        url = "https://mirror.example.org/v/model.onnx"
        with self.stream({"model.onnx_data": _mock_response(404),
                          "model.onnx": _mock_response(200, content=b"data")}):
            path = self.make_info(model_url=url).download_model()
        self.assertTrue(path.is_file())
        self.assertFalse((path.parent / "model.onnx_data").exists())

    def test_a_500_on_a_non_hub_sidecar_is_not_tolerated(self):
        url = "https://mirror.example.org/v/model.onnx"
        with self.stream({"model.onnx_data": _mock_response(500),
                          "model.onnx": _mock_response(200, content=b"data")}):
            with self.assertRaises(requests.exceptions.HTTPError):
                self.make_info(model_url=url).download_model()


class TestExternalDataGraphLoads(HubTestCase):
    """An external-data graph must load through onnxruntime, end to end.

    The graph names its weights by a relative path, and onnxruntime refuses to
    follow that reference outside the directory the graph itself resolves to.
    Every other test here stops at "the bytes arrived", which is why a layout
    onnxruntime rejects shipped: a real graph beside a linked sidecar reads as
    complete on disk and fails at load, with

        FAIL : External data path validation failed for initializer: W.
        Error: External data path escapes model directory.

    and every voice built that way answered with HTTP 500. This test loads it.
    """

    def _external_data_model(self):
        """Build a tiny graph whose one initializer lives in a sidecar."""
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper, TensorProto

        build = Path(tempfile.mkdtemp())
        self.addCleanup(
            lambda: __import__("shutil").rmtree(build, ignore_errors=True))
        weight = numpy_helper.from_array(
            np.ones((1024, 16), dtype=np.float32), name="W")
        graph = helper.make_graph(
            [helper.make_node("Add", ["X", "W"], ["Y"])], "g",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1024, 16])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1024, 16])],
            [weight])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.save(model, str(build / "model.onnx"), save_as_external_data=True,
                  location="model.onnx_data", all_tensors_to_one_file=True,
                  size_threshold=0)
        return ((build / "model.onnx").read_bytes(),
                (build / "model.onnx_data").read_bytes())

    def _stage_external_data_voice(self):
        graph, weights = self._external_data_model()
        self.hub.stage(f"{HUB}/model.onnx", graph)
        self.hub.stage(f"{HUB}/model.onnx_data", weights)
        return self.make_info()

    def test_the_graph_the_engine_is_handed_loads(self):
        info = self._stage_external_data_voice()
        session = onnxruntime.InferenceSession(
            str(info.download_model()), providers=["CPUExecutionProvider"])
        self.assertEqual([i.name for i in session.get_inputs()], ["X"])

    def test_a_copy_outside_the_cache_would_not_load(self):
        """The property under test, stated as its own failure.

        Copying the graph anywhere else — which is what phoonnx used to do —
        leaves the sidecar behind in the cache, and onnxruntime rejects it.

        The rejection is onnxruntime's, and older builds load the escaping
        path without complaint, so this control is skipped there rather than
        failing: it would otherwise report a missing guard in a dependency as
        a fault in phoonnx.
        """
        import shutil
        info = self._stage_external_data_voice()
        cached = info.download_model()
        elsewhere = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(elsewhere, ignore_errors=True))
        shutil.copy2(cached, elsewhere / "model.onnx")
        os.symlink(cached.parent / "model.onnx_data",
                   elsewhere / "model.onnx_data")

        try:
            onnxruntime.InferenceSession(
                str(elsewhere / "model.onnx"),
                providers=["CPUExecutionProvider"])
        except Exception as exc:
            self.assertIn("escapes model directory", str(exc))
        else:
            self.skipTest(
                f"onnxruntime {onnxruntime.__version__} does not enforce "
                "external-data path containment; the control cannot run")

    def test_a_self_hosted_external_data_graph_loads_too(self):
        graph, weights = self._external_data_model()
        url = "https://mirror.example.org/v/model.onnx"
        with self.stream({"model.onnx_data": _mock_response(200, content=weights),
                          "model.onnx": _mock_response(200, content=graph)}):
            path = self.make_info(model_url=url).download_model()
        session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"])
        self.assertEqual([i.name for i in session.get_inputs()], ["X"])

    def test_load_opens_a_real_session_on_an_external_data_graph(self):
        """``load()`` itself opens the session for a voice with no published
        config.json, so this exercises the whole path with real onnxruntime."""
        info = self._stage_external_data_voice()
        info.engine = Engine.PHOONNX
        with patch("phoonnx.model_manager.TTSVoice") as voice_cls:
            info.load()
        session = voice_cls.call_args.kwargs["session"]
        self.assertEqual([i.name for i in session.get_inputs()], ["X"])


class TestLoad(HubTestCase):
    def test_load_hands_every_artifact_path_to_ttsvoice(self):
        self.hub.stage(f"{HUB}/model.onnx", b"onnxdata")
        self.hub.stage(f"{HUB}/config.json", {"sample_rate": 22050})
        self.hub.stage(f"{HUB}/tokens.txt", "a\nb\n")
        info = self.make_info(config_url=f"{HUB}/config.json",
                              tokens_url=f"{HUB}/tokens.txt")
        with patch("phoonnx.model_manager.TTSVoice") as voice_cls, \
                patch("phoonnx.model_manager.get_phonemizer"):
            info.load()
        kwargs = voice_cls.load.call_args.kwargs
        self.assertEqual(Path(kwargs["model_path"]).read_bytes(), b"onnxdata")
        self.assertEqual(json.loads(Path(kwargs["config_path"]).read_text()),
                         {"sample_rate": 22050})
        self.assertIsNone(kwargs["vocab_path"], "this voice has no vocab.json")


class TestEngineParams(HubTestCase):
    def test_empty_for_single_stage_engine(self):
        self.assertEqual(self.make_info().engine_params(), {})

    def test_vocoder_style_and_speaker_encoder_resolve(self):
        for name in ("vocoder.onnx", "style.bin", "speaker_encoder.onnx"):
            self.hub.stage(f"{HUB}/{name}", b"payload")
        self.hub.stage(f"{HUB}/vocoder.json", {"hop": 256})
        info = self.make_info(vocoder_url=f"{HUB}/vocoder.onnx",
                              vocoder_config_url=f"{HUB}/vocoder.json",
                              vocoder_type="hifigan",
                              style_url=f"{HUB}/style.bin",
                              speaker_encoder_url=f"{HUB}/speaker_encoder.onnx")
        params = info.engine_params()
        self.assertEqual(params["vocoder_type"], "hifigan")
        self.assertEqual(params["vocoder_config"], {"hop": 256})
        for key in ("vocoder_path", "style_path", "speaker_encoder_path"):
            self.assertTrue(str(params[key]).startswith(str(self.hub_root)))

    def test_a_parametric_vocoder_needs_no_model_file(self):
        self.hub.stage(f"{HUB}/vocoder.json", {"n_fft": 1024})
        info = self.make_info(vocoder_type="griffin_lim",
                              vocoder_config_url=f"{HUB}/vocoder.json")
        params = info.engine_params()
        self.assertEqual(params["vocoder_type"], "griffin_lim")
        self.assertEqual(params["vocoder_config"], {"n_fft": 1024})
        self.assertNotIn("vocoder_path", params)


class TestDownloadAll(HubTestCase):
    def test_every_artifact_a_voice_needs_is_fetched(self):
        self.hub.stage(f"{HUB}/model.onnx", b"onnxdata")
        self.hub.stage(f"{HUB}/config.json", {"sample_rate": 22050})
        self.hub.stage(f"{HUB}/vocab.json", {"a": 1})
        self.hub.stage(f"{HUB}/tokenizer_config.json", {"unk_token": "<unk>"})
        info = self.make_info(config_url=f"{HUB}/config.json",
                              vocab_url=f"{HUB}/vocab.json",
                              tokenizer_config_url=f"{HUB}/tokenizer_config.json")
        model_path = info.download_all()
        self.assertEqual(model_path.read_bytes(), b"onnxdata")
        pulled = {name for _, name, _ in self.hub.downloads}
        self.assertLessEqual({"model.onnx", "config.json", "vocab.json",
                              "tokenizer_config.json"}, pulled)


class TestTTSModelInfoStringEnumCoercion(HubTestCase):
    def test_string_engine_alphabet_phoneme_type_coerced(self):
        info = self.make_info(engine="piper", alphabet="ipa", phoneme_type="espeak")
        self.assertEqual(info.engine, Engine.PIPER)
        self.assertEqual(info.alphabet, Alphabet.IPA)
        self.assertEqual(info.phoneme_type, PhonemeType.ESPEAK)

    def test_invalid_enum_value_raises(self):
        with self.assertRaises(ValueError):
            self.make_info(engine="not-a-real-engine")

    def test_display_name_placeholder_resolution(self):
        info = self.make_info(engine="piper", phoneme_type="espeak", display_name="{engine}/{phoneme_type}")
        self.assertEqual(info.display_name, "piper/espeak")


class ManagerTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.cache_path = os.path.join(self._tmpdir.name, "cache.json")
        expand_patcher = patch("phoonnx.model_manager.os.path.expanduser", return_value=self._tmpdir.name)
        self.addCleanup(expand_patcher.stop)
        expand_patcher.start()


class TestManagerLoadMalformed(ManagerTestCase):
    def test_load_skips_malformed_entries(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        manager.cache["good/voice"] = {
            "voice_id": "good/voice",
            "lang": "en-US",
            "model_url": "https://example.com/model.onnx",
        }
        manager.cache["missing_required_field"] = {"voice_id": "bad/voice"}
        manager.cache["bad_engine_value"] = {
            "voice_id": "bad2/voice",
            "lang": "en-US",
            "model_url": "https://example.com/model.onnx",
            "engine": "not-a-real-engine",
        }
        manager.cache.store()
        manager.load()
        self.assertIn("good/voice", manager.voices)
        self.assertNotIn("missing_required_field", manager.voices)
        self.assertNotIn("bad_engine_value", manager.voices)

    def test_merge_default_voices_skips_bad_entries_and_stores(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        with patch.object(manager.cache, "update") as mock_update:
            manager.merge_default_voices(store=True)
        self.assertTrue(mock_update.called)


class TestManagerRegistry(ManagerTestCase):
    def test_all_voices_and_supported_langs(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        v1 = TTSModelInfo(voice_id="a/1", lang="en-US", model_url="https://example.com/a.onnx")
        v2 = TTSModelInfo(voice_id="a/2", lang="pt-PT", model_url="https://example.com/b.onnx")
        manager.voices = {"a/1": v1, "a/2": v2}
        self.assertEqual(sorted(manager.all_voices, key=lambda v: v.voice_id), [v1, v2])
        self.assertEqual(manager.supported_langs, ["en-US", "pt-PT"])

    def test_clear_resets_voices_and_cache(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        manager.voices = {"x": MagicMock()}
        manager.cache["x"] = {"voice_id": "x"}
        manager.clear()
        self.assertEqual(manager.voices, {})
        self.assertEqual(len(manager.cache), 0)

    def test_add_voice_persists_to_cache(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        voice = TTSModelInfo(voice_id="a/1", lang="en-US", model_url="https://example.com/a.onnx")
        manager.add_voice(voice)
        self.assertIn("a/1", manager.voices)
        self.assertEqual(manager.cache["a/1"]["voice_id"], "a/1")

    def test_save_persists_all_voices(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        voice = TTSModelInfo(voice_id="a/1", lang="en-US", model_url="https://example.com/a.onnx")
        manager.voices = {"a/1": voice}
        manager.save()
        manager.cache.reload()
        self.assertIn("a/1", manager.cache)
        self.assertEqual(manager.cache["a/1"]["model_url"], "https://example.com/a.onnx")

    def test_get_lang_voices_filters_by_match(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        v_en = TTSModelInfo(voice_id="a/1", lang="en-US", model_url="https://example.com/a.onnx")
        v_pt = TTSModelInfo(voice_id="a/2", lang="pt-PT", model_url="https://example.com/b.onnx")
        manager.voices = {"a/1": v_en, "a/2": v_pt}
        result = manager.get_lang_voices("en-US")
        self.assertEqual(result, [v_en])

    def test_get_lang_voices_unknown_lang_returns_empty(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        v_en = TTSModelInfo(voice_id="a/1", lang="en-US", model_url="https://example.com/a.onnx")
        manager.voices = {"a/1": v_en}
        result = manager.get_lang_voices("zz-ZZ")
        self.assertEqual(result, [])


class TestGetAvailableVoiceIdsBySource(ManagerTestCase):
    def test_groups_by_source(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        result = manager.get_available_voice_ids_by_source()
        self.assertIn("piper", result)
        self.assertIsInstance(result["piper"], list)
        self.assertEqual(result["piper"], sorted(result["piper"]))


class TestDownloadVoiceById(ManagerTestCase):
    def test_voice_found_in_registry(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        voice = MagicMock()
        manager.voices = {"registry/voice": voice}
        result = manager.download_voice_by_id("registry/voice")
        self.assertTrue(result)
        voice.download_all.assert_called_once()

    def test_voice_found_only_in_on_disk_index(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        manager.voices = {}
        with patch.object(TTSModelInfo, "download_all") as mock_download:
            result = manager.download_voice_by_id("piper/ar_JO-kareem-low")
        self.assertTrue(result)
        mock_download.assert_called_once()

    def test_voice_not_found_anywhere_returns_false(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        manager.voices = {}
        result = manager.download_voice_by_id("totally/unknown-voice-id-xyz")
        self.assertFalse(result)


class TestFullSerializationRoundTrip(ManagerTestCase):
    """Adversarial: every optional TTSModelInfo field must survive save/reload
    and add_voice/reload, including enums, nested dicts, and the aux-graph
    URLs used by StyleTTS2/Kokoro, YourTTS, F5-TTS and Chatterbox."""

    def _fully_populated_info(self):
        return TTSModelInfo(
            voice_id="full/voice",
            lang="en-US",
            model_url="https://example.com/model.onnx",
            config_url="https://example.com/config.json",
            vocab_url="https://example.com/vocab.json",
            tokenizer_config_url="https://example.com/tokenizer_config.json",
            tokens_url="https://example.com/tokens.txt",
            phoneme_map_url="https://example.com/phoneme_map.json",
            phoneme_type="espeak",
            phonemizer_model="modern",
            alphabet="ipa",
            engine="piper",
            vocab_override={"a": 0, "b": 1},
            vocoder_url="https://example.com/vocoder.onnx",
            vocoder_config_url="https://example.com/vocoder.json",
            vocoder_type="hifigan",
            style_url="https://example.com/style.bin",
            speaker_encoder_url="https://example.com/enc.onnx",
            speaker_encoder_type="dvector",
            aux_model_urls={"preprocess_path": "https://example.com/pre.onnx",
                             "decode_path": "https://example.com/dec.onnx"},
            speech_encoder_url="https://example.com/speech_encoder.onnx",
            embed_tokens_url="https://example.com/embed_tokens.onnx",
            conditional_decoder_url="https://example.com/conditional_decoder.onnx",
            lang_tokens={"en-US": "en", "en-GB": "eg"},
            display_name="full voice",
        )

    def _assert_full_equality(self, original, reloaded):
        for f in original.__dataclass_fields__:
            self.assertEqual(getattr(original, f), getattr(reloaded, f), msg=f"field {f} mismatch")

    def test_save_round_trip_preserves_every_field(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        voice = self._fully_populated_info()
        manager.voices = {voice.voice_id: voice}
        manager.save()

        reloaded_manager = TTSModelManager(cache_path=self.cache_path)
        reloaded_manager.load()
        reloaded = reloaded_manager.voices["full/voice"]
        self._assert_full_equality(voice, reloaded)

    def test_add_voice_round_trip_preserves_every_field(self):
        manager = TTSModelManager(cache_path=self.cache_path)
        voice = self._fully_populated_info()
        manager.add_voice(voice)
        manager.cache.store()

        reloaded_manager = TTSModelManager(cache_path=self.cache_path)
        reloaded_manager.load()
        reloaded = reloaded_manager.voices["full/voice"]
        self._assert_full_equality(voice, reloaded)

    def test_bundled_index_style_dict_still_deserializes(self):
        # Matches the plain-dict-with-None-values format used by voice_index/*.json
        bundled_dict = {
            "voice_id": "chatterbox/base/en",
            "model_url": "https://example.com/language_model.onnx",
            "speech_encoder_url": "https://example.com/speech_encoder.onnx",
            "embed_tokens_url": "https://example.com/embed_tokens.onnx",
            "conditional_decoder_url": "https://example.com/conditional_decoder.onnx",
            "tokenizer_config_url": "https://example.com/tokenizer.json",
            "config_url": None,
            "vocab_url": None,
            "tokens_url": None,
            "phoneme_map_url": None,
            "phoneme_type": "unicode",
            "alphabet": "unicode",
            "engine": "chatterbox",
            "lang": "en-US",
        }
        info = TTSModelInfo(**bundled_dict)
        self.assertEqual(info.speech_encoder_url, bundled_dict["speech_encoder_url"])
        self.assertEqual(info.embed_tokens_url, bundled_dict["embed_tokens_url"])
        self.assertEqual(info.conditional_decoder_url, bundled_dict["conditional_decoder_url"])
        self.assertEqual(info.engine, Engine.CHATTERBOX)
        self.assertEqual(info.phoneme_type, PhonemeType.UNICODE)

    def test_to_dict_enum_fields_are_plain_strings(self):
        voice = self._fully_populated_info()
        d = voice.to_dict()
        self.assertEqual(d["engine"], "piper")
        self.assertEqual(d["alphabet"], "ipa")
        self.assertEqual(d["phoneme_type"], "espeak")
        # must be JSON serializable without a custom encoder
        json.dumps(d)


class TestVoiceIndexSources(unittest.TestCase):
    def test_every_bundled_index_is_listed(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        manager = TTSModelManager(cache_path=os.path.join(tmp.name, "cache.json"))
        base_path = TTSModelManager.voice_index_path()
        on_disk = {p.stem.lower() for p in base_path.glob("*.json")}
        listed = set(manager.get_available_voice_ids_by_source().keys())
        self.assertEqual(on_disk, listed)

    def test_index_files_cover_the_whole_directory(self):
        base_path = TTSModelManager.voice_index_path()
        self.assertEqual(set(TTSModelManager.voice_index_files()),
                         set(base_path.glob("*.json")))


if __name__ == "__main__":
    unittest.main()


class TestCatalogDoesNotTouchDisk(unittest.TestCase):
    """A voice's directory should exist because something was written into it,
    not because its catalog entry was constructed — building the catalog used to
    create one directory per voice (thousands) on every start."""

    def test_building_the_catalog_creates_no_directories(self):
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home}):
                mm = TTSModelManager()
                mm.merge_default_voices()
                self.assertTrue(mm.voices, "expected a populated catalog")
                created = [p for p in Path(home).rglob("*") if p.is_dir()]
                self.assertEqual(created, [], f"catalog created {len(created)} dirs")


# ---------------------------------------------------------------------------
# The shared HuggingFace cache
# ---------------------------------------------------------------------------

HUB = "https://huggingface.co/an-org/a-repo/resolve/main"


def _hub_500():
    """The hub client wraps a server error, keeping the response on it."""
    return HfHubHTTPError("500 Server Error", response=MagicMock(status_code=500))


class TestHfUrlParsing(unittest.TestCase):
    def test_non_hub_urls_are_declined(self):
        for url in ["https://example.com/model.onnx",
                    "https://huggingface.co/an-org/a-repo",
                    "https://not-huggingface.co/a/b/resolve/main/f.onnx"]:
            self.assertIsNone(_hf_fetch(url), url)

    def test_query_strings_are_ignored(self):
        with patch("phoonnx.model_manager.hf_hub_download",
                   return_value="/cache/f.onnx") as dl:
            _hf_fetch(f"{HUB}/model.onnx?download=true")
        dl.assert_called_once()
        self.assertEqual(dl.call_args.kwargs["filename"], "model.onnx")

    def test_nested_paths_survive(self):
        with patch("phoonnx.model_manager.hf_hub_download",
                   return_value="/cache/f.onnx") as dl:
            _hf_fetch(f"{HUB}/onnx/sub/model.onnx")
        self.assertEqual(dl.call_args.kwargs["filename"], "onnx/sub/model.onnx")


class TestHubImportedEagerly(unittest.TestCase):
    def test_the_hub_client_is_imported_at_module_scope(self):
        """Importing it inside the download call meant the first voice download
        decided where the hub cache and token live. Anything that had redirected
        ``expanduser`` by then — a test, a sandbox — got baked in for the life of
        the process."""
        import phoonnx.model_manager as mm
        self.assertTrue(callable(mm.hf_hub_download))
        from huggingface_hub import constants
        self.assertTrue(Path(constants.HF_TOKEN_PATH).parent.name,
                        "the token path must be resolved, not empty")


class TestUnwritableCachePath(unittest.TestCase):
    def test_unwritable_cache_dir_raises_clear_error(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(lambda: (os.chmod(tmpdir, 0o700), __import__("shutil").rmtree(tmpdir)))
        os.chmod(tmpdir, 0o500)
        bad_path = os.path.join(tmpdir, "sub", "cache.json")
        with self.assertRaises(PermissionError):
            TTSModelManager(cache_path=bad_path)


class TestDiskSize(unittest.TestCase):
    """Measuring a voice, against a real hub cache layout on disk.

    The byte budget is only as good as this measurement: a ``disk_size`` that
    silently returns 0 does not fail loudly, it disables the budget and lets
    memory run unbounded. So these tests build the cache the hub actually
    writes — blobs, refs, and a snapshot of symlinks into the blobs — and read
    it back through the real ``huggingface_hub`` resolver rather than a stub.
    """

    REPO = "an-org/a-repo"
    URL = "https://huggingface.co/an-org/a-repo/resolve/main/"

    def setUp(self):
        from huggingface_hub import constants
        self._cache = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self._cache.cleanup)
        self.cache = Path(self._cache.name)
        patcher = patch.object(constants, "HF_HUB_CACHE", str(self.cache))
        self.addCleanup(patcher.stop)
        patcher.start()
        # a direct (non-hub) download must not touch the developer's cache
        dpatch = patch("phoonnx.model_manager.HF_HUB_CACHE", str(self.cache))
        self.addCleanup(dpatch.stop)
        dpatch.start()

        self.root = self.cache / "models--an-org--a-repo"
        (self.root / "blobs").mkdir(parents=True)
        (self.root / "refs").mkdir()
        (self.root / "refs" / "main").write_text("cafebabe")
        self.snapshot = self.root / "snapshots" / "cafebabe"
        self.snapshot.mkdir(parents=True)

    def add(self, name, size):
        """Write ``name`` into the cache exactly as the hub would: a blob,
        with the snapshot entry a symlink pointing at it."""
        blob = self.root / "blobs" / f"{name}-sha"
        blob.write_bytes(b"x" * size)
        link = self.snapshot / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob)
        return blob

    def test_a_cached_file_resolves_to_the_blob_it_links_to(self):
        from phoonnx.model_manager import _cached_path
        blob = self.add("model.onnx", 1000)
        path = _cached_path(f"{self.URL}model.onnx")
        self.assertIsNotNone(path)
        self.assertTrue(Path(path).is_symlink())
        self.assertEqual(Path(path).resolve(), blob.resolve())
        self.assertEqual(Path(path).stat().st_size, 1000,
                         "stat must follow the link to the blob, not measure "
                         "the link itself")

    def test_the_weights_sidecar_is_counted_with_its_graph(self):
        # An omnivoice graph is a few MB and its .onnx_data is gigabytes;
        # missing the sidecar is the difference between a budget and a lie.
        from phoonnx.model_manager import _file_bytes
        self.add("model.onnx", 1000)
        self.add("model.onnx_data", 5000)
        self.assertEqual(_file_bytes(f"{self.URL}model.onnx"), 6000)

    def test_a_graph_without_a_sidecar_is_just_the_graph(self):
        from phoonnx.model_manager import _file_bytes
        self.add("vocoder.onnx", 2000)
        self.assertEqual(_file_bytes(f"{self.URL}vocoder.onnx"), 2000)

    def test_the_download_query_form_measures_the_same_file(self):
        from phoonnx.model_manager import _file_bytes
        self.add("model.onnx", 1000)
        self.add("model.onnx_data", 5000)
        self.assertEqual(_file_bytes(f"{self.URL}model.onnx?download=true"),
                         6000)

    def test_a_file_that_was_never_downloaded_measures_zero(self):
        from phoonnx.model_manager import _cached_path, _file_bytes
        self.assertIsNone(_cached_path(f"{self.URL}absent.onnx"))
        self.assertEqual(_file_bytes(f"{self.URL}absent.onnx"), 0)
        self.assertEqual(_file_bytes(""), 0)

    def test_never_goes_to_the_network(self):
        # A size question that can start a download, or block on a hub that is
        # slow to answer, is a size question that can hang synthesis.
        from phoonnx.model_manager import _file_bytes
        with patch("requests.get", side_effect=AssertionError("network!")), \
                patch("requests.head", side_effect=AssertionError("network!")):
            self.assertEqual(_file_bytes(f"{self.URL}absent.onnx"), 0)

    def test_every_graph_a_voice_loads_is_summed(self):
        self.add("model.onnx", 1000)
        self.add("model.onnx_data", 5000)
        self.add("vocoder.onnx", 2000)
        self.add("speaker.onnx", 400)
        self.add("aux.onnx", 100)
        info = TTSModelInfo(voice_id="v", lang="en-US",
                            model_url=f"{self.URL}model.onnx",
                            vocoder_url=f"{self.URL}vocoder.onnx",
                            speaker_encoder_url=f"{self.URL}speaker.onnx",
                            aux_model_urls={"extra": f"{self.URL}aux.onnx"})
        self.assertEqual(info.disk_size(), 8500)

    def test_one_file_named_by_two_fields_is_counted_once(self):
        self.add("model.onnx", 1000)
        self.add("model.onnx_data", 5000)
        info = TTSModelInfo(voice_id="v", lang="en-US",
                            model_url=f"{self.URL}model.onnx",
                            vocoder_url=f"{self.URL}model.onnx")
        self.assertEqual(info.disk_size(), 6000)

    def test_a_voice_that_was_never_fetched_measures_zero(self):
        info = TTSModelInfo(voice_id="v", lang="en-US",
                            model_url=f"{self.URL}cold.onnx")
        self.assertEqual(info.disk_size(), 0)

    def test_a_voice_hosted_outside_the_hub_is_measured_too(self):
        from phoonnx.model_manager import _file_bytes
        url = "https://models.example/voice/model.onnx"
        dest = _direct_dir(url)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "model.onnx").write_bytes(b"x" * 777)
        self.assertEqual(_file_bytes(url), 777)
        self.assertEqual(
            TTSModelInfo(voice_id="v", lang="en-US",
                         model_url=url).disk_size(), 777)

    def test_an_empty_off_hub_file_is_not_reported_as_downloaded(self):
        # ``_is_cached`` treats a zero-byte file as absent: an interrupted
        # download leaves one behind, and calling it cached would both skip the
        # refetch and measure a multi-gigabyte voice as free.
        from phoonnx.model_manager import _cached_path
        url = "https://models.example/voice/model.onnx"
        dest = _direct_dir(url)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "model.onnx").write_bytes(b"")
        self.assertIsNone(_cached_path(url))
