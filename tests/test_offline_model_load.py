"""Offline-loading behaviour of the ONNX model downloader.

Loading a voice whose graph is already cached probes for an optional
external-data sidecar. That probe must tolerate a network failure (offline / DNS
/ timeout) exactly like an HTTP 404 — the sidecar is simply absent — so a
fully-cached voice loads without the probe crashing, while a genuine sidecar is
still downloaded when the network is reachable.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from phoonnx.model_manager import TTSModelInfo, _direct_dir


class _FakeResponse:
    """Minimal stand-in for a streamed ``requests`` response usable as a
    context manager."""

    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self._content = content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def iter_content(self, chunk_size=8192):
        yield self._content


class TestOfflineModelLoad(unittest.TestCase):
    """This voice is self-hosted, so it takes the direct-download path. The hub
    client is never involved, and the sidecar probe behaves exactly as it does
    for a hub voice."""

    URL = "https://example.test/v/model.onnx"

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        # keep every download inside the test's own cache root
        patcher = patch("phoonnx.model_manager.HF_HUB_CACHE", self._tmpdir.name)
        self.addCleanup(patcher.stop)
        patcher.start()
        self.info = TTSModelInfo(voice_id="test/offline", lang="en",
                                 model_url=self.URL)
        self.tmp = _direct_dir(self.URL)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.model_path = self.tmp / "model.onnx"

    def test_sidecar_connection_error_does_not_propagate(self):
        """An offline sidecar probe must not crash a voice that has its graph."""
        self.model_path.write_bytes(b"onnx-graph")

        with patch("phoonnx.model_manager.requests.get",
                   side_effect=requests.exceptions.ConnectionError("offline")):
            # must not raise
            result = self.info.download_model()

        self.assertEqual(result, self.model_path)
        self.assertTrue(self.model_path.is_file())

    def test_missing_sidecar_404_is_tolerated(self):
        """A single-file voice whose sidecar 404s loads without error."""
        self.model_path.write_bytes(b"onnx-graph")

        with patch("phoonnx.model_manager.requests.get",
                   return_value=_FakeResponse(status_code=404)):
            self.info.download_model()

        self.assertFalse((self.tmp / "model.onnx_data").is_file())

    def test_genuine_sidecar_is_downloaded(self):
        """A voice that really needs external data downloads it when reachable."""
        self.model_path.write_bytes(b"onnx-graph")
        data_path = self.tmp / "model.onnx_data"

        def fake_get(url, *a, **kw):
            if url.endswith("_data"):
                return _FakeResponse(status_code=200, content=b"weights")
            return _FakeResponse(status_code=200, content=b"onnx-graph")

        with patch("phoonnx.model_manager.requests.get", side_effect=fake_get):
            self.info.download_model()

        self.assertTrue(data_path.is_file())
        self.assertEqual(data_path.read_bytes(), b"weights")


if __name__ == "__main__":
    unittest.main()
