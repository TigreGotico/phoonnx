"""Two downloads of the same artifact do not share one temporary file.

A fixed ``<target>.part`` name means concurrent fetches of the same URL write
into the same file. Their bytes interleave, and the one that fails first
deletes the file the other is still writing, so a request that should have
succeeded either fails or publishes a corrupt artifact.
"""
import threading
import unittest
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

from phoonnx import model_manager


BODY = b"x" * 4096
CHUNK = 256


class _Response:
    """A streamed response that parks mid-body until every caller arrives."""

    def __init__(self, barrier):
        self.headers = {"Content-Length": str(len(BODY))}
        self._barrier = barrier

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=CHUNK):
        halfway = len(BODY) // 2
        sent = 0
        while sent < len(BODY):
            if sent >= halfway and self._barrier is not None:
                self._barrier.wait(timeout=10)
                self._barrier = None
            yield BODY[sent:sent + CHUNK]
            sent += CHUNK


class TestConcurrentDownloadsOfOneArtifact(unittest.TestCase):

    def test_both_callers_get_the_whole_artifact(self):
        dest = Path(mkdtemp()) / "model.onnx"
        barrier = threading.Barrier(2)
        errors = []

        def fetch():
            try:
                model_manager._direct_stream("https://example.invalid/model.onnx", dest)
            except BaseException as e:
                errors.append(e)

        with patch.object(model_manager.requests, "get",
                          side_effect=lambda *a, **kw: _Response(barrier)):
            threads = [threading.Thread(target=fetch) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        self.assertEqual([], errors)
        self.assertEqual(BODY, dest.read_bytes())


if __name__ == "__main__":
    unittest.main()
