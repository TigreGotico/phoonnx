"""Two downloads of the same artifact do not share one temporary file.

A fixed ``<target>.part`` name means concurrent fetches of the same URL write
into the same file. Their bytes interleave, and the one that fails first
deletes the file the other is still writing, so a request that should have
succeeded either fails or publishes a corrupt artifact.
"""
import os
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


class TestPublishedArtifact(unittest.TestCase):
    """Properties of the file that ends up at the destination."""

    def _fetch(self, dest):
        class _Whole:
            headers = {"Content-Length": str(len(BODY))}

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size=CHUNK):
                yield BODY

        with patch.object(model_manager.requests, "get",
                          side_effect=lambda *a, **kw: _Whole()):
            model_manager._direct_stream("https://example.invalid/model.onnx", dest)

    def test_the_model_is_readable_by_more_than_its_downloader(self):
        """mkstemp creates 0600 and os.replace carries the mode onto the model.

        A shared cache is served by an account that is not always the one that
        filled it.
        """
        dest = Path(mkdtemp()) / "model.onnx"
        self._fetch(dest)

        umask = os.umask(0)
        os.umask(umask)
        self.assertEqual(0o666 & ~umask, dest.stat().st_mode & 0o777)

    def test_a_download_leaves_another_downloads_scratch_file_alone(self):
        """Nothing may sweep sibling scratch files on entry.

        A sweep cannot distinguish an abandoned file from one another thread is
        still writing, so it would restore the interleaving this module exists
        to prevent.
        """
        dest = Path(mkdtemp()) / "model.onnx"
        dest.parent.mkdir(parents=True, exist_ok=True)
        in_flight = dest.parent / (dest.name + ".aaaa.part")
        in_flight.write_bytes(b"another download is writing here")

        self._fetch(dest)

        self.assertTrue(in_flight.exists())
        self.assertEqual(b"another download is writing here", in_flight.read_bytes())


if __name__ == "__main__":
    unittest.main()
