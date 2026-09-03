"""Loading the reference clip a cloning request points at.

The URL comes from the caller, so the fetch is what needs guarding: it must
refuse addresses inside this network (on every redirect hop, not just the
first), refuse a body too large to be a few seconds of speech, give up on a
sender that trickles bytes to hold a worker open, and close every response it
opened along the way.
"""
import unittest
from unittest.mock import MagicMock, patch


class TestReferenceFromAUrl(unittest.TestCase):
    """A remote caller cannot put a file on the server's disk."""

    def test_a_url_is_downloaded_and_read(self):
        import numpy as np
        from phoonnx.reference_audio import load_reference_audio

        wav = self._wav_bytes()
        with patch("requests.get") as get:
            get.return_value.__enter__.return_value = self._response(wav)
            audio, sr = load_reference_audio("https://example.org/ref.wav")
        self.assertEqual(sr, 16000)
        self.assertGreater(len(audio), 0)
        self.assertEqual(np.asarray(audio).ndim, 1)

    def test_an_oversized_download_is_refused(self):
        from phoonnx.reference_audio import load_reference_audio
        with patch("requests.get") as get:
            get.return_value.__enter__.return_value = self._response(
                b"x" * (33 * 1024 * 1024))
            with self.assertRaises(ValueError) as caught:
                load_reference_audio("https://example.org/huge.wav")
        self.assertIn("larger than", str(caught.exception))

    def test_a_plain_path_is_still_read_directly(self):
        import tempfile, os
        from phoonnx.reference_audio import load_reference_audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(self._wav_bytes())
            path = f.name
        self.addCleanup(lambda: os.unlink(path))
        with patch("requests.get") as get:
            audio, sr = load_reference_audio(path)
        get.assert_not_called()
        self.assertEqual(sr, 16000)

    @staticmethod
    def _wav_bytes():
        import io, wave, struct
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(struct.pack("<1600h", *([1000] * 1600)))
        return buf.getvalue()

    @staticmethod
    def _response(payload: bytes):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.iter_content.return_value = [payload]
        # Not a redirect: the fetcher follows those itself so it can check
        # each hop, and a MagicMock is truthy.
        r.is_redirect = False
        r.is_permanent_redirect = False
        return r

    def setUp(self):
        # The host check does real DNS; the addresses it allows are covered
        # by TestTheFetchRefusesTheInsideOfTheNetwork.
        patcher = patch("phoonnx.reference_audio.check_reference_url")
        patcher.start()
        self.addCleanup(patcher.stop)


class TestTheFetchRefusesTheInsideOfTheNetwork(unittest.TestCase):
    """The URL comes from the caller, and the server sits on a private network.

    Without a check, a public TTS endpoint will fetch from its own loopback,
    its container neighbours, or a cloud metadata service on the caller's
    behalf and report back whether it worked.
    """

    def test_loopback_is_refused(self):
        from phoonnx.reference_audio import check_reference_url
        with self.assertRaises(ValueError) as caught:
            check_reference_url("http://127.0.0.1:9666/ref.wav")
        self.assertIn("inside this network", str(caught.exception))

    def test_a_private_range_is_refused(self):
        from phoonnx.reference_audio import check_reference_url
        for host in ("10.0.0.5", "192.168.1.10", "172.16.0.3",
                     "169.254.169.254"):
            with self.subTest(host=host):
                with self.assertRaises(ValueError):
                    check_reference_url(f"http://{host}/ref.wav")

    def test_a_non_http_scheme_is_refused(self):
        from phoonnx.reference_audio import check_reference_url
        for url in ("file:///etc/passwd", "ftp://example.org/a.wav",
                    "gopher://example.org/a.wav"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    check_reference_url(url)

    def test_a_public_address_is_allowed(self):
        from phoonnx.reference_audio import check_reference_url
        with patch("phoonnx.reference_audio.socket.getaddrinfo",
                   return_value=[(2, 1, 6, "", ("93.184.216.34", 80))]):
            check_reference_url("https://example.org/ref.wav")

    def test_a_redirect_into_the_network_is_refused(self):
        """A public URL that redirects inward is the easy way around a check
        that only looks at the address the caller typed."""
        from phoonnx import reference_audio

        redirect = MagicMock()
        redirect.is_redirect = True
        redirect.is_permanent_redirect = False
        redirect.url = "https://example.org/ref.wav"
        redirect.headers = {"Location": "http://169.254.169.254/latest/meta-data"}

        def resolve(host, *a, **kw):
            # Only the outward-looking name resolves publicly; the redirect
            # target is a literal address and resolves to itself.
            if host == "example.org":
                return [(2, 1, 6, "", ("93.184.216.34", 80))]
            return [(2, 1, 6, "", (host, 80))]

        with patch("requests.get") as get:
            get.return_value.__enter__.return_value = redirect
            with patch("phoonnx.reference_audio.socket.getaddrinfo", side_effect=resolve):
                with self.assertRaises(ValueError) as caught:
                    reference_audio.fetch_reference_audio("https://example.org/ref.wav")
        self.assertIn("inside this network", str(caught.exception))


class TestTheFetchGivesUp(unittest.TestCase):

    def test_a_trickle_cannot_hold_the_worker_forever(self):
        """requests' timeout is per read, so a sender dripping one byte at a
        time resets it endlessly and keeps a synthesis worker occupied."""
        import time
        from phoonnx import reference_audio

        def dribble(chunk_size=None):
            while True:
                time.sleep(0.05)
                yield b"x"

        response = MagicMock()
        response.raise_for_status.return_value = None
        response.is_redirect = False
        response.is_permanent_redirect = False
        response.iter_content.side_effect = dribble

        with patch("requests.get") as get, \
                patch("phoonnx.reference_audio.check_reference_url"):
            get.return_value.__enter__.return_value = response
            started = time.monotonic()
            with self.assertRaises(ValueError) as caught:
                reference_audio.fetch_reference_audio("https://example.org/slow.wav",
                                             deadline=0.5)
            elapsed = time.monotonic() - started
        self.assertIn("longer than", str(caught.exception))
        self.assertLess(elapsed, 5, "the deadline must actually end it")




class _FakeResponse:
    """Records whether close() was called, like requests.Response."""

    def __init__(self, url, redirect=False, permanent_redirect=False,
                 location=None, chunks=None):
        self.url = url
        self.is_redirect = redirect
        self.is_permanent_redirect = permanent_redirect
        self.headers = {"Location": location} if location else {}
        self._chunks = chunks or [b"abc"]
        self.closed = False

    def close(self):
        self.closed = True

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def iter_content(self, chunk_size=8192):
        for c in self._chunks:
            yield c


class TestRedirectedFetchClosesEveryResponse(unittest.TestCase):
    def test_final_response_is_closed(self):
        from phoonnx.reference_audio import fetch_reference_audio

        first = _FakeResponse("http://example.com/a", redirect=True,
                               location="http://example.com/b")
        second = _FakeResponse("http://example.com/b", chunks=[b"hello"])
        responses = [first, second]

        def fake_get(url, timeout=None, stream=None, allow_redirects=None):
            return responses.pop(0)

        with patch("requests.get", side_effect=fake_get), \
                patch("phoonnx.reference_audio.check_reference_url"):
            path = fetch_reference_audio("http://example.com/a")

        try:
            self.assertTrue(first.closed, "the intermediate hop must be closed")
            self.assertTrue(
                second.closed,
                "the final response (the one actually streamed to disk) "
                "must be closed too — it leaked its pooled connection")
        finally:
            import os
            os.unlink(path)

    def test_redirect_target_is_still_validated_per_hop(self):
        from phoonnx.reference_audio import fetch_reference_audio

        first = _FakeResponse("http://example.com/a", redirect=True,
                               location="http://example.com/b")
        second = _FakeResponse("http://example.com/b", chunks=[b"hello"])
        responses = [first, second]

        def fake_get(url, timeout=None, stream=None, allow_redirects=None):
            return responses.pop(0)

        with patch("requests.get", side_effect=fake_get), \
                patch("phoonnx.reference_audio.check_reference_url") as checker:
            path = fetch_reference_audio("http://example.com/a")
        import os
        os.unlink(path)

        checked = [c.args[0] for c in checker.call_args_list]
        self.assertIn("http://example.com/a", checked)
        self.assertIn("http://example.com/b", checked)


if __name__ == "__main__":
    unittest.main()
