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


class TestStdlibWavDecoding(unittest.TestCase):
    """Without soundfile, a WAV is decoded by its header's sample width."""

    RATE = 16000

    def setUp(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        # the patch drops at stop every module first imported under it, and numpy loads once per process
        import numpy  # noqa: F401
        import phoonnx.reference_audio  # noqa: F401
        patcher = patch.dict("sys.modules", {"soundfile": None})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _sine(self, n):
        import numpy as np
        t = np.arange(n) / self.RATE
        return 0.5 * np.sin(2 * np.pi * 440 * t)

    def _write(self, name, width, n, channels=1):
        import os, wave
        import numpy as np
        x = self._sine(n)
        if width == 1:
            data = (np.round(x * 127) + 128).astype(np.uint8).tobytes()
        elif width == 2:
            data = np.round(x * 32767).astype("<i2").tobytes()
        elif width == 3:
            ints = np.round(x * 8388607).astype("<i4")
            data = b"".join(int(v).to_bytes(3, "little", signed=True) for v in ints)
        else:
            data = np.round(x * 2147483647).astype("<i4").tobytes()
        if channels == 2:
            step = width
            data = b"".join(data[i:i + step] * 2 for i in range(0, len(data), step))
        path = os.path.join(self.dir, name)
        with wave.open(path, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(width)
            w.setframerate(self.RATE)
            w.writeframes(data)
        return path

    def _check_sine(self, width, n):
        import numpy as np
        from phoonnx.reference_audio import load_reference_audio
        audio, sr = load_reference_audio(self._write(f"w{width}_{n}.wav", width, n))
        self.assertEqual(sr, self.RATE)
        self.assertEqual(len(audio), n)
        self.assertAlmostEqual(float(np.abs(audio).max()), 0.5, delta=0.01)
        np.testing.assert_allclose(audio, self._sine(n), atol=0.01)

    def test_8_bit_sine(self):
        self._check_sine(1, self.RATE)

    def test_16_bit_sine(self):
        self._check_sine(2, self.RATE)

    def test_24_bit_sine(self):
        self._check_sine(3, self.RATE)

    def test_32_bit_sine(self):
        self._check_sine(4, self.RATE)

    def test_odd_length_8_bit_file_loads(self):
        self._check_sine(1, 101)

    def test_odd_length_24_bit_file_loads(self):
        self._check_sine(3, 101)

    def test_8_bit_silence_is_zero(self):
        import numpy as np, os, wave
        from phoonnx.reference_audio import load_reference_audio
        path = os.path.join(self.dir, "silence.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(1)
            w.setframerate(self.RATE)
            w.writeframes(b"\x80" * 100)
        audio, _ = load_reference_audio(path)
        np.testing.assert_array_equal(audio, np.zeros(100, dtype=np.float32))

    def test_16_bit_result_is_the_raw_samples_over_32768(self):
        import numpy as np, wave
        from phoonnx.reference_audio import load_reference_audio
        path = self._write("p16.wav", 2, 1000)
        with wave.open(path, "rb") as w:
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        audio, _ = load_reference_audio(path)
        self.assertEqual(audio.dtype, np.float32)
        np.testing.assert_array_equal(audio, raw.astype(np.float32) / 32768.0)

    def test_stereo_is_averaged_to_mono(self):
        import numpy as np
        from phoonnx.reference_audio import load_reference_audio
        audio, _ = load_reference_audio(self._write("st.wav", 3, 500, channels=2))
        self.assertEqual(len(audio), 500)
        np.testing.assert_allclose(audio, self._sine(500), atol=0.01)

    def test_unsupported_width_names_the_file(self):
        import os
        from phoonnx.reference_audio import load_reference_audio
        path = self._write("wide.wav", 4, 100)
        with open(path, "r+b") as f:
            f.seek(34)
            f.write((48).to_bytes(2, "little"))
        with self.assertRaises(ValueError) as caught:
            load_reference_audio(path)
        self.assertIn(path, str(caught.exception))
        self.assertIn("48-bit", str(caught.exception))
        self.assertIn("32-bit PCM", str(caught.exception))

    def test_float_wav_names_the_file(self):
        from phoonnx.reference_audio import load_reference_audio
        path = self._write("float.wav", 4, 100)
        with open(path, "r+b") as f:
            f.seek(20)
            f.write((3).to_bytes(2, "little"))
        with self.assertRaises(ValueError) as caught:
            load_reference_audio(path)
        self.assertIn(path, str(caught.exception))

    def test_non_wav_file_names_the_file(self):
        import os
        from phoonnx.reference_audio import load_reference_audio
        path = os.path.join(self.dir, "fake.flac")
        with open(path, "wb") as f:
            f.write(b"fLaC" + b"\x00" * 64)
        with self.assertRaises(ValueError) as caught:
            load_reference_audio(path)
        self.assertIn(path, str(caught.exception))


    def test_an_empty_or_cut_file_names_the_file(self):
        import os
        from phoonnx.reference_audio import load_reference_audio
        for name, data in [("empty.wav", b""), ("riff.wav", b"RIFF"),
                           ("header.wav", b"RIFF\x24\x00\x00\x00WAVE"),
                           ("fmt.wav", b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00")]:
            path = os.path.join(self.dir, name)
            with open(path, "wb") as f:
                f.write(data)
            with self.subTest(name):
                with self.assertRaises(ValueError) as caught:
                    load_reference_audio(path)
                self.assertIn(path, str(caught.exception))

    def test_a_file_cut_inside_a_frame_keeps_its_whole_frames(self):
        import os
        import numpy as np
        from phoonnx.reference_audio import load_reference_audio
        path = self._write("cut.wav", 3, 100, channels=2)
        # the header still claims 100 frames of 6 bytes; the data ends 4 bytes into frame 50
        with open(path, "r+b") as f:
            f.truncate(os.path.getsize(path) - 50 * 6 - 2)
        audio, _ = load_reference_audio(path)
        np.testing.assert_allclose(audio, self._sine(49), atol=0.01)

    def test_an_extensible_pcm_header(self):
        import os, sys
        import numpy as np
        from phoonnx.reference_audio import load_reference_audio
        samples = np.round(self._sine(100) * 32767).astype("<i2")
        pcm_guid = bytes.fromhex("0100000000001000800000aa00389b71")
        fmt = ((0xFFFE).to_bytes(2, "little") + (1).to_bytes(2, "little")
               + self.RATE.to_bytes(4, "little") + (self.RATE * 2).to_bytes(4, "little")
               + (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
               + (22).to_bytes(2, "little") + (16).to_bytes(2, "little")
               + (4).to_bytes(4, "little") + pcm_guid)
        data = samples.tobytes()
        body = (b"WAVE" + b"fmt " + len(fmt).to_bytes(4, "little") + fmt
                + b"data" + len(data).to_bytes(4, "little") + data)
        path = os.path.join(self.dir, "extensible.wav")
        with open(path, "wb") as f:
            f.write(b"RIFF" + len(body).to_bytes(4, "little") + body)
        if sys.version_info >= (3, 12):
            audio, sr = load_reference_audio(path)
            self.assertEqual(sr, self.RATE)
            np.testing.assert_array_equal(audio, samples.astype(np.float32) / 32768.0)
        else:
            with self.assertRaises(ValueError) as caught:
                load_reference_audio(path)
            message = str(caught.exception)
            self.assertIn(path, message)
            self.assertIn("WAVE_FORMAT_EXTENSIBLE", message)
            self.assertIn("soundfile", message)
            self.assertNotIn("is not a PCM WAV file", message)


if __name__ == "__main__":
    unittest.main()
