"""Two independent defects in phoonnx/voice.py.

``_fetch_reference_audio`` follows redirects manually (rebinding ``r`` inside
a ``with requests.get(...) as r:`` block) so the response actually streamed
to disk is never closed by the context manager, leaking a pooled connection
on every redirected fetch.

``synthesize_wav`` clears ``first_chunk`` before checking it, so silence gets
written before the first chunk too instead of only between chunks.
"""
import unittest
import wave
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, patch


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
        from phoonnx.voice import _fetch_reference_audio

        first = _FakeResponse("http://example.com/a", redirect=True,
                               location="http://example.com/b")
        second = _FakeResponse("http://example.com/b", chunks=[b"hello"])
        responses = [first, second]

        def fake_get(url, timeout=None, stream=None, allow_redirects=None):
            return responses.pop(0)

        with patch("requests.get", side_effect=fake_get), \
                patch("phoonnx.voice._check_reference_url"):
            path = _fetch_reference_audio("http://example.com/a")

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
        from phoonnx.voice import _fetch_reference_audio

        first = _FakeResponse("http://example.com/a", redirect=True,
                               location="http://example.com/b")
        second = _FakeResponse("http://example.com/b", chunks=[b"hello"])
        responses = [first, second]

        def fake_get(url, timeout=None, stream=None, allow_redirects=None):
            return responses.pop(0)

        with patch("requests.get", side_effect=fake_get), \
                patch("phoonnx.voice._check_reference_url") as checker:
            path = _fetch_reference_audio("http://example.com/a")
        import os
        os.unlink(path)

        checked = [c.args[0] for c in checker.call_args_list]
        self.assertIn("http://example.com/a", checked)
        self.assertIn("http://example.com/b", checked)


@dataclass
class _FakeChunk:
    sample_rate: int = 16000
    sample_width: int = 2
    sample_channels: int = 1
    audio_float_array: object = None
    phonemes: List[str] = field(default_factory=list)
    phoneme_ids: List[int] = field(default_factory=list)

    @property
    def audio_int16_bytes(self):
        return b"\x01\x02"


class TestSentenceSilencePlacement(unittest.TestCase):
    def test_silence_only_between_chunks_not_before_first(self):
        """``sentence_silence`` is hardcoded to 0.0 in the source, so the
        silence bytes computed from it are always ``b""`` and the placement
        bug is invisible on real output. To exercise the placement logic
        itself we patch the single ``bytes(...)`` call the function makes
        (for computing ``silence_int16_bytes``) so it stands in for a
        non-zero silence duration, then drive the real, unpatched
        ``synthesize_wav`` with a stubbed chunk generator.
        """
        from phoonnx.voice import TTSVoice

        voice = MagicMock(spec=TTSVoice)
        voice.config = MagicMock()
        voice.config.sample_rate = 16000

        chunks = [_FakeChunk(), _FakeChunk(), _FakeChunk()]
        voice.synthesize.return_value = iter(chunks)

        wav_file = MagicMock(spec=wave.Wave_write)

        silence_marker = b"SILENCE"
        with patch("phoonnx.voice.bytes", return_value=silence_marker):
            TTSVoice.synthesize_wav(voice, "hello. world. again.", wav_file)

        calls = [c.args[0] for c in wav_file.writeframes.call_args_list]
        self.assertEqual(
            calls.count(silence_marker), 2,
            "3 chunks must produce exactly 2 between-chunk silence writes")
        self.assertNotEqual(
            calls[0], silence_marker,
            "the first writeframes call must be the first audio chunk, "
            "never silence")
        self.assertEqual(calls, [
            b"\x01\x02",     # chunk 1
            silence_marker,  # silence before chunk 2
            b"\x01\x02",     # chunk 2
            silence_marker,  # silence before chunk 3
            b"\x01\x02",     # chunk 3
        ])


if __name__ == "__main__":
    unittest.main()
