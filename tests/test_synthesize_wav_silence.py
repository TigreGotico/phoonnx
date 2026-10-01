"""``synthesize_wav`` must write sentence silence only between chunks.

Clearing ``first_chunk`` before checking it wrote silence ahead of the first
chunk too, padding the start of every utterance.
"""
import unittest
import wave
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, patch


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
