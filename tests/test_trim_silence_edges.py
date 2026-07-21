"""trim_silence() must score every chunk of the clip, including the tail,
and must return a bounded (offset, duration) even when speech occupies only
a single chunk.

These tests drive trim_silence() with a deterministic fake detector instead
of the real Silero ONNX model, so they run everywhere without an onnxruntime
dependency (mirrors the _FakeSilenceDetector pattern already used in
tests/test_preprocess_pipeline.py).
"""
import unittest

import numpy as np

from phoonnx_train.norm_audio.trim import trim_silence

SR = 16000
SAMPLES_PER_CHUNK = 480


class _ChunkScriptDetector:
    """Reports speech/non-speech per call according to a fixed script,
    one entry per chunk in call order. Tracks reset() calls so tests can
    assert the per-utterance reset contract is preserved."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        self.calls = 0

    def __call__(self, audio_array, sample_rate=SR):
        idx = self.calls
        self.calls += 1
        if idx >= len(self.script):
            raise AssertionError(
                f"detector called for chunk {idx}, beyond scripted "
                f"{len(self.script)} chunks"
            )
        return 1.0 if self.script[idx] else 0.0


def _silence(num_chunks, samples_per_chunk=SAMPLES_PER_CHUNK):
    return np.zeros(num_chunks * samples_per_chunk, dtype=np.float32)


class TrimSilenceTailChunkTests(unittest.TestCase):
    def test_speech_only_in_final_chunk_is_detected_and_trimmed(self):
        # 5 chunks total; speech only in the very last one. The old chunk
        # loop primed/advanced such that the final chunk was never scored,
        # so this speech onset was silently dropped.
        script = [False, False, False, False, True]
        audio = _silence(len(script))
        det = _ChunkScriptDetector(script)

        offset, duration = trim_silence(
            audio, det, samples_per_chunk=SAMPLES_PER_CHUNK, sample_rate=SR,
            keep_chunks_before=0, keep_chunks_after=0,
        )

        self.assertEqual(det.calls, len(script), "final chunk was not scored")
        self.assertIsNotNone(duration)
        seconds_per_chunk = SAMPLES_PER_CHUNK / SR
        self.assertAlmostEqual(offset, 4 * seconds_per_chunk)
        self.assertAlmostEqual(duration, seconds_per_chunk)

    def test_speech_in_single_middle_chunk_yields_bounded_result(self):
        # Exactly one speech chunk: first_chunk gets set but last_chunk was
        # never assigned by the old code, so (first is not None and last is
        # not None) failed and trimming was skipped entirely -> (0.0, None).
        script = [False, False, True, False, False]
        audio = _silence(len(script))
        det = _ChunkScriptDetector(script)

        offset, duration = trim_silence(
            audio, det, samples_per_chunk=SAMPLES_PER_CHUNK, sample_rate=SR,
            keep_chunks_before=1, keep_chunks_after=1,
        )

        self.assertIsNotNone(duration, "lone speech chunk produced no trim window")
        seconds_per_chunk = SAMPLES_PER_CHUNK / SR
        self.assertAlmostEqual(offset, 1 * seconds_per_chunk)  # chunk 2 - 1
        self.assertAlmostEqual(duration, 3 * seconds_per_chunk)  # chunks 1..3 inclusive

    def test_single_speech_chunk_is_the_very_last_chunk(self):
        # Combines both bugs: the only speech is in the tail chunk.
        script = [False, False, False, True]
        audio = _silence(len(script))
        det = _ChunkScriptDetector(script)

        offset, duration = trim_silence(
            audio, det, samples_per_chunk=SAMPLES_PER_CHUNK, sample_rate=SR,
            keep_chunks_before=0, keep_chunks_after=2,
        )

        self.assertEqual(det.calls, len(script))
        self.assertIsNotNone(duration)
        seconds_per_chunk = SAMPLES_PER_CHUNK / SR
        self.assertAlmostEqual(offset, 3 * seconds_per_chunk)
        # last_chunk clamps to the final chunk index (3), keep_after=2 is bounded
        self.assertAlmostEqual(duration, 1 * seconds_per_chunk)

    def test_all_silence_returns_no_trim_window(self):
        script = [False] * 5
        audio = _silence(len(script))
        det = _ChunkScriptDetector(script)

        offset, duration = trim_silence(
            audio, det, samples_per_chunk=SAMPLES_PER_CHUNK, sample_rate=SR,
        )

        self.assertEqual(det.calls, len(script))
        self.assertEqual(offset, 0.0)
        self.assertIsNone(duration)

    def test_resets_detector_state_once_per_call(self):
        script = [True, False, False]
        audio = _silence(len(script))
        det = _ChunkScriptDetector(script)
        trim_silence(audio, det, samples_per_chunk=SAMPLES_PER_CHUNK, sample_rate=SR)
        self.assertEqual(det.reset_calls, 1)

    def test_repeated_calls_are_deterministic(self):
        script = [False, True, False, False, True]
        audio = _silence(len(script))
        det = _ChunkScriptDetector(script)

        first = trim_silence(
            audio.copy(), det, samples_per_chunk=SAMPLES_PER_CHUNK, sample_rate=SR,
        )
        second = trim_silence(
            audio.copy(), det, samples_per_chunk=SAMPLES_PER_CHUNK, sample_rate=SR,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
