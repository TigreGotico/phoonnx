"""trim_silence() must score every chunk of the clip, including the tail,
and must return a bounded (offset, duration) even when speech occupies only
a single chunk.

Most of these drive trim_silence() with a deterministic fake detector rather
than the real Silero ONNX model, which keeps the chunk-accounting cases exact
(mirrors the _FakeSilenceDetector pattern already used in
tests/test_preprocess_pipeline.py). That stub accepts a chunk of any length,
so it is blind to the model's own minimum; the last class drives the real
detector for the cases that turn on it.
"""
import unittest

import numpy as np

from phoonnx_train.norm_audio import trim
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
        self.chunk_lengths = []

    def reset(self):
        self.reset_calls += 1
        self.calls = 0

    def __call__(self, audio_array, sample_rate=SR):
        idx = self.calls
        self.chunk_lengths.append(len(audio_array))
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

    def test_tail_is_padded_only_up_to_the_detector_floor(self):
        # Padding a short tail out to a full chunk would re-score every tail
        # the detector already accepted, against a different amount of
        # padding than it saw before, and move boundaries on clips that
        # worked. Only a tail below the floor is touched.
        for tail, expected in ((64, trim.DETECTOR_MIN_SAMPLES),
                               (129, 129), (300, 300), (479, 479)):
            with self.subTest(tail=tail):
                audio = np.concatenate(
                    [_silence(2), np.zeros(tail, dtype=np.float32)])
                det = _ChunkScriptDetector([False, False, False])
                trim_silence(audio, det, samples_per_chunk=SAMPLES_PER_CHUNK,
                             sample_rate=SR)
                self.assertEqual(det.chunk_lengths[-1], expected)

    def test_full_chunks_are_never_padded(self):
        audio = _silence(3)
        det = _ChunkScriptDetector([False] * 3)
        trim_silence(audio, det, samples_per_chunk=SAMPLES_PER_CHUNK,
                     sample_rate=SR)
        self.assertEqual(det.chunk_lengths, [SAMPLES_PER_CHUNK] * 3)

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


def _speech_like(num_samples, seed=0):
    """A deterministic voiced signal the shipped Silero model scores as speech.

    A plain sine does not clear the threshold however loud it is, and real
    speech cannot be checked in as a fixture, so this is a source-filter
    synthesis: a glottal pulse train with an f0 contour driven through three
    time-varying formant resonators and a syllabic envelope.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(num_samples) / SR
    f0 = 130.0 + 25.0 * np.sin(2 * np.pi * 3.1 * t)
    source = (np.diff(np.floor(np.cumsum(f0) / SR), prepend=0) > 0).astype(float)
    source += 0.02 * rng.standard_normal(num_samples)

    formants = [(np.linspace(730, 300, num_samples), 80),
                (np.linspace(1090, 2300, num_samples), 90),
                (np.linspace(2440, 3000, num_samples), 120)]
    out = np.zeros(num_samples)
    for track, bandwidth in formants:
        r = np.exp(-np.pi * bandwidth / SR)
        y = np.zeros(num_samples)
        for i in range(num_samples):
            theta = 2 * np.pi * track[i] / SR
            a1, a2 = -2 * r * np.cos(theta), r * r
            y[i] = source[i] - a1 * (y[i - 1] if i else 0) - a2 * (y[i - 2] if i > 1 else 0)
        out += y
    out *= 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    return (0.9 * out / (np.max(np.abs(out)) + 1e-9)).astype(np.float32)


def _near_silence(num_chunks, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(num_chunks * SAMPLES_PER_CHUNK) * 1e-4).astype(np.float32)


class TrimSilenceRealDetectorTailTests(unittest.TestCase):
    """The two things the stub detector above cannot see.

    That stub scores a chunk of any length, so it is blind to the shipped
    Silero model's 129-sample floor: below that the model raises on the Pad
    node rather than returning a probability, and the exception propagates out
    of ``trim_silence`` and loses the whole utterance. ``onnxruntime`` is a
    base dependency and the model ships in package data, so these run
    unconditionally rather than behind a skip.
    """

    def setUp(self):
        from phoonnx_train.norm_audio import make_silence_detector

        self.detector = make_silence_detector()

    def test_tail_below_the_detector_floor_does_not_abort_the_utterance(self):
        # 64 samples of tail, half the 129 the model accepts. Unpadded this
        # raises InvalidArgument on Pad_27 and the caller loses the clip.
        audio = np.concatenate([_near_silence(10), _speech_like(1440, seed=3),
                                _speech_like(64, seed=2)])
        self.assertEqual(len(audio) % SAMPLES_PER_CHUNK, 64)

        offset_sec, duration_sec = trim_silence(
            audio, self.detector,
            samples_per_chunk=SAMPLES_PER_CHUNK, sample_rate=SR,
            keep_chunks_before=0, keep_chunks_after=0,
        )

        self.assertIsNotNone(duration_sec, "speech before the tail was lost")
        seconds_per_chunk = SAMPLES_PER_CHUNK / SR
        self.assertAlmostEqual(offset_sec, 10 * seconds_per_chunk)

    def test_tail_chunk_is_scored_rather_than_skipped(self):
        # Dropping the short tail unscored also avoids the exception, so the
        # test above cannot tell that regression from the fix. This one can:
        # the trim window has to reach the end of the tail chunk. keep_after
        # is 0 because the default 2 clamps to the last chunk either way and
        # would hide the difference.
        audio = np.concatenate([_near_silence(10), _speech_like(1440, seed=3),
                                _speech_like(300, seed=2)])
        num_chunks = (len(audio) + SAMPLES_PER_CHUNK - 1) // SAMPLES_PER_CHUNK

        offset_sec, duration_sec = trim_silence(
            audio, self.detector,
            samples_per_chunk=SAMPLES_PER_CHUNK, sample_rate=SR,
            keep_chunks_before=0, keep_chunks_after=0,
        )

        seconds_per_chunk = SAMPLES_PER_CHUNK / SR
        self.assertIsNotNone(duration_sec)
        self.assertAlmostEqual(
            offset_sec + duration_sec, num_chunks * seconds_per_chunk,
            msg="trim window stops short of the tail chunk",
        )


if __name__ == "__main__":
    unittest.main()
