"""Silero VAD trimming must be deterministic per utterance.

The detector is an RNN whose hidden state (``_h``/``_c``) is fed back on every
call. A detector reused across a dataset must not let one clip's leftover state
bleed into the next — otherwise the trim boundaries (and thus which clips
survive the "audio too short for its text" guard) depend on the order clips
happen to be processed in, which varies across runs under multiprocessing.
"""
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from phoonnx_train.norm_audio import make_silence_detector
from phoonnx_train.norm_audio.trim import trim_silence

_VAD_MODEL = (
    Path(__file__).resolve().parent.parent
    / "phoonnx_train" / "norm_audio" / "models" / "silero_vad.onnx"
)

SR = 16000


def _clip(seconds=1.0, freq=180.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    # a voiced-ish burst in the middle framed by near-silence
    speech = 0.6 * np.sin(2 * np.pi * freq * t) * (rng.random(t.shape) * 0.5 + 0.5)
    env = np.zeros_like(t)
    a, b = int(0.3 * len(t)), int(0.7 * len(t))
    env[a:b] = 1.0
    return (speech * env).astype(np.float32)


@unittest.skipUnless(_VAD_MODEL.is_file(), "silero_vad.onnx not present")
class VadTrimDeterminismTests(unittest.TestCase):
    def test_trim_is_independent_of_prior_utterances(self):
        target = _clip(seconds=1.2, freq=200.0, seed=1)

        # Fresh detector trims the target.
        fresh = make_silence_detector()
        base = trim_silence(target.copy(), fresh, sample_rate=SR)

        # A detector that already processed OTHER clips (its RNN state is now
        # non-zero) must trim the same target identically.
        contaminated = make_silence_detector()
        for seed in range(3):
            contaminated(_clip(seconds=0.8, freq=90.0 + 30 * seed, seed=100 + seed),
                         sample_rate=SR)
        self.assertFalse(
            np.allclose(contaminated._h, 0.0) and np.allclose(contaminated._c, 0.0),
            "test setup failed to contaminate the detector state",
        )

        after = trim_silence(target.copy(), contaminated, sample_rate=SR)
        self.assertEqual(base, after)

    def test_repeated_trims_on_same_detector_match(self):
        det = make_silence_detector()
        clip = _clip(seconds=1.0, freq=220.0, seed=7)
        first = trim_silence(clip.copy(), det, sample_rate=SR)
        # feed unrelated audio, then trim the same clip again
        det(_clip(seconds=0.5, freq=110.0, seed=8), sample_rate=SR)
        second = trim_silence(clip.copy(), det, sample_rate=SR)
        self.assertEqual(first, second)

    def test_trim_resets_state_before_first_chunk(self):
        # Guaranteed-adversarial: contaminate the detector, then record the
        # hidden state the FIRST in-trim detector call sees. It must be zero —
        # i.e. trim_silence reset it. Without the reset this state is non-zero.
        det = make_silence_detector()
        for seed in range(3):
            det(_clip(seconds=0.7, freq=100.0 + 20 * seed, seed=200 + seed),
                sample_rate=SR)

        from phoonnx_train.norm_audio.vad import SileroVoiceActivityDetector

        states = []
        real_call = SileroVoiceActivityDetector.__call__

        def _spy(self, audio_array, sample_rate=16000):
            states.append((self._h.copy(), self._c.copy()))
            return real_call(self, audio_array, sample_rate=sample_rate)

        with unittest.mock.patch.object(
            SileroVoiceActivityDetector, "__call__", _spy
        ):
            trim_silence(_clip(seconds=1.0, freq=210.0, seed=9), det, sample_rate=SR)
        self.assertTrue(states, "detector was never called during trim")
        first_h, first_c = states[0]
        self.assertTrue(np.allclose(first_h, 0.0))
        self.assertTrue(np.allclose(first_c, 0.0))

    def test_reset_zeroes_state(self):
        det = make_silence_detector()
        det(_clip(seconds=0.6, freq=150.0, seed=3), sample_rate=SR)
        det.reset()
        self.assertTrue(np.allclose(det._h, 0.0))
        self.assertTrue(np.allclose(det._c, 0.0))


if __name__ == "__main__":
    unittest.main()
