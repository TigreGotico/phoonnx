"""A loudness target keeps synthesized sentences level, where peak normalization does not."""
import sys
import unittest
from unittest import mock

import numpy as np

from phoonnx.config import SynthesisConfig
from phoonnx.engines.outetts import integrated_loudness
from phoonnx.voice import level_audio

RATE = 22050


def speechlike(amplitude: float, crest: float = 0.0, seconds: float = 1.5) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    y = amplitude * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    if crest:
        y[len(y) // 2] = crest  # a lone peak, the way a plosive makes one
    return y.astype(np.float32)


class TestPeakNormalizationIsTheDefault(unittest.TestCase):

    def test_default_scales_to_full_range(self):
        out = level_audio(speechlike(0.2), SynthesisConfig(), RATE)
        self.assertAlmostEqual(float(np.max(np.abs(out))), 1.0, places=5)

    def test_silence_stays_silence(self):
        out = level_audio(np.zeros(1000, np.float32), SynthesisConfig(), RATE)
        self.assertEqual(float(np.max(np.abs(out))), 0.0)

    def test_no_normalization_returns_the_audio_untouched(self):
        audio = speechlike(0.2)
        out = level_audio(audio, SynthesisConfig(normalize_audio=False), RATE)
        np.testing.assert_array_equal(out, audio)


class TestTargetLoudness(unittest.TestCase):

    def test_sentences_of_different_level_come_out_at_the_target(self):
        config = SynthesisConfig(target_loudness=-16.0)
        for amplitude in (0.9, 0.3, 0.05):
            with self.subTest(amplitude=amplitude):
                out = level_audio(speechlike(amplitude), config, RATE)
                self.assertAlmostEqual(integrated_loudness(out, RATE), -16.0, delta=0.5)

    def test_peaks_are_limited_below_full_scale(self):
        out = level_audio(speechlike(0.05, crest=0.9), SynthesisConfig(target_loudness=-10.0), RATE)
        self.assertLessEqual(float(np.max(np.abs(out))), 10 ** (-1.0 / 20) + 1e-4)

    def test_the_target_takes_precedence_over_peak_normalization(self):
        loud = level_audio(speechlike(0.9), SynthesisConfig(target_loudness=-30.0), RATE)
        self.assertLess(float(np.max(np.abs(loud))), 0.5)

    def test_silence_stays_silence(self):
        out = level_audio(np.zeros(2000, np.float32), SynthesisConfig(target_loudness=-16.0), RATE)
        self.assertEqual(float(np.max(np.abs(out))), 0.0)

    def test_without_scipy_it_warns_once_and_normalizes_the_peak(self):
        import phoonnx.voice as voice
        voice._LOUDNESS_WARNED = False
        blocked = {"scipy": None, "scipy.signal": None}
        with mock.patch.dict(sys.modules, blocked), \
                mock.patch.object(voice.LOG, "warning") as warning:
            out = level_audio(speechlike(0.2), SynthesisConfig(target_loudness=-16.0), RATE)
            level_audio(speechlike(0.2), SynthesisConfig(target_loudness=-16.0), RATE)
        self.assertAlmostEqual(float(np.max(np.abs(out))), 1.0, places=5)
        self.assertEqual(warning.call_count, 1)
        self.assertIn("scipy", warning.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
