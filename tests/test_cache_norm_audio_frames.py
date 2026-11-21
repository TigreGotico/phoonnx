"""`cache_norm_audio` reports the length of the spectrogram it caches.

The preprocessor's too-short guard needs one number per utterance, and reading
it back off disk afterwards is serial work in the parent process. The function
that writes the spectrogram already holds it.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch

from phoonnx_train.norm_audio import cache_norm_audio, make_silence_detector


def _speech_like(seconds: float, sample_rate: int = 22050) -> np.ndarray:
    """A voiced-sounding tone burst: the detector has to keep something, or
    the cached spectrogram is empty and the test measures nothing."""
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    f0 = 120.0
    signal = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    envelope = 0.5 * (1 - np.cos(2 * np.pi * np.clip(t / seconds, 0, 1)))
    return (0.4 * signal * envelope).astype(np.float32)


class TestCachedFrameCount(unittest.TestCase):
    def setUp(self):
        self.detector = make_silence_detector()

    def _run(self, tmp, wav):
        cache = tmp / "cache"
        cache.mkdir(exist_ok=True)   # the caller owns this; cache_norm_audio does not
        return cache_norm_audio(wav, cache, self.detector, 22050)

    def test_the_count_matches_the_spectrogram_that_was_written(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wav = tmp / "clip.wav"
            sf.write(wav, _speech_like(1.0), 22050)

            _norm, spec_path, frames = self._run(tmp, wav)
            self.assertGreater(frames, 0)
            self.assertEqual(frames,
                             torch.load(spec_path, map_location="cpu").size(-1))

    def test_a_cache_hit_reports_the_same_count_as_the_write(self):
        # The second call computes nothing and must still answer, because a
        # resumed or re-run preprocessing pass takes this path for every row.
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            wav = tmp / "clip.wav"
            sf.write(wav, _speech_like(1.0), 22050)

            _n1, spec_path, first = self._run(tmp, wav)
            self.assertTrue(spec_path.is_file())
            _n2, _s2, second = self._run(tmp, wav)
            self.assertEqual(first, second)

    def test_a_longer_clip_reports_more_frames(self):
        # Guards against returning a constant, which every equality above
        # would still accept.
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            short_wav, long_wav = tmp / "short.wav", tmp / "long.wav"
            sf.write(short_wav, _speech_like(0.5), 22050)
            sf.write(long_wav, _speech_like(2.0), 22050)

            _n, _s, short_frames = self._run(tmp, short_wav)
            _n, _s, long_frames = self._run(tmp, long_wav)
            self.assertGreater(long_frames, short_frames)


if __name__ == "__main__":
    unittest.main()
