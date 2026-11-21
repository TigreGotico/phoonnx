"""Tests for the numpy monotonic alignment search used in VITS training.

The numpy core is loaded by file path: the monotonic_align package wraps it in
torch tensors for training, and torch is not part of the test environment.
"""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

_CORE = (Path(__file__).parent.parent / "phoonnx_train" / "vits" /
         "monotonic_align" / "core_numpy.py")
spec = importlib.util.spec_from_file_location("core_numpy", _CORE)
core_numpy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core_numpy)
maximum_path_numpy = core_numpy.maximum_path_numpy


def reference_maximum_path(value: np.ndarray, t_s: int, t_t: int) -> np.ndarray:
    """Brute-force DP reference for a single sample: value [t_s, t_t]."""
    max_neg = -1e9
    best = np.full((t_s, t_t), max_neg, dtype=np.float64)
    for j in range(t_t):
        for x in range(t_s):
            if x > j or t_s - x > t_t - j:
                continue
            prev = 0.0 if (x == 0 and j == 0) else max(
                best[x, j - 1] if j > 0 and x <= j - 1 else max_neg,
                best[x - 1, j - 1] if j > 0 and x > 0 else max_neg,
            )
            best[x, j] = prev + value[x, j]
    path = np.zeros((t_s, t_t), dtype=np.float32)
    x = t_s - 1
    for j in range(t_t - 1, -1, -1):
        path[x, j] = 1
        if x > 0 and (x == j or best[x, j - 1] < best[x - 1, j - 1]):
            x -= 1
    return path


def make_batch(lengths, t_s_pad, t_t_pad, seed):
    rng = np.random.default_rng(seed)
    b = len(lengths)
    value = (rng.standard_normal((b, t_s_pad, t_t_pad)) * 10).astype(np.float32)
    mask = np.zeros((b, t_s_pad, t_t_pad), dtype=bool)
    for i, (t_s, t_t) in enumerate(lengths):
        mask[i, :t_s, :t_t] = True
    return value, mask


class TestMaximumPathNumpy(unittest.TestCase):
    def assert_valid_path(self, path, t_s, t_t):
        p = path[:t_s, :t_t]
        # one text position per frame
        np.testing.assert_array_equal(p.sum(axis=0), np.ones(t_t))
        # monotonic, no skips, starts at 0, ends at t_s - 1
        idx = p.argmax(axis=0)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], t_s - 1)
        self.assertTrue(np.all(np.isin(np.diff(idx), [0, 1])))

    def test_matches_reference(self):
        lengths = [(12, 40), (37, 37), (1, 25), (20, 33)]
        value, mask = make_batch(lengths, 40, 40, seed=0)
        out = maximum_path_numpy(value, mask)
        for i, (t_s, t_t) in enumerate(lengths):
            ref = reference_maximum_path(
                value[i].astype(np.float64) * mask[i], t_s, t_t)
            np.testing.assert_array_equal(out[i, :t_s, :t_t], ref, err_msg=f"sample {i}")
            self.assert_valid_path(out[i], t_s, t_t)

    def test_many_random_batches_are_valid(self):
        rng = np.random.default_rng(99)
        for trial in range(50):
            t_t = int(rng.integers(1, 120))
            t_s = int(rng.integers(1, t_t + 1))
            value, mask = make_batch([(t_s, t_t)], t_s, t_t, seed=trial)
            self.assert_valid_path(maximum_path_numpy(value, mask)[0], t_s, t_t)

    def test_padding_untouched(self):
        lengths = [(8, 20), (5, 14)]
        value, mask = make_batch(lengths, 15, 30, seed=1)
        out = maximum_path_numpy(value, mask)
        for i, (t_s, t_t) in enumerate(lengths):
            self.assertEqual(out[i, t_s:, :].sum(), 0)
            self.assertEqual(out[i, :, t_t:].sum(), 0)

    def test_more_text_than_frames_does_not_crash(self):
        # geometrically impossible alignment (phonemes > frames): must return,
        # not index out of bounds
        lengths = [(25, 10), (25, 25), (30, 5)]
        value, mask = make_batch(lengths, 30, 30, seed=2)
        out = maximum_path_numpy(value, mask)
        self.assertEqual(out.shape, value.shape)

    def test_single_frame_single_token(self):
        value, mask = make_batch([(1, 1)], 1, 1, seed=3)
        self.assertEqual(maximum_path_numpy(value, mask)[0, 0, 0], 1)

    def test_prefers_high_score_alignment(self):
        # a diagonal of high scores must be followed exactly
        t = 6
        value = np.full((1, t, t), -100.0, dtype=np.float32)
        np.fill_diagonal(value[0], 10.0)
        mask = np.ones((1, t, t), dtype=bool)
        out = maximum_path_numpy(value, mask)[0]
        np.testing.assert_array_equal(out, np.eye(t, dtype=np.float32))

    def test_input_not_mutated(self):
        value, mask = make_batch([(6, 11)], 6, 11, seed=4)
        before = value.copy()
        maximum_path_numpy(value, mask)
        np.testing.assert_array_equal(value, before)


if __name__ == "__main__":
    unittest.main()
