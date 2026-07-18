"""Tests for the numpy monotonic alignment search used in VITS training."""
import unittest

import numpy as np
import torch

from phoonnx_train.vits.monotonic_align import maximum_path


def reference_maximum_path(neg_cent: np.ndarray, t_t: int, t_s: int) -> np.ndarray:
    """Brute-force DP reference for a single sample: value [t_t, t_s]."""
    max_neg = -1e9
    value = np.full((t_t, t_s), max_neg, dtype=np.float64)
    for y in range(t_t):
        for x in range(t_s):
            if x > y or t_s - x > t_t - y:
                continue
            best = 0.0 if (x == 0 and y == 0) else max(
                value[y - 1, x] if y > 0 and x <= y - 1 else max_neg,
                value[y - 1, x - 1] if y > 0 and x > 0 else max_neg,
            )
            value[y, x] = best + neg_cent[y, x]
    path = np.zeros((t_t, t_s), dtype=np.float32)
    x = t_s - 1
    for y in range(t_t - 1, -1, -1):
        path[y, x] = 1
        if x > 0 and (x == y or value[y - 1, x] < value[y - 1, x - 1]):
            x -= 1
    return path


def make_batch(lengths, t_t_pad, t_s_pad, seed):
    rng = np.random.default_rng(seed)
    b = len(lengths)
    neg_cent = torch.from_numpy(
        rng.standard_normal((b, t_t_pad, t_s_pad)).astype(np.float32))
    mask = torch.zeros((b, t_t_pad, t_s_pad))
    for i, (t_t, t_s) in enumerate(lengths):
        mask[i, :t_t, :t_s] = 1
    return neg_cent, mask


class TestMaximumPath(unittest.TestCase):
    def assert_valid_path(self, path, t_t, t_s):
        p = path[:t_t, :t_s]
        # one text position per frame
        self.assertTrue(np.array_equal(p.sum(axis=1), np.ones(t_t)))
        # monotonic, no skips, starts at 0, ends at t_s - 1
        idx = p.argmax(axis=1)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], t_s - 1)
        self.assertTrue(np.all(np.isin(np.diff(idx), [0, 1])))

    def test_matches_reference(self):
        lengths = [(40, 12), (37, 37), (25, 1), (33, 20)]
        neg_cent, mask = make_batch(lengths, 40, 40, seed=0)
        out = maximum_path(neg_cent, mask).numpy()
        for i, (t_t, t_s) in enumerate(lengths):
            ref = reference_maximum_path(
                neg_cent[i].numpy().astype(np.float64), t_t, t_s)
            np.testing.assert_array_equal(out[i, :t_t, :t_s], ref, err_msg=f"sample {i}")
            self.assert_valid_path(out[i], t_t, t_s)

    def test_padding_untouched(self):
        lengths = [(20, 8), (14, 5)]
        neg_cent, mask = make_batch(lengths, 30, 15, seed=1)
        out = maximum_path(neg_cent, mask).numpy()
        for i, (t_t, t_s) in enumerate(lengths):
            self.assertEqual(out[i, t_t:, :].sum(), 0)
            self.assertEqual(out[i, :, t_s:].sum(), 0)

    def test_more_text_than_frames_does_not_crash(self):
        # geometrically impossible alignment (phonemes > frames): must return,
        # not read out of bounds
        lengths = [(10, 25), (25, 25), (5, 30)]
        neg_cent, mask = make_batch(lengths, 30, 30, seed=2)
        out = maximum_path(neg_cent, mask)
        self.assertEqual(out.shape, neg_cent.shape)

    def test_single_frame_single_token(self):
        neg_cent, mask = make_batch([(1, 1)], 1, 1, seed=3)
        out = maximum_path(neg_cent, mask).numpy()
        self.assertEqual(out[0, 0, 0], 1)

    def test_device_and_dtype_preserved(self):
        neg_cent, mask = make_batch([(12, 6)], 12, 6, seed=4)
        out = maximum_path(neg_cent.double(), mask)
        self.assertEqual(out.dtype, torch.float64)
        self.assertEqual(out.device, neg_cent.device)

    def test_gradient_isolation(self):
        # maximum_path must not require grad even when inputs do
        neg_cent, mask = make_batch([(15, 7)], 15, 7, seed=5)
        out = maximum_path(neg_cent.requires_grad_(True), mask)
        self.assertFalse(out.requires_grad)

    def test_prefers_low_cost_alignment(self):
        # a diagonal of high scores must be followed exactly
        t = 6
        neg_cent = torch.full((1, t, t), -100.0)
        for i in range(t):
            neg_cent[0, i, i] = 10.0
        mask = torch.ones((1, t, t))
        out = maximum_path(neg_cent, mask).numpy()[0]
        np.testing.assert_array_equal(out, np.eye(t, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
