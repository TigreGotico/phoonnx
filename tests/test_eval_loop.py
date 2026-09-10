"""Tests for the torch-free helpers behind the checkpoint evaluation loop.

phoonnx_train.eval_utils is loaded by file path so the suite runs without
torch installed (the eval loop itself needs torch, but its bookkeeping does
not).
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

_UTILS = Path(__file__).parent.parent / "phoonnx_train" / "eval_utils.py"
spec = importlib.util.spec_from_file_location("eval_utils", _UTILS)
eval_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_utils)


class TestParseNames(unittest.TestCase):
    def test_parse_epoch(self):
        self.assertEqual(eval_utils.parse_epoch("epoch=12-step=3400.ckpt"), 12)
        self.assertEqual(eval_utils.parse_epoch("epoch=0.ckpt"), 0)
        self.assertEqual(eval_utils.parse_epoch("epoch=007.ckpt"), 7)

    def test_parse_epoch_missing(self):
        self.assertIsNone(eval_utils.parse_epoch("last.ckpt"))
        self.assertIsNone(eval_utils.parse_epoch("epoch=.ckpt"))
        self.assertIsNone(eval_utils.parse_epoch("epoch=-3.ckpt"))
        self.assertIsNone(eval_utils.parse_epoch(""))

    def test_parse_step(self):
        self.assertEqual(eval_utils.parse_step("epoch=1-step=250.ckpt"), 250)
        self.assertEqual(eval_utils.parse_step("epoch=1.ckpt"), -1)
        self.assertEqual(eval_utils.parse_step("step=abc.ckpt"), -1)


class TestFindCheckpoints(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _touch(self, rel, mtime=None):
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        if mtime is not None:
            os.utime(p, (mtime, mtime))
        return p

    def test_finds_nested_and_ignores_no_epoch(self):
        a = self._touch("lightning_logs/version_0/checkpoints/epoch=3-step=9.ckpt")
        self._touch("lightning_logs/version_0/checkpoints/last.ckpt")
        self._touch("notes.txt")
        found = eval_utils.find_checkpoints(self.dir)
        self.assertEqual(found, {3: a})

    def test_duplicate_epoch_highest_mtime_wins(self):
        self._touch("v0/epoch=5-step=100.ckpt", mtime=1000)
        newer = self._touch("v1/epoch=5-step=200.ckpt", mtime=2000)
        found = eval_utils.find_checkpoints(self.dir)
        self.assertEqual(found[5], newer)

    def test_empty_dir(self):
        self.assertEqual(eval_utils.find_checkpoints(self.dir), {})


class TestMetricsCsv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.csv = Path(self.tmp.name) / "metrics.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def _row(self, epoch, mean):
        return {"epoch": epoch, "step": 1, "checkpoint": f"e{epoch}.ckpt",
                "utmos_mean": f"{mean:.4f}", "utmos_std": "0.1",
                "utmos_min": "1.0", "utmos_max": "5.0", "spk_sim_mean": "",
                "spk_sim_std": "", "spk_sim_min": "", "n_sentences": 10}

    def test_missing_file(self):
        self.assertEqual(eval_utils.done_epochs(self.csv), set())
        self.assertIsNone(eval_utils.best_mean_so_far(self.csv))

    def test_append_roundtrip_idempotency(self):
        eval_utils.append_metrics(self.csv, self._row(0, 2.5))
        eval_utils.append_metrics(self.csv, self._row(4, 3.1))
        self.assertEqual(eval_utils.done_epochs(self.csv), {0, 4})
        self.assertAlmostEqual(eval_utils.best_mean_so_far(self.csv), 3.1)
        # header written exactly once
        lines = self.csv.read_text().strip().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("epoch,step,checkpoint"))

    def test_missing_columns_padded_empty(self):
        eval_utils.append_metrics(self.csv, {"epoch": 1, "utmos_mean": "2.0"})
        line = self.csv.read_text().strip().splitlines()[1]
        self.assertEqual(len(line.split(",")),
                         len(eval_utils.METRICS_HEADER))

    def test_malformed_rows_skipped(self):
        self.csv.write_text(
            "epoch,step,checkpoint,utmos_mean,utmos_std,utmos_min,utmos_max,"
            "spk_sim_mean,spk_sim_std,spk_sim_min,n_sentences\n"
            "1,10,a.ckpt,3.5,0.1,3,4,,,,10\n"
            "garbage,x,b.ckpt,not-a-float,,,,,,,\n"
            "3,30,c.ckpt,2.9,0.1,2,3,,,,10\n"
            ",,,\n"
            "5\n")
        self.assertEqual(eval_utils.done_epochs(self.csv), {1, 3, 5})
        self.assertAlmostEqual(eval_utils.best_mean_so_far(self.csv), 3.5)

    def test_headerless_garbage_file(self):
        self.csv.write_text("complete nonsense\nwithout,any,header\n")
        self.assertEqual(eval_utils.done_epochs(self.csv), set())
        self.assertIsNone(eval_utils.best_mean_so_far(self.csv))

    def test_empty_file(self):
        self.csv.write_text("")
        self.assertEqual(eval_utils.done_epochs(self.csv), set())
        self.assertIsNone(eval_utils.best_mean_so_far(self.csv))


class TestReadSentences(unittest.TestCase):
    def test_strips_and_drops_blanks(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.txt"
            p.write_text("  hello \n\n\t\nworld\n   \n", encoding="utf-8")
            self.assertEqual(eval_utils.read_sentences(p), ["hello", "world"])

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.txt"
            p.write_text("", encoding="utf-8")
            self.assertEqual(eval_utils.read_sentences(p), [])


class TestSizeStable(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "ckpt"

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file(self):
        self.assertFalse(eval_utils.size_stable(self.path, wait=0,
                                                sleep=lambda _: None))

    def test_stable_file(self):
        self.path.write_bytes(b"abc")
        calls = []
        self.assertTrue(eval_utils.size_stable(self.path, wait=0,
                                               sleep=calls.append))
        self.assertEqual(len(calls), 1)  # one sleep to confirm stability

    def test_empty_file_never_stable(self):
        self.path.write_bytes(b"")
        self.assertFalse(eval_utils.size_stable(self.path, wait=0, tries=3,
                                                sleep=lambda _: None))

    def test_growing_file_unstable(self):
        self.path.write_bytes(b"a")

        def grow(_):
            with open(self.path, "ab") as f:
                f.write(b"a")

        self.assertFalse(eval_utils.size_stable(self.path, wait=0, tries=4,
                                                sleep=grow))

    def test_file_deleted_mid_wait(self):
        self.path.write_bytes(b"a")

        def grow_then_delete(_):
            if self.path.exists():
                with open(self.path, "ab") as f:
                    f.write(b"bb")  # keep it unstable, then remove
                self.path.unlink()

        self.assertFalse(eval_utils.size_stable(self.path, wait=0, tries=5,
                                                sleep=grow_then_delete))


class TestLargestWavs(unittest.TestCase):
    def test_orders_by_size_and_caps_n(self):
        with tempfile.TemporaryDirectory() as d:
            dp = Path(d)
            (dp / "small.wav").write_bytes(b"a")
            (dp / "big.wav").write_bytes(b"a" * 100)
            (dp / "mid.wav").write_bytes(b"a" * 10)
            (dp / "not_audio.txt").write_bytes(b"a" * 1000)
            top2 = eval_utils.largest_wavs(dp, 2)
            self.assertEqual([p.name for p in top2], ["big.wav", "mid.wav"])
            self.assertEqual(eval_utils.largest_wavs(dp, 0), [])

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(eval_utils.largest_wavs(Path(d), 5), [])


if __name__ == "__main__":
    unittest.main()
