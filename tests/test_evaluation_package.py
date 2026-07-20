"""Tests for the reusable checkpoint-evaluation package.

Covers selection (similarity gate + UTMOS-only fallback + atomic best writes),
tracker (old-format CSV read, failed-epoch marking, patience/stop.flag),
callbacks (stop-file honoured, scorer failure isolated), scorer determinism
(per-utterance reseed), and the eval_loop CLI (--once with a mocked engine).

Adversarial cases are included: high-UTMOS/low-sim rows must be excluded by the
gate, a scorer that raises must not stop training, and a checkpoint that fails
three times must never be retried.
"""
import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np

from phoonnx_train.evaluation.scorer import EvalRow
from phoonnx_train.evaluation.selection import SelectionPolicy
from phoonnx_train.evaluation.tracker import MetricsTracker, SUPERSET_HEADER


def _row(epoch, utmos, spk_sim=None, step=10):
    agg = {"utmos_mean": utmos, "utmos_std": 0.1, "utmos_min": utmos - 0.1,
           "utmos_max": utmos + 0.1}
    if spk_sim is not None:
        agg.update({"spk_sim_mean": spk_sim, "spk_sim_std": 0.0,
                    "spk_sim_min": spk_sim, "spk_sim_max": spk_sim})
    return EvalRow(epoch=epoch, step=step, checkpoint=f"epoch={epoch}.ckpt",
                   n_sentences=5, aggregates=agg)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
class TestSelectionPolicy(unittest.TestCase):
    def test_gate_excludes_high_utmos_low_sim(self):
        pol = SelectionPolicy(metric="utmos_mean", min_spk_sim=0.7)
        best = _row(1, utmos=3.0, spk_sim=0.8)
        # higher UTMOS but below the similarity floor -> not an improvement
        candidate = _row(2, utmos=4.5, spk_sim=0.5)
        self.assertFalse(pol.is_eligible(candidate))
        self.assertFalse(pol.is_improvement(candidate, best))

    def test_gate_allows_high_utmos_high_sim(self):
        pol = SelectionPolicy(metric="utmos_mean", min_spk_sim=0.7)
        best = _row(1, utmos=3.0, spk_sim=0.8)
        candidate = _row(2, utmos=4.5, spk_sim=0.75)
        self.assertTrue(pol.is_improvement(candidate, best))

    def test_no_sim_falls_back_to_utmos_only(self):
        pol = SelectionPolicy(metric="utmos_mean", min_spk_sim=0.7)
        # rows carry NO speaker score -> gate cannot apply, UTMOS-only
        best = _row(1, utmos=3.0)
        candidate = _row(2, utmos=3.5)
        self.assertTrue(pol.is_eligible(candidate))
        self.assertTrue(pol.is_improvement(candidate, best))

    def test_no_floor_means_no_gate(self):
        pol = SelectionPolicy(metric="utmos_mean", min_spk_sim=None)
        best = _row(1, utmos=3.0, spk_sim=0.9)
        candidate = _row(2, utmos=3.5, spk_sim=0.1)  # awful sim, but no floor
        self.assertTrue(pol.is_improvement(candidate, best))

    def test_first_candidate_is_improvement(self):
        pol = SelectionPolicy(min_spk_sim=0.7)
        self.assertTrue(pol.is_improvement(_row(1, 3.0, spk_sim=0.8), None))
        # ...unless it fails the gate
        self.assertFalse(pol.is_improvement(_row(1, 3.0, spk_sim=0.5), None))

    def test_equal_metric_is_not_improvement(self):
        pol = SelectionPolicy()
        self.assertFalse(pol.is_improvement(_row(2, 3.0), _row(1, 3.0)))

    def test_commit_best_writes_json_and_copies_ckpt(self):
        with TemporaryDirectory() as d:
            out = Path(d) / "eval"
            ckpt = Path(d) / "epoch=2.ckpt"
            ckpt.write_bytes(b"weights")
            work = Path(d) / "work"
            work.mkdir()
            (work / "utt00.wav").write_bytes(b"RIFF")
            row = _row(2, utmos=4.0, spk_sim=0.8)
            row.checkpoint = str(ckpt)
            pol = SelectionPolicy(min_spk_sim=0.7)
            pol.commit_best(row, out, work_dir=work)

            best = json.loads((out / "best.json").read_text())
            self.assertEqual(best["epoch"], 2)
            self.assertEqual(best["scores"]["utmos_mean"], 4.0)
            self.assertEqual((out / "best.ckpt").read_bytes(), b"weights")
            self.assertTrue((out / "samples" / "epoch2" / "utt00.wav").exists())
            # no leftover temp files
            self.assertFalse(list(out.glob("*.tmp")))

    def test_read_best_roundtrips(self):
        with TemporaryDirectory() as d:
            out = Path(d)
            ckpt = out / "e.ckpt"
            ckpt.write_bytes(b"x")
            row = _row(3, utmos=3.9, spk_sim=0.85)
            row.checkpoint = str(ckpt)
            SelectionPolicy().commit_best(row, out)
            back = SelectionPolicy.read_best(out)
            self.assertEqual(back.epoch, 3)
            self.assertAlmostEqual(back.value("utmos_mean"), 3.9)

    def test_read_best_missing_is_none(self):
        with TemporaryDirectory() as d:
            self.assertIsNone(SelectionPolicy.read_best(Path(d)))


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
class TestTracker(unittest.TestCase):
    def test_old_format_csv_read_and_append(self):
        with TemporaryDirectory() as d:
            out = Path(d)
            legacy = out / "metrics.csv"
            legacy.write_text(
                "epoch,step,checkpoint,utmos_mean,utmos_std,utmos_min,utmos_max,"
                "spk_sim_mean,spk_sim_std,spk_sim_min,n_sentences\n"
                "1,10,a.ckpt,3.5,0.1,3,4,,,,5\n", encoding="utf-8")
            tr = MetricsTracker(out)
            self.assertEqual(tr.done_epochs(), {1})
            # appending uses the file's OWN (legacy) header, staying well-formed
            tr.append(_row(2, 3.8).to_csv_row())
            with open(legacy, newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["epoch"], "2")
            # legacy file has no spk_sim_max column; row's extra key is dropped
            self.assertNotIn("spk_sim_max", rows[0])

    def test_fresh_file_gets_superset_header(self):
        with TemporaryDirectory() as d:
            tr = MetricsTracker(Path(d))
            tr.append(_row(1, 3.0, spk_sim=0.8).to_csv_row())
            header = (Path(d) / "metrics.csv").read_text().splitlines()[0]
            self.assertEqual(header.split(","), SUPERSET_HEADER)

    def test_failed_marking_stops_after_three(self):
        with TemporaryDirectory() as d:
            tr = MetricsTracker(Path(d), max_failures=3)
            self.assertFalse(tr.record_failure(7))
            self.assertFalse(tr.is_failed(7))
            self.assertFalse(tr.record_failure(7))
            self.assertTrue(tr.record_failure(7))  # third time -> failed
            self.assertTrue(tr.is_failed(7))
            self.assertIn(7, tr.failed_epochs())
            self.assertIn(7, tr.skip_epochs())

    def test_patience_and_stop_flag(self):
        with TemporaryDirectory() as d:
            out = Path(d)
            tr = MetricsTracker(out)
            for e in (1, 2, 3, 4):
                tr.append(_row(e, 3.0).to_csv_row())
            # best is epoch 1; epochs 2,3,4 are newer with no improvement
            self.assertEqual(tr.epochs_since_improvement(1), 3)
            # patience 2 exceeded -> stop.flag written with a reason
            self.assertTrue(tr.maybe_stop(best_epoch=1, patience=2))
            flag = out / "stop.flag"
            self.assertTrue(flag.exists())
            self.assertIn("patience", flag.read_text())
            # patience 5 not yet exceeded
            tr2 = MetricsTracker(Path(d) / "e2")
            for e in (1, 2, 3, 4):
                tr2.append(_row(e, 3.0).to_csv_row())
            self.assertFalse(tr2.maybe_stop(best_epoch=1, patience=5))
            self.assertFalse((Path(d) / "e2" / "stop.flag").exists())

    def test_patience_no_best_is_zero(self):
        with TemporaryDirectory() as d:
            tr = MetricsTracker(Path(d))
            self.assertEqual(tr.epochs_since_improvement(None), 0)
            self.assertFalse(tr.maybe_stop(None, patience=1))


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
class TestStopFileCallback(unittest.TestCase):
    def test_stops_when_flag_present(self):
        from phoonnx_train.evaluation.callbacks import StopFileCallback

        with TemporaryDirectory() as d:
            flag = Path(d) / "stop.flag"
            cb = StopFileCallback(flag)
            trainer = MagicMock()
            trainer.should_stop = False
            cb.on_train_epoch_end(trainer, MagicMock())
            self.assertFalse(trainer.should_stop)  # no flag yet
            flag.write_text("early stopping: patience\n")
            cb.on_train_epoch_end(trainer, MagicMock())
            self.assertTrue(trainer.should_stop)


class TestEvalScoreboardCallback(unittest.TestCase):
    def _callback(self, out, scorer, patience=None):
        from phoonnx_train.evaluation.callbacks import EvalScoreboardCallback

        tracker = MetricsTracker(out)
        selection = SelectionPolicy(metric="utmos_mean")
        ckpt_dir = out / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "epoch=0-step=1.ckpt").write_bytes(b"w")
        cb = EvalScoreboardCallback(scorer, tracker, selection, out,
                                    every_n_epochs=1, patience=patience,
                                    checkpoint_dir=ckpt_dir)
        return cb, tracker

    def _trainer(self, epoch=0):
        trainer = MagicMock()
        trainer.current_epoch = epoch
        trainer.should_stop = False
        return trainer

    def test_scorer_failure_does_not_crash_training(self):
        with TemporaryDirectory() as d:
            out = Path(d)
            scorer = MagicMock()
            scorer.score.side_effect = RuntimeError("boom")
            with patch("phoonnx_train.evaluation.callbacks.size_stable",
                       return_value=True):
                cb, tracker = self._callback(out, scorer)
                trainer = self._trainer()
                # must not raise
                cb.on_train_epoch_end(trainer, MagicMock())
            self.assertFalse(trainer.should_stop)
            # the failure was recorded (attempt 1)
            self.assertEqual(tracker._load_failed().get("0"), 1)

    def test_successful_score_updates_scoreboard(self):
        with TemporaryDirectory() as d:
            out = Path(d)
            scorer = MagicMock()
            scorer.score.return_value = _row(0, utmos=3.5)
            with patch("phoonnx_train.evaluation.callbacks.size_stable",
                       return_value=True):
                cb, tracker = self._callback(out, scorer)
                cb.on_train_epoch_end(self._trainer(), MagicMock())
            self.assertEqual(tracker.done_epochs(), {0})
            self.assertTrue((out / "best.json").exists())


# ---------------------------------------------------------------------------
# Scorer determinism (per-utterance reseed)
# ---------------------------------------------------------------------------
class TestScorerDeterminism(unittest.TestCase):
    def _scorer(self, out_capture):
        """Build a CheckpointScorer with encoder + engine faked out.

        The fake synth's output depends only on the torch RNG state, so if the
        per-utterance reseed works two runs produce identical wavs.
        """
        from phoonnx_train.evaluation import scorer as scorer_mod

        config = {"num_speakers": 1, "audio": {"sample_rate": 22050}}
        engine = MagicMock()

        def make_synth(*a, **k):
            import torch

            def synth(ids, scales, sid):
                # deterministic function of RNG state only
                return torch.rand(1000).numpy().astype(np.float32)
            return synth

        engine.eval_synthesize.side_effect = make_synth

        with patch.object(scorer_mod.CheckpointScorer, "__init__",
                          lambda self, *a, **k: None):
            s = scorer_mod.CheckpointScorer(engine, config, ["a", "b", "c"])
        # fill the attributes __init__ would have set
        s.engine = engine
        s.config = config
        s.sentences = ["a", "b", "c"]
        s.metrics = ["utmos"]
        s.seed = 1234
        s.vocoder_path = None
        s.device = "cpu"
        s.sample_rate = 22050
        s.scales = [0.667, 1.0, 0.8]
        s.speaker_id = None
        s.emb = None
        s.ref = None
        s.ph = None
        s.tokenizer = None
        s.lang = ""
        return s, scorer_mod

    def test_same_checkpoint_twice_identical(self):
        with TemporaryDirectory() as d:
            s, scorer_mod = self._scorer(d)
            # fake text_to_ids and utmos metric (avoid heavy phonemizer/model)
            with patch.object(scorer_mod, "text_to_ids",
                              side_effect=lambda t, *a: ([t], [1, 2, 3])), \
                 patch.dict(scorer_mod._METRIC_REGISTRY,
                            {"utmos": lambda wav, sr, text: float(np.mean(wav))}):
                w1 = Path(d) / "w1"
                w2 = Path(d) / "w2"
                row1 = s.score(Path(d) / "epoch=0.ckpt", 0, work_dir=w1)
                row2 = s.score(Path(d) / "epoch=0.ckpt", 0, work_dir=w2)
            self.assertEqual(row1.aggregates, row2.aggregates)
            # wavs byte-identical too
            for i in range(3):
                self.assertEqual((w1 / f"utt{i:02d}.wav").read_bytes(),
                                 (w2 / f"utt{i:02d}.wav").read_bytes())


# ---------------------------------------------------------------------------
# eval_loop CLI
# ---------------------------------------------------------------------------
class TestEvalLoopCLI(unittest.TestCase):
    def _config(self, path):
        path.write_text(json.dumps({
            "num_speakers": 1,
            "audio": {"sample_rate": 22050},
            "phoneme_type": "espeak",
            "alphabet": "ipa",
            "phoneme_id_map": {"a": 1},
            "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
        }), encoding="utf-8")

    def test_once_scores_fake_checkpoint(self):
        import click.testing

        from phoonnx_train import eval_loop

        with TemporaryDirectory() as d:
            d = Path(d)
            train_dir = d / "train"
            (train_dir / "ck").mkdir(parents=True)
            (train_dir / "ck" / "epoch=0-step=1.ckpt").write_bytes(b"w")
            cfg = d / "config.json"
            self._config(cfg)
            sents = d / "s.txt"
            sents.write_text("hello world\n", encoding="utf-8")
            out = d / "eval"

            fake_engine = MagicMock()
            fake_engine.eval_synthesize.return_value = (
                lambda ids, scales, sid: np.ones(2205, dtype=np.float32))

            with patch("phoonnx_train.eval_loop.get_engine",
                       return_value=fake_engine), \
                 patch("phoonnx_train.eval_loop.size_stable", return_value=True), \
                 patch("phoonnx_train.evaluation.scorer.build_encoder",
                       return_value=(None, None, "", 22050, [0.667, 1.0, 0.8])), \
                 patch("phoonnx_train.evaluation.scorer.text_to_ids",
                       return_value=(["h"], [1, 2, 3])), \
                 patch.dict("phoonnx_train.evaluation.scorer._METRIC_REGISTRY",
                            {"utmos": lambda wav, sr, text: 3.7}):
                runner = click.testing.CliRunner()
                result = runner.invoke(eval_loop.main, [
                    "--train-dir", str(train_dir),
                    "--config", str(cfg),
                    "--sentences", str(sents),
                    "--output-dir", str(out),
                    "--once",
                ])
            if result.exception:
                raise result.exception
            self.assertEqual(result.exit_code, 0)
            tr = MetricsTracker(out)
            self.assertEqual(tr.done_epochs(), {0})
            self.assertTrue((out / "best.json").exists())

    def test_once_emits_stop_flag_on_patience(self):
        import click.testing

        from phoonnx_train import eval_loop

        with TemporaryDirectory() as d:
            d = Path(d)
            train_dir = d / "train"
            ck = train_dir / "ck"
            ck.mkdir(parents=True)
            for e in range(3):
                (ck / f"epoch={e}-step={e}.ckpt").write_bytes(b"w")
            cfg = d / "config.json"
            self._config(cfg)
            sents = d / "s.txt"
            sents.write_text("hello\n", encoding="utf-8")
            out = d / "eval"

            fake_engine = MagicMock()
            fake_engine.eval_synthesize.return_value = (
                lambda ids, scales, sid: np.ones(2205, dtype=np.float32))

            # epoch 0 best (utmos 4), later epochs worse -> no improvement
            def utmos(wav, sr, text):
                return 4.0 if utmos.calls.pop(0) == 0 else 3.0
            utmos.calls = [0, 1, 2]

            with patch("phoonnx_train.eval_loop.get_engine",
                       return_value=fake_engine), \
                 patch("phoonnx_train.eval_loop.size_stable", return_value=True), \
                 patch("phoonnx_train.evaluation.scorer.build_encoder",
                       return_value=(None, None, "", 22050, [0.667, 1.0, 0.8])), \
                 patch("phoonnx_train.evaluation.scorer.text_to_ids",
                       return_value=(["h"], [1, 2, 3])), \
                 patch.dict("phoonnx_train.evaluation.scorer._METRIC_REGISTRY",
                            {"utmos": utmos}):
                runner = click.testing.CliRunner()
                result = runner.invoke(eval_loop.main, [
                    "--train-dir", str(train_dir),
                    "--config", str(cfg),
                    "--sentences", str(sents),
                    "--output-dir", str(out),
                    "--once", "--early-stop-patience", "2",
                ])
            if result.exception:
                raise result.exception
            self.assertEqual(result.exit_code, 0)
            self.assertTrue((train_dir / "stop.flag").exists())


if __name__ == "__main__":
    unittest.main()
