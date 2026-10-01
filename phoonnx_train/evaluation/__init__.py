"""Reusable checkpoint-evaluation package.

The same building blocks drive two modes:

* a **sidecar scoreboard** process (``phoonnx_train.eval_loop``) that watches a
  training directory, scores new checkpoints on CPU and maintains a scoreboard
  (metrics.csv, per-utterance scores, best.json/best.ckpt, kept wavs), and

* **in-training early stopping** (``EvalScoreboardCallback``) that runs the same
  scorer in-process every N epochs and stops the trainer once a patience budget
  is exhausted.

Both modes share:

* :class:`~phoonnx_train.evaluation.scorer.CheckpointScorer` — turns a checkpoint
  into a structured :class:`~phoonnx_train.evaluation.scorer.EvalRow`,
* :class:`~phoonnx_train.evaluation.selection.SelectionPolicy` — similarity-gated
  best-checkpoint selection,
* :class:`~phoonnx_train.evaluation.tracker.MetricsTracker` — metrics.csv
  bookkeeping, failed-epoch marking, patience counting and stop.flag emission.
"""
from phoonnx_train.evaluation.scorer import (CheckpointScorer, EvalRow,
                                             register_metric)
from phoonnx_train.evaluation.selection import SelectionPolicy
from phoonnx_train.evaluation.tracker import MetricsTracker

__all__ = [
    "CheckpointScorer",
    "EvalRow",
    "register_metric",
    "SelectionPolicy",
    "MetricsTracker",
]
