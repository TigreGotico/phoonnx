"""Lightning callbacks bridging the evaluation package into training.

* :class:`StopFileCallback` — watches a ``stop.flag`` file at each epoch end and
  stops the trainer when it appears. This is the **sidecar -> trainer bridge**:
  an out-of-process scoreboard writes the flag, the trainer honours it. It is a
  no-op while no flag exists, so it is always safe to add.

* :class:`EvalScoreboardCallback` — runs the same :class:`CheckpointScorer`
  in-process every N epochs (after ModelCheckpoint has saved), updates the shared
  tracker/selection files and stops the trainer once patience is exhausted.
  Scoring failures never crash training: they are logged and swallowed.
"""
import logging
from pathlib import Path
from typing import Optional

try:  # pytorch_lightning is a heavy optional dep at import time
    from pytorch_lightning.callbacks import Callback
except Exception:  # pragma: no cover - exercised indirectly
    Callback = object

from phoonnx_train.eval_utils import find_checkpoints, size_stable
from phoonnx_train.evaluation.scorer import write_epoch_perutt
from phoonnx_train.evaluation.selection import SelectionPolicy
from phoonnx_train.evaluation.tracker import MetricsTracker

_LOGGER = logging.getLogger(__name__)


class StopFileCallback(Callback):
    """Stop training when ``flag_path`` appears (external early-stop signal)."""

    def __init__(self, flag_path: Path):
        super().__init__()
        self.flag_path = Path(flag_path)

    def _check(self, trainer) -> None:
        if self.flag_path.exists():
            try:
                reason = self.flag_path.read_text(encoding="utf-8").strip()
            except OSError:
                reason = "(unreadable stop.flag)"
            _LOGGER.warning("stop flag %s present: %s; stopping training",
                            self.flag_path, reason)
            trainer.should_stop = True

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._check(trainer)

    def on_validation_end(self, trainer, pl_module) -> None:
        self._check(trainer)


class EvalScoreboardCallback(Callback):
    """In-process scoreboard + patience-based early stopping.

    Args:
        scorer: a configured :class:`CheckpointScorer`.
        tracker: shared :class:`MetricsTracker` (same output_dir as any sidecar).
        selection: :class:`SelectionPolicy`.
        output_dir: scoreboard dir (metrics.csv, best.ckpt, samples/, stop.flag).
        every_n_epochs: score every N epochs (>=1).
        patience: stop after this many scored epochs without improvement
            (``None``/0 disables stopping; scoring still runs).
        checkpoint_dir: where to look for ``epoch=*.ckpt`` (defaults to the
            trainer's checkpoint dir / default_root_dir).
    """

    def __init__(self, scorer, tracker: MetricsTracker, selection: SelectionPolicy,
                 output_dir: Path, every_n_epochs: int = 1,
                 patience: Optional[int] = None, checkpoint_dir: Optional[Path] = None):
        super().__init__()
        self.scorer = scorer
        self.tracker = tracker
        self.selection = selection
        self.output_dir = Path(output_dir)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        try:
            epoch = int(trainer.current_epoch)
            if (epoch + 1) % self.every_n_epochs != 0:
                return
            self._score_newest(trainer)
        except Exception:
            # Scoring must never crash training.
            _LOGGER.exception("EvalScoreboardCallback failed; continuing training")

    def _resolve_checkpoint_dir(self, trainer) -> Optional[Path]:
        if self.checkpoint_dir is not None:
            return self.checkpoint_dir
        cb = getattr(trainer, "checkpoint_callback", None)
        if cb is not None and getattr(cb, "dirpath", None):
            return Path(cb.dirpath)
        for attr in ("log_dir", "default_root_dir"):
            val = getattr(trainer, attr, None)
            if val:
                return Path(val)
        return None

    def _score_newest(self, trainer) -> None:
        ckpt_dir = self._resolve_checkpoint_dir(trainer)
        if ckpt_dir is None or not Path(ckpt_dir).exists():
            _LOGGER.warning("no checkpoint dir resolved yet; skipping scoreboard")
            return
        ckpts = find_checkpoints(ckpt_dir)
        skip = self.tracker.skip_epochs()
        todo = sorted(e for e in ckpts if e not in skip)
        if not todo:
            return
        epoch = todo[-1]
        ckpt = ckpts[epoch]
        if not size_stable(ckpt):
            _LOGGER.warning("checkpoint %s not stable yet; deferring", ckpt)
            return

        work_dir = self.output_dir / "_work" / f"epoch{epoch}"
        try:
            row = self.scorer.score(ckpt, epoch, work_dir=work_dir)
        except Exception:
            _LOGGER.exception("scoring epoch %d failed", epoch)
            self.tracker.record_failure(epoch)
            return

        best = self.selection.read_best(self.output_dir)
        self.tracker.append(row.to_csv_row())
        # Per-epoch per-utterance file for every scored epoch (overfit
        # diagnosis across epochs, not only the best epoch's samples).
        write_epoch_perutt(self.output_dir, row, self.scorer.metrics)
        if self.selection.is_improvement(row, best):
            self.selection.commit_best(row, self.output_dir, work_dir=work_dir)
            best = row

        best_epoch = best.epoch if best is not None else None
        if self.tracker.maybe_stop(best_epoch, self.patience):
            _LOGGER.warning("patience exhausted; stopping training at epoch %d", epoch)
            trainer.should_stop = True
