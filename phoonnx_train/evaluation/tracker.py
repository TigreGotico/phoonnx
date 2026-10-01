"""Scoreboard bookkeeping: metrics.csv, failed epochs, patience, stop.flag.

:class:`MetricsTracker` owns the durable state of an evaluation run:

* **metrics.csv** — a superset of the legacy header. Old files are read fine
  (missing columns come back empty) and appended to using their *own* existing
  header, so a run started against an old scoreboard keeps a valid CSV; a fresh
  file gets the superset header.
* **failed epochs** (``failed.json``) — a checkpoint that fails to load/score is
  retried at most ``max_failures`` times (default 3) and then recorded failed and
  skipped forever, with an ERROR log. No infinite retry loop.
* **patience** — ``epochs_since_improvement`` counts scored epochs newer than the
  current best (from best.json); ``write_stop_flag`` emits ``stop.flag`` with a
  reason line when the patience budget is exceeded.
"""
import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from phoonnx_train.eval_utils import done_epochs as _done_epochs

_LOGGER = logging.getLogger(__name__)

# Legacy header (VITS-only UTMOS scoreboard) plus spk_sim_max. Superset: any
# column not present in an old file is simply left empty on read.
SUPERSET_HEADER = [
    "epoch", "step", "checkpoint",
    "utmos_mean", "utmos_std", "utmos_min", "utmos_max",
    "spk_sim_mean", "spk_sim_std", "spk_sim_min", "spk_sim_max",
    "n_sentences",
]

STOP_FLAG = "stop.flag"


class MetricsTracker:
    def __init__(self, output_dir: Path, max_failures: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_csv = self.output_dir / "metrics.csv"
        self.failed_json = self.output_dir / "failed.json"
        self.max_failures = max_failures

    # ------------------------------------------------------------------
    # metrics.csv (backward compatible)
    # ------------------------------------------------------------------
    def done_epochs(self) -> Set[int]:
        return _done_epochs(self.metrics_csv)

    def read_rows(self) -> List[Dict[str, str]]:
        if not self.metrics_csv.exists():
            return []
        with open(self.metrics_csv, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _existing_header(self) -> Optional[List[str]]:
        if not self.metrics_csv.exists():
            return None
        with open(self.metrics_csv, newline="", encoding="utf-8") as f:
            try:
                header = next(csv.reader(f))
            except StopIteration:
                return None
        return header or None

    def append(self, row: Dict[str, object]) -> None:
        """Append one row. A new file gets ``SUPERSET_HEADER``; an existing file
        is appended to using its own header (extra keys dropped, missing keys
        written empty), so old-format scoreboards stay well-formed."""
        header = self._existing_header()
        new = header is None
        if new:
            header = SUPERSET_HEADER
        with open(self.metrics_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(header)
            w.writerow([row.get(k, "") for k in header])

    # ------------------------------------------------------------------
    # Failed-epoch marking (bounded retries)
    # ------------------------------------------------------------------
    def _load_failed(self) -> Dict[str, int]:
        if not self.failed_json.exists():
            return {}
        try:
            return dict(json.loads(self.failed_json.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            return {}

    def _save_failed(self, data: Dict[str, int]) -> None:
        tmp = self.failed_json.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, self.failed_json)

    def record_failure(self, epoch: int) -> bool:
        """Increment the failure count for ``epoch``. Returns True once the
        count reaches ``max_failures`` (epoch is now permanently failed)."""
        data = self._load_failed()
        count = int(data.get(str(epoch), 0)) + 1
        data[str(epoch)] = count
        self._save_failed(data)
        if count >= self.max_failures:
            _LOGGER.error(
                "epoch %d failed to score %d times; marking failed and "
                "skipping it permanently", epoch, count,
            )
            return True
        _LOGGER.warning("epoch %d failed to score (attempt %d/%d)",
                        epoch, count, self.max_failures)
        return False

    def is_failed(self, epoch: int) -> bool:
        return int(self._load_failed().get(str(epoch), 0)) >= self.max_failures

    def failed_epochs(self) -> Set[int]:
        return {int(e) for e, c in self._load_failed().items()
                if int(c) >= self.max_failures}

    def skip_epochs(self) -> Set[int]:
        """Epochs to never (re)score: already-done plus permanently failed."""
        return self.done_epochs() | self.failed_epochs()

    # ------------------------------------------------------------------
    # Patience / early stopping
    # ------------------------------------------------------------------
    def epochs_since_improvement(self, best_epoch: Optional[int]) -> int:
        """Number of scored epochs strictly newer than ``best_epoch``.

        With no best yet, returns 0.
        """
        if best_epoch is None:
            return 0
        return sum(1 for e in self.done_epochs() if e > best_epoch)

    def write_stop_flag(self, reason: str) -> Path:
        """Emit ``stop.flag`` with a reason line (idempotent overwrite)."""
        flag = self.output_dir / STOP_FLAG
        flag.write_text(reason.rstrip("\n") + "\n", encoding="utf-8")
        _LOGGER.warning("wrote %s: %s", flag, reason)
        return flag

    def maybe_stop(self, best_epoch: Optional[int], patience: Optional[int]) -> bool:
        """Write stop.flag when patience is set and exceeded. Returns whether a
        flag was written."""
        if not patience or patience <= 0:
            return False
        since = self.epochs_since_improvement(best_epoch)
        if since >= patience:
            self.write_stop_flag(
                f"early stopping: {since} epochs since improvement "
                f"(best epoch {best_epoch}) >= patience {patience}"
            )
            return True
        return False
