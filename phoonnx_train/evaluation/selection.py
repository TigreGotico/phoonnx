"""Similarity-gated best-checkpoint selection.

UTMOS alone can prefer a checkpoint whose voice has drifted away from the
target speaker (higher naturalness, wrong identity). :class:`SelectionPolicy`
therefore *gates* candidates on a minimum speaker similarity before comparing
the naturalness metric: when speaker scoring is active and a floor is set, a
candidate must clear the floor to be eligible at all; among eligible candidates
the higher metric wins.

On a new best it writes ``best.json`` (epoch, step, checkpoint, all scores) and
copies the checkpoint to ``best.ckpt`` (a copy, not a symlink, so it survives
checkpoint pruning), plus keeps the best epoch's wav samples.
"""
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from phoonnx_train.evaluation.scorer import EvalRow

_LOGGER = logging.getLogger(__name__)


class SelectionPolicy:
    def __init__(self, metric: str = "utmos_mean", min_spk_sim: Optional[float] = None):
        self.metric = metric
        self.min_spk_sim = min_spk_sim

    def is_eligible(self, row: EvalRow) -> bool:
        """A row must clear the similarity floor when speaker scoring is active
        and a floor is set. With no speaker score or no floor, always eligible
        (UTMOS-only fallback)."""
        if self.min_spk_sim is None or not row.has_speaker_score:
            return True
        sim = row.spk_sim_mean
        return sim is not None and sim >= self.min_spk_sim

    def is_improvement(self, row: EvalRow, best: Optional[EvalRow]) -> bool:
        """True when ``row`` is a new best: eligible, and strictly higher on
        ``metric`` than ``best`` (or there is no incumbent)."""
        if not self.is_eligible(row):
            return False
        value = row.value(self.metric)
        if value is None:
            return False
        if best is None:
            return True
        best_value = best.value(self.metric)
        if best_value is None:
            return True
        return value > best_value

    def commit_best(self, row: EvalRow, output_dir: Path,
                    work_dir: Optional[Path] = None) -> None:
        """Persist ``row`` as the new best: best.json, best.ckpt and, when a
        work_dir is given, the best epoch's wav samples under samples/."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": row.epoch,
            "step": row.step,
            "checkpoint": row.checkpoint,
            "metric": self.metric,
            "min_spk_sim": self.min_spk_sim,
            "scores": row.aggregates,
        }
        self._atomic_write_json(output_dir / "best.json", payload)

        src = Path(row.checkpoint)
        if src.is_file():
            self._atomic_copy(src, output_dir / "best.ckpt")
        else:
            _LOGGER.error("best checkpoint %s missing; best.ckpt not updated", src)

        if work_dir is not None and Path(work_dir).is_dir():
            dest = output_dir / "samples" / f"epoch{row.epoch}"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(work_dir, dest)
            _LOGGER.info("kept best-epoch samples at %s", dest)

        _LOGGER.info(
            "new best: epoch %d %s=%.4f%s",
            row.epoch, self.metric, row.value(self.metric),
            "" if row.spk_sim_mean is None else f" spk_sim={row.spk_sim_mean:.4f}",
        )

    @staticmethod
    def read_best(output_dir: Path) -> Optional[EvalRow]:
        """Reconstruct the incumbent best EvalRow from best.json, or None."""
        best_json = Path(output_dir) / "best.json"
        if not best_json.is_file():
            return None
        try:
            data = json.loads(best_json.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            _LOGGER.warning("could not read %s; treating as no incumbent best", best_json)
            return None
        return EvalRow(
            epoch=int(data.get("epoch", -1)),
            step=int(data.get("step", -1)),
            checkpoint=str(data.get("checkpoint", "")),
            n_sentences=0,
            aggregates=dict(data.get("scores", {})),
        )

    @staticmethod
    def _atomic_write_json(path: Path, payload) -> None:
        tmp = Path(path).with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _atomic_copy(src: Path, dest: Path) -> None:
        tmp = Path(dest).with_suffix(dest.suffix + ".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
