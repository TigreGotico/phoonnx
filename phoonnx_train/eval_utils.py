"""Torch-free helpers for the checkpoint evaluation loop.

Checkpoint discovery, epoch/step parsing, metrics.csv bookkeeping and the
partial-write guard live here so they can be imported (and tested) without
torch installed.
"""
import csv
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

METRICS_HEADER = ["epoch", "step", "checkpoint", "utmos_mean", "utmos_std",
                  "utmos_min", "utmos_max", "spk_sim_mean", "spk_sim_std",
                  "spk_sim_min", "n_sentences"]

CKPT_RE = re.compile(r"epoch=(\d+)")
STEP_RE = re.compile(r"step=(\d+)")


def parse_epoch(name: str) -> Optional[int]:
    """Epoch number from a checkpoint filename, or None if absent."""
    m = CKPT_RE.search(name)
    return int(m.group(1)) if m else None


def parse_step(name: str) -> int:
    """Step number from a checkpoint filename, or -1 if absent."""
    m = STEP_RE.search(name)
    return int(m.group(1)) if m else -1


def find_checkpoints(train_dir: Path) -> Dict[int, Path]:
    """All epoch=*.ckpt under train_dir as {epoch: path}.

    When several files share an epoch number the newest (highest mtime) wins.
    """
    out: Dict[int, Path] = {}
    for p in Path(train_dir).rglob("epoch=*.ckpt"):
        ep = parse_epoch(p.name)
        if ep is None:
            continue
        if ep not in out or p.stat().st_mtime > out[ep].stat().st_mtime:
            out[ep] = p
    return out


def done_epochs(metrics_csv: Path) -> Set[int]:
    """Epochs already recorded in metrics.csv (malformed rows are skipped)."""
    metrics_csv = Path(metrics_csv)
    if not metrics_csv.exists():
        return set()
    done: Set[int] = set()
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                done.add(int(row["epoch"]))
            except (KeyError, TypeError, ValueError):
                pass
    return done


def best_mean_so_far(metrics_csv: Path) -> Optional[float]:
    """Highest utmos_mean recorded in metrics.csv, or None."""
    metrics_csv = Path(metrics_csv)
    if not metrics_csv.exists():
        return None
    best: Optional[float] = None
    with open(metrics_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                m = float(row["utmos_mean"])
            except (KeyError, TypeError, ValueError):
                continue
            if best is None or m > best:
                best = m
    return best


def append_metrics(metrics_csv: Path, row: Dict[str, object]) -> None:
    """Append one summary row, writing the header on first use."""
    metrics_csv = Path(metrics_csv)
    new = not metrics_csv.exists()
    with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(METRICS_HEADER)
        w.writerow([row.get(k, "") for k in METRICS_HEADER])


def read_sentences(path: Path) -> List[str]:
    """Non-empty stripped lines of a sentences file."""
    return [line.strip() for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def size_stable(path: Path, wait: float = 5.0, tries: int = 6,
                sleep=time.sleep) -> bool:
    """Wait until file size stops changing (guards against partial writes)."""
    path = Path(path)
    last = -1
    for _ in range(max(1, tries)):
        try:
            s = path.stat().st_size
        except FileNotFoundError:
            return False
        if s == last and s > 0:
            return True
        last = s
        sleep(wait)
    return False


def largest_wavs(ref_dir: Path, n: int) -> Sequence[Path]:
    """The n largest .wav files in ref_dir (longer clips give steadier
    speaker embeddings)."""
    return sorted(Path(ref_dir).glob("*.wav"),
                  key=lambda p: p.stat().st_size, reverse=True)[:n]
