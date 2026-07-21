"""Corpus F0 (pitch) statistics for FastPitch training.

Torch-free (numpy + json only) so it can be unit-tested without the
training stack installed. The pitch target is z-scored with these corpus
statistics so its scale matches the other loss terms and the
``pitch_mul``/``pitch_add`` inference controls operate on a normalized
quantity (raw Hz would dwarf every other loss).
"""
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from phoonnx_train.vendor.f0 import EXTRACTOR_TAG

_LOG = logging.getLogger(__name__)

# Keyed by extraction method for the same reason as f0_cache_path: corpus
# mean/std computed from one extractor's tracks must not normalize another's.
STATS_FILENAME = f"pitch_stats-{EXTRACTOR_TAG}.json"


def f0_cache_path(audio_spec_path: Path) -> Path:
    """``<utterance>.spec.pt`` -> sidecar ``<utterance>.f0-<method>.npy``
    cache. The extraction-method tag is folded into the filename so a
    cache written by a previous F0 extractor is a clean miss instead of
    being silently reused."""
    return Path(str(audio_spec_path)).with_suffix("").with_suffix(f".f0-{EXTRACTOR_TAG}.npy")


def load_or_compute_pitch_stats(
    dataset_paths: Iterable[Path],
    f0_paths: List[Path],
) -> Tuple[float, float]:
    """Return corpus (mean, std) over voiced F0 frames.

    Stats are cached as ``pitch_stats-<method>.json`` (see ``STATS_FILENAME``) in the first dataset
    directory; a missing or malformed cache is recomputed from the
    ``f0_cache_path`` sidecar files. With no pitch caches at all, identity
    normalization ``(0.0, 1.0)`` is returned.
    """
    import numpy as np

    stats_path: Optional[Path] = None
    for p in dataset_paths:
        p = Path(p)
        if p.is_dir():
            stats_path = p / STATS_FILENAME
            break
    if stats_path and stats_path.is_file():
        try:
            stats = json.loads(stats_path.read_text())
            mean, std = float(stats["mean"]), float(stats["std"])
            if std > 0:
                return mean, std
            _LOG.warning("ignoring cached %s with non-positive std", stats_path)
        except (ValueError, KeyError, TypeError, OSError):
            _LOG.warning("ignoring malformed %s — recomputing", stats_path)

    voiced = []
    for cand in f0_paths:
        if cand.exists():
            f0 = np.load(cand)
            voiced.append(f0[f0 > 0])
    if not voiced:
        return 0.0, 1.0  # no pitch caches — identity normalization
    allv = np.concatenate(voiced)
    mean = float(allv.mean()) if allv.size else 0.0
    std = float(allv.std()) if allv.size else 1.0
    std = std or 1.0
    if stats_path:
        try:
            stats_path.write_text(json.dumps({"mean": mean, "std": std}))
        except OSError:
            pass
    return mean, std
