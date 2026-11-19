"""Sampling helpers shared by the codec-LM engines.

The samplers themselves stay in each engine: each one reproduces the order of its
upstream runtime (llama.cpp, vLLM, HuggingFace), and those orders select different
candidates.
"""
import numpy as np

#: Candidates sorted first when no top-k bounds the nucleus.
NUCLEUS_HEAD = 2048


def ranked(key: np.ndarray, size: int, reach: float | None = None) -> np.ndarray:
    """``np.argsort(key)[::-1]``, with only the ``size`` largest keys in order.

    The other indices follow unsorted. Falls back to the full sort on a tie among the
    keys a caller reads, or when the running sum of the head never reaches ``reach``.
    A caller reads the ``size`` largest keys, or with ``reach`` those up to the first
    whose running sum reaches it, plus the next.
    """
    if 0 < size < key.shape[0]:
        # a row filled with one low value (-inf, 0.0) is slow to partition, so leave it out
        above = key[key > key.min()]
        if size <= above.size:
            top = key >= np.partition(above, -size)[-size]
            head = np.flatnonzero(top)
            best = key[head]
            order = np.argsort(best)[::-1]
            head, best = head[order], best[order]
            # cumsum, not sum: callers cut on np.cumsum's rounding
            cut = best.size - 1 if reach is None else int(np.searchsorted(np.cumsum(best), reach))
            read = best[:cut + 2]
            if cut < best.size and np.all(read[:-1] > read[1:]):
                return np.concatenate([head, np.flatnonzero(~top)])
    return np.argsort(key)[::-1]


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def apply_repetition_penalty(scores: np.ndarray, seen: np.ndarray, penalty: float) -> np.ndarray:
    """Divide the score of tokens in ``seen`` by ``penalty`` (multiply when negative).

    >1 discourages repeats. ``seen`` is the recent-token window the caller chose.
    """
    if penalty == 1.0 or seen.size == 0:
        return scores
    out = scores.copy()
    idx = seen[(seen >= 0) & (seen < out.shape[0])]
    if idx.size:
        s = out[idx]
        out[idx] = np.where(s < 0, s * penalty, s / penalty)
    return out
