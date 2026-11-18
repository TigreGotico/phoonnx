"""Sampling helpers shared by the codec-LM engines.

The samplers themselves stay in each engine: each one reproduces the order of its
upstream runtime (llama.cpp, vLLM, HuggingFace), and those orders select different
candidates.
"""
import numpy as np


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
    idx = np.unique(seen[(seen >= 0) & (seen < out.shape[0])])
    if idx.size:
        s = out[idx]
        out[idx] = np.where(s < 0, s * penalty, s / penalty)
    return out
