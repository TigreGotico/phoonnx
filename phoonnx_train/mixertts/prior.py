"""Beta-binomial alignment prior for the Mixer-TTS aligner.

Torch/scipy-free (numpy + math.lgamma) so it can be unit-tested without
the training stack. Equivalent to NVIDIA NeMo's
``beta_binomial_prior_distribution`` (Apache-2.0,
nemo/collections/tts/torch/helpers.py), which computes the same
``scipy.stats.betabinom`` pmf: for mel frame ``i`` (1-based, of M), the
prior over the P phoneme positions is ``BetaBinom(P - 1, a=i,
b=M + 1 - i)`` — the "cigar along the diagonal" prior of the RAD-TTS /
one-TTS-alignment recipe (Badlani et al., 2021,
https://arxiv.org/abs/2108.10447; Mixer-TTS: Tatanov et al., 2021,
https://arxiv.org/abs/2110.03584).

NeMo interpolates cached priors of rounded sizes for speed
(``BetaBinomialInterpolator``); here the exact prior is computed and
LRU-cached per (mel_len, text_len) instead — exact, and free of the
scipy/ndimage dependency.
"""
from functools import lru_cache
from math import lgamma

import numpy as np


def _log_betabinom_pmf(k: np.ndarray, n: int, a: float, b: float) -> np.ndarray:
    """log pmf of BetaBinomial(n, a, b) at integer support ``k``."""
    # log C(n, k) + betaln(k + a, n - k + b) - betaln(a, b)
    lg = np.vectorize(lgamma)
    log_comb = lg(n + 1) - lg(k + 1) - lg(n - k + 1)
    betaln_num = lg(k + a) + lg(n - k + b) - lg(n + a + b)
    betaln_den = lgamma(a) + lgamma(b) - lgamma(a + b)
    return log_comb + betaln_num - betaln_den


def beta_binomial_prior_distribution(
    phoneme_count: int, mel_count: int, scaling: float = 1.0
) -> np.ndarray:
    """Return the [mel_count, phoneme_count] alignment prior matrix.

    Row ``i`` (0-based) is the BetaBinomial(P-1, scaling*(i+1),
    scaling*(M-i)) pmf over phoneme positions, matching NeMo/FastPitch's
    ``beta_binomial_prior_distribution`` exactly.
    """
    if phoneme_count < 1 or mel_count < 1:
        raise ValueError(
            f"phoneme_count and mel_count must be >= 1, "
            f"got {phoneme_count} and {mel_count}"
        )
    P, M = phoneme_count, mel_count
    x = np.arange(P)
    rows = []
    for i in range(1, M + 1):
        a, b = scaling * i, scaling * (M + 1 - i)
        rows.append(np.exp(_log_betabinom_pmf(x, P - 1, a, b)))
    return np.asarray(rows, dtype=np.float32)


@lru_cache(maxsize=256)
def _cached_prior(mel_count: int, phoneme_count: int) -> np.ndarray:
    return beta_binomial_prior_distribution(phoneme_count, mel_count)


class BetaBinomialPrior:
    """Callable returning the exact [mel_len, text_len] prior, LRU-cached.

    Drop-in for NeMo's ``BetaBinomialInterpolator`` minus the lossy
    zoom-interpolation approximation.
    """

    def __call__(self, mel_len: int, text_len: int) -> np.ndarray:
        return _cached_prior(int(mel_len), int(text_len))
