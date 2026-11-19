"""The samplers sort only their top candidates and still draw the tokens a full sort gives.

The oracle is each sampler with ``ranked`` replaced by the full sort, which is what the
samplers ran before ``ranked`` existed: the claim under test is that ``ranked`` never
changes a draw.
"""
from unittest import mock

import numpy as np
import pytest

from phoonnx.engines import llasa, neutts, orpheus, outetts, qwen3tts, sparktts
from phoonnx.engines._sampling import NUCLEUS_HEAD, apply_repetition_penalty, ranked

FULL_VOCAB = 151_936
KINDS = ("normal", "ties", "neg_inf", "peaked", "flat", "top_ties", "pair", "constant")


def full_sort(key, size, reach=None):
    return np.argsort(key)[::-1]


def logits(rng: np.random.Generator, size: int, kind: str) -> np.ndarray:
    """One float32 logit row of a given shape of distribution."""
    x = rng.normal(0.0, 3.0, size).astype(np.float32)
    if kind == "ties":
        x = np.round(x).astype(np.float32)
    elif kind == "neg_inf":
        x[rng.random(size) < 0.6] = -np.inf
        x[rng.integers(size)] = rng.normal()
    elif kind == "peaked":
        x[rng.integers(size)] = 40.0
    elif kind == "flat":
        x = (x * 1e-3).astype(np.float32)
    elif kind == "top_ties":
        best = np.argsort(x)[-60:]
        x[best[::2]] = x[best[-1]]
    elif kind == "pair" and size > 1:
        # one tie at a random place near the top, so some rows tie exactly at the cut
        best = np.argsort(x)[::-1][:60]
        place = rng.integers(best.size - 1)
        x[best[place + 1]] = x[best[place]]
    elif kind == "constant":
        x[:] = 1.5
    return x


def cases(seed: int, head: int, count: int = 400):
    """``count`` logit rows over the sizes around ``head`` and one full vocabulary."""
    rng = np.random.default_rng(seed)
    sizes = [1, head - 1, head, head + 1, 3 * head]
    for i in range(count):
        size = sizes[i % len(sizes)] if i % 20 else FULL_VOCAB
        yield i, logits(rng, max(size, 1), KINDS[i % len(KINDS)])


SAMPLERS = [
    # (name, module, sampler, settings without rng, candidates, has a cut)
    ("outetts", outetts, outetts._sample, (0.4, 40, 0.9, 0.05), 40, True),
    ("outetts-no-min-p", outetts, outetts._sample, (1.0, 40, 0.9, 0.0), 40, True),
    ("outetts-top-p-only", outetts, outetts._sample, (1.0, 0, 0.9, 0.05), 40, False),
    ("neutts", neutts, neutts._sample, (1.0, 50, 0.8), 50, True),
    ("neutts-hot", neutts, neutts._sample, (3.0, 50, 0.99), 50, True),
    ("sparktts", sparktts, sparktts.sample_token, (0.8, 50, 0.95), 50, True),
    ("sparktts-top-p-1", sparktts, sparktts.sample_token, (0.8, 50, 1.0), 50, True),
    ("sparktts-no-top-k", sparktts, sparktts.sample_token, (0.8, 0, 0.95), 50, False),
    ("orpheus", orpheus, orpheus._sample, (0.6, 0.8), NUCLEUS_HEAD, True),
    ("orpheus-hot", orpheus, orpheus._sample, (2.0, 0.99), NUCLEUS_HEAD, True),
    ("llasa", llasa, llasa._sample, (0.8, 0.95), NUCLEUS_HEAD, True),
    ("llasa-cold", llasa, llasa._sample, (0.3, 0.5), NUCLEUS_HEAD, True),
]


@pytest.mark.parametrize("name, module, sample, settings, head, cut", SAMPLERS,
                         ids=[s[0] for s in SAMPLERS])
def test_sampler_draws_the_same_token_as_a_full_sort(name, module, sample, settings, head, cut):
    real_argsort = np.argsort
    fast = fallback = 0
    for i, row in cases(sum(map(ord, name)), head):
        sizes = []

        def spy(a, *args, **kwargs):
            sizes.append(np.shape(a)[0])
            return real_argsort(a, *args, **kwargs)

        rng, rng_full = np.random.default_rng(i), np.random.default_rng(i)
        with mock.patch("numpy.argsort", spy):
            token = sample(row, *settings, rng)
        with mock.patch.object(module, "ranked", full_sort):
            assert token == sample(row, *settings, rng_full), (name, i, row.size)
        assert rng.bit_generator.state == rng_full.bit_generator.state, (name, i)
        if row.size > head and sizes:
            fast += max(sizes) < row.size
            fallback += max(sizes) == row.size
    print(f"{name}: fast {fast}, full sort {fallback}")
    if cut:
        assert fast and fallback, (fast, fallback)


@pytest.mark.parametrize("name, sample, settings, head", [
    ("outetts", outetts._sample, (0.4, 40, 0.9, 0.05), 40),
    ("neutts", neutts._sample, (1.0, 50, 0.8), 50),
    ("sparktts", sparktts.sample_token, (0.8, 50, 0.95), 50),
    ("orpheus", orpheus._sample, (0.6, 0.8), NUCLEUS_HEAD),
    ("llasa", llasa._sample, (0.8, 0.95), NUCLEUS_HEAD),
])
@pytest.mark.parametrize("seed", range(10))
def test_sampler_sorts_only_its_candidates(name, sample, settings, head, seed):
    # float64: a float32 row often ties inside llasa's 2,048 candidates, and a tie
    # takes the full sort that the differential test covers
    row = np.random.default_rng(seed).normal(0.0, 4.0, FULL_VOCAB)
    sizes = []
    real_argsort, real_sort = np.argsort, np.sort

    def spy(real):
        def call(a, *args, **kwargs):
            sizes.append(np.shape(a)[0])
            return real(a, *args, **kwargs)
        return call

    with mock.patch("numpy.argsort", spy(real_argsort)), mock.patch("numpy.sort", spy(real_sort)):
        sample(row, *settings, np.random.default_rng(seed))
    assert sizes and max(sizes) <= head, sizes


def test_ranked_head_matches_a_full_sort():
    rng = np.random.default_rng(3)
    for size in (1, 5, 40, 999):
        key = rng.permutation(1000).astype(np.float64)
        order = ranked(key, size)
        assert sorted(order.tolist()) == list(range(1000))
        np.testing.assert_array_equal(order[:size], np.argsort(key)[::-1][:size])


def test_ranked_sorts_everything_on_a_tie_or_a_short_reach():
    key = np.array([3.0, 1.0, 3.0, 0.5, 2.0])
    np.testing.assert_array_equal(ranked(key, 2), np.argsort(key)[::-1])
    probs = np.array([0.4, 0.1, 0.3, 0.2])
    assert ranked(probs, 2, reach=0.8).tolist() == [0, 2, 3, 1]
    assert ranked(probs, 2, reach=0.7).tolist() == [0, 2, 1, 3]


def test_ranked_with_reach_ignores_ties_past_the_cut():
    probs = np.array([0.5, 0.05, 0.3, 0.05, 0.1, 0.0])
    assert ranked(probs, 5, reach=0.7)[:3].tolist() == [0, 2, 4]
    np.testing.assert_array_equal(ranked(probs, 5, reach=0.95), np.argsort(probs)[::-1])


def test_ranked_sorts_everything_on_a_tie_just_after_the_cut():
    probs = np.array([0.4, 0.2, 0.2, 0.1, 0.05, 0.05])
    sizes = []
    real = np.argsort

    def spy(a, *args, **kwargs):
        sizes.append(np.shape(a)[0])
        return real(a, *args, **kwargs)

    with mock.patch("numpy.argsort", spy):
        ranked(probs, 4, 0.5)
    assert max(sizes) == probs.size, sizes


def test_repetition_penalties_ignore_repeated_ids():
    rng = np.random.default_rng(11)
    for _ in range(300):
        size = int(rng.integers(1, 5000))
        scores = rng.normal(0.0, 3.0, size).astype(np.float32)
        scores[rng.random(size) < 0.05] = 0.0
        seen = rng.integers(-3, size + 3, int(rng.integers(0, 300)))
        penalty = float(rng.choice([1.0, 1.1, 1.3, 0.9]))
        for penalize in (apply_repetition_penalty, outetts._apply_repetition_penalty):
            np.testing.assert_array_equal(penalize(scores, seen, penalty),
                                          penalize(scores, np.unique(seen), penalty))
        talker = rng.normal(0.0, 3.0, qwen3tts.TALKER_VOCAB).astype(np.float32)
        generated = rng.integers(0, qwen3tts.TALKER_VOCAB, int(rng.integers(1, 300)))
        step = int(rng.integers(0, 2 * qwen3tts.MIN_NEW_TOKENS + 1))
        np.testing.assert_array_equal(
            qwen3tts.apply_logits_processors(talker, generated.tolist(), penalty, step),
            qwen3tts.apply_logits_processors(talker, np.unique(generated).tolist(), penalty, step))
