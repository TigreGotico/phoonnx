"""The samplers sort only their top candidates and still draw the tokens they drew before.

The oracle for the differential tests is a verbatim copy of each sampler as it was when
it sorted the whole row. A copy of the old code is the right oracle here: the claim
under test is "same token for the same seed as before", and the old code is the only
independent definition of "before".
"""
import zlib
from unittest import mock

import numpy as np
import pytest

from phoonnx.engines import llasa, neutts, orpheus, outetts, qwen3tts, sparktts
from phoonnx.engines._sampling import NUCLEUS_HEAD, apply_repetition_penalty, ranked, softmax

FULL_VOCAB = 151_936


# ----------------------------------------------------------------------------------
# the samplers before the change, copied verbatim
# ----------------------------------------------------------------------------------
def outetts_before(scores, temperature, top_k, top_p, min_p, rng):
    s = np.asarray(scores, np.float32)
    if temperature <= 0:
        return int(np.argmax(s))
    s = s / float(temperature)
    if 0 < top_k < s.shape[0]:
        threshold = float(np.partition(s, -top_k)[-top_k])
        s = np.where(s < threshold, -np.inf, s)
    if 0 < top_p < 1:
        order = np.argsort(s)[::-1]
        cumulative = np.cumsum(softmax(s[order]))
        keep = np.ones(order.shape[0], bool)
        keep[1:] = cumulative[:-1] < top_p
        masked = np.full_like(s, -np.inf)
        masked[order[keep]] = s[order[keep]]
        s = masked
    if 0 < min_p < 1:
        probs = softmax(s)
        s = np.where(probs < min_p * probs.max(), -np.inf, s)
    return int(rng.choice(s.shape[0], p=softmax(s)))


def neutts_before(scores, temperature, top_k, top_p, rng):
    s = np.asarray(scores, np.float32)
    if temperature <= 0:
        return int(np.argmax(s))
    if 0 < top_k < s.shape[0]:
        threshold = float(np.partition(s, -top_k)[-top_k])
        s = np.where(s < threshold, -np.inf, s)
    if 0 < top_p < 1:
        order = np.argsort(s)[::-1]
        cumulative = np.cumsum(softmax(s[order]))
        keep = np.ones(order.shape[0], bool)
        keep[1:] = cumulative[:-1] < top_p
        masked = np.full_like(s, -np.inf)
        masked[order[keep]] = s[order[keep]]
        s = masked
    probs = softmax(s / float(temperature))
    return int(rng.choice(probs.shape[0], p=probs))


def sparktts_before(logits, temperature, top_k, top_p, rng):
    x = np.asarray(logits, np.float64).reshape(-1)
    if temperature > 0:
        x = x / max(temperature, 1e-5)
    else:
        return int(x.argmax())
    if top_k and 0 < top_k < x.size:
        x = np.where(x < np.partition(x, -top_k)[-top_k], -np.inf, x)
    x = x - x.max()
    probs = np.exp(x)
    probs /= probs.sum()
    order = np.argsort(probs)[::-1]
    cutoff = int(np.searchsorted(np.cumsum(probs[order]), top_p)) + 1
    keep = order[:max(1, cutoff)]
    p = probs[keep] / probs[keep].sum()
    return int(rng.choice(keep, p=p))


def orpheus_before(scores, temperature, top_p, rng):
    s = np.asarray(scores, np.float32)
    if temperature <= 0:
        return int(np.argmax(s))
    probs = softmax(s / float(temperature))
    if 0 < top_p < 1:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        keep = order[:max(1, int(np.searchsorted(cumulative, top_p)) + 1)]
        masked = np.zeros_like(probs)
        masked[keep] = probs[keep]
        probs = masked / masked.sum()
    return int(rng.choice(probs.shape[0], p=probs))


def llasa_before(scores, temperature, top_p, rng):
    s = np.asarray(scores, np.float64)
    if temperature <= 0:
        return int(np.argmax(s))
    probs = softmax(s / float(temperature))
    if 0 < top_p < 1:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        keep = np.ones(order.shape[0], bool)
        keep[1:] = cumulative[:-1] < top_p
        kept = order[keep]
        p = probs[kept] / probs[kept].sum()
        return int(rng.choice(kept, p=p))
    return int(rng.choice(probs.shape[0], p=probs))


def repetition_penalty_before(scores, seen, penalty):
    if penalty == 1.0 or seen.size == 0:
        return scores
    out = scores.copy()
    idx = np.unique(seen[(seen >= 0) & (seen < out.shape[0])])
    if idx.size:
        s = out[idx]
        out[idx] = np.where(s < 0, s * penalty, s / penalty)
    return out


def outetts_repetition_penalty_before(scores, seen, penalty):
    if penalty == 1.0 or seen.size == 0:
        return scores
    out = scores.copy()
    idx = np.unique(seen[(seen >= 0) & (seen < out.shape[0])])
    if idx.size:
        s = out[idx]
        out[idx] = np.where(s <= 0, s * penalty, s / penalty)
    return out


def qwen3_processors_before(logits, generated, repetition_penalty, step):
    x = np.asarray(logits, np.float64).reshape(-1).copy()
    if repetition_penalty != 1.0 and generated:
        seen = np.unique(np.asarray(generated, np.int64))
        scores = x[seen]
        x[seen] = np.where(scores < 0, scores * repetition_penalty,
                           scores / repetition_penalty)
    eos = x[qwen3tts.CODEC_EOS]
    x[qwen3tts.SUPPRESS_FROM:] = -np.inf
    x[qwen3tts.CODEC_EOS] = -np.inf if step < qwen3tts.MIN_NEW_TOKENS else eos
    return x


# ----------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------
KINDS = ("normal", "ties", "neg_inf", "peaked", "flat", "top_ties", "constant")


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
    elif kind == "constant":
        x[:] = 1.5
    return x


def cases(seed: int, head: int, count: int = 2000):
    """``count`` logit rows over the sizes around ``head`` and one full vocabulary."""
    rng = np.random.default_rng(seed)
    sizes = [1, head - 1, head, head + 1, 3 * head, FULL_VOCAB]
    for i in range(count):
        size = sizes[i % len(sizes)] if i % 10 else FULL_VOCAB
        yield i, logits(rng, max(size, 1), KINDS[i % len(KINDS)])


SAMPLERS = [
    # (name, new, before, settings without rng, candidates sorted when it can)
    ("outetts", outetts._sample, outetts_before, (0.4, 40, 0.9, 0.05), 40),
    ("outetts-no-min-p", outetts._sample, outetts_before, (1.0, 40, 0.9, 0.0), 40),
    ("outetts-top-p-only", outetts._sample, outetts_before, (1.0, 0, 0.9, 0.05), 40),
    ("outetts-top-k-only", outetts._sample, outetts_before, (1.0, 40, 1.0, 0.0), 40),
    ("neutts", neutts._sample, neutts_before, (1.0, 50, 0.8), 50),
    ("neutts-hot", neutts._sample, neutts_before, (3.0, 50, 0.99), 50),
    ("sparktts", sparktts.sample_token, sparktts_before, (0.8, 50, 0.95), 50),
    ("sparktts-top-p-1", sparktts.sample_token, sparktts_before, (0.8, 50, 1.0), 50),
    ("sparktts-no-top-k", sparktts.sample_token, sparktts_before, (0.8, 0, 0.95), 50),
    ("orpheus", orpheus._sample, orpheus_before, (0.6, 0.8), NUCLEUS_HEAD),
    ("orpheus-hot", orpheus._sample, orpheus_before, (2.0, 0.99), NUCLEUS_HEAD),
    ("llasa", llasa._sample, llasa_before, (0.8, 0.95), NUCLEUS_HEAD),
    ("llasa-cold", llasa._sample, llasa_before, (0.3, 0.5), NUCLEUS_HEAD),
]


@pytest.mark.parametrize("name, new, before, settings, head", SAMPLERS,
                         ids=[s[0] for s in SAMPLERS])
def test_sampler_draws_the_same_token_as_before(name, new, before, settings, head):
    for i, row in cases(zlib.crc32(name.encode()), head):
        rng_new, rng_before = np.random.default_rng(i), np.random.default_rng(i)
        token = new(row, *settings, rng_new)
        assert token == before(row, *settings, rng_before), (name, i, row.size)
        assert rng_new.bit_generator.state == rng_before.bit_generator.state, (name, i)


@pytest.mark.parametrize("name, new, settings, head", [
    ("outetts", outetts._sample, (0.4, 40, 0.9, 0.05), 40),
    ("neutts", neutts._sample, (1.0, 50, 0.8), 50),
    ("sparktts", sparktts.sample_token, (0.8, 50, 0.95), 50),
    ("orpheus", orpheus._sample, (0.6, 0.8), NUCLEUS_HEAD),
    ("llasa", llasa._sample, (0.8, 0.95), NUCLEUS_HEAD),
])
@pytest.mark.parametrize("seed", range(10))
def test_sampler_sorts_only_its_candidates(name, new, settings, head, seed):
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
        new(row, *settings, np.random.default_rng(seed))
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


def test_repetition_penalties_match_before():
    rng = np.random.default_rng(11)
    for i in range(2000):
        size = int(rng.integers(1, 5000))
        scores = rng.normal(0.0, 3.0, size).astype(np.float32)
        scores[rng.random(size) < 0.05] = 0.0
        seen = rng.integers(-3, size + 3, int(rng.integers(0, 300)))
        penalty = float(rng.choice([1.0, 1.1, 1.3, 0.9]))
        np.testing.assert_array_equal(apply_repetition_penalty(scores, seen, penalty),
                                      repetition_penalty_before(scores, seen, penalty))
        np.testing.assert_array_equal(outetts._apply_repetition_penalty(scores, seen, penalty),
                                      outetts_repetition_penalty_before(scores, seen, penalty))
        talker = rng.normal(0.0, 3.0, qwen3tts.TALKER_VOCAB).astype(np.float32)
        generated = rng.integers(0, qwen3tts.TALKER_VOCAB, int(rng.integers(0, 300))).tolist()
        step = int(rng.integers(0, 2 * qwen3tts.MIN_NEW_TOKENS + 1))
        np.testing.assert_array_equal(
            qwen3tts.apply_logits_processors(talker, generated, penalty, step),
            qwen3_processors_before(talker, generated, penalty, step))
