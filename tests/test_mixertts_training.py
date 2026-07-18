"""Tests for the Mixer-TTS training engine (phoonnx_train.engines.mixer).

Torch is not part of the test environment: the engine registry, config
handling and quality presets are all importable without torch (heavy
imports are deferred until a model is actually built), and the
beta-binomial alignment-prior math lives in the torch/scipy-free
``phoonnx_train.mixertts.prior`` module, loaded here by file path so the
mixertts package models (which need torch) are never imported.
"""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from phoonnx_train.engines import get_engine, list_engines
from phoonnx_train.engines.base import TrainingEngineConfig
from phoonnx_train.engines.mixer import (
    _QUALITY_PRESETS,
    MixerTTSTrainingEngine,
    resolve_module_kwargs,
)

_PRIOR = Path(__file__).parent.parent / "phoonnx_train" / "mixertts" / "prior.py"
spec = importlib.util.spec_from_file_location("mixer_prior", _PRIOR)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


# ---------------------------------------------------------------- registry
def test_engine_registered_with_alias():
    assert "mixer" in list_engines()
    assert "mixertts" in list_engines()
    assert isinstance(get_engine("mixer"), MixerTTSTrainingEngine)
    assert isinstance(get_engine("mixertts"), MixerTTSTrainingEngine)


def test_quality_presets():
    presets = MixerTTSTrainingEngine().quality_presets()
    assert set(presets) == {"x-low", "medium", "high"}
    # every Mixer block width derives from the symbol embedding dim
    dims = [presets[t]["symbols_embedding_dim"] for t in ("x-low", "medium", "high")]
    assert dims == sorted(dims) and all(d > 0 for d in dims)
    assert presets["high"]["symbols_embedding_dim"] == 384  # paper config


# ----------------------------------------------------------- config handling
def test_kwargs_accept_train_cli_extra_bag():
    # train.py merges preset kwargs + batch_size/validation_split/num_workers
    # into extra — the engine must accept the full bag
    eng = get_engine("mixer")
    cfg = TrainingEngineConfig(
        num_symbols=90,
        num_speakers=3,
        sample_rate=16000,
        extra={**eng.quality_presets()["x-low"], "batch_size": 4,
               "validation_split": 0.1, "num_workers": 0},
    )
    kw = resolve_module_kwargs(cfg)
    assert kw["num_symbols"] == 90
    assert kw["num_speakers"] == 3
    assert kw["sample_rate"] == 16000
    assert kw["symbols_embedding_dim"] == _QUALITY_PRESETS["x-low"]["symbols_embedding_dim"]
    assert kw["batch_size"] == 4


def test_kwargs_do_not_mutate_presets():
    before = copy.deepcopy(_QUALITY_PRESETS)
    cfg = TrainingEngineConfig(num_symbols=90, extra={"quality": "x-low", "batch_size": 2})
    kw = resolve_module_kwargs(cfg)
    kw["symbols_embedding_dim"] = -1
    assert _QUALITY_PRESETS == before


def test_kwargs_unknown_quality_falls_back_to_medium():
    cfg = TrainingEngineConfig(num_symbols=90, extra={"quality": "nonsense"})
    kw = resolve_module_kwargs(cfg)
    assert kw["symbols_embedding_dim"] == _QUALITY_PRESETS["medium"]["symbols_embedding_dim"]


def test_kwargs_extra_overrides_preset():
    cfg = TrainingEngineConfig(
        num_symbols=90, extra={"quality": "medium", "symbols_embedding_dim": 64},
    )
    assert resolve_module_kwargs(cfg)["symbols_embedding_dim"] == 64


def test_kwargs_config_wins_over_extra_symbol_count():
    # num_symbols/num_speakers/sample_rate come from the shared config,
    # not from a stray extra key
    cfg = TrainingEngineConfig(num_symbols=90, extra={"num_symbols": 5})
    assert resolve_module_kwargs(cfg)["num_symbols"] == 90


def test_kwargs_gan_flag_passes_through():
    cfg = TrainingEngineConfig(num_symbols=90, extra={"train_gan": True})
    assert resolve_module_kwargs(cfg)["train_gan"] is True


# --------------------------------------------------------- extra_preprocess
def test_extra_preprocess_missing_deps_returns_empty(monkeypatch, tmp_path):
    """Shared with the FastPitch engine: without pyworld/librosa it
    degrades to {} instead of crashing preprocessing."""
    engine = MixerTTSTrainingEngine()
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name in ("pyworld", "librosa"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = engine.extra_preprocess(tmp_path / "a.wav", tmp_path / "cache", 22050)
    assert result == {}
    assert not (tmp_path / "cache").exists()  # nothing written


# ------------------------------------------------------------- export guard
def test_export_onnx_missing_config_raises(tmp_path):
    engine = MixerTTSTrainingEngine()
    (tmp_path / "model.ckpt").write_bytes(b"not a checkpoint")
    with pytest.raises((FileNotFoundError, OSError)):
        engine.export_onnx(tmp_path / "model.ckpt", tmp_path / "missing.json", tmp_path)


def test_export_onnx_missing_checkpoint_raises(tmp_path):
    engine = MixerTTSTrainingEngine()
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        engine.export_onnx(tmp_path / "nope.ckpt", tmp_path / "config.json", tmp_path)


def test_export_onnx_malformed_config_raises(tmp_path):
    engine = MixerTTSTrainingEngine()
    (tmp_path / "model.ckpt").write_bytes(b"x")
    (tmp_path / "config.json").write_text("{not json")
    import json as _json
    with pytest.raises(_json.JSONDecodeError):
        engine.export_onnx(tmp_path / "model.ckpt", tmp_path / "config.json", tmp_path)


# ------------------------------------------------------- beta-binomial prior
def test_prior_shape_and_rows_sum_to_one():
    p = prior.beta_binomial_prior_distribution(11, 37)
    assert p.shape == (37, 11)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-4)  # each row is a pmf
    assert (p >= 0).all()


def test_prior_is_diagonal_cigar():
    # the argmax phoneme index must be monotonically non-decreasing over
    # mel frames — first frames attend to first phonemes, last to last
    p = prior.beta_binomial_prior_distribution(23, 200)
    peaks = p.argmax(axis=1)
    assert peaks[0] == 0 and peaks[-1] == 22
    assert (np.diff(peaks) >= 0).all()


def test_prior_matches_betabinom_pmf_exactly():
    # spot-check against the closed-form BetaBinomial pmf for tiny numbers:
    # M=1 frame, P=2 phonemes -> BetaBinom(n=1, a=1, b=1) = uniform
    p = prior.beta_binomial_prior_distribution(2, 1)
    assert np.allclose(p, [[0.5, 0.5]], atol=1e-6)


def test_prior_single_phoneme_and_single_frame():
    # degenerate boundaries must not divide by zero or return empties
    assert prior.beta_binomial_prior_distribution(1, 5).shape == (5, 1)
    assert np.allclose(prior.beta_binomial_prior_distribution(1, 5), 1.0)
    assert prior.beta_binomial_prior_distribution(7, 1).shape == (1, 7)


@pytest.mark.parametrize("P,M", [(0, 10), (10, 0), (-1, 5), (5, -1)])
def test_prior_rejects_nonpositive_sizes(P, M):
    with pytest.raises(ValueError):
        prior.beta_binomial_prior_distribution(P, M)


def test_prior_large_sizes_stay_finite():
    # long utterances: lgamma-space math must not overflow to inf/nan
    p = prior.beta_binomial_prior_distribution(400, 2000)
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-3)


def test_prior_callable_cache_returns_same_array():
    bb = prior.BetaBinomialPrior()
    a = bb(50, 10)
    b = bb(50, 10)
    assert a.shape == (50, 10)
    assert a is b  # LRU-cached exact prior, no recompute
