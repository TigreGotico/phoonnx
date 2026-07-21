"""Tests for the FastPitch / SpeedySpeech training engine
(phoonnx_train.engines.fastpitch).

Torch is not part of the test environment: the engine registry, config
handling and quality presets are all importable without torch (heavy
imports are deferred until a model is actually built), and the pitch
(F0) statistics rules live in the torch-free
``phoonnx_train.fastpitch.pitch_stats`` module, loaded here by file path
so the fastpitch package ``__init__`` (which needs torch) is never
imported.
"""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from phoonnx_train.engines import get_engine, list_engines
from phoonnx_train.engines.base import TrainingEngineConfig
from phoonnx_train.engines.fastpitch import (
    _QUALITY_PRESETS,
    ForwardTTSTrainingEngine,
    SpeedySpeechTrainingEngine,
    resolve_module_kwargs,
)

_PITCH_STATS = (
    Path(__file__).parent.parent / "phoonnx_train" / "fastpitch" / "pitch_stats.py"
)
spec = importlib.util.spec_from_file_location("fp_pitch_stats", _PITCH_STATS)
pitch_stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pitch_stats)


# ---------------------------------------------------------------- registry
def test_engines_registered():
    assert "fastpitch" in list_engines()
    assert "speedyspeech" in list_engines()
    assert isinstance(get_engine("fastpitch"), ForwardTTSTrainingEngine)
    assert isinstance(get_engine("speedyspeech"), SpeedySpeechTrainingEngine)


def test_quality_presets():
    presets = ForwardTTSTrainingEngine().quality_presets()
    assert set(presets) == {"x-low", "medium", "high"}
    for tier in presets.values():
        assert tier["hidden_channels"] > 0
        assert tier["encoder_num_layers"] > 0


# ----------------------------------------------------------- config handling
def test_kwargs_accept_train_cli_extra_bag():
    # train.py merges preset kwargs + batch_size/validation_split/num_workers
    # into extra — the engine must accept the full bag
    eng = get_engine("fastpitch")
    cfg = TrainingEngineConfig(
        num_symbols=90,
        num_speakers=3,
        sample_rate=16000,
        extra={**eng.quality_presets()["x-low"], "batch_size": 4,
               "validation_split": 0.1, "num_workers": 0},
    )
    kw = resolve_module_kwargs(cfg, eng.default_variant)
    assert kw["num_symbols"] == 90
    assert kw["num_speakers"] == 3
    assert kw["sample_rate"] == 16000
    assert kw["hidden_channels"] == _QUALITY_PRESETS["x-low"]["hidden_channels"]
    assert kw["batch_size"] == 4
    assert kw["variant"] == "fastpitch"


def test_kwargs_do_not_mutate_presets():
    before = copy.deepcopy(_QUALITY_PRESETS)
    cfg = TrainingEngineConfig(num_symbols=90, extra={"quality": "x-low", "batch_size": 2})
    kw = resolve_module_kwargs(cfg, "fastpitch")
    kw["hidden_channels"] = -1
    assert _QUALITY_PRESETS == before


def test_kwargs_unknown_quality_falls_back_to_medium():
    cfg = TrainingEngineConfig(num_symbols=90, extra={"quality": "nonsense"})
    kw = resolve_module_kwargs(cfg, "fastpitch")
    assert kw["hidden_channels"] == _QUALITY_PRESETS["medium"]["hidden_channels"]


def test_kwargs_extra_overrides_preset():
    cfg = TrainingEngineConfig(
        num_symbols=90, extra={"quality": "medium", "hidden_channels": 7},
    )
    assert resolve_module_kwargs(cfg, "fastpitch")["hidden_channels"] == 7


def test_kwargs_variant_defaults_per_engine():
    cfg = TrainingEngineConfig(num_symbols=90)
    assert resolve_module_kwargs(cfg, "fastpitch")["variant"] == "fastpitch"
    assert resolve_module_kwargs(cfg, "speedyspeech")["variant"] == "speedyspeech"
    assert (
        get_engine("fastpitch").default_variant,
        get_engine("speedyspeech").default_variant,
    ) == ("fastpitch", "speedyspeech")


def test_kwargs_explicit_variant_wins_over_default():
    cfg = TrainingEngineConfig(num_symbols=90, extra={"variant": "speedyspeech"})
    assert resolve_module_kwargs(cfg, "fastpitch")["variant"] == "speedyspeech"


def test_kwargs_config_wins_over_extra_symbol_count():
    # num_symbols/num_speakers/sample_rate come from the shared config,
    # not from a stray extra key
    cfg = TrainingEngineConfig(num_symbols=90, extra={"num_symbols": 5})
    assert resolve_module_kwargs(cfg, "fastpitch")["num_symbols"] == 90


# --------------------------------------------------------- extra_preprocess
def test_extra_preprocess_missing_deps_returns_empty(monkeypatch, tmp_path):
    """When librosa isn't importable, extra_preprocess degrades to {}."""
    engine = ForwardTTSTrainingEngine()
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name in ("librosa", "phoonnx_train.vendor.f0"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = engine.extra_preprocess(tmp_path / "a.wav", tmp_path / "cache", 22050)
    assert result == {}
    assert not (tmp_path / "cache").exists()  # nothing written


# ------------------------------------------------------------- pitch stats
def _write_f0(path: Path, voiced_value=200.0, n=50, voiced=slice(10, 40)):
    f0 = np.zeros(n, dtype=np.float32)
    f0[voiced] = voiced_value + 10.0 * np.random.RandomState(0).randn(
        len(range(*voiced.indices(n))))
    np.save(path, f0)
    return path


def test_f0_cache_path_strips_double_suffix(tmp_path):
    spec_path = tmp_path / "utt0.spec.pt"
    assert pitch_stats.f0_cache_path(spec_path) == tmp_path / "utt0.f0.npy"


def test_pitch_stats_computed_and_cached(tmp_path):
    f0_path = _write_f0(tmp_path / "utt0.f0.npy")
    mean, std = pitch_stats.load_or_compute_pitch_stats([tmp_path], [f0_path])
    assert 190 < mean < 210 and 0 < std < 30
    cached = json.loads((tmp_path / "pitch_stats.json").read_text())
    assert cached["mean"] == mean and cached["std"] == std
    # cached value is reused even if the f0 files disappear
    f0_path.unlink()
    assert pitch_stats.load_or_compute_pitch_stats([tmp_path], []) == (mean, std)


def test_pitch_stats_no_caches_is_identity(tmp_path):
    assert pitch_stats.load_or_compute_pitch_stats([tmp_path], []) == (0.0, 1.0)
    # missing f0 files are skipped, not an error
    assert pitch_stats.load_or_compute_pitch_stats(
        [tmp_path], [tmp_path / "nope.f0.npy"]) == (0.0, 1.0)


def test_pitch_stats_all_unvoiced_is_identity(tmp_path):
    f0 = np.zeros(50, dtype=np.float32)
    np.save(tmp_path / "utt0.f0.npy", f0)
    mean, std = pitch_stats.load_or_compute_pitch_stats(
        [tmp_path], [tmp_path / "utt0.f0.npy"])
    assert std > 0  # never a zero divisor


@pytest.mark.parametrize("payload", [
    "{not json}",
    "[]",
    json.dumps({"mean": 100.0}),                 # missing std
    json.dumps({"mean": "x", "std": "y"}),       # non-numeric
    json.dumps({"mean": 100.0, "std": 0.0}),     # zero std divisor
])
def test_pitch_stats_malformed_cache_recomputed(tmp_path, payload):
    (tmp_path / "pitch_stats.json").write_text(payload)
    f0_path = _write_f0(tmp_path / "utt0.f0.npy")
    mean, std = pitch_stats.load_or_compute_pitch_stats([tmp_path], [f0_path])
    assert 190 < mean < 210 and std > 0


def test_pitch_stats_no_dataset_dir_still_computes(tmp_path):
    f0_path = _write_f0(tmp_path / "utt0.f0.npy")
    mean, std = pitch_stats.load_or_compute_pitch_stats(
        [tmp_path / "dataset.jsonl"], [f0_path])  # file, not dir — no cache
    assert 190 < mean < 210
    assert not (tmp_path / "pitch_stats.json").exists()
