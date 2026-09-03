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


def _write_sine_wav(path: Path, freq_hz: float = 220.0, seconds: float = 0.5,
                    sr: int = 22050) -> None:
    import wave as wave_mod

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    samples = (0.3 * np.sin(2 * np.pi * freq_hz * t) * 32767).astype(np.int16)
    with wave_mod.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(samples.tobytes())


def test_extra_preprocess_f0_method_selects_cache_filename(tmp_path):
    """pyin (default) and dio (WORLD, opt-in) must write to distinct cache
    filenames for the same utterance — no silent cross-method reuse."""
    engine = ForwardTTSTrainingEngine()
    wav_path = tmp_path / "a.wav"
    _write_sine_wav(wav_path)
    cache_dir = tmp_path / "cache"

    pyin_result = engine.extra_preprocess(wav_path, cache_dir, 22050, f0_method="pyin")
    dio_result = engine.extra_preprocess(wav_path, cache_dir, 22050, f0_method="dio")

    assert pyin_result and dio_result
    pyin_path = Path(pyin_result["f0_path"])
    dio_path = Path(dio_result["f0_path"])
    assert pyin_path != dio_path
    assert pyin_path.name.endswith("f0-pyin.npy")
    assert dio_path.name.endswith("f0-dio.npy")
    assert pyin_path.is_file() and dio_path.is_file()

    pyin_f0 = np.load(pyin_path)
    dio_f0 = np.load(dio_path)
    assert np.any(pyin_f0 > 0)
    assert np.any(dio_f0 > 0)


# ------------------------------------------------------------- pitch stats
def _write_f0(path: Path, voiced_value=200.0, n=50, voiced=slice(10, 40)):
    f0 = np.zeros(n, dtype=np.float32)
    f0[voiced] = voiced_value + 10.0 * np.random.RandomState(0).randn(
        len(range(*voiced.indices(n))))
    np.save(path, f0)
    return path


def test_f0_cache_path_strips_double_suffix(tmp_path):
    from phoonnx_train.vendor.f0 import EXTRACTOR_TAG

    spec_path = tmp_path / "utt0.spec.pt"
    assert (pitch_stats.f0_cache_path(spec_path)
            == tmp_path / f"utt0.f0-{EXTRACTOR_TAG}.npy")


def test_f0_cache_path_keys_by_extraction_method(tmp_path):
    """A cache written under the pre-pyin (pyworld-era) naming scheme is a
    clean miss under the current key — it must not be silently reused with
    the new extractor — while a cache written under the current key round-
    trips (write, then read hits the same path)."""
    from phoonnx_train.vendor.f0 import EXTRACTOR_TAG

    spec_path = tmp_path / "utt0.spec.pt"
    legacy_path = tmp_path / "utt0.f0.npy"  # pre-tag (pyworld-era) filename
    current_path = pitch_stats.f0_cache_path(spec_path)

    assert current_path != legacy_path
    assert current_path.name == f"utt0.f0-{EXTRACTOR_TAG}.npy"

    # a stale pyworld-era cache sitting next to the spec is never picked up
    _write_f0(legacy_path)
    assert not current_path.exists()

    # writing under the current key makes it discoverable by the same
    # derivation used at read time
    _write_f0(current_path)
    assert pitch_stats.f0_cache_path(spec_path) == current_path
    assert current_path.exists()


def test_f0_cache_path_separates_pyin_and_world_methods(tmp_path):
    """dio/harvest are opt-in WORLD backends and must get their own cache
    filenames — never silently share a pyin-era (or each other's) cache."""
    spec_path = tmp_path / "utt0.spec.pt"
    pyin_path = pitch_stats.f0_cache_path(spec_path, method="pyin")
    dio_path = pitch_stats.f0_cache_path(spec_path, method="dio")
    harvest_path = pitch_stats.f0_cache_path(spec_path, method="harvest")

    assert len({pyin_path, dio_path, harvest_path}) == 3
    assert pyin_path.name == "utt0.f0-pyin.npy"
    assert dio_path.name == "utt0.f0-dio.npy"
    assert harvest_path.name == "utt0.f0-harvest.npy"


def test_stats_filename_separates_methods(tmp_path):
    assert pitch_stats.stats_filename("pyin") == "pitch_stats-pyin.json"
    assert pitch_stats.stats_filename("dio") == "pitch_stats-dio.json"
    assert pitch_stats.stats_filename("harvest") == "pitch_stats-harvest.json"
    assert len({
        pitch_stats.stats_filename("pyin"),
        pitch_stats.stats_filename("dio"),
        pitch_stats.stats_filename("harvest"),
    }) == 3


def test_load_or_compute_pitch_stats_keyed_by_method(tmp_path):
    """Stats computed under one method's cache file must not be read back
    (or overwritten) under a different method's stats filename."""
    pyin_f0 = _write_f0(tmp_path / "utt0.f0-pyin.npy", voiced_value=200.0)
    dio_f0 = _write_f0(tmp_path / "utt0.f0-dio.npy", voiced_value=300.0)

    pyin_mean, _ = pitch_stats.load_or_compute_pitch_stats([tmp_path], [pyin_f0], method="pyin")
    dio_mean, _ = pitch_stats.load_or_compute_pitch_stats([tmp_path], [dio_f0], method="dio")

    assert 190 < pyin_mean < 210
    assert 290 < dio_mean < 310
    assert (tmp_path / "pitch_stats-pyin.json").is_file()
    assert (tmp_path / "pitch_stats-dio.json").is_file()


def test_pitch_stats_computed_and_cached(tmp_path):
    f0_path = _write_f0(tmp_path / "utt0.f0.npy")
    mean, std = pitch_stats.load_or_compute_pitch_stats([tmp_path], [f0_path])
    assert 190 < mean < 210 and 0 < std < 30
    cached = json.loads((tmp_path / pitch_stats.STATS_FILENAME).read_text())
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
    (tmp_path / pitch_stats.STATS_FILENAME).write_text(payload)
    f0_path = _write_f0(tmp_path / "utt0.f0.npy")
    mean, std = pitch_stats.load_or_compute_pitch_stats([tmp_path], [f0_path])
    assert 190 < mean < 210 and std > 0


def test_pitch_stats_no_dataset_dir_still_computes(tmp_path):
    f0_path = _write_f0(tmp_path / "utt0.f0.npy")
    mean, std = pitch_stats.load_or_compute_pitch_stats(
        [tmp_path / "dataset.jsonl"], [f0_path])  # file, not dir — no cache
    assert 190 < mean < 210
    assert not (tmp_path / pitch_stats.STATS_FILENAME).exists()
