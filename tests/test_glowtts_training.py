"""Tests for the GlowTTS training engine (phoonnx_train.engines.glowtts).

Torch is not part of the test environment: the engine registry, config
handling and quality presets are all importable without torch (heavy
imports are deferred until a model is actually built or exported), and the
Monotonic Alignment Search dynamic-programming fallback is pure numpy,
loaded here by file path so the glowtts model modules (which need torch)
are never imported.

The torch-dependent behavior (Lightning training step, checkpoint resume,
ONNX export + onnxruntime inference at multiple sequence lengths,
GlowTTSAdapter.detect() on the exported graph, mel_fmin/mel_fmax metadata)
is exercised in the CPU smoke run documented in the pull request.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from phoonnx_train.engines import get_engine, list_engines
from phoonnx_train.engines.base import TrainingEngineConfig
from phoonnx_train.engines.glowtts import _QUALITY_PRESETS, GlowTTSTrainingEngine

_MAS = Path(__file__).parent.parent / "phoonnx_train" / "glowtts" / "monotonic_align.py"
spec = importlib.util.spec_from_file_location("glowtts_mas", _MAS)
mas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mas)


# ---------------------------------------------------------------- registry
def test_engine_registered():
    assert "glowtts" in list_engines()
    assert isinstance(get_engine("glowtts"), GlowTTSTrainingEngine)


def test_quality_presets_complete():
    presets = GlowTTSTrainingEngine().quality_presets()
    assert presets is _QUALITY_PRESETS
    assert set(presets) == {"x-low", "medium", "high"}
    for name, params in presets.items():
        assert params["hidden_channels"] > 0, name
        assert params["dec_n_blocks"] > 0, name
    # tiers are strictly ordered by capacity
    assert (presets["x-low"]["hidden_channels"]
            < presets["medium"]["hidden_channels"]
            < presets["high"]["hidden_channels"])


def test_engine_has_no_extra_cli_options():
    assert GlowTTSTrainingEngine().extra_cli_options() == []


def test_engine_module_importable_without_torch():
    """The engine module must stay importable in torch-free environments —
    heavy imports are deferred until a model is built or exported. Prove it
    by loading the module by file path with torch/lightning imports blocked."""

    class _Block:
        blocked = ("torch", "pytorch_lightning", "lightning")

        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in self.blocked:
                raise ImportError(f"import of {name!r} blocked by test")
            return None

    blocker = _Block()
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k.split(".")[0] in _Block.blocked}
    sys.meta_path.insert(0, blocker)
    try:
        path = (Path(__file__).parent.parent
                / "phoonnx_train" / "engines" / "glowtts.py")
        s = importlib.util.spec_from_file_location("glowtts_engine_torchfree", path)
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)  # must not raise
        assert mod.GlowTTSTrainingEngine().quality_presets()
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


def test_engine_config_extra_bag_roundtrip():
    config = TrainingEngineConfig(
        num_symbols=40, num_speakers=1, sample_rate=22050,
        extra={"hidden_channels": 32, "batch_size": 2},
    )
    assert config.extra["hidden_channels"] == 32
    assert config.num_symbols == 40


# ---------------------------------------------------------------- export fast-fail
def test_export_onnx_missing_checkpoint_fails_fast(tmp_path: Path):
    """A bad checkpoint path must raise FileNotFoundError before any heavy
    torch import happens (so it fails identically without torch installed)."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"audio": {"sample_rate": 22050}}', encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        GlowTTSTrainingEngine().export_onnx(
            tmp_path / "nope.ckpt", config_path, tmp_path)


def test_export_onnx_malformed_config_fails(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{not json", encoding="utf-8")
    ckpt = tmp_path / "x.ckpt"
    ckpt.write_bytes(b"whatever")
    with pytest.raises(json.JSONDecodeError):
        GlowTTSTrainingEngine().export_onnx(ckpt, config_path, tmp_path)


def test_export_onnx_missing_config_fails(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        GlowTTSTrainingEngine().export_onnx(
            tmp_path / "x.ckpt", tmp_path / "missing.json", tmp_path)


# ---------------------------------------------------------------- MAS numpy DP
def test_mas_numpy_monotonic_and_boundary():
    """Adversarial checks on the pure-numpy MAS DP: single-token,
    single-frame and ragged lengths must all yield valid monotonic
    surjective paths."""
    rng = np.random.default_rng(0)
    for t_y, t_x in [(1, 1), (5, 1), (7, 3), (12, 12)]:
        neg = rng.normal(size=(1, t_y, t_x)).astype(np.float32)
        p = mas._maximum_path_numpy(
            neg, np.array([t_y], np.int32), np.array([t_x], np.int32))
        # exactly one token per frame
        assert (p[0, :t_y].sum(axis=1) == 1).all(), (t_y, t_x)
        # monotonic non-decreasing token index; starts at 0, ends at t_x-1
        idx = p[0, :t_y].argmax(axis=1)
        assert idx[0] == 0 and idx[-1] == t_x - 1
        assert (np.diff(idx) >= 0).all() and (np.diff(idx) <= 1).all()


def test_mas_numpy_prefers_high_likelihood_path():
    """The DP must actually maximize the summed log-likelihood, not just
    return any monotonic path: with a strongly diagonal score matrix the
    chosen path is the diagonal."""
    t = 6
    neg = np.full((1, t, t), -10.0, dtype=np.float32)
    for i in range(t):
        neg[0, i, i] = 0.0
    p = mas._maximum_path_numpy(
        neg, np.array([t], np.int32), np.array([t], np.int32))
    assert (p[0].argmax(axis=1) == np.arange(t)).all()


def test_mas_numpy_ragged_batch_padding_untouched():
    """Batch entries with shorter true lengths must not write outside their
    valid region (padding rows/cols stay zero)."""
    rng = np.random.default_rng(1)
    neg = rng.normal(size=(2, 10, 8)).astype(np.float32)
    t_ys = np.array([10, 4], np.int32)
    t_xs = np.array([8, 3], np.int32)
    p = mas._maximum_path_numpy(neg, t_ys, t_xs)
    assert p[1, 4:, :].sum() == 0
    assert p[1, :, 3:].sum() == 0
    assert (p[1, :4].sum(axis=1) == 1).all()


def test_mas_numpy_zero_scores_still_valid_path():
    """Degenerate all-zero scores (ties everywhere) must still produce a
    well-formed monotonic path rather than looping or dropping frames."""
    t_y, t_x = 9, 4
    neg = np.zeros((1, t_y, t_x), dtype=np.float32)
    p = mas._maximum_path_numpy(
        neg, np.array([t_y], np.int32), np.array([t_x], np.int32))
    idx = p[0].argmax(axis=1)
    assert (p[0].sum(axis=1) == 1).all()
    assert idx[0] == 0 and idx[-1] == t_x - 1
    assert set(idx) == set(range(t_x))  # surjective: every token gets frames
