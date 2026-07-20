"""Pre-flight regression tests for the Matcha-TTS training path.

These guard the failure modes that would silently waste an expensive real
training run: mel front-end parameter drift between preprocessing and
training-time feature extraction, NaN/instability on degenerate batches,
padded frames leaking into the losses, unresumable checkpoints (optimizer /
epoch not restored), and a checkpoint loader that crashes on torch < 2.4.
"""
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# diffusers>=0.36 references torch.xpu at import, which does not exist on
# torch<2.4; the vendored matcha decoder imports diffusers, so skip the model
# tests cleanly when the installed pair is incompatible rather than erroring.
try:
    from phoonnx_train.matcha import MatchaTTS  # noqa: F401
    _MATCHA_IMPORTABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    _MATCHA_IMPORTABLE = False
    _MATCHA_IMPORT_ERR = exc

requires_matcha = pytest.mark.skipif(
    not _MATCHA_IMPORTABLE,
    reason=f"matcha model not importable: {_MATCHA_IMPORT_ERR if not _MATCHA_IMPORTABLE else ''}",
)


# ---------------------------------------------------------------------------
# 1. Mel/audio parameter consistency
# ---------------------------------------------------------------------------

def test_mel_frontend_matches_norm_audio_defaults():
    """The mel front-end Matcha trains on must use the same STFT geometry the
    preprocess step cached the normalized audio / spectrogram with; a mismatch
    trains and infers in different feature spaces."""
    from phoonnx_train.matcha import dataset as mds
    import inspect
    from phoonnx_train.norm_audio import cache_norm_audio

    sig = inspect.signature(cache_norm_audio)
    assert mds.N_FFT == sig.parameters["filter_length"].default == 1024
    assert mds.HOP_LENGTH == sig.parameters["hop_length"].default == 256
    assert mds.WIN_LENGTH == sig.parameters["window_length"].default == 1024


def test_mel_fmax_within_nyquist_for_default_sample_rate():
    """fmax must not exceed Nyquist for the training sample rate."""
    from phoonnx_train.matcha import dataset as mds
    # default matcha training sample rate is 22050 -> Nyquist 11025
    assert 0 <= mds.F_MIN < mds.F_MAX <= 22050 // 2


# ---------------------------------------------------------------------------
# Tiny model factory
# ---------------------------------------------------------------------------

def _tiny_model(n_vocab=20, n_feats=80, n_spks=1):
    from phoonnx_train.engines.matcha import (
        MatchaEngineConfig,
        _build_model_kwargs,
    )
    from phoonnx_train.matcha import MatchaTTS

    mcfg = MatchaEngineConfig(
        n_vocab=n_vocab,
        n_feats=n_feats,
        n_spks=n_spks,
        encoder_channels=64,
        encoder_filter_channels=128,
        encoder_filter_channels_dp=64,
        encoder_n_heads=2,
        encoder_n_layers=2,
        decoder_channels=[64, 64],
        decoder_num_heads=2,
        decoder_num_mid_blocks=1,
    )
    model = MatchaTTS(
        **_build_model_kwargs(mcfg),
        data_statistics={"mel_mean": -3.0, "mel_std": 1.5},
    )
    model.eval()
    return model


def _batch(text_len, mel_len, n_feats=80, n_vocab=20, seed=0):
    from phoonnx_train.matcha.utils import fix_len_compatibility

    g = torch.Generator().manual_seed(seed)
    y_max = fix_len_compatibility(mel_len)
    x = torch.zeros(1, text_len, dtype=torch.long)
    x[0, :text_len] = torch.randint(1, n_vocab, (text_len,), generator=g)
    y = torch.zeros(1, n_feats, y_max)
    y[0, :, :mel_len] = torch.randn(n_feats, mel_len, generator=g)
    return {
        "x": x,
        "x_lengths": torch.LongTensor([text_len]),
        "y": y,
        "y_lengths": torch.LongTensor([mel_len]),
        "spks": None,
        "durations": None,
    }


# ---------------------------------------------------------------------------
# 2. NaN / instability on adversarial batches
# ---------------------------------------------------------------------------

@requires_matcha
@pytest.mark.parametrize("text_len,mel_len", [(1, 4), (2, 4), (3, 8), (1, 1)])
def test_losses_finite_on_short_and_degenerate_batches(text_len, mel_len):
    """1-frame / single-phoneme clips must not produce NaN/Inf losses
    (log-of-zero, div-by-zero in the duration/prior/diff paths)."""
    model = _tiny_model()
    torch.manual_seed(0)
    batch = _batch(text_len, mel_len)
    losses = model.get_losses(batch)
    for name, loss in losses.items():
        assert torch.isfinite(loss).all(), f"{name} is not finite: {loss}"


# ---------------------------------------------------------------------------
# 4. Padded frames must be masked out of the losses
# ---------------------------------------------------------------------------

@requires_matcha
def test_padding_does_not_change_prior_and_duration_loss():
    """Appending zero-padded mel frames (with y_lengths unchanged) must not
    change the deterministic (prior/duration) losses — proof padded frames are
    masked out. If collate padding leaked into the loss, this diverges."""
    model = _tiny_model()
    from phoonnx_train.matcha.utils import fix_len_compatibility

    text_len, mel_len, n_feats = 5, 12, 80
    g = torch.Generator().manual_seed(3)
    x = torch.randint(1, 20, (1, text_len), generator=g)
    mel = torch.randn(n_feats, mel_len, generator=g)

    def prior_dur(y_pad_to):
        y = torch.zeros(1, n_feats, y_pad_to)
        y[0, :, :mel_len] = mel
        batch = {
            "x": x, "x_lengths": torch.LongTensor([text_len]),
            "y": y, "y_lengths": torch.LongTensor([mel_len]),
            "spks": None, "durations": None,
        }
        with torch.no_grad():
            dur, prior, _, _ = model(
                x=batch["x"], x_lengths=batch["x_lengths"],
                y=batch["y"], y_lengths=batch["y_lengths"], out_size=None,
            )
        return dur, prior

    tight = fix_len_compatibility(mel_len)
    dur_a, prior_a = prior_dur(tight)
    dur_b, prior_b = prior_dur(tight + 16)  # extra padded frames
    assert torch.allclose(prior_a, prior_b, atol=1e-5), (prior_a, prior_b)
    assert torch.allclose(dur_a, dur_b, atol=1e-5), (dur_a, dur_b)


# ---------------------------------------------------------------------------
# 5. Checkpoint loader must not depend on torch>=2.4 APIs
# ---------------------------------------------------------------------------

@requires_matcha
def test_load_checkpoint_is_torch_version_portable(tmp_path):
    """load_checkpoint must round-trip a Matcha checkpoint without touching
    torch.serialization.safe_globals (absent on torch<2.4) and tolerate the
    SimpleNamespace config objects stored in hyper_parameters."""
    from phoonnx_train.engines.matcha import MatchaTrainingEngine

    model = _tiny_model()
    ckpt_path = tmp_path / "m.ckpt"
    # a lightning-style checkpoint carries a state_dict plus non-tensor hparams
    torch.save({"state_dict": model.state_dict(),
                "epoch": 7, "global_step": 42}, ckpt_path)

    fresh = _tiny_model()
    engine = MatchaTrainingEngine()
    # must not raise AttributeError on torch<2.4
    engine.load_checkpoint(fresh, ckpt_path)
    # weights actually copied
    for k, v in model.state_dict().items():
        assert torch.equal(v, fresh.state_dict()[k]), k


def test_matcha_engine_source_has_no_safe_globals_reference():
    """Regression: torch.serialization.safe_globals does not exist on the
    project's supported torch floor (>=2.1); the engine must not reference it."""
    src = Path(__file__).resolve().parents[1] / "phoonnx_train" / "engines" / "matcha.py"
    assert "safe_globals" not in src.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 3. Resume must restore optimizer / epoch / global_step (via ckpt_path)
# ---------------------------------------------------------------------------

def test_plain_resume_passes_ckpt_path_to_fit(tmp_path, monkeypatch):
    """A plain --resume-from-checkpoint must hand the checkpoint to
    Trainer.fit(ckpt_path=...) so Lightning restores optimizer state, epoch and
    global_step. A weight-only manual load would silently reset them."""
    import json
    from click.testing import CliRunner
    import phoonnx_train.train as train_mod

    captured = {}

    class FakeTrainer:
        def __init__(self, *a, **k):
            pass

        def fit(self, model, ckpt_path=None):
            captured["ckpt_path"] = ckpt_path

    class FakeEngine:
        def quality_presets(self):
            return {"medium": {}}

        def create_model(self, config, dataset_paths):
            return object()

        def trainer_kwargs(self):
            return {}

        def load_checkpoint(self, *a, **k):
            captured["manual_load"] = True

    monkeypatch.setattr(train_mod, "Trainer", FakeTrainer)
    monkeypatch.setattr(train_mod, "get_engine", lambda name: FakeEngine())
    monkeypatch.setattr(train_mod, "list_engines", lambda: ["matcha"])

    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "config.json").write_text(json.dumps(
        {"num_symbols": 30, "num_speakers": 1, "audio": {"sample_rate": 22050}}))
    ckpt = tmp_path / "prev.ckpt"
    ckpt.write_bytes(b"x")

    runner = CliRunner()
    result = runner.invoke(train_mod.main, [
        "--dataset-dir", str(ds), "--engine", "matcha",
        "--max-epochs", "1", "--resume-from-checkpoint", str(ckpt),
    ])
    assert result.exit_code == 0, result.output
    assert captured.get("ckpt_path") == str(ckpt)
    # a true resume must NOT do a weight-only manual load (that skips optimizer)
    assert "manual_load" not in captured


def test_discard_encoder_resume_is_weight_only(tmp_path, monkeypatch):
    """--discard-encoder changes the architecture: it must stay a weight-only
    warm start (no ckpt_path, so optimizer state is not force-loaded into a
    mismatched param set)."""
    import json
    from click.testing import CliRunner
    import phoonnx_train.train as train_mod

    captured = {}

    class FakeTrainer:
        def __init__(self, *a, **k):
            pass

        def fit(self, model, ckpt_path=None):
            captured["ckpt_path"] = ckpt_path

    class FakeEngine:
        def quality_presets(self):
            return {"medium": {}}

        def create_model(self, config, dataset_paths):
            return object()

        def trainer_kwargs(self):
            return {}

        def load_checkpoint(self, *a, **k):
            captured["manual_load"] = True

    monkeypatch.setattr(train_mod, "Trainer", FakeTrainer)
    monkeypatch.setattr(train_mod, "get_engine", lambda name: FakeEngine())
    monkeypatch.setattr(train_mod, "list_engines", lambda: ["matcha"])

    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "config.json").write_text(json.dumps(
        {"num_symbols": 30, "num_speakers": 1, "audio": {"sample_rate": 22050}}))
    ckpt = tmp_path / "prev.ckpt"
    ckpt.write_bytes(b"x")

    runner = CliRunner()
    result = runner.invoke(train_mod.main, [
        "--dataset-dir", str(ds), "--engine", "matcha", "--max-epochs", "1",
        "--resume-from-checkpoint", str(ckpt), "--discard-encoder",
    ])
    assert result.exit_code == 0, result.output
    assert captured.get("ckpt_path") is None
    assert captured.get("manual_load") is True
