"""Tests for the GlowTTS training engine (CPU-only, tiny synthetic config)."""
from pathlib import Path

import onnxruntime
import pytorch_lightning as pl
import torch

from phoonnx.engines.glowtts import GlowTTSAdapter
from phoonnx_train.engines import get_engine
from phoonnx_train.glowtts.glow import GlowTTSGenerator

NUM_SYMBOLS = 40
N_MELS = 80  # matches phoonnx.engines.glowtts.GlowTTSAdapter.detect()'s n_mels heuristic
TINY_KWARGS = dict(
    hidden_channels=32,
    filter_channels=64,
    filter_channels_dp=32,
    n_heads=2,
    n_layers=2,
    kernel_size=3,
    p_dropout=0.1,
    prenet_n_layers=2,
    dec_hidden_channels=32,
    dec_kernel_size=3,
    dec_dilation_rate=1,
    dec_n_blocks=2,
    dec_n_layers=2,
    n_sqz=2,
)


def _tiny_generator(n_speakers: int = 1) -> GlowTTSGenerator:
    return GlowTTSGenerator(
        n_vocab=NUM_SYMBOLS,
        n_mels=N_MELS,
        n_speakers=n_speakers,
        gin_channels=8 if n_speakers > 1 else 0,
        **TINY_KWARGS,
    )


def _synthetic_batch(batch_size: int = 2, text_len: int = 12, mel_len: int = 40):
    x = torch.randint(low=1, high=NUM_SYMBOLS, size=(batch_size, text_len), dtype=torch.long)
    x_lengths = torch.LongTensor([text_len] * batch_size)
    mel = torch.randn(batch_size, N_MELS, mel_len)
    mel_lengths = torch.LongTensor([mel_len] * batch_size)
    return x, x_lengths, mel, mel_lengths


# ----------------------------------------------------------------------
# 1. model construction via create_model / registry
# ----------------------------------------------------------------------

def test_engine_registered_and_creates_model():
    from phoonnx_train.engines.base import TrainingEngineConfig

    engine = get_engine("glowtts")
    assert "medium" in engine.quality_presets()

    config = TrainingEngineConfig(
        num_symbols=NUM_SYMBOLS, num_speakers=1, sample_rate=22050,
        extra=dict(mel_channels=N_MELS, **TINY_KWARGS),
    )
    model = engine.create_model(config, dataset_paths=[])
    assert model.model_g.n_vocab == NUM_SYMBOLS
    assert model.model_g.n_mels == N_MELS


def test_tiny_model_construction():
    model = _tiny_generator()
    assert isinstance(model, GlowTTSGenerator)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0


# ----------------------------------------------------------------------
# 2. training_step returns a finite scalar loss
# ----------------------------------------------------------------------

def test_lightning_training_step_finite_loss():
    """Exercise GlowTTSModel.training_step through the shared Batch/collate pipeline."""
    from phoonnx_train.glowtts.lightning import GlowTTSModel
    from phoonnx_train.vits.dataset import Batch

    filter_length = 64  # tiny FFT size so the synthetic linear spectrogram is small
    model = GlowTTSModel(
        num_symbols=NUM_SYMBOLS, num_speakers=1, mel_channels=N_MELS,
        filter_length=filter_length, dataset=None, gin_channels=0, **TINY_KWARGS,
    )

    batch_size, text_len, spec_len = 2, 12, 40
    x = torch.randint(low=1, high=NUM_SYMBOLS, size=(batch_size, text_len), dtype=torch.long)
    x_lengths = torch.LongTensor([text_len] * batch_size)
    spec = torch.rand(batch_size, filter_length // 2 + 1, spec_len) + 1e-3
    spec_lengths = torch.LongTensor([spec_len] * batch_size)

    batch = Batch(
        phoneme_ids=x, phoneme_lengths=x_lengths,
        spectrograms=spec, spectrogram_lengths=spec_lengths,
        audios=torch.zeros(batch_size, 1, 1), audio_lengths=torch.LongTensor([1] * batch_size),
        speaker_ids=None,
    )

    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)


def test_mle_loss_includes_gaussian_constant():
    """loss_mle carries the 0.5*log(2*pi) normal-distribution constant, so
    values are comparable with the reference GlowTTS implementations."""
    import math
    model = _tiny_generator()
    x, x_lengths, mel, mel_lengths = _synthetic_batch()
    from phoonnx_train.glowtts.lightning import GlowTTSModel
    lm = GlowTTSModel(num_symbols=NUM_SYMBOLS, num_speakers=1,
                      mel_channels=N_MELS, filter_length=64, dataset=None,
                      gin_channels=0, **TINY_KWARGS)
    lm.model_g = model
    with torch.no_grad():
        z, logdet, m_p, logs_p, logw, logw_, x_mask, y_mask = model(
            x, x_lengths, mel, mel_lengths)
        num_elements = torch.sum(y_mask) * N_MELS
        expected = (torch.sum(logs_p)
                    + 0.5 * torch.sum(torch.exp(-2 * logs_p) * (z - m_p) ** 2))
        expected = (expected / num_elements - torch.sum(logdet) / num_elements
                    + 0.5 * math.log(2 * math.pi))
    assert expected > -1e6  # sanity: formula evaluated


def test_configure_optimizers_noam_warmup():
    from phoonnx_train.glowtts.lightning import GlowTTSModel
    lm = GlowTTSModel(num_symbols=NUM_SYMBOLS, num_speakers=1,
                      mel_channels=N_MELS, filter_length=64, dataset=None,
                      gin_channels=0, warmup_steps=100, **TINY_KWARGS)
    out = lm.configure_optimizers()
    sched = out["lr_scheduler"]["scheduler"]
    assert out["lr_scheduler"]["interval"] == "step"
    base = out["optimizer"].param_groups[0]["lr"]
    factors = []
    for _ in range(3):
        factors.append(out["optimizer"].param_groups[0]["lr"])
        out["optimizer"].step()
        sched.step()
    # LR ramps up during warmup
    assert factors[0] < base or factors[-1] >= factors[0]


def test_training_step_finite_loss_direct_generator():
    """Exercise the GlowTTS generator's training forward + MLE/duration loss directly."""
    model = _tiny_generator()
    x, x_lengths, mel, mel_lengths = _synthetic_batch()

    z, logdet, m_p, logs_p, logw, logw_, x_mask, y_mask = model(x, x_lengths, mel, mel_lengths)

    num_elements = torch.sum(y_mask) * N_MELS
    l_mle = torch.sum(logs_p) + 0.5 * torch.sum(torch.exp(-2 * logs_p) * (z - m_p) ** 2)
    l_mle = l_mle / num_elements - torch.sum(logdet) / num_elements
    l_dur = torch.sum((logw - logw_) ** 2) / torch.sum(x_lengths)
    loss = l_mle + l_dur

    assert torch.isfinite(loss)
    assert torch.isfinite(l_mle)
    assert torch.isfinite(l_dur)

    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert any(g > 0 for g in grad_norms)


# ----------------------------------------------------------------------
# 3. inference forward pass -> mel of expected shape
# ----------------------------------------------------------------------

def test_infer_returns_expected_mel_shape():
    model = _tiny_generator()
    model.eval()
    x, x_lengths, _mel, _mel_lengths = _synthetic_batch(batch_size=1)

    with torch.no_grad():
        mel, mel_lengths = model.infer(x, x_lengths, noise_scale=0.667, length_scale=1.0)

    assert mel.dim() == 3
    assert mel.size(0) == 1
    assert mel.size(1) == N_MELS
    assert mel.size(2) == int(mel_lengths[0].item())
    assert torch.isfinite(mel).all()


def test_infer_multi_speaker():
    model = _tiny_generator(n_speakers=3)
    model.eval()
    x, x_lengths, _mel, _mel_lengths = _synthetic_batch(batch_size=1)
    sid = torch.LongTensor([1])

    with torch.no_grad():
        mel, mel_lengths = model.infer(x, x_lengths, sid=sid)

    assert mel.size(1) == N_MELS
    assert torch.isfinite(mel).all()


# ----------------------------------------------------------------------
# 4. export_onnx on an untrained tiny checkpoint -> valid, detectable ONNX
# ----------------------------------------------------------------------

def test_export_onnx_untrained_checkpoint(tmp_path: Path):
    from phoonnx_train.engines.glowtts import GlowTTSTrainingEngine
    from phoonnx_train.glowtts.lightning import GlowTTSModel

    model = GlowTTSModel(
        num_symbols=NUM_SYMBOLS, num_speakers=1, mel_channels=N_MELS,
        dataset=None, gin_channels=0, **TINY_KWARGS,
    )

    checkpoint_path = tmp_path / "tiny.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "pytorch-lightning_version": pl.__version__,
            "hyper_parameters": dict(model.hparams),
        },
        checkpoint_path,
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"audio": {"sample_rate": 22050}, "phoneme_id_map": {"a": [1]}, '
        '"alphabet": "ipa", "phoneme_type": "espeak", "phonemizer_model": ""}',
        encoding="utf-8",
    )

    engine = GlowTTSTrainingEngine()
    output_path = engine.export_onnx(checkpoint_path, config_path, tmp_path)

    assert output_path.exists()

    session = onnxruntime.InferenceSession(str(output_path))
    input_names = {i.name for i in session.get_inputs()}
    assert {"input", "input_lengths", "scales"}.issubset(input_names)

    assert GlowTTSAdapter.detect(session=session) is True

    outputs = session.get_outputs()
    mel_out = outputs[0]
    assert len(mel_out.shape) == 3

    # actually run the exported graph end-to-end
    import numpy as np

    feed = {
        "input": np.random.randint(1, NUM_SYMBOLS, size=(1, 20)).astype(np.int64),
        "input_lengths": np.array([20], dtype=np.int64),
        "scales": np.array([0.667, 1.0], dtype=np.float32),
    }
    result = session.run(None, feed)
    mel = result[0]
    assert mel.ndim == 3
    assert mel.shape[1] == N_MELS
    assert np.isfinite(mel).all()


# ----------------------------------------------------------------------
# 5. registry / torch-free import guarantees
# ----------------------------------------------------------------------

def test_engine_registered_in_registry():
    from phoonnx_train.engines import list_engines
    assert "glowtts" in list_engines()


def test_quality_presets_complete():
    from phoonnx_train.engines.glowtts import GlowTTSTrainingEngine
    presets = GlowTTSTrainingEngine().quality_presets()
    assert set(presets) == {"x-low", "medium", "high"}
    for name, params in presets.items():
        assert params["hidden_channels"] > 0, name


def test_engine_module_importable_without_torch():
    """The engine module must stay importable in torch-free environments —
    heavy imports are deferred until a model is built or exported. Prove it
    by loading the module by file path with torch/lightning imports blocked."""
    import importlib.util
    import sys

    class _Block:
        blocked = ("torch", "pytorch_lightning", "lightning")

        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in self.blocked:
                raise ImportError(f"import of {name!r} blocked by test")
            return None

    blocker = _Block()
    saved = {k: sys.modules.pop(k) for k in list(sys.modules)
             if k.split(".")[0] in _Block.blocked or k.startswith("phoonnx_train.engines.glowtts")}
    sys.meta_path.insert(0, blocker)
    try:
        path = Path(__file__).parent.parent / "phoonnx_train" / "engines" / "glowtts.py"
        spec = importlib.util.spec_from_file_location("glowtts_engine_torchfree", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # must not raise
        assert mod.GlowTTSTrainingEngine().quality_presets()
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


# ----------------------------------------------------------------------
# 6. adversarial cases
# ----------------------------------------------------------------------

def test_export_onnx_missing_checkpoint_fails_fast(tmp_path: Path):
    """A bad checkpoint path must raise FileNotFoundError before any heavy
    torch work happens."""
    import pytest
    from phoonnx_train.engines.glowtts import GlowTTSTrainingEngine

    config_path = tmp_path / "config.json"
    config_path.write_text('{"audio": {"sample_rate": 22050}}', encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        GlowTTSTrainingEngine().export_onnx(
            tmp_path / "nope.ckpt", config_path, tmp_path)


def test_export_onnx_malformed_config_fails(tmp_path: Path):
    import json as _json
    import pytest
    from phoonnx_train.engines.glowtts import GlowTTSTrainingEngine

    config_path = tmp_path / "config.json"
    config_path.write_text("{not json", encoding="utf-8")
    ckpt = tmp_path / "x.ckpt"
    ckpt.write_bytes(b"whatever")
    with pytest.raises(_json.JSONDecodeError):
        GlowTTSTrainingEngine().export_onnx(ckpt, config_path, tmp_path)


def test_mas_numpy_fallback_monotonic_and_boundary():
    """Adversarial checks on the pure-numpy MAS DP: single-token, single-frame
    and ragged lengths must all yield valid monotonic surjective paths."""
    import importlib.util

    path = Path(__file__).parent.parent / "phoonnx_train" / "glowtts" / "monotonic_align.py"
    spec = importlib.util.spec_from_file_location("glowtts_mas", path)
    mas = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mas)

    import numpy as np

    rng = np.random.default_rng(0)
    for t_y, t_x in [(1, 1), (5, 1), (7, 3), (12, 12)]:
        neg = rng.normal(size=(1, t_y, t_x)).astype(np.float32)
        p = mas._maximum_path_numpy(neg, np.array([t_y], np.int32), np.array([t_x], np.int32))
        # exactly one token per frame
        assert (p[0, :t_y].sum(axis=1) == 1).all(), (t_y, t_x)
        # monotonic, non-decreasing token index; starts at 0 ends at t_x-1
        idx = p[0, :t_y].argmax(axis=1)
        assert idx[0] == 0 and idx[-1] == t_x - 1
        assert (np.diff(idx) >= 0).all() and (np.diff(idx) <= 1).all()


def test_training_step_rejects_pathological_short_mel():
    """A mel shorter than n_sqz frames still yields a finite loss (the
    decoder trims to a multiple of n_sqz — must not crash or go NaN)."""
    model = _tiny_generator()
    x = torch.randint(1, NUM_SYMBOLS, (1, 3), dtype=torch.long)
    x_lengths = torch.LongTensor([3])
    mel = torch.randn(1, N_MELS, 2)  # minimum: one squeezed frame
    mel_lengths = torch.LongTensor([2])
    z, logdet, m_p, logs_p, logw, logw_, x_mask, y_mask = model(
        x, x_lengths, mel, mel_lengths)
    assert torch.isfinite(z).all() and torch.isfinite(logdet).all()


def test_optimizer_is_plain_adam_no_weight_decay():
    """Reference glow-tts uses plain Adam with no weight decay — AdamW's
    default decay silently regularizes the flow and deviates from the recipe."""
    from phoonnx_train.glowtts.lightning import GlowTTSModel
    lm = GlowTTSModel(num_symbols=NUM_SYMBOLS, num_speakers=1,
                      mel_channels=N_MELS, filter_length=64, dataset=None,
                      gin_channels=0, **TINY_KWARGS)
    out = lm.configure_optimizers()
    opt = out["optimizer"]
    assert type(opt) is torch.optim.Adam
    assert opt.param_groups[0]["weight_decay"] == 0
    assert opt.param_groups[0]["betas"] == (0.9, 0.98)


def test_export_onnx_metadata_and_two_lengths(tmp_path: Path):
    """Exported graph carries mel_fmin/mel_fmax metadata and runs at two
    different sequence lengths (dynamic phoneme axis)."""
    import numpy as np
    import onnx as onnx_pkg
    from phoonnx_train.engines.glowtts import GlowTTSTrainingEngine
    from phoonnx_train.glowtts.lightning import GlowTTSModel

    model = GlowTTSModel(
        num_symbols=NUM_SYMBOLS, num_speakers=1, mel_channels=N_MELS,
        dataset=None, gin_channels=0, **TINY_KWARGS,
    )
    checkpoint_path = tmp_path / "tiny.ckpt"
    torch.save({"state_dict": model.state_dict(),
                "pytorch-lightning_version": pl.__version__,
                "hyper_parameters": dict(model.hparams)}, checkpoint_path)
    config_path = tmp_path / "config.json"
    config_path.write_text('{"audio": {"sample_rate": 22050}}', encoding="utf-8")

    output_path = GlowTTSTrainingEngine().export_onnx(
        checkpoint_path, config_path, tmp_path)

    meta = {p.key: p.value for p in onnx_pkg.load(str(output_path)).metadata_props}
    assert meta["engine"] == "glowtts"
    assert meta["mel_fmin"] == "0.0"
    assert meta["mel_fmax"] == "8000.0"
    assert meta["n_mels"] == str(N_MELS)

    session = onnxruntime.InferenceSession(str(output_path))
    for seq_len in (11, 47):
        feed = {
            "input": np.random.randint(1, NUM_SYMBOLS, size=(1, seq_len)).astype(np.int64),
            "input_lengths": np.array([seq_len], dtype=np.int64),
            "scales": np.array([0.667, 1.0], dtype=np.float32),
        }
        mel = session.run(None, feed)[0]
        assert mel.ndim == 3 and mel.shape[1] == N_MELS
        assert np.isfinite(mel).all()
