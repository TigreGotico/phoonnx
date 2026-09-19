"""CPU-only tests for the StyleTTS2 training engine.

Everything runs with a tiny random-init model (no auxiliary checkpoints, no
WavLM download): the point is that the full two-stage recipe — alignment,
segment slicing, style encoding, decoder reconstruction, duration/prosody
prediction — executes end-to-end and produces finite losses.
"""
from pathlib import Path

import numpy as np
import torch

from phoonnx_train.engines import get_engine, list_engines
from phoonnx_train.engines.base import TrainingEngineConfig
from phoonnx_train.engines.styletts2 import (
    StyleTTS2Config,
    StyleTTS2Module,
    StyleTTS2TrainingEngine,
    _QUALITY_PRESETS,
)

SR = 24000
HOP = 300

# Tiny model so a CPU forward/backward completes in seconds.
TINY_MODEL_PARAMS = {
    "dim_in": 16,
    "hidden_dim": 32,
    "max_conv_dim": 32,
    "n_layer": 1,
    "n_mels": 80,
    "n_token": 178,
    "max_dur": 50,
    "style_dim": 16,
    "dropout": 0.0,
    "decoder": {
        "type": "istftnet",
        "resblock_kernel_sizes": [3],
        "upsample_rates": [10, 6],
        # the yl4579 decoder trunk is fixed-width (1024/512) regardless of
        # hidden_dim, and its Generator has no conv_pre — must stay 512
        "upsample_initial_channel": 512,
        "resblock_dilation_sizes": [[1, 3, 5]],
        "upsample_kernel_sizes": [20, 12],
        "gen_istft_n_fft": 20,
        "gen_istft_hop_size": 5,
    },
    "diffusion": {
        "transformer": {
            "num_layers": 1,
            "num_heads": 2,
            "head_features": 8,
            "multiplier": 1,
        },
    },
}


def tiny_config(stage: str) -> StyleTTS2Config:
    return StyleTTS2Config(
        stage=stage,
        model_params=TINY_MODEL_PARAMS,
        # keep clips small and CPU-friendly
        max_len=192,
        batch_size=2,
        num_workers=0,
        use_slm=False,
        use_slm_adv=False,
        loss_params={"tma_epoch": 0, "diff_epoch": 10_000, "joint_epoch": 10_000}
        if stage == "second" else {"tma_epoch": 10_000},
    )


def synthetic_batch(batch_size: int = 2, n_frames: int = 120, n_tokens: int = 12):
    """Build a batch in the meldataset.Collater layout:
    waves, texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels
    """
    torch.manual_seed(0)
    np.random.seed(0)
    waves = [np.random.uniform(-0.3, 0.3, n_frames * 2 * HOP).astype(np.float32)
             for _ in range(batch_size)]
    texts = torch.randint(1, 100, (batch_size, n_tokens), dtype=torch.long)
    input_lengths = torch.full((batch_size,), n_tokens, dtype=torch.long)
    mels = torch.randn(batch_size, 80, n_frames * 2)
    mel_input_length = torch.full((batch_size,), n_frames * 2, dtype=torch.long)
    ref_mels = torch.randn(batch_size, 80, n_frames * 2)
    return [waves, texts, input_lengths, texts.clone(), input_lengths.clone(),
            mels, mel_input_length, ref_mels]


def _attach_manual_optimizers(module: StyleTTS2Module):
    """Wire just enough of the Trainer contract for training_step to run."""
    opts = module.configure_optimizers()
    module.optimizers = lambda: opts  # type: ignore[assignment]
    module.manual_backward = lambda loss, *a, **k: loss.backward(*a, **k)  # type: ignore[assignment]
    logged = {}
    module.log = lambda name, value, *a, **k: logged.__setitem__(name, value)  # type: ignore[assignment]
    module.log_dict = lambda d, *a, **k: logged.update(d)  # type: ignore[assignment]
    return logged


def test_engine_registered():
    assert "styletts2" in list_engines()
    assert isinstance(get_engine("styletts2"), StyleTTS2TrainingEngine)


def test_quality_presets():
    presets = StyleTTS2TrainingEngine().quality_presets()
    assert set(presets) == {"low", "medium", "high"}
    assert presets["high"]["decoder"]["type"] == "hifigan"
    assert presets is _QUALITY_PRESETS


def test_config_from_training_config():
    cfg = TrainingEngineConfig(num_symbols=200, num_speakers=4, sample_rate=24000,
                               extra={"stage": "second", "quality": "high",
                                      "use_slm": False,
                                      # train.py flattens preset/model keys into extra
                                      "hidden_dim": 64,
                                      "validation_split": 0.1})
    scfg = StyleTTS2Config.from_training_config(cfg)
    assert scfg.stage == "second"
    assert scfg.use_slm is False
    mp = scfg.resolved_model_params()
    assert mp["n_token"] == 200
    assert mp["multispeaker"] is True
    assert mp["decoder"]["type"] == "hifigan"
    assert mp["hidden_dim"] == 64


def test_finetune_enables_joint_training_from_epoch_zero():
    scfg = StyleTTS2Config(stage="finetune")
    lp = scfg.resolved_loss_params()
    assert lp["diff_epoch"] == 0
    assert lp["joint_epoch"] == 0
    # from-scratch second stage keeps the upstream schedule
    lp2 = StyleTTS2Config(stage="second").resolved_loss_params()
    assert lp2["diff_epoch"] == 20
    assert lp2["joint_epoch"] == 50


def test_monotonic_maximum_path():
    from phoonnx_train.styletts2.monotonic import mask_from_lens, maximum_path
    torch.manual_seed(0)
    value = torch.randn(2, 5, 9)
    in_lens = torch.tensor([5, 4])
    out_lens = torch.tensor([9, 7])
    mask = mask_from_lens(value, in_lens, out_lens)
    path = maximum_path(value, mask)
    assert path.shape == value.shape
    for b, (t_x, t_y) in enumerate(zip(in_lens, out_lens)):
        sub = path[b, :t_x, :t_y]
        # exactly one active text index per output frame, monotonically increasing
        assert torch.all(sub.sum(0) == 1)
        idx = sub.argmax(0)
        assert torch.all(idx[1:] - idx[:-1] >= 0)
        # path fully covers the text axis
        assert torch.all(sub.sum(1) >= 1)
        # nothing outside the valid region
        assert path[b, t_x:, :].sum() == 0
        assert path[b, :, t_y:].sum() == 0


def _finite(v):
    return torch.is_tensor(v) and torch.isfinite(v).all()


def test_stage_first_training_step_runs():
    module = StyleTTS2Module(tiny_config("first"))
    logged = _attach_manual_optimizers(module)
    module._training_step_first(synthetic_batch())
    assert "train_loss" in logged, "stage-1 step returned before computing losses"
    assert _finite(logged["train_loss"])
    assert _finite(logged["mel_loss"])


def test_stage_second_training_step_runs():
    module = StyleTTS2Module(tiny_config("second"))
    logged = _attach_manual_optimizers(module)
    module._training_step_second(synthetic_batch(), batch_idx=0)
    assert "train_loss" in logged, "stage-2 step returned before computing losses"
    for key in ("train_loss", "mel_loss", "dur_loss", "ce_loss", "F0_loss"):
        assert _finite(logged[key]), key


def test_stage_second_updates_predictor_only_before_joint():
    module = StyleTTS2Module(tiny_config("second"))
    _attach_manual_optimizers(module)
    dec_before = [p.detach().clone() for p in module.model.decoder.parameters()]
    pred_before = [p.detach().clone() for p in module.model.predictor.parameters()]
    module._training_step_second(synthetic_batch(), batch_idx=0)
    dec_after = list(module.model.decoder.parameters())
    pred_after = list(module.model.predictor.parameters())
    assert all(torch.equal(a, b) for a, b in zip(dec_before, dec_after)), \
        "decoder must stay frozen before joint_epoch"
    assert any(not torch.equal(a, b) for a, b in zip(pred_before, pred_after)), \
        "predictor must be updated in stage 2"


def test_net_checkpoint_roundtrip(tmp_path: Path):
    module = StyleTTS2Module(tiny_config("first"))
    ckpt = {"net": {k: module.model[k].state_dict() for k in module.model}}
    path = tmp_path / "first_stage.pth"
    torch.save(ckpt, path)

    module2 = StyleTTS2Module(tiny_config("first"))
    module2._load_net_checkpoint(path)
    for k in module.model:
        sd1 = module.model[k].state_dict()
        sd2 = module2.model[k].state_dict()
        for name in sd1:
            assert torch.equal(sd1[name], sd2[name]), f"{k}.{name} not restored"


def test_engine_create_model(tmp_path: Path):
    train_list = tmp_path / "train_list.txt"
    val_list = tmp_path / "val_list.txt"
    train_list.write_text("a.wav|h e l o|0\nb.wav|w o r l d|0\n")
    val_list.write_text("c.wav|t e s t|0\n")
    cfg = TrainingEngineConfig(num_symbols=178, num_speakers=1, sample_rate=SR,
                               extra={"stage": "first", "use_slm": False,
                                      "model_params": TINY_MODEL_PARAMS})
    engine = get_engine("styletts2")
    model = engine.create_model(cfg, [train_list, val_list])
    assert isinstance(model, StyleTTS2Module)
    assert len(model.train_list) == 2
    assert len(model.val_list) == 1
    assert model.cfg.root_path == str(tmp_path)
