"""Tests for the Matcha-TTS training engine (phoonnx_train.engines.matcha)."""
import copy
import json
from pathlib import Path

import pytest
import torch

from phoonnx_train.engines import get_engine
from phoonnx_train.engines.base import TrainingEngineConfig
from phoonnx_train.engines.matcha import (
    MatchaEngineConfig,
    MatchaTrainingEngine,
    _QUALITY_PRESETS,
)
from phoonnx_train.matcha.dataset import MatchaDataset, collate_matcha


@pytest.fixture()
def tiny_dataset(tmp_path):
    """Synthetic dataset in the phoonnx preprocessed format."""
    torch.manual_seed(0)
    lines = []
    for i in range(4):
        wav = torch.rand(11025) * 0.2 - 0.1
        wav_path = tmp_path / f"utt{i}.pt"
        torch.save(wav, wav_path)
        lines.append(
            {"phoneme_ids": [1 + (j + i) % 50 for j in range(12)],
             "audio_norm_path": str(wav_path)}
        )
    with open(tmp_path / "dataset.jsonl", "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return tmp_path


def test_registered():
    assert isinstance(get_engine("matcha"), MatchaTrainingEngine)


def test_config_accepts_train_cli_extra_bag():
    # train.py merges preset kwargs + batch_size/validation_split/num_workers
    # into extra — the engine must accept the full bag
    eng = get_engine("matcha")
    cfg = TrainingEngineConfig(
        num_symbols=90,
        num_speakers=1,
        extra={**eng.quality_presets()["x-low"], "batch_size": 4,
               "validation_split": 0.1, "num_workers": 0},
    )
    mcfg = MatchaEngineConfig.from_training_config(cfg)
    assert mcfg.n_vocab == 90
    assert mcfg.encoder_channels == 128
    assert mcfg.batch_size == 4


def test_config_does_not_mutate_presets():
    before = copy.deepcopy(_QUALITY_PRESETS)
    cfg = TrainingEngineConfig(num_symbols=90, extra={"quality": "x-low", "batch_size": 2})
    MatchaEngineConfig.from_training_config(cfg)
    assert _QUALITY_PRESETS == before


def test_config_unknown_quality_falls_back_to_medium():
    cfg = TrainingEngineConfig(num_symbols=90, extra={"quality": "nonsense"})
    mcfg = MatchaEngineConfig.from_training_config(cfg)
    assert mcfg.encoder_channels == _QUALITY_PRESETS["medium"]["encoder_channels"]


def test_dataset_skips_malformed_lines(tmp_path, tiny_dataset):
    with open(tiny_dataset / "dataset.jsonl", "a", encoding="utf-8") as f:
        f.write("{not json}\n")
        f.write(json.dumps({"phoneme_ids": []}) + "\n")  # empty ids
        f.write(json.dumps({"audio_norm_path": "/nope.pt"}) + "\n")  # no ids
    ds = MatchaDataset([tiny_dataset], sample_rate=11025)
    assert len(ds) == 4


def test_dataset_empty_raises(tmp_path):
    (tmp_path / "dataset.jsonl").write_text("")
    with pytest.raises(ValueError):
        MatchaDataset([tmp_path])


def test_dataset_max_phoneme_ids_filter(tiny_dataset):
    assert len(MatchaDataset([tiny_dataset], sample_rate=11025, max_phoneme_ids=12).utterances) == 4
    # all utterances have 12 ids -> a lower cap filters everything -> empty dataset raises
    with pytest.raises(ValueError):
        MatchaDataset([tiny_dataset], sample_rate=11025, max_phoneme_ids=5)


def test_collate_pads_to_unet_compatible_length(tiny_dataset):
    ds = MatchaDataset([tiny_dataset], sample_rate=11025)
    batch = collate_matcha([ds[0], ds[1]])
    assert batch["x"].shape[0] == 2
    assert batch["y"].shape[1] == 80
    assert batch["y"].shape[2] % 4 == 0  # 2 UNet downsamplings
    assert batch["spks"] is None
    assert int(batch["y_lengths"].max()) <= batch["y"].shape[2]


def test_create_model_and_training_step(tiny_dataset):
    eng = get_engine("matcha")
    cfg = TrainingEngineConfig(
        num_symbols=90, num_speakers=1, sample_rate=11025,
        extra={**eng.quality_presets()["x-low"], "batch_size": 2,
               "validation_split": 0.25, "num_workers": 0},
    )
    model = eng.create_model(cfg, dataset_paths=[tiny_dataset])
    # dataset statistics were computed and cached
    stats = json.loads((tiny_dataset / "matcha_stats.json").read_text())
    assert stats["mel_std"] > 0
    assert float(model.mel_std) == pytest.approx(stats["mel_std"])

    batch = next(iter(model.train_dataloader()))
    losses = model.get_losses(batch)
    total = sum(losses.values())
    assert torch.isfinite(total), losses
    total.backward()  # gradients flow
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)


def test_load_checkpoint_tolerates_shape_mismatch(tiny_dataset, tmp_path):
    eng = get_engine("matcha")
    base = {"quality": "x-low", "batch_size": 2, "mel_mean": 0.0, "mel_std": 1.0}
    m1 = eng.create_model(TrainingEngineConfig(num_symbols=90, extra=dict(base)), dataset_paths=[])
    ckpt_path = tmp_path / "m1.ckpt"
    torch.save({"state_dict": m1.state_dict()}, ckpt_path)
    # different vocab size -> embedding shape mismatch must be dropped, not crash
    m2 = eng.create_model(TrainingEngineConfig(num_symbols=120, extra=dict(base)), dataset_paths=[])
    eng.load_checkpoint(m2, ckpt_path)


def test_mas_matches_bruteforce_and_backends_agree():
    import itertools

    import numpy as np

    from phoonnx_train.matcha.mas import maximum_path, maximum_path_numpy

    def brute(v):
        ts, tt = v.shape
        best, bp = -1e18, None
        for bounds in itertools.combinations(range(1, tt), ts - 1):
            bounds = (0,) + bounds + (tt,)
            s = sum(v[i, bounds[i]:bounds[i + 1]].sum() for i in range(ts))
            if s > best:
                best, bp = s, bounds
        path = np.zeros((ts, tt), np.float32)
        for i in range(ts):
            path[i, bp[i]:bp[i + 1]] = 1
        return path

    rng = np.random.default_rng(0)
    for _ in range(5):
        v = rng.standard_normal((4, 9)).astype(np.float32)
        expected = brute(v)
        got_np = maximum_path_numpy(v[None], np.ones((1, 4, 9), bool))[0]
        got = maximum_path(torch.tensor(v)[None], torch.ones(1, 4, 9))[0].numpy()
        assert np.allclose(got_np, expected)
        assert np.allclose(got, expected)  # active backend (cython shim or numpy)

    # ragged batches: both backends agree
    torch.manual_seed(1)
    for _ in range(10):
        ts, tt = int(rng.integers(2, 12)), int(rng.integers(12, 40))
        val = torch.randn(2, ts, tt)
        mask = torch.ones(2, ts, tt)
        for i in range(2):
            mask[i, int(rng.integers(1, ts + 1)):, :] = 0
            mask[i, :, int(rng.integers(ts, tt + 1)):] = 0
        a = maximum_path(val.clone(), mask.clone()).numpy()
        c = maximum_path_numpy(val.numpy().astype(np.float32), mask.numpy().astype(bool))
        assert np.allclose(a, c)
