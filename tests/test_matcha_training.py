"""Tests for the Matcha-TTS training engine (phoonnx_train.engines.matcha).

Torch is not part of the test environment: the engine registry, config
handling and dataset.jsonl parsing are all importable without torch (heavy
imports are deferred until a model is actually built), and the jsonl parsing
rules live in the torch-free ``phoonnx_train.matcha.jsonl`` module, loaded
here by file path so the matcha package ``__init__`` (which needs torch) is
never imported.
"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from phoonnx_train.engines import get_engine, list_engines
from phoonnx_train.engines.base import TrainingEngineConfig
from phoonnx_train.engines.matcha import (
    MatchaEngineConfig,
    MatchaTrainingEngine,
    _QUALITY_PRESETS,
)

_JSONL = Path(__file__).parent.parent / "phoonnx_train" / "matcha" / "jsonl.py"
spec = importlib.util.spec_from_file_location("matcha_jsonl", _JSONL)
matcha_jsonl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matcha_jsonl)
load_dataset_lines = matcha_jsonl.load_dataset_lines


@pytest.fixture()
def tiny_jsonl(tmp_path):
    """dataset.jsonl in the phoonnx preprocessed format (metadata only)."""
    lines = [
        {"phoneme_ids": [1 + (j + i) % 50 for j in range(12)],
         "audio_norm_path": str(tmp_path / f"utt{i}.pt")}
        for i in range(4)
    ]
    path = tmp_path / "dataset.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


def test_registered():
    assert "matcha" in list_engines()
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


def test_config_ignores_unknown_extra_keys():
    cfg = TrainingEngineConfig(num_symbols=90, extra={"not_a_field": 1, "batch_size": 8})
    mcfg = MatchaEngineConfig.from_training_config(cfg)
    assert mcfg.batch_size == 8
    assert not hasattr(mcfg, "not_a_field")


def test_config_speaker_count_propagates():
    mcfg = MatchaEngineConfig.from_training_config(
        TrainingEngineConfig(num_symbols=90, num_speakers=7)
    )
    assert mcfg.n_spks == 7


def test_quality_presets_all_define_encoder_and_decoder():
    for name, preset in _QUALITY_PRESETS.items():
        assert preset["encoder_channels"] > 0, name
        assert len(preset["decoder_channels"]) >= 1, name


def test_jsonl_skips_malformed_lines(tiny_jsonl):
    with open(tiny_jsonl, "a", encoding="utf-8") as f:
        f.write("{not json}\n")
        f.write(json.dumps({"phoneme_ids": []}) + "\n")  # empty ids
        f.write(json.dumps({"audio_norm_path": "/nope.pt"}) + "\n")  # no ids
        f.write("\n")  # blank line
    assert len(list(load_dataset_lines(tiny_jsonl))) == 4


def test_jsonl_empty_yields_nothing(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text("")
    assert list(load_dataset_lines(path)) == []


def test_jsonl_max_phoneme_ids_filter(tiny_jsonl):
    assert len(list(load_dataset_lines(tiny_jsonl, max_phoneme_ids=12))) == 4
    # all utterances have 12 ids -> a lower cap filters everything
    assert list(load_dataset_lines(tiny_jsonl, max_phoneme_ids=5)) == []


def test_jsonl_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(load_dataset_lines(tmp_path / "nope.jsonl"))
