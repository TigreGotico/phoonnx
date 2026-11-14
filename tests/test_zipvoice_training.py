"""Tests for the ZipVoice training engine (phoonnx_train.engines.zipvoice).

Torch is not part of the test environment: the engine registry, preset and
config-bag handling are importable without torch (heavy imports are
deferred until a model is actually built), and the shared dataset.jsonl
parsing rules live in the torch-free ``phoonnx_train.matcha.jsonl`` module
— the same loader ``ZipVoiceDataset`` consumes — loaded here by file path
so no torch-needing package ``__init__`` is ever imported.
"""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from phoonnx_train.engines import get_engine, list_engines
from phoonnx_train.engines.base import BaseTrainingEngine, TrainingEngineConfig
from phoonnx_train.engines.zipvoice import (
    _MODEL_KEYS,
    _QUALITY_PRESETS,
    ZipVoiceTrainingEngine,
    split_model_params,
)

_JSONL = Path(__file__).parent.parent / "phoonnx_train" / "matcha" / "jsonl.py"
spec = importlib.util.spec_from_file_location("shared_jsonl", _JSONL)
shared_jsonl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared_jsonl)
load_dataset_lines = shared_jsonl.load_dataset_lines


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


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

def test_registered():
    assert "zipvoice" in list_engines()
    assert isinstance(get_engine("zipvoice"), ZipVoiceTrainingEngine)


def test_registry_returns_fresh_instances():
    assert get_engine("zipvoice") is not get_engine("zipvoice")


def test_implements_engine_contract():
    assert issubclass(ZipVoiceTrainingEngine, BaseTrainingEngine)
    eng = get_engine("zipvoice")
    presets = eng.quality_presets()
    assert isinstance(presets, dict) and presets


def test_unknown_engine_raises_keyerror():
    with pytest.raises(KeyError):
        get_engine("zipvoice-distill")


# ----------------------------------------------------------------------
# Presets / config-bag handling
# ----------------------------------------------------------------------

def test_presets_base_is_upstream_defaults():
    # "base" must be an empty override set → the vendored model's own
    # (upstream) defaults apply
    assert _QUALITY_PRESETS["base"] == {}


def test_presets_low_only_contains_model_keys():
    assert set(_QUALITY_PRESETS["low"]) <= _MODEL_KEYS


def test_split_routes_model_keys_and_trainer_keys():
    model_params, trainer = split_model_params(
        {"quality": "low", "batch_size": 2, "validation_split": 0.1,
         "num_workers": 0, "feat_dim": 100})
    assert model_params["feat_dim"] == 100
    assert model_params["text_encoder_dim"] == 32  # from the "low" preset
    assert trainer == {"batch_size": 2, "validation_split": 0.1,
                       "num_workers": 0}


def test_split_accepts_train_cli_extra_bag():
    # train.py merges the resolved preset kwargs + batch_size /
    # validation_split / num_workers into extra (no "quality" key) —
    # the split must route every preset key to the model
    eng = get_engine("zipvoice")
    extra = {**eng.quality_presets()["low"], "batch_size": 4,
             "validation_split": 0.05, "num_workers": 0}
    model_params, trainer = split_model_params(extra)
    assert model_params["fm_decoder_dim"] == 64
    assert "fm_decoder_dim" not in trainer
    assert trainer["batch_size"] == 4


def test_split_explicit_model_params_beat_preset():
    model_params, _ = split_model_params(
        {"quality": "low", "model_params": {"text_encoder_dim": 48}})
    assert model_params["text_encoder_dim"] == 48


def test_split_bare_key_beats_model_params_dict():
    model_params, _ = split_model_params(
        {"model_params": {"feat_dim": 80}, "feat_dim": 100})
    assert model_params["feat_dim"] == 100


def test_split_unknown_quality_falls_back_to_base():
    model_params, trainer = split_model_params({"quality": "nonsense"})
    assert model_params == {}
    assert trainer == {}


def test_split_does_not_mutate_inputs_or_presets():
    before = copy.deepcopy(_QUALITY_PRESETS)
    extra = {"quality": "low", "batch_size": 2, "feat_dim": 100}
    snapshot = dict(extra)
    split_model_params(extra)
    assert extra == snapshot
    assert _QUALITY_PRESETS == before


def test_split_empty_bag():
    assert split_model_params({}) == ({}, {})


def test_engine_config_extra_bag_survives_roundtrip():
    cfg = TrainingEngineConfig(num_symbols=90, num_speakers=1,
                               sample_rate=22050,
                               extra={"quality": "low", "batch_size": 2})
    model_params, trainer = split_model_params(cfg.extra)
    assert model_params["text_encoder_dim"] == 32
    assert trainer == {"batch_size": 2}
    # the config's own bag is untouched
    assert cfg.extra == {"quality": "low", "batch_size": 2}


# ----------------------------------------------------------------------
# dataset.jsonl parsing (the loader ZipVoiceDataset consumes)
# ----------------------------------------------------------------------

def test_jsonl_parses_valid_lines(tiny_jsonl):
    utts = list(load_dataset_lines(tiny_jsonl))
    assert len(utts) == 4
    assert all(u["phoneme_ids"] and u["audio_norm_path"] for u in utts)


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
