"""Every training engine's export writes a config that the native loader reads.

Each test trains nothing: it saves a freshly initialised tiny model as a
checkpoint, exports it with the engine under test, and loads the config the
exporter wrote through ``VoiceConfig.from_dict``. The loaded phoneme type,
alphabet, punctuation flag and phonemizer identity must equal those of the
training config.
"""
import dataclasses
import json
from pathlib import Path

import pytest
import torch

from phoonnx.config import VoiceConfig
from phoonnx_train.engines import get_engine
from phoonnx_train.engines.base import TrainingEngineConfig

IDENTITY = {
    "backend": "gruut", "distribution": "gruut", "version": "2.4.0",
    "lang": "en-US", "revision": None, "frontend": None,
}
TRAINING_CONFIG = {
    "phoonnx_version": "1.0.0",
    "lang_code": "en-US",
    "audio": {"sample_rate": 22050},
    "phoneme_type": "gruut",
    "alphabet": "ipa",
    "keep_punctuation": True,
    "phonemizer_identity": IDENTITY,
    "phoneme_id_map": {"_": 0, "^": 1, "$": 2, " ": 3, "a": 4, "b": 5},
    "num_symbols": 6,
    "num_speakers": 1,
}


def _lightning_checkpoint(engine_name: str, path: Path) -> None:
    import pytorch_lightning as pl

    config = TrainingEngineConfig(num_symbols=20, num_speakers=1, sample_rate=22050,
                                  extra={"quality": "x-low"})
    model = get_engine(engine_name).create_model(config, [])
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, accelerator="cpu",
                         enable_progress_bar=False)
    trainer.strategy.connect(model)
    trainer.save_checkpoint(str(path))


def _export(engine_name: str, tmp_path: Path, training_config: dict) -> Path:
    checkpoint = tmp_path / "last.ckpt"
    _lightning_checkpoint(engine_name, checkpoint)
    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(training_config), encoding="utf-8")
    get_engine(engine_name).export_onnx(checkpoint, config_path, tmp_path / "out")
    return tmp_path / "out"


def _assert_loads_with_training_facts(config_file: Path) -> None:
    written = json.loads(config_file.read_text(encoding="utf-8"))
    assert VoiceConfig.is_phoonnx(written)
    loaded = VoiceConfig.from_dict(written)
    assert loaded.phoneme_type.value == TRAINING_CONFIG["phoneme_type"]
    assert loaded.alphabet.value == TRAINING_CONFIG["alphabet"]
    assert loaded.keep_punctuation is TRAINING_CONFIG["keep_punctuation"]
    assert loaded.phonemizer_identity == IDENTITY
    assert written["phoonnx_version"] == TRAINING_CONFIG["phoonnx_version"]


@pytest.mark.parametrize("engine_name, config_name", [
    ("glowtts", "last.json"),
    ("fastpitch", "last.json"),
    ("mixer", "last.json"),
    ("matcha", "model.json"),
])
def test_exported_config_loads_with_training_facts(engine_name, config_name, tmp_path):
    out = _export(engine_name, tmp_path, TRAINING_CONFIG)
    _assert_loads_with_training_facts(out / config_name)


def test_zipvoice_exported_config_loads_with_training_facts(tmp_path):
    out = _export("zipvoice", tmp_path, TRAINING_CONFIG)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["engine"] == "zipvoice"
    assert config["audio"]["sample_rate"] == 24000
    _assert_loads_with_training_facts(out / "config.json")


def test_styletts2_exported_config_loads_with_training_facts(tmp_path):
    from phoonnx_train.engines.styletts2 import StyleTTS2Module
    from tests.test_styletts2_training import TINY_MODEL_PARAMS, tiny_config

    model_params = {**TINY_MODEL_PARAMS, "decoder": {
        "type": "hifigan",
        "resblock_kernel_sizes": [3],
        "upsample_rates": [10, 6],
        "upsample_initial_channel": 512,
        "resblock_dilation_sizes": [[1, 3, 5]],
        "upsample_kernel_sizes": [20, 12],
    }}
    training_config = {**TRAINING_CONFIG,
                       "styletts2": {"model_params": model_params, "use_slm": False}}
    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(training_config), encoding="utf-8")

    module = StyleTTS2Module(dataclasses.replace(tiny_config("first"), model_params=model_params))
    checkpoint = tmp_path / "first_stage.pth"
    torch.save({"net": {k: module.model[k].state_dict() for k in module.model}}, checkpoint)

    get_engine("styletts2").export_onnx(checkpoint, config_path, tmp_path / "out")
    config = json.loads((tmp_path / "out" / "config.json").read_text(encoding="utf-8"))
    assert config["engine"] == "styletts2"
    _assert_loads_with_training_facts(tmp_path / "out" / "config.json")


def test_styletts2_export_leaves_gradients_on(tmp_path):
    import torch
    test_styletts2_exported_config_loads_with_training_facts(tmp_path)
    assert torch.is_grad_enabled()
