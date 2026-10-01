"""Tests for the Vocos vocoder training helpers (phoonnx_train.vocos).

Torch is not part of the test environment: everything under test lives in
the torch-free ``phoonnx_train.vocos.data`` module, loaded here by file
path so no torch-importing sibling module is ever pulled in.
"""
import importlib.util
from pathlib import Path

import pytest

_DATA = Path(__file__).parent.parent / "phoonnx_train" / "vocos" / "data.py"
spec = importlib.util.spec_from_file_location("vocos_data", _DATA)
vocos_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vocos_data)

MelConfig = vocos_data.MelConfig
crop_samples = vocos_data.crop_samples
find_audio_files = vocos_data.find_audio_files
parse_warm_start = vocos_data.parse_warm_start


# ----------------------------------------------------------------------
# Mel configuration: regression-lock against the acoustic-model settings
# ----------------------------------------------------------------------

def test_mel_defaults_match_acoustic_models():
    """These values MUST equal the phoonnx acoustic-model mel settings
    (phoonnx_train/vits and the Matcha engine). A vocoder trained with
    different values produces garbage for every phoonnx mel model."""
    mel = MelConfig()
    assert mel.sample_rate == 22050
    assert mel.n_fft == 1024
    assert mel.hop_length == 256
    assert mel.win_length == 1024
    assert mel.n_mels == 80
    assert mel.fmin == 0.0
    assert mel.fmax == 8000.0


def test_mel_defaults_match_vits_config_source():
    """Cross-check against phoonnx_train/vits/config.py itself so the two
    files cannot drift silently."""
    cfg = (Path(__file__).parent.parent / "phoonnx_train" / "vits" / "config.py").read_text()
    mel = MelConfig()
    assert f"filter_length: int = {mel.n_fft}" in cfg
    assert f"hop_length: int = {mel.hop_length}" in cfg
    assert f"win_length: int = {mel.win_length}" in cfg
    assert f"mel_channels: int = {mel.n_mels}" in cfg
    assert f"sample_rate: int = {mel.sample_rate}" in cfg


def test_mel_config_is_frozen():
    with pytest.raises(Exception):
        MelConfig().n_fft = 2048


def test_frames_for_samples_hop_aligned():
    mel = MelConfig()
    assert mel.frames_for_samples(mel.hop_length * 100) == 100
    assert mel.frames_for_samples(22016) == 86


# ----------------------------------------------------------------------
# Crop math
# ----------------------------------------------------------------------

def test_crop_samples_rounds_down_to_hop_multiple():
    mel = MelConfig()
    n = crop_samples(1.0, mel)
    assert n == 22016  # 22050 rounded down to a multiple of 256
    assert n % mel.hop_length == 0


def test_crop_samples_exact_multiple_unchanged():
    mel = MelConfig(sample_rate=25600)
    assert crop_samples(1.0, mel) == 25600


@pytest.mark.parametrize("bad", [0, -1.0, -0.001])
def test_crop_samples_rejects_nonpositive(bad):
    with pytest.raises(ValueError):
        crop_samples(bad, MelConfig())


def test_crop_samples_rejects_sub_window_crop():
    # 0.01 s at 22050 Hz = 220 samples < n_fft 1024
    with pytest.raises(ValueError):
        crop_samples(0.01, MelConfig())


# ----------------------------------------------------------------------
# Audio file discovery
# ----------------------------------------------------------------------

def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_find_audio_files_recursive_sorted(tmp_path):
    _touch(tmp_path / "b.wav")
    _touch(tmp_path / "sub" / "a.wav")
    _touch(tmp_path / "sub" / "deep" / "c.flac")
    _touch(tmp_path / "notes.txt")
    files = find_audio_files([tmp_path])
    assert [f.name for f in files] == ["b.wav", "a.wav", "c.flac"] or files == sorted(files)
    assert len(files) == 3
    assert files == sorted(files)


def test_find_audio_files_multi_dir_merge_and_dedup(tmp_path):
    d1 = tmp_path / "d1"
    d2 = tmp_path / "d2"
    _touch(d1 / "x.wav")
    _touch(d2 / "y.wav")
    files = find_audio_files([d1, d2, d1])  # d1 given twice
    assert len(files) == 2


def test_find_audio_files_case_insensitive_extensions(tmp_path):
    _touch(tmp_path / "SHOUT.WAV")
    assert len(find_audio_files([tmp_path])) == 1


def test_find_audio_files_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_audio_files([tmp_path / "nope"])


def test_find_audio_files_no_dirs_raises():
    with pytest.raises(ValueError):
        find_audio_files([])


def test_find_audio_files_empty_dir_returns_empty(tmp_path):
    assert find_audio_files([tmp_path]) == []


# ----------------------------------------------------------------------
# Warm-start source parsing
# ----------------------------------------------------------------------

def test_warm_start_local_file(tmp_path):
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"x")
    assert parse_warm_start(str(ckpt)) == ("local", str(ckpt))


def test_warm_start_hf_repo_id():
    assert parse_warm_start("charactr/vocos-mel-24khz") == ("hf", "charactr/vocos-mel-24khz")


def test_warm_start_missing_local_path_raises(tmp_path):
    # looks like a path, does not exist -> must NOT be treated as a repo id
    with pytest.raises(ValueError):
        parse_warm_start(str(tmp_path / "nope.ckpt"))
    with pytest.raises(ValueError):
        parse_warm_start("./missing/model.bin")


@pytest.mark.parametrize("bad", ["", "   ", "no-slash", "a/b/c", "/a", "org/"])
def test_warm_start_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_warm_start(bad)


# ----------------------------------------------------------------------
# CLI surface (module source only — torch never imported)
# ----------------------------------------------------------------------

def test_train_cli_arg_surface():
    src = (Path(__file__).parent.parent / "phoonnx_train" / "train_vocos.py").read_text()
    for opt in ["--audio-dir", "--sample-rate", "--batch-size", "--crop-seconds",
                "--warm-start", "--max-epochs", "--checkpoint-epochs",
                "--default-root-dir", "--accelerator", "--devices", "--precision",
                "--resume-from-checkpoint"]:
        assert opt in src, f"train_vocos.py lost CLI option {opt}"
    assert '"audio_dirs", multiple=True' in src  # --audio-dir stays repeatable


def test_export_uses_registry_io_names_and_opset():
    src = (Path(__file__).parent.parent / "phoonnx_train" / "export_vocos.py").read_text()
    # the runtime VocosVocoder consumes 3 outputs (mag, real, imaginary)
    assert 'output_names=["mag", "x", "y"]' in src
    assert 'input_names=["mels"]' in src
    assert "OPSET_VERSION = 17" in src
