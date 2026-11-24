"""Known defects, each pinned as a strict xfail until its fix lands.

Each test states the behaviour the code should have. It fails on the current
code for the reason in its ``reason``, which names the issue that tracks it.
The fix PR removes the mark; a fix that lands without that edit turns the
strict xfail into a failure, so the pin cannot go stale.
"""
import importlib.util
import json
import logging
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import click.testing
import numpy as np
import pytest
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent


def _wav(path: Path, seconds: float = 0.5, sr: int = 22050) -> Path:
    tone = 0.2 * np.sin(2 * np.pi * 220 * np.arange(int(sr * seconds)) / sr)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, tone.astype(np.float32), sr)
    return path


# --- SuperTonic through the shared train and export CLIs -------------------

@pytest.mark.xfail(strict=True, raises=IsADirectoryError,
                   reason="supertonic-cli: train.py hands the dataset directory to stages that read a filelist")
def test_supertonic_trains_through_the_train_cli(tmp_path):
    from phoonnx_train.train import main

    dataset = tmp_path / "ds"
    _wav(dataset / "audio" / "utt0.wav")
    (dataset / "train.filelist").write_text("audio/utt0.wav|Hello there.|en\n", encoding="utf-8")

    def fit(model, **_):
        model.setup("fit")
        next(iter(model.train_dataloader()))

    with patch("phoonnx_train.train.Trainer") as trainer_cls:
        trainer_cls.return_value = MagicMock(fit=MagicMock(side_effect=fit))
        result = click.testing.CliRunner().invoke(main, [
            "--dataset-dir", str(dataset), "--engine", "supertonic", "--quality", "low",
            "--batch-size", "1", "--num-workers", "0", "--default-root-dir", str(tmp_path / "run"),
        ], catch_exceptions=False)
    assert result.exit_code == 0, result.output


@pytest.mark.xfail(strict=True, raises=KeyError,
                   reason="supertonic-cli: export_onnx.py has no way to pass the text_to_latent and "
                          "duration_predictor checkpoints")
def test_supertonic_export_cli_asks_for_the_stage_checkpoints(tmp_path):
    from phoonnx_train.export_onnx import cli

    checkpoint = tmp_path / "ae.ckpt"
    checkpoint.write_bytes(b"")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    result = click.testing.CliRunner().invoke(cli, [
        str(checkpoint), "--config", str(config), "--engine", "supertonic",
        "--output-dir", str(tmp_path / "out"),
    ], catch_exceptions=False)
    # a usage error that names the missing input, not a traceback
    assert result.exit_code == 2, result.output
    assert "text" in result.output and "latent" in result.output, result.output


# --- export_onnx --add-phoneme-alignment ------------------------------------

def _graph_without_ceil(path: Path) -> Path:
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    onnx.save(helper.make_model(graph), path)
    return path


def _export_with_alignment(tmp_path):
    from phoonnx_train import export_onnx

    graph = _graph_without_ceil(tmp_path / "model.onnx")
    engine = MagicMock(export_onnx=MagicMock(return_value=graph))
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    with patch.object(export_onnx, "get_engine", return_value=engine):
        return click.testing.CliRunner().invoke(export_onnx.cli, [
            str(checkpoint), "--config", str(config), "--output-dir", str(tmp_path / "out"),
            "--add-phoneme-alignment",
        ])


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="export-alignment: a failed --add-phoneme-alignment still exits 0")
def test_failed_alignment_export_exits_non_zero(tmp_path):
    result = _export_with_alignment(tmp_path)
    assert result.exit_code != 0, result.output


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="export-alignment: the error names a --tensor-name flag the CLI does not have")
def test_alignment_errors_name_only_real_flags(tmp_path, caplog):
    from phoonnx_train.export_onnx import cli

    with caplog.at_level(logging.DEBUG, logger="phoonnx_train.export_onnx"):
        _export_with_alignment(tmp_path)
    flags = {opt for param in cli.params for opt in param.opts}
    assert "--tensor-name" in flags or "--tensor-name" not in caplog.text, caplog.text


# --- preprocess --jsonl-audio-path / --jsonl-audio-spec-path ---------------

def _run_preprocess(args):
    from phoonnx.config import Alphabet
    from phoonnx_train import preprocess
    from scriptconv.phonemizers.base import GraphemePhonemizer, PhonemizerIdentity

    class Phonemizer(GraphemePhonemizer):
        def __init__(self):
            super().__init__(Alphabet.IPA)

        def identity(self, lang):
            return PhonemizerIdentity("dummy", "dummy", "0", lang)

    with patch.object(preprocess, "get_phonemizer", return_value=Phonemizer()):
        result = click.testing.CliRunner().invoke(
            preprocess.cli, [*args, "--max-workers", "1"], catch_exceptions=False)
    return SimpleNamespace(exit_code=result.exit_code, output=result.output)


def _preprocess(args):
    # preprocess forks its workers, and a fork after earlier tests started torch's
    # thread pools in this process deadlocks, so each run gets a fresh interpreter
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        return pool.submit(_run_preprocess, args).result()


def _manifest(out: Path):
    return [json.loads(line) for line in (out / "dataset.jsonl").read_text().splitlines() if line]


def _audio_override_row(tmp_path: Path, clip: Path, *extra: str) -> dict:
    source = tmp_path / "a.jsonl"
    source.write_text(json.dumps({"text": "hi", "audio": str(clip), "phon": "h i"}) + "\n")
    out = tmp_path / "out"
    result = _preprocess(["-i", str(source), "-o", str(out), "-l", "en", "--phonemes-column", "phon", *extra])
    assert result.exit_code == 0, result.output
    return _manifest(out)[0]


def test_jsonl_audio_path_rewrites_the_ljspeech_wavs_layout(tmp_path):
    clip = _wav(tmp_path / "LJSpeech" / "wavs" / "clip1.wav", sr=16000)
    row = _audio_override_row(tmp_path, clip, "--jsonl-audio-path", "/override/base")
    assert row["audio_path"] == "/override/base/wavs/clip1.wav", row["audio_path"]


def test_jsonl_audio_path_keeps_only_the_audio_folder_of_a_path_with_two_wav_folders(tmp_path):
    clip = _wav(tmp_path / "wav" / "x" / "wav" / "clip1.wav", sr=16000)
    row = _audio_override_row(tmp_path, clip, "--jsonl-audio-path", "/override/base")
    assert row["audio_path"] == "/override/base/wav/clip1.wav", row["audio_path"]


def test_jsonl_audio_spec_path_rewrites_a_custom_cache_dir(tmp_path):
    clip = _wav(tmp_path / "wav" / "clip1.wav", sr=16000)
    features = tmp_path / "features"
    row = _audio_override_row(tmp_path, clip, "--cache-dir", str(features),
                              "--jsonl-audio-spec-path", "/override/base")
    for key in ("audio_norm_path", "audio_spec_path"):
        path = Path(row[key])
        assert path.parent == Path("/override/base/features"), row[key]
        assert (features / path.name).is_file(), row[key]


def test_jsonl_audio_spec_path_keeps_the_cache_folder_of_the_default_cache_dir(tmp_path):
    clip = _wav(tmp_path / "wav" / "clip1.wav", sr=16000)
    row = _audio_override_row(tmp_path, clip, "--jsonl-audio-spec-path", "/override/base")
    for key in ("audio_norm_path", "audio_spec_path"):
        path = Path(row[key])
        assert path.parent == Path("/override/base/cache/22050"), row[key]
        assert (tmp_path / "out" / "cache" / "22050" / path.name).is_file(), row[key]


# --- preprocess calls a deprecated phoonnx API -----------------------------

@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="preprocess-deprecated-api: build_frontend calls the deprecated "
                          "phoonnx.config.get_phonemizer")
def test_preprocess_frontend_uses_no_deprecated_phoonnx_api():
    from phoonnx.config import Alphabet, PhonemeType
    from phoonnx_train import preprocess

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        preprocess.build_frontend(PhonemeType.GRAPHEMES, Alphabet.IPA, None,
                                  keep_punctuation=True, language="en", add_diacritics=False)
    moved = [str(w.message) for w in caught
             if issubclass(w.category, DeprecationWarning) and "phoonnx.config" in str(w.message)]
    assert not moved, moved


# --- scripts/bench_latency.py with a voice id ------------------------------

@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="bench-latency-config: resolve_model returns <dir>/model.json, which no index voice has")
def test_bench_latency_resolves_the_config_of_an_index_voice(tmp_path):
    spec = importlib.util.spec_from_file_location("bench_latency", REPO / "scripts" / "bench_latency.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.onnx").write_bytes(b"")
    (snapshot / "config.json").write_text(json.dumps({"phoneme_type": "espeak"}), encoding="utf-8")
    config_url = "https://huggingface.co/org/voice/resolve/rev/config.json"

    info = MagicMock(config_url=config_url)
    info.download_model.return_value = snapshot / "model.onnx"
    info.hub_path.side_effect = lambda url: snapshot / url.rsplit("/", 1)[-1] if url else None
    info.download_config.return_value = {"phoneme_type": "espeak"}
    manager = MagicMock(voices={"org/voice": info})
    manager.download_voice_by_id.return_value = True
    with patch("phoonnx.model_manager.TTSModelManager", return_value=manager):
        _, config_path = bench.resolve_model("org/voice", None)
    assert Path(config_path).is_file(), config_path


# --- log-domain duration outputs read as frame counts ----------------------

@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="log-duration-outputs: Matcha reads a 'logw' output as frame counts")
def test_matcha_does_not_read_logw_as_frame_counts():
    from phoonnx.engines.matcha import MatchaAdapter

    frames = np.array([[2.0, 3.0, 5.0]], np.float32)
    found = MatchaAdapter()._find_duration_output([np.log(frames)], ["logw"])
    assert found is None or np.allclose(found, frames), found


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="log-duration-outputs: Mixer-TTS reads a 'log_durations' output as frame counts")
def test_mixertts_does_not_read_log_durations_as_frame_counts():
    from phoonnx.engines.mixertts import MixerTTSAdapter

    frames = np.array([[2.0, 3.0, 5.0]], np.float32)
    found = MixerTTSAdapter()._find_duration_output([np.log(frames)], ["log_durations"])
    assert found is None or np.allclose(found, frames), found
