"""Known defects, each pinned as a strict xfail until its fix lands.

Each test states the behaviour the code should have. It fails on the current
code for the reason in its ``reason``, which names the issue that tracks it.
The fix PR removes the mark; a fix that lands without that edit turns the
strict xfail into a failure, so the pin cannot go stale.
"""
import hashlib
import importlib.util
import json
import logging
import multiprocessing
import re
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


def test_failed_alignment_export_exits_non_zero(tmp_path):
    result = _export_with_alignment(tmp_path)
    assert result.exit_code == 1, result.output
    assert "No Ceil tensor found" in result.output, result.output


def test_alignment_errors_name_only_real_flags(tmp_path, caplog):
    from phoonnx_train.export_onnx import cli

    with caplog.at_level(logging.DEBUG, logger="phoonnx_train.export_onnx"):
        result = _export_with_alignment(tmp_path)
    flags = {opt for param in cli.params for opt in param.opts}
    named = set(re.findall(r"--[a-z][a-z-]*", caplog.text + result.output))
    assert named <= flags, (named - flags, result.output, caplog.text)


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

    with patch.object(preprocess, "_voice_phonemizer", return_value=Phonemizer()):
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


def test_jsonl_audio_path_keeps_one_base_across_a_resume(tmp_path):
    def row(name: str) -> str:
        clip = _wav(tmp_path / "ds" / "wav" / name, sr=16000)
        return json.dumps({"text": "hi", "audio": str(clip), "phon": "h i"}) + "\n"

    source = tmp_path / "a.jsonl"
    source.write_text(row("spk1/a.wav") + row("spk2/b.wav"))
    out = tmp_path / "out"
    args = ["-i", str(source), "-o", str(out), "-l", "en", "--phonemes-column", "phon",
            "--jsonl-audio-path", "/override/base"]
    assert _preprocess(args).exit_code == 0
    with open(source, "a") as handle:
        handle.write(row("spk2/c.wav"))
    result = _preprocess([*args, "--resume"])
    assert result.exit_code == 0, result.output
    assert [r["audio_path"] for r in _manifest(out)] == [
        "/override/base/wav/spk1/a.wav", "/override/base/wav/spk2/b.wav", "/override/base/wav/spk2/c.wav"]


def test_jsonl_audio_path_keeps_the_first_base_when_a_resume_adds_a_folder(tmp_path):
    def row(name: str) -> str:
        clip = _wav(tmp_path / "ds" / "wav" / name, sr=16000)
        return json.dumps({"text": "hi", "audio": str(clip), "phon": "h i"}) + "\n"

    source = tmp_path / "a.jsonl"
    source.write_text(row("spk1/a.wav") + row("spk1/b.wav"))
    out = tmp_path / "out"
    args = ["-i", str(source), "-o", str(out), "-l", "en", "--phonemes-column", "phon",
            "--jsonl-audio-path", "/override/base"]
    assert _preprocess(args).exit_code == 0
    with open(source, "a") as handle:
        handle.write(row("spk2/c.wav"))
    result = _preprocess([*args, "--resume"])
    assert result.exit_code == 0, result.output
    assert [r["audio_path"] for r in _manifest(out)] == [
        "/override/base/spk1/a.wav", "/override/base/spk1/b.wav", "/override/base/spk2/c.wav"]


def test_jsonl_audio_path_refuses_a_resume_outside_the_first_base(tmp_path):
    first = tmp_path / "a.jsonl"
    first.write_text(json.dumps({"text": "hi", "audio": str(_wav(tmp_path / "ds1" / "wav" / "a.wav")),
                                 "phon": "h i"}) + "\n")
    second = tmp_path / "b.jsonl"
    second.write_text(json.dumps({"text": "hi", "audio": str(_wav(tmp_path / "ds2" / "wav" / "b.wav")),
                                  "phon": "h i"}) + "\n")
    out = tmp_path / "out"
    args = ["-o", str(out), "-l", "en", "--phonemes-column", "phon", "--jsonl-audio-path", "/override/base"]
    assert _preprocess(["-i", str(first), *args]).exit_code == 0
    result = _preprocess(["-i", str(first), "-i", str(second), *args, "--resume"])
    assert result.exit_code == 2, result.output
    assert f"its rows use {tmp_path / 'ds1'}, and the new rows need {tmp_path.parent}." in result.output, \
        result.output
    assert [r["audio_path"] for r in _manifest(out)] == ["/override/base/wav/a.wav"]


def test_a_resume_without_jsonl_audio_path_refuses_rows_written_with_it(tmp_path):
    def row(name: str) -> str:
        clip = _wav(tmp_path / "ds" / "wav" / name, sr=16000)
        return json.dumps({"text": "hi", "audio": str(clip), "phon": "h i"}) + "\n"

    source = tmp_path / "a.jsonl"
    source.write_text(row("a.wav"))
    out = tmp_path / "out"
    args = ["-i", str(source), "-o", str(out), "-l", "en", "--phonemes-column", "phon"]
    assert _preprocess([*args, "--jsonl-audio-path", "/override/base"]).exit_code == 0
    with open(source, "a") as handle:
        handle.write(row("b.wav"))
    result = _preprocess([*args, "--resume"])
    assert result.exit_code == 2, result.output
    assert "Pass the --jsonl-audio-path of the first run again." in result.output, result.output
    assert [r["audio_path"] for r in _manifest(out)] == ["/override/base/wav/a.wav"]


def _run_preprocess_over_embedded_audio(args, rows: int):
    import io

    from phoonnx_train import dataset_loaders
    from phoonnx_train.dataset_loaders import Utterance

    def loader(source, config):
        for index in range(rows):
            audio = io.BytesIO()
            sf.write(audio, np.zeros(8000, np.float32) + 0.1, 16000, format="WAV")
            yield Utterance(text="hi", audio_path=Path(""), row_id=f"{source}#{index}",
                            audio_bytes=audio.getvalue())

    dataset_loaders.register_loader("hf", loader)
    return _run_preprocess(args)


def test_a_resume_over_embedded_audio_without_overrides_keeps_going(tmp_path):
    # embedded rows keep a local path under <cache_dir>/audio that differs from their row_id
    out = tmp_path / "out"
    args = ["-i", "org/name", "-o", str(out), "-l", "en"]
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert pool.submit(_run_preprocess_over_embedded_audio, args, 1).result().exit_code == 0
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        result = pool.submit(_run_preprocess_over_embedded_audio, [*args, "--resume"], 2).result()
    assert result.exit_code == 0, result.output
    assert [Path(r["audio_path"]).parent for r in _manifest(out)] == [out / "cache" / "22050" / "audio"] * 2


def test_jsonl_audio_path_base_of_a_dataset_directory_is_that_directory(tmp_path):
    dataset = tmp_path / "ds"
    _wav(dataset / "wav" / "spk1" / "a.wav", sr=16000)
    _wav(dataset / "wav" / "spk1" / "b.wav", sr=16000)
    (dataset / "metadata.csv").write_text("spk1/a|hi\nspk1/b|hi there\n", encoding="utf-8")
    out = tmp_path / "out"
    result = _preprocess(["-i", str(dataset), "-o", str(out), "-l", "en",
                          "--jsonl-audio-path", "/override/base"])
    assert result.exit_code == 0, result.output
    assert sorted(r["audio_path"] for r in _manifest(out)) == [
        "/override/base/wav/spk1/a.wav", "/override/base/wav/spk1/b.wav"]


def _run_preprocess_recording_engine_paths(args):
    class Engine:
        def __init__(self):
            self.paths = []

        def extra_preprocess(self, utterance_audio_path, cache_dir, sample_rate, **kwargs):
            self.paths.append(str(utterance_audio_path))
            return {}

    engine = Engine()
    with patch("phoonnx_train.engines.get_engine", return_value=engine):
        result = _run_preprocess(args)
    return result, engine.paths


def test_jsonl_audio_path_leaves_the_engine_step_on_the_local_audio(tmp_path):
    clip = _wav(tmp_path / "wav" / "clip1.wav", sr=16000)
    source = tmp_path / "a.jsonl"
    source.write_text(json.dumps({"text": "hi", "audio": str(clip), "phon": "h i"}) + "\n")
    out = tmp_path / "out"
    args = ["-i", str(source), "-o", str(out), "-l", "en", "--phonemes-column", "phon",
            "--engine", "recorder", "--jsonl-audio-path", "/override/base"]
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        result, paths = pool.submit(_run_preprocess_recording_engine_paths, args).result()
    assert result.exit_code == 0, result.output
    assert paths == [str(clip)]
    assert _manifest(out)[0]["audio_path"] == "/override/base/wav/clip1.wav"


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


def _run_preprocess_with_cache_writing_engine(args):
    class Engine:
        # the YourTTS d-vector and FastPitch F0 cache layouts, without their models
        def extra_preprocess(self, utterance_audio_path, cache_dir, sample_rate, **kwargs):
            name = hashlib.sha1(str(utterance_audio_path).encode("utf-8")).hexdigest()
            (cache_dir / "dvec").mkdir(parents=True, exist_ok=True)
            (cache_dir / "dvec" / f"{name}.pt").write_bytes(b"")
            (cache_dir / f"{name}.f0.npy").write_bytes(b"")
            return {"d_vector_path": str(cache_dir / "dvec" / f"{name}.pt"),
                    "f0_path": str(cache_dir / f"{name}.f0.npy")}

    with patch("phoonnx_train.engines.get_engine", return_value=Engine()):
        return _run_preprocess(args)


def test_jsonl_audio_spec_path_rewrites_the_cache_files_of_the_engine_step(tmp_path):
    clip = _wav(tmp_path / "wav" / "clip1.wav", sr=16000)
    source = tmp_path / "a.jsonl"
    source.write_text(json.dumps({"text": "hi", "audio": str(clip), "phon": "h i"}) + "\n")
    out = tmp_path / "out"
    args = ["-i", str(source), "-o", str(out), "-l", "en", "--phonemes-column", "phon",
            "--engine", "cache-writer", "--jsonl-audio-spec-path", "/override/base"]
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        result = pool.submit(_run_preprocess_with_cache_writing_engine, args).result()
    assert result.exit_code == 0, result.output
    row = _manifest(out)[0]
    name = hashlib.sha1(str(clip).encode("utf-8")).hexdigest()
    assert row["d_vector_path"] == f"/override/base/cache/22050/dvec/{name}.pt", row["d_vector_path"]
    assert row["f0_path"] == f"/override/base/cache/22050/{name}.f0.npy", row["f0_path"]
    assert (out / "cache" / "22050" / "dvec" / f"{name}.pt").is_file()
    assert row["audio_path"] == str(clip), row["audio_path"]


# --- preprocess calls a deprecated phoonnx API -----------------------------

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


def _phoonnx_deprecations_of_a_run(args) -> list:
    from phoonnx_train import preprocess

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = click.testing.CliRunner().invoke(
            preprocess.cli, [*args, "--max-workers", "1"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return [str(w.message) for w in caught
            if issubclass(w.category, DeprecationWarning) and "phoonnx.config" in str(w.message)]


def test_a_preprocess_run_uses_no_deprecated_phoonnx_api(tmp_path):
    # no stub phonemizer, so the parent builds its own phonemizer and a worker frontend
    source = tmp_path / "a.jsonl"
    source.write_text(json.dumps({"text": "hi", "audio": "a.wav"}) + "\n")
    args = ["-i", str(source), "-o", str(tmp_path / "out"), "-l", "en", "--skip-audio",
            "--phoneme-type", "graphemes", "--alphabet", "unicode"]
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
        moved = pool.submit(_phoonnx_deprecations_of_a_run, args).result()
    assert not moved, moved


# --- scripts/bench_latency.py with a voice id ------------------------------

def _bench_latency():
    spec = importlib.util.spec_from_file_location("bench_latency", REPO / "scripts" / "bench_latency.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    return bench


def test_bench_latency_resolves_the_config_of_an_index_voice(tmp_path):
    bench = _bench_latency()

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
    assert config_path == str(snapshot / "config.json"), config_path


def test_bench_latency_says_when_it_uses_the_fallback_config(tmp_path, capsys):
    bench = _bench_latency()
    model = tmp_path / "voice.onnx"
    with patch("phoonnx.providers.make_session"), patch("phoonnx.engines.detect_engine"), \
            patch.object(bench, "TTSVoice"):
        bench.load_voice_timed(str(model), None)
    assert f"No config at {model}.json" in capsys.readouterr().err


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


# --- SynthesisConfig speaker and language names ------------------------------

def _voice(engine: str, **config):
    from phoonnx.config import VoiceConfig
    from phoonnx.engines import get_adapter
    from phoonnx.voice import TTSVoice

    letters = "abcdefghijklmnopqrstuvwxyz"
    config = {"phoonnx_version": "1.0", "engine": "phoonnx", "phoneme_type": "unicode",
              "alphabet": "unicode", "lang_code": "en-US", "audio": {"sample_rate": 16000},
              "phoneme_id_map": {"_": [0], "^": [1], "$": [2], **{c: [i + 3] for i, c in enumerate(letters)}},
              **config}
    names = ("input", "input_lengths", "scales", "sid", "lid")
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name=n) for n in names],
                              get_outputs=lambda: [SimpleNamespace(name="output")],
                              run=lambda outputs, feed: [np.zeros((1, 1, 64), np.float32)])
    return TTSVoice(session=session, config=VoiceConfig.from_dict(config), adapter=get_adapter(engine))


def test_an_unknown_language_code_names_the_codes_of_the_voice():
    from phoonnx.config import SynthesisConfig

    voice = _voice("vits", lang_id_map={"gl": 0, "pt": 1})
    with pytest.raises(ValueError) as caught:
        list(voice.synthesize("abc", SynthesisConfig(language="Jason")))
    assert "gl" in str(caught.value) and "pt" in str(caught.value), caught.value


def test_an_unknown_speaker_name_names_the_speakers_of_the_voice():
    from phoonnx.config import SynthesisConfig

    voice = _voice("glowtts", num_speakers=2, speaker_id_map={"alice": 0, "bob": 1})
    with pytest.raises(ValueError) as caught:
        list(voice.synthesize("abc", SynthesisConfig(speaker="Jason")))
    assert "alice" in str(caught.value) and "bob" in str(caught.value), caught.value


def test_a_json_round_trip_keeps_the_speaker_of_a_1x_speaker_id():
    from phoonnx.config import SynthesisConfig

    voice = _voice("vits", num_speakers=50, speaker_id_map={"40": 3})
    with pytest.warns(DeprecationWarning):
        config = SynthesisConfig(speaker_id="40")
    reloaded = SynthesisConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    with patch.object(voice.adapter, "synthesize", wraps=voice.adapter.synthesize) as engine:
        list(voice.synthesize("abc", config))
        list(voice.synthesize("abc", reloaded))
    original, round_tripped = (call.args[0].speaker_id for call in engine.call_args_list)
    assert round_tripped == original, (original, round_tripped)


def test_a_json_round_trip_keeps_the_language_of_a_1x_lang_id():
    from phoonnx.config import SynthesisConfig

    voice = _voice("vits", lang_id_map={"gl": 0, "1": 2, "pt": 1})
    with pytest.warns(DeprecationWarning):
        config = SynthesisConfig(lang_id="1")
    reloaded = SynthesisConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    with patch.object(voice.adapter, "synthesize", wraps=voice.adapter.synthesize) as engine:
        list(voice.synthesize("abc", config))
        list(voice.synthesize("abc", reloaded))
    original, round_tripped = (call.args[0].language_id for call in engine.call_args_list)
    assert round_tripped == original, (original, round_tripped)


# --- Matcha and ZipVoice vocoder_type ----------------------------------------

def test_matcha_builds_the_vocoder_of_a_vocoder_type():
    from phoonnx.engines.matcha import MatchaAdapter
    from phoonnx.engines.vocoders.griffinlim import GriffinLimVocoder

    adapter = MatchaAdapter()
    adapter.configure_from_params({"vocoder_type": "griffinlim"})
    assert isinstance(adapter.vocoder, GriffinLimVocoder), adapter.vocoder


def test_zipvoice_builds_the_vocoder_of_a_vocoder_type():
    from phoonnx.engines.vocoders.griffinlim import GriffinLimVocoder
    from phoonnx.engines.zipvoice import ZipVoiceAdapter

    adapter = ZipVoiceAdapter()
    adapter.configure(SimpleNamespace(engine_params={"vocoder_type": "griffinlim"}))
    assert isinstance(adapter.vocoder, GriffinLimVocoder), adapter.vocoder


# --- Mixer-TTS durations that round to 0 frames ------------------------------

def test_mixertts_gives_each_token_one_frame_at_least():
    import torch

    from phoonnx_train.mixertts.models.mixer_tts.mixer_tts import MixerTTSModel

    torch.manual_seed(0)
    model = MixerTTSModel(num_tokens=16, symbols_embedding_dim=16, encoder_kernel_sizes=[3],
                          encoder_num_layers=1, decoder_kernel_sizes=[15], decoder_num_layers=1,
                          duration_predictor_filter_size=16, pitch_predictor_filter_size=16,
                          energy_predictor_filter_size=16, n_speakers=1, n_emotions=1).eval()
    # a log duration of -10 is exp(-10) - 1 frames: every token rounds to 0
    model.duration_predictor.forward = lambda enc_out, enc_mask: torch.full(enc_out.shape[:2], -10.0)
    tokens = torch.tensor([[5, 6, 7, 8]])
    with torch.no_grad():
        spect = model.infer(text=tokens)
    assert spect.shape[1] >= tokens.shape[1], spect.shape


@pytest.mark.parametrize("log_dur, pace, frames", [
    (-10.0, 3.0, [4, 2]),  # 0 frames each: the floor gives a real token 1 frame
    (float(np.log(4.0)), 1.0, [12, 6]),  # 3 frames each: the floor changes nothing
])
def test_mixertts_frames_per_token_at_a_pace(log_dur, pace, frames):
    import torch

    from phoonnx_train.mixertts.models.mixer_tts import mixer_tts as mixer_tts_module

    torch.manual_seed(0)
    model = mixer_tts_module.MixerTTSModel(
        num_tokens=16, symbols_embedding_dim=16, encoder_kernel_sizes=[3],
        encoder_num_layers=1, decoder_kernel_sizes=[15], decoder_num_layers=1,
        duration_predictor_filter_size=16, pitch_predictor_filter_size=16,
        energy_predictor_filter_size=16, n_speakers=1, n_emotions=1).eval()
    # the predictor masks its output, so a padding token predicts 0 frames
    model.duration_predictor.forward = lambda enc_out, enc_mask: (
        torch.full(enc_out.shape[:2], log_dur) * enc_mask.squeeze(2))
    # the second row has 2 padding tokens (id 0), which get no frame
    tokens = torch.tensor([[5, 6, 7, 8], [5, 6, 0, 0]])
    captured = {}
    regulate = mixer_tts_module.regulate_len

    def spy(durations, enc_out, pace=1.0, **kwargs):
        out, dec_lens = regulate(durations, enc_out, pace=pace, **kwargs)
        captured["dec_lens"] = dec_lens.tolist()
        return out, dec_lens

    with patch.object(mixer_tts_module, "regulate_len", spy), torch.no_grad():
        model.infer(text=tokens, pace=pace)
    assert captured["dec_lens"] == frames


# --- OptiSpeech export phoneme map -------------------------------------------

TINY_OPTISPEECH = dict(
    dim=32, n_feats=40, hop_length=256, n_fft=256, win_length=256,
    encoder_intermediate_dim=48, decoder_intermediate_dim=48, encoder_num_layers=1,
    decoder_num_layers=1, vocoder_dim=32, vocoder_intermediate_dim=48, vocoder_num_layers=1,
    duration_intermediate_dim=32, pitch_intermediate_dim=32, energy_intermediate_dim=32,
    pitch_num_layers=1)


def test_an_optispeech_voice_feeds_the_ids_that_training_reads(tmp_path):
    import torch

    from phoonnx.voice import TTSVoice
    from phoonnx_train.engines import get_engine
    from phoonnx_train.engines.base import TrainingEngineConfig
    from tests import train_roundtrip as rt

    symbols = ["_", "^", "$", " ", ".", *"abcdefghijklmnopqrstuvwxyz"]
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "lang_code": "en", "phoneme_type": "graphemes", "alphabet": "graphemes",
        "audio": {"sample_rate": 22050}, "num_symbols": len(symbols),
        "phoneme_id_map": {s: [i] for i, s in enumerate(symbols)}}), encoding="utf-8")
    engine = get_engine("optispeech")
    torch.manual_seed(0)
    module = engine.create_model(TrainingEngineConfig(
        num_symbols=len(symbols), num_speakers=1, sample_rate=22050, extra=TINY_OPTISPEECH), [])
    # the untrained module stands in for a checkpoint: only the export is under test
    with patch.object(engine, "load_lightning_module", return_value=module):
        onnx_path = engine.export_onnx(tmp_path / "model.ckpt", config, tmp_path / "voice")
    # the ids that _JsonlTextWavDataset gives training for one row
    training_ids, _ = module.text_processor(rt.SPOKEN, None, split_sentences=False)
    voice = TTSVoice.load(onnx_path, config_path=onnx_path.with_suffix(".json"))
    _, fed = rt.speak(voice)
    assert list(training_ids) in fed, (list(training_ids), fed)


def test_an_optispeech_export_keeps_the_speaker_map_of_the_dataset(tmp_path):
    import torch

    from phoonnx_train.engines import get_engine
    from phoonnx_train.engines.base import TrainingEngineConfig

    speakers = {"alice": 0, "bob": 1}
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "lang_code": "en", "phoneme_type": "graphemes", "alphabet": "graphemes",
        "audio": {"sample_rate": 22050}, "num_symbols": 3, "phoneme_id_map": {"_": [0], "a": [1], "b": [2]},
        "num_speakers": 2, "speaker_id_map": speakers}), encoding="utf-8")
    engine = get_engine("optispeech")
    torch.manual_seed(0)
    module = engine.create_model(TrainingEngineConfig(
        num_symbols=3, num_speakers=2, sample_rate=22050, extra=TINY_OPTISPEECH), [])
    with patch.object(engine, "load_lightning_module", return_value=module):
        onnx_path = engine.export_onnx(tmp_path / "model.ckpt", config, tmp_path / "voice")
    written = json.loads(onnx_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert (written["num_speakers"], written["speaker_id_map"]) == (2, speakers)


# --- StyleTTS2 trailing pad and iSTFTNet export ------------------------------

@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#1032: StyleTTS2 training pads both ends, and the adapter pads the start only")
def test_styletts2_synthesis_pads_the_ids_as_training_does(tmp_path):
    from phoonnx.engines.base import AdapterSynthesisRequest
    from phoonnx.engines.styletts2 import StyleTTS2Adapter
    from phoonnx_train.styletts2.meldataset import FilePathDataset, TextCleaner

    text = "a cat sat."
    _wav(tmp_path / "u0.wav", sr=24000)
    training = FilePathDataset([f"u0.wav|{text}|0"], str(tmp_path), OOD_data=None)
    _, training_ids, _ = training._load_tensor(training.data_list[0])

    # a voice exported with the TextCleaner table gives the adapter these ids, with no pad
    ids = np.array([TextCleaner()(text)], np.int64)
    request = AdapterSynthesisRequest(phoneme_ids=ids, phoneme_lengths=np.array([ids.shape[1]]))
    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="tokens")])
    feed = StyleTTS2Adapter().build_feed_dict(request, session)
    assert feed["tokens"][0].tolist() == training_ids.tolist(), (feed["tokens"], training_ids)


@pytest.mark.xfail(strict=True, raises=RuntimeError,
                   reason="#1032: the TorchScript ONNX exporter cannot trace the complex tensors of "
                          "TorchSTFT.transform and TorchSTFT.inverse")
def test_the_istftnet_generator_exports_to_onnx(tmp_path):
    import torch

    from phoonnx_train.styletts2.Modules.istftnet import Generator
    from phoonnx_train.torch_compat import onnx_export_kwargs

    # the iSTFTNet layout of the low and medium presets, 32 channels wide in place of 512
    generator = Generator(style_dim=16, resblock_kernel_sizes=[3], upsample_rates=[10, 6],
                          upsample_initial_channel=32, resblock_dilation_sizes=[[1, 3, 5]],
                          upsample_kernel_sizes=[20, 12], gen_istft_n_fft=20, gen_istft_hop_size=5).eval()
    frames = 8
    args = (torch.randn(1, 32, frames), torch.randn(1, 16), torch.full((1, frames), 120.0))
    with torch.no_grad():
        torch.onnx.export(generator, args, str(tmp_path / "generator.onnx"),
                          input_names=["x", "s", "f0"], opset_version=17, **onnx_export_kwargs())
    assert (tmp_path / "generator.onnx").is_file()


# --- SuperTonic export of a model trained from zero --------------------------

@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#1033: the SuperTonic export writes no voice config and no voice style")
def test_a_supertonic_export_gives_a_voice_that_speaks(tmp_path):
    import torch

    from phoonnx.voice import TTSVoice
    from phoonnx_train.supertonic.autoencoder import SpeechAutoencoder
    from phoonnx_train.supertonic.config import tiny_config
    from phoonnx_train.supertonic.duration_predictor import DurationPredictor
    from phoonnx_train.supertonic.export_onnx import export_all
    from phoonnx_train.supertonic.text import CharTokenizer
    from phoonnx_train.supertonic.text_to_latent import TextToLatentModel
    from tests import train_roundtrip as rt

    tokenizer = CharTokenizer.build_from_texts(list(rt.TEXTS), ["en"])
    config = tiny_config(vocab_size=tokenizer.vocab_size)
    torch.manual_seed(0)
    paths = export_all(str(tmp_path), config=config, tokenizer=tokenizer,
                       autoencoder=SpeechAutoencoder(config.ae),
                       text_to_latent=TextToLatentModel(config.ttl, config.vocab_size),
                       duration_predictor=DurationPredictor(config.dp, config.vocab_size))
    voice_configs = [p for p in sorted(tmp_path.glob("*.json"))
                     if "lang_code" in json.loads(p.read_text(encoding="utf-8"))]
    assert voice_configs, f"the export wrote {sorted(p.name for p in tmp_path.iterdir())}"
    voice = TTSVoice.load(paths["vector_estimator"], config_path=voice_configs[0])
    audio, _ = rt.speak(voice)
    assert audio.size and np.isfinite(audio).all(), audio


# --- preprocess --resume and --jsonl-audio-spec-path -------------------------

@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#1067: a resume without --jsonl-audio-spec-path writes local cache paths beside "
                          "the rewritten ones")
def test_a_resume_without_jsonl_audio_spec_path_keeps_the_first_cache_base(tmp_path):
    def row(name: str) -> str:
        clip = _wav(tmp_path / "ds" / "wav" / name, sr=16000)
        return json.dumps({"text": "hi", "audio": str(clip), "phon": "h i"}) + "\n"

    source = tmp_path / "a.jsonl"
    source.write_text(row("a.wav"))
    out = tmp_path / "out"
    args = ["-i", str(source), "-o", str(out), "-l", "en", "--phonemes-column", "phon"]
    assert _preprocess([*args, "--jsonl-audio-spec-path", "/override/base"]).exit_code == 0
    with open(source, "a") as handle:
        handle.write(row("b.wav"))
    result = _preprocess([*args, "--resume"])
    if result.exit_code == 2:
        # a usage error is the other way the issue allows
        assert "--jsonl-audio-spec-path" in result.output, result.output
        assert len(_manifest(out)) == 1
    else:
        assert result.exit_code == 0, result.output
        folders = [Path(r[key]).parent for r in _manifest(out) for key in ("audio_norm_path", "audio_spec_path")]
        assert folders == [Path("/override/base/cache/22050")] * 4, folders
