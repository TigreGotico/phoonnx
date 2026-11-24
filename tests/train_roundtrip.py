"""Shared pieces of the train, export and run round trip (tests/test_train_export_run.py).

A toy dataset of a few short synthetic clips goes through phoonnx_train's own
preprocess with the grapheme phonemizer, so the ids training reads and the ids
synthesis feeds the graph both come from one encoder: the voice frontend that
preprocess and ``TTSVoice`` share.
"""
import json
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# One sentence each, so ``TTSVoice.synthesize`` makes one chunk per text.
TEXTS = (
    "hello there.",
    "a cat sat on the mat.",
    "go home now!",
    "one more line, then stop.",
)
SPOKEN = TEXTS[1]


def clip_seconds(text: str) -> float:
    """The length of the toy clip of ``text``."""
    return 0.6 + 0.05 * len(text)


def write_clips(folder: Path, sample_rate: int) -> Path:
    """Write one tone-plus-noise wav per text and a jsonl that lists them."""
    import soundfile as sf

    rng = np.random.default_rng(0)
    rows = []
    for i, text in enumerate(TEXTS):
        n = int(sample_rate * clip_seconds(text))
        t = np.arange(n) / sample_rate
        audio = 0.3 * np.sin(2 * np.pi * (150 + 40 * i) * t) + 0.02 * rng.standard_normal(n)
        wav = folder / f"u{i}.wav"
        sf.write(wav, audio.astype(np.float32), sample_rate)
        rows.append(json.dumps({"text": text, "audio": str(wav)}))
    source = folder / "src.jsonl"
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return source


def preprocess(folder: Path, sample_rate: int, engine: Optional[str] = None) -> Path:
    """Run phoonnx_train's preprocess on the toy clips and return the dataset folder.

    It runs in a child process: preprocess forks its workers, and a fork from a
    process whose torch thread pool is already running can hang.
    """
    source = write_clips(folder, sample_rate)
    data = folder / "data"
    args = [sys.executable, "-m", "phoonnx_train.preprocess", "-i", str(source), "-o", str(data),
            "-l", "en", "--phoneme-type", "graphemes", "--alphabet", "graphemes",
            "-r", str(sample_rate), "--max-workers", "1"]
    if engine:
        args += ["--engine", engine]
    result = subprocess.run(args, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len((data / "dataset.jsonl").read_text(encoding="utf-8").splitlines()) == len(TEXTS)
    return data


def dataset_ids(data: Path, text: str = SPOKEN) -> List[int]:
    """The ids that preprocess wrote for ``text``."""
    rows = [json.loads(line) for line in (data / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    return next(row["phoneme_ids"] for row in rows if row["text"] == text)


def dataset_config(data: Path) -> Dict[str, Any]:
    return json.loads((data / "config.json").read_text(encoding="utf-8"))


def export_config(folder: Path) -> Path:
    """The one phoonnx voice config that an export wrote beside its graph."""
    configs = [p for p in folder.glob("*.json") if not p.name.endswith(".piper.json")]
    assert len(configs) == 1, f"the export wrote {configs}"
    return configs[0]


def fit(module, steps: int, root: Path) -> Path:
    """Train ``module`` for ``steps`` optimizer steps on CPU and save a checkpoint."""
    import pytorch_lightning as pl
    import torch

    from phoonnx_train.torch_compat import trusting_torch_load

    torch.manual_seed(0)
    trainer = pl.Trainer(max_steps=steps, accelerator="cpu", devices=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, limit_val_batches=0,
                         num_sanity_val_steps=0, default_root_dir=root)
    with trusting_torch_load():
        trainer.fit(module)
    assert trainer.global_step >= 1, "the trainer took no step"
    checkpoint = root / "model.ckpt"
    trainer.save_checkpoint(checkpoint)
    return checkpoint


def speak(voice, text: str = SPOKEN, syn_config=None) -> Tuple[np.ndarray, List[List[int]]]:
    """Render ``text`` as one sentence with no peak normalisation; return the raw
    audio of the graph and each integer id row that any graph read.

    The recorder sits on ``onnxruntime.InferenceSession.run``, so it sees the ids
    of an engine that reads its text in an auxiliary graph too.
    """
    import onnxruntime

    from phoonnx.config import SynthesisConfig

    config = replace(syn_config or SynthesisConfig(), normalize_audio=False)
    fed: List[List[int]] = []
    run = onnxruntime.InferenceSession.run

    def recording_run(session, output_names, feed, *args, **kwargs):
        for value in feed.values():
            value = np.asarray(value)
            if value.ndim in (1, 2) and value.size > 1 and np.issubdtype(value.dtype, np.integer):
                fed.append(value.reshape(-1, value.shape[-1])[0].tolist())
        return run(session, output_names, feed, *args, **kwargs)

    onnxruntime.InferenceSession.run = recording_run
    try:
        utterances = list(voice.prepare(text, config))
        assert len(utterances) == 1, f"{text!r} gave {len(utterances)} utterances"
        audio = voice.render(utterances[0], config).audio_float_array
    finally:
        onnxruntime.InferenceSession.run = run
    return np.asarray(audio, dtype=np.float32), fed


# The audio floors. A tiny model trained a few steps makes loud noise. The quietest
# passing engine peaks at 0.52 (VITS) and has an RMS of 0.09 (Matcha), after the
# clip to [-1, 1]. The floors sit 10 and 17 times lower, at -26 and -46 dBFS. A
# silent mel through Griffin-Lim peaks near 5e-7, under the floors. Output scaled
# by 1e-6 falls under the floors only when the raw output peaks under 5e4 before
# the clip. Each test that keeps the floors trains its engine until it does.
PEAK_FLOOR = 0.05
RMS_FLOOR = 0.005


def assert_round_trip(training_ids: Sequence[int], fed: List[List[int]], audio: np.ndarray,
                      min_samples: int, level: bool = True) -> None:
    """The graph read the ids that training read, and the raw audio is long and finite:
    at least ``min_samples`` samples. With ``level``, it is also not near silence."""
    assert list(training_ids) in fed, (
        f"training read {list(training_ids)}; the graph read {fed}")
    assert audio.size >= min_samples, f"the voice gave {audio.size} samples, under {min_samples}"
    assert np.isfinite(audio).all(), "the voice gave non-finite audio"
    if not level:
        return
    peak, rms = float(np.abs(audio).max()), float(np.sqrt(np.mean(audio ** 2)))
    assert peak >= PEAK_FLOOR and rms >= RMS_FLOOR, (
        f"the voice gave near silence: peak {peak:.3g}, RMS {rms:.3g}")


def mel_profile_correlation(mel: np.ndarray, vocoder: Any, wav: Path) -> float:
    """Correlate the time-mean spectrum of ``mel`` with that of the clip ``wav``.

    The clip is analysed with the vocoder's own mel basis and the natural log of the
    magnitude, the scale on which the vocoder reads ``mel``. A constant mel gives 0.
    """
    import librosa
    import soundfile as sf

    audio, rate = sf.read(wav)
    audio = librosa.resample(audio, orig_sr=rate, target_sr=vocoder.sample_rate)
    spec = np.abs(librosa.stft(audio, n_fft=vocoder.n_fft, hop_length=vocoder.hop_length,
                               win_length=vocoder.win_length))
    target = np.log(np.maximum(1e-5, vocoder.mel_basis @ spec)).mean(axis=1)
    profile = np.asarray(mel, dtype=np.float64).reshape(vocoder.num_mels, -1).mean(axis=1)
    if profile.std() == 0:
        return 0.0
    return float(np.corrcoef(profile, target)[0, 1])


class KnownDefect(Exception):
    """The failure that a strict xfail names, told apart from any other failure."""


@contextmanager
def known_defect(error: type, pattern: str) -> Iterator[None]:
    """Raise :class:`KnownDefect` for an ``error`` whose message matches ``pattern``.

    A strict xfail with ``raises=KnownDefect`` then holds only for its own defect:
    any other error fails the test, and a fix makes it pass, which strict mode
    reports as a failure until the mark goes.
    """
    try:
        yield
    except error as exc:
        if re.search(pattern, str(exc)):
            raise KnownDefect(str(exc)) from exc
        raise
