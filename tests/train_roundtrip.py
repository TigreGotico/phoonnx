"""Shared pieces of the train, export and run round trip (tests/test_train_export_run.py).

A toy dataset of a few short synthetic clips goes through phoonnx_train's own
preprocess with the grapheme phonemizer, so the ids training reads and the ids
synthesis feeds the graph both come from one encoder: the voice frontend that
preprocess and ``TTSVoice`` share.
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# One sentence each, so ``TTSVoice.synthesize`` makes one chunk per text.
TEXTS = (
    "hello there.",
    "a cat sat on the mat.",
    "go home now!",
    "one more line, then stop.",
)
SPOKEN = TEXTS[1]


def write_clips(folder: Path, sample_rate: int) -> Path:
    """Write one tone-plus-noise wav per text and a jsonl that lists them."""
    import soundfile as sf

    rng = np.random.default_rng(0)
    rows = []
    for i, text in enumerate(TEXTS):
        n = int(sample_rate * (0.6 + 0.05 * len(text)))
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
    """Synthesize ``text``; return the audio and each integer id row that any graph read.

    The recorder sits on ``onnxruntime.InferenceSession.run``, so it sees the ids
    of an engine that reads its text in an auxiliary graph too.
    """
    import onnxruntime

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
        chunks = list(voice.synthesize(text, syn_config))
    finally:
        onnxruntime.InferenceSession.run = run
    assert len(chunks) == 1, f"{text!r} gave {len(chunks)} chunks"
    return chunks[0].audio_float_array, fed


def assert_round_trip(training_ids: Sequence[int], fed: List[List[int]], audio: np.ndarray) -> None:
    """The graph read the ids that training read, and the audio is not silence."""
    assert list(training_ids) in fed, (
        f"training read {list(training_ids)}; the graph read {fed}")
    audio = np.asarray(audio, dtype=np.float32)
    assert audio.size > 0, "the voice gave no audio"
    assert np.isfinite(audio).all(), "the voice gave non-finite audio"
    assert np.abs(audio).max() > 1e-4, "the voice gave silence"
