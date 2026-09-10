"""Torch-free helpers for Vocos vocoder training.

Everything in this module is importable without torch so the CLI surface,
mel configuration and dataset bookkeeping can be unit-tested in
environments where torch is not installed.
"""
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3")


@dataclass(frozen=True)
class MelConfig:
    """Mel-spectrogram settings for vocoder training.

    These MUST match the settings the acoustic model was trained with
    (``phoonnx_train.vits.mel_processing`` / the Matcha training engine),
    otherwise the vocoder is trained on features the acoustic model will
    never emit.  Do not change the defaults independently of those modules.
    """

    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0

    def as_dict(self) -> Dict:
        return asdict(self)

    def frames_for_samples(self, num_samples: int) -> int:
        """Number of mel frames produced for ``num_samples`` of audio.

        The phoonnx mel pipeline uses ``center=False`` with an explicit
        reflect pad of ``(n_fft - hop_length) // 2`` on each side, which
        yields exactly ``num_samples // hop_length`` frames when
        ``num_samples`` is a multiple of ``hop_length``.
        """
        padded = num_samples + (self.n_fft - self.hop_length)
        return (padded - self.n_fft) // self.hop_length + 1


def crop_samples(crop_seconds: float, mel: MelConfig) -> int:
    """Length (in samples) of a training crop, rounded down to a whole
    number of hops so waveform and mel frames stay aligned."""
    if crop_seconds <= 0:
        raise ValueError(f"crop_seconds must be > 0, got {crop_seconds}")
    samples = int(crop_seconds * mel.sample_rate)
    samples = (samples // mel.hop_length) * mel.hop_length
    if samples < mel.n_fft:
        raise ValueError(
            f"crop of {crop_seconds}s at {mel.sample_rate}Hz is shorter than "
            f"one FFT window ({mel.n_fft} samples)"
        )
    return samples


def find_audio_files(
    audio_dirs: Sequence[Union[str, Path]],
    extensions: Sequence[str] = AUDIO_EXTENSIONS,
) -> List[Path]:
    """Recursively collect audio files from one or more directories.

    Directories are searched recursively; results are deduplicated and
    sorted for deterministic dataset ordering.  A phoonnx preprocess output
    directory works as-is — its ``cache/<rate>`` folder holds the
    normalized wav files.
    """
    if not audio_dirs:
        raise ValueError("at least one audio directory is required")
    exts = {e.lower() for e in extensions}
    files = set()
    for d in audio_dirs:
        d = Path(d)
        if not d.is_dir():
            raise FileNotFoundError(f"audio directory not found: {d}")
        for root, _dirs, names in os.walk(d):
            for name in names:
                if os.path.splitext(name)[1].lower() in exts:
                    files.add(Path(root) / name)
    return sorted(files)


def parse_warm_start(source: str) -> Tuple[str, str]:
    """Classify a ``--warm-start`` value as a local file or a HF repo id.

    Returns ``("local", path)`` when the value points at an existing file,
    ``("hf", repo_id)`` when it looks like a HuggingFace ``org/name`` repo
    id.  Anything else raises ``ValueError`` — a mistyped path must fail
    loudly rather than trigger a surprise network download.
    """
    if not source or not source.strip():
        raise ValueError("empty warm-start source")
    source = source.strip()
    if os.path.isfile(source):
        return "local", source
    looks_like_path = (
        source.startswith((".", "/", "~"))
        or os.path.splitext(source)[1] in (".ckpt", ".bin", ".pt", ".pth", ".safetensors")
    )
    parts = source.split("/")
    if not looks_like_path and len(parts) == 2 and all(p.strip() for p in parts):
        return "hf", source
    raise ValueError(
        f"warm-start source {source!r} is neither an existing file nor a "
        f"HuggingFace repo id (expected 'org/name')"
    )
