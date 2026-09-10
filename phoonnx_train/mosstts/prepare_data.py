"""Pre-encode a dataset's audio with the frozen MOSS audio tokenizer.

The codec (``MOSS-Audio-Tokenizer-Nano``, RVQ-16 @ 12.5 fps, 48 kHz stereo) is **frozen**
and is never part of training. It runs exactly once, here, through the published ONNX
encode graph (``moss_audio_tokenizer_encode.onnx``) under onnxruntime — no torch codec, no
``trust_remote_code`` — and the resulting code matrices are written into the JSONL the
trainer consumes.

Input manifest, either

* a JSONL with ``{"audio": ..., "text": ..., "language": ..., "ref_audio": ...}`` records
  (upstream's ``prepare_data.py`` shape), or
* an LJSpeech-style ``metadata.csv`` (``<wav-stem>|<text>`` per line) plus ``--wav-dir``.

Output is the same JSONL with ``audio_codes`` (and ``ref_audio_codes`` when a reference
clip is given) attached::

    python -m phoonnx_train.mosstts.prepare_data \\
        --codec-encode-onnx models/moss_audio_tokenizer_encode.onnx \\
        --input-manifest data/train.jsonl \\
        --output-jsonl data/train.codes.jsonl
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from phoonnx_train.mosstts.dataset import dump_jsonl, load_jsonl

_LOGGER = logging.getLogger("mosstts.prepare_data")

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 2


def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Read *path* as ``[channels, samples]`` float32 plus its sample rate."""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise ImportError(
            "reading audio needs `soundfile` (pip install phoonnx[train])"
        ) from exc
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.T), int(sample_rate)


def resample(audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if sample_rate == target_sample_rate:
        return audio
    from math import gcd

    try:
        from scipy.signal import resample_poly
    except ImportError as exc:  # pragma: no cover
        raise ImportError("resampling needs `scipy` (pip install phoonnx[train])") from exc
    divisor = gcd(sample_rate, target_sample_rate)
    return resample_poly(
        audio, target_sample_rate // divisor, sample_rate // divisor, axis=-1
    ).astype(np.float32)


def fit_channels(audio: np.ndarray, target_channels: int) -> np.ndarray:
    """Match the codec's channel count: mono is duplicated, extra channels are averaged."""
    channels = audio.shape[0]
    if channels == target_channels:
        return audio
    if channels == 1:
        return np.repeat(audio, target_channels, axis=0)
    if target_channels == 1:
        return audio.mean(axis=0, keepdims=True)
    if channels > target_channels:
        return audio[:target_channels]
    raise ValueError(f"cannot map {channels} channels to {target_channels}")


class OnnxAudioTokenizer:
    """Frozen RVQ encoder: waveform -> ``[frames, n_vq]`` code matrix."""

    def __init__(
        self,
        encode_onnx: str,
        providers: Optional[Sequence[str]] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
    ) -> None:
        import onnxruntime

        self.session = onnxruntime.InferenceSession(
            str(encode_onnx),
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._input_names = [item.name for item in self.session.get_inputs()]

    def encode_file(self, path: Path, n_vq: Optional[int] = None) -> List[List[int]]:
        audio, sample_rate = read_audio(path)
        audio = fit_channels(resample(audio, sample_rate, self.sample_rate), self.channels)
        return self.encode_waveform(audio, n_vq=n_vq)

    def encode_waveform(self, audio: np.ndarray, n_vq: Optional[int] = None) -> List[List[int]]:
        batch = np.asarray(audio, dtype=np.float32)[None, ...]
        feed: Dict[str, np.ndarray] = {self._input_names[0]: batch}
        if len(self._input_names) > 1:
            feed[self._input_names[1]] = np.asarray([batch.shape[-1]], dtype=np.int32)
        outputs = self.session.run(None, feed)
        codes = np.asarray(outputs[0])
        frames = int(codes.shape[1])
        if len(outputs) > 1:
            lengths = np.asarray(outputs[1]).reshape(-1)
            if lengths.size:
                frames = int(lengths[0])
        codes = codes[0, :frames, :]
        if n_vq is not None:
            if n_vq > codes.shape[1]:
                raise ValueError(f"asked for n_vq={n_vq} but the codec emits {codes.shape[1]}")
            codes = codes[:, :n_vq]
        return codes.astype(np.int64).tolist()


def load_manifest(
    path: Path,
    wav_dir: Optional[Path] = None,
    delimiter: str = "|",
) -> List[Dict[str, Any]]:
    """Read a JSONL or LJSpeech-style CSV manifest into upstream's record shape."""
    if path.suffix.lower() in {".jsonl", ".json", ".ndjson"}:
        return load_jsonl(path)
    records: List[Dict[str, Any]] = []
    base = wav_dir or path.parent
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(delimiter)
        if len(parts) < 2:
            raise ValueError(f"{path}:{line_number} has no '{delimiter}' separated text column")
        stem, text = parts[0], parts[-1]
        audio_path = Path(stem)
        if not audio_path.suffix:
            audio_path = audio_path.with_suffix(".wav")
        records.append({"audio": str(base / audio_path), "text": text})
    return records


def resolve_audio_paths(records: Iterable[Dict[str, Any]], base_dir: Path) -> List[Dict[str, Any]]:
    resolved = []
    for record in records:
        record = dict(record)
        for key in ("audio", "ref_audio"):
            value = record.get(key)
            if isinstance(value, str) and value:
                candidate = Path(value)
                record[key] = str(candidate if candidate.is_absolute() else (base_dir / candidate))
        resolved.append(record)
    return resolved


def encode_records(
    records: List[Dict[str, Any]],
    tokenizer: OnnxAudioTokenizer,
    n_vq: Optional[int] = None,
    encode_reference: bool = True,
) -> List[Dict[str, Any]]:
    cache: Dict[str, List[List[int]]] = {}

    def encode(path: str) -> List[List[int]]:
        if path not in cache:
            cache[path] = tokenizer.encode_file(Path(path), n_vq=n_vq)
        return cache[path]

    for index, record in enumerate(records):
        if record.get("audio_codes") is None:
            audio_path = record.get("audio")
            if not isinstance(audio_path, str) or not audio_path:
                raise ValueError(f"record {index} has no `audio` path and no `audio_codes`")
            record["audio_codes"] = encode(audio_path)
            if not record["audio_codes"]:
                raise ValueError(f"record {index}: {audio_path} produced zero codec frames")
        if encode_reference and record.get("ref_audio_codes") is None:
            reference = record.get("ref_audio")
            if isinstance(reference, str) and reference:
                record["ref_audio_codes"] = encode(reference)
        if index and index % 50 == 0:
            _LOGGER.info("encoded %d/%d records", index, len(records))
    return records


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--codec-encode-onnx", required=True, help="moss_audio_tokenizer_encode.onnx")
    parser.add_argument("--input-manifest", required=True, help="JSONL or LJSpeech-style CSV")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--wav-dir", default=None, help="audio root for CSV manifests")
    parser.add_argument("--n-vq", type=int, default=None, help="keep only the first N codec layers")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--providers", nargs="*", default=None, help="onnxruntime execution providers")
    parser.add_argument(
        "--skip-reference-audio",
        dest="encode_reference",
        action="store_false",
        help="do not encode `ref_audio`",
    )
    parser.set_defaults(encode_reference=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manifest_path = Path(args.input_manifest).expanduser().resolve()
    records = load_manifest(
        manifest_path, wav_dir=Path(args.wav_dir).expanduser() if args.wav_dir else None
    )
    if not records:
        raise SystemExit(f"{manifest_path} has no records")
    records = resolve_audio_paths(records, manifest_path.parent)

    tokenizer = OnnxAudioTokenizer(
        args.codec_encode_onnx,
        providers=args.providers,
        sample_rate=args.sample_rate,
        channels=args.channels,
    )
    records = encode_records(
        records, tokenizer, n_vq=args.n_vq, encode_reference=args.encode_reference
    )
    output = dump_jsonl(records, args.output_jsonl)
    frames = sum(len(record["audio_codes"]) for record in records)
    _LOGGER.info(
        "wrote %d records (%d frames, %.1f s of audio) to %s",
        len(records), frames, frames / 12.5, output,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["OnnxAudioTokenizer", "load_manifest", "encode_records", "read_audio", "resample", "fit_channels"]
