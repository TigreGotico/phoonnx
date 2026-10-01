"""Multi-format dataset loading for training preprocessing.

Preprocessing consumes a stream of :class:`Utterance` objects regardless of how
the dataset is stored on disk. Every supported on-disk shape is loaded by a
named generator registered here, mirroring the scorer registry in
``phoonnx_train/quality_filter.py``: a loader takes ``(source, config)`` and
yields fully-populated :class:`Utterance` objects.

Supported formats (see :func:`detect_format` for auto-detection):

* ``ljspeech`` -- a directory with a pipe-delimited ``metadata.csv`` + ``wav(s)/``.
* ``jsonl`` -- a ``.jsonl`` file, one JSON object per line.
* ``parquet`` -- a ``.parquet`` file, a glob of shards, or a directory of shards.
* ``hf`` -- a Hugging Face ``org/name`` repo id loaded via ``datasets``.

The tabular formats (jsonl/parquet/hf) resolve columns by name with sensible
fallbacks (see the ``DEFAULT_*_COLUMNS`` tuples) and carry any unmapped columns
through under :attr:`Utterance.extras`. Hugging Face audio columns (and our own
parquet shards) frequently hold embedded audio *bytes* rather than a filesystem
path -- ``path`` is often ``None`` and casting does not inline the bytes -- so
the loaders read the bytes field explicitly into :attr:`Utterance.audio_bytes`
and audio-consuming code materializes them on demand via :func:`ensure_audio_path`.
"""
import csv
import dataclasses
import io
import json
import logging
import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from phoonnx.config import Alphabet, PhonemeType

_LOGGER = logging.getLogger("preprocess")

DEFAULT_TEXT_COLUMNS: Tuple[str, ...] = ("text", "sentence", "transcription", "transcript")
DEFAULT_AUDIO_COLUMNS: Tuple[str, ...] = ("audio",)
DEFAULT_SPEAKER_COLUMNS: Tuple[str, ...] = ("speaker", "speaker_id")
DEFAULT_PHONEMES_COLUMNS: Tuple[str, ...] = ()
DEFAULT_LANG_COLUMNS: Tuple[str, ...] = ()


@dataclass
class Utterance:
    """Represents a single utterance in the dataset."""
    text: str
    audio_path: Path
    speaker: Optional[str] = None
    speaker_id: Optional[int] = None
    phonemes: Optional[List[str]] = None
    phoneme_ids: Optional[List[int]] = None
    audio_norm_path: Optional[Path] = None
    audio_spec_path: Optional[Path] = None
    # engine-specific extras (e.g. yourtts d_vector_path, language_id;
    # fastpitch f0_path) merged from TrainingEngine.extra_preprocess
    d_vector_path: Optional[Path] = None
    language_id: Optional[int] = None
    f0_path: Optional[Path] = None
    # stable identity for cache/spec keying and resume when there is no
    # filesystem audio path (embedded-bytes rows)
    row_id: Optional[str] = None
    # embedded audio bytes (HF/parquet inline audio); decoded on demand
    audio_bytes: Optional[bytes] = None
    # True when phonemes came from a dataset column and must not be
    # re-phonemized, normalized, or case-mangled
    phonemes_precomputed: bool = False
    # unmapped source columns, carried through into dataset.jsonl
    extras: Dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> Dict[str, Any]:
        """Custom asdict to handle Path objects for JSON serialization.

        The embedded audio bytes and the internal precomputed-phonemes flag are
        dropped: bytes are not JSON serializable and only exist to feed audio
        processing, and the flag is a processing detail, not training data.
        """
        data = dataclasses.asdict(self)
        data.pop("audio_bytes", None)
        data.pop("phonemes_precomputed", None)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        return data


def get_text_casing(casing: str) -> Callable[[str], str]:
    """
    Returns a function to apply text casing based on a string name.

    Args:
        casing: The name of the casing function ('lower', 'upper', 'casefold', or 'ignore').

    Returns:
        A callable function (str) -> str.
    """
    if casing == "lower":
        return str.lower
    if casing == "upper":
        return str.upper
    if casing == "casefold":
        return str.casefold
    return lambda s: s


@dataclass
class PreprocessorConfig:
    """Dataclass to hold all runtime configuration, mimicking argparse.Namespace."""
    input_dir: Path
    output_dir: Path
    language: str
    sample_rate: int
    cache_dir: Path
    max_workers: int
    single_speaker: bool
    speaker_id: Optional[int]
    phoneme_type: PhonemeType
    alphabet: Alphabet
    phonemizer_model: str
    text_casing: str
    dataset_name: Optional[str]
    audio_quality: Optional[str]
    skip_audio: bool
    debug: bool
    add_diacritics: bool
    # multi-format loading
    dataset_format: str = "auto"
    text_column: Optional[str] = None
    audio_column: Optional[str] = None
    speaker_column: Optional[str] = None
    phonemes_column: Optional[str] = None
    lang_column: Optional[str] = None


# -----------------------------------------------------------------------------
# Loader registry
# -----------------------------------------------------------------------------
Loader = Callable[[str, PreprocessorConfig], Iterable[Utterance]]
_LOADER_REGISTRY: Dict[str, Loader] = {}


def register_loader(name: str, fn: Loader) -> None:
    """Register a named dataset loader generator.

    Args:
        name: format name (e.g. "ljspeech", "jsonl", "parquet", "hf").
        fn: callable ``(source, config) -> Iterable[Utterance]``.
    """
    _LOADER_REGISTRY[name] = fn


def known_loaders() -> List[str]:
    """Names of all registered loaders."""
    return list(_LOADER_REGISTRY)


def detect_format(source: str) -> str:
    """Guess a dataset format from a single input source.

    Rules:
        * an existing directory with ``metadata.csv`` -> ``ljspeech``.
        * an existing directory with ``*.parquet`` shards -> ``parquet``.
        * an existing ``.jsonl`` file -> ``jsonl``.
        * an existing ``.parquet`` file -> ``parquet``.
        * a glob pattern (``*?[]``) -> ``parquet`` (shard glob).
        * an ``org/name`` string that is not an existing path -> ``hf``.

    Raises:
        ValueError: if the source matches none of the rules.
    """
    path = Path(source)
    if path.exists():
        if path.is_dir():
            if (path / "metadata.csv").exists():
                return "ljspeech"
            if any(path.glob("*.parquet")):
                return "parquet"
            raise ValueError(
                f"cannot auto-detect dataset format for directory {source!r}: "
                "no metadata.csv and no .parquet shards found"
            )
        if path.suffix == ".jsonl":
            return "jsonl"
        if path.suffix == ".parquet":
            return "parquet"
        raise ValueError(
            f"cannot auto-detect dataset format for file {source!r}: "
            "expected a .jsonl or .parquet file"
        )
    if any(ch in source for ch in "*?[]"):
        return "parquet"
    if re.fullmatch(r"[\w.-]+/[\w.-]+", source):
        return "hf"
    raise ValueError(
        f"cannot auto-detect dataset format for {source!r}: not an existing "
        "path, a shard glob, or an 'org/name' Hugging Face repo id"
    )


def load_source(source: str, config: PreprocessorConfig) -> Iterable[Utterance]:
    """Load one source, dispatching by ``config.dataset_format`` (or auto)."""
    fmt = config.dataset_format
    if fmt == "auto":
        fmt = detect_format(source)
    if fmt not in _LOADER_REGISTRY:
        raise ValueError(f"unknown dataset format {fmt!r}; known: {', '.join(known_loaders())}")
    _LOGGER.info("Loading source %r as format %r", source, fmt)
    return _LOADER_REGISTRY[fmt](source, config)


# -----------------------------------------------------------------------------
# Column resolution (tabular loaders)
# -----------------------------------------------------------------------------
def _resolve_column(keys: Iterable[str], requested: Optional[str],
                    defaults: Tuple[str, ...], what: str, required: bool = False) -> Optional[str]:
    """Pick the column name to use for a field.

    An explicitly requested column must exist (else a loud error). Otherwise the
    first present default is used; if none match and the field is required, fail.
    """
    keyset = set(keys)
    if requested:
        if requested not in keyset:
            raise ValueError(
                f"requested {what} column {requested!r} not found; "
                f"available columns: {sorted(keyset)}"
            )
        return requested
    for candidate in defaults:
        if candidate in keyset:
            return candidate
    if required:
        raise ValueError(
            f"no {what} column found (tried {list(defaults)}); "
            f"available columns: {sorted(keyset)}; pass an explicit column flag"
        )
    return None


def _jsonable(value: Any) -> Any:
    """Coerce a source-column value into something json.dump can serialize.

    Passes primitives through, unwraps numpy scalars via ``.item()``, and drops
    anything else (bytes, arrays, nested audio mappings) by returning ``None``.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, dict):
        try:
            return value.item()
        except (ValueError, TypeError):
            return None
    return None


def _audio_from_value(value: Any) -> Tuple[Optional[str], Optional[bytes]]:
    """Extract (path, bytes) from a row's audio-column value.

    Handles a plain path string, raw bytes, and the Hugging Face Audio mapping
    ``{"bytes": ..., "path": ...}`` where ``path`` may be ``None`` and the audio
    only exists as embedded bytes.
    """
    if value is None:
        return None, None
    if isinstance(value, (bytes, bytearray)):
        return None, bytes(value)
    if isinstance(value, dict):
        raw = value.get("bytes")
        return value.get("path"), bytes(raw) if raw is not None else None
    return str(value), None


def _row_utterance(row: Dict[str, Any], cols: Dict[str, Optional[str]],
                   config: PreprocessorConfig, source: str, index: int) -> Optional[Utterance]:
    """Build an Utterance from a tabular row given resolved column names."""
    text = row.get(cols["text"])
    if text is None or str(text) == "":
        _LOGGER.warning("Skipping row %d with empty text in %s", index, source)
        return None

    audio_path, audio_bytes = (None, None)
    if cols["audio"]:
        audio_path, audio_bytes = _audio_from_value(row.get(cols["audio"]))

    if not config.skip_audio and audio_path is None and audio_bytes is None:
        _LOGGER.warning("Skipping row %d with no audio in %s", index, source)
        return None

    speaker = None
    if not config.single_speaker and cols["speaker"]:
        raw_speaker = row.get(cols["speaker"])
        speaker = str(raw_speaker) if raw_speaker is not None and str(raw_speaker) != "" else None

    phonemes = None
    precomputed = False
    if cols["phonemes"]:
        raw_ph = row.get(cols["phonemes"])
        if raw_ph is not None and str(raw_ph).strip():
            phonemes = str(raw_ph).split()
            precomputed = True

    row_id = str(audio_path) if audio_path else f"{source}#{index}"

    # The lang column is intentionally left in extras (carried through into
    # dataset.jsonl) rather than consumed here.
    mapped = {cols[k] for k in ("text", "audio", "speaker", "phonemes") if cols[k]}
    extras = {k: _jsonable(v) for k, v in row.items() if k not in mapped}
    extras = {k: v for k, v in extras.items() if v is not None}

    return Utterance(
        text=str(text),
        audio_path=Path(audio_path) if audio_path else Path(""),
        speaker=speaker,
        speaker_id=config.speaker_id,
        phonemes=phonemes,
        phonemes_precomputed=precomputed,
        row_id=row_id,
        audio_bytes=audio_bytes,
        extras=extras,
    )


def _resolve_columns(keys: Iterable[str], config: PreprocessorConfig) -> Dict[str, Optional[str]]:
    keys = list(keys)
    return {
        "text": _resolve_column(keys, config.text_column, DEFAULT_TEXT_COLUMNS, "text", required=True),
        "audio": _resolve_column(keys, config.audio_column, DEFAULT_AUDIO_COLUMNS, "audio"),
        "speaker": _resolve_column(keys, config.speaker_column, DEFAULT_SPEAKER_COLUMNS, "speaker"),
        "phonemes": _resolve_column(keys, config.phonemes_column, DEFAULT_PHONEMES_COLUMNS, "phonemes"),
        "lang": _resolve_column(keys, config.lang_column, DEFAULT_LANG_COLUMNS, "lang"),
    }


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def ljspeech_loader(source: str, config: PreprocessorConfig) -> Iterable[Utterance]:
    """
    Generator for LJSpeech-style dataset.
    Loads metadata and resolves audio file paths.

    Args:
        source: dataset directory containing ``metadata.csv`` and ``wav(s)/``.
        config: The configuration object containing dataset parameters.

    Yields:
        Utterance: A fully populated Utterance object.
    """
    dataset_dir = Path(source)
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        _LOGGER.error(f"Missing metadata file: {metadata_path}")
        return

    wav_dirs: List[Path] = [dataset_dir / "wav", dataset_dir / "wavs"]

    with open(metadata_path, "r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter="|")
        for row in reader:
            if len(row) < 2:
                _LOGGER.warning(f"Skipping malformed row: {row}")
                continue

            filename: str = row[0]
            text: str = row[-1]
            speaker: Optional[str] = None

            if not config.single_speaker and len(row) > 2:
                speaker = row[1]
            else:
                speaker = None

            wav_path: Optional[Path] = None
            for wav_dir in wav_dirs:
                potential_paths: List[Path] = [
                    wav_dir / filename,
                    wav_dir / f"{filename}.wav",
                    wav_dir / f"{filename.lstrip('0')}.wav"
                ]
                for path in potential_paths:
                    if path.exists():
                        wav_path = path
                        break
                if wav_path:
                    break

            if not config.skip_audio and not wav_path:
                _LOGGER.warning("Missing audio file for filename: %s", filename)
                continue

            if not config.skip_audio and wav_path and wav_path.stat().st_size == 0:
                _LOGGER.warning("Empty audio file: %s", wav_path)
                continue

            yield Utterance(
                text=text,
                audio_path=wav_path or Path(""),  # Use empty path if skipping audio, should not be used
                speaker=speaker,
                speaker_id=config.speaker_id,
                row_id=str(wav_path) if wav_path else filename,
            )


def jsonl_loader(source: str, config: PreprocessorConfig) -> Iterable[Utterance]:
    """Generator for a JSON-lines dataset (one JSON object per line).

    Columns are resolved from the first non-empty row and applied to all rows;
    malformed lines are logged and skipped rather than aborting the load.
    """
    cols: Optional[Dict[str, Optional[str]]] = None
    with open(source, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                _LOGGER.warning("Skipping malformed JSON on line %d in %s", index, source)
                continue
            if not isinstance(row, dict):
                _LOGGER.warning("Skipping non-object JSON on line %d in %s", index, source)
                continue
            if cols is None:
                cols = _resolve_columns(row.keys(), config)
            utt = _row_utterance(row, cols, config, source, index)
            if utt is not None:
                yield utt


def _iter_parquet_paths(source: str) -> List[Path]:
    path = Path(source)
    if path.is_dir():
        return sorted(path.glob("*.parquet"))
    if any(ch in source for ch in "*?[]"):
        from glob import glob
        return sorted(Path(p) for p in glob(source))
    return [path]


def parquet_loader(source: str, config: PreprocessorConfig) -> Iterable[Utterance]:
    """Generator for a parquet file, a glob of shards, or a directory of shards.

    Reads with pandas (pyarrow backend). Audio may be a path string or embedded
    bytes (our own shards inline audio bytes to stay under HF's per-dir file cap).
    """
    import pandas as pd

    shards = _iter_parquet_paths(source)
    if not shards:
        _LOGGER.error("No parquet shards found for %s", source)
        return

    cols: Optional[Dict[str, Optional[str]]] = None
    index = 0
    for shard in shards:
        frame = pd.read_parquet(shard)
        if cols is None:
            cols = _resolve_columns(frame.columns, config)
        for record in frame.to_dict(orient="records"):
            utt = _row_utterance(record, cols, config, source, index)
            index += 1
            if utt is not None:
                yield utt


def hf_loader(source: str, config: PreprocessorConfig) -> Iterable[Utterance]:
    """Generator for a Hugging Face dataset given an ``org/name`` repo id.

    Audio columns are cast to ``Audio(decode=False)`` so the embedded bytes are
    read explicitly (decoding/casting does not inline bytes, and ``path`` is
    frequently ``None`` for byte-backed audio).
    """
    import datasets

    loaded = datasets.load_dataset(source)
    splits = loaded.values() if isinstance(loaded, datasets.DatasetDict) else [loaded]

    index = 0
    for split in splits:
        audio_col = _resolve_column(split.column_names, config.audio_column,
                                    DEFAULT_AUDIO_COLUMNS, "audio")
        if audio_col and isinstance(split.features.get(audio_col), datasets.Audio):
            split = split.cast_column(audio_col, datasets.Audio(decode=False))
        cols = _resolve_columns(split.column_names, config)
        for record in split:
            utt = _row_utterance(record, cols, config, source, index)
            index += 1
            if utt is not None:
                yield utt


register_loader("ljspeech", ljspeech_loader)
register_loader("jsonl", jsonl_loader)
register_loader("parquet", parquet_loader)
register_loader("hf", hf_loader)


# -----------------------------------------------------------------------------
# Embedded-bytes audio materialization
# -----------------------------------------------------------------------------
def ensure_audio_path(utt: Utterance, cache_dir: Union[str, Path]) -> Path:
    """Return a filesystem audio path for an utterance, decoding embedded bytes.

    Path-backed utterances return their path unchanged (byte-identical to the
    existing flow). Byte-backed utterances have their audio decoded via
    soundfile and written once to ``<cache_dir>/audio/<row_id>.wav``; the path
    is cached on the utterance so repeated calls are cheap.

    Raises:
        soundfile.LibsndfileError / RuntimeError: if the embedded bytes cannot
            be decoded (corrupt audio), so callers can drop the sample loudly.
    """
    if str(utt.audio_path) not in ("", "."):
        return Path(utt.audio_path)
    if utt.audio_bytes is None:
        raise RuntimeError(f"utterance {utt.row_id!r} has neither an audio path nor audio bytes")

    import soundfile as sf

    key = utt.row_id or sha256(utt.audio_bytes).hexdigest()
    safe = sha256(key.encode("utf-8")).hexdigest()
    out = Path(cache_dir) / "audio" / f"{safe}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        data, sr = sf.read(io.BytesIO(utt.audio_bytes))
        tmp = out.with_name(out.name + ".tmp")
        sf.write(tmp, data, sr, format="WAV")
        tmp.rename(out)
    utt.audio_path = out
    return out
