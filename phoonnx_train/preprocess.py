#!/usr/bin/env python3
import dataclasses
import functools
import importlib.util
import itertools
import json
import logging
import os
import queue
from collections import Counter
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, packages_distributions, version
from multiprocessing import JoinableQueue, Process, Queue
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any, Set, Union, Callable

import click
import torch
from tqdm import tqdm

from phoonnx.config import PhonemeType, _voice_phonemizer, get_text_casing, Alphabet, voice_frontend
from scriptconv.diacritics import diacritize
from scriptconv.frontend import Stages, TextFrontend
from phoonnx.tokenizer import TTSTokenizer, DEFAULT_IPA_PHONEME_ID_MAP, DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, \
    phoneme_map_seed, untrained_map_symbols, \
    DEFAULT_EOS_TOKEN, DEFAULT_BLANK_WORD_TOKEN
from phoonnx.version import VERSION_STR
from phoonnx_train.dataset_loaders import (PreprocessorConfig, Utterance,
                                           ensure_audio_path,
                                           known_loaders, load_source)
from phoonnx_train.norm_audio import cache_norm_audio, make_silence_detector
from phoonnx_train.segments import segment_phonemes
from phoonnx_train.quality_filter import (FilterSpec, apply_quality_filters,
                                          configure_asr_model,
                                          configure_speaker_model,
                                          configure_vad_model, known_scorers,
                                          log_filter_summary, parse_filter_spec)

_LOGGER = logging.getLogger("preprocess")

# Base phoneme map
DEFAULT_SPECIAL_PHONEME_ID_MAP: Dict[str, int] = {
    DEFAULT_PAD_TOKEN: 0,
    DEFAULT_BOS_TOKEN: 1,
    DEFAULT_EOS_TOKEN: 2,
    DEFAULT_BLANK_WORD_TOKEN: 3,
}
MAX_PHONEMES = 256
# Path fields of a dataset.jsonl row that hold cache files, such as an engine
# step's d-vector. audio_path has its own override.
_CACHE_PATH_FIELDS: Tuple[str, ...] = tuple(
    field.name for field in dataclasses.fields(Utterance)
    if field.type == Optional[Path] and field.name != "audio_path")
# -----------------------------------------------------------------------------

def phonemizer_registry(phonemizer: Any) -> Tuple[str, str]:
    """Name the distribution the phonemizer class came from, and its version.

    This is the registry that resolved the phonemizer, not the grapheme-to-
    phoneme backend. For a phoneme type that shells out — espeak runs the system
    ``espeak-ng`` binary — the backend and its version are not captured here and
    nothing in the config identifies them.

    Returns the distribution name and version, or the top-level module and an
    empty version when the class comes from no installed distribution. Where
    several distributions provide the same top-level module the lowest name
    wins, so the field does not vary between machines with the same install.
    """
    module = type(phonemizer).__module__.split(".")[0]
    dist = sorted(packages_distributions().get(module, [module]))[0]
    try:
        return dist, version(dist)
    except PackageNotFoundError:
        return dist, ""


def phonemizer_identity(phonemizer: Any, lang: str) -> Dict[str, Optional[str]]:
    """Name the engine that phonemized the rows, as scriptconv reports it.

    The keys are ``backend``, ``distribution``, ``version`` (for espeak-ng, the
    version of the binary), ``lang`` (the resolved lect) and ``revision`` (a
    hash of the spec, code and data that make the phonemes, or None) and ``frontend`` (None, because the
    backend is called directly). Unlike :func:`phonemizer_registry`,
    this names the grapheme-to-phoneme engine itself.
    """
    return dataclasses.asdict(phonemizer.identity(lang))


class PathEncoder(json.JSONEncoder):
    """JSON encoder for Path objects."""

    def default(self, o: Any) -> Union[str, Any]:
        """
        Converts Path objects to strings for serialization.

        Args:
            o: The object to serialize.

        Returns:
            The serialized string representation or the default JSON serialization.
        """
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def build_frontend(phoneme_type: PhonemeType, alphabet: Alphabet, model: Optional[str],
                   keep_punctuation: bool, language: str, add_diacritics: bool) -> TextFrontend:
    """Build the text frontend a worker uses, the one a voice builds at inference.

    It is strict, so a row whose diacritizer or normaliser fails is dropped
    rather than trained on text without that stage. Module level, so a partial
    of it pickles.
    """
    phonemizer = _voice_phonemizer(phoneme_type, alphabet, model, keep_punctuation=keep_punctuation)
    return voice_frontend(phonemizer, language, Stages(diacritize=add_diacritics), strict=True)


def phonemize_worker(
        config: PreprocessorConfig,
        task_queue: JoinableQueue,
        result_queue: Queue,
        make_frontend: Callable[[], TextFrontend],
) -> None:
    """
    Worker process for phonemization and audio processing.

    Args:
        config: The configuration object containing runtime parameters.
        task_queue: Queue for receiving batches of Utterance objects.
        result_queue: Queue for sending processed results (Utterance, set of phonemes).
        make_frontend: Builds this worker's text frontend. The worker builds its
            own because a loaded phonemizer may hold native handles that do not
            pickle, and the ``spawn`` and ``forkserver`` start methods pickle
            every argument. ``forkserver`` is the Linux default from Python 3.14.
    """
    try:
        # Built on the first row that needs it: a run of precomputed phonemes
        # never reads text, so a backend that rejects the language must not stop it.
        frontend: Optional[TextFrontend] = None
        casing: Callable[[str], str] = get_text_casing(config.text_casing)
        silence_detector = make_silence_detector()

        while True:
            # Get a batch of utterances to process
            utterance_batch: Union[List[Utterance], None] = task_queue.get()
            if utterance_batch is None:
                # Signal to exit
                task_queue.task_done()
                break

            for utt in utterance_batch:
                try:
                    if utt.phonemes_precomputed:
                        # Phonemes come verbatim from a dataset column: use them
                        # as-is, never normalized or case-mangled.
                        if not utt.phonemes:
                            raise RuntimeError("empty precomputed phonemes")
                    else:
                        if frontend is None:
                            frontend = make_frontend()
                        # Normalize, case, diacritize and phonemize as synthesis does
                        utt.phonemes = frontend.phonemize_flat(
                            utt.text, Stages(diacritize=config.add_diacritics, casing=casing))
                        if not utt.phonemes:
                            raise RuntimeError(f"Phonemes not found for '{utt.text}'")

                    # Process audio if not skipping
                    if not config.skip_audio:
                        audio_path = ensure_audio_path(utt, config.cache_dir)
                        utt.audio_norm_path, utt.audio_spec_path, utt.spec_frames = cache_norm_audio(
                            audio_path,
                            config.cache_dir,
                            silence_detector,
                            config.sample_rate,
                        )

                    # Put the processed utterance and its phonemes into the result queue
                    # The result is a tuple of (Utterance, set of unique phonemes in that utterance)
                    symbols = utt.phonemes
                    if config.segment_map and all(len(p) == 1 for p in utt.phonemes):
                        # the tokenizer reads a longer item whole, so only single code points are cut
                        symbols = segment_phonemes("".join(utt.phonemes))
                    result_queue.put((utt, set(symbols)))
                except Exception:
                    _LOGGER.exception("Failed to process utterance: %s", utt.audio_path)
                    result_queue.put((None, set()))

            task_queue.task_done()

    except Exception:
        _LOGGER.exception("Worker process failed")
        # A non-zero exit code is how the parent learns the worker is gone.
        raise SystemExit(1)


def next_result(result_queue: Queue, task_queue: JoinableQueue,
                processes: List[Process]) -> Tuple[Optional[Utterance], Set[str]]:
    """Wait for the next worker result; stop every worker when one has failed.

    A worker that dies sends no result for the batch it held, and its stop
    signal stays in the task queue, so waiting longer would never end. The
    batches nobody will read stay in the pipe, so the queues' feeder threads
    are not joined at exit.
    """
    while True:
        try:
            return result_queue.get(timeout=1.0)
        except queue.Empty:
            failed = [proc for proc in processes if proc.exitcode not in (None, 0)]
            if failed:
                for proc in processes:
                    proc.terminate()
                task_queue.cancel_join_thread()
                result_queue.cancel_join_thread()
                raise RuntimeError(f"{len(failed)} preprocessing worker(s) stopped, the first with "
                                   f"exit code {failed[0].exitcode}; see the worker log above")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-i",
    "--input-dir",
    "input_sources",
    multiple=True,
    required=True,
    help="Dataset source. Repeatable (or comma-separated) to merge multiple "
         "datasets; per-source speaker ids are namespaced to avoid collisions. "
         "A source may be an LJSpeech-style directory (metadata.csv + wav(s)/), "
         "a .jsonl file, a .parquet file / shard glob / directory of shards, or "
         "a Hugging Face 'org/name' repo id (see --dataset-format).",
)
@click.option(
    "--dataset-format",
    "dataset_format",
    type=click.Choice(["auto", "ljspeech", "jsonl", "parquet", "hf"]),
    default="auto",
    show_default=True,
    help="Input format. 'auto' detects from each source: a directory with "
         "metadata.csv -> ljspeech; a .jsonl file -> jsonl; a .parquet file, "
         "shard glob, or directory of shards -> parquet; an 'org/name' string "
         "that is not an existing path -> hf.",
)
@click.option("--text-column", "text_column", default=None,
              help="Column holding transcript text (jsonl/parquet/hf). "
                   "Default: first of text, sentence, transcription, transcript.")
@click.option("--audio-column", "audio_column", default=None,
              help="Column holding audio path or embedded bytes "
                   "(jsonl/parquet/hf). Default: audio.")
@click.option("--speaker-column", "speaker_column", default=None,
              help="Column holding the speaker label (jsonl/parquet/hf). "
                   "Default: first of speaker, speaker_id.")
@click.option("--phonemes-column", "phonemes_column", default=None,
              help="Opt-in: column holding precomputed, whitespace-separated "
                   "phoneme symbols (jsonl/parquet/hf). Rows with a non-empty "
                   "value skip phonemization and are used verbatim; their "
                   "symbols are validated against the final phoneme map.")
@click.option("--lang-column", "lang_column", default=None,
              help="Column holding a per-row language code (jsonl/parquet/hf). "
                   "Carried through into dataset.jsonl extras. Default: unset.")
@click.option("--resume", "resume", is_flag=True,
              help="Skip rows already present (by row_id/audio path) in an "
                   "existing output dataset.jsonl and append only new rows. "
                   "Writes are atomic (temp file + rename). Incompatible with "
                   "--corpus-only-map, which cannot be reconstructed from "
                   "already-written rows.")
@click.option("--metrics-out", "metrics_out", type=click.Path(dir_okay=False, path_type=Path),
              default=None,
              help="Write every computed --filter metric value per row_id to "
                   "this parquet sidecar during filtering.")
@click.option("--metrics-in", "metrics_in", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help="Read a previously written metrics sidecar; with "
                   "--filter-from-columns its values are preferred over "
                   "recomputation.")
@click.option("--filter-from-columns", "filter_from_columns", is_flag=True,
              help="Make --filter prefer a per-row value from a dataset column "
                   "of the same name (or from --metrics-in) before computing it "
                   "on demand.")
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory to write output files for training (config.json, dataset.jsonl)",
)
@click.option(
    "-l",
    "--language",
    "language",
    required=True,
    help="phonemizer language code (e.g., 'en', 'es', 'fr')",
)
@click.option(
    "-c",
    "--prev-config",
    "prev_config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to a previous config.json from which to reuse phoneme_id_map. (for fine-tuning only)",
)
@click.option(
    "--drop-extra-phonemes",
    "drop_extra_phonemes",
    type=bool,
    default=True,
    help="If training data has more symbols than base model, discard new symbols. (for fine-tuning only)",
)
@click.option(
    "--corpus-only-map",
    "corpus_only_map",
    is_flag=True,
    default=False,
    help="Build the phoneme map only from symbols present in the corpus, instead of "
         "seeding it with the full default IPA table. Symbols outside the map fail at "
         "tokenization instead of mapping to embeddings the model never trained. "
         "Models preprocessed this way can only be fine-tuned from configs with a "
         "compatible (subset) map.",
)
@click.option(
    "--segment-map",
    "segment_map",
    is_flag=True,
    default=False,
    help="Give each phoneme segment one id: a symbol with its combining marks, "
         "its attaching modifier letters (ʲ ː ʰ ...), and the symbol joined by a "
         "tie bar. A features.json table needs this, because it describes whole "
         "segments. Implies --corpus-only-map.",
)
@click.option(
    "-r",
    "--sample-rate",
    "sample_rate",
    type=int,
    default=22050,
    help="Target sample rate for voice (hertz, Default: 22050)",
)
@click.option(
    "--cache-dir",
    "cache_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory to cache processed audio files. Defaults to <output-dir>/cache/<sample-rate>.",
)
@click.option(
    "-w",
    "--max-workers",
    "max_workers",
    type=click.IntRange(min=1),
    default=os.cpu_count() or 1,
    help="Maximum number of worker processes to use for parallel processing. Defaults to CPU count.",
)
@click.option(
    "--single-speaker",
    "single_speaker",
    is_flag=True,
    help="Force treating the dataset as single speaker, ignoring metadata speaker columns.",
)
@click.option(
    "--speaker-id",
    "speaker_id",
    type=int,
    default=None,
    help="Specify a fixed speaker ID (0, 1, etc.) for a single speaker dataset.",
)
@click.option(
    "--phoneme-type",
    "phoneme_type",
    type=click.Choice([p.value for p in PhonemeType]),
    default=PhonemeType.ESPEAK.value,
    help="Type of phonemes to use.",
)
@click.option(
    "--alphabet",
    "alphabet",
    type=click.Choice([a.value for a in Alphabet]),
    default=Alphabet.IPA.value,
    help="Phoneme alphabet to use (e.g., IPA).",
)
@click.option(
    "--phonemizer-model",
    "phonemizer_model",
    default="",
    help="Path or name of a custom phonemizer model, if applicable.",
)
@click.option(
    "--text-casing",
    "text_casing",
    type=click.Choice(("ignore", "lower", "upper", "casefold")),
    default="ignore",
    help="Casing applied to utterance text before phonemization.",
)
@click.option(
    "--dataset-name",
    "dataset_name",
    default=None,
    help="Name of dataset to put in config (default: name of <output_dir>/../).",
)
@click.option(
    "--audio-quality",
    "audio_quality",
    default=None,
    help="Audio quality description to put in config (default: name of <output_dir>).",
)
@click.option(
    "--skip-audio",
    "skip_audio",
    is_flag=True,
    help="Do not preprocess or cache audio files.",
)
@click.option(
    "--debug",
    "debug",
    is_flag=True,
    help="Print DEBUG messages to the console.",
)
@click.option(
    "--add-diacritics",
    "add_diacritics",
    is_flag=True,
    help="Add diacritics to text (phonemizer specific, e.g., to denote stress).",
)
@click.option(
    "--keep-punctuation/--no-keep-punctuation",
    "keep_punctuation",
    default=True,
    show_default=True,
    help="Keep clause punctuation (, . ? ! : ;) in the phonemes, as Piper does. "
         "Recorded in config.json so synthesis feeds the model the same symbols. "
         "--resume keeps whatever the existing config.json recorded.",
)
@click.option(
    "--jsonl-audio-path",
    default=None,
    help="Replace the base folder of each audio_path in dataset.jsonl: the parent of the "
         "folder that holds every audio file. In a dataset directory, that folder is the top "
         "folder with the audio, such as wav/, so the base is the dataset directory. "
         "The rest of each path stays."
)
@click.option(
    "--jsonl-audio-spec-path",
    default=None,
    help="Replace the base folder of each cache path in dataset.jsonl, such as audio_norm_path, "
         "audio_spec_path and the d_vector_path of --engine yourtts: "
         "the parent of --cache-dir, or --output-dir when --cache-dir is not given. "
         "The rest of each path stays."
)
@click.option(
    "--engine",
    default=None,
    help="run this training engine's extra feature extraction per utterance "
         "(e.g. 'yourtts' d-vectors, 'fastpitch' F0) and record the produced "
         "fields in dataset.jsonl",
)
@click.option(
    "--speaker-encoder-path",
    default=None,
    help="[--engine yourtts] path to the Coqui ResNet ONNX speaker encoder "
         "used to compute d-vectors",
)
@click.option(
    "--language-id",
    default=None,
    type=int,
    help="[--engine yourtts] language id recorded on every utterance "
         "(multilingual training)",
)
@click.option(
    "--f0-method",
    "f0_method",
    default="pyin",
    type=click.Choice(["pyin", "dio", "harvest"]),
    show_default=True,
    help="[--engine fastpitch/speedyspeech/mixer] ground-truth F0 (pitch) "
         "extractor: 'pyin' (librosa, default, no extra native dependency) "
         "or 'dio'/'harvest' (WORLD via pyworld, ~50x faster at "
         "preprocessing time, requires the 'train-pyworld' extra). Must "
         "match the f0_method the training engine config uses, since it "
         "selects which F0 sidecar cache filename is read at train time.",
)
@click.option(
    "--filter",
    "quality_filters",
    multiple=True,
    metavar="COLUMN:MIN:MAX",
    help="Drop utterances outside [MIN, MAX] on an on-demand-computed quality "
         "metric. Repeatable; a sample must pass every --filter to be kept. "
         "MIN or MAX may be empty for unbounded on that side. Metrics are "
         "computed fresh per sample (not read from precomputed dataset "
         "columns): 'wpm' (words per minute, arithmetic), 'snr' (energy-based "
         "signal-to-noise dB estimate, arithmetic), 'clipping' (fraction of "
         "near-full-scale samples, arithmetic), 'is_music_like' (0/1 "
         "onset-rhythmicity heuristic, not a trained classifier -- roughly "
         "25-30% error rate at any threshold, a coarse pre-filter only), "
         "'vad_ratio' (speech-activity fraction via vadonnx, see "
         "--vad-model), 'speaker_consistency' (min pairwise cosine "
         "similarity between windows via speakeronnx, see --speaker-model), "
         "'utmos' (SpeechMOS UTMOS naturalness), 'dnsmos_sig'/'dnsmos_bak'/"
         "'dnsmos_ovrl' (DNSMOS P.835), 'plcmos' (packet-loss-concealment "
         "quality, catches VoIP/dropped-packet artifacts), 'aecmos' "
         "(echo-cancellation quality, catches speakerphone/echo artifacts; "
         "plcmos/aecmos are most relevant to call-based corpora and require "
         "the 'speechmos' package), 'wer' (word error rate of an onnx_asr "
         "transcription against the sample's own text, see --asr-model; the "
         "most expensive filter, always evaluated last). Referencing an "
         "unknown column warns and skips that filter instead of failing. "
         "Examples: --filter utmos:3.0: --filter wpm:80:400 "
         "--filter is_music_like:0:0 --filter snr:15: --filter clipping:0:0.01 "
         "--filter vad_ratio:0.5: --filter speaker_consistency:0.6: "
         "--filter plcmos:3.0: --filter aecmos:3.0: --filter wer:0:0.3",
)
@click.option(
    "--vad-model",
    "vad_model",
    default="silero",
    show_default=True,
    help="vadonnx model name used by the 'vad_ratio' quality filter.",
)
@click.option(
    "--speaker-model",
    "speaker_model",
    default="wespeaker-resnet34",
    show_default=True,
    help="speakeronnx model name used by the 'speaker_consistency' quality filter.",
)
@click.option(
    "--asr-model",
    "asr_model",
    default="whisper-base",
    show_default=True,
    help="Model identifier or path loadable via onnx_asr.load_model(), used "
         "by the 'wer' quality filter. Must be onnx-asr-compatible; an "
         "unloadable value fails loudly instead of silently skipping 'wer'.",
)
def cli(
    input_sources: Tuple[str, ...],
    dataset_format: str,
    text_column: Optional[str],
    audio_column: Optional[str],
    speaker_column: Optional[str],
    phonemes_column: Optional[str],
    lang_column: Optional[str],
    resume: bool,
    metrics_out: Optional[Path],
    metrics_in: Optional[Path],
    filter_from_columns: bool,
    output_dir: Path,
    language: str,
    prev_config: Path,
    drop_extra_phonemes: bool,
    corpus_only_map: bool,
    segment_map: bool,
    sample_rate: int,
    cache_dir: Optional[Path],
    max_workers: Optional[int],
    single_speaker: bool,
    speaker_id: Optional[int],
    phoneme_type: str,
    alphabet: str,
    phonemizer_model: str,
    text_casing: str,
    dataset_name: Optional[str],
    audio_quality: Optional[str],
    skip_audio: bool,
    debug: bool,
    add_diacritics: bool,
    keep_punctuation: bool,
    jsonl_audio_path: Optional[str],
    jsonl_audio_spec_path: Optional[str],
    engine: Optional[str],
    speaker_encoder_path: Optional[str],
    language_id: Optional[int],
    f0_method: str,
    quality_filters: Tuple[str, ...],
    vad_model: str,
    speaker_model: str,
    asr_model: str,
) -> None:
    """
    Preprocess a TTS dataset into a JSONL and config suitable for training a VITS-style model.
    
    Builds a phoneme map, phonemizes texts, optionally normalizes audio, and writes a phoonnx-compatible
    config.json and dataset.jsonl in the output directory.
    
    Parameters:
        input_dir (Path): Root directory of the input dataset (e.g., LJSpeech-style).
        output_dir (Path): Directory where output config and dataset files will be written.
        language (str): Language code used by the phonemizer.
        prev_config (Path): Path to a previous dataset config to load an existing phoneme map (for finetuning).
        drop_extra_phonemes (bool): If True, discard phonemes that differ from prev_config to allow finetuning.
        sample_rate (int): Target audio sample rate for normalization.
        cache_dir (Optional[Path]): Directory to store cached normalized audio and spectrograms (defaults to output_dir/cache/<sample_rate>).
        max_workers (Optional[int]): Number of worker processes to use for phonemization and audio processing (defaults to CPU count).
        single_speaker (bool): Treat the entire dataset as a single speaker (overrides per-utterance speaker labels).
        speaker_id (Optional[int]): Fixed speaker ID to assign to all utterances (cannot be used with --single-speaker).
        phoneme_type (str): Phoneme type identifier used to initialize the phonemizer.
        alphabet (str): Alphabet identifier (e.g., IPA) used by the phonemizer.
        phonemizer_model (str): Model name or identifier for the phonemizer.
        text_casing (str): Text casing transform to apply before phonemization (e.g., "lower", "upper", "casefold").
        dataset_name (Optional[str]): Optional dataset name to store in the generated config (defaults to output directory name).
        audio_quality (Optional[str]): Optional audio quality label stored in the generated config.
        skip_audio (bool): If True, skip audio processing and only phonemize text.
        debug (bool): Enable debug logging.
        add_diacritics (bool): Instruct the inference settings in the config to add diacritics.
        keep_punctuation (bool): Keep clause punctuation in the phonemes and record it in the config.
        jsonl_audio_path (Optional[str]): Optional base path override for audio paths written into dataset.jsonl.
        jsonl_audio_spec_path (Optional[str]): Optional base path override for the cache paths in dataset.jsonl.
        quality_filters (Tuple[str, ...]): Repeatable 'column:min:max' quality-metric filters (e.g. 'utmos:3.0:'),
            applied at manifest-load time before phonemization/audio caching. See phoonnx_train.quality_filter.
        vad_model (str): vadonnx model name used by the 'vad_ratio' quality filter.
        speaker_model (str): speakeronnx model name used by the 'speaker_consistency' quality filter.
        asr_model (str): Model identifier/path loadable via onnx_asr.load_model(), used by the 'wer' quality filter.

    Raises:
        click.Abort: If mutually exclusive CLI options are provided (e.g., both --single-speaker and --speaker-id).
        ValueError: If finetuning with a previous config and the new dataset contains phonemes not present in that config and drop_extra_phonemes is False.
    """
    if metrics_out or metrics_in:
        # Checked first: the sidecar is written after the filter metrics are computed.
        try:
            import pandas  # noqa: F401
            import pyarrow  # noqa: F401
        except ImportError as e:
            raise click.UsageError(f"--metrics-out and --metrics-in need pandas and pyarrow "
                                   f"({e.name} is missing); install phoonnx[train-data]") from None

    # Split any comma-separated sources into a flat list.
    sources: List[str] = [s.strip() for spec in input_sources for s in spec.split(",") if s.strip()]

    # Create a config object from click arguments for easier passing
    config = PreprocessorConfig(
        input_dir=Path(sources[0]) if sources else Path(""),
        output_dir=output_dir,
        language=language,
        sample_rate=sample_rate,
        cache_dir=cache_dir or output_dir / "cache" / str(sample_rate),
        max_workers=max_workers or os.cpu_count() or 1,
        single_speaker=single_speaker,
        speaker_id=speaker_id,
        phoneme_type=PhonemeType(phoneme_type),
        alphabet=Alphabet(alphabet),
        phonemizer_model=phonemizer_model,
        text_casing=text_casing,
        dataset_name=dataset_name,
        audio_quality=audio_quality,
        skip_audio=skip_audio,
        debug=debug,
        add_diacritics=add_diacritics,
        dataset_format=dataset_format,
        text_column=text_column,
        audio_column=audio_column,
        speaker_column=speaker_column,
        phonemes_column=phonemes_column,
        lang_column=lang_column,
        segment_map=segment_map,
    )

    # Setup logging
    level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)
    logging.getLogger("numba").setLevel(logging.WARNING)

    # Validation
    if config.single_speaker and (config.speaker_id is not None):
        _LOGGER.fatal("--single-speaker and --speaker-id cannot both be provided")
        raise click.Abort()

    if segment_map and importlib.util.find_spec("phonematcher") is None:
        raise click.UsageError("--segment-map needs phonematcher: install phoonnx[train-features]")
    if segment_map and prev_config:
        raise click.UsageError(
            "--segment-map cannot be combined with --prev-config: the previous "
            "config's phoneme map is kept, and its own tokenizer reads the corpus.")
    # the default seed table holds single code points, not segments
    corpus_only_map = corpus_only_map or segment_map
    if resume and corpus_only_map:
        raise click.UsageError(
            "--resume cannot be combined with --corpus-only-map: resuming "
            "reprocesses only new rows, so it cannot reconstruct a corpus-only "
            "phoneme map from the already-written rows (symbols occurring only "
            "in those rows would be lost). Rerun without --resume, or without "
            "--corpus-only-map."
        )

    # Create directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    # Load all utterances, merging multiple sources. When more than one source
    # is given, speaker labels are namespaced by source ("<tag>:<speaker>") so
    # identical speaker ids across datasets do not collide.
    _LOGGER.info("Loading utterances from %d source(s)...", len(sources))
    utterances: List[Utterance] = []
    audio_dirs: List[str] = []
    for src_index, source in enumerate(sources):
        source_tag = Path(source).stem or Path(source).name or source
        source_dir = Path(source).absolute() if Path(source).is_dir() else None
        source_speakers: Set[str] = set()
        for utt in load_source(source, config):
            if len(sources) > 1 and utt.speaker is not None:
                source_speakers.add(utt.speaker)
                utt.speaker = f"{source_tag}:{utt.speaker}"
            utterances.append(utt)
            if not jsonl_audio_path:
                continue
            if str(utt.audio_path) != ".":
                audio_dir = utt.audio_path.absolute().parent
                # inside a dataset directory, the audio folder is its top folder, such as wav/
                if source_dir is not None and source_dir in audio_dir.parents:
                    audio_dir = source_dir / audio_dir.relative_to(source_dir).parts[0]
                audio_dirs.append(str(audio_dir))
            elif utt.audio_bytes is not None and not config.skip_audio:
                # ensure_audio_path decodes these rows into <cache_dir>/audio
                audio_dirs.append(str((config.cache_dir / "audio").absolute()))
        if len(sources) > 1 and source_speakers:
            _LOGGER.info("Source %r speakers namespaced as %s", source,
                         ", ".join(f"{source_tag}:{s}" for s in sorted(source_speakers)))

    # A path override replaces the base folder of each audio path and keeps the
    # rest. The audio base is the parent of the folder that holds every audio
    # file. It comes from every loaded row, before --resume or a filter drops
    # any, so each run over the same sources writes the same base.
    audio_base: Optional[Path] = Path(os.path.commonpath(audio_dirs)).parent if audio_dirs else None

    if not utterances:
        raise click.ClickException("No valid utterances found in dataset.")

    # Resume: skip rows already written to an existing dataset.jsonl.
    dataset_path = config.output_dir / "dataset.jsonl"
    resume_rows: List[Dict[str, Any]] = []
    if resume and dataset_path.exists():
        resume_rows = _read_jsonl(dataset_path)
        done_keys = {_row_key(r) for r in resume_rows}
        kept = [u for u in utterances if _utt_key(u) not in done_keys]
        _LOGGER.info("Resume: %d rows already done, %d new rows to process.",
                     len(resume_rows), len(kept))
        utterances = kept
        if not utterances:
            _LOGGER.info("Resume: nothing new to process.")
            return
        if audio_base is not None:
            # the new rows have to share the base the existing rows were written with
            first_base = _written_audio_base(resume_rows, Path(jsonl_audio_path), config.cache_dir)
            if first_base is None or not Path(os.path.commonpath(audio_dirs)).is_relative_to(first_base):
                raise click.UsageError(
                    f"--resume would write a second audio base into {dataset_path}: its rows use "
                    f"{first_base or 'a base that cannot be read back'}, and the new rows need {audio_base}. "
                    "Write the new rows to another output folder.")
            audio_base = first_base
        elif not jsonl_audio_path and any(_has_rewritten_audio_path(r) for r in resume_rows):
            raise click.UsageError(
                f"--resume would mix two audio bases in {dataset_path}: its rows were written with "
                "--jsonl-audio-path. Pass the --jsonl-audio-path of the first run again.")

    if quality_filters:
        configure_vad_model(vad_model)
        configure_speaker_model(speaker_model)
        configure_asr_model(asr_model)
        specs: List[FilterSpec] = [parse_filter_spec(f) for f in quality_filters]
        total_before = len(utterances)

        sidecar: Dict[str, Dict[str, float]] = {}
        if metrics_in:
            sidecar = _read_metrics_sidecar(metrics_in)

        value_source = None
        if filter_from_columns:
            def value_source(utt: Utterance, column: str) -> Optional[float]:
                if column in utt.extras and utt.extras[column] is not None:
                    try:
                        return float(utt.extras[column])
                    except (TypeError, ValueError):
                        return None
                return sidecar.get(utt.row_id or "", {}).get(column)

        recorded: Dict[str, Dict[str, float]] = {}
        metrics_sink = None
        if metrics_out:
            def metrics_sink(row_id: str, column: str, value: float) -> None:
                recorded.setdefault(row_id, {})[column] = value

        utterances, dropped_counts = apply_quality_filters(
            utterances,
            specs,
            audio_path_fn=lambda u: ensure_audio_path(u, config.cache_dir),
            text_fn=lambda u: u.text,
            id_fn=lambda u: u.row_id or str(u.audio_path),
            value_source=value_source,
            metrics_sink=metrics_sink,
        )
        log_filter_summary(total_before, dropped_counts, len(utterances))
        if metrics_out and recorded:
            _write_metrics_sidecar(metrics_out, recorded)
            _LOGGER.info("Wrote metrics sidecar for %d rows to %s", len(recorded), metrics_out)
        if not utterances:
            # A row the filter dropped never reaches dataset.jsonl, so a resume
            # meets it again as new; the manifest it resumes stays as it is.
            if resume_rows:
                _LOGGER.info("Resume: the filters dropped every new row, nothing new to write.")
                return
            raise click.ClickException("No utterances left after quality filtering.")

    num_utterances: int = len(utterances)
    _LOGGER.info("Found %d utterances.", num_utterances)

    # Count speakers and assign IDs
    speaker_counts: Counter[str] = Counter(u.speaker for u in utterances if u.speaker)
    is_multispeaker: bool = len(speaker_counts) > 1
    speaker_ids: Dict[str, int] = {}
    if is_multispeaker:
        _LOGGER.info("%s speakers detected", len(speaker_counts))
        # Assign speaker ids by most number of utterances first
        for speaker_id, (speaker, _) in enumerate(speaker_counts.most_common()):
            speaker_ids[speaker] = speaker_id
    else:
        _LOGGER.info("Single speaker dataset")

    # --- Single Pass: Process audio/phonemes and collect results ---
    _LOGGER.info("Starting single pass processing with %d workers...", config.max_workers)

    # A resumed manifest must not mix rows phonemized with and without
    # punctuation, or in two casings. A config written before keep_punctuation
    # existed was without it; one written before text_casing existed does not
    # say, so the command line stands.
    existing_config_path = config.output_dir / "config.json"
    if resume_rows and existing_config_path.is_file():
        existing = json.loads(existing_config_path.read_text(encoding="utf-8"))
        resumed_punctuation = bool(existing.get("keep_punctuation", False))
        if resumed_punctuation != keep_punctuation:
            _LOGGER.warning("Resume: keeping keep_punctuation=%s from the existing config.json",
                            resumed_punctuation)
        keep_punctuation = resumed_punctuation
        resumed_casing = existing.get("text_casing", config.text_casing)
        if resumed_casing != config.text_casing:
            _LOGGER.warning("Resume: keeping text_casing=%s from the existing config.json",
                            resumed_casing)
        config.text_casing = resumed_casing

    make_frontend = functools.partial(build_frontend, config.phoneme_type, config.alphabet,
                                      config.phonemizer_model, keep_punctuation, config.language,
                                      config.add_diacritics)
    # The parent keeps its own phonemizer for the alphabet and the provenance record.
    phonemizer = _voice_phonemizer(config.phoneme_type,
                                   config.alphabet,
                                   config.phonemizer_model,
                                   keep_punctuation=keep_punctuation)

    if any(not utt.phonemes_precomputed for utt in utterances):
        # A worker that cannot build its frontend (an unsupported language, a
        # missing backend) dies before it reads a task, and the run would wait
        # for its results forever.
        make_frontend()

    if config.add_diacritics and any(not utt.phonemes_precomputed for utt in utterances):
        # In a worker this failure is per utterance, so a missing backend would
        # drop every row and still exit 0.
        diacritize("a", config.language)

    batch_size: int = max(1, int(num_utterances / (config.max_workers * 2)))

    task_queue: "JoinableQueue[Optional[List[Utterance]]]" = JoinableQueue()
    # The result queue will hold tuples of (Utterance, set(phonemes))
    result_queue: "Queue[Tuple[Optional[Utterance], Set[str]]]" = Queue()

    # Start workers
    processes: List[Process] = [
        Process(
            target=phonemize_worker,
            args=(config, task_queue, result_queue, make_frontend)
        )
        for _ in range(config.max_workers)
    ]

    for proc in processes:
        proc.start()

    # Populate the task queue with batches
    task_count: int = 0
    for utt_batch in batched(utterances, batch_size):
        task_queue.put(utt_batch)
        task_count += len(utt_batch)

    # Signal workers to stop
    for _ in range(config.max_workers):
        task_queue.put(None)

    # Collect results from the queue with a progress bar. Phonemes read from a
    # dataset column (precomputed) never expand the phoneme map: they are kept
    # aside and validated against the final map so a mismatched inventory fails
    # loudly instead of silently growing the vocab.
    processed_utterances: List[Utterance] = []
    all_phonemes: Set[str] = set()
    precomputed_symbols: Set[str] = set()
    num_from_column: int = 0
    num_phonemized: int = 0
    for _ in tqdm(range(task_count), desc="Processing utterances"):
        result: Tuple[Optional[Utterance], Set[str]] = next_result(result_queue, task_queue, processes)
        utt, unique_phonemes = result
        if utt is not None:
            processed_utterances.append(utt)
            if utt.phonemes_precomputed:
                precomputed_symbols.update(unique_phonemes)
                num_from_column += 1
            else:
                all_phonemes.update(unique_phonemes)
                num_phonemized += 1

    # Wait for workers to finish
    task_queue.join()
    for proc in processes:
        proc.join()

    if phonemes_column:
        _LOGGER.info("Phoneme source split: %d rows from column %r, %d phonemized.",
                     num_from_column, phonemes_column, num_phonemized)

    # --- Build the final phoneme map from the collected phonemes ---
    _LOGGER.info("Building a phoneme map from collected dataset phonemes...")
    corpus_phonemes: Set[str] = set(all_phonemes) | precomputed_symbols

    if prev_config:
        with open(prev_config) as f:
            cfg = json.load(f)
        # flatten list, same models (eg. piper) use a list of ids
        prev_phoneme_id_map = {k: v if not isinstance(v, list) else v[0]
                               for k, v in cfg["phoneme_id_map"].items()}

        prev_num_symbols = cfg.get("num_symbols", MAX_PHONEMES)
        _LOGGER.info(f"Loaded phoneme map from previous config: '{prev_config}'")
        # The previous voice's tokenizer reads the corpus as the fine-tuned voice
        # will, whether its map holds code points or whole segments.
        all_phonemes, precomputed_symbols, corpus_phonemes = tokenizer_reading(cfg, processed_utterances)
        all_phonemes.update(prev_phoneme_id_map.keys())
        final_phoneme_id_map = prev_phoneme_id_map
        _LOGGER.info("previous phoneme map contains %d phonemes.", len(final_phoneme_id_map))
    else:
        prev_num_symbols = MAX_PHONEMES
        final_phoneme_id_map: Dict[str, int] = DEFAULT_SPECIAL_PHONEME_ID_MAP.copy()
        all_phonemes = phoneme_map_seed(all_phonemes,
                                        ipa=phonemizer.alphabet == Alphabet.IPA,
                                        include_defaults=not corpus_only_map)

    # Filter out tokens that are already in the map
    existing_keys: Set[str] = set(final_phoneme_id_map.keys())
    new_phonemes: List[str] = sorted([p for p in all_phonemes
                                      if p not in existing_keys]
                                     )

    _LOGGER.info("Collected %d new phonemes.", len(new_phonemes))

    finetune_error = prev_config and len(new_phonemes)
    if finetune_error:
        if not drop_extra_phonemes:
            raise ValueError("training data contains different phonemes than previous phoneme map! Can not finetune model")
        else:
            _LOGGER.error("training data contains different phonemes than previous phoneme map! "
                          "Discarding new phonemes to still allow model finetuning")

    current_id: int = len(final_phoneme_id_map)
    for pho in new_phonemes:
        if finetune_error:
            _LOGGER.info(f"Discarded phoneme: {pho}")
        else:
            final_phoneme_id_map[pho] = current_id
            current_id += 1
            _LOGGER.debug(f"New phoneme: {pho}")

    unused = untrained_map_symbols(final_phoneme_id_map, corpus_phonemes)
    if unused:
        _LOGGER.warning(
            "%d phoneme map symbols never occur in the corpus and their embeddings will "
            "not be trained; feeding them at inference produces undefined audio: %s",
            len(unused), " ".join(unused))

    if new_phonemes:
        _LOGGER.info("Final phoneme map contains %d phonemes.", len(final_phoneme_id_map))

    # Precomputed (column) phonemes must already be representable in the final
    # map; a mismatched inventory fails loudly instead of expanding the vocab.
    if precomputed_symbols:
        missing = sorted(precomputed_symbols - set(final_phoneme_id_map))
        if missing:
            raise ValueError(
                f"precomputed phonemes from column {phonemes_column!r} contain "
                f"{len(missing)} symbol(s) absent from the final phoneme map: "
                f"{' '.join(missing)}"
            )

    # --- Write the final config.json ---
    _LOGGER.info("Writing dataset config...")
    audio_quality = config.audio_quality or config.output_dir.name
    dataset_name = config.dataset_name or config.output_dir.parent.name

    # Phonemes taken from a dataset column were not produced here, so nothing
    # about this environment reconstructs them. Naming a registry over them
    # would be worse than naming nothing: a reader would believe the column is
    # reproducible from a version that never touched it.
    config_data: Dict[str, Any] = {
        "dataset": dataset_name,
        "audio": {
            "sample_rate": config.sample_rate,
            "quality": audio_quality,
        },
        "lang_code": config.language,
        "inference": {"noise_scale": 0.667,
                      "length_scale": 1,
                      "noise_w": 0.8,
                      "add_diacritics": config.add_diacritics},
        "alphabet": phonemizer.alphabet.value,
        "phoneme_type": config.phoneme_type.value,
        "phonemizer_model": config.phonemizer_model,
        "keep_punctuation": keep_punctuation,
        "text_casing": config.text_casing,
        "phoneme_id_map": final_phoneme_id_map,
        "num_symbols": prev_num_symbols if prev_config else len(final_phoneme_id_map),
        "num_speakers": len(speaker_counts) if is_multispeaker else 1,
        "speaker_id_map": speaker_ids,
        "phoonnx_version": VERSION_STR,
    }
    if segment_map:
        # the tokenizer finds a segment by its longest key first
        config_data["fold_compounds"] = True

    # The previous config, read before it is overwritten: a resume needs both
    # its counts and the registry version that produced them.
    prior_config: Dict[str, Any] = {}
    prior_config_path = config.output_dir / "config.json"
    if resume_rows and prior_config_path.is_file():
        prior_config = json.loads(prior_config_path.read_text(encoding="utf-8"))

    # The cache path override replaces the parent of --cache-dir, or
    # --output-dir by default, and keeps the rest.
    cache_base = config.cache_dir.parent if cache_dir else config.output_dir

    # --- Apply final phoneme IDs and write dataset.jsonl ---
    # Writes go to a temp file and are atomically renamed into place so an
    # interrupted run never leaves a half-written manifest. With --resume the
    # already-processed rows are re-emitted first, then the new rows appended.
    # Counted here rather than off the result queue: three paths below drop a
    # collected utterance before it reaches the manifest, so queue-side counts
    # describe more rows than the file holds.
    _LOGGER.info("Writing dataset.jsonl...")
    valid_utterances_count: int = 0
    written_phonemized: int = 0
    written_from_column: int = 0

    tokenizer = TTSTokenizer.from_phoonnx_config(config_data)

    dataset_tmp = config.output_dir / "dataset.jsonl.tmp"
    with open(dataset_tmp, "w", encoding="utf-8") as dataset_file:
        for row in resume_rows:
            json.dump(row, dataset_file, ensure_ascii=False, cls=PathEncoder)
            print("", file=dataset_file)
            valid_utterances_count += 1

        training_engine = None
        engine_kwargs = {}
        if engine:
            from phoonnx_train.engines import get_engine

            training_engine = get_engine(engine)
            if speaker_encoder_path:
                engine_kwargs["speaker_encoder_path"] = speaker_encoder_path
            if language_id is not None:
                engine_kwargs["language_id"] = language_id
            if f0_method:
                engine_kwargs["f0_method"] = f0_method

        for utt in processed_utterances:
            if is_multispeaker and utt.speaker is not None:
                if utt.speaker not in speaker_ids:
                    _LOGGER.error("Speaker '%s' not in speaker_id_map. This indicates an issue with your metadata.csv file.", utt.speaker)
                    continue
                utt.speaker_id = speaker_ids[utt.speaker]

            # Apply the final phoneme ID map to each utterance
            if utt.phonemes:
                utt.phoneme_ids = tokenizer.tokenize(utt.phonemes)

            if not utt.phoneme_ids:
                _LOGGER.warning("Skipping utterance with invalid phoneme_ids before writing: %s", utt.audio_path)
                continue

            # Monotonic alignment needs at least one spectrogram frame per
            # phoneme id; a shorter spectrogram means the audio does not
            # contain the full text (e.g. a truncated clip)
            # The worker that wrote or read the spectrogram measured it; an
            # utterance without a count is read back from disk.
            if utt.audio_spec_path is not None:
                spec_frames = utt.spec_frames
                if spec_frames is None:
                    spec_frames = torch.load(utt.audio_spec_path, map_location="cpu").size(-1)
                if spec_frames < len(utt.phoneme_ids):
                    _LOGGER.warning(
                        "Skipping utterance with more phonemes (%d) than spectrogram frames (%d), "
                        "audio is too short for its text: %s",
                        len(utt.phoneme_ids), spec_frames, utt.audio_path)
                    continue

            if training_engine is not None and utt.audio_path:
                extra = training_engine.extra_preprocess(
                    Path(utt.audio_path), config.cache_dir,
                    sample_rate, **engine_kwargs)
                for key, value in extra.items():
                    setattr(utt, key, value)

            # path overrides, so a dataset preprocessed on one machine trains on another.
            # They run after the engine step, which must read the local audio.
            if audio_base is not None and str(utt.audio_path) != ".":
                utt.audio_path = Path(jsonl_audio_path) / utt.audio_path.absolute().relative_to(audio_base)
            row = utt.asdict()
            if jsonl_audio_spec_path:
                for key in _CACHE_PATH_FIELDS:
                    if row[key] and Path(row[key]).is_relative_to(cache_base):
                        row[key] = str(Path(jsonl_audio_spec_path) / Path(row[key]).relative_to(cache_base))

            json.dump(
                row,
                dataset_file,
                ensure_ascii=False,
                cls=PathEncoder,
            )
            print("", file=dataset_file)
            valid_utterances_count += 1
            if utt.phonemes_precomputed:
                written_from_column += 1
            else:
                written_phonemized += 1

    if not valid_utterances_count:
        dataset_tmp.unlink()
        raise click.ClickException("No utterance was written: every row failed phonemization "
                                   "or the audio checks. The log names each one.")
    dataset_tmp.rename(config.output_dir / "dataset.jsonl")

    # --- Record which phonemizer produced the phonemes ---
    #
    # Written after the manifest so the counts describe the file that exists
    # beside them, and so an interrupted run leaves no config claiming rows
    # that were never written.
    #
    # A resumed run re-emits rows it never processed, so its own counts cover
    # only part of the manifest. Fold the previous config's counts in and the
    # pair keeps describing the whole file across any number of resumes. Two
    # cases cannot be folded and then all five fields are omitted rather than
    # left describing a subset: a previous config predating these keys, and one
    # written by a different registry version or phonemizer engine, whose rows
    # this run did not produce and must not be named over.
    registry, registry_version = phonemizer_registry(phonemizer)
    # Only a run that phonemized rows asks the engine. A column-only resume adds
    # no rows of an engine's making, so it keeps the earlier rows' engine, or
    # records none when the earlier config did not name one.
    identity: Optional[Dict[str, Optional[str]]] = (
        phonemizer_identity(phonemizer, config.language) if written_phonemized
        else prior_config.get("phonemizer_identity"))
    phonemized_total: Optional[int] = written_phonemized
    from_column_total: Optional[int] = written_from_column
    if resume_rows:
        prior_phonemized = prior_config.get("phonemes_phonemized")
        # A previous run that phonemized nothing named no registry, so there are
        # no rows of its making to mis-attribute and its counts fold whatever
        # version wrote them.
        same_producer = (
            prior_phonemized == 0
            or (prior_config.get("phonemizer_registry") == registry
                and prior_config.get("phonemizer_registry_version") == registry_version
                and (not written_phonemized
                     or prior_config.get("phonemizer_identity") == identity))
        )
        if (prior_phonemized is not None
                and "phonemes_from_column" in prior_config
                and same_producer):
            phonemized_total += prior_phonemized
            from_column_total += prior_config["phonemes_from_column"]
        else:
            phonemized_total = from_column_total = None

    if phonemized_total is not None:
        # Nothing was phonemized here, so no producer is claimed: the phonemes
        # came out of a dataset column and this run cannot vouch for them.
        if not phonemized_total:
            registry, registry_version, identity = "", "", {}
        config_data["phonemizer_registry"] = registry
        config_data["phonemizer_registry_version"] = registry_version
        if identity is not None:
            config_data["phonemizer_identity"] = identity
        config_data["phonemes_phonemized"] = phonemized_total
        config_data["phonemes_from_column"] = from_column_total

    config_tmp = config.output_dir / "config.json.tmp"
    with open(config_tmp, "w", encoding="utf-8") as config_file:
        json.dump(config_data, config_file, ensure_ascii=False, indent=2)
    config_tmp.rename(config.output_dir / "config.json")

    _LOGGER.info("Preprocessing complete. Wrote %d valid utterances to dataset.jsonl.", valid_utterances_count)


# -----------------------------------------------------------------------------

def _row_key(row: Dict[str, Any]) -> str:
    """Resume identity for an existing dataset.jsonl row: row_id or audio path."""
    return str(row.get("row_id") or row.get("audio_path") or "")


def _utt_key(utt: Utterance) -> str:
    """Resume identity for a freshly loaded utterance: row_id or audio path."""
    return str(utt.row_id or utt.audio_path or "")


def _written_audio_base(rows: List[Dict[str, Any]], jsonl_audio_path: Path, cache_dir: Path) -> Optional[Path]:
    """The local audio base that *rows* were written with under *jsonl_audio_path*, or None.

    A row of an audio file keeps its local path in ``row_id``. A row of embedded
    audio was decoded to ``<cache_dir>/audio`` under the same file name.
    """
    for row in rows:
        written = Path(row.get("audio_path") or "")
        if written == jsonl_audio_path or not written.is_relative_to(jsonl_audio_path):
            continue
        tail = written.relative_to(jsonl_audio_path).parts
        for local in (Path(row.get("row_id") or ""), cache_dir / "audio" / tail[-1]):
            local = local.absolute()
            if local.parts[-len(tail):] == tail:
                return Path(*local.parts[:-len(tail)])
    return None


def _has_rewritten_audio_path(row: Dict[str, Any]) -> bool:
    """Whether *row* was written with --jsonl-audio-path.

    A row of an audio file keeps its local path in ``row_id``. A row of embedded
    audio does not, so it never counts: its decoded file name is the hash of its
    ``row_id`` under either base.
    """
    row_id = row.get("row_id")
    audio_path = Path(row.get("audio_path") or "")
    return (bool(row_id) and audio_path != Path("") and audio_path != Path(row_id)
            and audio_path.name != f"{sha256(row_id.encode('utf-8')).hexdigest()}.wav")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read existing dataset.jsonl rows, tolerating a truncated final line."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                _LOGGER.warning("Ignoring truncated/malformed row while resuming: %.60s", line)
    return rows


def _read_metrics_sidecar(path: Path) -> Dict[str, Dict[str, float]]:
    """Load a metrics parquet sidecar into {row_id: {column: value}}."""
    import pandas as pd
    frame = pd.read_parquet(path)
    out: Dict[str, Dict[str, float]] = {}
    for record in frame.to_dict(orient="records"):
        row_id = str(record.get("row_id", ""))
        out[row_id] = {k: float(v) for k, v in record.items()
                       if k != "row_id" and v is not None and not pd.isna(v)}
    return out


def _write_metrics_sidecar(path: Path, recorded: Dict[str, Dict[str, float]]) -> None:
    """Write per-row computed metric values to a parquet sidecar (atomic)."""
    import pandas as pd
    rows = [{"row_id": row_id, **values} for row_id, values in recorded.items()]
    tmp = Path(str(path) + ".tmp")
    pd.DataFrame(rows).to_parquet(tmp)
    tmp.rename(path)


def tokenizer_reading(cfg: Dict[str, Any], utterances: List[Utterance]) -> Tuple[Set[str], Set[str], Set[str]]:
    """How the tokenizer of *cfg* reads *utterances*: the symbols it cannot read
    in phonemized rows and in dataset-column rows, and the map keys it uses."""
    phonemized, from_column = TTSTokenizer.from_phoonnx_config(cfg), TTSTokenizer.from_phoonnx_config(cfg)
    used: Set[int] = set()
    for utt in utterances:
        tokenizer = from_column if utt.phonemes_precomputed else phonemized
        used.update(tokenizer.encode(utt.phonemes))
    keys = {k for k, v in cfg["phoneme_id_map"].items()
            if any(i in used for i in (v if isinstance(v, list) else [v]))}
    return set(phonemized.not_found_characters), set(from_column.not_found_characters), keys


def batched(iterable: Iterable[Any], n: int) -> Iterable[List[Any]]:
    """
    Batch data from an iterable into lists of length n. The last batch may be shorter.

    Args:
        iterable: The input iterable to be batched.
        n: The desired size of each batch.

    Yields:
        List[Any]: A list representing a batch of items.
    """
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    batch = list(itertools.islice(it, n))
    while batch:
        yield batch
        batch = list(itertools.islice(it, n))


if __name__ == "__main__":
    cli()