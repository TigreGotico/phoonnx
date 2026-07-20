#!/usr/bin/env python3
import itertools
import json
import logging
import os
from collections import Counter
from multiprocessing import JoinableQueue, Process, Queue
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any, Set, Union, Callable

import click
import torch
from tqdm import tqdm

from phoonnx.config import PhonemeType, get_phonemizer, Alphabet
from phoonnx.phonemizers import Phonemizer
from phoonnx.tokenizer import TTSTokenizer, DEFAULT_IPA_PHONEME_ID_MAP, DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, \
    phoneme_map_seed, untrained_map_symbols, \
    DEFAULT_EOS_TOKEN, DEFAULT_BLANK_WORD_TOKEN
from phoonnx.util import normalize
from phoonnx.version import VERSION_STR
from phoonnx_train.dataset_loaders import (PreprocessorConfig, Utterance,
                                           ensure_audio_path, get_text_casing,
                                           known_loaders, load_source)
from phoonnx_train.norm_audio import cache_norm_audio, make_silence_detector
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
# -----------------------------------------------------------------------------

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


def phonemize_worker(
        config: PreprocessorConfig,
        task_queue: JoinableQueue,
        result_queue: Queue,
        phonemizer: Phonemizer,
) -> None:
    """
    Worker process for phonemization and audio processing.

    Args:
        config: The configuration object containing runtime parameters.
        task_queue: Queue for receiving batches of Utterance objects.
        result_queue: Queue for sending processed results (Utterance, set of phonemes).
        phonemizer: The initialized Phonemizer instance.
    """
    try:
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
                        # Normalize text (case, numbers, etc.)
                        utterance: str = casing(normalize(utt.text, config.language))

                        # Add diacritics
                        if config.add_diacritics:
                            utterance = phonemizer.add_diacritics(utterance, config.language)

                        # Phonemize the text
                        utt.phonemes = [p for p in phonemizer.phonemize_to_list(utterance, config.language)
                                        if p != "\n"] # HACK: not sure where this is coming from
                        if not utt.phonemes:
                            raise RuntimeError(f"Phonemes not found for '{utterance}'")

                    # Process audio if not skipping
                    if not config.skip_audio:
                        audio_path = ensure_audio_path(utt, config.cache_dir)
                        utt.audio_norm_path, utt.audio_spec_path = cache_norm_audio(
                            audio_path,
                            config.cache_dir,
                            silence_detector,
                            config.sample_rate,
                        )

                    # Put the processed utterance and its phonemes into the result queue
                    # The result is a tuple of (Utterance, set of unique phonemes in that utterance)
                    result_queue.put((utt, set(utt.phonemes)))
                except Exception:
                    _LOGGER.exception("Failed to process utterance: %s", utt.audio_path)
                    result_queue.put((None, set()))

            task_queue.task_done()

    except Exception:
        _LOGGER.exception("Worker process failed")


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
    "--jsonl-audio-path",
    default=None,
    help="override audio_path base directory (everything before '/wav') in generated dataset.jsonl"
)
@click.option(
    "--jsonl-audio-spec-path",
    default=None,
    help="override audio_norm_path/audio_spec_path base directory (everything before '/cache') in generated dataset.jsonl"
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
    jsonl_audio_path: Optional[str],
    jsonl_audio_spec_path: Optional[str],
    engine: Optional[str],
    speaker_encoder_path: Optional[str],
    language_id: Optional[int],
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
        jsonl_audio_path (Optional[str]): Optional base path override for audio paths written into dataset.jsonl.
        jsonl_audio_spec_path (Optional[str]): Optional base path override for cached audio/spec paths in dataset.jsonl.
        quality_filters (Tuple[str, ...]): Repeatable 'column:min:max' quality-metric filters (e.g. 'utmos:3.0:'),
            applied at manifest-load time before phonemization/audio caching. See phoonnx_train.quality_filter.
        vad_model (str): vadonnx model name used by the 'vad_ratio' quality filter.
        speaker_model (str): speakeronnx model name used by the 'speaker_consistency' quality filter.
        asr_model (str): Model identifier/path loadable via onnx_asr.load_model(), used by the 'wer' quality filter.

    Raises:
        click.Abort: If mutually exclusive CLI options are provided (e.g., both --single-speaker and --speaker-id).
        ValueError: If finetuning with a previous config and the new dataset contains phonemes not present in that config and drop_extra_phonemes is False.
    """
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
    for src_index, source in enumerate(sources):
        source_tag = Path(source).stem or Path(source).name or source
        source_speakers: Set[str] = set()
        for utt in load_source(source, config):
            if len(sources) > 1 and utt.speaker is not None:
                source_speakers.add(utt.speaker)
                utt.speaker = f"{source_tag}:{utt.speaker}"
            utterances.append(utt)
        if len(sources) > 1 and source_speakers:
            _LOGGER.info("Source %r speakers namespaced as %s", source,
                         ", ".join(f"{source_tag}:{s}" for s in sorted(source_speakers)))

    if not utterances:
        _LOGGER.error("No valid utterances found in dataset.")
        return

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
            _LOGGER.error("No utterances left after quality filtering.")
            return

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

    # Initialize the phonemizer only once in the main process
    phonemizer: Phonemizer = get_phonemizer(config.phoneme_type,
                                            config.alphabet,
                                            config.phonemizer_model)

    batch_size: int = max(1, int(num_utterances / (config.max_workers * 2)))

    task_queue: "JoinableQueue[Optional[List[Utterance]]]" = JoinableQueue()
    # The result queue will hold tuples of (Utterance, set(phonemes))
    result_queue: "Queue[Tuple[Optional[Utterance], Set[str]]]" = Queue()

    # Start workers
    processes: List[Process] = [
        Process(
            target=phonemize_worker,
            args=(config, task_queue, result_queue, phonemizer)
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
        result: Tuple[Optional[Utterance], Set[str]] = result_queue.get()
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
        "phoneme_id_map": final_phoneme_id_map,
        "num_symbols": prev_num_symbols if prev_config else len(final_phoneme_id_map),
        "num_speakers": len(speaker_counts) if is_multispeaker else 1,
        "speaker_id_map": speaker_ids,
        "phoonnx_version": VERSION_STR,
    }

    config_tmp = config.output_dir / "config.json.tmp"
    with open(config_tmp, "w", encoding="utf-8") as config_file:
        json.dump(config_data, config_file, ensure_ascii=False, indent=2)
    config_tmp.rename(config.output_dir / "config.json")

    # --- Apply final phoneme IDs and write dataset.jsonl ---
    # Writes go to a temp file and are atomically renamed into place so an
    # interrupted run never leaves a half-written manifest. With --resume the
    # already-processed rows are re-emitted first, then the new rows appended.
    _LOGGER.info("Writing dataset.jsonl...")
    valid_utterances_count: int = 0

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
            if utt.audio_spec_path is not None:
                spec_frames = torch.load(utt.audio_spec_path, map_location="cpu").size(-1)
                if spec_frames < len(utt.phoneme_ids):
                    _LOGGER.warning(
                        "Skipping utterance with more phonemes (%d) than spectrogram frames (%d), "
                        "audio is too short for its text: %s",
                        len(utt.phoneme_ids), spec_frames, utt.audio_path)
                    continue

            # apply path overrides if needed
            # this allows pre-processing the dataset in one system and then train in other
            if jsonl_audio_path:
                base_path, fname = str(utt.audio_path).split("/wav/")
                utt.audio_path = Path(f"{jsonl_audio_path}/wav/{fname}")
            if jsonl_audio_spec_path:
                base_path, fname = str(utt.audio_norm_path).split("/cache/")
                utt.audio_norm_path = Path(f"{jsonl_audio_spec_path}/cache/{fname}")
                base_path, fname = str(utt.audio_spec_path).split("/cache/")
                utt.audio_spec_path = Path(f"{jsonl_audio_spec_path}/cache/{fname}")

            if training_engine is not None and utt.audio_path:
                extra = training_engine.extra_preprocess(
                    Path(utt.audio_path), config.cache_dir,
                    sample_rate, **engine_kwargs)
                for key, value in extra.items():
                    setattr(utt, key, value)

            json.dump(
                utt.asdict(),
                dataset_file,
                ensure_ascii=False,
                cls=PathEncoder,
            )
            print("", file=dataset_file)
            valid_utterances_count += 1

    dataset_tmp.rename(config.output_dir / "dataset.jsonl")
    _LOGGER.info("Preprocessing complete. Wrote %d valid utterances to dataset.jsonl.", valid_utterances_count)


# -----------------------------------------------------------------------------

def _row_key(row: Dict[str, Any]) -> str:
    """Resume identity for an existing dataset.jsonl row: row_id or audio path."""
    return str(row.get("row_id") or row.get("audio_path") or "")


def _utt_key(utt: Utterance) -> str:
    """Resume identity for a freshly loaded utterance: row_id or audio path."""
    return str(utt.row_id or utt.audio_path or "")


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