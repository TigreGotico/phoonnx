#!/usr/bin/env python3
import csv
import dataclasses
import itertools
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from multiprocessing import JoinableQueue, Process, Queue
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any, Set, Union, Callable

import click
from phoonnx.util import normalize
from phoonnx.config import PhonemeType, get_phonemizer, Alphabet
from phoonnx.phonemizers import Phonemizer
from phoonnx.phoneme_ids import (
    phonemes_to_ids, DEFAULT_IPA_PHONEME_ID_MAP, DEFAULT_PAD_TOKEN,
    DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BLANK_WORD_TOKEN
)
from phoonnx_train.norm_audio import cache_norm_audio, make_silence_detector
from tqdm import tqdm
from phoonnx.version import VERSION_STR

_LOGGER = logging.getLogger("preprocess")

# Base phoneme map
DEFAULT_SPECIAL_PHONEME_ID_MAP: Dict[str, int] = {
    DEFAULT_PAD_TOKEN: 0,
    DEFAULT_BOS_TOKEN: 1,
    DEFAULT_EOS_TOKEN: 2,
    DEFAULT_BLANK_WORD_TOKEN: 3,
}

# -----------------------------------------------------------------------------

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

    def asdict(self) -> Dict[str, Any]:
        """Custom asdict to handle Path objects for JSON serialization."""
        data = dataclasses.asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        return data


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


def ljspeech_dataset(config: PreprocessorConfig) -> Iterable[Utterance]:
    """
    Generator for LJSpeech-style dataset.
    Loads metadata and resolves audio file paths.

    Args:
        config: The configuration object containing dataset parameters.

    Yields:
        Utterance: A fully populated Utterance object.
    """
    dataset_dir = config.input_dir
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

            # Ensure wav_path is Path or None, and is never accessed if skip_audio is true
            yield Utterance(
                text=text,
                audio_path=wav_path or Path(""), # Use empty path if skipping audio, should not be used
                speaker=speaker,
                speaker_id=config.speaker_id,
            )


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
                    # Normalize text (case, numbers, etc.)
                    utterance: str = casing(normalize(utt.text, config.language))

                    # Add diacritics
                    if config.add_diacritics:
                        utterance = phonemizer.add_diacritics(utterance, config.language)

                    # Phonemize the text
                    utt.phonemes = phonemizer.phonemize_to_list(utterance, config.language)
                    if not utt.phonemes:
                        raise RuntimeError(f"Phonemes not found for '{utterance}'")

                    # Process audio if not skipping
                    if not config.skip_audio:
                        utt.audio_norm_path, utt.audio_spec_path = cache_norm_audio(
                            utt.audio_path,
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
    "input_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory with audio dataset (e.g., containing metadata.csv and wavs/)",
)
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
    "-r",
    "--sample-rate",
    "sample_rate",
    type=int,
    required=True,
    help="Target sample rate for voice (hertz, e.g., 22050)",
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
    type=int,
    default=None,
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
def cli(
    input_dir: Path,
    output_dir: Path,
    language: str,
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
) -> None:
    """
    Preprocess a TTS dataset (e.g., LJSpeech format) for training a VITS-style model.
    This script handles text normalization, phonemization, and optional audio caching.
    """
    # Create a config object from click arguments for easier passing
    config = PreprocessorConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        language=language,
        sample_rate=sample_rate,
        cache_dir=cache_dir or output_dir / "cache" / str(sample_rate),
        max_workers=max_workers or os.cpu_count(),
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

    # Create directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    # Load all utterances from the dataset
    _LOGGER.info("Loading utterances from dataset...")
    utterances: List[Utterance] = list(ljspeech_dataset(config))
    if not utterances:
        _LOGGER.error("No valid utterances found in dataset.")
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

    # Collect results from the queue with a progress bar
    processed_utterances: List[Utterance] = []
    all_phonemes: Set[str] = set()
    for _ in tqdm(range(task_count), desc="Processing utterances"):
        result: Tuple[Optional[Utterance], Set[str]] = result_queue.get()
        utt, unique_phonemes = result
        if utt is not None:
            processed_utterances.append(utt)
            all_phonemes.update(unique_phonemes)

    # Wait for workers to finish
    task_queue.join()
    for proc in processes:
        proc.join()

    # --- Build the final phoneme map from the collected phonemes ---
    _LOGGER.info("Building a complete phoneme map from collected phonemes...")

    final_phoneme_id_map: Dict[str, int] = DEFAULT_SPECIAL_PHONEME_ID_MAP.copy()
    if phonemizer.alphabet == Alphabet.IPA:
        all_phonemes.update(DEFAULT_IPA_PHONEME_ID_MAP.keys())

    # Filter out special tokens that are already in the map
    existing_keys: Set[str] = set(final_phoneme_id_map.keys())
    new_phonemes: List[str] = sorted([p for p in all_phonemes if p not in existing_keys])

    current_id: int = len(final_phoneme_id_map)
    for pho in new_phonemes:
        final_phoneme_id_map[pho] = current_id
        current_id += 1

    _LOGGER.info("Final phoneme map contains %d symbols.", len(final_phoneme_id_map))

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
        "num_symbols": len(final_phoneme_id_map),
        "num_speakers": len(speaker_counts) if is_multispeaker else 1,
        "speaker_id_map": speaker_ids,
        "phoonnx_version": VERSION_STR,
    }

    with open(config.output_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(config_data, config_file, ensure_ascii=False, indent=2)

    # --- Apply final phoneme IDs and write dataset.jsonl ---
    _LOGGER.info("Writing dataset.jsonl...")
    valid_utterances_count: int = 0
    with open(config.output_dir / "dataset.jsonl", "w", encoding="utf-8") as dataset_file:
        for utt in processed_utterances:
            if is_multispeaker and utt.speaker is not None:
                if utt.speaker not in speaker_ids:
                    _LOGGER.error("Speaker '%s' not in speaker_id_map. This indicates an issue with your metadata.csv file.", utt.speaker)
                    continue
                utt.speaker_id = speaker_ids[utt.speaker]

            # Apply the final phoneme ID map to each utterance
            if utt.phonemes:
                utt.phoneme_ids = phonemes_to_ids(utt.phonemes, id_map=final_phoneme_id_map)

            if not utt.phoneme_ids:
                _LOGGER.warning("Skipping utterance with invalid phoneme_ids before writing: %s", utt.audio_path)
                continue

            json.dump(
                utt.asdict(),
                dataset_file,
                ensure_ascii=False,
                cls=PathEncoder,
            )
            print("", file=dataset_file)
            valid_utterances_count += 1

    _LOGGER.info("Preprocessing complete. Wrote %d valid utterances to dataset.jsonl.", valid_utterances_count)


# -----------------------------------------------------------------------------

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
