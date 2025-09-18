#!/usr/bin/env python3
import argparse
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
from typing import Dict, Iterable, List, Optional, Tuple, Any, Set, Union

from phoonnx.util import normalize
from phoonnx.config import PhonemeType, get_phonemizer, Alphabet
from phoonnx.phonemizers import Phonemizer
from phoonnx.phoneme_ids import (phonemes_to_ids, DEFAULT_IPA_PHONEME_ID_MAP, DEFAULT_PAD_TOKEN,
                                 DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BLANK_WORD_TOKEN)
from phoonnx_train.norm_audio import cache_norm_audio, make_silence_detector
from tqdm import tqdm

_VERSION = "0.0.0"
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
        """Custom asdict to handle Path objects."""
        data = dataclasses.asdict(self)
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        return data


class PathEncoder(json.JSONEncoder):
    """JSON encoder for Path objects."""

    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def get_text_casing(casing: str):
    """Returns a function to apply text casing based on a string."""
    if casing == "lower":
        return str.lower
    if casing == "upper":
        return str.upper
    if casing == "casefold":
        return str.casefold
    return lambda s: s


def ljspeech_dataset(args: argparse.Namespace) -> Iterable[Utterance]:
    """
    Generator for LJSpeech-style dataset.
    Loads metadata and resolves audio file paths.
    """
    dataset_dir = args.input_dir
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        _LOGGER.error(f"Missing metadata file: {metadata_path}")
        return

    wav_dirs = [dataset_dir / "wav", dataset_dir / "wavs"]

    with open(metadata_path, "r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter="|")
        for row in reader:
            if len(row) < 2:
                _LOGGER.warning(f"Skipping malformed row: {row}")
                continue

            filename: str = row[0]
            text: str = row[-1]
            speaker: Optional[str] = None

            if not args.single_speaker and len(row) > 2:
                speaker = row[1]
            else:
                speaker = None

            wav_path = None
            for wav_dir in wav_dirs:
                potential_paths = [wav_dir / filename,
                                   wav_dir / f"{filename}.wav",
                                   wav_dir / f"{filename.lstrip('0')}.wav"]
                for path in potential_paths:
                    if path.exists():
                        wav_path = path
                        break
                if wav_path:
                    break

            if not args.skip_audio and not wav_path:
                _LOGGER.warning("Missing audio file for filename: %s", filename)
                continue

            if not args.skip_audio and wav_path and wav_path.stat().st_size == 0:
                _LOGGER.warning("Empty audio file: %s", wav_path)
                continue

            yield Utterance(
                text=text,
                audio_path=wav_path,
                speaker=speaker,
                speaker_id=args.speaker_id,
            )


def phonemize_worker(
        args: argparse.Namespace,
        task_queue: JoinableQueue,
        result_queue: Queue,
        phonemizer: Phonemizer,
):
    """
    Worker process for phonemization and audio processing.
    Returns the utterance and the unique phonemes found in its batch.
    """
    try:
        casing = get_text_casing(args.text_casing)
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
                    # Phonemize the text
                    norm_utt = casing(normalize(utt.text, args.language))
                    utt.phonemes = phonemizer.phonemize_to_list(norm_utt, args.language)
                    if not utt.phonemes:
                        raise RuntimeError(f"Phonemes not found for '{norm_utt}'")
                    # Process audio if not skipping
                    if not args.skip_audio:
                        utt.audio_norm_path, utt.audio_spec_path = cache_norm_audio(
                            utt.audio_path,
                            args.cache_dir,
                            silence_detector,
                            args.sample_rate,
                        )

                    # Put the processed utterance and its phonemes into the result queue
                    result_queue.put((utt, set(utt.phonemes)))
                except Exception:
                    _LOGGER.exception("Failed to process utterance: %s", utt.audio_path)
                    result_queue.put((None, set()))

            task_queue.task_done()

    except Exception:
        _LOGGER.exception("Worker process failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess a TTS dataset for training a VITS-style model."
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory with audio dataset"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write output files for training",
    )
    parser.add_argument("--language", required=True, help="eSpeak-ng voice")
    parser.add_argument(
        "--sample-rate",
        type=int,
        required=True,
        help="Target sample rate for voice (hertz)",
    )
    parser.add_argument("--cache-dir", help="Directory to cache processed audio files")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument(
        "--single-speaker", action="store_true", help="Force single speaker dataset"
    )
    parser.add_argument(
        "--speaker-id", type=int, help="Add speaker id to single speaker dataset"
    )
    parser.add_argument(
        "--phoneme-type",
        choices=list(PhonemeType),
        default=PhonemeType.ESPEAK,
        help="Type of phonemes to use (default: espeak)",
    )
    parser.add_argument(
        "--alphabet",
        choices=list(Alphabet),
        default=Alphabet.IPA,
        help="Casing applied to utterance text",
    )
    parser.add_argument(
        "--phonemizer-model",
        default="",
        help="phonemizer model, if applicable",
    )
    parser.add_argument(
        "--text-casing",
        choices=("ignore", "lower", "upper", "casefold"),
        default="ignore",
        help="Casing applied to utterance text",
    )
    parser.add_argument(
        "--dataset-name",
        help="Name of dataset to put in config (default: name of <ouput_dir>/../)",
    )
    parser.add_argument(
        "--audio-quality",
        help="Audio quality to put in config (default: name of <output_dir>)",
    )
    parser.add_argument(
        "--skip-audio", action="store_true", help="Don't preprocess audio"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    # Setup
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)
    logging.getLogger("numba").setLevel(logging.WARNING)

    if args.single_speaker and (args.speaker_id is not None):
        _LOGGER.fatal("--single-speaker and --speaker-id cannot both be provided")
        return

    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else args.output_dir / "cache" / str(args.sample_rate)
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.phoneme_type = PhonemeType(args.phoneme_type)

    # Load all utterances from the dataset
    _LOGGER.info("Loading utterances from dataset...")
    utterances = list(ljspeech_dataset(args))
    if not utterances:
        _LOGGER.error("No valid utterances found in dataset.")
        return

    num_utterances = len(utterances)
    _LOGGER.info("Found %d utterances.", num_utterances)

    # Count speakers
    speaker_counts: Counter[str] = Counter(u.speaker for u in utterances if u.speaker)
    is_multispeaker = len(speaker_counts) > 1
    speaker_ids: Dict[str, int] = {}
    if is_multispeaker:
        _LOGGER.info("%s speakers detected", len(speaker_counts))
        # Assign speaker ids by most number of utterances first
        for speaker_id, (speaker, _) in enumerate(speaker_counts.most_common()):
            speaker_ids[speaker] = speaker_id
    else:
        _LOGGER.info("Single speaker dataset")

    # --- Single Pass: Process audio/phonemes and collect results ---
    # Set up multiprocessing
    args.max_workers = args.max_workers if args.max_workers is not None and args.max_workers > 0 else os.cpu_count()
    _LOGGER.info("Starting single pass processing with %d workers...", args.max_workers)

    # Initialize the phonemizer only once in the main process
    phonemizer = get_phonemizer(args.phoneme_type, args.alphabet, args.phonemizer_model)

    batch_size = max(1, int(num_utterances / (args.max_workers * 2)))

    task_queue: "Queue[Optional[List[Utterance]]]" = JoinableQueue()
    # The result queue will hold tuples of (Utterance, set(phonemes))
    result_queue: "Queue[Optional[Tuple[Utterance, Set[str]]]]" = Queue()

    # Start workers
    processes = [
        Process(
            target=phonemize_worker,
            args=(args, task_queue, result_queue, phonemizer)
        )
        for _ in range(args.max_workers)
    ]

    for proc in processes:
        proc.start()

    # Populate the task queue with batches
    task_count = 0
    for utt_batch in batched(utterances, batch_size):
        task_queue.put(utt_batch)
        task_count += len(utt_batch)

    # Signal workers to stop
    for _ in range(args.max_workers):
        task_queue.put(None)

    # Collect results from the queue with a progress bar
    processed_utterances: List[Utterance] = []
    all_phonemes: Set[str] = set()
    for _ in tqdm(range(task_count), desc="Processing utterances"):
        utt, unique_phonemes = result_queue.get()
        if utt is not None:
            processed_utterances.append(utt)
            all_phonemes.update(unique_phonemes)

    # Wait for workers to finish
    task_queue.join()
    for proc in processes:
        proc.join()

    # --- Build the final phoneme map from the collected phonemes ---
    _LOGGER.info("Building a complete phoneme map from collected phonemes...")

    final_phoneme_id_map = DEFAULT_SPECIAL_PHONEME_ID_MAP.copy()
    if phonemizer.alphabet == Alphabet.IPA:
        all_phonemes.update(DEFAULT_IPA_PHONEME_ID_MAP.keys())

    # Filter out special tokens that are already in the map
    existing_keys = set(final_phoneme_id_map.keys())
    new_phonemes = sorted([p for p in all_phonemes if p not in existing_keys])

    current_id = len(final_phoneme_id_map)
    for pho in new_phonemes:
        final_phoneme_id_map[pho] = current_id
        current_id += 1

    _LOGGER.info("Final phoneme map contains %d symbols.", len(final_phoneme_id_map))

    # --- Write the final config.json ---
    _LOGGER.info("Writing dataset config...")
    audio_quality = args.audio_quality or args.output_dir.name
    dataset_name = args.dataset_name or args.output_dir.parent.name

    config = {
        "dataset": dataset_name,
        "audio": {
            "sample_rate": args.sample_rate,
            "quality": audio_quality,
        },
        "lang_code": args.language,
        "inference": {"noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8},
        "alphabet": phonemizer.alphabet.value,
        "phoneme_type": args.phoneme_type.value,
        "phonemizer_model": args.phonemizer_model,
        "phoneme_id_map": final_phoneme_id_map,
        "num_symbols": len(final_phoneme_id_map),
        "num_speakers": len(speaker_counts) if is_multispeaker else 1,
        "speaker_id_map": speaker_ids,
        "phoonnx_version": _VERSION,
    }

    with open(args.output_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=2)

    # --- Apply final phoneme IDs and write dataset.jsonl ---
    _LOGGER.info("Writing dataset.jsonl...")
    valid_utterances_count = 0
    with open(args.output_dir / "dataset.jsonl", "w", encoding="utf-8") as dataset_file:
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

def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    batch = list(itertools.islice(it, n))
    while batch:
        yield batch
        batch = list(itertools.islice(it, n))


if __name__ == "__main__":
    main()
