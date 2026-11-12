"""Rápido training data over the shared phoonnx preprocessed format.

Each utterance of ``dataset.jsonl`` gives the phoneme ids written by
``phoonnx_train.preprocess`` and the cached, silence-trimmed audio. Rápido
reads its targets as VoxCPM2 AudioVAE latents: the trimmed audio is resampled
to 16 kHz and encoded to the encoder mean, ``[T, 64]`` at 25 Hz, cached as
``<sha>.rapido-latent.pt`` beside the ``<sha>.pt`` audio cache.

The model sees the preprocess ids without the interleaved blank token: Rápido
reads one embedding per letter and learns per-letter durations, so a blank
between every pair of phonemes would double the sequence with tokens that
carry no sound of their own. BOS and EOS stay. The exported voice config turns
blank insertion off, so the runtime tokenizer produces the same ids.
"""
import json
import logging
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from phoonnx.tokenizer import Vocabulary
from phoonnx_train.matcha.jsonl import load_dataset_lines

_LOGGER = logging.getLogger(__name__)

VAE_SAMPLE_RATE = 16000
LATENT_DIM = 64
STATS_FILE = "rapido_latent_stats.json"

# tokenizer flags of a Rápido voice: no blanks, BOS/EOS kept
RAPIDO_TOKENIZER_FLAGS: Dict[str, Any] = {
    "add_blank_char": False,
    "blank_at_start": False,
    "blank_at_end": False,
    "use_eos_bos": True,
    "add_blank_word": False,
    "word_sep_token": " ",
}


def blank_id(phoonnx_config: Mapping[str, Any]) -> Optional[int]:
    """Id of the blank token that ``preprocess.py`` interleaves, if any."""
    return Vocabulary.from_phoonnx_config(dict(phoonnx_config)).blank_id


def strip_blanks(ids: Sequence[int], blank: Optional[int]) -> List[int]:
    """Preprocess ids → the ids a Rápido model reads (blank tokens removed)."""
    return [int(i) for i in ids if i != blank]


def latent_cache_path(audio_norm_path: Path) -> Path:
    """``<sha>.pt`` normalized-audio cache → ``<sha>.rapido-latent.pt``."""
    return Path(audio_norm_path).with_suffix(".rapido-latent.pt")


def norm_audio_path(audio_path: Path, cache_dir: Path) -> Path:
    """The ``cache_norm_audio`` cache file of ``audio_path`` (same id scheme)."""
    cache_id = sha256(str(Path(audio_path).absolute()).encode()).hexdigest()
    return Path(cache_dir) / f"{cache_id}.pt"


@torch.no_grad()
def encode_latents(vae: torch.nn.Module, audio: torch.Tensor,
                   sample_rate: int) -> torch.Tensor:
    """Mono waveform ``[N]`` at ``sample_rate`` → VAE encoder mean ``[T, 64]``."""
    import torchaudio

    audio = audio.float().reshape(-1)
    if sample_rate != VAE_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, VAE_SAMPLE_RATE)
    device = next(vae.parameters()).device
    mu = vae.encode(audio.to(device)[None, None], VAE_SAMPLE_RATE)
    return mu[0].transpose(0, 1).float().cpu().contiguous()


def cache_latents(vae: torch.nn.Module, audio_norm_path: Path,
                  sample_rate: int) -> Path:
    """Encode one cached normalized-audio tensor and save its latents."""
    out = latent_cache_path(audio_norm_path)
    if not out.is_file():
        audio = torch.load(audio_norm_path, map_location="cpu")
        torch.save(encode_latents(vae, audio, sample_rate), out)
    return out


def dataset_jsonl(path: Path) -> Path:
    path = Path(path)
    return path / "dataset.jsonl" if path.is_dir() else path


def load_utterances(dataset_paths: Iterable[Path],
                    max_phoneme_ids: Optional[int] = None) -> List[Dict[str, Any]]:
    utterances: List[Dict[str, Any]] = []
    for path in dataset_paths:
        utterances.extend(load_dataset_lines(dataset_jsonl(path),
                                             max_phoneme_ids=max_phoneme_ids))
    return utterances


def ensure_latents(utterances: Sequence[Mapping[str, Any]], sample_rate: int,
                   device: str = "cpu") -> int:
    """Encode every utterance whose latent cache is missing; returns the count.

    The VAE is loaded only when at least one cache is missing, so a dataset
    preprocessed with ``--engine rapido`` never loads it here.
    """
    missing = [u for u in utterances
               if not latent_cache_path(u["audio_norm_path"]).is_file()]
    if not missing:
        return 0
    from phoonnx_train.rapido.audio_vae import load_audio_vae

    _LOGGER.info("encoding %d utterances to AudioVAE latents...", len(missing))
    vae = load_audio_vae(device=device)
    for utt in missing:
        cache_latents(vae, Path(utt["audio_norm_path"]), sample_rate)
    return len(missing)


def latent_statistics(utterances: Sequence[Mapping[str, Any]]) -> Tuple[List[float], List[float]]:
    """Per-dimension mean and std of the cached latents over every frame."""
    total = torch.zeros(LATENT_DIM, dtype=torch.float64)
    total_sq = torch.zeros(LATENT_DIM, dtype=torch.float64)
    frames = 0
    for utt in utterances:
        lat = torch.load(latent_cache_path(utt["audio_norm_path"])).double()
        total += lat.sum(0)
        total_sq += (lat ** 2).sum(0)
        frames += lat.shape[0]
    mean = total / frames
    std = (total_sq / frames - mean ** 2).clamp(min=1e-10).sqrt()
    return mean.tolist(), std.tolist()


def load_or_compute_statistics(dataset_paths: Sequence[Path],
                               utterances: Sequence[Mapping[str, Any]]) -> Tuple[List[float], List[float]]:
    """Latent statistics, cached as ``rapido_latent_stats.json`` beside the data."""
    first = Path(dataset_paths[0])
    cache = (first if first.is_dir() else first.parent) / STATS_FILE
    if cache.is_file():
        stats = json.loads(cache.read_text(encoding="utf-8"))
        if stats.get("utterances") == len(utterances):
            return stats["lat_mean"], stats["lat_std"]
    mean, std = latent_statistics(utterances)
    cache.write_text(json.dumps({"utterances": len(utterances),
                                 "lat_mean": mean, "lat_std": std}),
                     encoding="utf-8")
    return mean, std


class RapidoDataset(Dataset):
    """Yields (letter ids, raw latents ``[T, 64]``, speaker id)."""

    def __init__(self, utterances: Sequence[Mapping[str, Any]],
                 blank: Optional[int]) -> None:
        if not utterances:
            raise ValueError("no utterances to train on")
        self.utterances = list(utterances)
        self.blank = blank

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        utt = self.utterances[idx]
        return {
            "ids": torch.LongTensor(strip_blanks(utt["phoneme_ids"], self.blank)),
            "latents": torch.load(latent_cache_path(utt["audio_norm_path"])),
            "speaker": int(utt.get("speaker_id") or 0),
        }


def collate_rapido(batch: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
    """Zero-pad ids and latents; masks mark the real positions."""
    id_lens = torch.LongTensor([len(item["ids"]) for item in batch])
    lat_lens = torch.LongTensor([item["latents"].shape[0] for item in batch])
    ids = torch.zeros(len(batch), int(id_lens.max()), dtype=torch.long)
    latents = torch.zeros(len(batch), int(lat_lens.max()), LATENT_DIM)
    for i, item in enumerate(batch):
        ids[i, :id_lens[i]] = item["ids"]
        latents[i, :lat_lens[i]] = item["latents"]
    return {
        "ids": ids,
        "mask": torch.arange(ids.shape[1])[None] < id_lens[:, None],
        "latents": latents,
        "fmask": torch.arange(latents.shape[1])[None] < lat_lens[:, None],
        "speakers": torch.LongTensor([item["speaker"] for item in batch]),
    }
