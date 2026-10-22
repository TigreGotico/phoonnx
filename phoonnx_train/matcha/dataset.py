"""Text→mel dataset over the shared phoonnx preprocessed format.

Reads the same ``dataset.jsonl`` produced by ``phoonnx_train.preprocess``
(``phoneme_ids`` + ``audio_norm_path``) and computes the 80-bin log-mel
spectrogram Matcha-TTS trains on from the cached normalized audio.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from phoonnx_train.matcha.audio import mel_spectrogram
from phoonnx_train.matcha.jsonl import load_dataset_lines
from phoonnx_train.matcha.utils import fix_len_compatibility, normalize

_LOGGER = logging.getLogger(__name__)

# Matcha-TTS mel front-end (matches upstream configs/data/*.yaml and the
# Vocos/HiFi-GAN vocoders used at inference time)
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
F_MIN = 0
F_MAX = 8000


class MatchaDataset(Dataset):
    """Yields (phoneme_ids, normalized mel) pairs."""

    def __init__(
        self,
        dataset_paths: Sequence[Path],
        n_feats: int = 80,
        sample_rate: int = 22050,
        mel_mean: float = 0.0,
        mel_std: float = 1.0,
        max_phoneme_ids: Optional[int] = None,
    ) -> None:
        self.n_feats = n_feats
        self.sample_rate = sample_rate
        self.mel_mean = mel_mean
        self.mel_std = mel_std
        self.utterances: List[Dict] = []
        for dataset_path in dataset_paths:
            jsonl_path = Path(dataset_path)
            if jsonl_path.is_dir():
                jsonl_path = jsonl_path / "dataset.jsonl"
            self.utterances.extend(
                load_dataset_lines(jsonl_path, max_phoneme_ids=max_phoneme_ids)
            )
        if not self.utterances:
            raise ValueError(f"no utterances loaded from {list(dataset_paths)}")

    def mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Log-mel spectrogram [n_feats, T] of a mono waveform [T] in [-1, 1]."""
        return mel_spectrogram(
            audio.unsqueeze(0),
            N_FFT,
            self.n_feats,
            self.sample_rate,
            HOP_LENGTH,
            WIN_LENGTH,
            F_MIN,
            F_MAX,
            center=False,
        ).squeeze(0)

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        utt = self.utterances[idx]
        audio = torch.load(utt["audio_norm_path"]).float().squeeze()
        mel = normalize(self.mel(audio), self.mel_mean, self.mel_std)
        item = {
            "x": torch.LongTensor(utt["phoneme_ids"]),
            "y": mel,
        }
        if "speaker_id" in utt and utt["speaker_id"] is not None:
            item["spk"] = torch.LongTensor([int(utt["speaker_id"])])
        return item


def collate_matcha(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Zero-pad a batch; mel time axis is padded to a UNet-compatible length."""
    x_lengths = torch.LongTensor([item["x"].shape[0] for item in batch])
    y_lengths = torch.LongTensor([item["y"].shape[1] for item in batch])
    x_max = int(x_lengths.max())
    y_max = fix_len_compatibility(y_lengths.max())
    n_feats = batch[0]["y"].shape[0]

    x = torch.zeros(len(batch), x_max, dtype=torch.long)
    y = torch.zeros(len(batch), n_feats, y_max)
    for i, item in enumerate(batch):
        x[i, : item["x"].shape[0]] = item["x"]
        y[i, :, : item["y"].shape[1]] = item["y"]

    out = {
        "x": x,
        "x_lengths": x_lengths,
        "y": y,
        "y_lengths": y_lengths,
        "spks": None,
        "durations": None,
    }
    if "spk" in batch[0]:
        out["spks"] = torch.cat([item["spk"] for item in batch])
    return out


def compute_data_statistics(dataset: MatchaDataset) -> Tuple[float, float]:
    """Mean/std of the raw (un-normalized) log-mel over the whole dataset."""
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    for utt in dataset.utterances:
        audio = torch.load(utt["audio_norm_path"]).float().squeeze()
        mel = dataset.mel(audio)
        total_sum += float(mel.sum())
        total_sq_sum += float((mel**2).sum())
        total_count += mel.numel()
    mean = total_sum / total_count
    std = (total_sq_sum / total_count - mean**2) ** 0.5
    return mean, std


def load_or_compute_statistics(dataset_paths: Sequence[Path], dataset: MatchaDataset) -> Tuple[float, float]:
    """Dataset mel statistics, cached next to the first dataset dir."""
    cache_dir = Path(dataset_paths[0])
    if not cache_dir.is_dir():
        cache_dir = cache_dir.parent
    cache = cache_dir / "matcha_stats.json"
    if cache.is_file():
        with open(cache, "r", encoding="utf-8") as fh:
            stats = json.load(fh)
        return stats["mel_mean"], stats["mel_std"]
    _LOGGER.info("computing dataset mel statistics (one-time pass)...")
    mean, std = compute_data_statistics(dataset)
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump({"mel_mean": mean, "mel_std": std}, fh)
    _LOGGER.info("mel_mean=%.6f mel_std=%.6f (cached at %s)", mean, std, cache)
    return mean, std
