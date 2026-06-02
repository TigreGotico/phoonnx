from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import FloatTensor, LongTensor

from vits.dataset import PhoonnxDataset, UtteranceCollate as BaseUtteranceCollate
from vits.dataset import Batch as BaseBatch, UtteranceTensors


@dataclass
class DisentangledBatch(BaseBatch):
    """Batch with additional reference mel fields for disentangled training."""

    timbre_ref_mels: Optional[FloatTensor] = None
    artic_ref_mels: Optional[FloatTensor] = None
    prosody_ref_mels: Optional[FloatTensor] = None


class DisentangledUtteranceCollate:
    """Collate function that adds reference mel fields for disentangled training."""

    def __init__(self, is_multispeaker: bool, segment_size: int):
        self.is_multispeaker = is_multispeaker
        self.segment_size = segment_size

    def __call__(self, utterances: Sequence[UtteranceTensors]) -> DisentangledBatch:
        # Delegate base collation to the standard VITS collate
        base_collate = BaseUtteranceCollate(
            is_multispeaker=self.is_multispeaker,
            segment_size=self.segment_size,
        )
        base_batch = base_collate(utterances)

        batch = DisentangledBatch(
            phoneme_ids=base_batch.phoneme_ids,
            phoneme_lengths=base_batch.phoneme_lengths,
            spectrograms=base_batch.spectrograms,
            spectrogram_lengths=base_batch.spectrogram_lengths,
            audios=base_batch.audios,
            audio_lengths=base_batch.audio_lengths,
            speaker_ids=base_batch.speaker_ids,
        )

        # For disentangled training, use the target utterance's own mel
        # as all three reference mels by default. In a full implementation,
        # these would be sampled from a reference pool.
        batch.timbre_ref_mels = base_batch.spectrograms
        batch.artic_ref_mels = base_batch.spectrograms
        batch.prosody_ref_mels = base_batch.spectrograms

        return batch
