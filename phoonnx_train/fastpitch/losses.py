"""
Losses for the vendored ForwardTTS.

Adapted from coqui-ai/TTS ``TTS/tts/layers/losses.py`` (``ForwardTTSLoss``,
``ForwardSumLoss``) © Coqui GmbH, Mozilla Public License 2.0.
"""
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


class ForwardSumLoss(nn.Module):
    """CTC-based alignment loss over the soft attention (aligner) matrix."""

    def __init__(self, blank_logprob: float = -1.0):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob: torch.Tensor, in_lens: torch.Tensor,
                out_lens: torch.Tensor) -> torch.Tensor:
        """attn_logprob: [B, 1, T_de, T_en]."""
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(attn_logprob, (1, 0), value=self.blank_logprob)

        total_loss = attn_logprob.new_zeros(())
        b = attn_logprob.shape[0]
        for bid in range(b):
            target_seq = torch.arange(1, key_lens[bid] + 1, device=attn_logprob.device).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[
                : query_lens[bid], :, : key_lens[bid] + 1
            ]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss = total_loss + loss
        return total_loss / b


class BinaryAlignmentLoss(nn.Module):
    """Binarization loss: -log P(hard path) under the soft alignment."""

    def forward(self, alignment_hard: torch.Tensor, alignment_soft: torch.Tensor) -> torch.Tensor:
        log_sum = torch.log(torch.clamp(alignment_soft[alignment_hard == 1], min=1e-12)).sum()
        return -log_sum / alignment_hard.sum()


class ForwardTTSLoss(nn.Module):
    """Combined spectral + duration + pitch + aligner loss."""

    def __init__(
        self,
        spec_loss_alpha: float = 1.0,
        dur_loss_alpha: float = 0.1,
        pitch_loss_alpha: float = 0.1,
        aligner_loss_alpha: float = 1.0,
        binary_align_loss_alpha: float = 0.1,
    ):
        super().__init__()
        self.spec_loss_alpha = spec_loss_alpha
        self.dur_loss_alpha = dur_loss_alpha
        self.pitch_loss_alpha = pitch_loss_alpha
        self.aligner_loss_alpha = aligner_loss_alpha
        self.binary_align_loss_alpha = binary_align_loss_alpha
        self.aligner_loss = ForwardSumLoss()
        self.bin_loss = BinaryAlignmentLoss()

    @staticmethod
    def _masked_l1(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (torch.abs(x - y) * mask).sum() / torch.clamp(mask.sum() * x.shape[-1] / mask.shape[-1], min=1.0)

    def forward(
        self,
        decoder_output: torch.Tensor,          # [B, T_de, C_mel]
        decoder_target: torch.Tensor,          # [B, T_de, C_mel]
        decoder_output_lens: torch.Tensor,     # [B]
        dur_output: torch.Tensor,              # [B, T_en] log durations
        dur_target: torch.Tensor,              # [B, T_en]
        input_lens: torch.Tensor,              # [B]
        pitch_output: Optional[torch.Tensor] = None,   # [B, 1, T_en]
        pitch_target: Optional[torch.Tensor] = None,
        aligner_logprob: Optional[torch.Tensor] = None,
        alignment_hard: Optional[torch.Tensor] = None,
        alignment_soft: Optional[torch.Tensor] = None,
        binary_loss_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        from phoonnx_train.fastpitch.helpers import sequence_mask

        return_dict: Dict[str, torch.Tensor] = {}
        loss = decoder_output.new_zeros(())

        spec_mask = sequence_mask(decoder_output_lens, decoder_output.shape[1]).unsqueeze(-1)
        spec_loss = self._masked_l1(decoder_output, decoder_target, spec_mask)
        loss = loss + self.spec_loss_alpha * spec_loss
        return_dict["loss_spec"] = spec_loss

        in_mask = sequence_mask(input_lens, dur_output.shape[1])

        def _masked_mse(output, target, mask):
            # mean over VALID positions only — a plain F.mse_loss mean would
            # divide by padded elements too, scaling the gradient by the
            # batch's padding ratio
            mask = mask.to(output.dtype)
            diff2 = ((output - target) ** 2) * mask
            return diff2.sum() / mask.sum().clamp_min(1.0)

        log_dur_tgt = torch.log(dur_target.float() + 1)
        dur_loss = _masked_mse(dur_output, log_dur_tgt, in_mask)
        loss = loss + self.dur_loss_alpha * dur_loss
        return_dict["loss_dur"] = dur_loss

        if pitch_output is not None and pitch_target is not None:
            pitch_loss = _masked_mse(pitch_output, pitch_target,
                                     in_mask.unsqueeze(1))
            loss = loss + self.pitch_loss_alpha * pitch_loss
            return_dict["loss_pitch"] = pitch_loss

        if aligner_logprob is not None:
            aligner_loss = self.aligner_loss(aligner_logprob, input_lens, decoder_output_lens)
            loss = loss + self.aligner_loss_alpha * aligner_loss
            return_dict["loss_aligner"] = aligner_loss

        if alignment_hard is not None and alignment_soft is not None and self.binary_align_loss_alpha > 0:
            bin_loss = self.bin_loss(alignment_hard, alignment_soft)
            loss = loss + self.binary_align_loss_alpha * binary_loss_weight * bin_loss
            return_dict["loss_binary_alignment"] = bin_loss

        return_dict["loss"] = loss
        return return_dict
