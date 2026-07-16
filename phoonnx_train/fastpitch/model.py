"""
ForwardTTS acoustic model (FastPitch / SpeedySpeech).

Adapted from coqui-ai/TTS ``TTS/tts/models/forward_tts.py`` (© Coqui GmbH,
Mozilla Public License 2.0). Pure torch, self-contained.

Text/phoneme ids → 80-channel mel spectrogram. Non-autoregressive: an
encoder produces per-token states, per-token durations (and optionally
pitch/energy) are predicted, states are expanded to frame rate and a
decoder renders the mel. Durations are learned unsupervised with an
:class:`~phoonnx_train.fastpitch.aligner.AlignmentNetwork` + monotonic
alignment search.

Both model variants are configurations of this one class:

- **FastPitch**: FFT-transformer encoder/decoder, pitch predictor on.
- **SpeedySpeech**: residual conv-BN encoder/decoder, no pitch predictor.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import nn

from phoonnx_train.fastpitch.aligner import AlignmentNetwork
from phoonnx_train.fastpitch.helpers import (
    average_over_durations,
    generate_path,
    maximum_path,
    sequence_mask,
)
from phoonnx_train.fastpitch.layers import (
    DurationPredictor,
    FFTransformerBlock,
    ResidualConv1dBNBlock,
)


@dataclass
class ForwardTTSArgs:
    """Hyper-parameters of the ForwardTTS model (subset of coqui's)."""

    num_chars: int = 256
    out_channels: int = 80
    hidden_channels: int = 384

    # variant switches
    use_pitch: bool = True
    use_energy: bool = False
    use_aligner: bool = True
    encoder_type: str = "fftransformer"  # or "residual_conv_bn" (speedyspeech)
    decoder_type: str = "fftransformer"

    # fftransformer params
    num_heads: int = 1
    hidden_channels_ffn: int = 1024
    encoder_num_layers: int = 6
    decoder_num_layers: int = 6
    dropout: float = 0.1

    # residual conv-bn params (speedyspeech)
    encoder_num_res_blocks: int = 13
    decoder_num_res_blocks: int = 17
    res_kernel_size: int = 4

    # predictor heads
    predictor_hidden_channels: int = 256
    pitch_embedding_kernel_size: int = 3

    # multi-speaker
    num_speakers: int = 1
    speaker_embedding_channels: int = 0  # 0 -> use hidden_channels

    def __post_init__(self):
        if self.speaker_embedding_channels <= 0:
            self.speaker_embedding_channels = self.hidden_channels


def _build_coder(kind: str, args: "ForwardTTSArgs", num_layers: int,
                 num_res_blocks: int) -> nn.Module:
    if kind == "fftransformer":
        return FFTransformerBlock(
            in_out_channels=args.hidden_channels,
            num_heads=args.num_heads,
            hidden_channels_ffn=args.hidden_channels_ffn,
            num_layers=num_layers,
            dropout=args.dropout,
        )
    if kind == "residual_conv_bn":
        return ResidualConv1dBNBlock(
            in_channels=args.hidden_channels,
            out_channels=args.hidden_channels,
            hidden_channels=args.hidden_channels,
            kernel_size=args.res_kernel_size,
            num_res_blocks=num_res_blocks,
        )
    raise ValueError(f"Unknown encoder/decoder type: {kind!r}")


class ForwardTTS(nn.Module):
    def __init__(self, args: ForwardTTSArgs):
        super().__init__()
        self.args = args

        self.emb = nn.Embedding(args.num_chars, args.hidden_channels)
        self.encoder = _build_coder(args.encoder_type, args,
                                    args.encoder_num_layers, args.encoder_num_res_blocks)
        self.decoder = _build_coder(args.decoder_type, args,
                                    args.decoder_num_layers, args.decoder_num_res_blocks)
        self.mel_proj = nn.Conv1d(args.hidden_channels, args.out_channels, 1)

        self.duration_predictor = DurationPredictor(
            args.hidden_channels, args.predictor_hidden_channels, dropout=args.dropout
        )
        if args.use_pitch:
            self.pitch_predictor = DurationPredictor(
                args.hidden_channels, args.predictor_hidden_channels, dropout=args.dropout
            )
            self.pitch_emb = nn.Conv1d(
                1, args.hidden_channels,
                kernel_size=args.pitch_embedding_kernel_size,
                padding=args.pitch_embedding_kernel_size // 2,
            )
        if args.use_energy:
            self.energy_predictor = DurationPredictor(
                args.hidden_channels, args.predictor_hidden_channels, dropout=args.dropout
            )
            self.energy_emb = nn.Conv1d(
                1, args.hidden_channels,
                kernel_size=args.pitch_embedding_kernel_size,
                padding=args.pitch_embedding_kernel_size // 2,
            )
        if args.use_aligner:
            self.aligner = AlignmentNetwork(
                in_query_channels=args.out_channels,
                in_key_channels=args.hidden_channels,
            )
        if args.num_speakers > 1:
            # multi-speaker conditioning: additive speaker embedding (emb_g)
            self.emb_g = nn.Embedding(args.num_speakers, args.hidden_channels)

    # ------------------------------------------------------------------
    # shared pieces
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor, x_mask: torch.Tensor,
                g: Optional[torch.Tensor]) -> torch.Tensor:
        """x: [B, T_en] ids -> encoder states [B, C, T_en]."""
        o = self.emb(x).transpose(1, 2) * x_mask  # [B, C, T]
        o = self.encoder(o, x_mask)
        if g is not None:
            o = o + g
        return o * x_mask

    def _speaker_embedding(self, speaker: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.args.num_speakers > 1 and speaker is not None:
            return self.emb_g(speaker.view(-1)).unsqueeze(-1)  # [B, C, 1]
        return None

    @staticmethod
    def _expand(en: torch.Tensor, durations: torch.Tensor,
                x_mask: torch.Tensor, y_lengths: torch.Tensor) -> torch.Tensor:
        """Length-regulate encoder states to frame rate: [B, C, T_en] -> [B, C, T_de]."""
        y_mask = sequence_mask(y_lengths).unsqueeze(1).to(en.dtype)  # [B, 1, T_de]
        attn_mask = x_mask.transpose(1, 2) * y_mask  # [B, T_en, T_de]
        attn = generate_path(durations, attn_mask)
        return torch.matmul(en, attn), y_mask

    # ------------------------------------------------------------------
    # training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,             # [B, T_en] phoneme ids
        x_lengths: torch.Tensor,     # [B]
        y: torch.Tensor,             # [B, C_mel, T_de] target mel
        y_lengths: torch.Tensor,     # [B]
        pitch: Optional[torch.Tensor] = None,   # [B, 1, T_de] frame-level f0
        energy: Optional[torch.Tensor] = None,  # [B, 1, T_de]
        speaker: Optional[torch.Tensor] = None,  # [B]
    ) -> Dict[str, torch.Tensor]:
        x_mask = sequence_mask(x_lengths, x.shape[1]).unsqueeze(1).to(self.emb.weight.dtype)
        g = self._speaker_embedding(speaker)
        x_emb = self.emb(x).transpose(1, 2) * x_mask  # [B, C, T_en]
        en = self._encode(x, x_mask, g)

        # unsupervised alignment -> durations. The aligner is keyed on the
        # raw embedding table, decoupled from the encoder stack: aligner
        # gradients must not fight the decoder/spec objective inside the
        # shared encoder representation.
        y_mask_seq = sequence_mask(y_lengths, y.shape[2])
        attn_soft, attn_logp = self.aligner(y, x_emb, x_mask)
        # hard alignment via MAS on log-probs [B, T_en, T_de]
        attn_mask_full = x_mask.transpose(1, 2) * y_mask_seq.unsqueeze(1).to(x_mask.dtype)
        alignment_hard = maximum_path(
            attn_logp.squeeze(1).transpose(1, 2).contiguous(), attn_mask_full
        )
        durations = alignment_hard.sum(-1)  # [B, T_en]

        o_dur_log = self.duration_predictor(en.detach(), x_mask).squeeze(1)  # [B, T_en]

        o_pitch = avg_pitch = None
        if self.args.use_pitch and pitch is not None:
            o_pitch = self.pitch_predictor(en.detach(), x_mask)  # [B, 1, T_en]
            avg_pitch = average_over_durations(pitch, durations)  # [B, 1, T_en]
            en = en + self.pitch_emb(avg_pitch) * x_mask

        o_energy = avg_energy = None
        if self.args.use_energy and energy is not None:
            o_energy = self.energy_predictor(en.detach(), x_mask)
            avg_energy = average_over_durations(energy, durations)
            en = en + self.energy_emb(avg_energy) * x_mask

        o_en_ex, y_mask = self._expand(en, durations, x_mask, y_lengths)
        o_de = self.decoder(o_en_ex, y_mask)
        o_mel = self.mel_proj(o_de) * y_mask  # [B, C_mel, T_de]

        return {
            "model_outputs": o_mel.transpose(1, 2),        # [B, T_de, C_mel]
            "durations_log": o_dur_log,
            "durations": durations,
            "pitch_avg": avg_pitch,
            "pitch_avg_pred": o_pitch,
            "energy_avg": avg_energy,
            "energy_avg_pred": o_energy,
            "alignment_soft": attn_soft.squeeze(1).transpose(1, 2),  # [B, T_en, T_de]
            "alignment_logprob": attn_logp,
            "alignment_hard": alignment_hard,
        }

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,                       # [B, T_en]
        speaker: Optional[torch.Tensor] = None,
        pace: float = 1.0,
        pitch_mul: float = 1.0,
        pitch_add: float = 0.0,
    ) -> torch.Tensor:
        """Return mel [B, C_mel, T_de]. Tracer-safe (live-shape masks).

        Masks are derived with ``ones_like`` from a tensor whose shape
        already carries the relevant (dynamic) axis, rather than from
        Python ints read off ``.shape`` -- a plain ``torch.onnx.export``
        trace bakes the latter in as fixed constants, which breaks
        inference on any input length other than the one used for the
        dummy trace input.
        """
        # inference input is a single un-padded sequence: all-ones mask
        x_mask = torch.ones_like(x, dtype=self.emb.weight.dtype).unsqueeze(1)
        g = self._speaker_embedding(speaker)
        en = self._encode(x, x_mask, g)

        o_dur_log = self.duration_predictor(en, x_mask).squeeze(1)
        durations = torch.clamp((torch.exp(o_dur_log) - 1) / pace, min=0.0)
        durations = torch.round(durations).long().clamp(min=0)

        if self.args.use_pitch:
            o_pitch = self.pitch_predictor(en, x_mask)
            o_pitch = o_pitch * pitch_mul + pitch_add
            en = en + self.pitch_emb(o_pitch) * x_mask
        if self.args.use_energy:
            en = en + self.energy_emb(self.energy_predictor(en, x_mask)) * x_mask

        # Length regulation via repeat_interleave (ONNX-friendly, B=1).
        # Every token gets at least one output frame: this keeps the graph
        # free of data-dependent Python control flow (which a plain
        # torch.onnx.export trace would bake in as a fixed branch instead
        # of re-evaluating per input) and guarantees non-empty decoder
        # input even when an undertrained duration predictor outputs
        # near-zero/negative durations.
        durations = torch.clamp(durations, min=1)
        en_t = en.transpose(1, 2)  # [B, T_en, C]
        expanded = torch.repeat_interleave(en_t[0], durations[0], dim=0)  # [T_de, C]
        o_en_ex = expanded.unsqueeze(0).transpose(1, 2)  # [1, C, T_de]
        y_mask = torch.ones_like(o_en_ex[:, :1, :])

        o_de = self.decoder(o_en_ex, y_mask)
        return self.mel_proj(o_de)  # [B, C_mel, T_de]
