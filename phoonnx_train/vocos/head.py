"""ISTFT head for the Vocos vocoder (Siuzdak, 2023).

Instead of upsampling in the time domain, Vocos predicts STFT
coefficients — log-magnitude and phase — for every frame and inverts them
with a standard inverse STFT.  The head is a single linear projection to
``n_fft + 2`` channels, split into magnitude and phase.

Parameter names follow the reference layout (``out``) so Vocos-family
checkpoints warm-start with a near-identity key mapping.
"""
from typing import Tuple

import torch
from torch import nn


class ISTFTHead(nn.Module):
    """Hidden features ``[B, T, dim]`` → waveform ``[B, N]``.

    Uses centered ISTFT (``center=True``), which trims ``n_fft // 2``
    samples from each side — exactly what the phoonnx runtime Vocos
    adapter does after its numpy overlap-add ISTFT, so torch inference and
    the exported ONNX + runtime pipeline produce the same waveform.
    """

    def __init__(self, dim: int, n_fft: int, hop_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.out = nn.Linear(dim, n_fft + 2)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def coefficients(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict ``(mag, x_real, y_imag)``, each ``[B, n_fft//2+1, T]``.

        This is the exact output triple the phoonnx runtime Vocos adapter
        consumes, and what gets exported to ONNX.
        """
        h = self.out(x).transpose(1, 2)
        # explicit slicing instead of chunk/Split for clean ONNX opset-17 export
        half = self.n_fft // 2 + 1
        mag, phase = h[:, :half], h[:, half:]
        mag = torch.clip(torch.exp(mag), max=1e2)  # guard huge magnitudes
        return mag, torch.cos(phase), torch.sin(phase)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag, real, imag = self.coefficients(x)
        spec = torch.complex(mag * real, mag * imag)
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
        )
