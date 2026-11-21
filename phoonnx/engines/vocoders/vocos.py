"""
Vocos vocoder (also covers alVoCat, ``projecte-aina/alvocat-vocos-22khz``).

Vocos does not model audio in the time domain.  Its ONNX graph emits STFT
coefficients — magnitude and the real/imaginary parts of the phase — which
are combined into a complex spectrogram and inverted with an overlap-add
inverse STFT.

ONNX outputs (3 tensors):
  ``mag``  float32  [B, n_fft//2+1, T]   magnitude
  ``x``    float32  [B, n_fft//2+1, T]   real part
  ``y``    float32  [B, n_fft//2+1, T]   imaginary part
"""
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime

from phoonnx.engines.vocoders.base import BaseVocoder


def _periodic_hann(n: int) -> np.ndarray:
    """Periodic Hann window (equivalent to scipy hann(n, sym=False))."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def _overlap_add(frames: np.ndarray, hop_length: int) -> np.ndarray:
    """Overlap-add ``[B, T, win]`` frames; blocks run last to first to keep the loop's sum order."""
    B, T, win_length = frames.shape
    n_blocks = -(-win_length // hop_length)
    out = np.zeros((B, (T + n_blocks - 1) * hop_length))
    for j in reversed(range(n_blocks)):
        start = j * hop_length
        width = min(hop_length, win_length - start)
        out[:, start:start + T * hop_length].reshape(B, T, hop_length)[:, :, :width] += \
            frames[:, :, start:start + width]
    return out[:, :(T - 1) * hop_length + win_length]


class VocosVocoder(BaseVocoder):
    """ISTFT-based vocoder emitting STFT coefficients (Vocos / alVoCat)."""

    name = "vocos"

    @property
    def supports_denoise(self) -> bool:
        return True

    def _stft_params(self) -> Dict[str, int]:
        """Read n_fft / hop_length, tolerating flat or nested config layouts."""
        cfg = self.config or {}
        init_args = cfg.get("feature_extractor", {}).get("init_args", {})
        n_fft = cfg.get("n_fft", init_args.get("n_fft", 1024))
        hop_length = cfg.get("hop_length", init_args.get("hop_length", 256))
        return {"n_fft": int(n_fft), "hop_length": int(hop_length)}

    def mel_to_audio(self, mel: np.ndarray, denoise: bool = False) -> np.ndarray:
        mag, x_real, y_imag = self.session.run(None, {self.input_name: mel})
        spectrogram = mag * (x_real + 1j * y_imag)

        if denoise:
            spectrogram = self._denoise(mel, spectrogram)

        audio = self._istft_overlap_add(spectrogram)
        return audio.squeeze()

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _denoise(self, mel: np.ndarray, spectrogram: np.ndarray) -> np.ndarray:
        """Spectral subtraction using a zero-mel bias spectrogram."""
        mel_zeros = np.zeros_like(mel)
        mag_bias, x_bias, y_bias = self.session.run(
            None, {self.input_name: mel_zeros.astype(np.float32)}
        )
        spec_bias = mag_bias * (x_bias + 1j * y_bias)

        mag_spec = np.abs(spectrogram)
        mag_spec_bias = np.abs(spec_bias)

        strength = 0.0025
        mag_spec_denoised = np.clip(mag_spec - mag_spec_bias * strength, 0.0, None)

        angle = np.angle(spectrogram)
        return mag_spec_denoised * (np.cos(angle) + 1j * np.sin(angle))

    def _istft_overlap_add(self, spectrogram: np.ndarray) -> np.ndarray:
        """Inverse STFT with a Hann window and overlap-add (vectorized)."""
        params = self._stft_params()
        n_fft = params["n_fft"]
        hop_length = params["hop_length"]
        win_length = n_fft

        window = _periodic_hann(win_length)

        T = spectrogram.shape[-1]

        # Inverse FFT per frame, windowed: [B, T, win_length]
        ifft = np.fft.irfft(spectrogram.transpose(0, 2, 1), n=n_fft, axis=-1) * window

        y = _overlap_add(ifft, hop_length)
        window_envelope = _overlap_add(
            np.broadcast_to(window ** 2, (1, T, win_length)), hop_length)

        y /= np.maximum(window_envelope, 1e-11)
        # Vocos STFT uses center=True (signal padded by n_fft//2 each side); trim
        # that padding back on the inverse. Otherwise the edge frames, where the
        # window envelope is near-zero, blow up into large boundary artifacts.
        pad = n_fft // 2
        if y.shape[1] > 2 * pad:
            y = y[:, pad:-pad]
        return y.astype(np.float32)

    @staticmethod
    def detect(
        session: Optional[onnxruntime.InferenceSession] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if config and config.get("vocoder_type") in ("vocos", "alvocat"):
            return True
        if session is not None:
            # Vocos emits 3 tensors: magnitude + real + imaginary
            return len(session.get_outputs()) >= 3
        return False
