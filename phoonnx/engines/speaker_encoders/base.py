"""Speaker-encoder abstraction for d-vector based voice cloning.

A speaker encoder maps a reference waveform to a fixed-size speaker embedding
(d-vector) that conditions a cloning-capable model (e.g. YourTTS). Concrete
encoders register themselves so a voice can name its encoder by type, exactly like
the mel→waveform vocoders.
"""
from abc import ABC, abstractmethod

import numpy as np


class BaseSpeakerEncoder(ABC):
    """Reference audio -> speaker d-vector."""

    #: native sample rate the encoder ONNX expects
    sample_rate: int = 16000

    @abstractmethod
    def encode(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return the (1-D) speaker d-vector for ``audio`` (mono float32, any rate)."""

    @staticmethod
    def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sr_in == sr_out:
            return audio
        try:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr_in), int(sr_out))
            return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)
        except Exception:
            n = int(round(len(audio) * sr_out / sr_in))
            xp = np.linspace(0, 1, len(audio), endpoint=False)
            return np.interp(np.linspace(0, 1, n, endpoint=False), xp, audio).astype(np.float32)
