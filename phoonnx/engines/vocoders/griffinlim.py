"""
Griffin-Lim vocoder — a parametric (non-neural) mel→waveform vocoder.

Unlike the ONNX vocoders, this needs no model file: it inverts a mel
spectrogram back to audio with the Griffin-Lim algorithm, parameterised by the
acoustic model's mel settings. This makes *any* mel-producing engine
(GlowTTS, Matcha, …) synthesizable without a domain-matched neural vocoder —
lower fidelity, but universal.

The de-normalisation matches Coqui-TTS's ``AudioProcessor`` so coqui-exported
mels invert correctly (``db_to_amp = base**(x/spec_gain)``, symmetric norm).
"""
from typing import Any, Dict, Optional

import numpy as np

from phoonnx.engines.vocoders.base import BaseVocoder


class GriffinLimVocoder(BaseVocoder):
    """Parametric mel→audio vocoder (no ONNX model)."""

    def __init__(self, model_path: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 session: Optional[Any] = None):
        # model_path/session are unused — GL is purely parametric.
        super().__init__(model_path=None, config=config or {}, session=None)
        c = self.config
        self.sample_rate = int(c.get("sample_rate", 22050))
        self.n_fft = int(c.get("n_fft", c.get("fft_size", 1024)))
        self.hop_length = int(c.get("hop_length", 256))
        self.win_length = int(c.get("win_length", c.get("n_fft", 1024)))
        self.num_mels = int(c.get("num_mels", c.get("n_mels", 80)))
        self.mel_fmin = float(c.get("mel_fmin", 0.0))
        self.mel_fmax = c.get("mel_fmax", None)
        if self.mel_fmax in (0, "0"):
            self.mel_fmax = None
        # coqui AudioProcessor normalization params
        self.ref_level_db = float(c.get("ref_level_db", 20.0))
        self.min_level_db = float(c.get("min_level_db", -100.0))
        self.spec_gain = float(c.get("spec_gain", 1.0))
        self.max_norm = float(c.get("max_norm", 4.0))
        self.symmetric_norm = bool(c.get("symmetric_norm", True))
        self.signal_norm = bool(c.get("signal_norm", True))
        self.clip_norm = bool(c.get("clip_norm", True))
        self.power = float(c.get("power", 1.5))
        self.num_iters = int(c.get("griffin_lim_iters", 60))
        self._mel_basis = None
        self._inv_mel_basis = None

    @staticmethod
    def _librosa():
        try:
            import librosa
            return librosa
        except ImportError as e:
            raise ImportError(
                "The Griffin-Lim vocoder requires 'librosa'. Install it with "
                "`pip install librosa` (or `pip install phoonnx[train]`)."
            ) from e

    @property
    def mel_basis(self) -> np.ndarray:
        if self._mel_basis is None:
            librosa = self._librosa()
            fmax = self.mel_fmax if self.mel_fmax is not None else self.sample_rate / 2
            self._mel_basis = librosa.filters.mel(
                sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.num_mels,
                fmin=self.mel_fmin, fmax=fmax)
            self._inv_mel_basis = np.linalg.pinv(self._mel_basis)
        return self._mel_basis

    def _denormalize(self, S: np.ndarray) -> np.ndarray:
        if not self.signal_norm:
            return S
        S = S.copy()
        if self.symmetric_norm:
            if self.clip_norm:
                S = np.clip(S, -self.max_norm, self.max_norm)
            return ((S + self.max_norm) * (-self.min_level_db) / (2 * self.max_norm)) + self.min_level_db
        if self.clip_norm:
            S = np.clip(S, 0, self.max_norm)
        return (S * (-self.min_level_db) / self.max_norm) + self.min_level_db

    def mel_to_audio(self, mel: np.ndarray, denoise: bool = False) -> np.ndarray:
        librosa = self._librosa()
        m = np.asarray(mel, dtype=np.float32)
        if m.ndim == 3:
            m = m[0]            # [n_mels, T]
        _ = self.mel_basis      # build basis + inverse
        D = self._denormalize(m)
        S = np.power(10.0, (D + self.ref_level_db) / self.spec_gain)   # db_to_amp (base 10)
        linear = np.maximum(1e-10, np.dot(self._inv_mel_basis, S))      # mel -> linear magnitude
        audio = librosa.griffinlim(
            linear ** self.power, n_iter=self.num_iters,
            hop_length=self.hop_length, win_length=self.win_length, n_fft=self.n_fft)
        return audio.astype(np.float32)

    @property
    def supports_denoise(self) -> bool:
        return False

    @staticmethod
    def detect(session: Optional[Any] = None, config: Optional[Dict[str, Any]] = None) -> bool:
        # only selected explicitly via vocoder_type="griffinlim"; never auto-detected
        return False
