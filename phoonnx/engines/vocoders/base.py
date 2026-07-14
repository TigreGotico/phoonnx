"""
Vocoder abstraction for mel-spectrogram → waveform conversion.

Flow-matching and other two-stage acoustic models (Matcha-TTS, OptiSpeech,
…) emit a mel spectrogram and rely on a **separate** neural vocoder to
produce the final waveform.  The same acoustic model can be paired with
different vocoders, and those vocoders differ in their ONNX output layout:

  - **Vocos / alVoCat** — output STFT coefficients (magnitude + real +
    imaginary); the waveform is reconstructed with an inverse STFT.
  - **Wavenext / HiFi-GAN** — output the raw waveform directly (Wavenext is
    Vocos with the final ISTFT layer replaced by a trained linear layer).

A ``BaseVocoder`` hides those differences behind ``mel_to_audio`` so the
acoustic-model adapter never has to know which vocoder it is driving.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence

import numpy as np
import onnxruntime

from phoonnx.providers import ProviderSpec, make_session


def load_onnx_session(model_path: str,
                      providers: Optional[Sequence[ProviderSpec]] = None) -> onnxruntime.InferenceSession:
    """Load an ONNX vocoder on the given (or auto-detected) execution providers."""
    return make_session(model_path, providers=providers)


class BaseVocoder(ABC):
    """
    Translates a mel spectrogram into a 1-D float32 waveform for a single
    vocoder family.

    Subclasses **must** implement
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - ``mel_to_audio`` — mel ``[B, n_mels, T]`` → waveform ``[N]``

    Subclasses **may** override
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - ``detect`` — auto-detect whether an ONNX model belongs to this family
    - ``supports_denoise`` — whether ``denoise=True`` is honoured
    """

    name: str = "base"

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ):
        self.model_path = model_path
        self.config = config or {}
        self._session = session

    # ------------------------------------------------------------------
    # Mel preprocessing (config-driven, opt-in via flags)
    # ------------------------------------------------------------------

    def _preprocess_mel(self, mel: np.ndarray) -> np.ndarray:
        """
        Apply optional, config-flagged mel transforms before the vocoder runs.

        A converted ONNX vocoder declares in its config exactly what its training
        pipeline expected, so the same runtime handles vocoders with different
        input conventions:

        - ``stats_norm`` (bool) → standard-scale the mel with ``mel_mean`` /
          ``mel_std`` (per-channel), as Coqui's stats-normalized vocoders
          (e.g. multiband-MelGAN) expect.
        """
        mel = np.asarray(mel, dtype=np.float32)
        c = self.config
        if c.get("stats_norm") and c.get("mel_mean") is not None and c.get("mel_std") is not None:
            mean = np.asarray(c["mel_mean"], dtype=np.float32)
            std = np.asarray(c["mel_std"], dtype=np.float32)
            shape = (1, -1, 1) if mel.ndim == 3 else (-1, 1)
            mel = (mel - mean.reshape(shape)) / std.reshape(shape)
        return mel

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    @property
    def session(self) -> onnxruntime.InferenceSession:
        """Lazily load the vocoder ONNX session."""
        if self._session is None:
            if not self.model_path:
                raise RuntimeError(
                    f"{type(self).__name__} requires a vocoder 'model_path' "
                    f"(set engine_params.vocoder_path)"
                )
            self._session = load_onnx_session(self.model_path,
                                              providers=self.config.get("providers"))
        return self._session

    @property
    def input_name(self) -> str:
        """ONNX input name expected by the vocoder (usually ``mels``)."""
        return self.session.get_inputs()[0].name

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    @abstractmethod
    def mel_to_audio(self, mel: np.ndarray, denoise: bool = False) -> np.ndarray:
        """
        Convert a mel spectrogram ``[B, n_mels, T]`` to a 1-D float32
        waveform in ``[-1, 1]``.
        """

    # ------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------

    @property
    def supports_denoise(self) -> bool:
        """Whether this vocoder honours the ``denoise`` flag."""
        return False

    @staticmethod
    def detect(
        session: Optional[onnxruntime.InferenceSession] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Return ``True`` if this vocoder handles the given ONNX model."""
        return False
