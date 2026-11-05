"""
Raw-waveform vocoders — Wavenext and HiFi-GAN.

These vocoders model audio directly in the time domain: their ONNX graph
emits the waveform as a single tensor, so no inverse STFT is needed.

Wavenext (``BSC-LT/wavenext-mel``) is a modification of Vocos where the
final ISTFT layer is replaced by a trained linear layer that predicts
speech samples directly; HiFi-GAN is a GAN vocoder that likewise outputs
waveform samples.  From phoonnx's point of view they are interchangeable:
run the model, take the single output tensor.
"""
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime

from phoonnx.engines.vocoders.base import BaseVocoder


class RawWaveformVocoder(BaseVocoder):
    """Vocoder whose ONNX model outputs a waveform tensor directly."""

    name = "raw"

    #: vocoder_type strings that select this implementation
    _types: tuple = ("raw", "wavenext", "hifigan")

    def mel_to_audio(self, mel: np.ndarray, denoise: bool = False) -> np.ndarray:
        # denoise is a Vocos/ISTFT concept; raw vocoders bake it in (or omit).
        mel = self._preprocess_mel(mel)
        feed = {self.input_name: mel}
        # Some exports (e.g. tts_arabic's baked Vocos) take an extra scalar
        # 'denoise' input alongside the mel — feed it (default 0).
        for inp in self.session.get_inputs()[1:]:
            feed[inp.name] = np.array([1.0 if denoise else 0.0], dtype=np.float32)
        outputs = self.session.run(None, feed)
        return np.asarray(outputs[0], dtype=np.float32).squeeze()

    @classmethod
    def detect(
        cls,
        session: Optional[onnxruntime.InferenceSession] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if config and config.get("vocoder_type") in cls._types:
            return True
        if session is not None:
            # Raw vocoders emit a single waveform tensor.
            return len(session.get_outputs()) == 1
        return False


class WavenextVocoder(RawWaveformVocoder):
    """Wavenext vocoder (Vocos with the ISTFT head replaced by a linear layer)."""

    name = "wavenext"
    _types = ("wavenext",)


class HiFiGANVocoder(RawWaveformVocoder):
    """HiFi-GAN GAN vocoder."""

    name = "hifigan"
    _types = ("hifigan",)
