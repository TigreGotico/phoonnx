"""Coqui ResNet speaker encoder (YourTTS) — wav -> 512-d L2-normalised d-vector."""
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime

from phoonnx.engines.speaker_encoders.base import BaseSpeakerEncoder
from phoonnx.providers import make_session


class CoquiResNetSpeakerEncoder(BaseSpeakerEncoder):
    sample_rate = 16000

    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        self.session = make_session(model_path, providers=(config or {}).get("providers"))
        self._input = self.session.get_inputs()[0].name

    def encode(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = self.resample(audio, sample_rate, self.sample_rate)
        dv = self.session.run(None, {self._input: audio[None, :].astype(np.float32)})[0]
        return np.asarray(dv, dtype=np.float32).reshape(-1)
