"""StyleTTS2 style encoder — reference wav -> 256-d style (ref_p[128] ++ ref_s[128])."""
from typing import Any, Dict, Optional

import numpy as np
import onnxruntime

from phoonnx.engines.speaker_encoders.base import BaseSpeakerEncoder
from phoonnx.providers import make_session


class StyleTTS2StyleEncoder(BaseSpeakerEncoder):
    sample_rate = 24000

    def __init__(self, model_path: str, config: Optional[Dict[str, Any]] = None):
        self.session = make_session(model_path, providers=(config or {}).get("providers"))
        self._input = self.session.get_inputs()[0].name
        self._outs = [o.name for o in self.session.get_outputs()]

    def encode(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = self.resample(audio, sample_rate, self.sample_rate)
        out = dict(zip(self._outs, self.session.run(None, {self._input: audio[None, :].astype(np.float32)})))
        # concat acoustic (ref_p) + prosody (ref_s); the StyleTTS2 adapter splits it
        return np.concatenate([np.asarray(out["ref_p"], np.float32).reshape(-1),
                               np.asarray(out["ref_s"], np.float32).reshape(-1)])
