"""
YourTTS inference adapter.

YourTTS is a multilingual VITS conditioned on an external **512-d speaker d-vector**
(not a speaker-id embedding) plus a language id. This enables zero-shot voice
cloning: a reference clip is run through the bundled speaker encoder to produce the
d-vector, which conditions synthesis.

The ONNX graph takes ``input, input_lengths, scales, d_vector(1,512), langid`` and
returns a waveform. The d-vector is either:

- **bundled** — a fixed voice carries its speaker's d-vector in ``engine_params``
  (e.g. one of the model's training speakers), or
- **per-request** — supplied via ``params["d_vector"]`` for zero-shot cloning, as
  computed by :mod:`phoonnx.engines.speaker_encoder` from reference audio.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter


class YourTTSAdapter(BaseOnnxAdapter):
    """Adapter for YourTTS (multilingual VITS + d-vector conditioning)."""

    def __init__(self, d_vector: Optional[np.ndarray] = None, langid: int = 0):
        self.d_vector = None if d_vector is None else np.asarray(d_vector, np.float32).reshape(1, -1)
        self.langid = int(langid)

    def default_params(self) -> Dict[str, float]:
        return {"noise_scale": 0.667, "length_scale": 1.0, "noise_w_scale": 0.8}

    def configure(self, voice_config: Any) -> None:
        """Load the bundled speaker d-vector + language id from ``engine_params``."""
        ep = getattr(voice_config, "engine_params", None) or {}
        if self.d_vector is None and ep.get("d_vector") is not None:
            self.d_vector = np.asarray(ep["d_vector"], np.float32).reshape(1, -1)
        if "langid" in ep:
            self.langid = int(ep["langid"])

    def build_feed_dict(
        self,
        request: AdapterSynthesisRequest,
        session: onnxruntime.InferenceSession,
    ) -> Dict[str, np.ndarray]:
        p = request.params
        d = self.default_params()
        scales = np.array(
            [p.get("noise_scale", d["noise_scale"]),
             p.get("length_scale", d["length_scale"]),
             p.get("noise_w_scale", d["noise_w_scale"])],
            dtype=np.float32,
        )
        # d-vector: per-request (cloning) wins over the bundled fixed-voice vector
        dv = p.get("d_vector")
        dv = np.asarray(dv, np.float32).reshape(1, -1) if dv is not None else self.d_vector
        langid = int(p.get("langid", self.langid))
        args: Dict[str, np.ndarray] = {
            "input": request.phoneme_ids,
            "input_lengths": request.phoneme_lengths,
            "scales": scales,
            "langid": np.array([langid], dtype=np.int64),
        }
        if dv is not None:
            args["d_vector"] = dv.astype(np.float32)
        return self._filter_inputs(args, session)

    def parse_outputs(
        self,
        outputs: List[np.ndarray],
        request: AdapterSynthesisRequest,
    ) -> AdapterSynthesisResult:
        wav = max(outputs, key=lambda o: np.asarray(o).size)
        return AdapterSynthesisResult(audio=np.asarray(wav, dtype=np.float32).reshape(-1))

    @staticmethod
    def detect(
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> bool:
        return bool(config and config.get("engine") == "yourtts")
