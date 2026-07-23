"""Adapter for ShamiVITS / HamsVITS ONNX models.

ShamiVITS (https://huggingface.co/Tushe/shami-tts) is a VITS variant for Levantine
Arabic / English code-switching.  Its ONNX export has a distinct I/O contract from
standard piper/coqui/phoonnx VITS:

  Inputs:  phoneme_ids      (int64, shape [1, T])
           phoneme_lengths  (int64, shape [1])
           language_ids     (int64, shape [1, T])  -- per-phoneme language stream
           [optional] speaker_id (int64, shape [1])
  Output:  waveform         (float32, shape [1, samples])

There is no ``scales`` input: ``noise_scale`` / ``speaking_rate`` are baked into the
exported graph (the upstream checkpoint is trained with deterministic duration and is
intended to be used at ``length_scale=1.0``).
"""
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime

from phoonnx.engines.base import (
    AdapterSynthesisRequest,
    AdapterSynthesisResult,
    BaseOnnxAdapter,
)


class ShamiAdapter(BaseOnnxAdapter):
    """Adapter for ShamiVITS-family ONNX models."""

    def build_feed_dict(
        self,
        request: AdapterSynthesisRequest,
        session: onnxruntime.InferenceSession,
    ) -> Dict[str, np.ndarray]:
        args: Dict[str, np.ndarray] = {
            "phoneme_ids": request.phoneme_ids,
            "phoneme_lengths": request.phoneme_lengths,
        }

        # The graph *requires* language_ids (it is how detect() recognises the
        # model), so never omit it: without a per-phoneme language stream, fall
        # back to a single-language one so synthesis degrades to monolingual
        # rather than failing onnxruntime's required-input check.
        args["language_ids"] = (
            request.language_ids if request.language_ids is not None
            else np.zeros_like(np.asarray(request.phoneme_ids), dtype=np.int64)
        )

        # Single-speaker graphs may prune the speaker_id input entirely.
        if request.speaker_id is not None and request.speaker_id > 0:
            args["speaker_id"] = np.array([request.speaker_id], dtype=np.int64)

        return self._filter_inputs(args, session)

    def parse_outputs(
        self,
        outputs: List[np.ndarray],
        request: AdapterSynthesisRequest,
    ) -> AdapterSynthesisResult:
        audio = outputs[0].squeeze()
        return AdapterSynthesisResult(audio=audio)

    def default_params(self) -> Dict[str, float]:
        # Scales are baked into the exported graph; expose only dummy defaults so the
        # generic TTSVoice layer does not complain about missing keys.
        return {
            "noise_scale": 0.667,
            "length_scale": 1.0,
            "noise_w_scale": 0.8,
        }

    @staticmethod
    def detect(
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> bool:
        if config is not None:
            engine = config.get("engine", "")
            if engine in ("shami", "hams"):
                return True

        if session is not None:
            inputs = {inp.name for inp in session.get_inputs()}
            if "phoneme_ids" in inputs and "language_ids" in inputs:
                return True

        return False

    def parse_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # No runtime scales for ShamiVITS; return empty so the baked-in defaults win.
        return {}

    def param_labels(self) -> Dict[str, str]:
        return {
            "noise_scale": "Noise (baked-in)",
            "length_scale": "Speed (baked-in)",
            "noise_w_scale": "Noise W (baked-in)",
        }
