"""
VITS / Piper / Mimic3 / Coqui inference adapter.

All of these engines share the same ONNX I/O contract (they are all
VITS variants):

  Inputs:  input (int64), input_lengths (int64), scales (float32[3]),
           [optional] sid (int64), langid (int64)
  Outputs: output (float32) — raw waveform

The ``scales`` tensor packs [noise_scale, length_scale, noise_w_scale].

Some exports use alternate input names (``x`` / ``x_length`` for MMS,
``input_ids`` / ``attention_mask`` for HuggingFace).  The adapter
provides all variants and filters to the ones the model actually
accepts.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime

from phoonnx.engines.base import (
    AdapterSynthesisRequest,
    AdapterSynthesisResult,
    BaseOnnxAdapter,
)


class VitsAdapter(BaseOnnxAdapter):
    """Adapter for VITS-family ONNX models (piper, mimic3, coqui, phoonnx)."""

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    def build_feed_dict(
        self,
        request: AdapterSynthesisRequest,
        session: onnxruntime.InferenceSession,
    ) -> Dict[str, np.ndarray]:
        """Build the ONNX feed dict for a VITS-family model."""
        params = request.params
        defaults = self.default_params()
        noise_scale = np.float32(params.get("noise_scale", defaults["noise_scale"]))
        length_scale = np.float32(params.get("length_scale", defaults["length_scale"]))
        noise_w_scale = np.float32(params.get("noise_w_scale", defaults["noise_w_scale"]))

        phoneme_ids_array = request.phoneme_ids          # already (1, T)
        phoneme_ids_lengths = request.phoneme_lengths     # already (1,)
        attention_mask = np.ones_like(phoneme_ids_array, dtype=np.int64)

        # Provide every known alias — we filter to actual model inputs below
        args: Dict[str, np.ndarray] = {
            # piper / phoonnx style
            "input": phoneme_ids_array,
            "input_lengths": phoneme_ids_lengths,
            # MMS style
            "x": phoneme_ids_array,
            "x_length": phoneme_ids_lengths,
            # HuggingFace style
            "input_ids": phoneme_ids_array,
            "attention_mask": attention_mask,
            # scales
            "scales": np.array(
                [noise_scale, length_scale, noise_w_scale], dtype=np.float32
            ),
            "noise_scale": np.array([noise_scale], dtype=np.float32),
            "noise_scale_w": np.array([noise_w_scale], dtype=np.float32),
            "length_scale": np.array([length_scale], dtype=np.float32),
        }

        # Optional speaker / language ids
        if request.language_id is not None:
            args["langid"] = np.array([request.language_id], dtype=np.int64)
        if request.speaker_id is not None:
            args["sid"] = np.array([request.speaker_id], dtype=np.int64)

        return self._filter_inputs(args, session)

    def parse_outputs(
        self,
        outputs: List[np.ndarray],
        request: AdapterSynthesisRequest,
    ) -> AdapterSynthesisResult:
        """Extract audio waveform from VITS ONNX outputs."""
        # VITS returns a single output tensor — squeeze to 1-D
        audio = outputs[0].squeeze()
        return AdapterSynthesisResult(audio=audio)

    def default_params(self) -> Dict[str, float]:
        return {
            "noise_scale": 0.667,
            "length_scale": 1.0,
            "noise_w_scale": 0.8,
        }

    # ------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------

    @staticmethod
    def detect(
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> bool:
        # Config-based detection
        if config is not None:
            # Explicit engine tag
            engine = config.get("engine", "")
            if engine in ("vits", "piper", "mimic3", "coqui", "phoonnx"):
                return True
            # Piper signature
            if "phoneme_id_map" in config and "phoneme_type" in config:
                return True
            # Phoonnx signature
            if "phoonnx_version" in config or "piper_version" in config:
                return True
            # Coqui signature
            if "characters" in config and isinstance(config.get("characters"), dict):
                return True
            # Mimic3 signature
            if "phonemizer" in config and "phonemes" in config:
                return True

        # Session-based detection: check for "scales" input
        if session is not None:
            input_names = {inp.name for inp in session.get_inputs()}
            if "scales" in input_names:
                return True

        return False

    def parse_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract VITS inference parameters from a JSON config dict."""
        inference = config.get("inference", {})
        defaults = self.default_params()
        return {
            "noise_scale": inference.get("noise_scale", defaults["noise_scale"]),
            "length_scale": inference.get("length_scale", defaults["length_scale"]),
            "noise_w_scale": inference.get("noise_w", defaults["noise_w_scale"]),
        }

    def param_labels(self) -> Dict[str, str]:
        """Human-readable labels for VITS synthesis parameters."""
        return {
            "noise_scale": "Noise",
            "length_scale": "Speed",
            "noise_w_scale": "Noise W",
        }
