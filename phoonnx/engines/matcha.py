"""
Matcha-TTS inference adapter.

Matcha-TTS (flow-matching TTS) uses a **two-stage** ONNX pipeline:

1. **Mel model** — flow-matching acoustic model: phoneme IDs → mel spectrogram.
2. **Vocoder** — a *separate* ONNX model: mel spectrogram → waveform.

The vocoder is not part of the acoustic model and is published as its own
artifact (Vocos/alVoCat, Wavenext, …) that differs in output layout.  The
adapter therefore delegates mel→waveform to a :class:`BaseVocoder` obtained
from :mod:`phoonnx.engines.vocoders`, built from ``engine_params``::

    "engine_params": {
        "vocoder_path": "alvocat-vocos-22khz.onnx",
        "vocoder_type": "vocos",          # optional; auto-detected if omitted
        "vocoder_config": {"n_fft": 1024, "hop_length": 256}
    }

ONNX inputs (mel model):
  ``x``          int64    [B, T]   phoneme IDs (interspersed with blank=0)
  ``x_lengths``  int64    [B]      sequence lengths
  ``scales``     float32  [2]      [temperature, length_scale]
  ``spks``       int64    [B]      speaker ID (optional)

ONNX outputs (mel model):
  ``mel``         float32  [B, n_mels, T_mel]
  ``mel_lengths`` int64    [B]
"""
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime

from phoonnx.engines.base import (
    AdapterSynthesisRequest,
    AdapterSynthesisResult,
    BaseOnnxAdapter,
)
from phoonnx.engines.vocoders import build_vocoder
from phoonnx.engines.vocoders.base import BaseVocoder


class MatchaAdapter(BaseOnnxAdapter):
    """Adapter for Matcha-TTS ONNX models (flow-matching mel model + vocoder)."""

    def __init__(self, vocoder: Optional[BaseVocoder] = None):
        self.vocoder = vocoder
        # Raw engine params, retained so the vocoder can be built lazily if the
        # adapter was selected before configure() ran.
        self._engine_params: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, voice_config: Any) -> None:
        """Build the vocoder from ``voice_config.engine_params``."""
        engine_params = getattr(voice_config, "engine_params", None) or {}
        self.configure_from_params(engine_params)

    def configure_from_params(self, engine_params: Dict[str, Any]) -> None:
        """Build the vocoder directly from an engine_params dict."""
        self._engine_params = engine_params or {}
        if self.vocoder is None and self._engine_params.get("vocoder_path"):
            self.vocoder = build_vocoder(
                model_path=self._engine_params.get("vocoder_path"),
                vocoder_type=self._engine_params.get("vocoder_type"),
                config=self._engine_params.get("vocoder_config") or {},
            )

    def _require_vocoder(self) -> BaseVocoder:
        if self.vocoder is None:
            # Last-chance lazy build (adapter injected without configure()).
            self.configure_from_params(self._engine_params)
        if self.vocoder is None:
            raise RuntimeError(
                "Matcha-TTS requires a vocoder. Set 'vocoder_path' in "
                "config.engine_params (and optionally 'vocoder_type')."
            )
        return self.vocoder

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    def build_feed_dict(
        self,
        request: AdapterSynthesisRequest,
        session: onnxruntime.InferenceSession,
    ) -> Dict[str, np.ndarray]:
        """Build the ONNX feed dict for the Matcha mel model."""
        params = request.params
        defaults = self.default_params()
        temperature = np.float32(params.get("temperature", defaults["temperature"]))
        length_scale = np.float32(params.get("length_scale", defaults["length_scale"]))

        # Phoneme IDs come from phoonnx's tokenizer, configured with blank_id=0
        # so interspersed blanks are already present; the adapter passes them
        # straight through (no re-intersperse).
        x = request.phoneme_ids.astype(np.int64)
        x_lengths = request.phoneme_lengths.astype(np.int64)

        args: Dict[str, np.ndarray] = {
            "x": x,
            "x_lengths": x_lengths,
            "scales": np.array([temperature, length_scale], dtype=np.float32),
        }

        spk_id = params.get("spk_id", request.speaker_id)
        if spk_id is not None:
            args["spks"] = np.array([int(spk_id)], dtype=np.int64)

        return self._filter_inputs(args, session)

    def parse_outputs(
        self,
        outputs: List[np.ndarray],
        request: AdapterSynthesisRequest,
    ) -> AdapterSynthesisResult:
        """
        Produce the waveform from the mel model's outputs.

        Matcha ships in two forms:

        - **two-stage** — the model outputs a mel spectrogram ``[B, n_mels, T]``
          (rank 3); the configured vocoder reconstructs the waveform.
        - **end-to-end** — a fused model (e.g. ``*_wavenext_e2e.onnx``) outputs
          the waveform directly ``[B, T]`` / ``[T]`` (rank ≤ 2); no vocoder is
          needed.

        The output rank decides which path to take, so the same adapter drives
        both without extra configuration.
        """
        out0 = np.asarray(outputs[0])

        if out0.ndim < 3:
            # End-to-end fused model: the output already is the waveform.
            return AdapterSynthesisResult(audio=out0.squeeze().astype(np.float32))

        mel = out0
        # Trim to the valid mel length if the mel model reports it.
        if len(outputs) > 1 and outputs[1] is not None:
            try:
                valid = int(np.asarray(outputs[1]).reshape(-1)[0])
                if 0 < valid <= mel.shape[-1]:
                    mel = mel[..., :valid]
            except (ValueError, IndexError):
                pass

        vocoder = self._require_vocoder()
        denoise = bool(request.params.get("denoise", True)) and vocoder.supports_denoise
        audio = vocoder.mel_to_audio(mel.astype(np.float32), denoise=denoise)
        return AdapterSynthesisResult(audio=np.asarray(audio).squeeze())

    def default_params(self) -> Dict[str, float]:
        return {
            "temperature": 0.667,
            "length_scale": 1.0,
        }

    # ------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------

    @staticmethod
    def detect(
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> bool:
        if config is not None:
            if config.get("engine") in ("matcha", "matcha-tts"):
                return True
            if config.get("model_type") == "matcha":
                return True
            # Matcha voices carry their vocoder under engine_params.
            engine_params = config.get("engine_params") or {}
            if engine_params.get("vocoder_path") or "vocoder_path" in config:
                return True
        if session is not None:
            inputs = {inp.name for inp in session.get_inputs()}
            if {"x", "x_lengths", "scales"}.issubset(inputs):
                return True
        return False

    def parse_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Matcha per-call params and remember engine_params."""
        params: Dict[str, Any] = {}
        inference = config.get("inference", {})
        params["temperature"] = inference.get("temperature", 0.667)
        params["length_scale"] = inference.get("length_scale", 1.0)
        params["denoise"] = inference.get("denoise", True)

        engine_params = config.get("engine_params", {})
        if engine_params:
            self.configure_from_params(engine_params)
        params["spk_id"] = engine_params.get("spk_id")
        return params

    def param_labels(self) -> Dict[str, str]:
        return {
            "temperature": "Temperature",
            "length_scale": "Length Scale",
        }
