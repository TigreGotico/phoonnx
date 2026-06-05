"""
GlowTTS inference adapter (Larynx — the mimic3/piper precursor).

GlowTTS is a flow-based acoustic model: text -> mel spectrogram. Like
Matcha-TTS it is **two-stage** — a separate vocoder (Larynx ships HiFi-GAN)
turns the mel into a waveform, so this adapter reuses
:mod:`phoonnx.engines.vocoders`.

ONNX inputs (glow_tts generator):
  ``input``         int64    [B, T]   phoneme IDs (gruut)
  ``input_lengths`` int64    [B]      sequence lengths
  ``scales``        float32  [2]      [noise_scale, length_scale]

ONNX outputs:
  mel spectrogram ``[B, n_mels, T_mel]`` (Larynx also emits an intermediate
  tensor; the mel is found by its ``n_mels`` axis).

The vocoder is built from ``engine_params`` (``vocoder_path`` / ``vocoder_type``)
exactly as for Matcha-TTS.
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


class GlowTTSAdapter(BaseOnnxAdapter):
    """Adapter for GlowTTS / Larynx ONNX models (flow-matching mel + vocoder)."""

    def __init__(self, vocoder: Optional[BaseVocoder] = None):
        self.vocoder = vocoder
        self._engine_params: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, voice_config: Any) -> None:
        self.configure_from_params(getattr(voice_config, "engine_params", None) or {})

    def configure_from_params(self, engine_params: Dict[str, Any]) -> None:
        self._engine_params = engine_params or {}
        # build the vocoder from a model file (hifigan/vocos/…) OR, for a
        # parametric vocoder like Griffin-Lim, from vocoder_type + config alone.
        if self.vocoder is None and (self._engine_params.get("vocoder_path")
                                     or self._engine_params.get("vocoder_type")):
            self.vocoder = build_vocoder(
                model_path=self._engine_params.get("vocoder_path"),
                vocoder_type=self._engine_params.get("vocoder_type"),
                config=self._engine_params.get("vocoder_config") or {},
            )

    def _require_vocoder(self) -> BaseVocoder:
        if self.vocoder is None:
            self.configure_from_params(self._engine_params)
        if self.vocoder is None:
            raise RuntimeError(
                "GlowTTS requires a vocoder. Set 'vocoder_path' in "
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
        params = request.params
        defaults = self.default_params()
        noise_scale = np.float32(params.get("noise_scale", defaults["noise_scale"]))
        length_scale = np.float32(params.get("length_scale", defaults["length_scale"]))
        args: Dict[str, np.ndarray] = {
            "input": request.phoneme_ids.astype(np.int64),
            "input_lengths": request.phoneme_lengths.astype(np.int64),
            "scales": np.array([noise_scale, length_scale], dtype=np.float32),
        }
        if request.speaker_id is not None:
            args["sid"] = np.array([int(request.speaker_id)], dtype=np.int64)
        return self._filter_inputs(args, session)

    def parse_outputs(
        self,
        outputs: List[np.ndarray],
        request: AdapterSynthesisRequest,
    ) -> AdapterSynthesisResult:
        arrays = [np.asarray(o) for o in outputs if o is not None]
        # the mel is the rank-3 tensor with an n_mels axis (Larynx emits an extra
        # intermediate output, so pick by shape rather than position)
        mel = next((a for a in arrays if a.ndim == 3 and 16 <= a.shape[1] <= 256), None)
        if mel is None:
            mel = max(arrays, key=lambda a: a.size)
        vocoder = self._require_vocoder()
        denoise = bool(request.params.get("denoise", False)) and vocoder.supports_denoise
        audio = vocoder.mel_to_audio(mel.astype(np.float32), denoise=denoise)
        return AdapterSynthesisResult(audio=np.asarray(audio).reshape(-1))

    def default_params(self) -> Dict[str, float]:
        return {"noise_scale": 0.667, "length_scale": 1.0}

    # ------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------

    @staticmethod
    def detect(
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> bool:
        if config is not None:
            if config.get("engine") in ("glowtts", "glow_tts", "larynx"):
                return True
            engine_params = config.get("engine_params") or {}
            if config.get("model_type") == "glow_tts":
                return True
        if session is not None:
            inputs = {inp.name for inp in session.get_inputs()}
            outs = session.get_outputs()
            # input/input_lengths/scales + a mel ([B, n_mels, T]) output
            mel_out = any(len(o.shape) == 3 and o.shape[1] in (80, 0) for o in outs)
            if {"input", "input_lengths", "scales"}.issubset(inputs) and mel_out:
                return True
        return False

    def param_labels(self) -> Dict[str, str]:
        return {"noise_scale": "Noise", "length_scale": "Length Scale"}
