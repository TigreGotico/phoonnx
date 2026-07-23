"""
Mixer-TTS inference adapter.

Mixer-TTS (NVIDIA, https://arxiv.org/abs/2110.03584) is a non-autoregressive,
MLP-Mixer / FastPitch-style acoustic model: text -> 80-channel mel. Like
Matcha-TTS and GlowTTS it is **two-stage** — a separate vocoder turns the mel
into a waveform — so this adapter reuses :mod:`phoonnx.engines.vocoders`
(the reference models pair with Vocos / HiFi-GAN mels).

ONNX inputs:
  ``token_ids`` int64   [B, T]   IPA symbol ids (espeak)
  ``pace``      float32 [1]      speaking-rate scale (>1 faster)
  ``speaker``   int32   [1]      speaker id (single-speaker LJ = 0)
  ``emotion``   int32   [1]      emotion id (0)
  ``pitch_mul`` float32 [1]      pitch multiplier
  ``pitch_add`` float32 [1]      pitch offset

ONNX output:
  ``mel_spec`` ``[B, 80, T_mel]``
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


class MixerTTSAdapter(BaseOnnxAdapter):
    """Adapter for Mixer-TTS ONNX models (mel + separate vocoder)."""

    # Standard Mixer-TTS/FastPitch ONNX exports emit only ``mel_spec`` — no
    # duration output. These candidates cover the raw duration-predictor
    # tensor some FastPitch training code calls "durations" (in mel frames),
    # so a future re-export exposing it lights up alignment automatically;
    # with today's typical exports this resolves to None (see
    # docs/alignment.md).
    DURATION_OUTPUT_NAMES = ["durations", "dur", "log_durations"]

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
                "Mixer-TTS requires a vocoder. Set 'vocoder_path' (or "
                "'vocoder_type') in config.engine_params."
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
        p = request.params
        args: Dict[str, np.ndarray] = {
            "token_ids": request.phoneme_ids.astype(np.int64),
            "pace": np.array([float(p.get("pace", p.get("length_scale", 1.0)))], dtype=np.float32),
            "speaker": np.array([int(request.speaker_id or 0)], dtype=np.int32),
            "emotion": np.array([int(p.get("emotion", 0))], dtype=np.int32),
            "pitch_mul": np.array([float(p.get("pitch_mul", 1.0))], dtype=np.float32),
            "pitch_add": np.array([float(p.get("pitch_add", 0.0))], dtype=np.float32),
        }
        return self._filter_inputs(args, session)

    def parse_outputs(
        self,
        outputs: List[np.ndarray],
        request: AdapterSynthesisRequest,
        output_names: Optional[List[str]] = None,
    ) -> AdapterSynthesisResult:
        arrays = [np.asarray(o) for o in outputs if o is not None]
        mel = next((a for a in arrays if a.ndim == 3 and 16 <= a.shape[1] <= 256), None)
        if mel is None:
            mel = max(arrays, key=lambda a: a.size)
        vocoder = self._require_vocoder()
        denoise = bool(request.params.get("denoise", False)) and vocoder.supports_denoise
        audio = vocoder.mel_to_audio(mel.astype(np.float32), denoise=denoise)

        extras: Dict[str, Any] = {}
        # Mel-frame durations, when the export exposes them. Converted to
        # audio samples uniformly by TTSVoice.phoneme_ids_to_audio via
        # VoiceConfig.hop_length (mel frame rate == vocoder hop_length for
        # the standard HiFi-GAN/Vocos vocoders this adapter pairs with).
        durations = self._find_duration_output(outputs, output_names)
        if durations is not None:
            extras["phoneme_id_samples"] = durations.squeeze()

        return AdapterSynthesisResult(audio=np.asarray(audio).reshape(-1), extras=extras)

    def default_params(self) -> Dict[str, float]:
        return {"pace": 1.0, "pitch_mul": 1.0, "pitch_add": 0.0, "emotion": 0}

    # ------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------

    @staticmethod
    def detect(
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> bool:
        if config is not None and config.get("engine") in ("mixertts", "mixer_tts"):
            return True
        if session is not None:
            inputs = {inp.name for inp in session.get_inputs()}
            outs = {o.name for o in session.get_outputs()}
            # the pace/pitch_mul/pitch_add control inputs are Mixer-TTS specific
            if {"token_ids", "pace", "pitch_mul"}.issubset(inputs):
                return True
            if "mel_spec" in outs and "token_ids" in inputs:
                return True
        return False

    def param_labels(self) -> Dict[str, str]:
        return {"pace": "Pace", "pitch_mul": "Pitch ×", "pitch_add": "Pitch +"}
