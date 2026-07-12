"""F5-TTS / Habibi-TTS inference adapter — flow-matching DiT with Euler ODE.

F5-TTS (`SWivid/F5-TTS <https://github.com/SWivid/F5-TTS>`_) is a
diffusion-transformer (DiT) TTS model using **flow matching** with Euler ODE
integration.  It is the architecture behind Habibi-TTS (Arabic dialectal) and
several community fine-tunes.

The ONNX export (e.g. `DakeQQ/F5-TTS-ONNX
<https://github.com/DakeQQ/F5-TTS-ONNX>`_) produces three graphs:

  preprocess : ref_audio + text_ids + max_duration
               → noise, rope_cos/sin, cat_mel_text, ref_signal_len
  transformer: noise, rope, cat_mel_text, time_step
               → denoised, time_step           (run in a loop, NFE steps)
  decode     : denoised, ref_signal_len
               → waveform                      (Vocos + ISTFT baked in)

This adapter overrides :meth:`synthesize` to drive the iterative ODE loop
across the three graphs.  The voice's primary model is the ``transformer``;
the ``preprocess`` and ``decode`` graphs, plus an optional standalone Vocos
vocoder, are loaded in :meth:`configure` from ``engine_params``.

**engine_params** keys (set in the voice config JSON)::

    "engine_params": {
        "preprocess_path": "F5_Preprocess.onnx",
        "decode_path":     "F5_Decode.onnx",         // optional if vocoder provided
        "vocoder_path":    "vocos-mel-24khz.onnx",   // optional, prefer over decode
        "vocoder_type":    "vocos",
        "nfe":             32,
        "cfg_strength":    2.0,
        "sway_coefficient": -1.0,
        "target_rms":      0.15,
        "sample_rate":     24000,
        "speed":           1.0
    }

The adapter also works for **Habibi-TTS** and any other model built on the
F5-TTS architecture (same ONNX layout, different weights).
"""
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime

from phoonnx.engines.base import (
    AdapterSynthesisRequest,
    AdapterSynthesisResult,
    BaseOnnxAdapter,
)


class F5TTSAdapter(BaseOnnxAdapter):
    """Adapter for F5-TTS / Habibi-TTS (flow-matching DiT, Euler ODE)."""

    SAMPLE_RATE = 24000

    def __init__(self):
        self.preprocess: Optional[onnxruntime.InferenceSession] = None
        self.decode: Optional[onnxruntime.InferenceSession] = None
        self.vocoder = None
        self._nfe: int = 32
        self._cfg_strength: float = 2.0
        self._sway_coefficient: float = -1.0
        self._target_rms: float = 0.15
        self._speed: float = 1.0

    def default_params(self) -> Dict[str, float]:
        return {
            "nfe": float(self._nfe),
            "cfg_strength": self._cfg_strength,
            "sway_coefficient": self._sway_coefficient,
            "target_rms": self._target_rms,
            "speed": self._speed,
        }

    def configure(self, voice_config: Any) -> None:
        """Load auxiliary ONNX graphs and optional vocoder from engine_params."""
        ep = getattr(voice_config, "engine_params", None) or {}
        sess = lambda p: onnxruntime.InferenceSession(
            str(p), providers=["CPUExecutionProvider"]
        )

        if self.preprocess is None and ep.get("preprocess_path"):
            self.preprocess = sess(ep["preprocess_path"])

        if self.decode is None and ep.get("decode_path"):
            self.decode = sess(ep["decode_path"])

        if self.vocoder is None and ep.get("vocoder_path"):
            from phoonnx.engines.vocoders import build_vocoder
            self.vocoder = build_vocoder(
                ep["vocoder_path"], ep.get("vocoder_type"), ep
            )

        if ep.get("sample_rate"):
            self.SAMPLE_RATE = int(ep["sample_rate"])
        if "nfe" in ep:
            self._nfe = int(ep["nfe"])
        if "cfg_strength" in ep:
            self._cfg_strength = float(ep["cfg_strength"])
        if "sway_coefficient" in ep:
            self._sway_coefficient = float(ep["sway_coefficient"])
        if "target_rms" in ep:
            self._target_rms = float(ep["target_rms"])
        if "speed" in ep:
            self._speed = float(ep["speed"])

    def synthesize(
        self,
        request: AdapterSynthesisRequest,
        session: onnxruntime.InferenceSession,
    ) -> AdapterSynthesisResult:
        """Run the full F5-TTS pipeline: preprocess → ODE loop → decode."""
        if self.preprocess is None:
            raise RuntimeError(
                "F5-TTS voice missing 'preprocess_path' in engine_params"
            )
        p = request.params
        ref_audio = p.get("reference_audio")
        ref_text_tokens = p.get("ref_text_tokens")
        if ref_audio is None or ref_text_tokens is None:
            raise RuntimeError(
                "F5-TTS needs a reference clip + transcription "
                "(params 'reference_audio' and 'ref_text_tokens')"
            )

        speed = float(p.get("speed", self._speed))
        nfe = int(p.get("nfe", self._nfe))
        cfg_strength = np.float32(p.get("cfg_strength", self._cfg_strength))
        sway_coef = np.float32(p.get("sway_coefficient", self._sway_coefficient))

        # ref_audio: (audio_array, sample_rate) tuple
        ref_audio_arr = np.asarray(ref_audio[0], np.float32).reshape(1, 1, -1)
        if ref_audio[1] != self.SAMPLE_RATE:
            ref_audio_arr = np.asarray(
                self._resample(ref_audio_arr.reshape(-1), ref_audio[1], self.SAMPLE_RATE),
                np.float32,
            ).reshape(1, 1, -1)

        # Combine ref_text + gen_text token IDs
        ref_text_ids = np.asarray(ref_text_tokens, np.int32).reshape(1, -1)
        gen_text_ids = request.phoneme_ids.astype(np.int32).reshape(1, -1)
        text_ids = np.concatenate([ref_text_ids, gen_text_ids], axis=1)

        # max_duration: estimated from ref audio length and text length ratio
        ref_audio_len = ref_audio_arr.shape[-1]
        ref_text_len = ref_text_ids.shape[1]
        gen_text_len = gen_text_ids.shape[1]
        ref_audio_frames = ref_audio_len // 256 + 1  # HOP_LENGTH = 256
        max_duration = np.array(
            [ref_audio_frames + int(ref_audio_frames / max(ref_text_len, 1) * gen_text_len / speed)],
            dtype=np.int64,
        )

        # --- Preprocess ---
        pre_out = self.preprocess.run(None, {
            "audio": ref_audio_arr.astype(np.float32),
            "text_ids": text_ids,
            "max_duration": max_duration,
        })
        # outputs: noise, rope_cos_q, rope_sin_q, rope_cos_k, rope_sin_k,
        #          cat_mel_text, cat_mel_text_drop, ref_signal_len
        noise = pre_out[0]
        rope_cos_q = pre_out[1]
        rope_sin_q = pre_out[2]
        rope_cos_k = pre_out[3]
        rope_sin_k = pre_out[4]
        cat_mel_text = pre_out[5]
        cat_mel_text_drop = pre_out[6]
        ref_signal_len = pre_out[7]

        # --- Flow-matching ODE loop (Euler method) ---
        time_step = np.array([0], dtype=np.int32)
        for _ in range(nfe):
            noise, time_step = session.run(None, {
                "noise": noise,
                "rope_cos_q": rope_cos_q,
                "rope_sin_q": rope_sin_q,
                "rope_cos_k": rope_cos_k,
                "rope_sin_k": rope_sin_k,
                "cat_mel_text": cat_mel_text,
                "cat_mel_text_drop": cat_mel_text_drop,
                "time_step": time_step,
            })

        # --- Decode ---
        if self.vocoder is not None:
            # mel path: strip ref prefix, transpose to [1, n_mels, T], vocoder
            ref_len = int(ref_signal_len) if np.ndim(ref_signal_len) == 0 else int(ref_signal_len.reshape(-1)[0])
            mel = noise[:, ref_len:].transpose(0, 2, 1).astype(np.float32)
            audio = np.asarray(self.vocoder.mel_to_audio(mel), np.float32).reshape(-1)
        elif self.decode is not None:
            audio = self.decode.run(None, {
                "denoised": noise,
                "ref_signal_len": ref_signal_len,
            })[0]
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            audio = audio.reshape(-1)
        else:
            raise RuntimeError(
                "F5-TTS requires either 'decode_path' or 'vocoder_path' "
                "in engine_params"
            )

        return AdapterSynthesisResult(audio=audio)

    # --- ABC stubs (unused — synthesize() drives the loop) ---

    def build_feed_dict(self, request, session):
        raise NotImplementedError("F5-TTS is iterative — use synthesize()")

    def parse_outputs(self, outputs, request):
        raise NotImplementedError("F5-TTS is iterative — use synthesize()")

    @staticmethod
    def detect(
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> bool:
        if config is not None:
            if config.get("engine") in ("f5tts", "f5-tts", "habibi"):
                return True
            ep = config.get("engine_params") or {}
            if ep.get("preprocess_path") and ep.get("nfe") is not None:
                return True
        if session is not None:
            inputs = {inp.name for inp in session.get_inputs()}
            if {"noise", "rope_cos_q", "cat_mel_text", "time_step"}.issubset(inputs):
                return True
        return False

    def param_labels(self) -> Dict[str, str]:
        return {
            "nfe": "NFE Steps",
            "cfg_strength": "CFG Strength",
            "sway_coefficient": "Sway Coefficient",
            "target_rms": "Target RMS",
            "speed": "Speed",
        }

    @staticmethod
    def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
        if sr == target_sr:
            return audio
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, target_sr)
            return resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
        except Exception:
            n = int(round(len(audio) * target_sr / sr))
            return np.interp(
                np.linspace(0, len(audio) - 1, n),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
