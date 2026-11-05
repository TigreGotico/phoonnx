"""
FastPitch inference adapter.

FastPitch (NVIDIA, https://arxiv.org/abs/2006.06873) is a non-autoregressive,
FastSpeech2-style acoustic model: text -> 80-channel mel, with pace and pitch
control. Its exported ONNX inference contract is identical to Mixer-TTS
(``token_ids`` + ``pace``/``speaker``/``pitch_mul``/``pitch_add`` -> ``mel_spec``),
so this adapter reuses the Mixer-TTS feed/parse logic (mel -> separate vocoder).

The two engines are told apart by the native config's ``engine`` field, since a
FastPitch and a Mixer-TTS ONNX are indistinguishable by their I/O alone.
"""
from typing import Any, Dict, Optional

import onnxruntime

from phoonnx.engines.mixertts import MixerTTSAdapter


class FastPitchAdapter(MixerTTSAdapter):
    """Adapter for FastPitch ONNX models (shares the Mixer-TTS contract)."""

    @staticmethod
    def detect(
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> bool:
        # FastPitch and Mixer-TTS have the same ONNX I/O, so route by the config
        # engine field only (the native config disambiguates).
        return bool(config and config.get("engine") in ("fastpitch", "fast_pitch"))

    def param_labels(self) -> Dict[str, str]:
        return {"pace": "Pace", "pitch_mul": "Pitch ×", "pitch_add": "Pitch +"}
