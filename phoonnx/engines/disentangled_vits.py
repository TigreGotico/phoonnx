"""
Disentangled VITS inference adapter.

Extends the base VITS adapter with additional ONNX inputs for timbre,
articulation, and prosody reference mel spectrograms, plus an optional
emotion id.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime

from phoonnx.engines.vits import VitsAdapter


class DisentangledVitsAdapter(VitsAdapter):
    """Adapter for Disentangled VITS ONNX models."""

    def build_feed_dict(
        self,
        request,
        session: onnxruntime.InferenceSession,
    ) -> Dict[str, np.ndarray]:
        """Build ONNX feed dict including disentangled reference mels."""
        feed = super().build_feed_dict(request, session)
        params = request.params

        for key in ("timbre_ref_mel", "artic_ref_mel", "prosody_ref_mel"):
            if key in params:
                val = params[key]
                if isinstance(val, str):
                    val = self._load_mel(val)
                elif not isinstance(val, np.ndarray):
                    val = np.array(val, dtype=np.float32)
                feed[key] = val

        emotion_id = params.get("emotion_id")
        if emotion_id is not None:
            if isinstance(emotion_id, str):
                emotion_id = _DEFAULT_EMOTION_MAP.get(emotion_id, 0)
            feed["emotion_id"] = np.array([emotion_id], dtype=np.int64)

        return self._filter_inputs(feed, session)

    @staticmethod
    def detect(
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> bool:
        if config is not None:
            if config.get("model_type") == "disentangled-vits":
                return True
        if session is not None:
            inputs = {inp.name for inp in session.get_inputs()}
            if "timbre_ref_mel" in inputs and "artic_ref_mel" in inputs:
                return True
        return False

    def parse_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        base = super().parse_config(config)
        for key in ("timbre_dim", "artic_dim", "prosody_dim"):
            if key in config:
                base[key] = config[key]
        return base

    @staticmethod
    def _load_mel(audio_path: str) -> np.ndarray:
        import librosa
        import torch
        from vits.mel_processing import mel_spectrogram_torch

        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        wav = torch.from_numpy(y).unsqueeze(0)
        mel = mel_spectrogram_torch(
            wav,
            n_fft=1024,
            num_mels=80,
            sampling_rate=22050,
            hop_size=256,
            win_size=1024,
            fmin=0.0,
            fmax=None,
        )
        return mel.numpy()


_DEFAULT_EMOTION_MAP = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "fearful": 4,
}
