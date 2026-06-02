"""
Matcha-TTS inference adapter.

Matcha-TTS (flow-matching TTS) uses a two-stage ONNX pipeline:

1. **Mel model** — flow-matching acoustic model that converts phoneme IDs
   → mel spectrogram.
2. **Vocoder** — Vocos-style vocoder that converts mel spectrogram → waveform.

Both models are separate ONNX files.  The adapter holds the vocoder session
internally and runs the full pipeline in ``parse_outputs``.

ONNX inputs (mel model):
  ``x``          int64      [B, T]      phoneme IDs (interspersed with 0)
  ``x_lengths``  int64      [B]         sequence lengths
  ``scales``     float32    [2]         [temperature, length_scale]
  ``spks``       int64      [B]         speaker ID

ONNX outputs (mel model):
  ``mel``        float32    [B, n_mels, T_mel]
  ``mel_lengths`` int64     [B]

ONNX inputs (vocoder):
  ``mels``       float32    [B, n_mels, T_mel]

ONNX outputs (vocoder):
  ``mag``        float32    [B, n_fft//2+1, T]
  ``x``          float32    [B, n_fft//2+1, T]   real part
  ``y``          float32    [B, n_fft//2+1, T]   imaginary part

The vocoder outputs are combined into a complex spectrogram, then inverse
STFT with overlap-add produces the final waveform.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime
from scipy.fft import irfft
from scipy.signal.windows import hann

from phoonnx.engines.base import (
    AdapterSynthesisRequest,
    AdapterSynthesisResult,
    BaseOnnxAdapter,
)


class MatchaAdapter(BaseOnnxAdapter):
    """Adapter for Matcha-TTS ONNX models (flow-matching TTS + Vocos vocoder)."""

    def __init__(self, vocoder_path: Optional[str] = None,
                 vocoder_config: Optional[Dict[str, Any]] = None):
        self.vocoder_path = vocoder_path
        self.vocoder_config = vocoder_config or {}
        self._vocoder_session: Optional[onnxruntime.InferenceSession] = None

    # ------------------------------------------------------------------
    # Vocoder lazy loading
    # ------------------------------------------------------------------

    def _load_vocoder(self) -> onnxruntime.InferenceSession:
        if self._vocoder_session is not None:
            return self._vocoder_session
        if not self.vocoder_path:
            raise RuntimeError(
                "MatchaAdapter requires 'vocoder_path' in config.engine_params"
            )
        opts = onnxruntime.SessionOptions()
        self._vocoder_session = onnxruntime.InferenceSession(
            self.vocoder_path, sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        return self._vocoder_session

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    def build_feed_dict(
        self,
        request: AdapterSynthesisRequest,
        session: onnxruntime.InferenceSession,
    ) -> Dict[str, np.ndarray]:
        """Build ONNX feed dict for the Matcha mel model."""
        params = request.params
        defaults = self.default_params()
        temperature = np.float32(params.get("temperature", defaults["temperature"]))
        length_scale = np.float32(params.get("length_scale", defaults["length_scale"]))

        # Phoneme IDs from the request are produced by phoonnx's tokenizer.
        # For Matcha the tokenizer should be configured with blank_id=0 so
        # that the interspersed blanks are already present in the sequence.
        x = request.phoneme_ids.astype(np.int64)
        x_lengths = request.phoneme_lengths.astype(np.int64)

        args: Dict[str, np.ndarray] = {
            "x": x,
            "x_lengths": x_lengths,
            "scales": np.array([temperature, length_scale], dtype=np.float32),
        }

        # Optional speaker ID
        spk_id = request.params.get("spk_id", request.speaker_id)
        if spk_id is not None:
            args["spks"] = np.array([int(spk_id)], dtype=np.int64)

        return self._filter_inputs(args, session)

    def parse_outputs(
        self,
        outputs: List[np.ndarray],
        request: AdapterSynthesisRequest,
    ) -> AdapterSynthesisResult:
        """Run vocoder on mel output and reconstruct waveform."""
        # outputs[0] = mel, outputs[1] = mel_lengths
        mel = outputs[0]

        # Run vocoder
        vocoder = self._load_vocoder()
        mag, x_real, y_imag = vocoder.run(None, {"mels": mel})

        # Combine into complex spectrogram
        spectrogram = mag * (x_real + 1j * y_imag)

        # Optional denoising
        denoise = request.params.get("denoise", True)
        if denoise:
            spectrogram = self._denoise(mel, spectrogram, vocoder)

        # Inverse STFT + overlap-add
        audio = self._istft_overlap_add(spectrogram)
        return AdapterSynthesisResult(audio=audio.squeeze())

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
            engine = config.get("engine", "")
            if engine in ("matcha", "matcha-tts"):
                return True
            if config.get("model_type") == "matcha":
                return True
            # Matcha signature: has vocoder_path and mel model inputs x/x_lengths/scales
            if "vocoder_path" in config:
                return True
        if session is not None:
            inputs = {inp.name for inp in session.get_inputs()}
            if {"x", "x_lengths", "scales"}.issubset(inputs):
                return True
        return False

    def parse_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Matcha-specific params from JSON config."""
        params: Dict[str, Any] = {}
        inference = config.get("inference", {})
        params["temperature"] = inference.get("temperature", 0.667)
        params["length_scale"] = inference.get("length_scale", 1.0)
        params["denoise"] = inference.get("denoise", True)

        # Engine-specific config
        engine_params = config.get("engine_params", {})
        self.vocoder_path = engine_params.get("vocoder_path", self.vocoder_path)
        self.vocoder_config = engine_params.get("vocoder_config", self.vocoder_config)
        params["spk_id"] = engine_params.get("spk_id")

        return params

    def param_labels(self) -> Dict[str, str]:
        return {
            "temperature": "Temperature",
            "length_scale": "Length Scale",
        }

    # ------------------------------------------------------------------
    # Vocoder post-processing
    # ------------------------------------------------------------------

    def _denoise(
        self,
        mel: np.ndarray,
        spectrogram: np.ndarray,
        vocoder: onnxruntime.InferenceSession,
    ) -> np.ndarray:
        """Spectral subtraction denoising using a bias spectrogram."""
        mel_rand = np.zeros_like(mel)
        mag_bias, x_bias, y_bias = vocoder.run(None, {"mels": mel_rand.astype(np.float32)})
        spec_bias = mag_bias * (x_bias + 1j * y_bias)

        # Magnitudes
        spec = np.stack([np.real(spectrogram), np.imag(spectrogram)], axis=-1)
        mag_spec = np.sqrt(np.sum(spec ** 2, axis=-1))

        spec_bias_c = np.stack([np.real(spec_bias), np.imag(spec_bias)], axis=-1)
        mag_spec_bias = np.sqrt(np.sum(spec_bias_c ** 2, axis=-1))

        # Subtract
        strength = 0.0025
        mag_spec_denoised = mag_spec - mag_spec_bias * strength
        mag_spec_denoised = np.clip(mag_spec_denoised, 0.0, None)

        # Reconstruct complex spectrogram with original phase
        angle = np.arctan2(np.imag(spectrogram), np.real(spectrogram))
        return mag_spec_denoised * (np.cos(angle) + 1j * np.sin(angle))

    def _istft_overlap_add(self, spectrogram: np.ndarray) -> np.ndarray:
        """Inverse STFT with Hann window and overlap-add."""
        cfg = self.vocoder_config.get("feature_extractor", {}).get("init_args", {})
        n_fft = cfg.get("n_fft", 1024)
        hop_length = cfg.get("hop_length", 256)
        win_length = n_fft

        window = hann(win_length, sym=False)
        pad = (win_length - hop_length) // 2

        B, N, T = spectrogram.shape

        # Inverse FFT per frame
        ifft = irfft(spectrogram, n=n_fft, axis=1)
        ifft *= window[None, :, None]

        # Overlap-add
        output_size = (T - 1) * hop_length + win_length
        y = np.zeros((B, output_size))
        for b in range(B):
            for t in range(T):
                y[b, t * hop_length:t * hop_length + win_length] += ifft[b, :, t]

        # Window envelope normalization
        window_sq = np.expand_dims(window ** 2, axis=0)
        window_envelope = np.zeros((B, output_size))
        for b in range(B):
            for t in range(T):
                window_envelope[b, t * hop_length:t * hop_length + win_length] += window_sq[0]

        y /= np.maximum(window_envelope, 1e-11)
        return y
