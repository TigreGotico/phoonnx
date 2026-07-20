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

    # GlowTTS derives durations from the flow's log-determinant (``logw`` /
    # ``w_ceil`` in the reference Coqui implementation). ``w_ceil`` is already
    # the integer duration (ceil'd) and safe to consume directly. ``logw`` is
    # log-domain and would need ``exp()`` + rounding before use as a duration;
    # ``_find_duration_output`` has no such transform, so "logw" is
    # deliberately NOT listed here — matching it would silently feed raw
    # log-domain values through as if they were sample counts. Standard
    # Larynx/Coqui exports don't expose either as a graph output today, so
    # this resolves to None on those checkpoints; listed so a re-export
    # exposing ``w_ceil`` lights up automatically (see docs/alignment.md).
    DURATION_OUTPUT_NAMES = ["durations", "dur", "w_ceil"]

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
        output_names: Optional[List[str]] = None,
    ) -> AdapterSynthesisResult:
        arrays = [np.asarray(o) for o in outputs if o is not None]
        # Larynx glow_tts emits two rank-3 ``[B, n_mels, T]`` tensors. The mel the
        # vocoder consumes is **output index 0** — a signal-normalized mel in the
        # symmetric ``[-max_norm, max_norm]`` domain (values cluster near
        # ``[0.5, 0.6]``); output 1 is an intermediate that is NOT the vocoder mel.
        # This mirrors rhasspy larynx, which always takes ``model.run(...)[0]``.
        mels = [a for a in arrays if a.ndim == 3 and 16 <= a.shape[1] <= 256]
        mel = (mels[0] if mels else max(arrays, key=lambda a: a.size)).astype(np.float32)
        # Larynx glow_tts emits a *signal-normalized* mel (values ~[0, 1]); Coqui
        # glow_tts emits a *log-domain* mel (negative values) the vocoder consumes
        # directly, like Matcha. Invert only Larynx's normalization; pass a
        # log-domain mel through unchanged.
        if float(mel.min()) >= -0.5:
            mel = self._larynx_mel_to_vocoder(mel)
        vocoder = self._require_vocoder()
        denoise = bool(request.params.get("denoise", False)) and vocoder.supports_denoise
        audio = vocoder.mel_to_audio(mel.astype(np.float32), denoise=denoise)

        extras: Dict[str, Any] = {}
        durations = self._find_duration_output(outputs, output_names)
        if durations is not None:
            extras["phoneme_id_samples"] = durations.squeeze()

        return AdapterSynthesisResult(audio=np.asarray(audio).reshape(-1), extras=extras)

    def _larynx_mel_to_vocoder(self, mel: np.ndarray) -> np.ndarray:
        """
        Reproduce rhasspy larynx's mel post-processing between glow_tts and
        HiFi-GAN. The glow_tts output is a Coqui signal-normalized mel; larynx
        inverts that and applies the vocoder's dynamic-range compression before
        synthesis (``denormalize`` → ``db_to_amp`` → ``dynamic_range_compression``).

        Defaults are larynx's published glow_tts audio settings (identical across
        the rhasspy German/Dutch/… glow voices, e.g. de-de_eva_k-glow_tts:
        ``signal_norm`` symmetric, ``ref_level_db=20``, ``min_level_db=-100``,
        ``max_norm=1.0``, ``spec_gain=1.0``). They reproduce the genuine larynx
        render bit-for-bit. A voice may override any value via
        ``engine_params['audio']`` (the Coqui ``config['audio']`` block).
        """
        audio = self._engine_params.get("audio") or {}
        signal_norm = audio.get("signal_norm", True)
        symmetric = audio.get("symmetric_norm", True)
        clip_norm = audio.get("clip_norm", True)
        max_norm = float(audio.get("max_norm", 1.0))
        min_level_db = float(audio.get("min_level_db", -100.0))
        ref_level_db = float(audio.get("ref_level_db", 20.0))
        spec_gain = float(audio.get("spec_gain", 1.0))
        convert_db_to_amp = audio.get("convert_db_to_amp", True)
        do_drc = audio.get("do_dynamic_range_compression", True)

        if signal_norm:
            if symmetric:
                if clip_norm:
                    mel = np.clip(mel, -max_norm, max_norm)
                mel = ((mel + max_norm) * -min_level_db / (2 * max_norm)) + min_level_db
            else:
                if clip_norm:
                    mel = np.clip(mel, 0, max_norm)
                mel = (mel * -min_level_db / max_norm) + min_level_db
            mel = mel + ref_level_db
        if convert_db_to_amp:
            mel = np.power(10.0, mel / spec_gain)
        if do_drc:
            mel = np.log(np.clip(mel, 1e-5, None))
        return mel.astype(np.float32)

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
