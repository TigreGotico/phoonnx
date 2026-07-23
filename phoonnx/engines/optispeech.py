"""
OptiSpeech inference adapter.

OptiSpeech is a non-autoregressive FastSpeech2-style TTS with GAN
vocoder.  Its ONNX contract differs from VITS:

  Inputs:
    x          (int64)     — phoneme IDs
    x_lengths  (int64)     — sequence lengths
    scales     (float32[3])— [d_factor, p_factor, e_factor]
    sids       (int64)     — speaker IDs (optional, multi-speaker)
    lids       (int64)     — language IDs (optional, multi-language)

  Outputs:
    wav          (float32) — waveform [B, 1, T]
    wav_lengths  (int64)   — sample counts [B]
    durations    (int64)   — per-phoneme durations [B, T_text]

  Metadata:
    All config is embedded in the ONNX model under the ``"inference"``
    metadata key as a JSON blob containing sample_rate, inference_args,
    input_symbols, special_symbols, text_processor, speakers, languages.

Scale semantics:
    scales[0] = d_factor  (duration scale — >1 slower, <1 faster)
    scales[1] = p_factor  (pitch scale)
    scales[2] = e_factor  (energy scale)

This is fundamentally different from VITS where scales =
[noise_scale, length_scale, noise_w_scale].
"""
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime

from phoonnx.engines.base import (
    AdapterSynthesisRequest,
    AdapterSynthesisResult,
    BaseOnnxAdapter,
)

LOG = logging.getLogger(__name__)


class OptiSpeechAdapter(BaseOnnxAdapter):
    """Adapter for OptiSpeech ONNX models."""

    # OptiSpeech's own contract names this output "durations" (see module
    # docstring); other names are speculative fallbacks for foreign exports.
    DURATION_OUTPUT_NAMES = ["durations", "dur", "phoneme_id_samples"]

    def __init__(self):
        self._onnx_meta: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    def build_feed_dict(
        self,
        request: AdapterSynthesisRequest,
        session: onnxruntime.InferenceSession,
    ) -> Dict[str, np.ndarray]:
        params = request.params
        d_factor = np.float32(params.get("d_factor", params.get("length_scale", 1.0)))
        p_factor = np.float32(params.get("p_factor", 1.0))
        e_factor = np.float32(params.get("e_factor", 1.0))

        args: Dict[str, np.ndarray] = {
            "x": request.phoneme_ids,
            "x_lengths": request.phoneme_lengths,
            "scales": np.array([d_factor, p_factor, e_factor], dtype=np.float32),
        }

        # Multi-speaker
        if request.speaker_id is not None:
            args["sids"] = np.array([request.speaker_id], dtype=np.int64)

        # Multi-language
        if request.language_id is not None:
            args["lids"] = np.array([request.language_id], dtype=np.int64)

        return self._filter_inputs(args, session)

    def parse_outputs(
        self,
        outputs: List[np.ndarray],
        request: AdapterSynthesisRequest,
        output_names: Optional[List[str]] = None,
    ) -> AdapterSynthesisResult:
        # OptiSpeech returns [wav, wav_lengths, durations]. ``durations`` is a
        # per-phoneme frame count from the internal duration predictor (like
        # VITS's w_ceil) — exposed as ``phoneme_id_samples`` so
        # ``TTSVoice.phoneme_ids_to_audio`` picks it up uniformly and scales
        # it to samples via ``VoiceConfig.hop_length``.
        audio = outputs[0].squeeze()

        extras: Dict[str, Any] = {}
        if len(outputs) > 1:
            extras["wav_lengths"] = outputs[1]

        durations = self._find_duration_output(outputs, output_names)
        if durations is None and len(outputs) > 2 and outputs[2] is not None:
            durations = outputs[2]
        if durations is not None:
            # Keep the historical "durations" key for back-compat, and also
            # expose the value as "phoneme_id_samples" — the name
            # TTSVoice.phoneme_ids_to_audio looks for uniformly across engines.
            extras["durations"] = durations
            extras["phoneme_id_samples"] = np.asarray(durations).squeeze()

        return AdapterSynthesisResult(audio=audio, extras=extras)

    def default_params(self) -> Dict[str, float]:
        return {
            "d_factor": 1.0,
            "p_factor": 1.0,
            "e_factor": 1.0,
        }

    # ------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------

    @staticmethod
    def detect(
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> bool:
        # 1. Check ONNX metadata for OptiSpeech signature
        if session is not None:
            try:
                meta = session.get_modelmeta().custom_metadata_map
                if "inference" in meta:
                    infer = json.loads(meta["inference"])
                    # OptiSpeech metadata always contains text_processor + input_symbols
                    if "text_processor" in infer and "input_symbols" in infer:
                        return True
                if meta.get("model_type") == "optispeech":
                    return True
            except Exception:
                pass

            # 2. Heuristic: has "x" + "x_lengths" inputs but NOT "input" or "input_lengths"
            input_names = {inp.name for inp in session.get_inputs()}
            output_names = {out.name for out in session.get_outputs()}
            if (
                "x" in input_names
                and "x_lengths" in input_names
                and "input" not in input_names
                and "input_lengths" not in input_names
            ):
                # Also check that outputs include wav + wav_lengths + durations
                if "wav" in output_names or "durations" in output_names:
                    return True

        # 3. Config-based
        if config is not None:
            engine = config.get("engine", "")
            if engine == "optispeech":
                return True

        return False

    def parse_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract default d/p/e factors from config if present."""
        inference_args = config.get("inference_args", {})
        return {
            "d_factor": inference_args.get("d_factor", 1.0),
            "p_factor": inference_args.get("p_factor", 1.0),
            "e_factor": inference_args.get("e_factor", 1.0),
        }

    def configure_tokenizer(self, tokenizer):
        """
        OptiSpeech models embed ``text_processor`` config in ONNX metadata
        that controls whether blanks / BOS/EOS should be inserted.  Respect
        those flags so the model receives the exact token sequence it was
        trained on.
        """
        if self._onnx_meta is None:
            return tokenizer
        tp = self._onnx_meta.get("text_processor", {})
        if tp.get("add_blank") is False:
            tokenizer.add_blank_char = False
            tokenizer.blank_at_start = False
            tokenizer.blank_at_end = False
        if tp.get("add_bos_eos") is False:
            tokenizer.use_eos_bos = False
        return tokenizer

    def parse_onnx_meta(
        self, session: onnxruntime.InferenceSession,
    ) -> Dict[str, Any]:
        """
        Extract the full config from OptiSpeech's embedded ONNX metadata.

        Returns a dict with keys:
            name, sample_rate, inference_args, input_symbols,
            special_symbols, speakers, languages, text_processor
        """
        try:
            meta = session.get_modelmeta().custom_metadata_map
            if "inference" in meta:
                self._onnx_meta = json.loads(meta["inference"])
                return self._onnx_meta
        except Exception:
            LOG.debug("Failed to parse OptiSpeech ONNX metadata", exc_info=True)
        return {}

    def param_labels(self) -> Dict[str, str]:
        return {
            "d_factor": "Speed",
            "p_factor": "Pitch",
            "e_factor": "Energy",
        }
