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
from phoonnx.providers import make_session


class VitsAdapter(BaseOnnxAdapter):
    """Adapter for VITS-family ONNX models (piper, mimic3, coqui, phoonnx)."""

    # phoonnx's own exports name the alignment output "phoneme_id_samples";
    # other candidate names are speculative, for future/foreign exports that
    # expose a raw duration-predictor tensor under a common name.
    DURATION_OUTPUT_NAMES = ["phoneme_id_samples", "durations", "w_ceil", "dur"]

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

        # ``scales`` packs [noise, length, noise_w].  Most piper/phoonnx exports
        # declare it as a 1-D ``float32[3]``; some (e.g. the HiTZ Basque VITS
        # exports) declare it batched as ``float32[batch, 3]``.  Match whatever
        # rank the model declares so onnxruntime does not reject the feed.
        scales = np.array([noise_scale, length_scale, noise_w_scale], dtype=np.float32)
        scales_rank = self._input_rank(session, "scales")
        if scales_rank is not None and scales_rank >= 2:
            scales = scales.reshape(1, -1)

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
            "scales": scales,
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
        output_names: Optional[List[str]] = None,
    ) -> AdapterSynthesisResult:
        """Extract audio waveform from VITS ONNX outputs.

        Some phoonnx exports emit a second output tensor with per-phoneme
        sample counts (used for alignment). When present, it is exposed via
        ``AdapterSynthesisResult.extras["phoneme_id_samples"]``.

        Detection prefers matching by name (``DURATION_OUTPUT_NAMES``) when
        ``output_names`` is available; falls back to the historical
        positional convention (second output tensor) for sessions that don't
        expose output names (e.g. minimal test mocks), preserving prior
        behavior.
        """
        # VITS returns a single output tensor — squeeze to 1-D
        audio = outputs[0].squeeze()
        extras: Dict[str, Any] = {}
        durations = self._find_duration_output(outputs, output_names)
        if durations is None and len(outputs) > 1 and outputs[1] is not None:
            durations = outputs[1]
        if durations is not None:
            extras["phoneme_id_samples"] = np.asarray(durations).squeeze()
        return AdapterSynthesisResult(audio=audio, extras=extras)

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


class VitsPriorSplitAdapter(VitsAdapter):
    """VITS split at the **prior** boundary (before the flow), rather than at
    the flow/HiFiGAN boundary that :class:`~phoonnx.engines.vits_streaming.VitsStreamingAdapter`
    uses. This is the cut Inflect-v2's official ONNX release
    (``owensong/Inflect-Micro-v2-ONNX`` / ``-Nano-v2-ONNX``) ships:

      duration (primary session) : tokens, lengths, length_scale
                                    -> m_p_exp, logs_p_exp, y_mask
      decode   (aux graph)       : m_p_exp, logs_p_exp, y_mask, zp_noise, noise_scale
                                    -> waveform

    Unlike the streaming split, the latent noise for the flow prior
    (``zp_noise``) is sampled **outside** the graph rather than internally via
    ``torch.randn_like`` -- upstream's own reference script
    (``onnx/inference_onnx.py``) does this so a JS/WASM caller can seed it.
    ``synthesize`` reproduces that: it seeds ``numpy.random.default_rng`` with
    the ``seed`` control param (default 0) and draws a same-shaped standard
    normal array for ``zp_noise``. ``length_scale``/``noise_scale`` follow the
    same VITS convention as the single-graph adapter (``length_scale`` > 1 is
    slower, ``noise_scale`` is the flow's stochasticity); ``noise_w_scale`` is
    accepted but unused, matching upstream's non-stochastic (``use_sdp=False``)
    duration predictor.

    The primary session is the *duration* graph; *decode* is loaded in
    :meth:`configure` from ``engine_params["decode_path"]`` (the same
    auxiliary-graph mechanism :mod:`~phoonnx.engines.zipvoice` uses).
    """

    #: (m_p_exp, logs_p_exp, y_mask) is the exact output signature Inflect's
    #: duration.onnx exposes; other prior-split exports could differ, but no
    #: alternative is known, so this is not currently configurable.
    _DURATION_OUTPUTS = ("m_p_exp", "logs_p_exp", "y_mask")

    def __init__(self) -> None:
        self.decode: Optional[onnxruntime.InferenceSession] = None

    def configure(self, voice_config: Any) -> None:
        ep = getattr(voice_config, "engine_params", None) or {}
        if self.decode is None and ep.get("decode_path"):
            self.decode = make_session(ep["decode_path"], providers=ep.get("providers"))

    def default_params(self) -> Dict[str, float]:
        defaults = dict(super().default_params())
        defaults["seed"] = 0.0
        return defaults

    def param_labels(self) -> Dict[str, str]:
        labels = dict(super().param_labels())
        labels["seed"] = "Seed"
        return labels

    def synthesize(
        self,
        request: AdapterSynthesisRequest,
        session: onnxruntime.InferenceSession,
    ) -> AdapterSynthesisResult:
        if self.decode is None:
            raise RuntimeError(
                "VitsPriorSplitAdapter has no decode graph. Set "
                "engine_params['decode_path'] to the decode .onnx."
            )
        params = request.params
        defaults = self.default_params()
        noise_scale = np.array(params.get("noise_scale", defaults["noise_scale"]), dtype=np.float32)
        length_scale = np.array(params.get("length_scale", defaults["length_scale"]), dtype=np.float32)
        seed = int(params.get("seed", defaults["seed"]))

        tokens = np.asarray(request.phoneme_ids, dtype=np.int64).reshape(1, -1)
        lengths = np.asarray(request.phoneme_lengths, dtype=np.int64).reshape(-1)

        m_p_exp, logs_p_exp, y_mask = session.run(
            list(self._DURATION_OUTPUTS),
            {"tokens": tokens, "lengths": lengths, "length_scale": length_scale},
        )

        rng = np.random.default_rng(seed)
        zp_noise = rng.standard_normal(m_p_exp.shape).astype(np.float32)

        waveform = self.decode.run(
            ["waveform"],
            {
                "m_p_exp": m_p_exp,
                "logs_p_exp": logs_p_exp,
                "y_mask": y_mask,
                "zp_noise": zp_noise,
                "noise_scale": noise_scale,
            },
        )[0]
        audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
        return AdapterSynthesisResult(audio=audio)

    # build_feed_dict / parse_outputs are required by the ABC but unused — synthesize()
    # drives the two-graph (duration + decode) sequence directly. VitsAdapter's
    # versions would build a single-graph VITS feed / parse a single-graph output,
    # neither of which matches the duration graph's tokens/lengths/length_scale ->
    # m_p_exp/logs_p_exp/y_mask contract, so they are overridden to fail loudly
    # instead of silently misfeeding the duration graph if ever called.
    def build_feed_dict(
        self,
        request: AdapterSynthesisRequest,
        session: onnxruntime.InferenceSession,
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError("VitsPriorSplitAdapter is two-graph — use synthesize()")

    def parse_outputs(
        self,
        outputs: List[np.ndarray],
        request: AdapterSynthesisRequest,
        output_names: Optional[List[str]] = None,
    ) -> AdapterSynthesisResult:
        raise NotImplementedError("VitsPriorSplitAdapter is two-graph — use synthesize()")

    @staticmethod
    def detect(
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> bool:
        """A prior-split voice needs a decode graph *and* a duration graph whose
        outputs match the known ``m_p_exp``/``logs_p_exp``/``y_mask`` signature —
        the config flag alone isn't enough (a plain single-graph VITS voice
        must never be mistaken for one that needs a second graph it doesn't
        have)."""
        if not config:
            return False
        engine_params = config.get("engine_params") or {}
        if not engine_params.get("decode_path"):
            return False
        if session is None:
            return False
        try:
            names = {o.name for o in session.get_outputs()}
        except Exception:
            return False
        return set(VitsPriorSplitAdapter._DURATION_OUTPUTS).issubset(names)
