"""Ito (Lokutor) ONNX adapter — the export of
``scripts/conversion/ito/export_ito_onnx.py``.

The graph contract (batch size 1):

    input  int64  [1, L]   ito token ids, leading pad included
    speed  float  [1]      speaking rate (1.0 = the trained pace)
    wav    float  [1, T*hop]
    durations int64 [1, L] frames per token (pad included)
    mel    float  [1, 100, T]
    f0     float  [1, T]

The acoustic model conditions on a style vector baked into the graph and
uses a deterministic excitation-noise buffer baked in the same way, so a
voice needs no sidecar artifacts — one ONNX file plus its config.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime

from .base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter

_PAD_ID = 0  # "$", the StyleTTS 2 table's pad — the first row of the embedding
_MAX_TOKENS = 400  # the chip's per-sentence budget; the model was trained on it


class ItoAdapter(BaseOnnxAdapter):
    """Adapter for the single-graph ito export (English, 24 kHz)."""

    DURATION_OUTPUT_NAMES = ["durations", "dur"]

    def __init__(self):
        self._pad_id = _PAD_ID
        self._max_tokens = _MAX_TOKENS

    def default_params(self) -> Dict[str, float]:
        return {"speed": 1.0}

    def configure(self, voice_config: Any) -> None:
        """The pad id comes from the voice's vocabulary ("$" for the ito
        table); the token budget may be overridden through engine_params."""
        tokenizer = voice_config.tokenizer
        if tokenizer is not None and tokenizer.pad_id is not None:
            self._pad_id = int(tokenizer.pad_id)
        ep = getattr(voice_config, "engine_params", None) or {}
        if ep.get("max_tokens"):
            self._max_tokens = int(ep["max_tokens"])

    def build_feed_dict(
        self,
        request: AdapterSynthesisRequest,
        session: onnxruntime.InferenceSession,
    ) -> Dict[str, np.ndarray]:
        ids = np.asarray(request.phoneme_ids, dtype=np.int64)
        if ids.shape[1] + 1 > self._max_tokens:
            raise ValueError(
                f"{ids.shape[1] + 1} tokens (pad included) exceeds the ito "
                f"per-sentence budget of {self._max_tokens}: split the text "
                f"into shorter sentences")
        # the ito front end was trained with the pad token prepended to every
        # utterance (ito.text.phonemes_to_ids emits ``[0, *ids]``)
        ids = np.pad(ids, ((0, 0), (1, 0)), constant_values=self._pad_id)
        # the standard rate knob is length_scale (as for VITS): higher is
        # faster — the graph divides the predicted durations by it.
        # ``speed`` is the direct alias used when the adapter is called
        # with explicit params rather than through TTSVoice's merge.
        speed = request.params.get("length_scale")
        if speed is None:
            speed = request.params.get("speed", self.default_params()["speed"])
        speed = np.float32(speed)
        args: Dict[str, np.ndarray] = {
            "input": ids, "input_ids": ids, "tokens": ids,  # name aliases
            "speed": np.array([speed], dtype=np.float32),
        }
        return self._filter_inputs(args, session)

    def parse_outputs(
        self,
        outputs: List[np.ndarray],
        request: AdapterSynthesisRequest,
        output_names: Optional[List[str]] = None,
    ) -> AdapterSynthesisResult:
        wav = max(outputs, key=lambda o: np.asarray(o).size)
        extras: Dict[str, Any] = {}
        durations = self._find_duration_output(outputs, output_names)
        if durations is not None:
            # one entry per token, the pad included — one longer than the
            # requested phoneme ids; TTSVoice's length check degrades to
            # "alignments unavailable" on that mismatch
            extras["phoneme_id_samples"] = np.asarray(durations).squeeze()
        return AdapterSynthesisResult(audio=np.asarray(wav, dtype=np.float32).reshape(-1),
                                      extras=extras)

    @staticmethod
    def detect(
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> bool:
        return bool(config and config.get("engine") in ("ito",))

    def param_labels(self) -> Dict[str, str]:
        return {"speed": "Speed"}
