"""
StyleTTS2-family inference adapter.

Covers StyleTTS2 (https://arxiv.org/abs/2306.07691) and its distilled derivative
Kokoro. Both consolidate to the same single-graph ONNX contract:

    tokens (int64) [+ attention_mask] [+ style (1, 256)] + speed -> waveform

- **StyleTTS2** (e.g. the stitched DDATT pipeline) bakes the reference style into
  the graph, so it only needs ``input_ids`` + ``attention_mask`` + ``speed``.
- **Kokoro** takes an explicit per-voice ``style`` vector, packed per token-length
  ([N, 256]); the adapter holds the pack and selects ``style_pack[len(tokens)]``.

The model is end-to-end (no separate vocoder). Tokenisation is the StyleTTS2 vocab
(``$``-padded at both ends); the phonemizer is espeak (StyleTTS2) or misaki (Kokoro).
"""
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter

_PAD_ID = 0  # "$" in the StyleTTS2 vocab


class StyleTTS2Adapter(BaseOnnxAdapter):
    """Adapter for StyleTTS2 / Kokoro single-graph ONNX models."""

    # StyleTTS2/Kokoro's internal duration predictor (``pred_dur``) is not
    # exposed as a graph output in standard single-waveform exports today;
    # listed so a future re-export lights up alignment automatically (see
    # docs/usage.md). Note the token sequence fed to the graph is padded
    # (see build_feed_dict), so a duration tensor here would be one-or-two
    # tokens longer than ``request.phoneme_ids`` — TTSVoice's length check
    # already degrades to "alignments unavailable" rather than crashing on
    # that mismatch.
    DURATION_OUTPUT_NAMES = ["durations", "dur", "pred_dur"]

    def __init__(self, style_pack: Optional[np.ndarray] = None,
                 speaker_encoder: Optional[Any] = None):
        # [N, 256] per-voice style indexed by token length (Kokoro); None when the
        # reference style is baked into the graph (StyleTTS2).
        self.style_pack = None if style_pack is None else np.asarray(style_pack, dtype=np.float32)
        # for zero-shot cloning models: reference wav -> 256-d style (ref_p ++ ref_s)
        self.speaker_encoder = speaker_encoder
        # id the token sequence is padded with. Defaults to the yl4579/Kokoro
        # vocab's "$" (id 0); configure() replaces it with the voice's own pad
        # token, which is not id 0 for every StyleTTS2 lineage (the ProxectoNos
        # Galician voices pad with the word separator, id 1).
        self._pad_id = _PAD_ID

    def default_params(self) -> Dict[str, float]:
        return {"speed": 1.0}

    def configure(self, voice_config: Any) -> None:
        """Load the per-voice style pack and (for cloning) the speaker encoder from
        ``voice_config.engine_params``.

        Kokoro ships one ``style`` artifact per voice — a flat float32 blob that
        reshapes to ``[N, 256]`` (N style rows indexed by token length). A cloning
        StyleTTS2 voice instead carries a ``speaker_encoder_path``.
        """
        tokenizer = voice_config.tokenizer
        if tokenizer is not None and tokenizer.pad_id is not None:
            self._pad_id = int(tokenizer.pad_id)
        ep = getattr(voice_config, "engine_params", None) or {}
        style_path = ep.get("style_path")
        if self.style_pack is None and style_path:
            self.style_pack = np.fromfile(style_path, dtype=np.float32).reshape(-1, 256)
        if self.speaker_encoder is None and ep.get("speaker_encoder_path"):
            from phoonnx.engines.speaker_encoders import build_speaker_encoder
            self.speaker_encoder = build_speaker_encoder(
                ep["speaker_encoder_path"], ep.get("speaker_encoder_type"), ep)

    def build_feed_dict(
        self,
        request: AdapterSynthesisRequest,
        session: onnxruntime.InferenceSession,
    ) -> Dict[str, np.ndarray]:
        ids = np.asarray(request.phoneme_ids, dtype=np.int64)
        # Kokoro's style pack is indexed by the token count BEFORE padding
        # (upstream kokoro-onnx does voices[voice][len(tokens)] and pads after),
        # so capture it here rather than reading it back off the padded ids.
        unpadded_len = ids.shape[1]
        # Padding with the pad id ($): Kokoro pads BOTH ends (its tokenizer emits
        # [0, *tokens, 0]); the plain StyleTTS2 lineage (single reference style or
        # a baked-in style) pads the START only -- a trailing pad makes those
        # models decode a noise burst at the end. Kokoro is the only case here
        # with a multi-row [N, 256] style pack.
        _trailing = 1 if (self.style_pack is not None and self.style_pack.shape[0] > 1) else 0
        ids = np.pad(ids, ((0, 0), (1, _trailing)), constant_values=self._pad_id)
        speed = np.float32(request.params.get(
            "speed", request.params.get("length_scale", self.default_params()["speed"])))
        args: Dict[str, np.ndarray] = {
            "input_ids": ids, "tokens": ids,                       # name aliases
            "attention_mask": np.ones_like(ids, dtype=np.int32),
            "speed": np.array([speed], dtype=np.float32),
        }
        # resolve the conditioning style: clone from a reference clip > Kokoro pack
        style = None
        ref = request.params.get("reference_audio")
        if ref is not None and self.speaker_encoder is not None:
            audio, sr = ref
            style = self.speaker_encoder.encode(audio, sr).reshape(1, -1).astype(np.float32)
        elif self.style_pack is not None:
            style = self.style_pack[
                min(unpadded_len, self.style_pack.shape[0] - 1)
            ].reshape(1, -1).astype(np.float32)
        if style is not None:
            args["style"] = style          # Kokoro
            args["style_vec"] = style       # alexbeatnik StyleTTS2
            args["s_prev"] = style          # Iam314rock StyleTTS2
            if style.shape[1] >= 256:       # cloning StyleTTS2: split into ref_p + ref_s
                args["ref"] = style[:, :128]
                args["s"] = style[:, 128:256]
        elif any(i.name in ("style", "style_vec", "s_prev") for i in session.get_inputs()):
            # The graph conditions on a style vector; without one onnxruntime only
            # reports a missing required input, which says nothing about what the
            # voice is actually missing.
            raise ValueError(
                "this StyleTTS2/Kokoro voice needs a conditioning style: pass a "
                "speaker_reference to synthesize(), or give the voice a "
                "engine_params['style_path']"
            )
        return self._filter_inputs(args, session)

    def parse_outputs(
        self,
        outputs: List[np.ndarray],
        request: AdapterSynthesisRequest,
        output_names: Optional[List[str]] = None,
    ) -> AdapterSynthesisResult:
        # the only meaningful output is the waveform (1-D)
        wav = max(outputs, key=lambda o: np.asarray(o).size)
        extras: Dict[str, Any] = {}
        durations = self._find_duration_output(outputs, output_names)
        if durations is not None:
            extras["phoneme_id_samples"] = np.asarray(durations).squeeze()
        return AdapterSynthesisResult(audio=np.asarray(wav, dtype=np.float32).reshape(-1), extras=extras)

    @staticmethod
    def detect(
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> bool:
        return bool(config and config.get("engine") in ("styletts2", "kokoro"))

    def param_labels(self) -> Dict[str, str]:
        return {"speed": "Speed"}
