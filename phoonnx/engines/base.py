"""
Base adapter for ONNX TTS inference.

Each TTS architecture (VITS, OptiSpeech, Matcha-TTS, etc.) has different:
  - ONNX input names and semantics
  - Scale parameter meanings
  - Output tensor layouts
  - Config/metadata formats

An OnnxAdapter encapsulates these differences behind a single interface
so that TTSVoice.phoneme_ids_to_audio works identically for any engine.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime


@dataclass
class AdapterSynthesisRequest:
    """Architecture-agnostic container for a single synthesis call."""

    phoneme_ids: np.ndarray
    """int64 array of shape (1, T)."""

    phoneme_lengths: np.ndarray
    """int64 array of shape (1,)."""

    speaker_id: Optional[int] = None
    language_id: Optional[int] = None

    # Engine-specific key→value (noise_scale, d_factor, …).
    # Populated from SynthesisConfig + VoiceConfig defaults.
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterSynthesisResult:
    """Architecture-agnostic container returned after synthesis."""

    audio: np.ndarray
    """1-D float32 array in [-1, 1]."""

    # Optional extras (durations, pitch, energy, attention, …)
    extras: Dict[str, Any] = field(default_factory=dict)


class BaseOnnxAdapter(ABC):
    """
    Translates between the generic phoonnx voice layer and a specific
    ONNX model layout.

    Subclasses **must** implement
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - ``build_feed_dict``  — SynthesisRequest → ONNX feed dict
    - ``parse_outputs``    — raw ONNX outputs  → SynthesisResult
    - ``default_params``   — engine-default control parameters

    Subclasses **may** override
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - ``detect``           — auto-detect whether a config/session belongs
                             to this engine
    - ``parse_config``     — pull engine-specific fields from JSON config
    - ``parse_onnx_meta``  — pull engine-specific fields from ONNX metadata
    - ``param_labels``     — human-readable names for UI sliders
    """

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    @abstractmethod
    def build_feed_dict(
        self,
        request: AdapterSynthesisRequest,
        session: onnxruntime.InferenceSession,
    ) -> Dict[str, np.ndarray]:
        """
        Build the exact ``{name: array}`` dict that goes into
        ``session.run(None, feed_dict)``.
        """

    @abstractmethod
    def parse_outputs(
        self,
        outputs: List[np.ndarray],
        request: AdapterSynthesisRequest,
    ) -> AdapterSynthesisResult:
        """
        Extract the audio waveform (and optional extras) from the raw
        list of ONNX output tensors.
        """

    @abstractmethod
    def default_params(self) -> Dict[str, float]:
        """
        Return the engine's default scale / control parameters.

        Keys must match what the adapter reads from
        ``request.params``.  Examples::

            VITS:        {"noise_scale": 0.667, "length_scale": 1.0, "noise_w_scale": 0.8}
            OptiSpeech:  {"d_factor": 1.0, "p_factor": 1.0, "e_factor": 1.0}
        """

    # ------------------------------------------------------------------
    # Optional
    # ------------------------------------------------------------------

    def encode_text(self, text: str, voice: Any, syn_config: Any) -> List[List[int]]:
        """Turn (already-preprocessed) text into the model's input token-id sequences,
        one per sentence.

        Every engine receives text and produces the ids its model consumes. The default
        is the phoneme pipeline — phonemize, then vocab-tokenize. Text-token engines
        (e.g. Chatterbox, whose subword BPE does its own normalization) override this to
        BPE the raw text directly. ``voice`` exposes the phonemizer / tokenizer.
        """
        return [voice.phonemes_to_ids(p) for p in voice.phonemize(text) if p]

    def synthesize(
        self,
        request: AdapterSynthesisRequest,
        session: onnxruntime.InferenceSession,
    ) -> AdapterSynthesisResult:
        """Run full inference for one request.

        Default is the single static-graph path: ``build_feed_dict`` →
        ``session.run`` → ``parse_outputs``. Multi-ONNX / iterative engines
        (e.g. flow-matching, where ``session`` is one of several graphs and the
        output is produced by a sampling loop) override this to run their own
        pipeline. ``session`` is the voice's primary model; any auxiliary graphs
        are loaded in :meth:`configure` from ``engine_params``.
        """
        feed = self.build_feed_dict(request, session)
        outputs = session.run(None, feed)
        return self.parse_outputs(outputs, request)

    @staticmethod
    def detect(
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> bool:
        """Return ``True`` if this adapter handles the given model."""
        return False

    def parse_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract engine-specific parameters from a JSON config dict.

        Returned dict is stored on VoiceConfig and forwarded as
        ``request.params`` at synthesis time.
        """
        return {}

    def parse_onnx_meta(
        self, session: onnxruntime.InferenceSession,
    ) -> Dict[str, Any]:
        """
        Extract engine-specific parameters from ONNX model metadata.
        """
        return {}

    def param_labels(self) -> Dict[str, str]:
        """Human-readable labels for each scale param (for UIs)."""
        return {k: k for k in self.default_params()}

    def configure(self, voice_config: Any) -> None:
        """
        Set up engine-specific runtime state from the resolved VoiceConfig.

        Called once when a ``TTSVoice`` is constructed, after the adapter has
        been selected.  Engines that need extra ONNX models (e.g. Matcha-TTS
        loading its vocoder from ``voice_config.engine_params``) build them
        here.  Default: no-op.
        """
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expected_inputs(session: onnxruntime.InferenceSession) -> set:
        return {inp.name for inp in session.get_inputs()}

    @staticmethod
    def _filter_inputs(
        args: Dict[str, np.ndarray],
        session: onnxruntime.InferenceSession,
    ) -> Dict[str, np.ndarray]:
        expected = {inp.name for inp in session.get_inputs()}
        return {k: v for k, v in args.items() if k in expected}
