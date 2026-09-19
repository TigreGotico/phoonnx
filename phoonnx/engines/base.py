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

    # Per-phoneme language IDs for models that consume a parallel language stream
    # (e.g. ShamiVITS).  Shape (1, T), dtype int64, when provided.
    language_ids: Optional[np.ndarray] = None

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
    - ``DURATION_OUTPUT_NAMES`` — candidate ONNX output names that carry a
                             per-phoneme duration/frame-count tensor, used by
                             :meth:`_find_duration_output` for the alignment
                             feature (see ``docs/usage.md``).

    Adapters are stateless between requests
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    One adapter instance serves every request a voice receives, and a server
    runs those concurrently. Anything belonging to a single request — the
    reference clip, its transcription, sampling seeds, decoded prompts —
    therefore rides on the :class:`AdapterSynthesisRequest` and comes back in
    the :class:`AdapterSynthesisResult`; writing it to ``self`` hands one
    caller's voice to the next caller.

    What may live on the instance is setup: what ``SETUP_METHODS`` derive from
    the voice's config, plus the individual values a request-path helper is
    allowed to memoize, named one by one in ``MEMOIZED_WRITES``. A memoized
    value must be safe to serve to the next request — a lazily opened
    auxiliary session, a KV-cache shape read off the model, a cache keyed on
    the input it was derived from. ``tests/test_adapter_statelessness.py``
    holds every adapter to this.
    """

    # Methods that may assign anything to ``self``: the ones that run once,
    # from the voice's config, before any request is served.
    SETUP_METHODS: frozenset = frozenset({
        "__init__", "configure", "configure_from_params",
        "parse_config", "parse_onnx_meta", "load_voice",
    })

    # ``{method: {attribute}}`` — the exact writes a request-path method may
    # make, each one a claim that the value is either a property of the model
    # or a cache keyed on what produced it. Scoped per attribute so exempting
    # a method does not blanket-exempt everything else it touches.
    MEMOIZED_WRITES: Dict[str, frozenset] = {}

    # Candidate ONNX output names that expose per-phoneme durations for this
    # engine family (matched case-insensitively against
    # ``session.get_outputs()`` names by :meth:`_find_duration_output`).
    # Empty by default: engines opt in per-family. Listing a name here does
    # NOT guarantee a given checkpoint's export has it — real exports vary;
    # this only makes detection automatic once a model *does* expose one.
    DURATION_OUTPUT_NAMES: List[str] = []

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
        output_names: Optional[List[str]] = None,
    ) -> AdapterSynthesisResult:
        """
        Extract the audio waveform (and optional extras) from the raw
        list of ONNX output tensors.

        ``output_names`` — the names reported by ``session.get_outputs()``,
        positionally aligned with ``outputs`` when available (``None`` if the
        session doesn't expose them, e.g. in unit tests with a bare mock).
        Engines that support alignment use this (via
        :meth:`_find_duration_output`) to locate a durations tensor by name
        rather than by hardcoded position.
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
        try:
            return self.parse_outputs(
                outputs, request, output_names=self._safe_output_names(session)
            )
        except TypeError:
            # Third-party / test subclasses that predate the ``output_names``
            # parameter and don't accept **kwargs — fall back to the
            # original two-argument call so they keep working unmodified.
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
    def _input_rank(
        session: onnxruntime.InferenceSession, name: str,
    ) -> Optional[int]:
        """Return the declared rank (len of shape) of input ``name``, or ``None``
        if the model has no such input. Used to match a fed tensor's rank to what
        the model expects (e.g. ``scales`` declared as ``[3]`` vs ``[batch, 3]``).
        """
        for inp in session.get_inputs():
            if inp.name == name:
                shape = getattr(inp, "shape", None)
                if shape is None:
                    return None
                return len(shape)
        return None

    @staticmethod
    def _filter_inputs(
        args: Dict[str, np.ndarray],
        session: onnxruntime.InferenceSession,
    ) -> Dict[str, np.ndarray]:
        expected = {inp.name for inp in session.get_inputs()}
        return {k: v for k, v in args.items() if k in expected}

    @staticmethod
    def _safe_output_names(
        session: onnxruntime.InferenceSession,
    ) -> Optional[List[str]]:
        """Best-effort ``session.get_outputs()`` names, positionally aligned
        with ``session.run(None, ...)``'s return list.

        Never raises: a mocked/minimal session (as used in unit tests) simply
        yields ``None``, and callers must treat that as "duration detection by
        name is unavailable" rather than crashing normal synthesis.
        """
        try:
            return [getattr(o, "name", None) for o in session.get_outputs()]
        except Exception:
            return None

    def _find_duration_output(
        self,
        outputs: List[np.ndarray],
        output_names: Optional[List[str]],
    ) -> Optional[np.ndarray]:
        """Locate a per-phoneme duration/frame-count tensor among ``outputs``
        by matching ``output_names`` against ``self.DURATION_OUTPUT_NAMES``.

        Returns the raw tensor (still in the model's native unit — mel
        frames for two-stage engines, audio-decoder frames for single-stage
        ones; converted to samples uniformly in ``TTSVoice.phoneme_ids_to_audio``
        via ``VoiceConfig.hop_length``), or ``None`` if this particular
        export doesn't expose one. Never raises — callers degrade to "no
        alignment available" when this returns ``None``.
        """
        if not output_names or not self.DURATION_OUTPUT_NAMES:
            return None
        candidates = {c.lower() for c in self.DURATION_OUTPUT_NAMES}
        for name, arr in zip(output_names, outputs):
            if name and name.lower() in candidates and arr is not None:
                return np.asarray(arr)
        return None
