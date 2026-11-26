"""
Base adapter for ONNX TTS inference.

Each TTS architecture (VITS, OptiSpeech, Matcha-TTS, etc.) has different:
  - ONNX input names and semantics
  - Scale parameter meanings
  - Output tensor layouts
  - Config/metadata formats

An OnnxAdapter encapsulates these differences behind a single interface
so that TTSVoice.render works identically for any engine.
"""
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Mapping, Optional

import numpy as np
import onnxruntime
from quebra_frases import span_indexed_sentence_tokenize
from scriptconv.frontend import Stages, TextFrontend

from phoonnx.encoders import SIDE_STREAMS, Encoder, SymbolEncoder, TextEncoder


def pack_sentences(paragraph: str, max_len: int) -> List[str]:
    """Pack whole sentences of ``paragraph`` into chunks of at most ``max_len`` characters.

    A chunk is a slice of ``paragraph``, so the gap between two sentences stays as the
    text has it: none in Chinese or Japanese, a space in Latin script. A sentence longer
    than ``max_len`` is a chunk of its own.
    """
    chunks: List[str] = []
    start = end = None
    for sentence_start, sentence_end, _ in span_indexed_sentence_tokenize(paragraph):
        if start is not None and sentence_end - start > max_len:
            chunks.append(paragraph[start:end].strip())
            start = None
        if start is None:
            start = sentence_start
        end = sentence_end
    if start is not None:
        chunks.append(paragraph[start:end].strip())
    return chunks


def _warn_removed_hook(name: str, replacement: str) -> None:
    from phoonnx.config import _removal_version
    warnings.warn(f"BaseOnnxAdapter.{name} is deprecated and will be removed in phoonnx "
                  f"{_removal_version()}. Use {replacement}.",
                  DeprecationWarning, stacklevel=3)

INLINE_BLOCK = re.compile(r"\[\[(.*?)\]\]")
"""An inline phoneme block: the shortest ``[[...]]`` on one line, as scriptconv's frontend reads it."""


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

    # Per-phoneme stress (0 none, 1 primary, 2 secondary) for models with a stress
    # input (e.g. tiny-voca).  Shape (1, T), dtype int64, when provided.
    stress: Optional[np.ndarray] = None

    # Engine-specific key→value (noise_scale, d_factor, …).
    # Populated from SynthesisConfig + VoiceConfig defaults.
    params: Dict[str, Any] = field(default_factory=dict)

    text: Optional[str] = None
    """The chunk's text, per ``Utterance.text``."""


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
    - ``parse_onnx_meta``  — pull engine-specific fields from ONNX metadata
    - ``max_input_tokens`` — the most ids one model call reads; a longer
                             request is cut (see :meth:`_cut`)
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

    What may live on the instance is setup: what ``__init__``, ``configure``,
    ``configure_from_params``, ``parse_onnx_meta`` and ``load_voice`` derive
    from the voice's config, plus the individual values a request-path helper is
    allowed to memoize, named one by one in ``MEMOIZED_WRITES``. A memoized
    value must be safe to serve to the next request — a lazily opened
    auxiliary session, a KV-cache shape read off the model, a cache keyed on
    the input it was derived from. ``tests/test_adapter_statelessness.py``
    holds every adapter to this.
    """

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

    # True for an adapter that reads the text of a phonemizing voice itself, in ``read_text``.
    READS_TEXT: bool = False

    # True for an adapter whose engine has speaker names of its own. The voice then
    # passes a name that its ``speaker_id_map`` lacks to the engine, and does not reject it.
    SPEAKER_NAMES: bool = False

    # True for an adapter that runs a VITS graph, whose embedding has ``num_symbols`` rows.
    VITS_GRAPH: bool = False

    # The most ids one model call reads. ``None`` means the engine is never cut.
    # An engine that sets it also sets ``bos``, ``eos``, ``word_gap`` and
    # ``clause_marks`` (a set of ids) in ``configure``; :meth:`_cut` reads them.
    max_input_tokens: Optional[int] = None
    bos: int
    eos: int
    word_gap: int
    clause_marks: set

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

    def length_scale_params(self, length_scale: float) -> Dict[str, float]:
        """The engine's own speed parameters for a VITS-style ``length_scale``.

        A length scale above 1 makes the speech slower. An engine that reads
        ``length_scale`` itself keeps this default, which returns nothing. An
        engine with a speed key of its own returns that key, scaled the way the
        engine reads it.
        """
        return {}

    def pitch_shift_params(self, semitones: float, params: Mapping[str, Any],
                           session: Any = None, speaker: Optional[int] = None) -> Dict[str, float]:
        """The engine's own pitch keys for a shift of *semitones*, or nothing.

        *params* are the engine's settings as the voice resolves them (the engine's defaults,
        the voice's ``engine_params`` and the call's ``extra``), so a pitch the voice already
        sets is shifted from where it is. *session* is the voice's model, for an engine whose
        graph may lack the inputs. *speaker* is the speaker id the call resolves to, for an
        engine whose pitch depends on the speaker. ``None`` means the engine's first speaker
        (speaker 0): a single-speaker voice, or a multi-speaker voice with no default speaker,
        whose engine is fed speaker 0. Returning nothing leaves the shift to the finished audio
        (:func:`phoonnx.audio_fx.pitch_shift`). An engine that returns keys makes the shift
        itself, so the formants stay where the voice has them.
        """
        return {}

    def pitch_range_params(self, pitch_range: float, params: Mapping[str, Any],
                           session: Any = None, speaker: Optional[int] = None) -> Dict[str, float]:
        """The engine's own keys for a pitch range of *pitch_range* times the voice's, or nothing.

        ``<prosody range>`` reaches only an engine that returns keys here; on any other it
        has no effect. *params*, *session* and *speaker* are as for :meth:`pitch_shift_params`.
        """
        return {}

    @staticmethod
    def _graph_inputs(session: Any) -> Optional[set]:
        """The names of the inputs of *session*, or ``None`` when there is no session to ask."""
        try:
            return {inp.name for inp in session.get_inputs()}
        except Exception:
            return None

    @property
    def takes_symbols(self) -> bool:
        """Whether the model reads phoneme symbols, so an inline ``[[...]]`` block can reach it.

        An adapter that overrides :meth:`encode_text` reads text, and one that
        sets this to False has no symbol input at all.
        """
        return type(self).encode_text is BaseOnnxAdapter.encode_text

    def encode_text(self, text: str, voice: Any, syn_config: Any) -> List[List[int]]:
        """Turn text into the model's input token-id sequences, one per sentence.

        *text* comes from ``voice.frontend.prepare``, so it is normalised and
        diacritized already. Every engine receives text and produces the ids its
        model consumes. The default is the phoneme pipeline — phonemize with no
        text stage, then vocab-tokenize. Text-token engines (e.g. Chatterbox,
        whose subword BPE does its own normalization) override this to BPE the
        text directly. ``voice`` exposes the frontend / tokenizer.
        """
        stages = Stages(normalize=False, diacritize=False)
        return [voice.tokenizer.tokenize(p) for p in voice.frontend.phonemize(text, stages) if p]

    def make_encoder(self, voice_config: Any, frontend: TextFrontend, tokenizer: Any) -> Encoder:
        """The encoder that turns text into this engine's ids.

        The default reads the symbols of the frontend. An adapter that overrides
        :meth:`encode_text` gets a :class:`TextEncoder` that hands the whole prepared text to
        ``encode_text``, which cuts it itself. An engine that tokenizes each chunk itself returns
        a ``TextEncoder`` of its own.
        """
        if type(self).encode_text is BaseOnnxAdapter.encode_text:
            return SymbolEncoder(frontend, tokenizer)
        voice = SimpleNamespace(config=voice_config, frontend=frontend, tokenizer=tokenizer)
        return TextEncoder(frontend, lambda text, config: self.encode_text(text, voice, config),
                           whole_text=True)

    def read_text(self, text: str, voice: Any, config: Any, stages: Any) -> Iterator[Any]:
        """Yield the :class:`~phoonnx.voice.Utterance` objects of *text*, one per engine call.

        Only an adapter with ``READS_TEXT`` is asked. An engine that needs more than ids from
        the text, such as the word each token belongs to, reads it here, through
        ``voice.frontend`` and ``voice.tokenizer``, and puts what it learned in
        ``Utterance.params``. Those values join the request params of that utterance.
        """
        raise NotImplementedError

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
        are loaded in :meth:`configure` from ``engine_params``. A request longer
        than ``max_input_tokens`` runs as several pieces whose audio is joined;
        a cut request returns no extras.
        """
        results = []
        for piece in self._cut(request):
            feed = self.build_feed_dict(piece, session)
            outputs = session.run(None, feed)
            results.append(self.parse_outputs(outputs, piece, output_names=self._safe_output_names(session)))
        if len(results) == 1:
            return results[0]
        return AdapterSynthesisResult(audio=np.concatenate([r.audio for r in results]))

    def synthesize_stream(
        self,
        request: AdapterSynthesisRequest,
        session: onnxruntime.InferenceSession,
    ) -> Iterator[AdapterSynthesisResult]:
        """Yield the audio of one request in parts; joined, they are :meth:`synthesize`'s audio.

        An engine that makes audio in steps yields each part once it exists. The
        default yields the one result of :meth:`synthesize`.
        """
        yield self.synthesize(request, session)

    @staticmethod
    def detect(
        config: Optional[Dict[str, Any]] = None,
        session: Optional[onnxruntime.InferenceSession] = None,
    ) -> bool:
        """Return ``True`` if this adapter handles the given model."""
        return False

    def parse_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deprecated: nothing calls it. Engine settings come from ``default_params``."""
        _warn_removed_hook("parse_config", "default_params")
        return {}

    def parse_onnx_meta(
        self, session: onnxruntime.InferenceSession,
    ) -> Dict[str, Any]:
        """
        Extract engine-specific parameters from ONNX model metadata.
        """
        return {}

    def param_labels(self) -> Dict[str, str]:
        """Deprecated: nothing calls it. The labels are the ``default_params`` keys."""
        _warn_removed_hook("param_labels", "default_params")
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

    def _cut(self, request: AdapterSynthesisRequest) -> List[AdapterSynthesisRequest]:
        """*request* cut into pieces of at most ``max_input_tokens`` ids.

        The request's ids run from ``bos`` to ``eos``. A piece ends after the
        last clause mark that fits, else before the last word gap that fits; the
        word gap at the cut is dropped. Every piece gets ``bos`` and ``eos``
        again, and each side stream is cut at the same places. The request's own
        ``bos`` and ``eos`` keep their stream values; every ``bos`` and ``eos``
        the cut adds gets the neutral value 0.
        """
        limit = self.max_input_tokens
        if limit is None or request.phoneme_ids.shape[1] <= limit:
            return [request]
        body = request.phoneme_ids[0].tolist()[1:-1]
        room = limit - 2
        spans = []
        start = 0
        while len(body) - start > room:
            window = range(start + room - 1, start, -1)
            cut = next((i + 1 for i in window if body[i] in self.clause_marks), None)
            if cut is not None:
                end, resume = cut, cut + (body[cut] == self.word_gap)
            else:
                gap = next((i for i in range(start + room, start, -1) if body[i] == self.word_gap), None)
                if gap is None:
                    raise ValueError(f"one word is longer than {limit} tokens")
                end, resume = gap, gap + 1
            spans.append((start, end))
            start = resume
        spans.append((start, len(body)))
        streams = {name: getattr(request, name)[0].tolist()
                   for name in SIDE_STREAMS if getattr(request, name) is not None}
        pieces = []
        last = len(spans) - 1
        for index, (start, end) in enumerate(spans):
            piece_ids = [self.bos] + body[start:end] + [self.eos]
            fields = {name: np.array([[values[0] if index == 0 else 0]
                                      + values[1:-1][start:end]
                                      + [values[-1] if index == last else 0]], dtype=np.int64)
                      for name, values in streams.items()}
            pieces.append(replace(request,
                                  phoneme_ids=np.array([piece_ids], dtype=np.int64),
                                  phoneme_lengths=np.array([len(piece_ids)], dtype=np.int64),
                                  **fields))
        return pieces

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
        ones; converted to samples uniformly in ``TTSVoice.render``
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
