"""
TTSVoice — architecture-agnostic TTS voice.

This module is the main user-facing synthesis interface.  It delegates
all ONNX-specific I/O to an engine adapter (``phoonnx.engines``),
which means the same TTSVoice class works for VITS or
any future architecture without code changes here.
"""
import copy
import hashlib
import json
import os.path
import re
import shutil
import tempfile
import threading
import wave
import warnings
import weakref
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (Any, Callable, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple,
                    Union, Dict)

import numpy as np
import onnxruntime
from linguatag import closest_lang
from scriptconv.frontend import FrontendConfig, Stages, TextFrontend

from phoonnx.alignment import UNSET, alignment_model_path, ensure_alignment_session
from phoonnx.audio import write_wav
from phoonnx.config import VoiceConfig, SynthesisConfig, SpeakerReference, Alphabet, _LegacyId, PhonemeType, _voice_phonemizer, \
    _removal_version, check_lang_supported, get_text_casing, voice_frontend, without_normalizer
from scriptconv.graph import DEFAULT_GRAPH
from phoonnx.engines import detect_engine, get_adapter
from phoonnx.encoders import SIDE_STREAMS, Encoder, SymbolEncoder, Utterance, stages_of
from phoonnx.engines.base import (
    AdapterSynthesisRequest,
    AdapterSynthesisResult,
    BaseOnnxAdapter,
    INLINE_BLOCK,
)
from scriptconv.phonemizers.base import BasePhonemizer, PhonemizerIdentity
from phoonnx.providers import ProviderSpec, make_session, resolve_providers, resolve_threads
from phoonnx.reference_audio import load_reference_audio
from phoonnx import audio_fx
from phoonnx.ssml import parse_ssml
from scriptconv.phonemizers.base import PhonemizedChunks
from phoonnx.errors import FrontendMismatchError, NonFiniteAudioError, VoiceLoadError
from phoonnx.tokenizer import TTSTokenizer
from phoonnx.util import LOG


# Alias kept for callers that imported the sentinel from here before it moved.
_UNSET = UNSET

# Spellings of the r-coloured vowels for a voice whose map lacks the first one,
# best first. ɹ comes before r because English phonemizers write ɹ.
_RHOTIC_FALLBACKS = {
    "ɝ": ("ɜɹ", "ɚ", "əɹ"),
    "ɚ": ("əɹ",),
}
# espeak-ng writes the stressed r-coloured vowel of en-us and en-gb as ɜː, never ɜɹ.
_ESPEAK_NURSE = "ɜː"


def _respell_rhotics(ipa: str, phoneme_map: Mapping[str, int],
                     phoneme_type: Optional[PhonemeType] = None) -> str:
    """Replace each r-coloured vowel that ``phoneme_map`` cannot encode with the
    first fallback it can. For an espeak voice, ɝ becomes ɜː first. A vowel with no
    such fallback stays in place, so the tokenizer warns about it."""
    def held(spelling: str) -> bool:
        return all(c in phoneme_map for c in spelling)

    for vowel, fallbacks in _RHOTIC_FALLBACKS.items():
        if vowel == "ɝ" and phoneme_type == PhonemeType.ESPEAK:
            fallbacks = (_ESPEAK_NURSE, *fallbacks)
        if not held(vowel):
            ipa = ipa.replace(vowel, next(filter(held, fallbacks), vowel))
    return ipa


def _phonemic_chunks(text: str, alphabet: Optional[Alphabet]) -> PhonemizedChunks:
    """Split an already-phonemic string into per-sentence symbol lists.

    Space-separated notations (ARPA) tokenize on whitespace; char-based
    alphabets tokenize per character.
    """
    chunks = BasePhonemizer.chunk_text(text)
    if alphabet == Alphabet.ARPA:
        return [chunk.split() for chunk, _, _ in chunks if chunk.strip()]
    return [[c for c in chunk] for chunk, _, _ in chunks if chunk]


_SKIPPED_BLOCKS = re.compile(r"\s*(?:\[\[.*?\]\]\s*)+")


def _gap(match: "re.Match[str]") -> str:
    """What stands where a run of skipped blocks was: nothing when a mark follows the last block
    directly, else the whitespace around or between them with the most line breaks, else one space."""
    spaces = INLINE_BLOCK.split(match[0])[::2]
    following = match.string[match.end():match.end() + 1]
    if not spaces[-1] and not following.isalnum():
        return ""
    widest = max(spaces, key=lambda space: space.count("\n"))
    return widest if "\n" in widest else " "


def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> str:
    """Write mono float32 ``audio`` to ``path`` as 16-bit PCM WAV.

    voiceclonnx is a file-in/file-out API, so the in-memory chunks have to land on
    disk for the conversion hop.
    """
    pcm = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes((pcm * 32767.0).astype("<i2").tobytes())
    return path


def _unlink(path: str) -> None:
    with suppress(OSError):
        os.unlink(path)


class _VoiceConversion:
    """A loaded voiceclonnx cloner together with the reference WAV it reads.

    The two are one resource. voiceclonnx re-reads the reference file on every
    ``clone_voice`` call, so the file has to outlive every request still holding
    this object — and no longer. Tying its lifetime to the object rather than to
    the voice's cache slot means a later call for a different speaker can replace
    the cache without pulling the file out from under a synthesis already
    streaming, and the file still goes away once nobody is converting to it.
    """

    def __init__(self, cloner, ref_path: str):
        self.cloner = cloner
        self.ref_path = ref_path
        weakref.finalize(self, _unlink, ref_path)


# Directory of user phonetic spellings (``<lang>.txt``) applied over the packaged ones.
USER_SPELLINGS_DIR_ENV = "PHOONNX_PHONETIC_SPELLINGS_DIR"


@dataclass
class PhoneticSpellings:
    replacements: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_lang(lang: str, locale_path: str = f"{os.path.dirname(__file__)}/locale",
                  user_dir: Optional[str] = None):
        """The spellings for *lang*: the packaged file, then the user's.

        A user directory (``user_dir``, or ``$PHOONNX_PHONETIC_SPELLINGS_DIR``) may hold
        ``<lang>.txt`` files in the same ``word: spelling`` format, named by primary language
        (``fr.txt``) and/or full tag (``fr-CA.txt``). They are applied over the packaged file,
        the full tag over the primary language, so a name, a brand or a place the voice gets
        wrong can be respelled without touching the package."""
        replacements: Dict[str, str] = {}
        found = False
        lang2 = closest_lang(lang, os.listdir(locale_path))
        if lang2 is not None:
            replacements.update(
                PhoneticSpellings.from_path(f"{locale_path}/{lang2}/phonetic_spellings.txt").replacements)
            found = True
        user_dir = user_dir if user_dir is not None else os.environ.get(USER_SPELLINGS_DIR_ENV)
        if user_dir:
            tag = lang.replace("_", "-")
            for name in dict.fromkeys((tag.split("-")[0], tag)):
                for candidate in dict.fromkeys((name, name.lower())):
                    path = os.path.join(user_dir, f"{candidate}.txt")
                    if not os.path.isfile(path):
                        continue
                    try:
                        user = PhoneticSpellings.from_path(path).replacements
                    except UnicodeDecodeError as e:
                        LOG.warning(f"Skipping phonetic spellings file that is not UTF-8: {path} ({e})")
                        break
                    # apply() matches ignoring case, so a user word replaces every packaged
                    # word that differs from it only in case
                    for word, spelling in user.items():
                        for key in [k for k in replacements if k.lower() == word.lower()]:
                            del replacements[key]
                        replacements[word] = spelling
                    found = True
                    break
        if not found:
            raise FileNotFoundError(f"Spellings file for '{lang}' not found")
        return PhoneticSpellings(replacements)

    @staticmethod
    def from_path(spellings_file: str):
        replacements = {}
        with open(spellings_file, encoding="utf-8-sig") as f:
            lines = f.read().split("\n")
            for l in lines:
                l = l.strip()
                if not l or l.startswith("#"):
                    continue
                if ":" not in l:
                    LOG.warning(f"Skipping malformed phonetic spelling line: {l!r}")
                    continue
                word, spelling = l.split(":", 1)
                replacements[word.strip()] = spelling.strip()
        return PhoneticSpellings(replacements)

    def apply(self, text: str) -> str:
        for k, v in self.replacements.items():
            # Use regex to ensure word boundaries
            pattern = r'\b' + re.escape(k) + r'\b'
            # Replace using regex with case insensitivity
            text = re.sub(pattern, v, text, flags=re.IGNORECASE)
        return text


_LOUDNESS_WARNED = False


def level_audio(audio: np.ndarray, syn_config: "SynthesisConfig", sample_rate: int) -> np.ndarray:
    """Set a synthesized sentence's level: a loudness target if asked for, else the peak."""
    global _LOUDNESS_WARNED
    if not audio.size:
        return audio
    if syn_config.audio.target_loudness is not None:
        try:
            from phoonnx.audio_fx import normalize_loudness
            return normalize_loudness(audio, float(syn_config.audio.target_loudness), rate=sample_rate)
        except ImportError:
            if not _LOUDNESS_WARNED:
                _LOUDNESS_WARNED = True
                LOG.warning("target_loudness needs scipy, which is not installed: "
                            "using peak normalization")
    if syn_config.audio.normalize:
        max_val = np.max(np.abs(audio))
        if max_val < 1e-8:
            # Prevent division by zero
            return np.zeros_like(audio)
        return audio / max_val
    return audio


@dataclass
class PhonemeAlignment:
    """A single phoneme paired with the number of audio samples it spans."""

    phoneme: str
    num_samples: int


def _stretched_counts(counts: Sequence[int], ratio: float) -> np.ndarray:
    """Sample counts scaled by `ratio`, rounded at the running total so they sum to the scaled total."""
    edges = np.rint(np.cumsum(counts, dtype=np.float64) * ratio).astype(np.int64)
    return np.diff(edges, prepend=0)


@dataclass
class AudioChunk:
    """Chunk of raw audio."""

    sample_rate: int
    """Rate of chunk samples in Hertz."""

    sample_width: int
    """Width of chunk samples in bytes."""

    sample_channels: int
    """Number of channels in chunk samples."""

    audio_float_array: np.ndarray
    """Audio data as float numpy array in [-1, 1]."""

    phonemes: List[str] = field(default_factory=list)
    """Phonemes that produced this audio chunk (empty for text-token models)."""

    phoneme_ids: List[int] = field(default_factory=list)
    """Phoneme/token ids that produced this audio chunk."""

    phoneme_id_samples: Optional[np.ndarray] = None
    """Number of audio samples for each phoneme id — only populated when
    ``synthesize(include_alignments=True)`` and the model exposes durations."""

    phoneme_alignments: Optional[List[PhonemeAlignment]] = None
    """Per-phoneme (phoneme, num_samples) alignments — only populated when
    ``synthesize(include_alignments=True)`` and the model exposes durations."""

    _audio_int16_array: Optional[np.ndarray] = None
    _audio_int16_bytes: Optional[bytes] = None
    _MAX_WAV_VALUE: float = 32767.0

    @property
    def audio_int16_array(self) -> np.ndarray:
        """
        Get audio as an int16 numpy array.

        :return: Audio data as int16 numpy array.
        """
        if self._audio_int16_array is None:
            self._audio_int16_array = np.clip(
                self.audio_float_array * self._MAX_WAV_VALUE, -self._MAX_WAV_VALUE, self._MAX_WAV_VALUE
            ).astype(np.int16)

        return self._audio_int16_array

    @property
    def audio_int16_bytes(self) -> bytes:
        """
        Get audio as 16-bit PCM bytes.

        :return: Audio data as signed 16-bit sample bytes.
        """
        return self.audio_int16_array.tobytes()


# engine values that name a VITS graph, which streams once it is split
_VITS_ENGINES = ("vits", "vits_streaming", "piper", "mimic3", "coqui", "phoonnx", "transformers")


def streaming_graphs(model_path, config_dict: dict, config_dir: str,
                     engine_params: Optional[dict] = None) -> Tuple[Any, dict, bool]:
    """Give the graph to open, the config ``engine_params`` of a voice, and whether it streams.

    A config names its decoder graph relative to its own folder. A voice whose
    config sets ``"streaming": true`` and names no decoder graph gets one: its
    single VITS graph is split into an encoder and a decoder, cached next to
    it, so it can stream with no re-export. A graph that does not split loads
    as a normal voice. The voice streams when its config sets ``"streaming": true``,
    it has a decoder graph, and its engine is unset or a VITS one; it then reads
    the encoder graph through the ``vits_streaming`` adapter.
    """
    params = config_dict.get("engine_params") or {}
    if params.get("decoder_path") and not os.path.isabs(params["decoder_path"]):
        params = {**params, "decoder_path": os.path.join(config_dir, params["decoder_path"])}
    has_decoder = bool(params.get("decoder_path") or (engine_params or {}).get("decoder_path"))
    if config_dict.get("streaming") and not has_decoder:
        try:
            from phoonnx.engines.vits_split import ensure_split_vits
            model_path, decoder_path = ensure_split_vits(str(model_path))
            params = {**params, "decoder_path": decoder_path}
            has_decoder = True
        except Exception as e:
            LOG.warning(f"Streaming requested but auto-split of '{model_path}' failed ({e}); "
                        f"loading as a normal single-graph voice.")
    engine = getattr(config_dict.get("engine"), "value", config_dict.get("engine")) or "vits"
    streams = bool(config_dict.get("streaming")) and has_decoder and str(engine).lower() in _VITS_ENGINES
    return model_path, params, streams


def check_num_symbols(config: VoiceConfig, adapter: BaseOnnxAdapter, source: object) -> None:
    """Raise VoiceLoadError when a VITS-family vocabulary uses an id the embedding does not have.

    num_symbols sizes the embedding of a VITS-family graph only; other engines
    carry the field without using it.
    """
    if not adapter.VITS_GRAPH or not isinstance(config.tokenizer, TTSTokenizer):
        return
    ids = [i for value in config.tokenizer.vocabulary.char2idx.values()
           for i in (value if isinstance(value, list) else [value])]
    if ids and max(ids) >= config.num_symbols:
        raise VoiceLoadError(f"{source}: the vocabulary uses id {max(ids)}, but the model "
                             f"has num_symbols={config.num_symbols}")


@dataclass
class TTSVoice:
    session: onnxruntime.InferenceSession
    config: VoiceConfig
    phonetic_spellings: Optional[PhoneticSpellings] = None
    phonemizer: Optional[BasePhonemizer] = None
    adapter: Optional[BaseOnnxAdapter] = None

    _sr_engine: Any = field(default=None, init=False, repr=False)
    _sr_loaded: bool = field(default=False, init=False, repr=False)
    # Post-synthesis voice-conversion state (``SynthesisConfig.audio.voice_conversion``).
    # ``(key, conversion)`` in one attribute, where ``key`` is the (engine,
    # reference content) the ``_VoiceConversion`` was built for. A later call naming
    # a different engine or target speaker, or the same path holding a different
    # recording, rebuilds instead of silently reusing the wrong voice. One attribute
    # means a reader never sees one request's key next to another request's conversion.
    _vc: Optional[Tuple[tuple, _VoiceConversion]] = field(default=None, init=False, repr=False)
    # Path to the ONNX model file backing ``session``, kept so a later
    # ``include_alignments=True`` call can locate-and-patch a model that was
    # exported without the alignment output (see ``_ensure_alignment_session``).
    # ``None`` for voices constructed directly from a session (e.g. tests),
    # which simply skip the on-demand-surgery path.
    model_path: Optional[str] = None
    # Cache for the on-demand alignment-surgery outcome: unset (sentinel
    # object, distinguishable from a real ``None`` "surgery ran and failed"
    # result) until the first ``include_alignments=True`` call actually needs
    # it, so a voice that never asks for alignments never even imports
    # ``onnx``. Set to an ``onnxruntime.InferenceSession`` on success, or
    # ``None`` on a cached negative result (tried once, no dice).
    _alignment_session: Any = field(default=UNSET, repr=False, compare=False)
    # The text frontend, the phonemizer it was built for, and the values it was
    # built from. ``frontend`` rebuilds it when one of them changes.
    _frontend: Optional[TextFrontend] = field(default=None, init=False, repr=False, compare=False)
    _frontend_phonemizer: Optional[BasePhonemizer] = field(default=None, init=False, repr=False,
                                                           compare=False)
    _frontend_key: Any = field(default=None, init=False, repr=False, compare=False)
    # The frontend that with_frontend gave this voice; it replaces the one built from the phonemizer.
    _pinned_frontend: Optional[TextFrontend] = field(default=None, init=False, repr=False, compare=False)
    # The encoder, with the frontend, tokenizer and adapter it was built from.
    _encoder: Optional[Encoder] = field(default=None, init=False, repr=False, compare=False)
    _encoder_parts: Tuple[Any, ...] = field(default=(), init=False, repr=False, compare=False)
    # The thread of the provenance check; None when the voice has nothing to check.
    _provenance_thread: Optional[threading.Thread] = field(default=None, init=False, repr=False,
                                                           compare=False)

    def __post_init__(self):
        """
        Initialize optional phonetic resources after dataclass construction.
        
        Attempts to load phonetic spellings for the voice's language and, if a phonemizer was not provided, selects and assigns one based on the voice configuration.
        
        Notes:
        - If no phonetic spellings file is found for the configured language, the absence is ignored.
        """
        try:
            self.phonetic_spellings = PhoneticSpellings.from_lang(self.config.lang_code)
        except FileNotFoundError:
            pass

        # Name the voice by its model filename (e.g. "tdt-TL_joao"), not the
        # full HF snapshot path -- ``model_path`` is a cache path, not a
        # human-facing identifier.
        voice_name = Path(self.model_path).stem if self.model_path else "<voice>"

        # Phonemizer
        if self.phonemizer is None:
            # Fail fast (typed) if no scriptconv backend serves this voice's
            # language, instead of the voice loading fine and crashing mid
            # synthesis with an opaque ValueError once text is sent.
            check_lang_supported(
                voice_name,
                self.config.phonemizer_lang or self.config.lang_code,
                self.config.phoneme_type,
            )
            self.phonemizer = _voice_phonemizer(
                self.config.phoneme_type,
                self.config.alphabet,
                self.config.phonemizer_model,
                keep_punctuation=self.config.keep_punctuation,
            )

        # Engine adapter — auto-detect if not explicitly provided
        if self.adapter is None:
            engine_name = self.config.engine.value if self.config.engine else None
            try:
                # Try by engine name first
                if engine_name and engine_name not in (
                    "piper", "mimic3", "coqui", "phoonnx", "transformers",
                ):
                    self.adapter = get_adapter(engine_name)
                else:
                    # piper/mimic3/coqui/phoonnx/transformers are all VITS-style
                    # architectures (tokens in, waveform out) and share the vits adapter.
                    self.adapter = get_adapter("vits")
            except KeyError:
                # Fall back to auto-detection
                self.adapter = detect_engine(session=self.session)

        # Let the adapter set up engine-specific runtime state (e.g. Matcha
        # building its vocoder from config.engine_params).
        self.adapter.configure(self.config)
        self._provenance_thread = self._start_provenance_check(voice_name)

    def _start_provenance_check(self, voice_name: str) -> Optional[threading.Thread]:
        """Start :meth:`_check_provenance` in a daemon thread, or return None when there is nothing to check.

        The frontend is built here, on the load path as before. Only its
        ``identity()``, which can take seconds, runs in the thread, so the
        thread and a first synthesis never build the frontend twice. A frontend
        that fails to build is built again in the thread, which reports it.
        """
        if not self._has_provenance_block():
            return None
        try:
            frontend = self.frontend
        except Exception:
            frontend = None
        thread = threading.Thread(target=self._check_provenance, args=(voice_name, frontend),
                                  name=f"phoonnx-provenance-{voice_name}", daemon=True)
        thread.start()
        return thread

    def _has_provenance_block(self) -> bool:
        recorded = self.config.phonemizer_identity
        return not (recorded is None or recorded == {} or self.config.alphabet == Alphabet.GRAPHEMES)

    def _check_provenance(self, voice_name: str, frontend: Optional[TextFrontend] = None) -> None:
        """Warn when the running phonemizer differs from the one the voice was trained with.

        A load runs it in a background thread (:meth:`_start_provenance_check`)
        and never waits for it, so the warning comes when the thread ends. The config's
        ``phonemizer_identity`` names the phonemizer of training. A field that
        differs from the running identity, by
        :meth:`PhonemizerIdentity.differs_from`, gives one warning. A block that
        cannot be read, or a running phonemizer that cannot name itself, gives
        one warning too. The voice loads in every case. A voice with no block
        (``phonemizer_identity`` ``None`` or ``{}``), or one whose engine reads
        text, is not checked.

        A block with no ``frontend`` was trained through the backend alone, so
        it is compared with the backend's identity. A block with a ``frontend``
        is compared with :meth:`TextFrontend.identity`.
        """
        if not self._has_provenance_block():
            return
        recorded = self.config.phonemizer_identity
        try:
            trained = PhonemizerIdentity(**recorded)
            if frontend is None:
                frontend = self.frontend
            # identity() leaves the engine cache alone. An engine that runs for it, as
            # the tugaphone probe, runs in a child process, so it needs no lock against a synthesis.
            running = frontend.identity()
            if trained.frontend is None:
                # Preprocessing calls the backend directly and records no frontend version.
                running = frontend.backend.identity(running.lang)
        except Exception as err:
            LOG.warning(f"Voice {voice_name}: cannot compare phonemizer_identity {recorded!r} "
                        f"with the running phonemizer ({err}). The voice loads without the check.")
            return
        differing = "; ".join(
            f"{name}: trained {getattr(trained, name)!r}, running {getattr(running, name)!r}"
            for name in trained.differs_from(running))
        if differing:
            LOG.warning(f"Voice {voice_name} was trained with another phonemizer: {differing}. "
                        "The voice loads, but its pronunciation can differ from training.")

    def warmup(self) -> None:
        """
        Run one throwaway inference to pay ONNX Runtime's first-call kernel
        selection / graph optimization cost outside the first real request.

        Builds the smallest valid synthesis request (a single phoneme id)
        and routes it through the adapter's own ``synthesize`` (which calls
        ``build_feed_dict`` → ``session.run`` → ``parse_outputs``), so it
        exercises the exact same code path as a real call rather than
        hand-rolled input names. Adapters that cannot make sense of a
        minimal input (or that need extra state ``configure`` didn't set
        up) fail this quietly: warmup is an optimization, not a
        requirement, so any error is logged at debug level and swallowed.
        """
        try:
            request = AdapterSynthesisRequest(
                phoneme_ids=np.array([[1]], dtype=np.int64),
                phoneme_lengths=np.array([1], dtype=np.int64),
                speaker_id=0,
                language_id=0,
                params=dict(self.adapter.default_params()),
            )
            self.adapter.synthesize(request, self.session)
        except Exception as err:
            LOG.debug(
                f"warmup skipped for adapter '{type(self.adapter).__name__}': {err}"
            )

    @property
    def tokenizer(self) -> TTSTokenizer:
        """
        Return the tokenizer configured for this voice.
        
        Returns:
            TTSTokenizer: The TTSTokenizer instance from the voice configuration.
        """
        return self.config.tokenizer

    @staticmethod
    def load(
            model_path: Union[str, Path],
            config_path: Optional[Union[str, Path]] = None,
            vocab_path: Optional[Union[str, Path]] = None,
            tokenizer_config_path: Optional[Union[str, Path]] = None,
            phonemes_txt: Optional[str] = None,
            phoneme_map: Optional[str] = None,
            lang_code: Optional[str] = None,
            phoneme_type_str: Optional[str] = None,
            alphabet_str: Optional[str] = None,
            engine_params: Optional[Dict[str, Any]] = None,
            providers: Optional[Sequence[ProviderSpec]] = None,
            use_cuda: bool = False,
            warmup: bool = False,
            phonemizer_lang: Optional[str] = None,
            threads: Optional[int] = None,
            frontend: Optional[Union[FrontendConfig, TextFrontend]] = None,
    ) -> "TTSVoice":
        """
        Load a TTS voice ONNX model and its configuration into a TTSVoice instance.
        
        Parameters:
            model_path (str | Path): Path to the ONNX voice model file.
            config_path (str | Path, optional): Path to the JSON voice configuration. If omitted, defaults to model_path + ".json".
                A voice with no config file raises :class:`~phoonnx.errors.VoiceLoadError`.
            phonemes_txt (str, optional): Path to a tokens file (``.txt`` or ``.json``) to override the config's phoneme list.
            phoneme_map (str, optional): Optional phoneme mapping specification or file path used to map phonemes (may be None).
            lang_code (str, optional): Language code to override or set in the loaded voice configuration.
            phonemizer_lang (str, optional): The language the phonemizer reads when it differs from
                ``lang_code``. Overrides the configuration's own.
            phoneme_type_str (str, optional): Phoneme type identifier to override the configuration (for example, "arpabet" or "ipa").
            alphabet_str (str, optional): Alphabet override to pass into the VoiceConfig during load.
            providers (Sequence, optional): Ordered ONNX Runtime execution providers, e.g.
                ``["ROCMExecutionProvider", "CPUExecutionProvider"]``. When omitted, the
                ``PHOONNX_ONNX_PROVIDERS`` environment variable is used, falling back to
                auto-detecting the best provider the installed runtime offers. The resolved
                list also drives the auxiliary graphs an engine loads (vocoders, speaker
                encoders, ...).
            use_cuda (bool): Deprecated alias for ``providers=["CUDAExecutionProvider"]``.
            warmup (bool): Run a throwaway inference (see :meth:`TTSVoice.warmup`)
                before returning, so the first real ``synthesize`` call does not
                pay ONNX Runtime's first-call kernel-selection cost.
            threads (int, optional): Threads per graph operator in every session of the voice
                (``intra_op_num_threads``). When omitted, the ``PHOONNX_ORT_THREADS`` environment
                variable is used, falling back to ONNX Runtime's one thread per core.

        Returns:
            TTSVoice: A TTSVoice instance prepared with the loaded ONNX session and merged configuration.
        """
        if config_path is None:
            config_path = f"{model_path}.json"

        if not os.path.isfile(config_path):
            raise VoiceLoadError(f"No voice config at {config_path}; pass config_path to TTSVoice.load")
        with open(config_path, "r", encoding="utf-8") as config_file:
            config_dict = json.load(config_file)

        vocab_dict = {}
        tokenizer_dict = {}
        if vocab_path and os.path.isfile(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as vocab_file:
                vocab_dict = json.load(vocab_file)
            if tokenizer_config_path and os.path.isfile(tokenizer_config_path):
                with open(tokenizer_config_path, "r", encoding="utf-8") as tokenizer_file:
                    tokenizer_dict = json.load(tokenizer_file)
        resolved_providers = resolve_providers(providers, use_cuda=use_cuda)
        resolved_threads = resolve_threads(threads)

        model_path, config_params, streams = streaming_graphs(
            model_path, config_dict, os.path.dirname(os.path.abspath(config_path)), engine_params)
        if config_params:
            config_dict["engine_params"] = config_params

        session = make_session(model_path, providers=resolved_providers, threads=resolved_threads)

        # Auto-detect engine adapter from config + session
        adapter = get_adapter("vits_streaming") if streams else detect_engine(config=config_dict, session=session)

        # Engine-specific params (e.g. Matcha's vocoder_path) come either from
        # the caller (the voice manager injects locally-downloaded paths) or
        # from the model's own config JSON.
        engine_params = engine_params or config_dict.get("engine_params") or {}
        # Engines load auxiliary graphs of their own (vocoders, speaker encoders,
        # text encoders); they run on the same providers and threads as the voice itself.
        engine_params = dict(engine_params)
        engine_params.setdefault("providers", resolved_providers)
        if resolved_threads is not None:
            engine_params.setdefault("threads", resolved_threads)

        voice_config = VoiceConfig.from_dict(
            config_dict,
            vocab=vocab_dict,
            tokenizer_config=tokenizer_dict,
            alphabet=alphabet_str,
            tokens_txt=phonemes_txt,
            lang_code=lang_code,
            phonemizer_lang=phonemizer_lang,
            phoneme_type=phoneme_type_str,
            engine_params=engine_params,
        )

        if "num_symbols" in config_dict:
            check_num_symbols(voice_config, adapter, config_path)

        voice = TTSVoice(
            config=voice_config,
            session=session,
            adapter=adapter,
            model_path=str(model_path),
        )
        if frontend is not None:
            voice = voice.with_frontend(frontend)
        if warmup:
            voice.warmup()
        return voice

    def with_frontend(self, frontend: Union[FrontendConfig, TextFrontend],
                      probe_text: Optional[str] = None) -> "TTSVoice":
        """A voice that reads text through *frontend*, and shares this voice's session, tokenizer and adapter.

        The new frontend must give the symbols this voice was trained on: the same
        alphabet, and the same choice to keep or drop punctuation. With *probe_text*,
        the new frontend phonemizes it, and every symbol must be in the voice
        vocabulary. A raw-text engine reads no symbols, so it skips the probe.
        Later phonemizer changes do not reach the new voice: its frontend stays fixed.

        Raises:
            FrontendMismatchError: the alphabet or the punctuation setting differs,
                or the probe gives a symbol that the vocabulary lacks.
        """
        if isinstance(frontend, FrontendConfig):
            frontend = TextFrontend(frontend)
        own = self.frontend
        for name, new, old in (("alphabet", frontend.backend.alphabet, own.backend.alphabet),
                               ("keep_punctuation", frontend.config.keep_punctuation,
                                own.config.keep_punctuation)):
            if new != old:
                raise FrontendMismatchError(f"the frontend gives {name} {new!r}, and the voice reads {old!r}")
        if probe_text and self.adapter.takes_symbols:
            tokenizer = copy.copy(self.tokenizer)
            tokenizer.not_found_characters = set()
            encoder = (self.adapter.make_encoder(self.config, frontend, tokenizer)
                       if isinstance(self.adapter, BaseOnnxAdapter) else SymbolEncoder(frontend, tokenizer))
            for _ in encoder.encode(probe_text, SynthesisConfig()):
                pass
            if tokenizer.not_found_characters:
                raise FrontendMismatchError(
                    f"the frontend gives symbols that the voice vocabulary lacks: "
                    f"{sorted(tokenizer.not_found_characters)}")
        voice = copy.copy(self)
        voice.phonemizer = frontend.backend
        voice._pinned_frontend = frontend
        voice._encoder = None
        return voice

    @property
    def frontend(self) -> TextFrontend:
        """The text frontend that takes this voice's text to phonemes.

        It reads :attr:`phonemizer`. The phonemizer's ``normalizer`` becomes the
        frontend's normalisation hook, so it runs before diacritization, and the
        frontend reads a copy of the phonemizer without it. Every voice runs the
        hook. The frontend skips it for a backend that names ``normalize`` in its
        ``owned_stages``, because that backend normalises the text itself. Such a
        phonemizer has no ``normalizer``, so its frontend gets
        :func:`scriptconv.frontend.spoken_normalizer`, which reads only a ``<say-as>``.

        Assigning another phonemizer to :attr:`phonemizer`, as the Vosk voices
        do once their dictionary is known, reaches the frontend through
        :meth:`TextFrontend.with_backend`. A change to the phoneme type, the
        language, the text casing, or the phonemizer's ``keep_punctuation`` or
        ``normalizer`` builds a new frontend. The frontend applies the voice's
        ``text_casing`` after normalisation, as training did.
        """
        if self._pinned_frontend is not None:
            return self._pinned_frontend
        phonemizer = self.phonemizer
        key = (self.config.phoneme_type, self.config.lang_code, self.config.phonemizer_lang,
               self.config.text_casing, phonemizer.keep_punctuation, phonemizer.normalizer)
        if self._frontend is None or key != self._frontend_key:
            casing = None if self.config.text_casing == "ignore" else get_text_casing(self.config.text_casing)
            self._frontend = voice_frontend(phonemizer, self.config.lang_code, Stages(casing=casing),
                                            phonemizer_lang=self.config.phonemizer_lang)
            self._frontend_key = key
        elif phonemizer is not self._frontend_phonemizer:
            self._frontend = self._frontend.with_backend(without_normalizer(phonemizer))
        self._frontend_phonemizer = phonemizer
        return self._frontend

    @property
    def encoder(self) -> Encoder:
        """The encoder that turns text into the ids this voice's engine reads.

        The adapter builds it with :meth:`BaseOnnxAdapter.make_encoder`. An object that is
        not a ``BaseOnnxAdapter`` gets a :class:`SymbolEncoder`. The encoder is built once and
        rebuilt when :attr:`frontend`, the tokenizer or the adapter is another object, so
        assigning another phonemizer, as the Vosk voices do, reaches it.
        """
        parts = (self.frontend, self.tokenizer, self.adapter)
        if self._encoder is None or any(new is not old for new, old in zip(parts, self._encoder_parts)):
            frontend, tokenizer, adapter = parts
            if isinstance(adapter, BaseOnnxAdapter):
                self._encoder = adapter.make_encoder(self.config, frontend, tokenizer)
            else:
                self._encoder = SymbolEncoder(frontend, tokenizer)
            self._encoder_parts = parts
        return self._encoder

    def _with_voice_defaults(self, syn_config: SynthesisConfig) -> SynthesisConfig:
        """*syn_config* with the voice's own text settings where it sets none."""
        stages = syn_config.stages
        diacritize = stages.diacritize
        if diacritize is None:
            diacritize = self.config.add_diacritics
        return replace(syn_config, stages=replace(
            stages, diacritize=diacritize,
            diacritizer_model=stages.diacritizer_model or self.config.diacritizer_model))

    def _stages(self, syn_config: SynthesisConfig) -> Stages:
        """The text stages of one synthesis call: an explicit setting wins, then the voice's own."""
        return stages_of(self._with_voice_defaults(syn_config))

    def phonemize(self, text: str, lang: Optional[str] = None) -> PhonemizedChunks:
        """
        Text to phonemes grouped by sentence, with no diacritizer.

        Inline ``[[phoneme]]`` blocks reach the output as written.

        :param text: Text to phonemize.
        :param lang: Language code to phonemize in; defaults to the voice's own
            ``lang_code``. Used to phonemize a cloning reference transcription in
            *its* language (which may differ from the target's).
        :return: List of phonemes for each sentence.
        """
        return self.frontend.phonemize(text, Stages(diacritize=False, lang=lang))

    def phonemes_to_ids(self, phonemes: list[str]) -> list[int]:
        """Deprecated: use ``voice.tokenizer.tokenize``, or :meth:`prepare` for text."""
        warnings.warn("TTSVoice.phonemes_to_ids is deprecated and will be removed in phoonnx "
                      f"{_removal_version()}. Use TTSVoice.tokenizer.tokenize, or "
                      "TTSVoice.prepare for text.", DeprecationWarning, stacklevel=2)
        return self.tokenizer.tokenize(phonemes)

    def _iter_phonemes(
            self, text: str, src_alphabet: Alphabet, tgt_alphabet: Alphabet, stages: Stages,
    ) -> Iterable[Tuple[List[str], Dict[str, List[int]], Optional[str]]]:
        """Yield ``(phonemes, streams, text)`` per chunk of *text*, by whichever
        conversion its alphabet calls for. One job: getting phonemes.

        ``streams`` holds the per-symbol values a phonemizer gives beside its
        symbols, keyed by ``AdapterSynthesisRequest`` field, such as Shami's
        ``language_ids`` and dizione's ``stress``. Phonemic input carries none.
        The third element is the chunk's staged source text from
        ``Segment.text``; phonemic input has none, so it is None."""
        # Already-phonemic input: transcode to the model's alphabet through
        # scriptconv's graph (a no-op when src == tgt), then split.
        if src_alphabet != Alphabet.GRAPHEMES:
            try:
                text = DEFAULT_GRAPH.convert(text, src_alphabet.value, tgt_alphabet.value,
                                             lang=self.config.lang_code)
            except ValueError:
                LOG.debug("no conversion path %s -> %s; passing through", src_alphabet, tgt_alphabet)
            if (src_alphabet == Alphabet.ARPA and tgt_alphabet == Alphabet.IPA
                    and isinstance(self.tokenizer, TTSTokenizer)):
                text = _respell_rhotics(text, self.tokenizer.vocabulary.char2idx,
                                        self.config.phoneme_type)
            for phonemes in _phonemic_chunks(text, tgt_alphabet):
                yield phonemes, {}, None
            return

        # Graphemes: the text frontend reads the whole text, then yields one chunk at a time.
        for segment in self.frontend.segments(text, stages):
            yield list(segment.phonemes), {name: list(values) for name, values in segment.streams.items()
                                           if name in SIDE_STREAMS}, segment.text

    @property
    def _voice_name(self) -> str:
        return Path(self.model_path).stem if self.model_path else "<voice>"

    def _inline_blocks(self, text: str, alphabet: Optional[Alphabet]) -> str:
        """*text* with each ``[[...]]`` block in the voice's *alphabet*, or with the blocks left out.

        A block holds IPA. An adapter that takes no symbols has no input for it, so
        the blocks are left out with one warning. For any other voice, a block is
        converted from IPA to *alphabet* where scriptconv has a path, and kept as
        written where it has none. ARPA symbols have more than one character, and the
        frontend adds a block one character at a time, so an ARPA voice keeps the IPA.
        """
        blocks = INLINE_BLOCK.findall(text)
        if not blocks:
            return text
        if not self.adapter.takes_symbols:
            LOG.warning("Voice %s takes no symbol input: skipping the inline blocks %s", self._voice_name,
                        ["[[%s]]" % block for block in blocks])
            return _SKIPPED_BLOCKS.sub(_gap, text).strip()
        if alphabet in (None, Alphabet.IPA, Alphabet.ARPA):
            return text

        def convert(block: "re.Match[str]") -> str:
            try:
                symbols = DEFAULT_GRAPH.convert(block[1], Alphabet.IPA.value, alphabet.value,
                                                lang=self.config.lang_code)
            except ValueError:
                return block[0]
            return f"[[{symbols}]]"

        return INLINE_BLOCK.sub(convert, text)

    def prepare(self, text: str, config: Optional[SynthesisConfig] = None) -> Iterator[Utterance]:
        """Turn *text* into model input: one :class:`Utterance` per engine call.

        This is the text half of :meth:`synthesize`. It applies the phonetic
        spellings and reads the text through :attr:`frontend` with the stages of
        *config*. :attr:`encoder` turns each chunk into ids, a sentence unless
        ``config.stages.chunk`` names another size. Input in a phonemic alphabet
        (``config.input_alphabet``) is converted to the voice's alphabet, is not
        phonemized, and goes to ``encoder.encode_symbols``.

        The method is a generator, so the second chunk is read only after
        the caller has rendered the first. An utterance with no ids is not
        yielded.

        :param text: The text to read.
        :param config: The synthesis settings. Only the text settings apply.
        :raises ValueError: If ``config.language`` is a code that the
            ``lang_id_map`` of a multilingual voice lacks.
        """
        config = self._with_voice_defaults(config or SynthesisConfig())
        # An unknown language code fails here, before the phonemizer reads it.
        self._language_id(config.language)
        LOG.debug("text=%s", text)
        spoken = text
        if self.phonetic_spellings and config.phonetic_spellings:
            spoken = self.phonetic_spellings.apply(spoken)

        src_alphabet = config.input_alphabet if config.input_alphabet is not None else Alphabet.GRAPHEMES
        if src_alphabet == Alphabet.GRAPHEMES:
            spoken = self._inline_blocks(spoken, self.config.alphabet)
            if self.adapter.READS_TEXT is True:
                yield from self.adapter.read_text(spoken, self, config, stages_of(config))
            else:
                yield from self.encoder.encode(spoken, config)
            return

        if not self.adapter.takes_symbols:
            LOG.warning("Voice %s takes no symbol input: phonetic input in %s is not supported",
                        self._voice_name, src_alphabet.value)
        for symbols, _, _ in self._iter_phonemes(spoken, src_alphabet, self.config.alphabet,
                                                 stages_of(config)):
            if symbols:
                utterance = self.encoder.encode_symbols(symbols, config)
                if utterance.ids:
                    yield utterance

    def _load_super_resolution(self, syn_config: SynthesisConfig):
        """
        Lazily load the audio super-resolution engine when enabled in ``syn_config``.

        Reads ``syn_config.audio.super_resolution`` (a bool) and
        ``syn_config.audio.super_resolution_model`` (the ``audiosronnx`` engine name).
        Disabled — the default — is a strict no-op: ``audiosronnx`` is never imported
        and synthesis is untouched. When explicitly enabled, ``audiosronnx`` is
        imported lazily and the chosen engine is loaded once and cached on the
        instance; a missing install raises a clear ImportError naming the extra rather
        than silently downgrading a feature the caller asked for.

        Returns:
            The loaded SR engine, or None when disabled.
        """
        if not syn_config.audio.super_resolution:
            return None
        if self._sr_loaded:
            return self._sr_engine

        engine = syn_config.audio.super_resolution_model or "novasr"
        try:
            from audiosronnx import load_sr
        except ImportError as e:
            raise ImportError(
                "audio.super_resolution is enabled but 'audiosronnx' is not installed; "
                "install it with `pip install phoonnx[audiosr]` (or set "
                "audio.super_resolution=False)") from e

        self._sr_engine = load_sr(engine=engine)
        self._sr_loaded = True
        LOG.info(f"Audio super-resolution enabled (engine={engine}); "
                 "output will be upscaled to 48 kHz")
        return self._sr_engine

    @staticmethod
    def _maybe_upscale(audio: np.ndarray, sr_engine, in_sr: int):
        """
        Upscale ``audio`` via the super-resolution engine, if one is loaded.

        Returns ``(audio, sample_rate)`` — untouched when ``sr_engine`` is None, and
        the upscaled audio together with the rate the engine reports otherwise, so the
        caller always emits a sample rate that matches the samples it got. Falls back
        to the native audio/rate (with a warning) if the engine raises mid-stream: a
        runtime SR failure must never break synthesis that is already underway.
        """
        if sr_engine is None:
            return audio, in_sr
        try:
            up, out_sr = sr_engine.upscale(audio, in_sr)
            return up.astype(np.float32), out_sr
        except Exception as exc:
            LOG.warning(f"Super-resolution failed ({exc}); yielding native "
                        f"{in_sr} Hz audio instead")
            return audio, in_sr

    def _load_voice_conversion(
            self, syn_config: SynthesisConfig) -> Optional[_VoiceConversion]:
        """
        Lazily build the ``voiceclonnx`` cloner for ``syn_config.audio.voice_conversion``.

        Disabled — no reference, the default — is a strict no-op:
        ``voiceclonnx`` is never imported and synthesis is untouched. When enabled,
        the cloner and its materialized reference WAV are cached as one
        :class:`_VoiceConversion`, so repeated calls naming the same target speaker
        do not reload the ONNX sessions. The object is handed to the caller, which
        holds it for the whole of its synthesis; a later call for a different
        speaker replaces the cache slot without disturbing it.

        The reference is normalised by
        :func:`phoonnx.reference_audio.load_reference_audio`, the same guarded
        loader ``reference`` uses, so URLs are fetched under its address and
        size limits and non-WAV clips are decoded rather than handed to voiceclonnx
        raw. The cache is keyed on the resulting *samples*, not on the path or URL
        that produced them, because a path is not a stable identity for a
        recording: a reference rewritten in place under the same name is a
        different speaker, and serving the previous one would be a wrong answer.
        The cost of that is reading — and for a URL, fetching — the reference once
        per synthesis call.

        A missing install raises an ImportError naming the extra. A missing or
        unreadable reference raises rather than degrading: skipping the conversion
        would emit the *source* speaker, which is a wrong answer, not a
        reduced-quality one.

        Returns:
            The :class:`_VoiceConversion` to convert with, or None when disabled.
        """
        target = syn_config.audio.voice_conversion
        ref = None if target is None else target.audio
        if ref is None:
            return None

        engine = syn_config.audio.voice_conversion_engine or "openvoice"
        if isinstance(ref, (str, Path)) and not str(ref).lower().startswith(
                ("http://", "https://")) and not os.path.isfile(str(ref)):
            raise FileNotFoundError(
                f"audio.voice_conversion points at no readable file: {str(ref)!r}")

        audio, sr = load_reference_audio(ref)
        key = (engine, hashlib.sha1(audio.tobytes()).hexdigest(), int(sr))
        cached = self._vc
        if cached is not None and cached[0] == key:
            return cached[1]

        try:
            from voiceclonnx import VoiceCloner
        except ImportError as e:
            raise ImportError(
                "audio.voice_conversion is set but 'voiceclonnx' is not installed; "
                "install it with `pip install phoonnx[vc]` (or set "
                "audio.voice_conversion=None)") from e

        cloner = VoiceCloner(engine=engine)
        fd, ref_path = tempfile.mkstemp(prefix="phoonnx_vc_ref_", suffix=".wav")
        os.close(fd)
        try:
            _write_wav(ref_path, audio, sr)
        except BaseException:
            _unlink(ref_path)
            raise

        conversion = _VoiceConversion(cloner, ref_path)
        self._vc = (key, conversion)
        LOG.info(f"Voice conversion enabled (engine={engine}); output will be "
                 f"converted to the reference speaker at {cloner.sample_rate} Hz")
        return conversion

    @staticmethod
    def _maybe_convert(audio: np.ndarray, vc: Optional[_VoiceConversion],
                       in_sr: int) -> Tuple[np.ndarray, int]:
        """
        Convert ``audio`` to the target speaker, if a conversion is loaded.

        Returns ``(audio, sample_rate)`` — untouched when ``vc`` is None, and the
        converted audio at the VC engine's own rate otherwise, so the caller always
        emits a sample rate that matches the samples it got. Nothing is resampled on
        the way in: voiceclonnx accepts any input rate and resamples internally, so
        the chunk reaches it at the voice's native rate.

        Conversion is applied per chunk, not per utterance: every voiceclonnx engine
        is stateless across calls and conditions only on the reference, so a
        per-sentence application is equivalent to a whole-utterance one while keeping
        synthesis streaming. Sentences shorter than roughly 0.3 s carry too little
        content for the content encoders and pass through unconverted rather than
        turning into artefacts.

        Unlike super-resolution, a runtime failure is not swallowed: returning the
        source speaker after the caller asked for a different one is a wrong answer,
        not a degraded one.
        """
        if vc is None:
            return audio, in_sr
        if audio.size < int(0.3 * in_sr):
            LOG.debug("chunk too short for voice conversion; passing through")
            return audio, in_sr

        tmpdir = tempfile.mkdtemp(prefix="phoonnx_vc_")
        try:
            src = os.path.join(tmpdir, "src.wav")
            dst = os.path.join(tmpdir, "out.wav")
            _write_wav(src, audio, in_sr)
            vc.cloner.clone_voice(src, vc.ref_path, dst)
            return load_reference_audio(dst)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def synthesize(
            self,
            text: str,
            syn_config: Optional[SynthesisConfig] = None,
            include_alignments: bool = False,
    ) -> Iterable[AudioChunk]:
        """
        Synthesize speech from input text, yielding one AudioChunk per sentence.

        This is :meth:`render` over :meth:`prepare`: each utterance that
        ``prepare`` yields becomes one chunk. The voice conversion and
        super-resolution stages of ``syn_config`` load once for the whole call.

        Parameters:
            text (str): The input text to synthesize.
            syn_config (Optional[SynthesisConfig]): Optional synthesis options (e.g., phonetic_spellings, stages, audio). If omitted, a default SynthesisConfig is used.
            include_alignments (bool): When True, and the model exposes a
                per-phoneme duration output, each yielded AudioChunk also carries
                ``phoneme_id_samples`` and reconstructed ``phoneme_alignments``.
                When False (the default) this is a strict no-op — the exact same
                inference runs and the same audio is produced, only the extra
                per-sentence alignment reconstruction is skipped and both fields
                stay ``None``. Models that don't expose durations degrade
                gracefully: the audio is unchanged and both fields stay ``None``.

        Returns:
            Iterable[AudioChunk]: An iterable that yields one AudioChunk per synthesized sentence. Each AudioChunk contains a float32 audio array, sample rate taken from the voice config, 2-byte sample width, and 1 channel.
        """
        if syn_config is None:
            syn_config = SynthesisConfig()
        # Loaded before the text is read, so a missing extra or reference fails first.
        sr_engine = self._load_super_resolution(syn_config)
        vc = self._load_voice_conversion(syn_config)
        for utterance in self.prepare(text, syn_config):
            yield self._render(utterance, syn_config, include_alignments, sr_engine, vc)

    def stream(self, text: str, config: Optional[SynthesisConfig] = None) -> Iterator[AudioChunk]:
        """Synthesize *text* and yield its audio as the engine makes it.

        An engine with ``synthesize_stream`` yields parts of a sentence, and any other
        engine one chunk per sentence. The parts join to the audio of :meth:`synthesize`.
        Peak normalisation, a loudness target, voice conversion and super-resolution need
        the whole sentence, so with any of them on each sentence is one chunk. The first
        part of a sentence carries its phonemes and ids.
        """
        if config is None:
            config = SynthesisConfig()
        sr_engine = self._load_super_resolution(config)
        vc = self._load_voice_conversion(config)
        whole_sentence = (config.audio.normalize or config.audio.target_loudness is not None
                          or sr_engine is not None or vc is not None
                          or self._pace_stretch(config) != 1.0)
        for utterance in self.prepare(text, config):
            if whole_sentence:
                yield self._render(utterance, config, False, sr_engine, vc)
                continue
            phonemes, ids = list(utterance.symbols or []), list(utterance.ids)
            request = self._request(utterance.ids, self._with_params(config, utterance),
                                    utterance.streams, utterance.text)
            for part in self.adapter.synthesize_stream(request, self.session):
                yield AudioChunk(sample_rate=self.config.sample_rate, sample_width=2,
                                 sample_channels=1, audio_float_array=self._level(part.audio, config),
                                 phonemes=phonemes, phoneme_ids=ids)
                phonemes, ids = [], []

    def render(self, utterance: Utterance, config: Optional[SynthesisConfig] = None, *,
               alignments: bool = False) -> AudioChunk:
        """Turn one :class:`Utterance` into audio: the audio half of :meth:`synthesize`.

        The engine reads the ids and streams of *utterance* with the engine
        settings of *config*. The audio is then levelled, scaled by
        ``config.audio.volume`` and clipped to [-1, 1]. Voice conversion and
        super-resolution run last, when *config* asks for them, and the chunk
        reports the sample rate that they give.

        :param utterance: An utterance from :meth:`prepare`, or one built by the
            caller with ids for this voice.
        :param config: The synthesis settings. The text settings do not apply.
        :param alignments: As ``include_alignments`` in :meth:`synthesize`.
        :return: One chunk, also when the model gives no durations.
        """
        if config is None:
            config = SynthesisConfig()
        return self._render(utterance, config, alignments,
                            self._load_super_resolution(config), self._load_voice_conversion(config))

    def _render(self, utterance: Utterance, syn_config: SynthesisConfig, include_alignments: bool,
                sr_engine, vc: Optional[_VoiceConversion]) -> AudioChunk:
        """:meth:`render` with the super-resolution engine and the voice conversion already loaded."""
        # Special-token ids for alignment reconstruction, read only when alignments are requested.
        _idx2char: dict = {}
        _blank_id = _bos_id = _eos_id = None
        if include_alignments:
            _tok = self.config.tokenizer
            _vocab = getattr(_tok, "vocabulary", None)
            if _vocab is not None:
                _idx2char = _vocab.idx2char
                _blank_id = _tok.blank_id
                _bos_id = _vocab.bos_id
                _eos_id = _vocab.eos_id
            else:
                # Tokenizers such as BPETokenizer/ChatterboxMTLTokenizer expose
                # no ``vocabulary`` attribute — alignment reconstruction is not
                # possible for them. Degrade to "alignments unavailable"
                # instead of raising, since alignment is an optional feature.
                include_alignments = False

        syn_config = self._with_params(syn_config, utterance)
        audio, phoneme_id_samples = self._infer(
            utterance.ids, syn_config, include_alignments, utterance.streams,
            text=utterance.text)

        factor = self._pace_stretch(syn_config)
        mono = np.squeeze(np.asarray(audio, dtype=np.float32))
        if factor != 1.0 and mono.ndim <= 1:  # a stretch is for one channel of samples
            mono = mono.reshape(-1)
            stretched = audio_fx.time_stretch(mono, factor, self.config.sample_rate)
            if phoneme_id_samples is not None:
                phoneme_id_samples = _stretched_counts(phoneme_id_samples,
                                                       len(stretched) / max(1, mono.size))
            audio = stretched

        audio = self._level(audio, syn_config)

        phoneme_alignments = self._reconstruct_alignments(
            list(utterance.ids), phoneme_id_samples,
            _idx2char, _blank_id, _bos_id, _eos_id,
        ) if phoneme_id_samples is not None else None

        # VC and SR each change the rate of the samples, so the chunk must
        # report the rate that came back with them, not the voice's native one
        audio, chunk_sr = self._maybe_convert(
            audio, vc, self.config.sample_rate)
        audio, chunk_sr = self._maybe_upscale(audio, sr_engine, chunk_sr)

        return AudioChunk(
            sample_rate=chunk_sr,
            sample_width=2,
            sample_channels=1,
            audio_float_array=audio,
            phonemes=list(utterance.symbols or []),
            phoneme_ids=list(utterance.ids),
            phoneme_id_samples=phoneme_id_samples,
            phoneme_alignments=phoneme_alignments,
        )

    def _voice_length_scale(self) -> Any:
        """The voice's own length scale, the one a call's is measured against: the voice's
        ``engine_params``, then its ``inference`` block, then the engine's default, then 1.
        It is the order in which :meth:`_request` merges them."""
        engine_params = getattr(self.config, "engine_params", None) or {}
        for value in (engine_params.get("length_scale"), self.config.length_scale,
                      self.adapter.default_params().get("length_scale")):
            if value is not None:
                return value
        return 1.0

    def _pace_stretch(self, syn_config: SynthesisConfig) -> float:
        """How much ``audio.stretch_pace`` stretches this call's audio (1.0: not at all).

        Only an engine without pace control is stretched. The factor is the call's length
        scale (``extra`` first, then ``length_scale``) over the voice's own
        (:meth:`_voice_length_scale`), within 0.5 to 2.
        """
        if not syn_config.audio.stretch_pace or self.adapter.PACE_CONTROL:
            return 1.0
        call = syn_config.extra.get("length_scale", syn_config.length_scale)
        if call is None:
            return 1.0
        try:
            factor = float(call) / float(self._voice_length_scale())
        except (TypeError, ValueError, ZeroDivisionError):
            return 1.0
        if not np.isfinite(factor) or factor <= 0 or abs(factor - 1.0) < 1e-3:
            return 1.0
        return min(2.0, max(0.5, factor))

    @staticmethod
    def _with_params(syn_config: SynthesisConfig, utterance: Utterance) -> SynthesisConfig:
        """*syn_config* with the params of *utterance* joined to its ``extra``."""
        if not utterance.params:
            return syn_config
        return replace(syn_config, extra={**syn_config.extra, **utterance.params})

    def _level(self, audio: np.ndarray, syn_config: SynthesisConfig) -> np.ndarray:
        """The engine's *audio* levelled, scaled by ``syn_config.audio.volume`` and clipped to [-1, 1].

        Raises :class:`NonFiniteAudioError` when the engine returned NaN or inf, which
        the int16 cast would write as silence.
        """
        if not np.isfinite(audio).all():
            raise NonFiniteAudioError(self.model_path or "<voice>")
        audio = level_audio(audio, syn_config, self.config.sample_rate)
        if syn_config.audio.volume != 1.0:
            audio = audio * syn_config.audio.volume
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    def _alignment_model_path(self) -> Optional[Path]:
        """Where this voice's alignment-patched model is cached, if it has one."""
        if not self.model_path:
            return None
        return alignment_model_path(self.model_path)

    def _ensure_alignment_session(self) -> Optional[onnxruntime.InferenceSession]:
        """At-most-once, best-effort attempt to retrofit a duration output onto
        this voice's model.

        Returns an alternate session with the duration tensor exposed as an
        extra output, or ``None`` when this model cannot produce one. The
        outcome — including the negative one — is cached on the instance, so
        the surgery in :mod:`phoonnx.alignment` is attempted at most once per
        voice.
        """
        if self._alignment_session is not UNSET:
            return self._alignment_session

        if not self.model_path:
            LOG.debug("no alignment output: this voice's model_path is "
                      "unknown, cannot retrofit one")
            result = None
        else:
            try:
                providers = self.session.get_providers()
                threads = self.session.get_session_options().intra_op_num_threads
            except Exception:
                providers = threads = None
            result = ensure_alignment_session(self.model_path, providers, threads)

        self._alignment_session = result
        return result

    @staticmethod
    def _reconstruct_alignments(
            phoneme_ids: List[int],
            phoneme_id_samples: np.ndarray,
            idx2char: dict,
            blank_id: Optional[int],
            bos_id: Optional[int],
            eos_id: Optional[int],
    ) -> Optional[List[PhonemeAlignment]]:
        """Fold per-id sample counts onto the real phonemes.

        Walk ``phoneme_ids`` and their per-id sample counts together. Blank/bos
        durations are absorbed forward into the next real phoneme; eos duration
        is absorbed backward into the last real phoneme. This works regardless
        of the tokenizer's blank_at_start/end settings. Returns ``None`` when
        the sample counts don't line up one-to-one with the ids (e.g. a padded
        model input), so callers degrade to "alignments unavailable" rather
        than emitting a wrong alignment.
        """
        if np.size(phoneme_id_samples) != len(phoneme_ids):
            return None
        alignments: List[PhonemeAlignment] = []
        pending: int = 0
        for pid, n_samples in zip(phoneme_ids, phoneme_id_samples.tolist()):
            if pid == bos_id or pid == blank_id:
                pending += n_samples
            elif pid == eos_id:
                if alignments:
                    prev = alignments[-1]
                    alignments[-1] = PhonemeAlignment(prev.phoneme, prev.num_samples + n_samples)
            else:
                token = idx2char.get(pid, "?")
                alignments.append(PhonemeAlignment(token, pending + n_samples))
                pending = 0
        # Absorb any leftover pending (e.g. trailing blank with no phoneme after).
        if pending and alignments:
            prev = alignments[-1]
            alignments[-1] = PhonemeAlignment(prev.phoneme, prev.num_samples + pending)
        return alignments or None

    def synthesize_wav(
            self,
            text: str,
            wav_file: wave.Wave_write,
            syn_config: Optional[SynthesisConfig] = None,
            set_wav_format: bool = True,
    ) -> None:
        """
        Synthesize and write WAV audio from text.

        :param text: Text to synthesize.
        :param wav_file: WAV file writer.
        :param syn_config: Synthesis configuration.
        :param set_wav_format: True if the WAV format should be set automatically.
        """

        # The voice's rate is set first, so text that gives no audio still leaves a valid WAV.
        write_wav(self.synthesize(text, syn_config=syn_config), wav_file,
                  sample_rate=self.config.sample_rate, set_wav_format=set_wav_format)

    def synthesize_ssml(
            self,
            ssml: str,
            syn_config: Optional[SynthesisConfig] = None,
            lang: Optional[str] = None,
            include_alignments: bool = False,
            on_mark: Optional[Callable[[str, float], None]] = None,
            seed: Optional[int] = None,
    ) -> Iterable[AudioChunk]:
        """Synthesize SSML, yielding silence, speech and sound chunks in order.

        The markup is read by :func:`phoonnx.ssml.parse_ssml`. ``<break>``, ``<p>`` and ``<s>``
        become silence; ``<prosody rate>`` and ``<emphasis>`` change the length scale (a rate of
        1.25 is ``length_scale / 1.25``); ``<prosody volume>`` scales the samples; ``<prosody
        pitch>`` is made by an engine with pitch controls of its own (``pitch_shift_params``), and
        else shifts the finished audio (:mod:`phoonnx.audio_fx`), as ``vocal-tract-length`` always
        does; ``<prosody range>`` reaches only an engine with a range control
        (``pitch_range_params``);
        ``<prosody duration>`` and ``amazon:max-duration`` stretch the speech to the time asked
        for (within a factor of two); ``<lang>`` (or ``xml:lang`` on any element) picks the language
        id of a multilingual voice when it has one, and the language of an engine that is told the
        tag itself (SuperTonic; a monolingual voice says the words in its own
        language); :meth:`scriptconv.frontend.TextFrontend.say_as` of :attr:`frontend` reads
        ``<say-as>``; ``<sub>`` and ``<phoneme>`` are read as words and phonemes;
        ``<amazon:effect>`` whispers, softens or compresses the audio; ``<amazon:auto-breaths>`` and
        ``<amazon:breath>`` add breaths; an expletive is a bleep. A text that is not SSML is
        synthesized as it is.

        Silences are exactly ``rate * ms // 1000`` samples, and speech that starts or stops on a
        loud sample is faded over a few milliseconds so a join does not click.

        :param ssml: A ``<speak>`` document, or plain text.
        :param syn_config: Base synthesis configuration; each piece adjusts a copy.
        :param lang: Language of the text when the markup does not say (default: the voice's).
        :param include_alignments: As in :meth:`synthesize`.
        :param on_mark: Called as ``on_mark(name, seconds)`` for each ``<mark name>``, just before
            the audio that follows it is yielded; `seconds` is the time since the start of the
            audio, counted from the samples yielded so far. A mark written before a ``<break>`` is
            called before its silence is yielded, one written after it after.
        :param seed: Seeds the noise of whispers and breaths: the same markup and seed give the
            same samples. Without it they use the defaults of :mod:`phoonnx.audio_fx`.
        """
        base = syn_config or SynthesisConfig()
        base_length = base.length_scale
        if base_length is None:
            # the voice's own length scale, in the order _request merges it
            base_length = (getattr(self.config, "engine_params", None) or {}).get("length_scale")
        if base_length is None:
            base_length = self.config.length_scale
        if base_length is None:
            base_length = self.adapter.default_params().get("length_scale", 1.0)
        voice_lang = lang or self.config.lang_code
        # Set by the first speech chunk: voice conversion and super-resolution can move it off
        # the model's own rate, and every chunk of one stream must share it.
        rate: Optional[int] = None
        waiting: deque = deque()  # sounds and marks ahead of the first speech, made once rate is known
        written = 0  # samples yielded so far
        noises = 0  # whispers and breaths made so far: each gets its own seed

        def elapsed() -> float:
            return written / rate if rate else 0.0

        def passed(chunk: AudioChunk) -> AudioChunk:
            nonlocal written, rate
            rate = chunk.sample_rate
            written += len(chunk.audio_float_array)
            return chunk

        def next_seed() -> Dict[str, int]:
            nonlocal noises
            noises += 1
            return {} if seed is None else {"seed": seed + noises}

        def silence(samples: int, at_rate: int) -> AudioChunk:
            return AudioChunk(sample_rate=at_rate, sample_width=2, sample_channels=1,
                              audio_float_array=np.zeros(samples, dtype=np.float32))

        def tone(kind: str, ms: int, volume: float, at_rate: int) -> AudioChunk:
            samples = audio_fx.beep(at_rate, ms) if kind == "beep" else audio_fx.breath(
                at_rate, ms, volume, **next_seed())
            exact = at_rate * ms // 1000  # not int(at_rate * ms / 1000): a float can land under it
            if len(samples) != exact:
                samples = np.pad(samples[:exact], (0, max(0, exact - len(samples))))
            return AudioChunk(sample_rate=at_rate, sample_width=2, sample_channels=1,
                              audio_float_array=samples)

        def lead_in(pause_ms: int, marks: Sequence[str], offsets: Sequence[int],
                    at_rate: int) -> Iterable[AudioChunk]:
            """A pause with its marks inside it, each at the millisecond it was written at."""
            offsets = list(offsets) + [pause_ms] * (len(marks) - len(offsets))
            done_ms = 0
            for name, at in zip(marks, offsets):
                at = min(max(at, done_ms), pause_ms)
                if at > done_ms:
                    yield passed(silence(at_rate * at // 1000 - at_rate * done_ms // 1000, at_rate))
                    done_ms = at
                if on_mark is not None:
                    on_mark(name, elapsed())
            if pause_ms > done_ms:
                yield passed(silence(at_rate * pause_ms // 1000 - at_rate * done_ms // 1000, at_rate))

        def sound(piece: Any, at_rate: int) -> Iterable[AudioChunk]:
            """A piece with no words: its pause and marks, then its sound if it has one."""
            yield from lead_in(piece.pause_ms, piece.marks, piece.mark_pauses, at_rate)
            if piece.sound is not None:
                kind, ms, volume = piece.sound
                yield passed(tone(kind, ms, volume, at_rate))

        for piece in parse_ssml(ssml, voice_lang, self.frontend, self.adapter.takes_symbols):
            if piece.sound is not None or not piece.text:
                if rate is None:
                    waiting.append(piece)
                else:
                    yield from sound(piece, rate)
                continue
            lead = [piece.pause_ms, list(piece.marks), list(piece.mark_pauses)]

            def before_audio(at_rate: int) -> Iterable[AudioChunk]:
                """The sounds still waiting, then the pause and the marks, ahead of a piece's audio."""
                while waiting:
                    yield from sound(waiting.popleft(), at_rate)
                pause_ms, marks, offsets = lead
                lead[:] = [0, [], []]
                yield from lead_in(pause_ms, marks, offsets, at_rate)

            settings: Dict[str, Any] = {}
            if piece.rate != 1.0:
                settings["length_scale"] = base_length / piece.rate
            if piece.volume != 1.0:
                settings["audio"] = replace(base.audio, volume=base.audio.volume * piece.volume)
            spoken_in = piece.voice_lang or piece.lang
            lang_id = self._lang_id_for(spoken_in)
            if lang_id is not None:
                settings["language"] = lang_id
            elif self.takes_language(spoken_in):
                settings["language"] = spoken_in
            native, shifted = self._native_pitch(piece, base)
            if native:
                settings["extra"] = {**base.extra, **native}
            piece_config = replace(base, **settings) if settings else base
            # The engine made the prosody pitch itself: only the vocal tract is left to the audio.
            effects_piece = replace(piece, pitch=getattr(piece, "tract", 0.0)) if shifted else piece
            if rate is not None:
                # The silence and the marks ahead of this speech do not wait for it to be made: a
                # listener should not wait for the model to hear the pause they asked for.
                yield from before_audio(rate)
            spoken = self.synthesize(piece.text, syn_config=piece_config,
                                     include_alignments=include_alignments)
            if piece.duration_ms is not None or piece.max_duration_ms is not None:
                spoken = self._fitted(list(spoken), piece)
            for index, chunk in enumerate(spoken):
                # Silence is made at the rate of the speech that follows it, which post-processing
                # (e.g. super-resolution) may have moved off the model's native rate.
                yield from before_audio(chunk.sample_rate)
                if piece.breaths and index and index % piece.breaths[2] == 0:
                    yield passed(tone("breath", piece.breaths[1], piece.breaths[0], chunk.sample_rate))
                chunk = self._with_effects(chunk, effects_piece,
                                           seed=None if seed is None else seed + 1000 + index)
                yield passed(self._declicked(chunk))
            # A piece that made no audio still owes its pause and its marks.
            yield from before_audio(rate or self.config.sample_rate)
        for piece in waiting:  # no words at all: the model's own rate
            yield from sound(piece, self.config.sample_rate)

    def _native_pitch(self, piece: Any, base: SynthesisConfig) -> Tuple[Dict[str, Any], bool]:
        """The engine's own pitch keys for a piece's ``<prosody pitch>`` and ``range``, and
        whether they make the pitch shift (else it is left to the finished audio).

        The keys build on the engine's settings as the call resolves them: its defaults, the
        voice's ``engine_params``, then the call's ``extra``. The range is applied first, then
        the shift. The part of the pitch that ``vocal-tract-length`` asks for is never sent to
        the engine: it moves the formants too, which only the audio shift does.
        """
        # pitch and tract are each within +-12 st, so their difference can reach 24
        semitones = max(-12.0, min(12.0, piece.pitch - getattr(piece, "tract", 0.0)))
        pitch_range = getattr(piece, "pitch_range", 1.0)
        if not semitones and pitch_range == 1.0:
            return {}, False
        params = {**self.adapter.default_params(),
                  **(getattr(self.config, "engine_params", None) or {}), **base.extra}
        session = getattr(self, "session", None)
        speaker = self._pitch_speaker(base)
        keys: Dict[str, Any] = {}
        if pitch_range != 1.0:
            ranged = self.adapter.pitch_range_params(pitch_range, params, session, speaker)
            if isinstance(ranged, Mapping):
                keys.update(ranged)
                params.update(ranged)
        shifted = False
        if semitones:
            shift = self.adapter.pitch_shift_params(semitones, params, session, speaker)
            if isinstance(shift, Mapping) and shift:
                keys.update(shift)
                shifted = True
        return keys, shifted

    def _pitch_speaker(self, syn_config: SynthesisConfig) -> Optional[int]:
        """The speaker id a call resolves to, as :meth:`_request` resolves it: the call's
        speaker (a name through the voice's ``speaker_id_map``), else the voice's default on a
        multi-speaker voice. ``None`` when there is none, which the engine reads as speaker 0:
        a single-speaker voice, a multi-speaker voice without a default, or a name only the
        engine knows."""
        config = self.config
        try:
            speaker = self._speaker_id(syn_config.speaker)
        except (ValueError, AttributeError):
            return None
        if speaker is None and (getattr(config, "num_speakers", 1) or 1) > 1:
            speaker = getattr(config, "default_speaker_id", None)
        try:
            return None if speaker is None else int(speaker)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _fitted(chunks: list, piece: Any) -> list:
        """`chunks` stretched or squeezed together to the piece's ``duration`` (or under its maximum).

        The factor is the same for every chunk and stays within 0.5 to 2: past that the speech is
        not speech any more, and the nearest the limit allows is as close as it gets. The phoneme
        samples and alignments of a chunk are stretched with its audio, so they still sum to it.
        """
        spoken = sum(len(c.audio_float_array) / c.sample_rate for c in chunks)
        if not chunks or spoken <= 0:
            return chunks
        target = (piece.duration_ms / 1000.0) if piece.duration_ms is not None else spoken
        if piece.max_duration_ms is not None:
            target = min(target, piece.max_duration_ms / 1000.0)
        factor = min(2.0, max(0.5, target / spoken))
        if abs(factor - 1.0) < 0.02:
            return chunks
        fitted = []
        for chunk in chunks:
            audio = audio_fx.time_stretch(np.asarray(chunk.audio_float_array, dtype=np.float32),
                                          factor, chunk.sample_rate)
            ratio = len(audio) / max(1, len(chunk.audio_float_array))
            samples, alignments = chunk.phoneme_id_samples, chunk.phoneme_alignments
            if samples is not None:
                samples = _stretched_counts(samples, ratio)
            if alignments is not None:
                counts = _stretched_counts([a.num_samples for a in alignments], ratio)
                alignments = [PhonemeAlignment(a.phoneme, int(n)) for a, n in zip(alignments, counts)]
            fitted.append(replace(chunk, audio_float_array=audio, phoneme_id_samples=samples,
                                  phoneme_alignments=alignments,
                                  _audio_int16_array=None, _audio_int16_bytes=None))
        return fitted

    @staticmethod
    def _declicked(chunk: AudioChunk) -> AudioChunk:
        """`chunk`, faded in or out where it would start or stop on a loud sample (else itself)."""
        samples = np.asarray(chunk.audio_float_array, dtype=np.float32)
        smoothed = audio_fx.declick(samples, chunk.sample_rate)
        if smoothed is samples or len(samples) < 2:
            return chunk
        return replace(chunk, audio_float_array=smoothed, _audio_int16_array=None, _audio_int16_bytes=None)

    @staticmethod
    def _with_effects(chunk: AudioChunk, piece: Any, seed: Optional[int] = None) -> AudioChunk:
        """`chunk` with the piece's pitch shift and effects applied (the chunk itself if none)."""
        if not piece.pitch and not piece.effects:
            return chunk
        samples = np.asarray(chunk.audio_float_array, dtype=np.float32)
        if piece.pitch:
            samples = audio_fx.pitch_shift(samples, chunk.sample_rate, piece.pitch)
        if "whisper" in piece.effects:
            samples = audio_fx.whisper(samples, chunk.sample_rate, **({} if seed is None else {"seed": seed}))
        if "soft" in piece.effects:
            samples = audio_fx.soft(samples, chunk.sample_rate)
        if "drc" in piece.effects:
            samples = audio_fx.compress(samples, chunk.sample_rate)
        return replace(chunk, audio_float_array=samples,
                       _audio_int16_array=None, _audio_int16_bytes=None)

    def _speaker_id(self, speaker: Optional[Union[int, str]]) -> Optional[Union[int, str]]:
        """The speaker that the engine gets: the id of a name in the voice's
        ``speaker_id_map``, or else *speaker* as it is. An engine with its own
        speaker names, such as Magpie, reads a name that the map lacks.

        Raises:
            ValueError: If *speaker* is a name that the map lacks and the engine has no
                speaker names of its own.
        """
        if isinstance(speaker, str) and not isinstance(speaker, _LegacyId):
            if speaker in self.config.speaker_id_map:
                return int(self.config.speaker_id_map[speaker])
            # A digit string is an id, which the engine reads with int().
            if not self.adapter.SPEAKER_NAMES and not speaker.isdigit():
                raise ValueError(f"voice {self._voice_name} has no speaker {speaker!r}. "
                                 f"Its speakers are: {sorted(self.config.speaker_id_map)}")
        return speaker

    def _language_id(self, language: Optional[Union[int, str]]) -> Optional[Union[int, str]]:
        """The language that the engine gets: the id of a code in the voice's
        ``lang_id_map``, or else *language* as it is. A code on a voice with no map
        gives ``None``, because the code then only sets the text-stage language.

        Raises:
            ValueError: If *language* is a code that the map of a multilingual voice lacks.
        """
        if not isinstance(language, str) or isinstance(language, _LegacyId):
            return language
        language_id = self._lang_id_for(language)
        if language_id is None and self.config.lang_id_map:
            raise ValueError(f"voice {self._voice_name} has no language {language!r}. "
                             f"Its languages are: {sorted(self.config.lang_id_map)}")
        return language_id

    def takes_language(self, code: Optional[str]) -> bool:
        """Whether this voice speaks the language *code* when a call asks for it as
        ``SynthesisConfig.language``, through an engine that is told the tag itself
        (SuperTonic). A voice with a ``lang_id_map`` takes the codes of its map instead."""
        if not code or getattr(self.config, "lang_id_map", None):
            return False
        asks = getattr(self.adapter, "takes_language", None)
        return callable(asks) and asks(code) is True

    def _lang_id_for(self, tag: Optional[str]) -> Optional[int]:
        """The language id of a multilingual voice for *tag*, or None (monolingual, or unknown)."""
        mapping = self.config.lang_id_map
        if not tag or not mapping:
            return None
        wanted = tag.replace("_", "-").lower()
        for key in (wanted, wanted.split("-")[0]):
            for name, ident in mapping.items():
                if name.replace("_", "-").lower() == key:
                    return int(ident)
        return None

    def synthesize_ssml_wav(
            self,
            ssml: str,
            wav_file: wave.Wave_write,
            syn_config: Optional[SynthesisConfig] = None,
            set_wav_format: bool = True,
            lang: Optional[str] = None,
            on_mark: Optional[Callable[[str, float], None]] = None,
            seed: Optional[int] = None,
    ) -> None:
        """Synthesize SSML and write it as WAV audio (see :meth:`synthesize_ssml`)."""
        write_wav(self.synthesize_ssml(ssml, syn_config=syn_config, lang=lang, on_mark=on_mark,
                                       seed=seed),
                  wav_file, sample_rate=self.config.sample_rate, set_wav_format=set_wav_format)

    def _prompt_token_ids(self, text: str, lang: Optional[str] = None) -> list[int]:
        """Tokenize a cloning reference transcription into prompt token ids, reusing
        the voice's own phonemizer + tokenizer (for in-context engines like ZipVoice).

        ``lang`` phonemizes the transcription in the reference's own language (which
        may differ from the target's); defaults to the voice's ``lang_code``.
        """
        ids: list[int] = []
        for phonemes in self.phonemize(text, lang):
            if phonemes:
                ids.extend(self.tokenizer.tokenize(phonemes))
        return ids

    def phoneme_ids_to_audio(
            self, phoneme_ids: list[int], syn_config: Optional[SynthesisConfig] = None,
            include_alignments: bool = False,
            streams: Optional[Dict[str, List[int]]] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Deprecated: use :meth:`render`, which always returns an :class:`AudioChunk`.

        This returns the engine's audio before :meth:`render` levels, scales and
        clips it: an array, or ``(audio, phoneme_id_samples)`` when
        ``include_alignments`` is True.
        """
        warnings.warn("TTSVoice.phoneme_ids_to_audio is deprecated and will be removed in phoonnx "
                      f"{_removal_version()}. Use TTSVoice.render.", DeprecationWarning, stacklevel=2)
        audio, phoneme_id_samples = self._infer(phoneme_ids, syn_config or SynthesisConfig(),
                                                include_alignments, streams)
        return (audio, phoneme_id_samples) if include_alignments else audio

    def _request(
            self, phoneme_ids: Sequence[int], syn_config: SynthesisConfig,
            streams: Optional[Mapping[str, Sequence[int]]] = None,
            text: Optional[str] = None,
    ) -> AdapterSynthesisRequest:
        """The engine request for one id sequence, with the parameters merged from the
        adapter, the voice config and *syn_config*."""
        phoneme_ids_array = np.expand_dims(
            np.array(phoneme_ids, dtype=np.int64), 0
        )
        phoneme_ids_lengths = np.array(
            [phoneme_ids_array.shape[1]], dtype=np.int64
        )
        # Merge defaults from adapter → VoiceConfig → SynthesisConfig
        params = dict(self.adapter.default_params())
        # Override with voice-config level defaults
        params.update({
            k: v for k, v in {
                "noise_scale": self.config.noise_scale,
                "length_scale": self.config.length_scale,
                "noise_w_scale": self.config.noise_w_scale,
            }.items() if v is not None
        })
        # Override with any engine-specific extras from voice config
        params.update(self.config.engine_params)
        # The voice's own length scale, before the call changes it
        voice_length = params.get("length_scale")
        # Override with per-call SynthesisConfig
        if syn_config.noise_scale is not None:
            params["noise_scale"] = syn_config.noise_scale
        if syn_config.length_scale is not None:
            params["length_scale"] = syn_config.length_scale
        if syn_config.noise_w_scale is not None:
            params["noise_w_scale"] = syn_config.noise_w_scale
        # Engine-specific extras from SynthesisConfig
        params.update(syn_config.extra)
        # An engine with a speed key of its own gets it from the length scale,
        # unless the voice config or the call names that key itself. A call that
        # changes the length scale (an SSML rate, or a length_scale of its own)
        # still scales a speed the voice config names: the configured speed is the
        # voice's normal pace, and the call's change is relative to the voice's own
        # length scale. A speed the call names wins outright.
        if "length_scale" in params:
            named, extra = self.config.engine_params, syn_config.extra
            own = self.adapter.length_scale_params(params["length_scale"])
            params.update({key: value for key, value in own.items()
                           if key not in named and key not in extra})
            base = 1.0 if voice_length is None else voice_length
            if params["length_scale"] != base:
                normal = self.adapter.length_scale_params(base)
                params.update({key: float(named[key]) * value / normal[key]
                               for key, value in own.items()
                               if key in named and key not in extra and normal.get(key)
                               and isinstance(named[key], (int, float))
                               and not isinstance(named[key], bool)})
        # Voice cloning: hand the cloning adapter the reference clip and — for
        # in-context engines (ZipVoice) — the prompt tokens of its transcription.
        reference = syn_config.reference or SpeakerReference()
        if reference.audio is not None:
            params["reference_audio"] = load_reference_audio(reference.audio)
        if reference.text:
            reference_text, reference_lang = reference.text, reference.lang
            params["prompt_tokens"] = self._prompt_token_ids(reference_text, reference_lang)
            # The text, for text-token adapters (OmniVoice, Spark-TTS) that tokenize the
            # reference transcription themselves. It is normalised as the target text is,
            # so the model reads both in one form. It goes on the per-call request, not on
            # adapter state, so concurrent calls on one adapter never see each other's reference.
            params["speaker_reference_text"] = self.frontend.prepare(
                reference_text, Stages(diacritize=False, lang=reference_lang))
        if syn_config.exaggeration is not None:
            params["exaggeration"] = syn_config.exaggeration
        if syn_config.temperature is not None:
            params["temperature"] = syn_config.temperature
        if syn_config.top_p is not None:
            params["top_p"] = syn_config.top_p

        side_streams = {}
        for name, values in (streams or {}).items():
            if len(values) != len(phoneme_ids):
                raise ValueError(f"side stream {name!r} has {len(values)} values for {len(phoneme_ids)} phoneme ids")
            side_streams[name] = np.array([values], dtype=np.int64)

        speaker_id = self._speaker_id(syn_config.speaker)
        if speaker_id is None and self.config.num_speakers > 1:
            speaker_id = self.config.default_speaker_id
        language_id = self._language_id(syn_config.language)

        return AdapterSynthesisRequest(
            phoneme_ids=phoneme_ids_array,
            phoneme_lengths=phoneme_ids_lengths,
            speaker_id=speaker_id,
            language_id=language_id,
            params=params,
            text=text,
            **side_streams,
        )

    def _infer(
            self, phoneme_ids: Sequence[int], syn_config: SynthesisConfig,
            include_alignments: bool = False,
            streams: Optional[Mapping[str, Sequence[int]]] = None,
            text: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run the engine on one id sequence: the engine step of :meth:`render`.

        Delegates all ONNX I/O to ``self.adapter`` so the same code
        path works for VITS or any future engine.

        :param phoneme_ids: List of phoneme ids.
        :param syn_config: Synthesis configuration.
        :param include_alignments: When True, also return per-phoneme sample counts.
        :param streams: Per-id side streams, keyed by ``AdapterSynthesisRequest``
            field, e.g. ``{"language_ids": [...]}``; each holds one value per id.
        :param text: The chunk's text, as ``Utterance.text`` gives it.
        :return: ``(audio, phoneme_id_samples)``. The audio is a float numpy
            array, not normalized. ``phoneme_id_samples`` is the per-id audio-sample
            count (durations scaled from model frames via ``VoiceConfig.hop_length``),
            or ``None`` when alignments are not requested or the model doesn't
            expose a duration output. The audio is byte-for-byte identical either
            way — the same inference runs.
        """
        request = self._request(phoneme_ids, syn_config, streams, text)

        # Once a prior call already retrofitted a working alignment session
        # (see ``_ensure_alignment_session``), route straight to it instead
        # of running inference on the original session first every time.
        session_to_use = self.session
        if (
            include_alignments
            and self._alignment_session is not UNSET
            and self._alignment_session is not None
        ):
            session_to_use = self._alignment_session

        # Delegate to the adapter (single-graph engines use the default
        # build_feed_dict → run → parse_outputs; iterative engines override).
        result = self.adapter.synthesize(request, session_to_use)

        if not include_alignments:
            return result.audio, None

        phoneme_id_samples = result.extras.get("phoneme_id_samples")
        if phoneme_id_samples is None and session_to_use is self.session:
            # This model's session exposes no duration output. Try (at most
            # once per voice) retrofitting one via load-time graph surgery —
            # see ``_ensure_alignment_session`` — and re-run this same
            # request on the patched session if that worked. Every failure
            # mode there degrades to ``None`` here exactly like a model that
            # genuinely has no duration predictor; the outcome (including a
            # negative one) is cached so this only ever runs once per voice.
            alignment_session = self._ensure_alignment_session()
            if alignment_session is not None:
                result = self.adapter.synthesize(request, alignment_session)
                phoneme_id_samples = result.extras.get("phoneme_id_samples")

        if phoneme_id_samples is None:
            # Alignment is not available from this voice model.
            return result.audio, None

        # Model durations are in native frames — convert to audio samples.
        # ``np.atleast_1d`` guards against adapters that squeeze a
        # single-token (1,1,1) duration tensor down to a 0-d array, which
        # would otherwise blow up ``len()`` in ``_reconstruct_alignments``.
        phoneme_id_samples = (
            np.atleast_1d(np.asarray(phoneme_id_samples)) * self.config.hop_length
        ).astype(np.int64)

        return result.audio, phoneme_id_samples
