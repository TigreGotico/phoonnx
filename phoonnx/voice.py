"""
TTSVoice — architecture-agnostic TTS voice.

This module is the main user-facing synthesis interface.  It delegates
all ONNX-specific I/O to an engine adapter (``phoonnx.engines``),
which means the same TTSVoice class works for VITS or
any future architecture without code changes here.
"""
import hashlib
import json
import os.path
import re
import shutil
import tempfile
import wave
import weakref
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import onnxruntime
from langcodes import closest_match
from quebra_frases import sentence_tokenize

from phoonnx.alignment import UNSET, alignment_model_path, ensure_alignment_session
from phoonnx.config import PhonemeType, VoiceConfig, SynthesisConfig, Alphabet, get_phonemizer, get_conversion, \
    check_lang_supported
from scriptconv.graph import DEFAULT_GRAPH
from phoonnx.engines import detect_engine, get_adapter
from phoonnx.engines.base import (
    AdapterSynthesisRequest,
    AdapterSynthesisResult,
    BaseOnnxAdapter,
)
from scriptconv.phonemizers.base import BasePhonemizer
from phoonnx.providers import ProviderSpec, make_session, resolve_providers
from phoonnx.reference_audio import load_reference_audio
from scriptconv.phonemizers.base import PhonemizedChunks
from phoonnx.tokenizer import TTSTokenizer
from phoonnx.util import LOG


_PHONEME_BLOCK_PATTERN = re.compile(r"(\[\[.*?\]\])")

# Alias kept for callers that imported the sentinel from here before it moved.
_UNSET = UNSET


def _phonemic_chunks(text: str, alphabet: Optional[Alphabet]) -> PhonemizedChunks:
    """Split an already-phonemic string into per-sentence symbol lists.

    Space-separated notations (ARPA) tokenize on whitespace; char-based
    alphabets tokenize per character.
    """
    chunks = BasePhonemizer.chunk_text(text)
    if alphabet == Alphabet.ARPA:
        return [chunk.split() for chunk, _, _ in chunks if chunk.strip()]
    return [[c for c in chunk] for chunk, _, _ in chunks if chunk]


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


@dataclass
class PhoneticSpellings:
    replacements: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def from_lang(lang: str, locale_path: str = f"{os.path.dirname(__file__)}/locale"):
        langs = os.listdir(locale_path)
        lang2, distance = closest_match(lang, langs)
        if distance <= 10:
            spellings_file = f"{locale_path}/{lang2}/phonetic_spellings.txt"
            return PhoneticSpellings.from_path(spellings_file)
        raise FileNotFoundError(f"Spellings file for '{lang}' not found")

    @staticmethod
    def from_path(spellings_file: str):
        replacements = {}
        with open(spellings_file) as f:
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


@dataclass
class PhonemeAlignment:
    """A single phoneme paired with the number of audio samples it spans."""

    phoneme: str
    num_samples: int


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


@dataclass
class TTSVoice:
    session: onnxruntime.InferenceSession
    config: VoiceConfig
    phonetic_spellings: Optional[PhoneticSpellings] = None
    phonemizer: Optional[BasePhonemizer] = None
    adapter: Optional[BaseOnnxAdapter] = None

    _sr_engine: Any = field(default=None, init=False, repr=False)
    _sr_loaded: bool = field(default=False, init=False, repr=False)
    # Post-synthesis voice-conversion state (``SynthesisConfig.vc_reference``).
    # ``_vc_key`` is the (engine, reference content) the cached ``_VoiceConversion``
    # was built for, so a later call naming a different engine or target speaker —
    # or the same path holding a different recording — rebuilds instead of silently
    # reusing the wrong voice.
    _vc: Optional[_VoiceConversion] = field(default=None, init=False, repr=False)
    _vc_key: Any = field(default=None, init=False, repr=False)
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

        # Phonemizer
        if self.phonemizer is None:
            # Fail fast (typed) if no scriptconv backend serves this voice's
            # language, instead of the voice loading fine and crashing mid
            # synthesis with an opaque ValueError once text is sent. Name the
            # voice by its model filename (e.g. "tdt-TL_joao"), not the full
            # HF snapshot path -- ``model_path`` is a cache path, not a
            # human-facing identifier.
            voice_name = Path(self.model_path).stem if self.model_path else "<voice>"
            check_lang_supported(
                voice_name,
                self.config.lang_code,
                self.config.phoneme_type,
            )
            self.phonemizer = get_phonemizer(
                self.config.phoneme_type,
                self.config.alphabet,
                self.config.phonemizer_model,
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
    ) -> "TTSVoice":
        """
        Load a TTS voice ONNX model and its configuration into a TTSVoice instance.
        
        Parameters:
            model_path (str | Path): Path to the ONNX voice model file.
            config_path (str | Path, optional): Path to the JSON voice configuration. If omitted, defaults to model_path + ".json".
            phonemes_txt (str, optional): Optional phonemes definition or file content to override the config's phoneme list.
            phoneme_map (str, optional): Optional phoneme mapping specification or file path used to map phonemes (may be None).
            lang_code (str, optional): Language code to override or set in the loaded voice configuration.
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

        Returns:
            TTSVoice: A TTSVoice instance prepared with the loaded ONNX session and merged configuration.
        """
        if config_path is None:
            config_path = f"{model_path}.json"

        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as config_file:
                config_dict = json.load(config_file)
        else:
            config_dict = {"phoneme_type": "unicode", "alphabet": "unicode"}

        vocab_dict = {}
        tokenizer_dict = {}
        if vocab_path and os.path.isfile(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as vocab_file:
                vocab_dict = json.load(vocab_file)
            if tokenizer_config_path and os.path.isfile(tokenizer_config_path):
                with open(tokenizer_config_path, "r", encoding="utf-8") as tokenizer_file:
                    tokenizer_dict = json.load(tokenizer_file)
        resolved_providers = resolve_providers(providers, use_cuda=use_cuda)

        # Auto-split: a voice may request streaming on a monolithic single-graph
        # VITS model. When it declares "streaming": true but ships no decoder
        # graph, split the model into an encoder/decoder pair on the fly (cached
        # next to the model) so it can stream without a re-export. If the model
        # is not a splittable VITS, fall back to loading it as a normal voice.
        _cfg_ep = config_dict.get("engine_params") or {}
        _has_decoder = _cfg_ep.get("decoder_path") or (engine_params or {}).get("decoder_path")
        if config_dict.get("streaming") and not _has_decoder:
            try:
                from phoonnx.engines.vits_split import ensure_split_vits
                enc_path, dec_path = ensure_split_vits(str(model_path))
                model_path = enc_path
                config_dict["engine_params"] = {**_cfg_ep, "decoder_path": dec_path}
            except Exception as e:
                LOG.warning(f"Streaming requested but auto-split of "
                            f"'{model_path}' failed ({e}); loading as a normal "
                            f"single-graph voice.")

        session = make_session(model_path, providers=resolved_providers)

        # Auto-detect engine adapter from config + session
        adapter = detect_engine(config=config_dict, session=session)

        # Engine-specific params (e.g. Matcha's vocoder_path) come either from
        # the caller (the voice manager injects locally-downloaded paths) or
        # from the model's own config JSON.
        engine_params = engine_params or config_dict.get("engine_params") or {}
        # Engines load auxiliary graphs of their own (vocoders, speaker encoders,
        # text encoders); they run on the same providers as the voice itself.
        engine_params = dict(engine_params)
        engine_params.setdefault("providers", resolved_providers)

        voice_config = VoiceConfig.from_dict(
            config_dict,
            vocab=vocab_dict,
            tokenizer_config=tokenizer_dict,
            alphabet=alphabet_str,
            tokens_txt=phonemes_txt,
            lang_code=lang_code,
            phoneme_type=phoneme_type_str,
            engine_params=engine_params,
        )

        voice = TTSVoice(
            config=voice_config,
            session=session,
            adapter=adapter,
            model_path=str(model_path),
        )
        if warmup:
            voice.warmup()
        return voice

    def phonemize(self, text: str, lang: Optional[str] = None) -> PhonemizedChunks:
        """
        Text to phonemes grouped by sentence.

        :param text: Text to phonemize.
        :param lang: Language code to phonemize in; defaults to the voice's own
            ``lang_code``. Used to phonemize a cloning reference transcription in
            *its* language (which may differ from the target's).
        :return: List of phonemes for each sentence.
        """
        lang = lang or self.config.lang_code
        phonemes: list[list[str]] = []

        text_parts = _PHONEME_BLOCK_PATTERN.split(text)

        for i, text_part in enumerate(text_parts):
            if text_part.startswith("[["):
                # Phonemes
                if not phonemes:
                    # Start new sentence
                    phonemes.append([])
                # the preceding text chunk may have come back as an immutable
                # sequence; ensure the current sentence is a mutable list before
                # appending the inline phonemes to it
                if not isinstance(phonemes[-1], list):
                    phonemes[-1] = list(phonemes[-1])

                if (i > 0) and (text_parts[i - 1].endswith(" ")):
                    phonemes[-1].append(" ")

                phonemes[-1].extend(list(text_part[2:-2].strip()))  # Ensure characters are split

                if (i < (len(text_parts)) - 1) and (text_parts[i + 1].startswith(" ")):
                    phonemes[-1].append(" ")

                continue

            if not text_part:
                continue
            phonemes.extend(list(chunk) for chunk in self.phonemizer.phonemize(text_part, lang))

        if phonemes and (not phonemes[-1]):
            # Remove empty phonemes
            phonemes.pop()

        return phonemes

    def phonemes_to_ids(self, phonemes: list[str]) -> list[int]:
        """
        Convert a sequence of phoneme tokens (or characters for grapheme models) into token IDs.
        
        Parameters:
            phonemes (list[str]): Sequence of phoneme strings or individual characters to be tokenized.
        
        Returns:
            list[int]: Token IDs corresponding to each input phoneme or character.
        """
        token_ids =  self.tokenizer.tokenize(phonemes)
        return token_ids

    def _text_parts(self, text: str) -> Iterable[str]:
        """Split into sentences — always, as a time-to-first-audio measure (keep
        sentence N+1 off the critical path of sentence N's audio)."""
        for sentence in sentence_tokenize(text):
            if sentence.strip():
                yield sentence

    def _iter_phonemes(
            self, text: str, src_alphabet: Alphabet, tgt_alphabet: Alphabet,
            route, prepare_text,
    ) -> Iterable[List[str]]:
        """Yield per-sentence phoneme lists for *text*, by whichever conversion
        its alphabet calls for. One job: getting phonemes."""
        # Already-phonemic input: transcode to the model's alphabet through
        # scriptconv's graph (a no-op when src == tgt), then split.
        if src_alphabet != Alphabet.GRAPHEMES:
            try:
                text = DEFAULT_GRAPH.convert(text, src_alphabet.value, tgt_alphabet.value,
                                             lang=self.config.lang_code)
            except ValueError:
                LOG.debug("no conversion path %s -> %s; passing through", src_alphabet, tgt_alphabet)
            yield from _phonemic_chunks(text, tgt_alphabet)
            return

        # Inline [[phoneme]] overrides bypass the phonemizer and merge across the
        # whole text, so they stay whole-text rather than per sentence.
        if _PHONEME_BLOCK_PATTERN.search(text):
            yield from self.phonemize(prepare_text(text))
            return

        # Graphemes: the voice's conversion route, one sentence at a time.
        for sentence in self._text_parts(text):
            yield from route.convert(sentence, "text", tgt_alphabet.value)

    def _iter_synthesis_ids(
            self, text, syn_config,
            src_alphabet: Alphabet, tgt_alphabet: Alphabet,
    ) -> Iterable[tuple]:
        """Lazily yield ``(phonemes, phoneme_ids)`` per sentence — ``phonemes`` is
        ``None`` for text-token models, whose adapter owns text→ids outright."""
        route, prepare_text = get_conversion(
            self.phonemizer, self.config, syn_config, tgt_alphabet)

        if src_alphabet == tgt_alphabet == Alphabet.GRAPHEMES:
            for ids in self.adapter.encode_text(prepare_text(text), self, syn_config):
                yield None, ids
            return

        for phonemes in self._iter_phonemes(text, src_alphabet, tgt_alphabet,
                                            route, prepare_text):
            if phonemes:
                yield phonemes, self.phonemes_to_ids(phonemes)

    def _load_super_resolution(self, syn_config: SynthesisConfig):
        """
        Lazily load the audio super-resolution engine when enabled in ``syn_config``.

        Reads ``syn_config.super_resolution`` (a bool) and
        ``syn_config.super_resolution_model`` (the ``audiosronnx`` engine name).
        Disabled — the default — is a strict no-op: ``audiosronnx`` is never imported
        and synthesis is untouched. When explicitly enabled, ``audiosronnx`` is
        imported lazily and the chosen engine is loaded once and cached on the
        instance; a missing install raises a clear ImportError naming the extra rather
        than silently downgrading a feature the caller asked for.

        Returns:
            The loaded SR engine, or None when disabled.
        """
        if not syn_config.super_resolution:
            return None
        if self._sr_loaded:
            return self._sr_engine

        engine = syn_config.super_resolution_model or "novasr"
        try:
            from audiosronnx import load_sr
        except ImportError as e:
            raise ImportError(
                "super_resolution is enabled but 'audiosronnx' is not installed; "
                "install it with `pip install phoonnx[audiosr]` (or set "
                "super_resolution=False)") from e

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
        Lazily build the ``voiceclonnx`` cloner for ``syn_config.vc_reference``.

        Disabled — ``vc_reference is None``, the default — is a strict no-op:
        ``voiceclonnx`` is never imported and synthesis is untouched. When enabled,
        the cloner and its materialized reference WAV are cached as one
        :class:`_VoiceConversion`, so repeated calls naming the same target speaker
        do not reload the ONNX sessions. The object is handed to the caller, which
        holds it for the whole of its synthesis; a later call for a different
        speaker replaces the cache slot without disturbing it.

        The reference is normalised by
        :func:`phoonnx.reference_audio.load_reference_audio`, the same guarded
        loader ``speaker_reference`` uses, so URLs are fetched under its address and
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
        ref = syn_config.vc_reference
        if ref is None:
            return None

        engine = syn_config.vc_engine or "openvoice"
        if isinstance(ref, (str, Path)) and not str(ref).lower().startswith(
                ("http://", "https://")) and not os.path.isfile(str(ref)):
            raise FileNotFoundError(
                f"vc_reference points at no readable file: {str(ref)!r}")

        audio, sr = load_reference_audio(ref)
        key = (engine, hashlib.sha1(audio.tobytes()).hexdigest(), int(sr))
        if self._vc is not None and self._vc_key == key:
            return self._vc

        try:
            from voiceclonnx import VoiceCloner
        except ImportError as e:
            raise ImportError(
                "vc_reference is set but 'voiceclonnx' is not installed; "
                "install it with `pip install phoonnx[vc]` (or set "
                "vc_reference=None)") from e

        cloner = VoiceCloner(engine=engine)
        fd, ref_path = tempfile.mkstemp(prefix="phoonnx_vc_ref_", suffix=".wav")
        os.close(fd)
        try:
            _write_wav(ref_path, audio, sr)
        except BaseException:
            _unlink(ref_path)
            raise

        self._vc = _VoiceConversion(cloner, ref_path)
        self._vc_key = key
        LOG.info(f"Voice conversion enabled (engine={engine}); output will be "
                 f"converted to the reference speaker at {cloner.sample_rate} Hz")
        return self._vc

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

        Generates sentence-level audio by phonemizing the input text and synthesizing each sentence into a float32 PCM audio array in the range [-1.0, 1.0]. If enabled in the synthesis configuration, user-provided phonetic spellings and diacritic augmentation are applied before phonemization. Output audio may be normalized and volume-scaled according to the configuration; samples are clipped to [-1.0, 1.0].

        Parameters:
            text (str): The input text to synthesize.
            syn_config (Optional[SynthesisConfig]): Optional synthesis options (e.g., enable_phonetic_spellings, add_diacritics, normalize_audio, volume). If omitted, a default SynthesisConfig is used.
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

        LOG.debug("text=%s", text)

        # optional post-synthesis super-resolution; disabled (the default) means the
        # loop below is bit-for-bit the native path and audiosronnx is not imported
        sr_engine = self._load_super_resolution(syn_config)

        # optional post-synthesis voice conversion; disabled (the default) means the
        # loop below is bit-for-bit the native path and voiceclonnx is not imported.
        # VC runs before SR so the upscaler sees the final timbre and the 48 kHz rate
        # is what actually reaches the caller.
        vc = self._load_voice_conversion(syn_config)

        # Text preprocessing — engine-agnostic, applied to the whole text before
        # any conversion: user pronunciation overrides.
        if self.phonetic_spellings and syn_config.enable_phonetic_spellings:
            text = self.phonetic_spellings.apply(text)

        # Alphabet dispatch (unified model + language-aware phonemizers):
        #   src == tgt          -> grapheme/text-token models use the adapter;
        #                          already-phonemic input in the model's own
        #                          alphabet passes straight through (no re-phonemize).
        #   src == GRAPHEMES    -> phonemize with the model's phonemizer, keeping the
        #                          per-phoneme language IDs that language-aware
        #                          phonemizers (e.g. Shami) provide.
        #   otherwise           -> already-phonemic input in a different alphabet is
        #                          transcoded to the model's alphabet through the
        #                          conversion graph (scriptconv notation edges).
        src_alphabet = (syn_config.alphabet
                        if syn_config.alphabet is not None
                        else Alphabet.GRAPHEMES)
        tgt_alphabet = self.config.alphabet

        # Per-sentence phonemize -> tokenize is done lazily so the first sentence
        # reaches session.run before later sentences are converted,
        # cutting time-to-first-audio on multi-sentence input.
        id_stream = self._iter_synthesis_ids(
            text, syn_config, src_alphabet, tgt_alphabet,
        )

        # Special-token ids for alignment reconstruction — read once, only when
        # alignments are requested (keeps the default path free of this work).
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

        for phonemes, phoneme_ids in id_stream:
            if not phoneme_ids:
                continue

            phoneme_id_samples: Optional[np.ndarray] = None
            audio_result = self.phoneme_ids_to_audio(
                phoneme_ids, syn_config, include_alignments=include_alignments,
            )
            if isinstance(audio_result, tuple):
                audio, phoneme_id_samples = audio_result
            else:
                audio = audio_result

            if syn_config.normalize_audio:
                max_val = np.max(np.abs(audio))
                if max_val < 1e-8:
                    # Prevent division by zero
                    audio = np.zeros_like(audio)
                else:
                    audio = audio / max_val

            if syn_config.volume != 1.0:
                audio = audio * syn_config.volume

            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

            phoneme_alignments = self._reconstruct_alignments(
                phoneme_ids, phoneme_id_samples,
                _idx2char, _blank_id, _bos_id, _eos_id,
            ) if phoneme_id_samples is not None else None

            # VC and SR each change the rate of the samples, so the chunk must
            # report the rate that came back with them, not the voice's native one
            audio, chunk_sr = self._maybe_convert(
                audio, vc, self.config.sample_rate)
            audio, chunk_sr = self._maybe_upscale(audio, sr_engine, chunk_sr)

            yield AudioChunk(
                sample_rate=chunk_sr,
                sample_width=2,
                sample_channels=1,
                audio_float_array=audio,
                phonemes=list(phonemes) if phonemes else [],
                phoneme_ids=list(phoneme_ids),
                phoneme_id_samples=phoneme_id_samples,
                phoneme_alignments=phoneme_alignments,
            )

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
            except Exception:
                providers = None
            result = ensure_alignment_session(self.model_path, providers)

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

        # 16-bit samples for silence
        sentence_silence = 0.0  # Seconds of silence after each sentence
        silence_int16_bytes = bytes(
            int(self.config.sample_rate * sentence_silence * 2)
        )
        if set_wav_format:
            # Set the format up front, from the voice: text that produces no
            # audio at all (an empty or punctuation-only utterance) must still
            # leave a valid, empty WAV behind. Waiting for the first chunk left
            # the writer unconfigured, so closing it raised
            # "wave.Error: # channels not specified" instead.
            wav_file.setframerate(self.config.sample_rate)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)

        first_chunk = True
        for audio_chunk in self.synthesize(text, syn_config=syn_config):
            if not first_chunk:
                wav_file.writeframes(silence_int16_bytes)

            if first_chunk:
                if set_wav_format:
                    # the chunk is authoritative once we have one
                    wav_file.setframerate(audio_chunk.sample_rate)
                    wav_file.setsampwidth(audio_chunk.sample_width)
                    wav_file.setnchannels(audio_chunk.sample_channels)

                first_chunk = False

            wav_file.writeframes(audio_chunk.audio_int16_bytes)


    def _prompt_token_ids(self, text: str, lang: Optional[str] = None) -> list[int]:
        """Tokenize a cloning reference transcription into prompt token ids, reusing
        the voice's own phonemizer + tokenizer (for in-context engines like ZipVoice).

        ``lang`` phonemizes the transcription in the reference's own language (which
        may differ from the target's); defaults to the voice's ``lang_code``.
        """
        ids: list[int] = []
        for phonemes in self.phonemize(text, lang):
            if phonemes:
                ids.extend(self.phonemes_to_ids(phonemes))
        return ids

    def phoneme_ids_to_audio(
            self, phoneme_ids: list[int], syn_config: Optional[SynthesisConfig] = None,
            include_alignments: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Synthesize raw audio from phoneme ids.

        Delegates all ONNX I/O to ``self.adapter`` so the same code
        path works for VITS or any future engine.

        :param phoneme_ids: List of phoneme ids.
        :param syn_config: Synthesis configuration.
        :param include_alignments: When True, also return per-phoneme sample counts.
        :return: Audio float numpy array (unnormalized, in range [-1, 1]).

        When ``include_alignments`` is True the return value is instead a
        ``(audio, phoneme_id_samples)`` tuple, where ``phoneme_id_samples`` is
        the per-id audio-sample count (durations scaled from model frames via
        ``VoiceConfig.hop_length``), or ``None`` when this model doesn't expose
        a duration output. The audio itself is byte-for-byte identical to the
        no-alignment call — the same inference runs either way.
        """
        syn_config = syn_config or SynthesisConfig()

        # Build the architecture-agnostic request
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
        if hasattr(self.config, 'engine_params'):
            params.update(self.config.engine_params)
        # Override with per-call SynthesisConfig
        if syn_config.noise_scale is not None:
            params["noise_scale"] = syn_config.noise_scale
        if syn_config.length_scale is not None:
            params["length_scale"] = syn_config.length_scale
        if syn_config.noise_w_scale is not None:
            params["noise_w_scale"] = syn_config.noise_w_scale
        # Engine-specific extras from SynthesisConfig
        params.update(syn_config.extra_params)
        # Voice cloning: hand the cloning adapter the reference clip and — for
        # in-context engines (ZipVoice) — the prompt tokens of its transcription.
        if syn_config.speaker_reference is not None:
            params["reference_audio"] = load_reference_audio(syn_config.speaker_reference)
        if syn_config.speaker_reference_text:
            params["prompt_tokens"] = self._prompt_token_ids(
                syn_config.speaker_reference_text, syn_config.speaker_reference_lang)
            # Raw string, for text-token adapters (OmniVoice, Spark-TTS) that tokenize
            # the reference transcription themselves rather than through the shared
            # phonemizer. Carried on the per-call request rather than adapter state so
            # concurrent calls sharing one adapter instance never see each other's
            # reference transcription.
            params["speaker_reference_text"] = syn_config.speaker_reference_text
        if syn_config.exaggeration is not None:
            params["exaggeration"] = syn_config.exaggeration
        if syn_config.temperature is not None:
            params["temperature"] = syn_config.temperature
        if syn_config.top_p is not None:
            params["top_p"] = syn_config.top_p

        request = AdapterSynthesisRequest(
            phoneme_ids=phoneme_ids_array,
            phoneme_lengths=phoneme_ids_lengths,
            speaker_id=syn_config.speaker_id or 0,
            language_id=syn_config.lang_id or 0,
            params=params,
        )

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
            return result.audio

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

if __name__ == "__main__":
    from scriptconv.phonemizers.gl import CotoviaPhonemizer
    from scriptconv.phonemizers.he import PhonikudPhonemizer
    from scriptconv.phonemizers.mul import (EspeakPhonemizer, EpitranPhonemizer, GruutPhonemizer, ByT5Phonemizer)

    syn_config = SynthesisConfig(enable_phonetic_spellings=True)

    # test hebrew piper
    model = "/home/miro/PycharmProjects/phoonnx_tts/phonikud/model.onnx"
    config = "/home/miro/PycharmProjects/phoonnx_tts/phonikud/model.config.json"

    voice = TTSVoice.load(model_path=model, config_path=config, use_cuda=False)

    print("\n################")
    # hebrew phonemes (raw input model)
    pho = PhonikudPhonemizer(diacritics=True)
    sentence = "הכוח לשנות מתחיל ברגע שבו אתה מאמין שזה אפשרי!"
    sentence = pho.phonemize_string(sentence, "he")

    print("## piper hebrew (raw)")
    print("-", voice.config.phoneme_type)
    slug = f"piper_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
    with wave.open(f"{slug}.wav", "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file, syn_config)

    print("\n################")
    sentence = "הכוח לשנות מתחיל ברגע שבו אתה מאמין שזה אפשרי!"
    voice.config.phoneme_type = PhonemeType.PHONIKUD
    voice.phonemizer = pho

    print("## piper hebrew (phonikud)")
    print("-", voice.config.phoneme_type)
    slug = f"piper_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
    with wave.open(f"{slug}.wav", "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file, syn_config)

    exit()
    # test piper
    model = "/home/miro/PycharmProjects/phoonnx_tts/miro_en-GB.onnx"
    config = "/home/miro/PycharmProjects/phoonnx_tts/piper_espeak.json"

    voice = TTSVoice.load(model_path=model, config_path=config, use_cuda=False)
    byt5_phonemizer = ByT5Phonemizer()
    gruut_phonemizer = GruutPhonemizer()
    espeak_phonemizer = EspeakPhonemizer()
    epitran_phonemizer = EpitranPhonemizer()
    cotovia_phonemizer = CotoviaPhonemizer()

    sentence = "A rainbow is a meteorological phenomenon that is caused by reflection, refraction and dispersion of light in water droplets resulting in a spectrum of light appearing in the sky. It takes the form of a multi-colored circular arc. Rainbows caused by sunlight always appear in the section of sky directly opposite the Sun."

    print("\n################")
    print("## piper")
    for phonemizer_type, phonemizer in [
        (PhonemeType.ESPEAK, espeak_phonemizer),
        (PhonemeType.BYT5, byt5_phonemizer),
        (PhonemeType.GRUUT, gruut_phonemizer),
        (PhonemeType.EPITRAN, epitran_phonemizer)
    ]:
        voice.config.phoneme_type = phonemizer_type
        voice.phonemizer = phonemizer
        print("-", phonemizer_type)

        slug = f"piper_{phonemizer_type.value}_{voice.config.lang_code}"
        with wave.open(f"{slug}.wav", "wb") as wav_file:
            voice.synthesize_wav(sentence, wav_file, syn_config)

    print("\n################")
    print("## mimic3")
    model = "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/generator.onnx"
    config = "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/config.json"
    phonemes_txt = "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/phonemes.txt"
    phoneme_map = "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/phoneme_map.txt"

    voice = TTSVoice.load(model_path=model, config_path=config,
                          phonemes_txt=phonemes_txt, phoneme_map=phoneme_map,
                          use_cuda=False)
    for phonemizer_type, phonemizer in [
        (PhonemeType.ESPEAK, espeak_phonemizer),
        (PhonemeType.BYT5, byt5_phonemizer),
        (PhonemeType.GRUUT, gruut_phonemizer),
        (PhonemeType.EPITRAN, epitran_phonemizer)
    ]:
        voice.config.phoneme_type = phonemizer_type
        voice.phonemizer = phonemizer
        print("-", phonemizer_type)
        slug = f"mimic3_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
        with wave.open(f"{slug}.wav", "wb") as wav_file:
            voice.synthesize_wav(sentence, wav_file, syn_config)

    # Test grapheme model directly
    print("\n################")
    print("## coqui vits")
    model = "/home/miro/PycharmProjects/phoonnx_tts/celtia_vits/model.onnx"
    config = "/home/miro/PycharmProjects/phoonnx_tts/celtia_vits/config.json"

    sentence = "Este é un sistema de conversión de texto a voz en lingua galega baseado en redes neuronais artificiais. Ten en conta que as funcionalidades incluídas nesta páxina ofrécense unicamente con fins de demostración. Se tes algún comentario, suxestión ou detectas algún problema durante a demostración, ponte en contacto connosco."

    voice = TTSVoice.load(model_path=model, config_path=config,
                          use_cuda=False, lang_code="gl-ES")
    print("-", voice.config.phoneme_type)
    print(voice.config)
    phones = voice.phonemize(sentence)
    print(phones)
    print(voice.phonemes_to_ids(phones[0]))

    slug = f"vits_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
    with wave.open(f"{slug}.wav", "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file, syn_config)

    # Test cotovia phonemizer
    print("\n################")
    print("## cotovia coqui vits")
    model = "/home/miro/PycharmProjects/phoonnx_tts/sabela_cotovia/model.onnx"
    config = "/home/miro/PycharmProjects/phoonnx_tts/sabela_cotovia/config.json"

    sentence = "Este é un sistema de conversión de texto a voz en lingua galega baseado en redes neuronais artificiais. Ten en conta que as funcionalidades incluídas nesta páxina ofrécense unicamente con fins de demostración. Se tes algún comentario, suxestión ou detectas algún problema durante a demostración, ponte en contacto connosco."

    voice = TTSVoice.load(model_path=model, config_path=config,
                          use_cuda=False, lang_code="gl-ES")
    print("-", voice.config.phoneme_type)
    print(voice.config)
    phones = voice.phonemize(sentence)
    print(phones)
    print(voice.phonemes_to_ids(phones[0]))

    slug = f"vits_{voice.config.phoneme_type.value}_{voice.config.lang_code}"
    with wave.open(f"{slug}.wav", "wb") as wav_file:
        voice.synthesize_wav(sentence, wav_file, syn_config)
