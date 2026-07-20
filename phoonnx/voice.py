"""
TTSVoice — architecture-agnostic TTS voice.

This module is the main user-facing synthesis interface.  It delegates
all ONNX-specific I/O to an engine adapter (``phoonnx.engines``),
which means the same TTSVoice class works for VITS or
any future architecture without code changes here.
"""
import json
import logging
import os.path
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import onnxruntime
from langcodes import closest_match
from quebra_frases import sentence_tokenize

from phoonnx.config import PhonemeType, VoiceConfig, SynthesisConfig, Alphabet, get_phonemizer
from phoonnx.alphabet_convert import convert as alphabet_convert
from phoonnx.engines import detect_engine, get_adapter
from phoonnx.engines.base import (
    AdapterSynthesisRequest,
    AdapterSynthesisResult,
    BaseOnnxAdapter,
)
from phoonnx.phonemizers import Phonemizer
from phoonnx.providers import ProviderSpec, make_session, resolve_providers
from phoonnx.phonemizers.base import BasePhonemizer, PhonemizedChunks
from phoonnx.tokenizer import TTSTokenizer
from phoonnx.util import LOG


_PHONEME_BLOCK_PATTERN = re.compile(r"(\[\[.*?\]\])")


def _phonemic_chunks(text: str, alphabet: Optional[Alphabet]) -> PhonemizedChunks:
    """Split an already-phonemic string into per-sentence symbol lists.

    Space-separated notations (ARPA) tokenize on whitespace; char-based
    alphabets tokenize per character.
    """
    chunks = BasePhonemizer.chunk_text(text)
    if alphabet == Alphabet.ARPA:
        return [chunk.split() for chunk, _, _ in chunks if chunk.strip()]
    return [[c for c in chunk] for chunk, _, _ in chunks if chunk]


def _load_reference_audio(ref: Any) -> tuple:
    """Normalise a cloning reference to ``(mono float32 audio, sample_rate)``.

    Accepts an ``(audio, sample_rate)`` tuple or a path to an audio file.
    """
    if isinstance(ref, tuple) and len(ref) == 2:
        audio, sr = ref
        return np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)
    try:
        import soundfile as sf
        audio, sr = sf.read(str(ref), dtype="float32")
    except ImportError:
        with wave.open(str(ref), "rb") as w:
            sr, ch = w.getframerate(), w.getnchannels()
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
            if ch > 1:
                audio = audio.reshape(-1, ch).mean(axis=1)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32).reshape(-1), int(sr)


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
                    logging.warning(f"Skipping malformed phonetic spelling line: {l!r}")
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
    phonemizer: Optional[Phonemizer] = None
    adapter: Optional[BaseOnnxAdapter] = None

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
                if engine_name and engine_name not in ("piper", "mimic3", "coqui"):
                    self.adapter = get_adapter(engine_name)
                else:
                    # For piper/mimic3/coqui all use the vits adapter
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

    def _diacritize(self, text: str, diacritizer_model: Optional[str]) -> str:
        """Apply (Arabic/Hebrew) diacritics for the voice's language."""
        return self.phonemizer.add_diacritics(
            text, self.config.lang_code, model=diacritizer_model
        )

    def _text_parts(
            self, text: str, do_diacritics: bool, diacritizer_model: Optional[str]
    ) -> Iterable[str]:
        """Yield the text unit(s) to phonemize.

        Without diacritics the whole text is yielded once so the phonemizer's own
        normalization + chunking runs over it exactly as before (guaranteeing the
        lazy stream produces the same phonemes as a single whole-text call). With
        diacritics enabled the text is split into sentences and each is
        diacritized independently — Arabic/Hebrew diacritizer models are
        sentence-level, and doing it per sentence keeps the first sentence off the
        critical path of the rest.
        """
        if not do_diacritics:
            yield text
            return
        for sentence in sentence_tokenize(text):
            if not sentence.strip():
                continue
            yield self._diacritize(sentence, diacritizer_model)

    def _iter_synthesis_ids(
            self,
            text: str,
            syn_config: SynthesisConfig,
            do_diacritics: bool,
            diacritizer_model: Optional[str],
            src_alphabet: Alphabet,
            tgt_alphabet: Alphabet,
    ) -> Iterable[tuple]:
        """Lazily yield ``(phonemes, phoneme_ids, language_ids)`` per sentence.

        ``phonemes`` is the per-sentence symbol list (``None`` for text-token
        models whose adapter owns text→ids); it is carried through only so the
        yielded :class:`AudioChunk` can expose the phonemes that produced it.

        Preserves the exact dispatch semantics of :meth:`synthesize` while
        deferring per-sentence phonemize/diacritize/tokenize work until each item
        is pulled from the generator.
        """
        lang = self.config.lang_code

        if src_alphabet == tgt_alphabet:
            if src_alphabet == Alphabet.GRAPHEMES:
                # Grapheme / text-token models: the adapter owns text -> token ids
                # (some do their own whole-text normalization, e.g. subword BPE),
                # so encode the whole text once; audio is still produced per
                # sentence downstream.
                if do_diacritics:
                    text = self._diacritize(text, diacritizer_model)
                for ids in self.adapter.encode_text(text, self, syn_config):
                    yield None, ids, None
            else:
                # Already-phonemic input in the model's own alphabet: pass through.
                # Diacritization is orthographic (grapheme-level) only; never
                # apply it to already-phonemic input.
                for phonemes in _phonemic_chunks(text, tgt_alphabet):
                    if phonemes:
                        yield phonemes, self.phonemes_to_ids(phonemes), None
            return

        if src_alphabet == Alphabet.GRAPHEMES:
            # Normal phoneme model — the hot path. Language-aware phonemizers (e.g.
            # Shami) provide per-phoneme language IDs; the two streams are produced
            # together per sentence so they can never fall out of alignment.
            lazy_lang_ids = getattr(
                self.phonemizer, "phonemize_with_language_ids_lazy", None)
            if _PHONEME_BLOCK_PATTERN.search(text):
                # Inline [[phoneme]] blocks are literal overrides that must bypass
                # the phonemizer entirely — checked before any language-aware
                # dispatch so they are never sent to it verbatim. Inline [[phoneme]]
                # blocks merge into adjacent sentences; keep the (eager) whole-text
                # handling that owns that merge logic.
                if do_diacritics:
                    text = self._diacritize(text, diacritizer_model)
                for phonemes in self.phonemize(text):
                    if phonemes:
                        yield phonemes, self.phonemes_to_ids(phonemes), None
            elif lazy_lang_ids is not None:
                for part in self._text_parts(text, do_diacritics, diacritizer_model):
                    for phonemes, language_ids in lazy_lang_ids(part, lang):
                        if phonemes:
                            yield phonemes, self.phonemes_to_ids(phonemes), language_ids
            elif hasattr(self.phonemizer, "phonemize_with_language_ids"):
                # Language-aware phonemizer without a lazy variant: eager fallback.
                if do_diacritics:
                    text = self._diacritize(text, diacritizer_model)
                sentence_phonemes, sentence_language_ids = (
                    self.phonemizer.phonemize_with_language_ids(text, lang))
                for phonemes, language_ids in zip(sentence_phonemes, sentence_language_ids):
                    if phonemes:
                        yield phonemes, self.phonemes_to_ids(phonemes), language_ids
            else:
                # Lazy per-sentence phonemization; a phonemizer without a lazy
                # variant falls back to its eager whole-text phonemize().
                phonemize_lazy = getattr(self.phonemizer, "phonemize_lazy", None)
                for part in self._text_parts(text, do_diacritics, diacritizer_model):
                    sentences = (phonemize_lazy(part, lang) if phonemize_lazy is not None
                                 else self.phonemizer.phonemize(part, lang))
                    for phonemes in sentences:
                        if phonemes:
                            yield phonemes, self.phonemes_to_ids(phonemes), None
            return

        # Already-phonemic input in a different alphabet: transcode to the model's
        # alphabet through the conversion graph. Diacritization is orthographic
        # (grapheme-level) only; never apply it to already-phonemic input.
        converted = alphabet_convert(
            text,
            lang=lang,
            src=src_alphabet,
            tgt=tgt_alphabet,
            phoneme_type=self.config.phoneme_type,
        )
        if isinstance(converted, list):
            # PhonemizedChunks returned by a phonemization edge
            sentence_phonemes = converted
            if _PHONEME_BLOCK_PATTERN.search(text):
                sentence_phonemes = self.phonemize(text)
        else:
            # Script-converted string — split into single-char token lists.
            sentence_phonemes = _phonemic_chunks(converted, tgt_alphabet)
        for phonemes in sentence_phonemes:
            if phonemes:
                yield phonemes, self.phonemes_to_ids(phonemes), None

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

        # Text preprocessing — engine-agnostic, operates on the text before encoding:
        # user pronunciation overrides (whole-text) + (Arabic/Hebrew) diacritics,
        # which are applied per sentence inside the lazy stream below.
        if self.phonetic_spellings and syn_config.enable_phonetic_spellings:
            text = self.phonetic_spellings.apply(text)

        # A per-call value wins; when unset (None) defer to the voice config so a
        # model that ships undiacritized is not force-diacritized by the caller.
        do_diacritics = syn_config.add_diacritics
        if do_diacritics is None:
            do_diacritics = self.config.add_diacritics
        # An explicit per-call diacritizer model wins, otherwise fall back to the
        # voice config's own choice.
        diacritizer_model = syn_config.diacritizer_model or self.config.diacritizer_model

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
        # reaches session.run before later sentences are phonemized/diacritized,
        # cutting time-to-first-audio on multi-sentence input.
        id_stream = self._iter_synthesis_ids(
            text, syn_config, do_diacritics, diacritizer_model,
            src_alphabet, tgt_alphabet,
        )

        # Special-token ids for alignment reconstruction — read once, only when
        # alignments are requested (keeps the default path free of this work).
        _idx2char: dict = {}
        _blank_id = _bos_id = _eos_id = None
        if include_alignments:
            _tok = self.config.tokenizer
            _vocab = _tok.vocabulary
            _idx2char = _vocab.idx2char
            _blank_id = _tok.blank_id
            _bos_id = _vocab.bos_id
            _eos_id = _vocab.eos_id

        for phonemes, phoneme_ids, language_ids in id_stream:
            if not phoneme_ids:
                continue

            phoneme_id_samples: Optional[np.ndarray] = None
            audio_result = self.phoneme_ids_to_audio(
                phoneme_ids, syn_config, language_ids=language_ids,
                include_alignments=include_alignments,
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

            yield AudioChunk(
                sample_rate=self.config.sample_rate,
                sample_width=2,
                sample_channels=1,
                audio_float_array=audio,
                phonemes=list(phonemes) if phonemes else [],
                phoneme_ids=list(phoneme_ids),
                phoneme_id_samples=phoneme_id_samples,
                phoneme_alignments=phoneme_alignments,
            )

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
        if len(phoneme_id_samples) != len(phoneme_ids):
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
        first_chunk = True
        for audio_chunk in self.synthesize(text, syn_config=syn_config):
            if first_chunk:
                if set_wav_format:
                    # Set audio format on first chunk
                    wav_file.setframerate(audio_chunk.sample_rate)
                    wav_file.setsampwidth(audio_chunk.sample_width)
                    wav_file.setnchannels(audio_chunk.sample_channels)

                first_chunk = False

            if not first_chunk:
                wav_file.writeframes(silence_int16_bytes)

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
            language_ids: Optional[list[int]] = None,
            include_alignments: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        Synthesize raw audio from phoneme ids.

        Delegates all ONNX I/O to ``self.adapter`` so the same code
        path works for VITS or any future engine.

        :param phoneme_ids: List of phoneme ids.
        :param syn_config: Synthesis configuration.
        :param language_ids: Optional per-phoneme language IDs (for language-aware engines).
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
        language_ids_array = None
        if language_ids is not None:
            language_ids_array = np.expand_dims(
                np.array(language_ids, dtype=np.int64), 0
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
            params["reference_audio"] = _load_reference_audio(syn_config.speaker_reference)
        if syn_config.speaker_reference_text:
            params["prompt_tokens"] = self._prompt_token_ids(
                syn_config.speaker_reference_text, syn_config.speaker_reference_lang)
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
            language_ids=language_ids_array,
            params=params,
        )

        # Delegate to the adapter (single-graph engines use the default
        # build_feed_dict → run → parse_outputs; iterative engines override).
        result = self.adapter.synthesize(request, self.session)

        if not include_alignments:
            return result.audio

        phoneme_id_samples = result.extras.get("phoneme_id_samples")
        if phoneme_id_samples is None:
            # Alignment is not available from this voice model.
            return result.audio, None

        # Model durations are in native frames — convert to audio samples.
        phoneme_id_samples = (
            np.asarray(phoneme_id_samples) * self.config.hop_length
        ).astype(np.int64)

        return result.audio, phoneme_id_samples

if __name__ == "__main__":
    from phoonnx.phonemizers.gl import CotoviaPhonemizer
    from phoonnx.phonemizers.he import PhonikudPhonemizer
    from phoonnx.phonemizers.mul import (EspeakPhonemizer, EpitranPhonemizer, GruutPhonemizer, ByT5Phonemizer)

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
