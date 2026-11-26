import copy
import dataclasses
import os
import sys
import warnings
from dataclasses import MISSING, asdict, dataclass, field, fields, replace
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Union, Dict

import text2spoken

from phoonnx.util import LOG, is_arabic, normalize_lang
from phoonnx.version import VERSION_STR
from phoonnx.errors import UnsupportedVoiceLanguage
from phoonnx.tokenizer import (TTSTokenizer, Vocabulary, BlankBetween,
                                 DEFAULT_BLANK_WORD_TOKEN, DEFAULT_BLANK_TOKEN,
                                 DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN)

DEFAULT_HOP_LENGTH = 256


class Engine(str, Enum):
    """voices trained with these frameworks are explicitly supported.
    This mainly affects the format of .json file and possibly tokenization"""
    PHOONNX = "phoonnx"
    VITS = "vits"  # single-graph VITS, the adapter the format names below fall back to
    VITS_STREAMING = "vits_streaming"  # VITS split at the flow/HiFiGAN boundary
    VITS_PRIOR_SPLIT = "vits_prior_split"  # VITS split at the prior boundary
    PIPER = "piper"
    MIMIC3 = "mimic3"
    COQUI = "coqui"
    TRANSFORMERS = "transformers"
    MATCHA = "matcha"  # flow-matching mel model + separate vocoder
    OPTISPEECH = "optispeech"  # FastSpeech2-style acoustic + GAN vocoder
    GLOWTTS = "glowtts"  # flow-based mel model + separate vocoder (Larynx)
    MIXERTTS = "mixertts"  # MLP-Mixer/FastPitch-style mel model + separate vocoder
    FASTPITCH = "fastpitch"  # FastSpeech2-style mel model + separate vocoder
    STYLETTS2 = "styletts2"  # StyleTTS2 / Kokoro end-to-end (tokens + style -> wav)
    YOURTTS = "yourtts"  # multilingual VITS conditioned on a speaker d-vector (cloning)
    ZIPVOICE = "zipvoice"  # flow-matching, in-context cloning (iterative ODE loop)
    SHAMI = "shami"  # Levantine Arabic / English code-switching (HamsVITS)
    F5TTS = "f5tts"  # F5-TTS / Habibi-TTS: DiT flow-matching, Euler ODE (iterative)
    CHATTERBOX = "chatterbox"  # autoregressive codec-LM, d-vector cloning + exaggeration
    SUPERTONIC = "supertonic"  # Supertone SuperTonic: 4-graph flow-matching, raw-text (no phonemizer)
    NEUTTS = "neutts"  # NeuTTS Air / VieNeu / Akiti: Qwen3 codec-LM + NeuCodec decoder
    POCKETTTS = "pockettts"  # Kyutai Pocket TTS: 5-graph flow-matching codec LM, raw-text (no phonemizer)
    SPARKTTS = "sparktts"  # Spark-TTS: Qwen2 codec-LM + BiCodec, preset or zero-shot speakers
    QWEN3TTS = "qwen3tts"  # Qwen3-TTS: talker + code predictor, 16 code groups, 12.5 Hz codec
    OUTETTS = "outetts"  # OuteTTS 1.0: Llama/Qwen codec-LM + DAC.speech decoder, 23 languages
    ARKTTS = "arktts"  # ArkTTS (Audio8 / Zortzi): DualAR codec-LM, 10 codebooks, 44.1 kHz codec
    OMNIVOICE = "omnivoice"  # OmniVoice (k2-fsa): masked-diffusion codec LM, 600+ languages
    INDIC_PARLER = "indic_parler"  # AI4Bharat Indic Parler-TTS: T5 encoder + AR DAC codec LM
    LLASA = "llasa"  # Llasa (HKUST): LLaMA codec-LM + XCodec2 decoder, 50 Hz single codebook
    ORPHEUS = "orpheus"  # Orpheus (Canopy Labs): Llama codec-LM + SNAC decoder, emotive tags
    MAGPIE = "magpie"  # NVIDIA Magpie-TTS: encoder-decoder codec LM, 8 codebooks, 12 languages
    MOSSTTS = "mosstts"  # MOSS-TTS-Nano: autoregressive RVQ-16 codec-LM, zero-shot cloning @48kHz
    VOSK = "vosk"  # alphacep vosk-tts: VITS + dictionary/rule Russian g2p
    RAPIDO = "rapido"  # Rápido: flow-matching DiT over VoxCPM2 AudioVAE latents @48kHz
    TINYVOCA = "tinyvoca"  # tiny-voca: dizione tokens + stress, source-filter waveform head
    XVITS = "xvits"  # x-vits: Piper/VITS with flow-matching prosody and word emphasis (5 graphs)

    @classmethod
    def _missing_(cls, value: object) -> Optional["Engine"]:
        """Resolve a name a published config may use to the member that owns it.

        The alias resolves to the *same* member, so every comparison against
        ``Engine.STYLETTS2`` keeps working and a config is not routed to a
        second, separate value.
        """
        if isinstance(value, str):
            canonical = ENGINE_ALIASES.get(value.strip().lower())
            if canonical is not None:
                return cls(canonical)
        return None


#: Names a published config may use for an engine above. Kokoro is the
#: StyleTTS2 graph under another name, and the sherpa-onnx and kokoro-onnx
#: tooling writes "kokoro", so a config copied from there must not be refused.
#: Kept beside the enum rather than inside it: a plain attribute in an Enum
#: body becomes a member.
ENGINE_ALIASES = {"kokoro": Engine.STYLETTS2.value}


# Alphabet and PhonemeType are wire-format enums shared with scriptconv;
# PhonemeType is scriptconv's Phonemizer under its historical name.  Aliasing
# (not redefinition) keeps enum identity: values stored in voice configs and
# pickles resolve to the same class everywhere.
from scriptconv.phonemizers.enums import Alphabet, Phonemizer as PhonemeType
from scriptconv.frontend import FrontendConfig, Stages, TextFrontend, spoken_normalizer

TEXT_CASINGS: Dict[str, Callable[[str], str]] = {
    "ignore": lambda s: s,
    "lower": str.lower,
    "upper": str.upper,
    "casefold": str.casefold,
}


def get_text_casing(casing: str) -> Callable[[str], str]:
    """The casing function named by ``casing`` (a key of ``TEXT_CASINGS``); any other name leaves the text as is."""
    return TEXT_CASINGS.get(casing, TEXT_CASINGS["ignore"])


@dataclass
class VoiceConfig:
    """TTS model configuration"""

    num_symbols: int
    """Number of phonemes."""

    num_speakers: int
    """Number of speakers."""

    num_langs: int
    """Number of langs."""

    sample_rate: int
    """Sample rate of output audio."""

    lang_code: Optional[str]
    """Name of espeak-ng voice or alphabet."""

    phoneme_type: PhonemeType
    """Dual-role field: conversion backend **and** tokenisation recipe.

    ``phoneme_type`` serves two tightly-coupled purposes that are
    intentionally unified, not accidentally merged:

    1. **Conversion backend** – which graphemes→phoneme implementation to
       call (e.g. ``espeak``, ``gruut``, ``misaki_en``).  This is the
       *how* of the graphemes→phoneme conversion (scriptconv routes it).

    2. **Tokenisation recipe** – the token vocabulary and splitting rules
       for the model's input layer are built to match the output of the
       chosen backend.  Swapping the backend without a matching vocabulary
       would produce incorrect token IDs.

    Relationship to other alphabet fields:

    +----------------------------+------------------------------+-------------------------------+
    | concept                    | answers                      | example                       |
    +============================+==============================+===============================+
    | ``VoiceConfig.alphabet``   | WHAT the model eats          | ``Alphabet.IPA``              |
    +----------------------------+------------------------------+-------------------------------+
    | ``phoneme_type``           | HOW to get there (convert    | ``PhonemeType.ESPEAK``        |
    |                            | backend + tokenisation)      |                               |
    +----------------------------+------------------------------+-------------------------------+
    | ``SynthesisConfig.alphabet`` | WHAT the user's text is    | ``Alphabet.GRAPHEMES`` / None |
    +----------------------------+------------------------------+-------------------------------+
    """

    alphabet: Optional[Alphabet]
    """Alphabet (token space) that the model was trained on.

    This is the *target* of every text-to-phoneme conversion step and
    must match the model's vocabulary exactly.  Typical values:

    * ``Alphabet.IPA`` — espeak / gruut / misaki phoneme models.
    * ``Alphabet.UNICODE`` — grapheme/character-level models (piper ``text``
      type, Coqui VITS grapheme models).
    * ``Alphabet.ARPA`` — ARPABET-trained models (rare).
    * ``Alphabet.HANGUL`` — Korean Hangul-input models.
    * ``Alphabet.HIRA`` / ``Alphabet.KANA`` — Japanese script models.
    """

    phonemizer_model: Optional[str]
    """for phonemizers that allow changing base model """

    phonemizer_lang: Optional[str] = None
    """The language the phonemizer reads when it differs from ``lang_code``, the text
    language: the eSpeak voice ``en`` of an en-US Piper voice. Normalisation keeps
    ``lang_code``."""

    speaker_id_map: Mapping[str, int] = field(default_factory=dict)
    """Speaker -> id"""

    default_speaker_id: Optional[int] = None
    """The speaker of a multi-speaker voice when the call names none. ``None`` is speaker 0."""

    lang_id_map: Mapping[str, int] = field(default_factory=dict)
    """lang-code -> id"""

    # Info about what framework was used to train the model
    engine: Engine = Engine.PHOONNX

    # Inference scales from the config's ``inference`` block; None lets the engine's
    # ``default_params()`` decide.
    length_scale: Optional[float] = None
    noise_scale: Optional[float] = None
    noise_w_scale: Optional[float] = None
    add_diacritics: bool = None # arabic and hebrew
    # diacritizer model for languages that need one; scriptconv routes it to the
    # right backend: Arabic text2tashkeel model name (e.g. "rawi-ensemble"),
    # Hebrew phonikud ONNX path (None -> scriptconv auto-provisions + caches), or
    # stressonnx model. None lets each backend pick its own default.
    diacritizer_model: Optional[str] = None

    # samples per model frame — used to convert per-phoneme durations (in model
    # frames) to audio samples for the phoneme-alignment feature (see
    # docs/usage.md). Matches the vocoder/decoder hop length; 256 for the
    # standard 22.05 kHz VITS/HiFi-GAN exports.
    hop_length: int = DEFAULT_HOP_LENGTH

    # tokenization settings
    tokenizer: Optional[TTSTokenizer] = None
    blank_at_start: bool = True
    blank_at_end: bool = True
    pad_token: Optional[str] = DEFAULT_PAD_TOKEN
    blank_token: Optional[str] = DEFAULT_PAD_TOKEN
    bos_token: Optional[str] = DEFAULT_BOS_TOKEN
    eos_token: Optional[str] = DEFAULT_EOS_TOKEN
    word_sep_token: Optional[str] = DEFAULT_BLANK_WORD_TOKEN
    blank_between: BlankBetween = BlankBetween.TOKENS_AND_WORDS

    # Adapter-specific parameters parsed from config JSON
    engine_params: Dict[str, Any] = field(default_factory=dict)

    # Optional BCP47/lang-code -> internal language-token map. Lets a voice override how
    # its lang_code becomes the model's language token (e.g. dialect models that repurpose
    # the base tokens with a literal token string). Empty -> derive the token from lang_code.
    lang_tokens: Dict[str, str] = field(default_factory=dict)

    phonemizer_identity: Optional[Dict[str, Any]] = None
    """The phonemizer that made the training phonemes, as preprocessing records it.

    The keys are those of :class:`scriptconv.phonemizers.base.PhonemizerIdentity`.
    :class:`phoonnx.voice.TTSVoice` compares it with the running phonemizer at
    load and warns on a difference. ``None`` for a voice that records none.
    """

    keep_punctuation: bool = False
    """Whether the model was trained with clause punctuation (``, . ? ! : ;``).

    Synthesis keeps punctuation only when this is true, so a model is fed the
    symbols it learned and nothing else. Piper voices are trained with it.
    phoonnx datasets record it in ``config.json`` from preprocessing; a phoonnx
    config without the key was trained without it.
    """

    text_casing: str = "ignore"
    """The casing training applied to the text before phonemization: ``lower``,
    ``upper``, ``casefold`` or ``ignore``. Synthesis applies the same casing,
    after normalisation, so the model reads the text as it learned it.
    phoonnx datasets record it in ``config.json`` from preprocessing; a config
    without the key applies none.
    """

    def __post_init__(self):
        """
        Finalize dataclass defaults after initialization.
        
        If `add_diacritics` is None, sets it to True when `lang_code` names Arabic or an individual Arabic language in Arabic script, and to False otherwise. Ensures `lang_code` is set to "und" when not provided.
        """
        # cast strings to enum for consistency
        if not isinstance(self.engine, Engine) and isinstance(self.engine, str):
            self.engine = Engine(self.engine)
        if not isinstance(self.alphabet, Alphabet) and isinstance(self.alphabet, str):
            self.alphabet = Alphabet(self.alphabet)
        if self.alphabet is None:
            # The alphabet names the model's token space and is what the
            # conversion routes to, so it can never be absent. Vocab- and
            # tokens-file voices (transformers, sherpa) carry no alphabet of
            # their own; they are character models, like every other branch that
            # falls back here.
            self.alphabet = Alphabet.UNICODE
        if not isinstance(self.phoneme_type, PhonemeType) and isinstance(self.phoneme_type, str):
            self.phoneme_type = PhonemeType(self.phoneme_type)
        if self.phoneme_type is None:
            # Vocab- and tokens-file voices (transformers, sherpa) carry no
            # phoneme_type of their own; they are character models, like the
            # alphabet fallback above. GRAPHEMES (not UNICODE): the grapheme
            # phonemizer case-folds and NFC-composes, so lowercase precomposed
            # vocabs (the MMS shape) keep matching instead of dropping OOV
            # codepoints, and it matches what the shipped index declares for
            # every voice of this shape.
            self.phoneme_type = PhonemeType.GRAPHEMES

        if self.add_diacritics is None:
            self.add_diacritics = False
            if self.lang_code and is_arabic(self.lang_code):
                self.add_diacritics = True

        self.lang_code = normalize_lang(self.lang_code or "und", default_region=False)

    @staticmethod
    def is_mimic3(config: dict[str, Any]) -> bool:
        """Whether *config* is a mimic3 voice."""
        from phoonnx.compat import LoadRequest, Mimic3Loader
        return Mimic3Loader.detect(LoadRequest(config=config))

    @staticmethod
    def is_piper(config: dict[str, Any]) -> bool:
        """Whether *config* is a piper voice."""
        from phoonnx.compat import LoadRequest, PiperLoader
        return PiperLoader.detect(LoadRequest(config=config))

    @staticmethod
    def is_vosk(config: dict[str, Any]) -> bool:
        """Whether *config* is an alphacep vosk-tts voice."""
        from phoonnx.compat import LoadRequest, VoskLoader
        return VoskLoader.detect(LoadRequest(config=config))

    @staticmethod
    def is_coqui_vits(config: dict[str, Any]) -> bool:
        """Whether *config* is a Coqui VITS grapheme voice."""
        from phoonnx.compat import CoquiLoader, LoadRequest
        return CoquiLoader.detect(LoadRequest(config=config))

    @staticmethod
    def is_phoonnx(config: dict[str, Any]) -> bool:
        """Whether *config* is a native phoonnx voice."""
        from phoonnx.compat import LoadRequest, PhoonnxLoader
        return PhoonnxLoader.detect(LoadRequest(config=config))

    @property
    def frontend(self) -> FrontendConfig:
        """What the text frontend is built from. The voice adds the normaliser, a callable."""
        return FrontendConfig(lang=self.lang_code, phonemizer=self.phoneme_type, alphabet=self.alphabet,
                              model=self.phonemizer_model, phonemizer_lang=self.phonemizer_lang,
                              stages=Stages(diacritize=self.add_diacritics,
                                            diacritizer_model=self.diacritizer_model),
                              keep_punctuation=self.keep_punctuation)

    @staticmethod
    def from_dict(config: dict[str, Any], **options: Any) -> "VoiceConfig":
        """Build a VoiceConfig from a config dict and its companion files.

        A config with a ``phoonnx_version`` key is native. Any other config is a third-party
        format: it still loads, with a ``DeprecationWarning``, and
        :func:`phoonnx.compat.convert_config` writes the native config once. ``options`` are the
        keyword parameters of :func:`phoonnx.compat.build_voice_config`.

        Raises:
            ValueError: If the format needs a companion file that was not given.
        """
        from phoonnx.compat import build_voice_config

        if "phoonnx_version" not in config:
            warnings.warn("The config is not in the native phoonnx format. phoonnx "
                          f"{_removal_version()} reads only the native format. Convert it once with "
                          "phoonnx.compat.convert_config.", DeprecationWarning, stacklevel=2)
        return build_voice_config(config, **options)

    def to_dict(self) -> Dict[str, Any]:
        """The native ``config.json`` dict, which :meth:`from_dict` reads back with no warning.

        A raw-text tokenizer (Chatterbox) has no symbol table, so its vocabulary is empty;
        the voice still needs ``bpe_tokenizer_json`` at load. A missing pad is written as null,
        which reads back as no pad. Another special token that the vocabulary lacks is written
        as the phoonnx default, which is what a read gives, and the flags that need that token
        are written off, so the ids stay the same.
        """
        tok = self.tokenizer
        if isinstance(tok, TTSTokenizer):
            voc = tok.vocabulary
            flags = {"add_blank_char": tok.add_blank_char and voc.blank_id is not None,
                     "add_blank_word": tok.add_blank_word,
                     "use_eos_bos": tok.use_eos_bos and voc.bos_id is not None and voc.eos_id is not None,
                     "blank_at_start": tok.blank_at_start and voc.blank_id is not None,
                     "blank_at_end": tok.blank_at_end,
                     "fold_compounds": tok.fold_compounds}
            if tok.blank_after is not None:
                flags["blank_after"] = sorted(tok.blank_after)
        else:
            voc = Vocabulary(char2idx={})
            flags = dict.fromkeys(("add_blank_char", "add_blank_word", "use_eos_bos", "blank_at_start",
                                   "blank_at_end"), False)
            flags["fold_compounds"] = True
        # an unset scale stays out of the file, so a reader uses the engine default
        scales = {"noise_scale": self.noise_scale, "length_scale": self.length_scale, "noise_w": self.noise_w_scale}
        native = {
            "phoonnx_version": VERSION_STR,
            "engine": self.engine.value,
            "phoneme_type": self.phoneme_type.value,
            "alphabet": self.alphabet.value,
            "lang_code": self.lang_code,
            **({"phonemizer_lang": self.phonemizer_lang} if self.phonemizer_lang else {}),
            "audio": {"sample_rate": self.sample_rate},
            "hop_length": self.hop_length,
            "num_symbols": self.num_symbols,
            "num_speakers": self.num_speakers,
            "num_langs": self.num_langs,
            "speaker_id_map": dict(self.speaker_id_map),
            "default_speaker_id": self.default_speaker_id,
            "lang_id_map": dict(self.lang_id_map),
            "lang_tokens": dict(self.lang_tokens),
            "phonemizer_model": self.phonemizer_model,
            "keep_punctuation": self.keep_punctuation,
            "text_casing": self.text_casing,
            "inference": {
                **{key: value for key, value in scales.items() if value is not None},
                "add_diacritics": self.add_diacritics,
                "diacritizer_model": self.diacritizer_model,
            },
            "phoneme_id_map": dict(voc.char2idx),
            "pad": voc.pad, "blank": voc.blank or DEFAULT_BLANK_TOKEN,
            "bos": voc.bos or DEFAULT_BOS_TOKEN, "eos": voc.eos or DEFAULT_EOS_TOKEN,
            "blank_word": voc.blank_word,
            **flags,
            "word_sep_token": self.word_sep_token,
            "blank_between": self.blank_between.value,
            "engine_params": {k: v for k, v in self.engine_params.items() if k not in ("providers", "threads")},
        }
        if self.phonemizer_identity is not None:
            native["phonemizer_identity"] = dict(self.phonemizer_identity)
        return native

    def to_native_dict(self) -> Dict[str, Any]:
        """Deprecated. Use :meth:`to_dict`."""
        _warn_moved("VoiceConfig.to_native_dict", "VoiceConfig.to_dict")
        return self.to_dict()


@dataclass(frozen=True)
class SpeakerReference:
    """A recording of the speaker to imitate."""

    audio: Any = None
    """A path or URL of a wav file, or an ``(audio, sample_rate)`` tuple. ``None`` means no reference."""

    text: Optional[str] = None
    """The transcription of :attr:`audio`. In-context cloning engines (ZipVoice) need it;
    d-vector engines (YourTTS, StyleTTS2) ignore it."""

    lang: Optional[str] = None
    """The language of :attr:`text`. ``None`` takes the voice's language."""


@dataclass(frozen=True)
class AudioPost:
    """What happens to the engine's audio before it is returned."""

    normalize: bool = True
    """Scale each sentence to the full sample range."""

    target_loudness: Optional[float] = None
    """The integrated loudness, in LUFS, of each sentence (for example ``-16.0``), with the
    peaks limited to -1 dBFS. ``None`` keeps :attr:`normalize`. It needs ``scipy``; without
    it a warning is logged once and :attr:`normalize` applies."""

    volume: float = 1.0
    """A multiplier for the samples (< 1 is quieter, > 1 is louder)."""

    super_resolution: bool = False
    """Upscale each chunk to 48 kHz with ``audiosronnx`` (the ``audiosr`` extra)."""

    super_resolution_model: Optional[str] = None
    """The ``audiosronnx`` engine. ``None`` selects ``"novasr"``."""

    voice_conversion: Optional[SpeakerReference] = None
    """Convert each chunk to the speaker of this reference with ``voiceclonnx``. It works with
    every engine, but only the timbre changes. :attr:`SynthesisConfig.reference` clones in
    the engine, so the prosody follows the reference too. ``None`` imports nothing."""

    voice_conversion_engine: Optional[str] = None
    """The ``voiceclonnx`` engine. ``None`` selects ``"openvoice"``."""


# The 1.x name of each setting, and the path of its new home.
_1X_NAMES = {
    "alphabet": "input_alphabet",
    "enable_phonetic_spellings": "phonetic_spellings",
    "speaker_id": "speaker",
    "lang_id": "language",
    "extra_params": "extra",
    "speaker_reference": "reference.audio",
    "speaker_reference_text": "reference.text",
    "speaker_reference_lang": "reference.lang",
    "add_diacritics": "stages.diacritize",
    "diacritizer_model": "stages.diacritizer_model",
    "chunk": "stages.chunk",
    "normalize_audio": "audio.normalize",
    "target_loudness": "audio.target_loudness",
    "volume": "audio.volume",
    "super_resolution": "audio.super_resolution",
    "super_resolution_model": "audio.super_resolution_model",
    "vc_reference": "audio.voice_conversion.audio",
    "vc_engine": "audio.voice_conversion_engine",
}

_GROUPS = {"stages": Stages, "audio": AudioPost, "reference": SpeakerReference,
           "voice_conversion": SpeakerReference}

# The 1.x field order, which positional arguments follow.
_1X_ORDER = ("alphabet", "speaker_id", "lang_id", "speaker_reference", "speaker_reference_text",
             "speaker_reference_lang", "exaggeration", "temperature", "top_p", "length_scale",
             "noise_scale", "noise_w_scale", "normalize_audio", "target_loudness", "volume",
             "enable_phonetic_spellings", "add_diacritics", "diacritizer_model", "chunk",
             "super_resolution", "super_resolution_model", "vc_reference", "vc_engine", "extra_params")


def _warn_1x(old: str, new: str) -> None:
    """A ``DeprecationWarning`` attributed to the first caller outside this module and
    :mod:`dataclasses`, so that ``from_dict`` and ``replace`` warn at the caller's line."""
    level, frame = 2, sys._getframe(1)
    while frame.f_back is not None and frame.f_code.co_filename in (__file__, dataclasses.__file__):
        level, frame = level + 1, frame.f_back
    warnings.warn(f"{old} is deprecated and will be removed in phoonnx {_removal_version()}. Use {new}.",
                  DeprecationWarning, stacklevel=level)


class _LegacyId(str):
    """A string given as the 1.x ``speaker_id`` or ``lang_id``.

    1.x passed it to the engine as it was: most engines read it with ``int()``, and
    Magpie reads a voice name. So the voice passes it on unread. No voice map and no
    text stage reads it, and ``speaker_id="40"`` stays id 40 on a voice whose map
    has a speaker named "40".
    """


def _as_group(group: type, value: Any) -> Any:
    """*value*, or the *group* that *value* describes when it is a mapping."""
    return group(**value) if isinstance(value, Mapping) else value


def _with_path(group: Any, path: str, value: Any) -> Any:
    """A copy of *group* with the field at the dotted *path* set to *value*."""
    head, _, rest = path.partition(".")
    if rest:
        value = _with_path(getattr(group, head) or _GROUPS[head](), rest, value)
    return replace(group, **{head: value})


@dataclass(frozen=True, init=False)
class SynthesisConfig:
    """What one synthesis call asks for. A ``None`` takes the voice's own value.

    The 1.x flat names (``speaker_id``, ``add_diacritics``, ``normalize_audio`` and the
    others in ``_1X_NAMES``) still work as keywords and attributes, with a
    ``DeprecationWarning`` that names the new field.
    """

    stages: Stages = Stages()
    """The text stages: ``diacritize``, ``diacritizer_model``, ``normalize``, ``chunk`` and ``lang``."""

    speaker: Optional[Union[int, str]] = None
    """The speaker of a multi-speaker voice: an id, or a name in the voice's
    ``speaker_id_map``. A name that the map lacks goes to an engine with names of its
    own (Magpie, Orpheus); any other engine gets a ``ValueError`` that lists the names
    of the voice. ``None`` takes the voice's own speaker, or speaker 0."""

    language: Optional[Union[int, str]] = None
    """The language of a multi-language voice: an id, or a code in the voice's
    ``lang_id_map``. A code that the map of a multilingual voice lacks gives a
    ``ValueError`` that lists the codes of the voice. A code also sets ``stages.lang``
    when that is ``None``."""

    length_scale: Optional[float] = None
    """The phoneme length scale (< 1 is faster, > 1 is slower)."""

    noise_scale: Optional[float] = None
    """The generator noise."""

    noise_w_scale: Optional[float] = None
    """The phoneme width noise."""

    temperature: Optional[float] = None
    """The sampling temperature of autoregressive engines; ``0`` is greedy decoding."""

    top_p: Optional[float] = None
    """The nucleus sampling cutoff of autoregressive engines."""

    exaggeration: Optional[float] = None
    """The expressiveness (0.0 to 1.0) of engines that support it (Chatterbox)."""

    reference: Optional[SpeakerReference] = None
    """The speaker that a cloning engine imitates."""

    extra: Mapping[str, Any] = field(default_factory=dict)
    """Engine-specific settings of one call, such as ``d_factor`` or ``voice``."""

    audio: AudioPost = AudioPost()
    """What happens to the audio after the engine call."""

    phonetic_spellings: bool = True
    """Apply the voice's phonetic spellings to the text."""

    input_alphabet: Optional[Alphabet] = None
    """The alphabet of the caller's text. ``None`` means graphemes. Set it for text
    that is already phonemic, such as IPA, so that it is converted and not phonemized."""

    def __init__(self, *args: Any, **kwargs: Any):
        if len(args) > len(_1X_ORDER):
            raise TypeError(f"SynthesisConfig takes at most {len(_1X_ORDER)} positional arguments")
        if args:
            _warn_1x("Positional construction of SynthesisConfig", "keywords")
        for name, value in zip(_1X_ORDER, args):
            if name in kwargs:
                raise TypeError(f"SynthesisConfig got multiple values for argument {name!r}")
            kwargs[name] = value
        renamed = {name: kwargs.pop(name) for name in list(kwargs) if name in _1X_NAMES}
        names = {f.name for f in fields(self)}
        for name in kwargs.keys() - names:
            raise TypeError(f"SynthesisConfig got an unexpected keyword argument {name!r}")
        for f in fields(self):
            default = f.default_factory() if f.default is MISSING else f.default
            object.__setattr__(self, f.name, kwargs.get(f.name, default))
        object.__setattr__(self, "stages", _as_group(Stages, self.stages))
        object.__setattr__(self, "reference", _as_group(SpeakerReference, self.reference))
        if isinstance(self.audio, Mapping):
            audio = dict(self.audio)
            audio["voice_conversion"] = _as_group(SpeakerReference, audio.get("voice_conversion"))
            object.__setattr__(self, "audio", AudioPost(**audio))
        for name, value in renamed.items():
            path = _1X_NAMES[name]
            _warn_1x(f"SynthesisConfig.{name}", f"SynthesisConfig.{path}")
            if name in ("speaker_id", "lang_id") and isinstance(value, str):
                value = _LegacyId(value)
            head, _, rest = path.partition(".")
            if rest:
                value = _with_path(getattr(self, head) or _GROUPS[head](), rest, value)
            object.__setattr__(self, head, value)
        if self.input_alphabet is not None:
            object.__setattr__(self, "input_alphabet", Alphabet(self.input_alphabet))

    def __getattr__(self, name: str) -> Any:
        if name not in _1X_NAMES:
            raise AttributeError(f"'SynthesisConfig' object has no attribute {name!r}")
        _warn_1x(f"SynthesisConfig.{name}", f"SynthesisConfig.{_1X_NAMES[name]}")
        value = self
        for part in _1X_NAMES[name].split("."):
            value = None if value is None else getattr(value, part)
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SynthesisConfig":
        """The config that *data* describes: the output of :meth:`to_dict`, or the 1.x
        flat names, which give a ``DeprecationWarning``."""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """The config as plain data, which :meth:`from_dict` reads back and
        ``phoonnx/synthesis_config.schema.json`` describes. A 1.x ``speaker_id`` or
        ``lang_id`` string is written under its 1.x key, so that it reaches the engine
        as it is after :meth:`from_dict` too.

        Raises:
            ValueError: If ``stages.casing`` is set, or a reference holds samples
                and not a path or URL.
        """
        def reference(ref: Optional[SpeakerReference]) -> Optional[Dict[str, Any]]:
            if ref is None:
                return None
            if not isinstance(ref.audio, (str, os.PathLike, type(None))):
                raise ValueError("a SpeakerReference with samples has no plain-data form: use a path or URL")
            return {"audio": None if ref.audio is None else os.fspath(ref.audio), "text": ref.text, "lang": ref.lang}

        data = {f.name: getattr(self, f.name) for f in fields(self)}
        stages = asdict(self.stages)
        if stages.pop("casing") is not None:
            raise ValueError("Stages.casing is a callable and has no plain-data form")
        if self.stages.chunk is not None:
            stages["chunk"] = self.stages.chunk.value
        data["stages"] = stages
        data["audio"] = {**asdict(self.audio), "voice_conversion": reference(self.audio.voice_conversion)}
        data["reference"] = reference(self.reference)
        data["extra"] = dict(self.extra)
        if self.input_alphabet is not None:
            data["input_alphabet"] = self.input_alphabet.value
        for field_name, name_1x in (("speaker", "speaker_id"), ("language", "lang_id")):
            if isinstance(data[field_name], _LegacyId):
                data[name_1x] = str(data.pop(field_name))
        return data


def check_lang_supported(voice: str, lang_code: Optional[str],
                         phoneme_type: PhonemeType) -> None:
    """Eagerly verify a voice's phonemizer chain serves its language.

    Only checks backends that actually restrict languages -- scriptconv's
    registry query resolves the backend *class* (a lazy import; no
    construction of the wrapper instance itself) and calls its ``get_lang``
    classmethod when present. This is cheaper than instantiating the
    phonemizer, but not free: a few backends' ``get_lang`` (e.g.
    orthography2ipa) construct a lightweight G2P engine to answer, and
    others (euskaphone, arbtok) import their optional backing package to
    check. A missing optional package therefore surfaces as ``ImportError``
    here, not ``ValueError`` -- that is not an unsupported-language verdict,
    so it is skipped, not rejected. A grapheme or unicode passthrough returns
    the tag unchanged, so it accepts any language. Unresolvable phoneme types
    are skipped (``get_phonemizer`` raises its own error for those).

    Raises:
        UnsupportedVoiceLanguage: If the resolved backend cannot serve
            ``lang_code``.
    """
    if not lang_code or phoneme_type is None:
        return
    from scriptconv.phonemizers.registry import get_phonemizer_class
    try:
        phonemizer_cls = get_phonemizer_class(phoneme_type)
    except (KeyError, ImportError, ValueError):
        return
    try:
        resolved = phonemizer_cls.get_lang(lang_code)
    except ImportError:
        return
    except ValueError as e:
        raise UnsupportedVoiceLanguage(voice, lang_code, phoneme_type) from e
    # "und" names no language: a passthrough keeps it, but a fuzzy matcher maps it to some language
    if lang_code.split("-")[0].lower() == "und" and resolved != lang_code:
        raise UnsupportedVoiceLanguage(voice, lang_code, phoneme_type)


def _removal_version() -> str:
    """The release that removes a name deprecated in this one: the next major."""
    from phoonnx.version import VERSION_MAJOR
    return f"{VERSION_MAJOR + 1}.0.0"


def _warn_moved(old: str, new: str) -> None:
    warnings.warn(f"{old} is deprecated and will be removed in phoonnx {_removal_version()}. "
                  f"Use {new}.", DeprecationWarning, stacklevel=3)


def _voice_phonemizer(phoneme_type: PhonemeType,
                      alphabet: Alphabet = Alphabet.IPA,
                      model: Optional[str] = None,
                      keep_punctuation: bool = False) -> 'Phonemizer':
    """The phonemizer of a voice, with :func:`text2spoken.normalize` as its ``normalizer``.

    The backend comes from scriptconv's registry, which also holds the
    license-quarantined mantoq and KoG2P and provisions the Hebrew phonikud
    model. :class:`phoonnx.voice.TTSVoice`
    moves the ``normalizer`` into its :class:`scriptconv.frontend.TextFrontend`,
    so it runs before diacritization. ``keep_punctuation`` keeps clause
    punctuation in the phonemes; it must match how the model was trained (see
    :attr:`VoiceConfig.keep_punctuation`).

    Raises:
        ValueError: If the provided `phoneme_type` is not supported.
    """
    phoneme_type = PhonemeType(phoneme_type)

    from scriptconv.phonemizers import get_phonemizer as _sc_get
    # dizione gives the voice its own tokens, with the ids and stress of each segment
    options = {"tokens": True} if phoneme_type == PhonemeType.DIZIONE else {}
    try:
        phonemizer = _sc_get(phoneme_type, alphabet=alphabet, model=model, **options)
    except KeyError:
        raise ValueError(f"unsupported phoneme_type: {phoneme_type}")

    if "normalize" not in phonemizer.owned_stages:  # dizione normalises raw text itself
        phonemizer.normalizer = text2spoken.normalize
    phonemizer.keep_punctuation = keep_punctuation
    return phonemizer


def without_normalizer(phonemizer: 'Phonemizer') -> 'Phonemizer':
    """*phonemizer*, or a shallow copy of it with no ``normalizer``."""
    if phonemizer.normalizer is None:
        return phonemizer
    backend = copy.copy(phonemizer)
    backend.normalizer = None
    return backend


def voice_frontend(phonemizer: 'Phonemizer', lang: str, stages: Stages = Stages(),
                   strict: bool = False, phonemizer_lang: Optional[str] = None) -> TextFrontend:
    """The text frontend of a voice that reads *phonemizer*, in language *lang*.

    *phonemizer_lang* is the language the backend reads when it differs from *lang*.

    The phonemizer's ``normalizer`` becomes the frontend's normalisation hook
    and the frontend reads a copy of the phonemizer without it. The frontend
    skips the hook for a backend that names ``normalize`` in its
    ``owned_stages``, because that backend normalises the text itself. Such a
    phonemizer has no ``normalizer``, so its frontend gets
    :func:`scriptconv.frontend.spoken_normalizer`, which reads only a ``<say-as>``.
    Inference (:attr:`phoonnx.voice.TTSVoice.frontend`) and dataset building
    (``phoonnx_train.preprocess``) both build their frontend here, so a text
    gives the same phonemes in both. With *strict*, a failed stage raises
    instead of letting the text continue without it.
    """
    normalizer = phonemizer.normalizer
    if normalizer is None and "normalize" in phonemizer.owned_stages:
        normalizer = spoken_normalizer()
    return TextFrontend(
        FrontendConfig(lang=lang, normalizer=normalizer, keep_punctuation=phonemizer.keep_punctuation,
                       stages=stages, strict=strict, phonemizer_lang=phonemizer_lang),
        backend=without_normalizer(phonemizer))


def get_phonemizer(phoneme_type: PhonemeType,
                   alphabet: Alphabet = Alphabet.IPA,
                   model: Optional[str] = None,
                   keep_punctuation: bool = False) -> 'Phonemizer':
    """Deprecated. A phonemizer whose own ``normalizer`` is :func:`text2spoken.normalize`.

    Use :attr:`phoonnx.voice.TTSVoice.frontend`, or build a
    :class:`scriptconv.frontend.TextFrontend`.

    Raises:
        ValueError: If the provided `phoneme_type` is not supported.
    """
    _warn_moved("phoonnx.config.get_phonemizer", "TTSVoice.frontend or scriptconv.frontend.TextFrontend")
    return _voice_phonemizer(phoneme_type, alphabet, model, keep_punctuation)


def get_conversion(phonemizer, voice_config: "VoiceConfig",
                   syn_config: "SynthesisConfig", tgt_alphabet: Alphabet):
    """Deprecated. A voice's ``text -> tgt_alphabet`` conversion for one synthesis call.

    Use :attr:`phoonnx.voice.TTSVoice.frontend`: its ``phonemize_lazy`` replaces
    the graph, and its ``prepare`` replaces ``prepare_text``.

    Returns ``(graph, prepare_text)``:

    * ``graph`` — a scriptconv ``ConversionGraph`` whose phoneme edge is the
      voice's own (lazy) phonemizer. The edge yields ``(phonemes, streams)`` per
      segment of ``phonemize_segments``: ``streams`` maps an ``AdapterSynthesisRequest`` field name to one
      value per symbol, and is empty unless the phonemizer tags its symbols.
      When the voice vocalizes, scriptconv's ``text -> text-diacritized`` edge is added and the direct edge omitted, so
      routing is forced through the vocalizer by *topology* rather than a runtime
      flag. Language and model are closed over, so callers route with
      ``graph.convert(text, "text", tgt_alphabet.value)`` and carry no conversion
      context of their own.
    * ``prepare_text`` — the same vocalization as a plain ``str -> str``
      transform, for paths that feed raw text to something other than the phoneme
      route (text-token models, inline ``[[phoneme]]`` overrides). Identity when
      the voice does not vocalize.

    Undervocalized scripts (Arabic, Hebrew) need their vowel marks restored
    before G2P or the pronunciation is ambiguous; that restoration is scriptconv's
    and lives entirely behind these two values.
    """
    _warn_moved("phoonnx.config.get_conversion", "TTSVoice.frontend")
    from phoonnx.encoders import SIDE_STREAMS
    from scriptconv.graph import ConversionGraph, Edge
    from scriptconv.diacritics import DIACRITIZED, diacritize

    # explicit per-call setting wins, else the voice's own
    enabled = syn_config.stages.diacritize
    if enabled is None:
        enabled = voice_config.add_diacritics
    model = syn_config.stages.diacritizer_model or voice_config.diacritizer_model
    lang, alpha = voice_config.lang_code, tgt_alphabet.value

    def phonemize(text, **_):
        # phonemize_segments is a BasePhonemizer method, so every phonemizer has it:
        # lazy, so segment N+1 stays off the critical path of N.
        for segment in phonemizer.phonemize_segments(text, lang):
            yield list(segment.phonemes), {name: list(values) for name, values in segment.streams.items()
                                           if name in SIDE_STREAMS}

    graph = ConversionGraph()
    if not enabled:
        graph.register(Edge("text", alpha, phonemize))
        return graph, (lambda text: text)

    def _vocalize(text, **_):
        try:
            return diacritize(text, lang, diacritizer_model=model)
        except Exception as e:
            LOG.warning(f"diacritization failed for lang={lang}: {e} — synthesizing unstressed text")
            return text

    graph.register(Edge("text", DIACRITIZED, _vocalize))
    graph.register(Edge(DIACRITIZED, alpha, phonemize))
    return graph, _vocalize
