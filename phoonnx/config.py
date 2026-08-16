import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Union, Dict
from phoonnx.util import LOG, normalize_lang
from phoonnx.tokenizer import (TTSTokenizer, Vocabulary, BlankBetween,
                                 DEFAULT_BLANK_WORD_TOKEN, DEFAULT_BLANK_TOKEN,
                                 DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN)

DEFAULT_NOISE_SCALE = 0.667
DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_NOISE_W_SCALE = 0.8
DEFAULT_HOP_LENGTH = 256


class Engine(str, Enum):
    """voices trained with these frameworks are explicitly supported.
    This mainly affects the format of .json file and possibly tokenization"""
    PHOONNX = "phoonnx"
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


# Alphabet and PhonemeType are wire-format enums shared with scriptconv;
# PhonemeType is scriptconv's Phonemizer under its historical name.  Aliasing
# (not redefinition) keeps enum identity: values stored in voice configs and
# pickles resolve to the same class everywhere.
from scriptconv.phonemizers.enums import Alphabet, Phonemizer as PhonemeType

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

    speaker_id_map: Mapping[str, int] = field(default_factory=dict)
    """Speaker -> id"""

    lang_id_map: Mapping[str, int] = field(default_factory=dict)
    """lang-code -> id"""

    # Info about what framework was used to train the model
    engine: Engine = Engine.PHOONNX

    # Inference settings
    length_scale: float = DEFAULT_LENGTH_SCALE
    noise_scale: float = DEFAULT_NOISE_SCALE
    noise_w_scale: float = DEFAULT_NOISE_W_SCALE
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

    def __post_init__(self):
        """
        Finalize dataclass defaults after initialization.
        
        If `add_diacritics` is None, sets it to False; if `lang_code` is present and starts with "ar", sets `add_diacritics` to True. Ensures `lang_code` is set to "und" when not provided.
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
            if self.lang_code and self.lang_code.startswith("ar"):
                self.add_diacritics = True

        self.lang_code = normalize_lang(self.lang_code or "und")

    @staticmethod
    def is_mimic3(config: dict[str, Any]) -> bool:
        """Whether *config* is a mimic3 voice."""
        from phoonnx.config_loaders import LoadRequest, Mimic3Loader
        return Mimic3Loader.detect(LoadRequest(config=config))

    @staticmethod
    def is_piper(config: dict[str, Any]) -> bool:
        """Whether *config* is a piper voice."""
        from phoonnx.config_loaders import LoadRequest, PiperLoader
        return PiperLoader.detect(LoadRequest(config=config))

    @staticmethod
    def is_coqui_vits(config: dict[str, Any]) -> bool:
        """Whether *config* is a Coqui VITS grapheme voice."""
        from phoonnx.config_loaders import CoquiLoader, LoadRequest
        return CoquiLoader.detect(LoadRequest(config=config))

    @staticmethod
    def is_phoonnx(config: dict[str, Any]) -> bool:
        """Whether *config* is a native phoonnx voice."""
        from phoonnx.config_loaders import LoadRequest, PhoonnxLoader
        return PhoonnxLoader.detect(LoadRequest(config=config))

    @staticmethod
    def from_dict(config: dict[str, Any],  # phoonnx/piper/coqui/mimic3
                  vocab: Optional[Dict[str, Any]] = None,  # transformers
                  tokenizer_config: Optional[Dict[str, Any]] = None,  # transformers
                  tokens_txt: Optional[str] = None,  # sherpa/mimic3
                  lang_code: Optional[str] = None,
                  phoneme_type: Optional[Union[str, PhonemeType]] = None,
                  alphabet: Optional[Union[str, Alphabet]] = None,
                  engine: Optional[Union[str, Engine]] = None,
                   engine_params: Optional[Dict[str, Any]] = None,
                   bpe_tokenizer_json: Optional[str] = None,
                   lang_tokens: Optional[Dict[str, str]] = None) -> "VoiceConfig":
        """
        Build a VoiceConfig from a model configuration dictionary and its optional
        companion files.

        The config's format is recognised by the loader registry in
        :mod:`phoonnx.config_loaders`, which yields the fields that differ per
        format (tokenizer, engine, phoneme type, alphabet, language, diacritics);
        :func:`~phoonnx.config_loaders.resolve_overrides` then folds in the
        caller's overrides, and everything format-independent — audio, inference
        scales, speaker and language maps, special tokens — is read straight off
        the config here.

        Parameters:
            config: Parsed model configuration dictionary. Loaders may normalise
                it in place (special tokens, a defaulted sample rate).
            vocab: Token vocabulary of a transformers export.
            tokenizer_config: ``tokenizer_config.json`` of a transformers export.
            tokens_txt: Path to an external tokens file (``.txt`` or ``.json``),
                required by mimic3 voices and by sherpa-onnx style models.
            bpe_tokenizer_json: Path to a ``tokenizer.json``, required by Chatterbox.
            lang_code, phoneme_type, alphabet, engine: Overrides that win over the
                config's own values, except where the format pins them.
            engine_params: Locally-resolved adapter parameters; these win over the
                config's own ``engine_params``.
            lang_tokens: Overrides the config's BCP47 -> language-token map.

        Raises:
            ValueError: If the detected format needs a companion file that was
                not provided.
        """
        from phoonnx.config_loaders import LoadRequest, load_voice_fields

        loaded = load_voice_fields(LoadRequest(
            config=config,
            vocab=vocab,
            tokenizer_config=tokenizer_config,
            tokens_txt=tokens_txt,
            bpe_tokenizer_json=bpe_tokenizer_json,
            lang_code=lang_code or config.get("lang_code"),
            phoneme_type=phoneme_type or config.get("phoneme_type"),
            alphabet=alphabet or config.get("alphabet"),
            engine=engine,
        ))
        LOG.debug(f"phonemizer: {loaded.phoneme_type}")

        inference = config.get("inference", {})
        return VoiceConfig(
            tokenizer=loaded.tokenizer,
            num_langs=config.get("num_langs", 1),
            num_symbols=config.get("num_symbols", 256),
            num_speakers=config.get("num_speakers", 1),
            sample_rate=config.get("audio", {}).get("sample_rate", 16000),
            noise_scale=inference.get("noise_scale", DEFAULT_NOISE_SCALE),
            length_scale=inference.get("length_scale", DEFAULT_LENGTH_SCALE),
            noise_w_scale=inference.get("noise_w", DEFAULT_NOISE_W_SCALE),
            add_diacritics=loaded.add_diacritics,
            diacritizer_model=loaded.diacritizer_model,
            hop_length=config.get("hop_length", DEFAULT_HOP_LENGTH),
            lang_code=loaded.lang_code,
            alphabet=loaded.alphabet,
            engine=loaded.engine,
            phonemizer_model=config.get("phonemizer_model"),
            phoneme_type=loaded.phoneme_type,
            speaker_id_map=config.get("speaker_id_map", {}),
            blank_between=BlankBetween(loaded.blank_between),
            blank_at_start=config.get("blank_at_start", True),
            blank_at_end=config.get("blank_at_end", True),
            pad_token=config.get("pad"),
            blank_token=config.get("blank"),
            bos_token=config.get("bos"),
            eos_token=config.get("eos"),
            word_sep_token=config.get("word_sep_token") or config.get("blank_word", " "),
            # config's own engine_params (e.g. a baked YourTTS d-vector) merged with
            # any locally-resolved paths the manager passes in (the latter win).
            engine_params={**(config.get("engine_params") or {}), **(engine_params or {})},
            lang_tokens=lang_tokens or config.get("lang_tokens") or {},
            lang_id_map=config.get("lang_id_map", {}),
        )

    def to_native_dict(self) -> Dict[str, Any]:
        """
        Serialize this config to a **native phoonnx** ``config.json`` dict.

        The result loads back through the ``is_phoonnx`` path in
        :meth:`from_dict` (it carries ``phoonnx_version``), folding the
        tokenizer vocabulary into ``phoneme_id_map`` and recording the
        tokenizer flags + special tokens explicitly, so any model round-trips
        without relying on the foreign-config detection heuristics.
        """
        try:
            from phoonnx.version import VERSION as _v
            version = ".".join(str(p) for p in _v) if isinstance(_v, (tuple, list)) else str(_v)
        except Exception:
            version = "1.0"

        tok = self.tokenizer
        voc = tok.vocabulary
        return {
            "phoonnx_version": version,
            "engine": self.engine.value if self.engine else "phoonnx",
            "phoneme_type": self.phoneme_type.value,
            "alphabet": self.alphabet.value if self.alphabet else "unicode",
            "lang_code": self.lang_code,
            "audio": {"sample_rate": self.sample_rate},
            "hop_length": self.hop_length,
            "num_symbols": self.num_symbols,
            "num_speakers": self.num_speakers,
            "num_langs": self.num_langs,
            "speaker_id_map": dict(self.speaker_id_map or {}),
            "lang_id_map": dict(self.lang_id_map or {}),
            "lang_tokens": dict(self.lang_tokens or {}),
            "phonemizer_model": self.phonemizer_model,
            "inference": {
                "noise_scale": self.noise_scale,
                "length_scale": self.length_scale,
                "noise_w": self.noise_w_scale,
                "add_diacritics": self.add_diacritics,
                "diacritizer_model": self.diacritizer_model,
            },
            "phoneme_id_map": dict(voc.char2idx),
            "pad": voc.pad, "blank": voc.blank, "bos": voc.bos, "eos": voc.eos,
            "add_blank_char": tok.add_blank_char,
            "add_blank_word": tok.add_blank_word,
            "use_eos_bos": tok.use_eos_bos,
            "blank_at_start": tok.blank_at_start,
            "blank_at_end": tok.blank_at_end,
            "word_sep_token": self.word_sep_token,
            "blank_between": self.blank_between.value if self.blank_between else "tokens_and_words",
            "engine_params": dict(self.engine_params or {}),
        }


@dataclass
class SynthesisConfig:
    """Configuration for synthesis."""

    alphabet: Optional['Alphabet'] = None
    """Alphabet of the *caller's* input text.

    ``None`` means the text is plain graphemes (the default for virtually
    every use-case).  Set this when passing pre-converted text, for example
    IPA strings or Hangul, so that scriptconv's conversion graph can skip the
    phonemization step and apply the correct script-conversion
    instead.

    Relationship to other alphabet fields:

    +----------------------------+------------------------------+-------------------------------+
    | concept                    | answers                      | example                       |
    +============================+==============================+===============================+
    | ``VoiceConfig.alphabet``   | WHAT the model eats          | ``Alphabet.IPA``              |
    +----------------------------+------------------------------+-------------------------------+
    | ``VoiceConfig.phoneme_type`` | HOW to get there (convert  | ``PhonemeType.ESPEAK``        |
    |                            | backend + tokenisation)      |                               |
    +----------------------------+------------------------------+-------------------------------+
    | ``SynthesisConfig.alphabet`` | WHAT the user's text is    | ``Alphabet.GRAPHEMES`` / None |
    +----------------------------+------------------------------+-------------------------------+
    """

    speaker_id: Optional[int] = None
    """Index of speaker to use (multi-speaker voices only)."""

    lang_id: Optional[int] = None
    """Index of lang to use (multi-lang voices only)."""

    speaker_reference: Optional[Any] = None
    """Reference audio for zero-shot voice cloning (cloning engines). A path to a wav
    file, or an ``(audio, sample_rate)`` tuple. The cloning adapter turns it into the
    conditioning signal (a d-vector, or — for in-context engines like ZipVoice — the
    prompt mel)."""

    speaker_reference_text: Optional[str] = None
    """Transcription of ``speaker_reference``, required by **in-context** cloning
    engines (ZipVoice): the voice tokenizes it into the prompt tokens that prefix
    generation. Ignored by d-vector engines (YourTTS, StyleTTS2)."""

    speaker_reference_lang: Optional[str] = None
    """Language of ``speaker_reference_text`` (e.g. ``pt`` for a Portuguese clip),
    used to phonemize the reference in *its* language — which may differ from the
    target text's. Defaults to the voice's ``lang_code``. Enables cross-lingual
    cloning (a Portuguese reference speaking English). In-context engines only."""

    exaggeration: Optional[float] = None
    """Expressiveness / emotional intensity (0.0–1.0, default 0.5) for engines that
    support it (Chatterbox). Higher = more exaggerated prosody. Ignored otherwise."""

    temperature: Optional[float] = None
    """Sampling temperature for autoregressive engines (Chatterbox, default 0.8).
    Higher = more varied/expressive; ``0`` = deterministic greedy decoding."""

    top_p: Optional[float] = None
    """Nucleus (top-p) sampling cutoff for autoregressive engines (default 0.95)."""

    length_scale: Optional[float] = None
    """Phoneme length scale (< 1 is faster, > 1 is slower)."""

    noise_scale: Optional[float] = None
    """Amount of generator noise to add."""

    noise_w_scale: Optional[float] = None
    """Amount of phoneme width noise to add."""

    normalize_audio: bool = True
    """Enable/disable scaling audio samples to fit full range."""

    volume: float = 1.0
    """Multiplier for audio samples (< 1 is quieter, > 1 is louder)."""

    enable_phonetic_spellings: bool = True

    """for arabic and hebrew models. ``None`` (the default) defers to the voice
    config's own ``add_diacritics`` so a model that ships undiacritized (e.g.
    grapheme F5-TTS voices) is not force-diacritized by the caller."""
    add_diacritics: Optional[bool] = None

    # diacritizer model name (for languages that need one — e.g. Arabic uses text2tashkeel
    # models like "rawi-ensemble"). ``None`` (the default) defers to the voice config's
    # own ``diacritizer_model`` choice; an explicit value here overrides it.
    diacritizer_model: Optional[str] = None

    # post-synthesis audio super-resolution via ``audiosronnx`` (pure-ONNX). Off by
    # default; when ``True`` each synthesized chunk is upscaled to 48 kHz before being
    # yielded and the chunk's ``sample_rate`` reports the upscaled rate. ``audiosronnx``
    # is imported lazily and only when enabled; it ships in the ``[audiosr]`` extra.
    super_resolution: bool = False

    # super-resolution model name (an ``audiosronnx`` engine, e.g. ``"novasr"`` or
    # ``"lavasr"``). ``None`` (the default) selects ``"novasr"``. Ignored unless
    # ``super_resolution`` is True.
    super_resolution_model: Optional[str] = None

    # Engine-specific per-call params (d_factor, p_factor, e_factor, …)
    extra_params: Dict[str, Any] = field(default_factory=dict)


class UnsupportedVoiceLanguage(ValueError):
    """Raised at voice load when no phonemizer backend serves the voice's language.

    Surfacing this eagerly (instead of letting scriptconv's ``ValueError:
    unsupported language code`` bubble up mid-synthesis, after the user has
    already picked the voice and sent text) turns an opaque runtime crash
    into an actionable, typed failure at load time.
    """

    def __init__(self, voice: str, lang_code: str, phoneme_type: "PhonemeType"):
        self.voice = voice
        self.lang_code = lang_code
        self.phoneme_type = phoneme_type
        super().__init__(
            f"voice {voice!r}: no phonemizer backend supports lang "
            f"{lang_code!r} for phoneme_type {phoneme_type.value!r}"
        )


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
    so it is treated the same as "no get_lang": skip, don't reject. Backends
    without ``get_lang`` (grapheme/unicode passthroughs) accept any language
    and are silently skipped, as are unresolvable phoneme types
    (``get_phonemizer`` raises its own error for those).

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
    get_lang = getattr(phonemizer_cls, "get_lang", None)
    if get_lang is None:
        return
    try:
        get_lang(lang_code)
    except ImportError:
        return
    except ValueError as e:
        raise UnsupportedVoiceLanguage(voice, lang_code, phoneme_type) from e


def get_phonemizer(phoneme_type: PhonemeType,
                   alphabet: Alphabet = Alphabet.IPA,
                   model: Optional[str] = None) -> 'Phonemizer':
    """
    Create a phonemizer instance for the specified phonemeization strategy.

    Delegates to scriptconv's registry, injecting phoonnx's text normalizer
    (number/date expansion) so behavior matches the historical in-tree
    phonemizers exactly.  All backends, including the license-quarantined
    mantoq/KoG2P (vendored in scriptconv under their own licenses), come from
    scriptconv, which also auto-provisions the Hebrew phonikud model.

    Raises:
        ValueError: If the provided `phoneme_type` is not supported.
    """
    from phoonnx.util import normalize as _normalize
    phoneme_type = PhonemeType(phoneme_type)

    from scriptconv.phonemizers import get_phonemizer as _sc_get
    try:
        phonemizer = _sc_get(phoneme_type, alphabet=alphabet, model=model)
    except KeyError:
        raise ValueError(f"unsupported phoneme_type: {phoneme_type}")

    phonemizer.normalizer = _normalize
    return phonemizer


def get_conversion(phonemizer, voice_config: "VoiceConfig",
                   syn_config: "SynthesisConfig", tgt_alphabet: Alphabet):
    """Build a voice's ``text -> tgt_alphabet`` conversion for one synthesis call.

    Returns ``(graph, prepare_text)``:

    * ``graph`` — a scriptconv ``ConversionGraph`` whose phoneme edge is the
      voice's own (lazy) phonemizer. When the voice vocalizes, scriptconv's
      ``text -> text-diacritized`` edge is added and the direct edge omitted, so
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
    from scriptconv.graph import ConversionGraph, Edge
    from scriptconv.diacritics import DIACRITIZED, diacritize

    # explicit per-call setting wins, else the voice's own
    enabled = syn_config.add_diacritics
    if enabled is None:
        enabled = voice_config.add_diacritics
    model = syn_config.diacritizer_model or voice_config.diacritizer_model
    lang, alpha = voice_config.lang_code, tgt_alphabet.value

    # phonemize_lazy is a BasePhonemizer method, so every phonemizer has it: a
    # per-sentence generator, keeping sentence N+1 off the critical path of N.
    def phonemize(text, **_):
        return phonemizer.phonemize_lazy(text, lang)

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




if __name__ == "__main__":  # pragma: no cover
    config_files = [
        "/home/miro/PycharmProjects/phoonnx_tts/sabela_cotovia_vits.json",
        "/home/miro/PycharmProjects/phoonnx_tts/celtia_vits.json",
        "/home/miro/PycharmProjects/phoonnx_tts/mimic3_gruut.json",
        "/home/miro/PycharmProjects/phoonnx_tts/mimic3_espeak.json",
        "/home/miro/PycharmProjects/phoonnx_tts/mimic3_epitran.json",
        "/home/miro/PycharmProjects/phoonnx_tts/mimic3_symbols.json",
        "/home/miro/PycharmProjects/phoonnx_tts/piper_espeak.json",
        "/home/miro/PycharmProjects/phoonnx_tts/vits-coqui-pt-cv/config.json",
        "/home/miro/PycharmProjects/phoonnx_tts/phonikud/model.config.json"
    ]
    phoneme_txts = [
        None,
        None,
        "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/phonemes.txt",
        "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/phonemes.txt",
        "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/phonemes.txt",
        "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/phonemes.txt",
        None,
        None,
        None
    ]
    print("Testing model config file parsing\n###############")
    for idx, cfile in enumerate(config_files):
        print(f"\nConfig file: {cfile}")
        with open(cfile) as f:
            config = json.load(f)
        print("Mimic3:", VoiceConfig.is_mimic3(config))
        print("Piper:", VoiceConfig.is_piper(config))
        print("Coqui:", VoiceConfig.is_coqui_vits(config))
        print("Phoonx:", VoiceConfig.is_phoonnx(config))
        cfg = VoiceConfig.from_dict(config, phoneme_txts[idx])
        print(cfg)
