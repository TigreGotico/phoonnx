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

        if self.add_diacritics is None:
            self.add_diacritics = False
            if self.lang_code and self.lang_code.startswith("ar"):
                self.add_diacritics = True

        self.lang_code = normalize_lang(self.lang_code or "und")

    @staticmethod
    def is_mimic3(config: dict[str, Any]) -> bool:
        # https://huggingface.co/mukowaty/mimic3-voices

        # mimic3 models indicate a phonemizer strategy in their config
        if ("phonemizer" not in config or
                not isinstance(config["phonemizer"], str)):
            return False

        # mimic3 models include a "phonemes" section with token info
        if "phonemes" not in config or not isinstance(config["phonemes"], dict):
            return False

        # validate phonemizer type as expected by mimic3
        phonemizer = config["phonemizer"]
        # class Phonemizer(str, Enum):
        #     SYMBOLS = "symbols"
        #     GRUUT = "gruut"
        #     ESPEAK = "espeak"
        #     EPITRAN = "epitran"
        if phonemizer not in ["symbols", "gruut", "espeak", "epitran"]:
            return False

        return True

    @staticmethod
    def is_piper(config: dict[str, Any]) -> bool:
        if "piper_version" in config:
            return True
        # an explicitly declared non-piper engine wins over shape-sniffing: a
        # canonical config (e.g. engine="coqui" with an espeak phoneme_id_map)
        # must not be mistaken for piper just because it phonemizes with espeak.
        engine = config.get("engine")
        if engine and engine not in ("piper", Engine.PIPER, Engine.PIPER.value):
            return False
        # piper models indicate a phonemizer strategy in their config
        if ("phoneme_type" not in config or
                not isinstance(config["phoneme_type"], str)):
            return False

        # piper models include a "phoneme_id_map" section mapping phonemes to int
        pid = config.get("phoneme_id_map")
        if not isinstance(pid, dict) or not pid:
            return False

        # piper's phoneme_id_map values are *lists* ([id, ...]); a flat
        # phoneme -> int map is the canonical phoonnx/coqui shape, not piper.
        if not all(isinstance(v, (list, tuple)) for v in pid.values()):
            return False

        # validate phonemizer type as expected by piper
        phonemizer = config["phoneme_type"]
        if phonemizer not in ["text", "espeak"]:
            return False

        return True

    @staticmethod
    def is_coqui_vits(config: dict[str, Any]) -> bool:
        # coqui vits grapheme models include a "characters" section with token info
        if "characters" not in config or not isinstance(config["characters"], dict):
            return False

        # double check this was trained with coqui
        if config["characters"].get("characters_class", "") not in ["TTS.tts.models.vits.VitsCharacters",
                                                                    "TTS.tts.utils.text.characters.Graphemes"]:
            return False

        return True

    @staticmethod
    def is_phoonnx(config: dict[str, Any]) -> bool:
        return "phoonnx_version" in config

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
        Create a VoiceConfig from a model configuration dictionary and optional external phoneme data.
        
        Builds a VoiceConfig by detecting the model engine (Phoonnx, Piper, Mimic3, Transformers or Coqui), deriving tokenizer and alphabet, and applying model-specific defaults and inference settings. Provided optional arguments override corresponding values found in the config.
        
        Parameters:
            config (dict[str, Any]): Parsed model configuration dictionary.
            tokens_txt (Optional[str]): Path to an external tokens file (.txt or .json) used to build or override the tokenizer vocabulary.
            lang_code (Optional[str]): Language code to override the config's language selection.
            phoneme_type (Optional[PhonemeType]): Phoneme type name to override the config's phoneme_type value.
            alphabet (Optional[Alphabet]): Alphabet name to override or supply the resulting VoiceConfig alphabet.
        
        Returns:
            VoiceConfig: A populated VoiceConfig instance with tokenizer, alphabet, engine, phoneme_type, inference settings, and token tokens derived from the inputs.
        
        Raises:
            ValueError: If the config is identified as a Mimic3 model but no phonemes_txt is provided.
        """
        blank_type = BlankBetween.TOKENS_AND_WORDS
        lang_code = lang_code or config.get("lang_code")
        phoneme_type = phoneme_type or config.get("phoneme_type")
        alphabet = alphabet or config.get("alphabet")
        diacritics = False
        ar_diacritizer_model = None

        if (engine == Engine.CHATTERBOX or
                (isinstance(engine, str) and engine == "chatterbox") or
                config.get("engine") == "chatterbox"):
            # Chatterbox tokenizes raw text with its own BPE; the multilingual variant
            # uses ChatterboxMTLTokenizer (detected by the [SPACE] token).
            engine = Engine.CHATTERBOX
            if not bpe_tokenizer_json:
                raise ValueError("Chatterbox voices require a tokenizer.json (bpe_tokenizer_json)")
            from phoonnx.tokenizer import load_chatterbox_tokenizer
            tokenizer = load_chatterbox_tokenizer(bpe_tokenizer_json)
            phoneme_type = phoneme_type or PhonemeType.UNICODE
            alphabet = alphabet or Alphabet.UNICODE
            # Arabic/Hebrew need vocalization (niqqud/tashkeel) for correct pronunciation.
            diacritics = (lang_code or "").lower().startswith(("ar", "he"))
            config.setdefault("audio", {}).setdefault("sample_rate", 24000)
            config.setdefault("num_symbols", tokenizer._tok.get_vocab_size())

        elif (engine == Engine.NEUTTS or
                (isinstance(engine, str) and engine == "neutts") or
                config.get("engine") == "neutts"):
            # NeuTTS consumes raw text: the adapter owns the whole text -> id path
            # (espeak-ng phonemization, prompt assembly, then the checkpoint's own BPE,
            # all loaded from engine_params), so no phonemizer/tokenizer vocabulary is
            # built here. Alphabet.GRAPHEMES routes TTSVoice.synthesize() through
            # adapter.encode_text(). NeuCodec output is 24 kHz mono.
            engine = Engine.NEUTTS
            phoneme_type = phoneme_type or PhonemeType.UNICODE
            alphabet = alphabet or Alphabet.GRAPHEMES
            config.setdefault("audio", {}).setdefault("sample_rate", 24000)
            tokenizer = TTSTokenizer(
                Vocabulary(char2idx={}, pad=DEFAULT_PAD_TOKEN),
                add_blank_char=False,
                add_blank_word=False,
                use_eos_bos=False,
                blank_at_end=False,
                blank_at_start=False,
            )

        elif (engine == Engine.SUPERTONIC or
                (isinstance(engine, str) and engine == "supertonic") or
                config.get("engine") == "supertonic"):
            # SuperTonic consumes raw text directly: the adapter owns text -> id
            # conversion via its own unicode-codepoint indexer (loaded from
            # engine_params), so no phonemizer/tokenizer vocabulary is needed here.
            # Alphabet.GRAPHEMES routes TTSVoice.synthesize() through
            # adapter.encode_text() instead of the phonemizer pipeline.
            engine = Engine.SUPERTONIC
            phoneme_type = phoneme_type or PhonemeType.UNICODE
            alphabet = alphabet or Alphabet.GRAPHEMES
            config.setdefault("audio", {}).setdefault("sample_rate", 44100)
            tokenizer = TTSTokenizer(
                Vocabulary(char2idx={}, pad=DEFAULT_PAD_TOKEN),
                add_blank_char=False,
                add_blank_word=False,
                use_eos_bos=False,
                blank_at_end=False,
                blank_at_start=False,
            )

        elif (engine == Engine.POCKETTTS or
                (isinstance(engine, str) and engine == "pockettts") or
                config.get("engine") == "pockettts"):
            # Pocket TTS consumes raw text directly: the adapter tokenizes with the
            # bundle's own SentencePiece model (loaded from engine_params), so no
            # phonemizer or vocabulary is needed here. Alphabet.GRAPHEMES routes
            # TTSVoice.synthesize() through adapter.encode_text().
            engine = Engine.POCKETTTS
            phoneme_type = phoneme_type or PhonemeType.UNICODE
            alphabet = alphabet or Alphabet.GRAPHEMES
            config.setdefault("audio", {}).setdefault("sample_rate", 24000)
            tokenizer = TTSTokenizer(
                Vocabulary(char2idx={}, pad=DEFAULT_PAD_TOKEN),
                add_blank_char=False,
                add_blank_word=False,
                use_eos_bos=False,
                blank_at_end=False,
                blank_at_start=False,
            )

        elif VoiceConfig.is_phoonnx(config):
            engine = engine or config.get("engine") or Engine.PHOONNX

            lang_code = lang_code or config.get("lang_code")
            phoneme_type = phoneme_type or config.get("phoneme_type", PhonemeType.ESPEAK)
            alphabet = alphabet or Alphabet(config.get("alphabet", "ipa"))
            diacritics = config.get("inference", {}).get("add_diacritics", True)
            ar_diacritizer_model = config.get("inference", {}).get("diacritizer_model", None)

            # Preserve the model's own special tokens when present (a native
            # config may use any pad/blank/bos/eos); fall back to phoonnx defaults.
            config["pad"] = config.get("pad") or DEFAULT_PAD_TOKEN
            config["blank"] = config.get("blank") or DEFAULT_BLANK_TOKEN
            config["bos"] = config.get("bos") or DEFAULT_BOS_TOKEN
            config["eos"] = config.get("eos") or DEFAULT_EOS_TOKEN

            tokenizer = TTSTokenizer.from_phoonnx_config(config)

        # check if model was trained for PiperTTS
        elif VoiceConfig.is_piper(config):
            engine =  engine or Engine.PIPER

            lang_code = lang_code or (config.get("language", {}).get("code") or
                         config.get("espeak", {}).get("voice"))
            diacritics = (lang_code or "").startswith("ar")
            phoneme_type = phoneme_type or config.get("phoneme_type", PhonemeType.ESPEAK)
            if phoneme_type == "text":
                phoneme_type = PhonemeType.UNICODE
                alphabet = Alphabet.UNICODE
            elif phoneme_type == "pygoruut":
                # special case: neurlang models
                phoneme_type = PhonemeType.GORUUT
                alphabet = Alphabet.IPA
            else:
                alphabet = alphabet or Alphabet.IPA

            # not configurable in piper
            config["pad"] =  DEFAULT_PAD_TOKEN
            config["blank"] = DEFAULT_BLANK_TOKEN
            config["blank_word"] = DEFAULT_BLANK_WORD_TOKEN
            config["bos"] = DEFAULT_BOS_TOKEN
            config["eos"] = DEFAULT_EOS_TOKEN

            tokenizer = TTSTokenizer.from_piper_config(config)

        # check if model was trained for Mimic3
        elif VoiceConfig.is_mimic3(config):
            engine =  engine or Engine.MIMIC3

            if not tokens_txt:
                raise ValueError("mimic3 models require an external phonemes.txt file in addition to the config")
            lang_code = config.get("text_language")
            phoneme_type = phoneme_type or config.get("phonemizer", PhonemeType.GRUUT)
            # read phoneme settings
            phoneme_cfg = config.get("phonemes", {})
            blank_type = BlankBetween(phoneme_cfg.get("blank_between", "tokens_and_words"))
            config.update(phoneme_cfg)

            if phoneme_type == "symbols":
                # Mimic3 "symbols" models are grapheme models
                # symbol map comes from phonemes_txt
                phoneme_type = PhonemeType.GRAPHEMES
                alphabet = Alphabet.UNICODE
            else:
                alphabet = alphabet or Alphabet.IPA

            tokenizer = TTSTokenizer.from_mimic3_config(config, tokens_txt)

        # check if model was trained with Coqui
        elif VoiceConfig.is_coqui_vits(config):
            engine =  engine or Engine.COQUI
            phoneme_type = phoneme_type or PhonemeType.GRAPHEMES
            alphabet = alphabet or Alphabet.UNICODE

            # NOTE: lang code usually not provided and often wrong :(
            ds = config.get("datasets", [])
            if ds and not lang_code:
                lang_code = ds[0].get("language")

            tokenizer = TTSTokenizer.from_coqui_config(config)
        # for models trained with transformers
        elif vocab:
            add_blank = True
            if tokenizer_config:
                add_blank = tokenizer_config.get("add_blank", add_blank)
                lang_code = tokenizer_config.get("language", lang_code)
                config["blank"] = tokenizer_config.get("pad_token", config.get("blank", DEFAULT_BLANK_TOKEN))

            tokenizer = TTSTokenizer(
                Vocabulary(char2idx=vocab, blank=config.get("blank", DEFAULT_BLANK_TOKEN)),
                add_blank_char=add_blank,
                add_blank_word=False,
                use_eos_bos=False,
                blank_at_end=add_blank,
                blank_at_start=add_blank
            )

        # for sherpa-onnx style models with tokens.txt only
        elif tokens_txt:
            if tokens_txt.endswith(".txt"):
                # mimic3 / MMS / sherpa
                with open(tokens_txt, "r", encoding="utf-8") as ids_file:
                    tokenizer = TTSTokenizer(
                        Vocabulary.from_tokens_txt(ids_file.read()),
                        add_blank_char=True,
                        add_blank_word=False,
                        use_eos_bos=True,
                        blank_at_end=True,
                        blank_at_start=True
                    )

            elif tokens_txt.endswith(".json"):
                with open(tokens_txt, "r", encoding="utf-8") as ids_file:
                    tokenizer = TTSTokenizer(
                        Vocabulary(char2idx=json.load(ids_file), pad=config["pad"]),
                        add_blank_char=True,
                        add_blank_word=False,
                        use_eos_bos=True,
                        blank_at_end=True,
                        blank_at_start=True
                    )

        # Canonical config: the model ships its own phoneme_id_map (and declares
        # engine / phoneme_type / alphabet). Use those values directly --
        # config-shape engine detection is deprecated, so honour what the config
        # declares instead of guessing or rejecting it.
        else:
            engine = engine or config.get("engine") or Engine.PHOONNX
            phoneme_type = phoneme_type or config.get("phoneme_type", PhonemeType.GRAPHEMES)
            alphabet = alphabet or Alphabet(config.get("alphabet", "ipa"))
            diacritics = config.get("inference", {}).get("add_diacritics", False)
            ar_diacritizer_model = config.get("inference", {}).get("diacritizer_model", ar_diacritizer_model)
            config["pad"] = config.get("pad") or DEFAULT_PAD_TOKEN
            config["blank"] = config.get("blank") or DEFAULT_BLANK_TOKEN
            config["bos"] = config.get("bos") or DEFAULT_BOS_TOKEN
            config["eos"] = config.get("eos") or DEFAULT_EOS_TOKEN
            tokenizer = TTSTokenizer.from_phoonnx_config(config)
        phoneme_type = PhonemeType(phoneme_type) if isinstance(phoneme_type, str) else phoneme_type
        LOG.debug(f"phonemizer: {phoneme_type}")
        inference = config.get("inference", {})
        engine = engine or Engine.PHOONNX
        return VoiceConfig(
            tokenizer=tokenizer,
            num_langs=config.get("num_langs", 1),
            num_symbols=config.get("num_symbols", 256),
            num_speakers=config.get("num_speakers", 1),
            sample_rate=config.get("audio", {}).get("sample_rate", 16000),
            noise_scale=inference.get("noise_scale", DEFAULT_NOISE_SCALE),
            length_scale=inference.get("length_scale", DEFAULT_LENGTH_SCALE),
            noise_w_scale=inference.get("noise_w", DEFAULT_NOISE_W_SCALE),
            add_diacritics=diacritics,
            diacritizer_model=ar_diacritizer_model,
            hop_length=config.get("hop_length", DEFAULT_HOP_LENGTH),
            lang_code=lang_code,
            alphabet=Alphabet(alphabet) if isinstance(alphabet, str) else alphabet,
            engine=Engine(engine) if isinstance(engine, str) else engine,
            phonemizer_model=config.get("phonemizer_model"),
            phoneme_type=PhonemeType(phoneme_type) if isinstance(phoneme_type, str) else phoneme_type,
            speaker_id_map=config.get("speaker_id_map", {}),
            blank_between=BlankBetween(blank_type) if isinstance(blank_type, str) else blank_type,
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
            "phoneme_type": self.phoneme_type.value if self.phoneme_type else "graphemes",
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
        return diacritize(text, lang, diacritizer_model=model)

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
