"""
Per-format ``config.json`` loaders.

A voice config can arrive in any of a dozen shapes: phoonnx's own native
format, a piper or mimic3 or Coqui checkpoint, a bare transformers vocabulary,
a sherpa ``tokens.txt``, or one of the codec-LM engines that ship no phoneme
vocabulary at all.  Each shape is handled by a loader that knows how to
recognise it (:meth:`ConfigLoader.detect`) and how to turn it into the handful
of fields that vary per format (:meth:`ConfigLoader.load`).

:func:`load_voice_fields` walks :data:`LOADERS` in order and returns the first
match, falling back to :class:`CanonicalLoader`.  **Order is significant** and
mirrors the historical detection ladder:

* the raw-text codec-LM loaders come first because they key off a declared
  engine name and must win before any shape-sniffing runs;
* ``phoonnx`` precedes ``piper`` because a native config that phonemizes with
  espeak carries a piper-shaped ``phoneme_id_map`` and would otherwise be
  claimed by the piper loader;
* ``vocab`` (transformers) and ``tokens_txt`` (sherpa) come last because they
  are decided by what the caller passed alongside the config, not by the config
  itself, and any real config shape should win over them.

A loader may consult the caller's values — a format's own defaults sometimes
depend on them — but it never decides the precedence between them and its own.
:func:`resolve_overrides` does that once, for every format, right after the
loader returns.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Type, Union

from scriptconv.phonemizers.enums import Alphabet, Phonemizer as PhonemeType

from phoonnx.config import Engine

from phoonnx.tokenizer import (TTSTokenizer, Vocabulary, BlankBetween,
                               DEFAULT_BLANK_WORD_TOKEN, DEFAULT_BLANK_TOKEN,
                               DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN)


@dataclass
class LoadRequest:
    """Everything a loader may look at.

    ``lang_code``, ``phoneme_type`` and ``alphabet`` arrive pre-merged: the
    caller's kwarg if it gave one, else the config's own value, so a loader
    that needs to branch on the effective value (piper's ``text`` models,
    mimic3's ``symbols`` models) sees what the caller asked for.  ``engine`` is
    the caller's kwarg alone — the config's ``engine`` key is a per-format
    concern and each loader decides what to do with it.
    """

    config: Dict[str, Any]
    vocab: Optional[Dict[str, Any]] = None
    tokenizer_config: Optional[Dict[str, Any]] = None
    tokens_txt: Optional[str] = None
    bpe_tokenizer_json: Optional[str] = None
    lang_code: Optional[str] = None
    phoneme_type: Optional[Union[str, PhonemeType]] = None
    alphabet: Optional[Union[str, Alphabet]] = None
    engine: Optional[Union[str, Any]] = None


@dataclass
class LoadedFields:
    """The format-specific half of a :class:`~phoonnx.config.VoiceConfig`.

    Every value may be ``None``; the dataclass' own ``__post_init__`` owns the
    final defaulting (alphabet, lang code, diacritics).  ``pinned`` names the
    fields the format fixes for correctness — see :func:`resolve_overrides`.
    """

    tokenizer: Optional[TTSTokenizer] = None
    engine: Optional[Any] = None
    lang_code: Optional[str] = None
    phoneme_type: Optional[Union[str, PhonemeType]] = None
    alphabet: Optional[Union[str, Alphabet]] = None
    add_diacritics: Optional[bool] = False
    diacritizer_model: Optional[str] = None
    blank_between: Union[str, BlankBetween] = BlankBetween.TOKENS_AND_WORDS
    pinned: FrozenSet[str] = field(default_factory=frozenset)


def resolve_overrides(loaded: LoadedFields, request: LoadRequest) -> LoadedFields:
    """Apply the caller's overrides to a loader's output, in place.

    The precedence, for ``lang_code``, ``phoneme_type``, ``alphabet`` and
    ``engine``:

    1. a field the loader **pinned** wins over everything — the format binds it
       to its tokenizer vocabulary, so honouring the caller would produce
       wrong token ids (piper's ``text``/``pygoruut`` models, mimic3's
       ``symbols`` models and its ``text_language``, a transformers
       ``tokenizer_config`` language, a codec-LM engine named by the config);
    2. otherwise the caller's kwarg wins;
    3. otherwise the loader's value, which already folded in the config's own.

    Every step is null-safe by ``or``: a config carrying an explicit ``null``
    falls through to the next source rather than pinning ``None``.
    ``engine`` additionally defaults to :data:`~phoonnx.config.Engine.PHOONNX`
    when nothing supplies one.
    """
    if "lang_code" not in loaded.pinned:
        loaded.lang_code = request.lang_code or loaded.lang_code
    if "phoneme_type" not in loaded.pinned:
        loaded.phoneme_type = request.phoneme_type or loaded.phoneme_type
    if "alphabet" not in loaded.pinned:
        loaded.alphabet = request.alphabet or loaded.alphabet
    if "engine" not in loaded.pinned:
        loaded.engine = request.engine or loaded.engine or Engine.PHOONNX
    return loaded


class ConfigLoader:
    """Base class for a per-format loader."""

    @classmethod
    def detect(cls, request: LoadRequest) -> bool:
        """Whether this loader owns *request*.

        Takes the whole request rather than just the config because two
        formats (transformers, sherpa) are identified by the companion files
        the caller passed, not by the config's own keys.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, request: LoadRequest) -> LoadedFields:
        """Build the format-specific fields, without applying caller overrides.

        May mutate ``request.config`` — several formats normalise their special
        tokens or fold a nested section into the top level, and the values are
        read back out of the config when the VoiceConfig is assembled.

        Raises ``ValueError`` when the format needs a companion file the caller
        did not pass.
        """
        raise NotImplementedError


def _adapter_owned_tokenizer() -> TTSTokenizer:
    """An empty tokenizer, for engines whose adapter owns the text -> id path."""
    return TTSTokenizer(
        Vocabulary(char2idx={}, pad=DEFAULT_PAD_TOKEN),
        add_blank_char=False,
        add_blank_word=False,
        use_eos_bos=False,
        blank_at_end=False,
        blank_at_start=False,
    )


class RawTextLoader(ConfigLoader):
    """Codec-LM engines that consume raw text.

    The adapter owns the whole text -> id path (prompt assembly plus the
    checkpoint's own subword tokenizer, loaded from ``engine_params``), so no
    phoneme vocabulary is built here and ``Alphabet.GRAPHEMES`` routes
    ``TTSVoice.synthesize()`` through ``adapter.encode_text()``.  Subclasses
    supply only the engine, its codec's sample rate, and — where the adapter
    wants codepoints rather than words — a different alphabet.

    These loaders pin ``engine``: they are selected *by* the engine name, so
    the name is the identity of the branch, not a preference.
    """

    ENGINE: Any = None
    SAMPLE_RATE: int = 24000
    ALPHABET: Alphabet = Alphabet.GRAPHEMES

    @classmethod
    def detect(cls, request: LoadRequest) -> bool:
        name = cls.ENGINE.value
        return request.engine == name or request.config.get("engine") == name

    @classmethod
    def load(cls, request: LoadRequest) -> LoadedFields:
        request.config.setdefault("audio", {}).setdefault("sample_rate", cls.SAMPLE_RATE)
        return LoadedFields(
            tokenizer=_adapter_owned_tokenizer(),
            engine=cls.ENGINE,
            lang_code=request.lang_code,
            phoneme_type=request.phoneme_type or PhonemeType.UNICODE,
            alphabet=request.alphabet or cls.ALPHABET,
            pinned=frozenset({"engine"}),
        )


class ChatterboxLoader(RawTextLoader):
    """Chatterbox: raw text through its own BPE, loaded from ``tokenizer.json``.

    The multilingual variant uses ChatterboxMTLTokenizer, detected by the
    ``[SPACE]`` token.  Unlike the other codec-LM engines it keeps
    ``Alphabet.UNICODE``, and it vocalizes Arabic and Hebrew, which need
    niqqud/tashkeel restored for a correct pronunciation.
    """

    ENGINE = Engine.CHATTERBOX
    SAMPLE_RATE = 24000
    ALPHABET = Alphabet.UNICODE

    @classmethod
    def load(cls, request: LoadRequest) -> LoadedFields:
        if not request.bpe_tokenizer_json:
            raise ValueError("Chatterbox voices require a tokenizer.json (bpe_tokenizer_json)")
        from phoonnx.tokenizer import load_chatterbox_tokenizer
        tokenizer = load_chatterbox_tokenizer(request.bpe_tokenizer_json)
        config = request.config
        config.setdefault("audio", {}).setdefault("sample_rate", cls.SAMPLE_RATE)
        config.setdefault("num_symbols", tokenizer._tok.get_vocab_size())
        return LoadedFields(
            tokenizer=tokenizer,
            engine=Engine.CHATTERBOX,
            lang_code=request.lang_code,
            phoneme_type=request.phoneme_type or PhonemeType.UNICODE,
            alphabet=request.alphabet or cls.ALPHABET,
            add_diacritics=(request.lang_code or "").lower().startswith(("ar", "he")),
            pinned=frozenset({"engine"}),
        )


class PhoonnxLoader(ConfigLoader):
    """phoonnx's own native format."""

    @classmethod
    def detect(cls, request: LoadRequest) -> bool:
        return "phoonnx_version" in request.config

    @classmethod
    def load(cls, request: LoadRequest) -> LoadedFields:
        config = request.config
        inference = config.get("inference", {})
        # before any config write: an unknown alphabet raises, and the caller's
        # dict must come back untouched when it does
        alphabet = request.alphabet or Alphabet(config.get("alphabet", "ipa"))

        # Preserve the model's own special tokens when present (a native
        # config may use any pad/blank/bos/eos); fall back to phoonnx defaults.
        config["pad"] = config.get("pad") or DEFAULT_PAD_TOKEN
        config["blank"] = config.get("blank") or DEFAULT_BLANK_TOKEN
        config["bos"] = config.get("bos") or DEFAULT_BOS_TOKEN
        config["eos"] = config.get("eos") or DEFAULT_EOS_TOKEN

        return LoadedFields(
            tokenizer=TTSTokenizer.from_phoonnx_config(config),
            engine=config.get("engine"),
            lang_code=request.lang_code or config.get("lang_code"),
            phoneme_type=request.phoneme_type or config.get("phoneme_type", PhonemeType.ESPEAK),
            alphabet=alphabet,
            add_diacritics=inference.get("add_diacritics", True),
            diacritizer_model=inference.get("diacritizer_model", None),
        )


class PiperLoader(ConfigLoader):
    """piper checkpoints.

    ``text`` and ``pygoruut`` models pin both the phoneme type and the
    alphabet: their vocabulary is the model's own character or goruut token
    set, so a caller-supplied alphabet would mis-tokenise.
    """

    @classmethod
    def detect(cls, request: LoadRequest) -> bool:
        config = request.config
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

        return config["phoneme_type"] in ("text", "espeak")

    @classmethod
    def load(cls, request: LoadRequest) -> LoadedFields:
        config = request.config

        lang_code = request.lang_code or (config.get("language", {}).get("code") or
                                          config.get("espeak", {}).get("voice"))
        phoneme_type = request.phoneme_type or config.get("phoneme_type", PhonemeType.ESPEAK)
        alphabet = request.alphabet
        pinned = set()
        if phoneme_type == "text":
            phoneme_type, alphabet = PhonemeType.UNICODE, Alphabet.UNICODE
            pinned = {"phoneme_type", "alphabet"}
        elif phoneme_type == "pygoruut":
            # special case: neurlang models
            phoneme_type, alphabet = PhonemeType.GORUUT, Alphabet.IPA
            pinned = {"phoneme_type", "alphabet"}
        else:
            alphabet = alphabet or Alphabet.IPA

        # not configurable in piper
        config["pad"] = DEFAULT_PAD_TOKEN
        config["blank"] = DEFAULT_BLANK_TOKEN
        config["blank_word"] = DEFAULT_BLANK_WORD_TOKEN
        config["bos"] = DEFAULT_BOS_TOKEN
        config["eos"] = DEFAULT_EOS_TOKEN

        return LoadedFields(
            tokenizer=TTSTokenizer.from_piper_config(config),
            engine=Engine.PIPER,
            lang_code=lang_code,
            phoneme_type=phoneme_type,
            alphabet=alphabet,
            add_diacritics=(lang_code or "").startswith("ar"),
            pinned=frozenset(pinned),
        )


class Mimic3Loader(ConfigLoader):
    """mimic3 voices (https://huggingface.co/mukowaty/mimic3-voices).

    The language always comes from the config's ``text_language`` — mimic3
    gruut vocabularies are language-specific, so it is pinned.  ``symbols``
    models are grapheme models whose symbol map lives in the tokens file, so
    they pin the phoneme type and alphabet too.
    """

    @classmethod
    def detect(cls, request: LoadRequest) -> bool:
        config = request.config
        # mimic3 models indicate a phonemizer strategy in their config
        if ("phonemizer" not in config or
                not isinstance(config["phonemizer"], str)):
            return False

        # mimic3 models include a "phonemes" section with token info
        if "phonemes" not in config or not isinstance(config["phonemes"], dict):
            return False

        return config["phonemizer"] in ("symbols", "gruut", "espeak", "epitran")

    @classmethod
    def load(cls, request: LoadRequest) -> LoadedFields:
        config = request.config
        if not request.tokens_txt:
            raise ValueError("mimic3 models require an external phonemes.txt file in addition to the config")

        phoneme_type = request.phoneme_type or config.get("phonemizer", PhonemeType.GRUUT)
        phoneme_cfg = config.get("phonemes", {})
        blank_between = BlankBetween(phoneme_cfg.get("blank_between", "tokens_and_words"))
        config.update(phoneme_cfg)

        pinned = {"lang_code"}
        if phoneme_type == "symbols":
            phoneme_type, alphabet = PhonemeType.GRAPHEMES, Alphabet.UNICODE
            pinned |= {"phoneme_type", "alphabet"}
        else:
            alphabet = request.alphabet or Alphabet.IPA

        return LoadedFields(
            tokenizer=TTSTokenizer.from_mimic3_config(config, request.tokens_txt),
            engine=Engine.MIMIC3,
            lang_code=config.get("text_language"),
            phoneme_type=phoneme_type,
            alphabet=alphabet,
            blank_between=blank_between,
            pinned=frozenset(pinned),
        )


class CoquiLoader(ConfigLoader):
    """Coqui VITS grapheme models."""

    @classmethod
    def detect(cls, request: LoadRequest) -> bool:
        characters = request.config.get("characters")
        if not isinstance(characters, dict):
            return False
        # double check this was trained with coqui
        return characters.get("characters_class", "") in ("TTS.tts.models.vits.VitsCharacters",
                                                          "TTS.tts.utils.text.characters.Graphemes")

    @classmethod
    def load(cls, request: LoadRequest) -> LoadedFields:
        config = request.config
        # NOTE: lang code usually not provided and often wrong :(
        ds = config.get("datasets", [])
        lang_code = request.lang_code
        if ds and not lang_code:
            lang_code = ds[0].get("language")

        return LoadedFields(
            tokenizer=TTSTokenizer.from_coqui_config(config),
            engine=Engine.COQUI,
            lang_code=lang_code,
            phoneme_type=request.phoneme_type or PhonemeType.GRAPHEMES,
            alphabet=request.alphabet or Alphabet.UNICODE,
        )


class TransformersLoader(ConfigLoader):
    """Models exported from transformers, identified by a companion ``vocab``.

    A ``tokenizer_config`` names the language the vocabulary was built for,
    which pins ``lang_code``.
    """

    @classmethod
    def detect(cls, request: LoadRequest) -> bool:
        return bool(request.vocab)

    @classmethod
    def load(cls, request: LoadRequest) -> LoadedFields:
        config, tok_cfg = request.config, request.tokenizer_config
        add_blank = True
        lang_code = request.lang_code
        pinned = set()
        if tok_cfg:
            add_blank = tok_cfg.get("add_blank", add_blank)
            if tok_cfg.get("language"):
                lang_code = tok_cfg["language"]
                pinned = {"lang_code"}
            config["blank"] = tok_cfg.get("pad_token", config.get("blank", DEFAULT_BLANK_TOKEN))

        tokenizer = TTSTokenizer(
            Vocabulary(char2idx=request.vocab, blank=config.get("blank", DEFAULT_BLANK_TOKEN)),
            add_blank_char=add_blank,
            add_blank_word=False,
            use_eos_bos=False,
            blank_at_end=add_blank,
            blank_at_start=add_blank,
        )
        return LoadedFields(
            tokenizer=tokenizer,
            lang_code=lang_code,
            phoneme_type=request.phoneme_type,
            alphabet=request.alphabet,
            pinned=frozenset(pinned),
        )


class TokensTxtLoader(ConfigLoader):
    """sherpa-onnx style models that ship only a tokens file."""

    @classmethod
    def detect(cls, request: LoadRequest) -> bool:
        return bool(request.tokens_txt)

    @classmethod
    def load(cls, request: LoadRequest) -> LoadedFields:
        path = request.tokens_txt
        with open(path, "r", encoding="utf-8") as ids_file:
            if path.endswith(".txt"):
                # mimic3 / MMS / sherpa
                vocabulary = Vocabulary.from_tokens_txt(ids_file.read())
            elif path.endswith(".json"):
                vocabulary = Vocabulary(char2idx=json.load(ids_file), pad=request.config["pad"])
            else:
                raise ValueError(f"unsupported tokens file (expected .txt or .json): {path}")

        tokenizer = TTSTokenizer(
            vocabulary,
            add_blank_char=True,
            add_blank_word=False,
            use_eos_bos=True,
            blank_at_end=True,
            blank_at_start=True,
        )
        return LoadedFields(
            tokenizer=tokenizer,
            lang_code=request.lang_code,
            phoneme_type=request.phoneme_type,
            alphabet=request.alphabet,
        )


class CanonicalLoader(ConfigLoader):
    """The fallback: a config that ships its own ``phoneme_id_map``.

    Config-shape engine detection is deprecated, so a config that declares its
    engine, phoneme type and alphabet is honoured as-is rather than guessed at
    or rejected.  Unlike the native phoonnx format it does not assume espeak
    phonemes or diacritization.
    """

    @classmethod
    def detect(cls, request: LoadRequest) -> bool:
        return True

    @classmethod
    def load(cls, request: LoadRequest) -> LoadedFields:
        config = request.config
        inference = config.get("inference", {})
        # before any config write: an unknown alphabet raises, and the caller's
        # dict must come back untouched when it does
        alphabet = request.alphabet or Alphabet(config.get("alphabet", "ipa"))

        config["pad"] = config.get("pad") or DEFAULT_PAD_TOKEN
        config["blank"] = config.get("blank") or DEFAULT_BLANK_TOKEN
        config["bos"] = config.get("bos") or DEFAULT_BOS_TOKEN
        config["eos"] = config.get("eos") or DEFAULT_EOS_TOKEN

        return LoadedFields(
            tokenizer=TTSTokenizer.from_phoonnx_config(config),
            engine=config.get("engine"),
            lang_code=request.lang_code,
            phoneme_type=request.phoneme_type or config.get("phoneme_type", PhonemeType.GRAPHEMES),
            alphabet=alphabet,
            add_diacritics=inference.get("add_diacritics", False),
            diacritizer_model=inference.get("diacritizer_model", None),
        )


def _raw_text_loader(engine: Engine, sample_rate: int) -> Type[RawTextLoader]:
    """A :class:`RawTextLoader` for an engine that differs only in sample rate."""
    return type(f"{engine.name.title()}Loader", (RawTextLoader,),
                {"ENGINE": engine, "SAMPLE_RATE": sample_rate,
                 "__doc__": RawTextLoader.__doc__})


LOADERS: List[Type[ConfigLoader]] = [
    ChatterboxLoader,
    _raw_text_loader(Engine.LLASA, 16000),
    _raw_text_loader(Engine.NEUTTS, 24000),
    _raw_text_loader(Engine.ORPHEUS, 24000),
    _raw_text_loader(Engine.MOSSTTS, 48000),
    _raw_text_loader(Engine.SUPERTONIC, 44100),
    _raw_text_loader(Engine.POCKETTTS, 24000),
    _raw_text_loader(Engine.SPARKTTS, 16000),
    _raw_text_loader(Engine.INDIC_PARLER, 44100),
    _raw_text_loader(Engine.OMNIVOICE, 24000),
    _raw_text_loader(Engine.MAGPIE, 22050),
    _raw_text_loader(Engine.ARKTTS, 44100),
    _raw_text_loader(Engine.QWEN3TTS, 24000),
    _raw_text_loader(Engine.OUTETTS, 24000),
    PhoonnxLoader,
    PiperLoader,
    Mimic3Loader,
    CoquiLoader,
    TransformersLoader,
    TokensTxtLoader,
    CanonicalLoader,
]


def load_voice_fields(request: LoadRequest) -> LoadedFields:
    """Detect *request*'s format, load it, and apply the caller's overrides."""
    for loader in LOADERS:
        if loader.detect(request):
            return resolve_overrides(loader.load(request), request)
    raise RuntimeError("no config loader matched")  # pragma: no cover
