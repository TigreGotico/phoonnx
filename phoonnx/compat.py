"""
Conversion of third-party voice configs to the native phoonnx ``config.json``.

phoonnx reads one format, its own: ``phoneme_id_map``, the special tokens and the
text keys of :meth:`phoonnx.config.VoiceConfig.to_dict`. A voice can arrive in any
of a dozen other shapes: a piper or mimic3 or Coqui checkpoint, a bare transformers
vocabulary, a sherpa ``tokens.txt``, or one of the codec-LM engines that ship no
phoneme vocabulary at all. :func:`convert_config` turns each of them into a native
config once. :func:`~phoonnx.config.VoiceConfig.from_dict` still loads a third-party config
through these loaders, with a ``DeprecationWarning``.

Each shape is handled by a loader that knows how to recognise it
(:meth:`ConfigLoader.detect`) and how to turn it into the handful of fields that
vary per format (:meth:`ConfigLoader.load`). The tokenizer constructors of the
third-party formats (:func:`from_piper_config` and the others) live here too.

:func:`load_voice_fields` walks :data:`LOADERS` in order and returns the first
match, falling back to :class:`CanonicalLoader`.  **Order is significant** and
mirrors the historical detection ladder:

* the raw-text codec-LM loaders come first because they key off a declared
  engine name and must win before any shape-sniffing runs;
* ``phoonnx`` precedes ``piper`` because a native config that phonemizes with
  espeak carries a piper-shaped ``phoneme_id_map`` and would otherwise be
  claimed by the piper loader;
* ``vosk`` precedes ``piper``, but for a different reason than ``phoonnx``
  does: :class:`PiperLoader` already rejects a config with no ``phoneme_type``
  (unless it carries ``piper_version``), and a genuine vosk config has
  neither. Ordering only guards the engine-declared branch — a config naming
  ``engine: vosk`` that also happens to carry ``piper_version`` — from being
  claimed by :class:`PiperLoader`'s unconditional ``piper_version`` check
  first;
* ``vocab`` (transformers) and ``tokens_txt`` (sherpa) come last because they
  are decided by what the caller passed alongside the config, not by the config
  itself, and any real config shape should win over them.

A loader may consult the caller's values — a format's own defaults sometimes
depend on them — but it never decides the precedence between them and its own.
:func:`resolve_overrides` does that once, for every format, right after the
loader returns.
"""
import copy
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Type, Union

from scriptconv.phonemizers.enums import Alphabet, Phonemizer as PhonemeType

from phoonnx.config import Engine

from phoonnx.tokenizer import (TTSTokenizer, Vocabulary, BlankBetween,
                               DEFAULT_BLANK_WORD_TOKEN, DEFAULT_BLANK_TOKEN,
                               DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN)
from phoonnx.util import LOG, is_arabic


def vocabulary_from_piper_config(cfg: Dict[str, Any], decompose: bool = False) -> 'Vocabulary':
    """
    Creates a Vocabulary instance from a Piper configuration dictionary.

    The Piper config format assumes `phoneme_id_map` values are lists where the ID is the first element.

    Parameters:
        cfg: The Piper configuration dictionary.
        decompose: Give a precomposed symbol the ids of its canonical
            decomposition (NFD), as Piper does for the models it trains.

    Returns:
        A Vocabulary instance.
    """
    # Piper format has value as list of [id, ...]
    char2idx: Dict[str, Union[int, List[int]]] = {
        char: idx[0] for char, idx in cfg.get("phoneme_id_map", {}).items()}
    if decompose:
        # Piper decomposes the espeak-ng phonemes (NFD) before the id lookup, so
        # "ç" reaches the model as "c" and U+0327 and its own id is never trained
        # (piper1-gpl src/piper/phonemize_espeak.py at v1.5.0, and
        # src/phonemize.cpp of rhasspy/piper-phonemize for the older voices).
        for char in list(char2idx):
            parts = unicodedata.normalize("NFD", char)
            if parts != char:
                char2idx[char] = [char2idx[part] for part in parts if part in char2idx]
    pad: Optional[str] = cfg.get("pad") or DEFAULT_PAD_TOKEN
    eos: Optional[str] = cfg.get("eos") or DEFAULT_EOS_TOKEN
    bos: Optional[str] = cfg.get("bos") or DEFAULT_BOS_TOKEN
    blank: Optional[str] = cfg.get("blank") or DEFAULT_BLANK_TOKEN
    return Vocabulary(char2idx=char2idx, pad=pad, eos=eos, bos=bos, blank=blank)


def vocabulary_from_mimic3_config(cfg: Dict[str, Any], tokens_txt: str) -> 'Vocabulary':
    """
    Build a Vocabulary from a Mimic3 configuration dictionary and the contents of a tokens.txt file.

    Parameters:
        cfg (Dict[str, Any]): Mimic3 configuration dict; special token names are read from the "phonemes" section.
        tokens_txt (str): Raw text of a tokens.txt file that maps token strings to numeric IDs.

    Returns:
        Vocabulary: A Vocabulary populated from the tokens file with special tokens (pad, bos, eos, blank, blank_word) set from the configuration when present.
    """
    voc: 'Vocabulary' = vocabulary_from_tokens_txt(tokens_txt, id_first=True)
    voc.pad = cfg.get("phonemes", {}).get("pad") or cfg.get("phonemes", {}).get("phoneme_separator")
    voc.eos = cfg.get("phonemes", {}).get("eos")
    voc.bos = cfg.get("phonemes", {}).get("bos")
    voc.blank = cfg.get("phonemes", {}).get("blank")
    voc.blank_word = cfg.get("phonemes", {}).get("blank_word") or cfg.get("phonemes", {}).get("word_separator")
    return voc


def vocabulary_from_tokens_txt(tokens_txt: str, id_first: bool = False) -> 'Vocabulary':
    """
    Creates a Vocabulary instance by parsing a tokens.txt style string (ID token per line).

    Parameters:
        tokens_txt: A string where each line is formatted as "ID token" (e.g., "0 <pad>").

    Returns:
        A Vocabulary instance with character-to-index mapping.
    """
    char2idx: Dict[str, int] = {}
    blank_tok = None
    for line in tokens_txt.split("\n"):
        try:
            if id_first:
                idx_str, token = line.split(" ", 1)
            else:
                token, idx_str = line.split(" ", 1)
            char2idx[token] = int(idx_str)
            if int(idx_str) == 0:
                blank_tok = token
        except ValueError:
            # Skip empty lines or malformed lines
            pass

    return Vocabulary(char2idx=char2idx, blank=blank_tok)


def vocabulary_from_coqui_config(cfg: Dict[str, Any]) -> 'Vocabulary':
    """
    Build a Vocabulary from a Coqui TTS configuration dictionary.

    Supports Coqui character classes `TTS.tts.models.vits.VitsCharacters` and
    `TTS.tts.utils.text.characters.Graphemes`; maps configured characters, punctuations,
    and special tokens (pad, bos, eos, blank) to integer IDs according to Coqui's settings.

    Parameters:
        cfg (Dict[str, Any]): Coqui TTS configuration dictionary containing a "characters" section.

    Returns:
        Vocabulary: A Vocabulary whose char2idx maps each token to its index and whose
        special-token fields (pad, eos, bos, blank) are set from the configuration.

    Raises:
        ValueError: If the configuration specifies an unsupported Coqui tokenizer class.
    """
    characters_cfg: Dict[str, Any] = cfg.get("characters", {})
    pad: Optional[str] = characters_cfg.get("pad")
    eos: Optional[str] = characters_cfg.get("eos")
    bos: Optional[str] = characters_cfg.get("bos")
    blank: Optional[str] = characters_cfg.get("blank")
    punctuations: Optional[str] = characters_cfg.get("punctuations")
    characters: Optional[str] = characters_cfg.get("characters")
    clazz: str = characters_cfg.get("characters_class", "N/A")
    sort: bool = characters_cfg.get("is_sorted", False)
    unique: bool = characters_cfg.get("is_unique", False)
    vocab: List[str]

    if clazz == "TTS.tts.models.vits.VitsCharacters":
        vocab = list(punctuations) + list(characters)
        if pad:
            vocab.insert(0, pad)
        if cfg.get("add_blank"):
            blank = blank or "<BLNK>"
        if blank:
            vocab.append(blank)
    elif clazz == "TTS.tts.utils.text.characters.Graphemes":
        vocab = list(characters)
        if unique:
            # NOTE: deduplication in coqui does not preserve order
            # MUST be used together with is_sorted
            vocab = list(set(vocab))
        if sort:
            vocab = sorted(vocab)
        vocab = [blank, *vocab] if blank is not None and len(blank) > 0 else vocab
        vocab = [bos, *vocab] if bos is not None and len(bos) > 0 else vocab
        vocab = [eos, *vocab] if eos is not None and len(eos) > 0 else vocab
        vocab = [pad, *vocab] if pad is not None and len(pad) > 0 else vocab
        vocab = vocab + list(punctuations)
    else:
        raise ValueError(f"unsupported coqui tokenizer: {clazz}")

    return Vocabulary(char2idx={char: idx for idx, char in enumerate(vocab)},
                      pad=pad,
                      eos=eos,
                      bos=bos,
                      blank=blank)


def from_piper_config(cfg: Dict[str, Any], decompose: bool = False) -> 'TTSTokenizer':
    """
    Create a TTSTokenizer configured from a Piper configuration.

    Parameters:
        cfg (Dict[str, Any]): Piper configuration dictionary containing phoneme/token mappings and optional special token names.
        decompose (bool): See :func:`vocabulary_from_piper_config`.

    Returns:
        TTSTokenizer: Tokenizer configured with the Vocabulary derived from `cfg` and Piper-oriented defaults for blank and BOS/EOS handling.
    """
    voc: Vocabulary = vocabulary_from_piper_config(cfg, decompose)
    # Default settings for Piper
    add_blank: bool = True
    blank_at_end: bool = True
    blank_at_start: bool = True
    use_eos_bos: bool = True
    add_blank_word: bool = False
    return TTSTokenizer(voc, add_blank_char=add_blank, add_blank_word=add_blank_word,
                        blank_at_end=blank_at_end, blank_at_start=blank_at_start,
                        use_eos_bos=use_eos_bos)


def from_vosk_config(cfg: Dict[str, Any]) -> 'TTSTokenizer':
    """
    Create a TTSTokenizer for an alphacep vosk-tts voice.

    Tokenization matches piper (blank id 0 interspersed between every token
    with leading/trailing blanks, BOS/EOS wrapping) — which reproduces
    vosk_tts's own ``^ 0 p 0 p … 0 $`` id stream exactly. Compound folding
    is disabled because the phonemizer already emits complete phoneme
    tokens, so merging neighbours (``s`` + ``h`` -> ``sh``) would be wrong.
    """
    voc: Vocabulary = vocabulary_from_piper_config(cfg)
    return TTSTokenizer(voc, add_blank_char=True, add_blank_word=False,
                        blank_at_end=True, blank_at_start=True,
                        use_eos_bos=True, fold_compounds=False)


def from_mimic3_config(cfg: Dict[str, Any], tokens_txt: str) -> 'TTSTokenizer':
    """
    Factory method to create a TTSTokenizer from a Mimic3 configuration and tokens file.

    Parameters:
        cfg: The Mimic3 configuration dictionary.
        tokens_txt: The content of the tokens.txt file.

    Returns:
        A configured TTSTokenizer instance.
    """
    voc: Vocabulary = vocabulary_from_mimic3_config(cfg, tokens_txt)
    phonemes_cfg: Dict[str, Any] = cfg.get("phonemes", {})
    blank_between: str = phonemes_cfg.get("blank_between", BlankBetween.TOKENS_AND_WORDS)
    blank_at_end: bool = phonemes_cfg.get("blank_at_end", True)
    blank_at_start: bool = phonemes_cfg.get("blank_at_start", True)
    use_eos_bos: bool = phonemes_cfg.get("auto_bos_eos", True)

    add_blank: bool = blank_between != BlankBetween.WORDS  # intersperse blank char
    add_blank_word: bool = blank_between != BlankBetween.TOKENS

    return TTSTokenizer(voc, add_blank_char=add_blank, add_blank_word=add_blank_word,
                        blank_at_end=blank_at_end, blank_at_start=blank_at_start,
                        use_eos_bos=use_eos_bos)


def from_tokens_txt(tokens_txt: str, id_first=False) -> 'TTSTokenizer':
    """
    Create a TTSTokenizer from the contents of a tokens.txt file using conservative defaults.

    Parameters:
        tokens_txt (str): Contents of a tokens.txt file that maps token strings to IDs.

    Returns:
        TTSTokenizer: A tokenizer built from the parsed vocabulary configured with
        add_blank_char=True, add_blank_word=True, blank_at_end=True, blank_at_start=True,
        and use_eos_bos=True.
    """
    voc: Vocabulary = vocabulary_from_tokens_txt(tokens_txt, id_first)
    add_blank_word: bool = False
    add_blank: bool = True
    blank_at_end: bool = True
    blank_at_start: bool = True
    use_eos_bos: bool = False
    return TTSTokenizer(voc, add_blank_char=add_blank, add_blank_word=add_blank_word,
                        blank_at_end=blank_at_end, blank_at_start=blank_at_start,
                        use_eos_bos=use_eos_bos)


def from_coqui_config(cfg: Dict[str, Any]) -> 'TTSTokenizer':
    """
    Create a TTSTokenizer configured from a Coqui TTS configuration.

    Interprets the following Coqui config keys:
    - "add_blank": enable inter-character blank insertion and control blank placement at start/end.
    - "enable_eos_bos_chars": enable wrapping token sequences with BOS/EOS.
    This factory does not enable word-level blank mapping (blank_word is False).

    Parameters:
        cfg (Dict[str, Any]): Coqui configuration dictionary.

    Returns:
        TTSTokenizer: Tokenizer instance configured according to the provided Coqui config.
    """
    voc: Vocabulary = vocabulary_from_coqui_config(cfg)
    add_blank_word: bool = False
    # Coqui typically controls blank insertion via 'add_blank' flag
    add_blank: bool = cfg.get("add_blank", False)
    blank_at_end: bool = cfg.get("add_blank", False)
    blank_at_start: bool = cfg.get("add_blank", False)
    use_eos_bos: bool = cfg.get("enable_eos_bos_chars", False)
    return TTSTokenizer(voc, add_blank_char=add_blank, add_blank_word=add_blank_word,
                        blank_at_end=blank_at_end, blank_at_start=blank_at_start,
                        use_eos_bos=use_eos_bos)


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
    keep_punctuation: bool = False
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
    phoneme vocabulary is built here and ``adapter.make_encoder()`` returns an
    encoder that reads text.  Subclasses
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
        lang = request.lang_code or ""
        return LoadedFields(
            tokenizer=tokenizer,
            engine=Engine.CHATTERBOX,
            lang_code=request.lang_code,
            phoneme_type=request.phoneme_type or PhonemeType.UNICODE,
            alphabet=request.alphabet or cls.ALPHABET,
            add_diacritics=is_arabic(lang) or lang.lower().startswith("he"),
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
        # "pad": null is a vocabulary with no pad.
        config.setdefault("pad", DEFAULT_PAD_TOKEN)
        config["blank"] = config.get("blank") or DEFAULT_BLANK_TOKEN
        config["bos"] = config.get("bos") or DEFAULT_BOS_TOKEN
        config["eos"] = config.get("eos") or DEFAULT_EOS_TOKEN

        return LoadedFields(
            tokenizer=TTSTokenizer.from_phoonnx_config(config),
            engine=config.get("engine"),
            lang_code=request.lang_code or config.get("lang_code"),
            phoneme_type=request.phoneme_type or config.get("phoneme_type", PhonemeType.ESPEAK),
            alphabet=alphabet,
            add_diacritics=inference.get("add_diacritics"),
            diacritizer_model=inference.get("diacritizer_model", None),
            blank_between=config.get("blank_between", BlankBetween.TOKENS_AND_WORDS),
        )


class VoskLoader(ConfigLoader):
    """alphacep vosk-tts voices: piper-shaped VITS with a dictionary/rule
    Russian g2p.

    Indexed voices name ``engine: vosk``, resolved by the engine-declared
    branch below; this loader also recognises a *local* vosk model directory
    with no declared engine, whose ``config.json`` is piper-shaped
    (list-valued ``phoneme_id_map``, an ``espeak`` stanza) but — unlike a real
    piper config — carries no ``phoneme_type``. :class:`PiperLoader` already
    rejects a config with no ``phoneme_type`` (its own ``piper_version`` fast
    path aside), so the missing key is what keeps a genuine vosk config out of
    :class:`PiperLoader`'s hands regardless of probing order; the
    romanised-Russian phoneme inventory (``a0``/``a1``/``bj``/``sch`` …) is
    the second signal that keeps a piper-shaped config with some *other*,
    unrelated vocabulary from being mistaken for vosk.
    """

    @classmethod
    def detect(cls, request: LoadRequest) -> bool:
        config = request.config
        # an explicitly declared engine is authoritative in both directions:
        # a config naming vosk is trusted even without the tell-tale
        # inventory, and one naming anything else is never claimed by it.
        engine = config.get("engine")
        if engine in ("vosk", Engine.VOSK, Engine.VOSK.value):
            pid = config.get("phoneme_id_map")
            return isinstance(pid, dict) and bool(pid)
        if engine:
            return False
        pid = config.get("phoneme_id_map")
        if not isinstance(pid, dict) or not pid:
            return False
        # a real vosk config carries no phoneme_type of its own - piper does.
        # Without this, a piper (or piper-shaped) config that merely happens
        # to also define "a0"/"sch" phonemes (a custom vocabulary, a fixture)
        # would be stolen from PiperLoader by the inventory check alone.
        if config.get("phoneme_type"):
            return False
        return "a0" in pid and "sch" in pid

    @classmethod
    def load(cls, request: LoadRequest) -> LoadedFields:
        config = request.config

        # fixed special tokens (ids 0/1/2 in every vosk phoneme_id_map)
        config["pad"] = DEFAULT_PAD_TOKEN
        config["blank"] = DEFAULT_BLANK_TOKEN
        config["bos"] = DEFAULT_BOS_TOKEN
        config["eos"] = DEFAULT_EOS_TOKEN

        # the config's own phoneme_type/espeak stanza describes how the
        # dictionary/rule g2p was *derived*, not how this loader tokenizes;
        # the vocabulary is the fixed romanised-Russian phoneme set, so both
        # are pinned rather than inherited from the config or the caller.
        return LoadedFields(
            tokenizer=from_vosk_config(config),
            engine=Engine.VOSK,
            lang_code=request.lang_code or config.get("lang_code") or "ru",
            phoneme_type=PhonemeType.VOSK,
            alphabet=Alphabet.VOSK,
            pinned=frozenset({"phoneme_type", "alphabet"}),
        )


# Piper phoneme types that name a phonemizer of their own, and the scriptconv
# backend and alphabet that give the same symbols. Each backend gives one
# symbol per code point (a pinyin initial, final or tone digit for g2pW), and
# Piper merges no vowel clusters for these types.
PIPER_PHONEMIZERS: Dict[str, Tuple[str, Alphabet]] = {
    "pinyin": ("g2pw", Alphabet.PINYIN),
    "hebrew": ("hebrew_ipa", Alphabet.IPA),
    "japanese": ("openjtalk_prosody", Alphabet.IPA),
    "thai": ("tltk", Alphabet.IPA),
    "lithuanian": ("espeak_lt_accent", Alphabet.IPA),
}

# The symbols after which a Piper pinyin voice reads a blank: the tone digits,
# the pause marks and the space (GROUP_END_PHONEMES of
# src/piper/phonemize_chinese.py, OHF-Voice/piper1-gpl v1.8.0).
PINYIN_GROUP_END = frozenset("12345。？！.?!—…、，：；,:; ")

# The multi-character keys of Piper's default phoneme id map, at their default ids
# (DEFAULT_PHONEME_ID_MAP of src/piper/phoneme_ids.py, OHF-Voice/piper1-gpl v1.5.0).
# Piper merges them into one id only for a config that lists them in vowel_clusters.
PIPER_DIPHTHONGS = {"aɪ": [161], "aʊ": [162], "ɔɪ": [163], "eɪ": [164], "oʊ": [165]}


class PiperLoader(ConfigLoader):
    """piper checkpoints.

    ``text`` and ``pygoruut`` models pin both the phoneme type and the
    alphabet: their vocabulary is the model's own character or goruut token
    set, so a caller-supplied alphabet would mis-tokenise. The Piper types of
    :data:`PIPER_PHONEMIZERS` pin them too, and read each symbol as one id
    with no compound folding. Hebrew voices diacritize with Nakdimon first,
    as Piper does, and pinyin voices put a blank only after the symbols of
    :data:`PINYIN_GROUP_END`.

    Piper trains with clause punctuation, so its voices keep it at synthesis.
    phoonnx's own ``--piper`` export side-cars are piper-shaped too, but their
    models were trained by phoonnx, without punctuation unless the config says
    otherwise. They are recognised by a ``piper_version`` starting with
    ``phoonnx`` or by an ``alphabet`` key, which Piper never writes.

    Piper phonemizes with the espeak-ng voice that ``espeak.voice`` names
    (``es-419`` for an ``es_MX`` voice), and it reads the phonemes decomposed
    (NFD). The ``vowel_clusters`` of a Piper 1.5 config, such as ``a`` + ``ɪ``,
    each have their own key in ``phoneme_id_map``, and the tokenizer's compound
    folding gives such a key its own id. A Piper config with no
    ``vowel_clusters`` whose only multi-character keys are the diphthongs of
    Piper's default map (:data:`PIPER_DIPHTHONGS`) was trained on the split
    ids, so it folds nothing. Other trainers write Piper configs too, with
    keys of their own that they fold, and those configs keep the folding.
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

        return config["phoneme_type"] in ("text", "espeak", *PIPER_PHONEMIZERS)

    @staticmethod
    def made_by_phoonnx(config: Dict[str, Any]) -> bool:
        """Whether *config* is a phoonnx export side-car rather than a Piper config."""
        return (str(config.get("piper_version", "")).startswith("phoonnx")
                or "alphabet" in config)

    @classmethod
    def trained_on_split_diphthongs(cls, config: Dict[str, Any]) -> bool:
        """Whether *config* is Piper's default map with no ``vowel_clusters``, so Piper reads
        ``aɪ`` as ``a`` and ``ɪ``."""
        compounds = {key: ids for key, ids in config["phoneme_id_map"].items() if len(key) > 1}
        return (not config.get("vowel_clusters") and not cls.made_by_phoonnx(config)
                and bool(compounds) and compounds.items() <= PIPER_DIPHTHONGS.items())

    @classmethod
    def load(cls, request: LoadRequest) -> LoadedFields:
        config = request.config

        phoneme_type = request.phoneme_type or config.get("phoneme_type", PhonemeType.ESPEAK)
        espeak_voice = config.get("espeak", {}).get("voice")
        lang_code = (request.lang_code or (espeak_voice if phoneme_type == PhonemeType.ESPEAK else None)
                     or config.get("language", {}).get("code") or espeak_voice)
        alphabet = request.alphabet
        pinned = set()
        piper_type = None
        if phoneme_type == "text":
            phoneme_type, alphabet = PhonemeType.UNICODE, Alphabet.UNICODE
            pinned = {"phoneme_type", "alphabet"}
        elif phoneme_type == "pygoruut":
            # special case: neurlang models
            phoneme_type, alphabet = PhonemeType.GORUUT, Alphabet.IPA
            pinned = {"phoneme_type", "alphabet"}
        elif phoneme_type in PIPER_PHONEMIZERS:
            piper_type = phoneme_type
            phoneme_type, alphabet = PIPER_PHONEMIZERS[piper_type]
            pinned = {"phoneme_type", "alphabet"}
        else:
            alphabet = alphabet or Alphabet.IPA

        # not configurable in piper
        config["pad"] = DEFAULT_PAD_TOKEN
        config["blank"] = DEFAULT_BLANK_TOKEN
        config["blank_word"] = DEFAULT_BLANK_WORD_TOKEN
        config["bos"] = DEFAULT_BOS_TOKEN
        config["eos"] = DEFAULT_EOS_TOKEN

        tokenizer = from_piper_config(config, decompose=not cls.made_by_phoonnx(config))
        add_diacritics, diacritizer_model = is_arabic(lang_code or ""), None
        if piper_type is not None or cls.trained_on_split_diphthongs(config):
            tokenizer.fold_compounds = False
        if piper_type == "pinyin":
            tokenizer.blank_after = PINYIN_GROUP_END
        elif piper_type == "hebrew":
            add_diacritics, diacritizer_model = True, "nakdimon"

        return LoadedFields(
            tokenizer=tokenizer,
            engine=Engine.PIPER,
            lang_code=lang_code,
            phoneme_type=phoneme_type,
            alphabet=alphabet,
            add_diacritics=add_diacritics,
            diacritizer_model=diacritizer_model,
            keep_punctuation=not cls.made_by_phoonnx(config),
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
        with open(request.tokens_txt, encoding="utf-8") as tokens_file:
            tokens_txt = tokens_file.read()

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
            tokenizer=from_mimic3_config(config, tokens_txt),
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
            tokenizer=from_coqui_config(config),
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
        # transformers' VitsTokenizer inserts token id 0 between tokens, whatever
        # its pad_token is; a <pad> with another id is outside what the model saw.
        config["blank"] = next((token for token, idx in request.vocab.items() if idx == 0),
                               config.get("blank", DEFAULT_BLANK_TOKEN))

        tokenizer = TTSTokenizer(
            Vocabulary(char2idx=request.vocab, blank=config["blank"]),
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
                vocabulary = vocabulary_from_tokens_txt(ids_file.read())
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
    phonemes.
    """

    @classmethod
    def detect(cls, request: LoadRequest) -> bool:
        return True

    @classmethod
    def load(cls, request: LoadRequest) -> LoadedFields:
        config = request.config
        inference = config.get("inference", {})
        # the catch-all loader sees every format, so "language" is not
        # always a piper-style {"code": ...} stanza
        language = config.get("language")
        lang_code = request.lang_code or (
            language.get("code") if isinstance(language, dict) else None)
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
            lang_code=lang_code,
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
    VoskLoader,
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


def build_voice_config(config: Dict[str, Any],
                       vocab: Optional[Dict[str, Any]] = None,
                       tokenizer_config: Optional[Dict[str, Any]] = None,
                       tokens_txt: Optional[str] = None,
                       lang_code: Optional[str] = None,
                       phoneme_type: Optional[Union[str, PhonemeType]] = None,
                       alphabet: Optional[Union[str, Alphabet]] = None,
                       engine: Optional[Union[str, Engine]] = None,
                       engine_params: Optional[Dict[str, Any]] = None,
                       bpe_tokenizer_json: Optional[str] = None,
                       lang_tokens: Optional[Dict[str, str]] = None):
    """Build a :class:`~phoonnx.config.VoiceConfig` from a config of any format and its companion files.

    Loaders may normalise *config* in place. The overrides win over the config's
    own values, except where the format pins them; ``engine_params`` win over the
    config's ``engine_params``. Raises ``ValueError`` when the format needs a
    companion file that was not given.
    """
    from phoonnx.config import DEFAULT_HOP_LENGTH, VoiceConfig

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
        noise_scale=inference.get("noise_scale"),
        length_scale=inference.get("length_scale"),
        noise_w_scale=inference.get("noise_w"),
        add_diacritics=loaded.add_diacritics,
        diacritizer_model=loaded.diacritizer_model,
        hop_length=config.get("hop_length", DEFAULT_HOP_LENGTH),
        lang_code=loaded.lang_code,
        alphabet=loaded.alphabet,
        engine=loaded.engine,
        phonemizer_model=config.get("phonemizer_model"),
        keep_punctuation=bool(config.get("keep_punctuation", loaded.keep_punctuation)),
        text_casing=config.get("text_casing", "ignore"),
        phonemizer_identity=config.get("phonemizer_identity"),
        phoneme_type=loaded.phoneme_type,
        speaker_id_map=config.get("speaker_id_map", {}),
        default_speaker_id=config.get("default_speaker_id"),
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


def _read_json(source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """A copy of a dict, or the parsed JSON file at a path."""
    if isinstance(source, dict):
        return copy.deepcopy(source)
    return json.loads(Path(source).read_text(encoding="utf-8"))


def convert_config(path_or_dict: Union[str, Path, Dict[str, Any]], *,
                   vocab: Optional[Union[str, Path, Dict[str, Any]]] = None,
                   tokens: Optional[Union[str, Path]] = None,
                   tokenizer_config: Optional[Union[str, Path, Dict[str, Any]]] = None,
                   **overrides: Any) -> Dict[str, Any]:
    """Turn a voice config of any format into a native phoonnx ``config.json`` dict.

    The result is what :meth:`phoonnx.config.VoiceConfig.to_dict` writes. It loads through
    :meth:`~phoonnx.config.VoiceConfig.from_dict` with no companion file and no override, and
    gives the same token ids as the original config does with them. A Chatterbox config
    still needs ``bpe_tokenizer_json`` at load. A native config comes back in normal form.

    Parameters:
        path_or_dict: The config, as a dict or the path of a JSON file.
        vocab: The ``vocab.json`` of a transformers export, as a dict or a path.
        tokens: Path of a ``tokens.txt``, or a ``.json`` token map, which mimic3
            voices and sherpa-onnx style models need.
        tokenizer_config: The ``tokenizer_config.json`` of a transformers export.
        overrides: ``lang_code``, ``phoneme_type``, ``alphabet``, ``engine``,
            ``engine_params``, ``bpe_tokenizer_json`` and ``lang_tokens``, with the
            meaning they have in :meth:`phoonnx.config.VoiceConfig.from_dict`.

    Raises:
        ValueError: If the format needs a companion file that was not given.
    """
    return build_voice_config(
        _read_json(path_or_dict),
        vocab=_read_json(vocab) if vocab else None,
        tokenizer_config=_read_json(tokenizer_config) if tokenizer_config else None,
        tokens_txt=str(tokens) if tokens else None,
        **overrides,
    ).to_dict()
