"""The typed failures phoonnx raises, in one place.

Every class here derives from :class:`PhoonnxError`, so one
``except PhoonnxError`` catches each of them. A class with a builtin base has
it as a second base: ``except ValueError`` catches
:class:`UnsupportedVoiceLanguage` and ``except RuntimeError`` catches
:class:`VoiceExceedsMemoryBudget`.

Callers that want to react to a failure rather than just report it need to tell
four situations apart, and each has its own exception here.

A *missing language* is a request phoonnx cannot serve because no voice covers
the language at all; the voice manager signals that with a plain ``ValueError``
from the catalogue lookup, since there is no voice object to describe.

An *unsupported language* is a voice that exists but whose language no
phonemizer backend serves — :class:`UnsupportedVoiceLanguage`. It is raised at
load time rather than mid-synthesis, so a caller can fall back to another voice
before the user has sent any text.

A *config defect* is a model whose own shipped artifacts phoonnx cannot read:
:class:`UnsupportedTokenizer` and :class:`UnsupportedSentencePieceModel` mark a
tokenizer or SentencePiece model in a format the loaders do not implement.
Retrying will not help; the model needs re-exporting or a new loader.

A *resource refusal* is phoonnx declining work it could start but could not
finish: :class:`VoiceExceedsMemoryBudget` refuses a voice whose weights alone
exceed the cache's memory budget, because loading it is a guaranteed OOM kill
rather than a slow path. The refusal is the useful outcome — one failed
request instead of a restart loop.

A voice that cannot load is a :class:`VoiceLoadError`; a frontend that does
not fit the voice and a config in a format phoonnx does not read are its two
named cases. A voice that an export target cannot hold is
:class:`ExportNotSupported`.

The classes are defined here and imported by the modules that raise them, so
an import through one of those modules, such as
``phoonnx.config.UnsupportedVoiceLanguage``, gives the same class. This module
imports nothing from phoonnx, so importing it stays cheap.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scriptconv.phonemizers.enums import Phonemizer as PhonemeType

__all__ = [
    "ExportNotSupported",
    "FrontendMismatchError",
    "PhoonnxError",
    "PrecisionUnavailable",
    "UnsupportedConfigError",
    "UnsupportedSentencePieceModel",
    "UnsupportedTokenizer",
    "UnsupportedVoiceLanguage",
    "VoiceExceedsMemoryBudget",
    "VoiceLoadError",
]


class PhoonnxError(Exception):
    """The base of the typed failures in this module."""


class VoiceLoadError(PhoonnxError):
    """A voice cannot load: it has no config, or the model's ``num_symbols``
    differs from the config's symbol table."""


class FrontendMismatchError(VoiceLoadError):
    """A frontend does not fit the voice: its alphabet or punctuation differs,
    or it writes a symbol the voice's vocabulary lacks."""


class UnsupportedConfigError(VoiceLoadError):
    """A ``config.json`` is in a format phoonnx does not read."""


class PrecisionUnavailable(VoiceLoadError, ValueError):
    """A voice has no files for the precision a caller asked for."""

    def __init__(self, voice_id: str, precision: str):
        self.voice_id = voice_id
        self.precision = precision
        other = "full" if precision == "int8" else "int8"
        super().__init__(
            f"voice {voice_id!r} has no {precision} files; precision 'auto' "
            f"or {other!r} loads it")


class ExportNotSupported(PhoonnxError):
    """The export target cannot hold this voice, for example a phonemizer or
    an engine the target lacks."""


class UnsupportedTokenizer(PhoonnxError):
    """The tokenizer.json declares a component this reader does not implement."""


class UnsupportedSentencePieceModel(PhoonnxError):
    """The .model declares a feature this reader does not implement."""


class UnsupportedVoiceLanguage(PhoonnxError, ValueError):
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


class VoiceExceedsMemoryBudget(PhoonnxError, RuntimeError):
    """A voice's on-disk size alone is larger than ``max_loaded_bytes``.

    Loading it is not a degraded path, it is a guaranteed OOM kill: a 15.4 GB
    voice against a 5 GB budget in an 8 GiB cgroup killed the process, and the
    container restarted straight back into the same load, over and over. The
    size is known before the load is attempted whenever the voice was already
    on disk from an earlier attempt (its own weights survive the restart even
    though the process does not), which is exactly the case that used to
    loop. Refusing here turns that loop into one failed request.
    """
