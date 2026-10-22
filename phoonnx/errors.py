"""The typed failures phoonnx raises, in one place.

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

These classes are defined next to the code that raises them and re-exported
here; both import paths resolve to the same class, so ``except`` clauses
written against either one catch the same failures.
"""

from phoonnx._bpe import UnsupportedTokenizer
from phoonnx._sentencepiece import UnsupportedSentencePieceModel
from phoonnx.config import UnsupportedVoiceLanguage
from phoonnx.voice_cache import VoiceExceedsMemoryBudget

__all__ = [
    "UnsupportedSentencePieceModel",
    "UnsupportedTokenizer",
    "UnsupportedVoiceLanguage",
    "VoiceExceedsMemoryBudget",
]
