"""Inline ``[[phoneme]]`` blocks in TTSVoice.phonemize.

Regression tests for a crash where text wrapped in ``[[ ]]`` (fed to bypass the
phonemizer and inject phonemes/IPA directly) raised
``'tuple' object has no attribute 'extend'``. Root cause: BasePhonemizer.phonemize
returned the raw ``(str, str, bool)`` tuple form for empty text, so the leading
empty part before a block landed a tuple in the sentence list that the block
branch then tried to mutate.
"""
from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.voice import TTSVoice


def _voice(phonemizer):
    v = TTSVoice.__new__(TTSVoice)
    v.phonemizer = phonemizer

    class _Cfg:
        lang_code = "ar"

    v.config = _Cfg()
    return v


class _ListPhonemizer:
    """Well-behaved: PhonemizedChunks == list[list[str]], [] for empty."""

    def phonemize(self, text, lang):
        return [list(text)] if text else []


class _TuplePhonemizer:
    """Adversarial: returns immutable tuples (and the old malformed empty form)."""

    def phonemize(self, text, lang):
        return [tuple(text)] if text else [("", "", True)]


def test_base_phonemize_empty_returns_empty_list():
    # the root fix: empty text must yield no sentences, not the raw tuple form
    class _P(BasePhonemizer):
        def phonemize_string(self, text, lang):
            return text

    p = _P.__new__(_P)
    assert p.phonemize("", "ar") == []


def test_inline_block_only():
    out = _voice(_ListPhonemizer()).phonemize("[[abc]]")
    assert out == [["a", "b", "c"]]


def test_inline_block_after_text():
    out = _voice(_ListPhonemizer()).phonemize("hi [[xy]]")
    assert out and all(isinstance(s, list) for s in out)
    assert "x" in out[-1] and "y" in out[-1]


def test_inline_block_between_text():
    out = _voice(_ListPhonemizer()).phonemize("[[ab]] cd")
    assert out and all(isinstance(s, list) for s in out)


def test_voice_robust_to_tuple_returning_phonemizer():
    # even a phonemizer that returns tuples must not break inline blocks
    out = _voice(_TuplePhonemizer()).phonemize("[[xy]]")
    assert out and "x" in out[-1] and "y" in out[-1]
