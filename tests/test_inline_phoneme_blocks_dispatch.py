"""Inline [[phoneme]] override blocks must never reach a phonemizer verbatim,
including language-aware phonemizers exposing phonemize_with_language_ids_lazy.
"""
import types
import unittest
from unittest.mock import MagicMock

import numpy as np

from phoonnx.config import Alphabet, SynthesisConfig
from phoonnx.voice import AudioChunk, TTSVoice


class _LangAwarePhonemizer:
    """Mock language-aware phonemizer recording every text it is asked to phonemize."""

    def __init__(self, events):
        self.events = events

    def phonemize_with_language_ids_lazy(self, text, lang):
        self.events.append(text)
        for sentence in text.split(". "):
            sentence = sentence.strip()
            if sentence:
                yield list(sentence), [0] * len(sentence)

    def phonemize(self, text, lang):
        # Non-language-aware fallback used by the inline-[[phoneme]]-block path.
        self.events.append(text)
        return [list(sentence.strip()) for sentence in text.split(". ") if sentence.strip()]


def _make_voice(phonemizer):
    voice = TTSVoice.__new__(TTSVoice)
    voice.phonetic_spellings = None
    voice.phonemizer = phonemizer
    voice.config = types.SimpleNamespace(
        alphabet=Alphabet.IPA,
        lang_code="en",
        add_diacritics=False,
        diacritizer_model=None,
        sample_rate=22050,
    )
    voice.adapter = MagicMock()
    voice.phonemes_to_ids = lambda phonemes: [1] * len(phonemes)
    voice.phoneme_ids_to_audio = lambda phoneme_ids, syn_config=None, language_ids=None, include_alignments=False: np.zeros(4, dtype=np.float32)
    return voice


class TestInlinePhonemeBlockDispatchOrder(unittest.TestCase):
    def test_inline_block_bypasses_language_aware_phonemizer(self):
        events = []
        voice = _make_voice(_LangAwarePhonemizer(events))
        text = "hello [[wɜːld]] there."

        list(voice.synthesize(text, SynthesisConfig(normalize_audio=False)))

        for seen_text in events:
            self.assertNotIn("[[", seen_text,
                              f"inline phoneme block leaked into language-aware phonemizer input: {seen_text!r}")
