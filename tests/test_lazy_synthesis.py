"""Tests for lazy, per-sentence synthesis in phoonnx.voice.TTSVoice.

Cover three properties of the lazy restructure:

1. Equivalence — a whole-text ``phonemize`` produces exactly the concatenation
   of the per-sentence ``phonemize_lazy`` stream (engine-independent: real
   grapheme/unicode phonemizers + a mocked phonemizer).
2. Ordering — sentence 2 is only phonemized *after* sentence 1 has reached
   ``session.run`` (proved by recording call order through mocks).
3. Language-ID alignment — the lazy Shami stream yields the same phoneme lists
   and parallel language-ID lists as the eager whole-text call.
"""
import types
import unittest
from unittest.mock import MagicMock

import numpy as np

from phoonnx.config import Alphabet, SynthesisConfig
from scriptconv.phonemizers.base import (
    BasePhonemizer,
    GraphemePhonemizer,
    UnicodeCodepointPhonemizer,
)
from phoonnx.voice import AudioChunk, TTSVoice


MULTISENTENCE = (
    "The quick brown fox jumps over the lazy dog. "
    "A rainbow is a phenomenon caused by refraction of light. "
    "Voice assistants rely on low latency to feel responsive."
)


class _RecordingPhonemizer(BasePhonemizer):
    """Grapheme-style phonemizer that records every phonemize_string call.

    Uses the base normalization-free char split so per-sentence and whole-text
    calls are trivially comparable; ``events`` captures ordering.
    """

    def __init__(self, events=None):
        super().__init__(Alphabet.IPA)
        self.events = events if events is not None else []

    def phonemize_string(self, text: str, lang: str) -> str:
        self.events.append(("phonemize", text))
        return text


class TestPhonemizeLazyEquivalence(unittest.TestCase):
    """list(phonemize_lazy(text)) == phonemize(text) for any input."""

    def _assert_equivalent(self, phonemizer, text, lang="en"):
        eager = phonemizer.phonemize(text, lang)
        lazy = list(phonemizer.phonemize_lazy(text, lang))
        self.assertEqual(lazy, eager)

    def test_grapheme_phonemizer_multisentence(self):
        self._assert_equivalent(GraphemePhonemizer(), MULTISENTENCE)

    def test_unicode_phonemizer_multisentence(self):
        self._assert_equivalent(UnicodeCodepointPhonemizer(), MULTISENTENCE)

    def test_mocked_phonemizer_multisentence(self):
        self._assert_equivalent(_RecordingPhonemizer(), MULTISENTENCE)

    def test_empty_text(self):
        for ph in (GraphemePhonemizer(), UnicodeCodepointPhonemizer(),
                   _RecordingPhonemizer()):
            self.assertEqual(list(ph.phonemize_lazy("", "en")), [])
            self.assertEqual(ph.phonemize("", "en"), [])

    def test_single_sentence(self):
        self._assert_equivalent(GraphemePhonemizer(), "just one sentence here")

    def test_delimiter_splitting_matches(self):
        # ':' / ';' split intra-sentence chunks — both paths must chunk identically.
        self._assert_equivalent(GraphemePhonemizer(),
                                "first part: second part; third part. And a new one.")

    def test_lazy_defers_phonemize_string_calls(self):
        """phonemize_string must not run before the generator is pulled."""
        events = []
        ph = _RecordingPhonemizer(events)
        gen = ph.phonemize_lazy(MULTISENTENCE, "en")
        self.assertEqual(events, [])  # nothing phonemized yet
        first = next(gen)
        self.assertTrue(first)
        self.assertEqual(len(events), 1)  # only the first sentence so far


def _make_lazy_voice(events, alphabet=Alphabet.IPA):
    """Build a TTSVoice wired to a recording phonemizer + recording synth step.

    Grapheme input -> IPA model, so synthesis takes the lazy per-sentence phoneme
    branch. ``events`` records both phonemize and session.run in call order.
    """
    voice = TTSVoice.__new__(TTSVoice)
    voice.phonetic_spellings = None
    voice.phonemizer = _RecordingPhonemizer(events)

    config = types.SimpleNamespace(
        alphabet=alphabet,
        lang_code="en",
        add_diacritics=False,
        diacritizer_model=None,
        sample_rate=22050,
    )
    voice.config = config
    voice.adapter = MagicMock()

    voice.phonemes_to_ids = lambda phonemes: [1] * len(phonemes)

    def _synth(phoneme_ids, syn_config=None, language_ids=None, include_alignments=False):
        events.append(("run", tuple(phoneme_ids)))
        return np.zeros(4, dtype=np.float32)

    voice.phoneme_ids_to_audio = _synth
    return voice


class TestLazyOrdering(unittest.TestCase):
    def test_sentence2_phonemized_after_sentence1_run(self):
        events = []
        voice = _make_lazy_voice(events)
        gen = voice.synthesize(MULTISENTENCE, SynthesisConfig(normalize_audio=False))

        first_chunk = next(gen)
        self.assertIsInstance(first_chunk, AudioChunk)

        # After the first yielded chunk: sentence 1 was phonemized AND run, but
        # sentence 2 has NOT been phonemized yet.
        kinds = [e[0] for e in events]
        self.assertEqual(kinds, ["phonemize", "run"],
                         f"expected only sentence 1 work before first chunk, got {events}")

        # Drain the rest — now every sentence is phonemized and run, and each
        # sentence's phonemize precedes its own run.
        list(gen)
        kinds = [e[0] for e in events]
        # phonemize/run strictly interleaved: phon, run, phon, run, ...
        self.assertEqual(kinds, ["phonemize", "run"] * 3)

    def test_no_work_before_first_pull(self):
        events = []
        voice = _make_lazy_voice(events)
        gen = voice.synthesize(MULTISENTENCE, SynthesisConfig(normalize_audio=False))
        self.assertEqual(events, [])  # generator not yet consumed


class TestLanguageIdAlignment(unittest.TestCase):
    def test_shami_lazy_matches_eager(self):
        try:
            from scriptconv.phonemizers.shami import ShamiPhonemizer
        except Exception as e:  # pragma: no cover - dependency-gated
            self.skipTest(f"Shami unavailable: {e}")

        ph = ShamiPhonemizer()
        text = "مرحبا كيف حالك؟ Hello there. شكرا جزيلا."

        eager_phonemes, eager_ids = ph.phonemize_with_language_ids(text, "ar")
        lazy = list(ph.phonemize_with_language_ids_lazy(text, "ar"))
        lazy_phonemes = [p for p, _ in lazy]
        lazy_ids = [i for _, i in lazy]

        self.assertEqual(lazy_phonemes, eager_phonemes)
        self.assertEqual(lazy_ids, eager_ids)
        # Alignment: each sentence's phoneme list and language-id list match length.
        for phonemes, ids in lazy:
            self.assertEqual(len(phonemes), len(ids))


if __name__ == "__main__":
    unittest.main()
