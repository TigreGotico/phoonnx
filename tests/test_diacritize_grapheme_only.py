"""Diacritization is orthographic (grapheme-level); it must never run over
phonemic (e.g. IPA) input, regardless of add_diacritics. phoonnx delegates
diacritization to scriptconv (scriptconv.diacritics.diacritize) — these tests
replace the Arabic diacritizer with a recorder so no diacritizer backend is
needed.
"""
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import scriptconv.diacritics as scd
from phoonnx.config import Alphabet, PhonemeType, SynthesisConfig
from phoonnx.voice import TTSVoice
from scriptconv.phonemizers.base import BasePhonemizer


class _RecordingPhonemizer(BasePhonemizer):
    def __init__(self, seen):
        super().__init__()
        self.seen = seen

    def phonemize_string(self, text, lang):
        self.seen.append(text)
        return text


class _RecordingDiacritizer:
    def __init__(self, calls):
        self.calls = calls

    def diacritize(self, text):
        self.calls.append(text)
        return text


def _make_voice(model_alphabet, seen):
    voice = TTSVoice.__new__(TTSVoice)
    voice.phonetic_spellings = None
    voice.phonemizer = _RecordingPhonemizer(seen)
    voice.config = types.SimpleNamespace(
        phoneme_type=PhonemeType.UNICODE,
        alphabet=model_alphabet,
        lang_code="ar",
        add_diacritics=True,
        diacritizer_model=None,
        sample_rate=22050,
    )
    voice.adapter = MagicMock()
    voice.phonemes_to_ids = lambda phonemes: [1] * len(phonemes)
    voice.phoneme_ids_to_audio = lambda phoneme_ids, syn_config=None, language_ids=None, include_alignments=False: np.zeros(4, dtype=np.float32)
    return voice


class TestDiacritizeGraphemeOnly(unittest.TestCase):
    def test_phonemic_ipa_input_is_never_diacritized(self):
        calls, seen = [], []
        voice = _make_voice(Alphabet.IPA, seen)
        syn_config = SynthesisConfig(alphabet=Alphabet.IPA, normalize_audio=False)

        with patch.object(scd, "_tashkeel", lambda *a, **k: _RecordingDiacritizer(calls)):
            list(voice.synthesize("mafʕuːl", syn_config))

        self.assertEqual(calls, [], f"diacritizer was called on phonemic input: {calls}")
        self.assertEqual(seen, [], "phonemic input reached the phonemizer")

    def test_grapheme_input_is_still_diacritized_when_enabled(self):
        calls, seen = [], []
        voice = _make_voice(Alphabet.IPA, seen)
        syn_config = SynthesisConfig(alphabet=Alphabet.GRAPHEMES, normalize_audio=False)

        with patch.object(scd, "_tashkeel", lambda *a, **k: _RecordingDiacritizer(calls)):
            list(voice.synthesize("mrhba", syn_config))

        self.assertEqual(calls, ["mrhba"])
        self.assertEqual(seen, ["mrhba"])
