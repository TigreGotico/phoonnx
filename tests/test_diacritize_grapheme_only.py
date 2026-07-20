"""_diacritize is orthographic (grapheme-level); it must never run over
phonemic (e.g. IPA) input, regardless of add_diacritics.
"""
import types
import unittest
from unittest.mock import MagicMock

import numpy as np

from phoonnx.config import Alphabet, SynthesisConfig
from phoonnx.voice import TTSVoice


def _make_voice(model_alphabet, diacritize_calls):
    voice = TTSVoice.__new__(TTSVoice)
    voice.phonetic_spellings = None
    voice.phonemizer = MagicMock()
    voice.config = types.SimpleNamespace(
        alphabet=model_alphabet,
        lang_code="ar",
        add_diacritics=True,
        diacritizer_model="rawi-ensemble",
        sample_rate=22050,
    )
    voice.adapter = MagicMock()
    voice.phonemes_to_ids = lambda phonemes: [1] * len(phonemes)
    voice.phoneme_ids_to_audio = lambda phoneme_ids, syn_config=None, language_ids=None, include_alignments=False: np.zeros(4, dtype=np.float32)

    def _diacritize(text, diacritizer_model):
        diacritize_calls.append(text)
        return text

    voice._diacritize = _diacritize
    return voice


class TestDiacritizeGraphemeOnly(unittest.TestCase):
    def test_phonemic_ipa_input_is_never_diacritized(self):
        calls = []
        voice = _make_voice(Alphabet.IPA, calls)
        syn_config = SynthesisConfig(alphabet=Alphabet.IPA, normalize_audio=False)

        list(voice.synthesize("mafʕuːl", syn_config))

        self.assertEqual(calls, [], f"diacritizer was called on phonemic input: {calls}")

    def test_grapheme_input_is_still_diacritized_when_enabled(self):
        calls = []
        voice = _make_voice(Alphabet.IPA, calls)
        syn_config = SynthesisConfig(alphabet=Alphabet.GRAPHEMES, normalize_audio=False)
        voice.phonemizer.phonemize_lazy = None
        voice.phonemizer.phonemize = MagicMock(return_value=[list("test")])

        list(voice.synthesize("mrhba", syn_config))

        self.assertEqual(calls, ["mrhba"])
