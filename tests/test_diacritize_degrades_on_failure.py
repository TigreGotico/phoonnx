"""Diacritization is optional prosody enrichment. If the backend raises
(e.g. stressonnx.UnsupportedLanguageError for a language it doesn't cover),
synthesis must degrade to the unstressed input text instead of 500ing —
matching the degradation contract already used by phoonnx.lang_preprocess's
russian_add_stress. Regression for a production failure where a regional
voice (e.g. ru-RU) hard-failed synthesis because the diacritize graph edge
propagated the exception unguarded.
"""
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import scriptconv.diacritics as scd
from phoonnx.config import Alphabet, SynthesisConfig
from phoonnx.util import LOG
from phoonnx.voice import TTSVoice


def _make_voice():
    voice = TTSVoice.__new__(TTSVoice)
    voice.phonetic_spellings = None
    voice.phonemizer = MagicMock()
    voice.phonemizer.phonemize_lazy = MagicMock(return_value=iter([list("test")]))
    voice.config = types.SimpleNamespace(
        alphabet=Alphabet.IPA,
        lang_code="ru-RU",
        add_diacritics=True,
        diacritizer_model=None,
        sample_rate=22050,
    )
    voice.adapter = MagicMock()
    voice.phonemes_to_ids = lambda phonemes: [1] * len(phonemes)
    voice.phoneme_ids_to_audio = lambda phoneme_ids, syn_config=None, language_ids=None, include_alignments=False: np.zeros(4, dtype=np.float32)
    return voice


class TestDiacritizeDegradesOnFailure(unittest.TestCase):
    def test_diacritize_failure_degrades_to_unstressed_text_with_warning(self):
        voice = _make_voice()
        syn_config = SynthesisConfig(alphabet=Alphabet.GRAPHEMES, normalize_audio=False)

        def _raise(t, l="und", **k):
            raise RuntimeError("UnsupportedLanguageError('ru-RU')")

        with patch.object(scd, "diacritize", side_effect=_raise), \
             patch.object(LOG, "warning") as mock_warn:
            list(voice.synthesize("privet", syn_config))

        voice.phonemizer.phonemize_lazy.assert_called_once()
        called_text = voice.phonemizer.phonemize_lazy.call_args[0][0]
        self.assertEqual(called_text, "privet")
        mock_warn.assert_called()


if __name__ == "__main__":
    unittest.main()
