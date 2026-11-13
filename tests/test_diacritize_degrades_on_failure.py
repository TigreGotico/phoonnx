"""Diacritization is optional prosody enrichment. If the backend raises
(e.g. stressonnx.UnsupportedLanguageError for a language it doesn't cover),
synthesis must degrade to the unstressed input text instead of 500ing —
matching the degradation contract already used by phoonnx.lang_preprocess's
russian_add_stress. Regression for a production failure where a regional
voice (e.g. ru-RU) hard-failed synthesis because the diacritize stage
propagated the exception unguarded.
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
    def __init__(self):
        super().__init__()
        self.seen = []

    def phonemize_string(self, text, lang):
        self.seen.append(text)
        return text


def _make_voice():
    voice = TTSVoice.__new__(TTSVoice)
    voice.phonetic_spellings = None
    voice.phonemizer = _RecordingPhonemizer()
    voice.config = types.SimpleNamespace(
        phoneme_type=PhonemeType.UNICODE,
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

        def _raise(*_, **__):
            raise RuntimeError("UnsupportedLanguageError('ru-RU')")

        with patch.object(scd, "_stress", side_effect=_raise), \
             self.assertLogs("scriptconv.frontend", "WARNING") as logs:
            chunks = list(voice.synthesize("privet", syn_config))

        self.assertEqual(voice.phonemizer.seen, ["privet"])
        self.assertEqual([chunk.phonemes for chunk in chunks], [list("privet")])
        self.assertIn("diacritize failed for lang='ru-RU'", logs.output[0])


if __name__ == "__main__":
    unittest.main()
