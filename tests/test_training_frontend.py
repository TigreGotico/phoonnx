"""Dataset building reads text through the frontend that synthesis uses.

``preprocess.phonemize_worker`` phonemizes each row with
``TextFrontend.phonemize_flat``, built by ``phoonnx.config.voice_frontend``, the
function ``TTSVoice.frontend`` builds from. So a row's training symbols and ids
equal those ``TTSVoice.prepare`` gives for the same text and config.

The diacritizer is replaced with a recorder, so no model is needed. The unicode
backend needs no espeak. Expected values come from ``TTSVoice.prepare``, a path
the training code does not call.
"""
import queue
import types
import unittest
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock, patch

import scriptconv.diacritics as scd

from phoonnx.config import Alphabet, PhonemeType, SynthesisConfig, VoiceConfig
from phoonnx.voice import TTSVoice
from phoonnx_train import preprocess
from phoonnx_train.dataset_loaders import PreprocessorConfig, Utterance
from phoonnx_train.evaluation.scorer import text_to_ids


def _tokenizer():
    """Ids from the symbols alone, so two paths agree only if their symbols agree."""
    return types.SimpleNamespace(tokenize=lambda symbols: [1] + [ord(s[0]) for s in symbols] + [2])


class _Fatha:
    """Stands for text2tashkeel: adds one fatha to each text it reads."""

    def diacritize(self, text):
        return text + "َ"


def _voice(lang, add_diacritics=False):
    config = VoiceConfig(num_symbols=1, num_speakers=1, num_langs=1, sample_rate=22050,
                         lang_code=lang, phoneme_type=PhonemeType.UNICODE, alphabet=Alphabet.UNICODE,
                         phonemizer_model=None, add_diacritics=add_diacritics)
    voice = TTSVoice(config=config, session=MagicMock(), adapter=MagicMock())
    voice.config.tokenizer = _tokenizer()
    return voice


def _training_symbols(lang, texts, add_diacritics=False, text_casing="ignore"):
    """The phonemes ``phonemize_worker`` writes for each text, or None for a dropped row."""
    config = PreprocessorConfig(
        input_dir=Path("."), output_dir=Path("."), language=lang, sample_rate=16000,
        cache_dir=Path("."), max_workers=1, single_speaker=True, speaker_id=None,
        phoneme_type=PhonemeType.UNICODE, alphabet=Alphabet.UNICODE, phonemizer_model="",
        text_casing=text_casing, dataset_name=None, audio_quality=None, skip_audio=True,
        debug=False, add_diacritics=add_diacritics)
    tasks, results = queue.Queue(), queue.Queue()
    tasks.put([Utterance(text=text, audio_path=Path("a.wav")) for text in texts])
    tasks.put(None)
    make_frontend = partial(preprocess.build_frontend, config.phoneme_type, config.alphabet,
                            None, False, lang, add_diacritics)
    with patch.object(preprocess, "make_silence_detector", return_value=MagicMock()):
        preprocess.phonemize_worker(config, tasks, results, make_frontend)
    rows = []
    while not results.empty():
        utt, _ = results.get()
        rows.append(utt.phonemes if utt else None)
    return rows


def _synthesis_ids(voice, text, chunk=None):
    syn = replace(SynthesisConfig(), enable_phonetic_spellings=False, chunk=chunk)
    return [list(u.ids) for u in voice.prepare(text, syn)]


class TestTrainingIdsEqualSynthesisIds(unittest.TestCase):
    ROWS = {
        # digits, an abbreviation and punctuation inside one sentence
        "en": ["Dr. Smith paid 3 dollars, then left!", "It is 5 o'clock (almost)."],
        "pt": ["O Dr. Silva tem 21 anos, e viu 3 gatos.", "Que dia lindo!"],
    }

    def test_one_sentence_rows(self):
        for lang, texts in self.ROWS.items():
            voice = _voice(lang)
            symbols = _training_symbols(lang, texts)
            for text, row in zip(texts, symbols):
                with self.subTest(lang=lang, text=text):
                    self.assertEqual([_tokenizer().tokenize(row)], _synthesis_ids(voice, text))

    def test_a_row_of_two_sentences_reads_as_one_chunk(self):
        text = "Dr. Smith paid 3 dollars. He left."
        [row] = _training_symbols("en", [text])
        self.assertEqual([_tokenizer().tokenize(row)], _synthesis_ids(_voice("en"), text, chunk="none"))

    def test_diacritics_are_added_per_sentence_as_synthesis_adds_them(self):
        text = "عندي 3 قطط. هذا بيت."
        voice = _voice("ar", add_diacritics=True)
        with patch.object(scd, "_tashkeel", lambda *_, **__: _Fatha()):
            [row] = _training_symbols("ar", [text], add_diacritics=True)
            expected = _synthesis_ids(voice, text, chunk="none")
        self.assertEqual([_tokenizer().tokenize(row)], expected)
        self.assertEqual(row.count("َ"), 2)

    def test_a_row_whose_diacritization_fails_is_dropped(self):
        class _FailsOnCats(_Fatha):
            def diacritize(self, text):
                if "قطط" in text:
                    raise RuntimeError("diacritizer failed")
                return super().diacritize(text)

        with patch.object(scd, "_tashkeel", lambda *_, **__: _FailsOnCats()):
            rows = _training_symbols("ar", ["هذا بيت.", "عندي قطط."], add_diacritics=True)
        self.assertEqual([row is None for row in rows], [False, True])

    def test_a_row_is_normalised_and_cased_to_a_known_text(self):
        # The readings are written by hand: the unicode backend gives one symbol per character.
        cases = [("en", "ignore", "Dr. Smith paid 3 dollars.", "Doctor Smith paid three dollars"),
                 ("en", "lower", "Dr. Smith paid 3 dollars.", "doctor smith paid three dollars"),
                 ("pt", "ignore", "O Dr. Silva tem 21 anos.", "O Doutor Silva tem vinte e um anos")]
        for lang, casing, text, reading in cases:
            with self.subTest(lang=lang, casing=casing):
                self.assertEqual(_training_symbols(lang, [text], text_casing=casing), [list(reading)])

    def test_the_scorer_reads_text_as_the_dataset_does(self):
        from phoonnx.config import get_phonemizer, voice_frontend
        text = "Dr. Smith paid 3 dollars, then left!"
        frontend = voice_frontend(get_phonemizer(PhonemeType.UNICODE, Alphabet.UNICODE), "en")
        phonemes, ids = text_to_ids(text, frontend, _tokenizer(), "en")
        [row] = _training_symbols("en", [text])
        self.assertEqual(phonemes, row)
        self.assertEqual(ids, _tokenizer().tokenize(row))

    def test_the_scorer_diacritizes_a_config_written_by_preprocess(self):
        from phoonnx_train.evaluation.scorer import build_encoder
        config = {"lang_code": "ar", "phoneme_type": "unicode", "alphabet": "unicode",
                  "phonemizer_model": "", "keep_punctuation": False, "audio": {"sample_rate": 22050},
                  "inference": {"noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8,
                                "add_diacritics": True},
                  "phoneme_id_map": {"_": [0], "^": [1], "$": [2]}, "num_symbols": 3}
        text = "عندي 3 قطط. هذا بيت."
        with patch.object(scd, "_tashkeel", lambda *_, **__: _Fatha()):
            frontend, *_ = build_encoder(config)
            phonemes, _ = text_to_ids(text, frontend, _tokenizer(), "ar")
        self.assertEqual(phonemes.count("َ"), 2)


if __name__ == "__main__":
    unittest.main()
