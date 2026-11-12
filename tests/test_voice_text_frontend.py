"""TTSVoice takes text to phonemes through scriptconv's TextFrontend.

The contract (``speech-stack-contract.md``, section phoonnx) says: "``TTSVoice``
builds one ``TextFrontend`` at load, and maps ``SynthesisConfig`` overrides to
``Stages`` for each call." The frontend runs inline blocks, sentences,
normalisation, diacritization and the backend, in that order.

The diacritizer is replaced with a recorder, so no model is needed. Every
expected value comes from a function outside the voice text path: phoonnx's own
normaliser, a backend called alone, or hand-written symbols.
"""
import json
import os
import tempfile
import unittest
import warnings
from unittest.mock import MagicMock, patch

import scriptconv.diacritics as scd
from scriptconv.phonemizers.mul import EspeakPhonemizer
from scriptconv.phonemizers.shami import ShamiPhonemizer

from phoonnx.config import (Alphabet, PhonemeType, SynthesisConfig, VoiceConfig,
                            get_conversion, get_phonemizer)
from phoonnx.util import normalize
from phoonnx.version import VERSION_MAJOR
from phoonnx.voice import TTSVoice

ARABIC_DIGITS = "عندي 3 قطط."


class _RecordingDiacritizer:
    """Stands for text2tashkeel: records each text and adds one fatha."""

    def __init__(self, seen):
        self.seen = seen

    def diacritize(self, text):
        self.seen.append(text)
        return text + "َ"


def _recording_tashkeel(seen):
    return patch.object(scd, "_tashkeel", lambda *_, **__: _RecordingDiacritizer(seen))


def _voice(lang, phoneme_type=PhonemeType.UNICODE, alphabet=Alphabet.UNICODE,
           add_diacritics=False, keep_punctuation=False):
    config = VoiceConfig(num_symbols=1, num_speakers=1, num_langs=1, sample_rate=22050,
                         lang_code=lang, phoneme_type=phoneme_type, alphabet=alphabet,
                         phonemizer_model=None, add_diacritics=add_diacritics,
                         keep_punctuation=keep_punctuation)
    voice = TTSVoice(config=config, session=MagicMock(), adapter=MagicMock())
    voice.phonemes_to_ids = lambda phonemes: list(range(len(phonemes)))
    return voice


def _synthesis_phonemes(voice, text, syn_config=None):
    return [phonemes for phonemes, _ in voice._iter_synthesis_ids(
        text, syn_config or SynthesisConfig(), Alphabet.GRAPHEMES, voice.config.alphabet)]


class TestStageOrder(unittest.TestCase):
    """Normalisation runs before diacritization, as training runs it."""

    def test_digits_reach_the_diacritizer_as_words(self):
        voice = _voice("ar", add_diacritics=True)
        seen = []
        with _recording_tashkeel(seen):
            _synthesis_phonemes(voice, ARABIC_DIGITS)
        self.assertEqual(seen, [normalize(ARABIC_DIGITS, "ar")])
        self.assertNotIn("3", seen[0])

    def test_an_inline_block_never_reaches_the_diacritizer(self):
        voice = _voice("ar", add_diacritics=True)
        seen = []
        with _recording_tashkeel(seen):
            phonemes = _synthesis_phonemes(voice, "عندي [[ab]] قطط.")
        self.assertEqual(seen, ["عندي", "قطط."])
        self.assertEqual(phonemes, [list("عنديَ") + [" ", "a", "b", " "], list("قططَ")])

    def test_the_reference_path_runs_no_diacritizer(self):
        voice = _voice("ar", add_diacritics=True)
        seen = []
        with _recording_tashkeel(seen):
            phonemes = voice.phonemize(ARABIC_DIGITS)
        self.assertEqual(seen, [])
        self.assertEqual(phonemes, [list(normalize("عندي 3 قطط", "ar"))])


class TestInlineBlocks(unittest.TestCase):
    """A block is the shortest ``[[...]]`` on one line, at the place it starts."""

    def test_text_after_an_unclosed_block_is_read(self):
        voice = _voice("en")
        self.assertEqual(voice.phonemize("[[x] y"), [["x", " ", "y"]])

    def test_two_blocks_with_a_space_between_are_one_sentence(self):
        voice = _voice("en")
        self.assertEqual(voice.phonemize("[[a]] [[b]]"), [["a", " ", " ", "b"]])
        self.assertEqual(_synthesis_phonemes(voice, "[[a]] [[b]]"), [["a", " ", " ", "b"]])


class TestTextTokenEngines(unittest.TestCase):
    """An engine that reads text gets diacritics and no normalisation."""

    def test_encode_text_gets_digits_as_written(self):
        voice = _voice("ar", PhonemeType.GRAPHEMES, Alphabet.GRAPHEMES, add_diacritics=True)
        seen = []
        with _recording_tashkeel(seen):
            list(voice._iter_synthesis_ids(ARABIC_DIGITS, SynthesisConfig(),
                                           Alphabet.GRAPHEMES, Alphabet.GRAPHEMES))
        self.assertEqual(seen, [ARABIC_DIGITS])
        voice.adapter.encode_text.assert_called_once()
        self.assertEqual(voice.adapter.encode_text.call_args[0][0], ARABIC_DIGITS + "َ")

    def test_a_per_call_setting_wins_over_the_voice(self):
        voice = _voice("ar", PhonemeType.GRAPHEMES, Alphabet.GRAPHEMES, add_diacritics=True)
        seen = []
        with _recording_tashkeel(seen):
            list(voice._iter_synthesis_ids(ARABIC_DIGITS, SynthesisConfig(add_diacritics=False),
                                           Alphabet.GRAPHEMES, Alphabet.GRAPHEMES))
        self.assertEqual(seen, [])
        self.assertEqual(voice.adapter.encode_text.call_args[0][0], ARABIC_DIGITS)


class TestNormaliser(unittest.TestCase):
    """The voice's normaliser is phoonnx's own, chosen by phoneme type."""

    def test_espeak_keeps_contractions_and_expands_digits(self):
        voice = _voice("en-US", PhonemeType.ESPEAK, Alphabet.IPA)
        text = "I don't have 5 cats"
        expected = EspeakPhonemizer().phonemize(normalize(text, "en-US", expand_contractions=False), "en-US")
        self.assertEqual(_synthesis_phonemes(voice, text), expected)

    def test_the_normaliser_runs_once_per_sentence(self):
        voice = _voice("en")
        seen = []
        voice.phonemizer.normalizer = lambda text, lang: seen.append(text) or text
        self.assertEqual(_synthesis_phonemes(voice, "ab. cd."), [list("ab"), list("cd")])
        self.assertEqual(seen, ["ab.", "cd."])

    def test_shami_reads_digits_with_its_own_front_end(self):
        voice = _voice("ar", PhonemeType.SHAMI, Alphabet.IPA)
        text = "عندي 12 قطة"
        self.assertEqual(_synthesis_phonemes(voice, text, SynthesisConfig(add_diacritics=False)),
                         ShamiPhonemizer().phonemize(text, "ar"))


class TestPhonemizerChanges(unittest.TestCase):
    """A change to ``TTSVoice.phonemizer`` after the first call reaches synthesis."""

    def test_a_vosk_dictionary_set_late_is_read(self):
        """TTSModelInfo.load builds the Vosk phonemizer again once the dictionary is known."""
        from phoonnx.config import _voice_phonemizer
        voice = _voice("ru-RU", PhonemeType.VOSK, Alphabet.VOSK)
        self.assertEqual(_synthesis_phonemes(voice, "привет"), [["p", "rj", "i0", "vj", "e0", "t"]])
        with tempfile.TemporaryDirectory() as tmp:
            dictionary = os.path.join(tmp, "dictionary")
            with open(dictionary, "w", encoding="utf-8") as f:
                f.write("привет pj rj i1 vj e0 t\n")
            voice.phonemizer = _voice_phonemizer(PhonemeType.VOSK, Alphabet.VOSK, dictionary)
            self.assertEqual(_synthesis_phonemes(voice, "привет"),
                             [["pj", "rj", "i1", "vj", "e0", "t"]])

    def test_a_phonemizer_of_another_type_is_read(self):
        from phoonnx.config import _voice_phonemizer
        voice = _voice("ru-RU", PhonemeType.VOSK, Alphabet.VOSK)
        _synthesis_phonemes(voice, "привет")
        voice.phonemizer = _voice_phonemizer(PhonemeType.UNICODE, Alphabet.UNICODE)
        self.assertEqual(_synthesis_phonemes(voice, "привет"), [list("привет")])

    def test_keep_punctuation_set_on_the_phonemizer_is_read(self):
        voice = _voice("en")
        self.assertEqual(_synthesis_phonemes(voice, "ab, cd."), [list("ab cd")])
        voice.phonemizer.keep_punctuation = True
        self.assertEqual(_synthesis_phonemes(voice, "ab, cd."), [list("ab, cd.")])


class TestDeprecatedShims(unittest.TestCase):
    removal = f"{VERSION_MAJOR + 1}.0.0"

    def test_get_phonemizer_warns_and_still_normalises(self):
        with self.assertWarns(DeprecationWarning) as caught:
            phonemizer = get_phonemizer(PhonemeType.UNICODE)
        self.assertIn(f"removed in phoonnx {self.removal}", str(caught.warning))
        self.assertEqual(phonemizer.phonemize("2 ab", "en"), [list(normalize("2 ab", "en"))])

    def test_get_conversion_warns_and_still_converts(self):
        voice = _voice("en")
        with self.assertWarns(DeprecationWarning) as caught:
            graph, prepare = get_conversion(voice.phonemizer, voice.config, SynthesisConfig(),
                                            Alphabet.UNICODE)
        self.assertIn(f"removed in phoonnx {self.removal}", str(caught.warning))
        self.assertEqual(list(graph.convert("ab", "text", "unicode")), [["a", "b"]])
        self.assertEqual(prepare("ab"), "ab")

    def test_a_voice_uses_no_deprecated_name(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            voice = _voice("en")
            self.assertEqual(_synthesis_phonemes(voice, "ab"), [["a", "b"]])
            self.assertEqual(voice.phonemize("ab"), [["a", "b"]])


class TestOldConfigs(unittest.TestCase):
    def test_a_voice_with_no_phonemizer_identity_loads(self):
        config = VoiceConfig.from_dict({"lang_code": "en", "phoneme_type": "unicode",
                                        "alphabet": "unicode",
                                        "phoneme_id_map": {"_": [0], "a": [1], "b": [2]}})
        voice = TTSVoice(config=config, session=MagicMock(), adapter=MagicMock())
        self.assertEqual([ids for _, ids in voice._iter_synthesis_ids(
            "ab", SynthesisConfig(), Alphabet.GRAPHEMES, Alphabet.UNICODE)],
            [voice.phonemes_to_ids(["a", "b"])])

    def test_a_mimic3_config_with_a_string_phonemizer_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            tokens = os.path.join(tmp, "phonemes.txt")
            with open(tokens, "w", encoding="utf-8") as f:
                f.write("0 _\n1 ^\n2 $\n3 #\n4 a\n5 b\n")
            config = VoiceConfig.from_dict(json.loads(json.dumps({
                "phonemizer": "symbols", "text_language": "en",
                "phonemes": {"blank_between": "tokens_and_words"}})), tokens_txt=tokens)
        self.assertEqual(config.phoneme_type, PhonemeType.GRAPHEMES)
        voice = TTSVoice(config=config, session=MagicMock(), adapter=MagicMock())
        self.assertEqual(voice.phonemize("ab"), [["a", "b"]])


if __name__ == "__main__":
    unittest.main()
