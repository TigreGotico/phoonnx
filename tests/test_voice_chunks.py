"""TTSVoice cuts text into chunks after the frontend's stages, at the size the call names.

The contract (``speech-stack-contract.md``, section "The order of the stages")
says: "Every path through ``TextFrontend`` reads the text first and cuts it into
chunks after", and "The caller chooses the chunk size per call, with
``Stages(chunk=...)``". A phonemizer with side streams (Shami's language ids,
dizione's stress) takes the same path as every other phonemizer, so its
streams stay in step with its symbols through ``TextFrontend.segments``.

Every expected value comes from outside the voice text path: symbols written by
hand from the contract's table of chunk sizes, or the Shami phonemizer called
alone on phoonnx's normalised text. tests/test_tinyvoca.py holds the dizione
side of the same path.
"""
import types
from unittest.mock import MagicMock

import pytest
from scriptconv.phonemizers.shami import ShamiPhonemizer

from phoonnx.config import Alphabet, PhonemeType, SynthesisConfig, VoiceConfig
from phoonnx.util import normalize
from phoonnx.voice import TTSVoice


def _unicode_voice():
    """A voice with no normalisation hook: phoonnx's own normaliser joins the paragraphs of a text."""
    config = VoiceConfig(num_symbols=1, num_speakers=1, num_langs=1, sample_rate=22050,
                         lang_code="en", phoneme_type=PhonemeType.UNICODE, alphabet=Alphabet.UNICODE,
                         phonemizer_model=None, add_diacritics=False, keep_punctuation=False)
    voice = TTSVoice(config=config, session=MagicMock(), adapter=MagicMock())
    voice.config.tokenizer = types.SimpleNamespace(tokenize=lambda phonemes: list(range(len(phonemes))))
    voice.phonemizer.normalizer = None
    return voice


def _shami_voice():
    config = VoiceConfig(num_symbols=1, num_speakers=1, num_langs=1, sample_rate=22050,
                         lang_code="ar", phoneme_type=PhonemeType.SHAMI, alphabet=Alphabet.IPA,
                         phonemizer_model=None, add_diacritics=False, keep_punctuation=False)
    voice = TTSVoice(config=config, session=MagicMock(), adapter=MagicMock())
    voice.config.tokenizer = types.SimpleNamespace(tokenize=lambda phonemes: list(range(len(phonemes))))
    return voice


def _utterances(voice, text, **settings):
    return list(voice.prepare(text, SynthesisConfig(enable_phonetic_spellings=False, **settings)))


class TestChunkSize:
    """``SynthesisConfig.chunk`` reaches ``Stages(chunk=...)``: one utterance per chunk."""

    TEXT = "Ab, cd. Ef.\n\nGh."

    @pytest.mark.parametrize("chunk, expected", [
        (None, ["Ab cd", "Ef", "Gh"]),
        ("sentence", ["Ab cd", "Ef", "Gh"]),
        ("clause", ["Ab", "cd", "Ef", "Gh"]),
        ("paragraph", ["Ab cd Ef", "Gh"]),
        ("none", ["Ab cd Ef Gh"]),
    ])
    def test_one_utterance_per_chunk(self, chunk, expected):
        utterances = _utterances(_unicode_voice(), self.TEXT, chunk=chunk)
        assert ["".join(u.symbols) for u in utterances] == expected

    def test_an_unknown_size_raises(self):
        with pytest.raises(ValueError):
            _utterances(_unicode_voice(), self.TEXT, chunk="word")


class TestShamiStageOrder:
    """Shami's language ids follow the contract order: normalise the whole text, then cut."""

    TEXT = "عندي 12 قطة. I have 3 cats."

    def test_the_normaliser_reads_the_whole_text_once(self):
        voice = _shami_voice()
        seen = []
        voice.phonemizer.normalizer = lambda text, lang: seen.append(text) or normalize(text, lang)
        _utterances(voice, self.TEXT)
        assert seen == [self.TEXT]

    def test_language_ids_are_those_of_the_normalised_sentences(self):
        utterances = _utterances(_shami_voice(), self.TEXT)
        expected = list(ShamiPhonemizer().phonemize_with_language_ids_lazy(normalize(self.TEXT, "ar"), "ar"))
        assert [list(u.symbols) for u in utterances] == [phonemes for phonemes, _ in expected]
        assert [list(u.streams["language_ids"]) for u in utterances] == [ids for _, ids in expected]
        assert {1, 2} <= {i for u in utterances for i in u.streams["language_ids"]}

    def test_a_block_keeps_the_language_ids_of_the_text_around_it(self):
        """The text keeps its own ids, the word break before the block takes Shami's
        neutral id, the block's symbols take 0, and the text after the block starts a sentence."""
        shami = ShamiPhonemizer()
        (before, before_ids), = shami.phonemize_with_language_ids_lazy("مرحبا", "ar")
        (after, after_ids), = shami.phonemize_with_language_ids_lazy("شكرا", "ar")
        neutral = shami.word_break().streams["language_ids"][0]
        utterances = _utterances(_shami_voice(), "مرحبا [[m a]] شكرا")
        assert [list(u.symbols) for u in utterances] == [before[:-1] + [" ", "m", " ", "a"] + before[-1:], after]
        assert [list(u.streams["language_ids"]) for u in utterances] == [
            before_ids[:-1] + [neutral, 0, 0, 0] + before_ids[-1:], after_ids]
