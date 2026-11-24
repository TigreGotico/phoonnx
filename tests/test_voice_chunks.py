"""TTSVoice cuts text into chunks after the frontend's stages, at the size the call names.

The contract (``speech-stack-contract.md``, section "The order of the stages")
says: "Every path through ``TextFrontend`` reads the text first and cuts it into
chunks after", and "The caller chooses the chunk size per call, with
``Stages(chunk=...)``". A phonemizer with side streams (Shami's language ids,
dizione's stress) takes the same path as every other phonemizer, so its
streams stay in step with its symbols through ``TextFrontend.segments``.

Every expected value comes from outside the voice text path: symbols written by
hand from the contract's table of chunk sizes, or the Shami phonemizer called
alone on text2spoken's normalised text. tests/test_tinyvoca.py holds the
dizione side of the same path.
"""
import types
from unittest.mock import MagicMock

import pytest
import text2spoken
from scriptconv.frontend import Stages
from scriptconv.phonemizers.shami import ShamiPhonemizer

from phoonnx.config import Alphabet, PhonemeType, SynthesisConfig, VoiceConfig
from phoonnx.voice import TTSVoice


def _unicode_voice(normalise: bool = False):
    """A voice whose symbols are the characters of the text.

    With ``normalise``, the voice keeps its text2spoken normaliser; without it,
    the voice has no normalisation hook and reads the text as written.
    """
    config = VoiceConfig(num_symbols=1, num_speakers=1, num_langs=1, sample_rate=22050,
                         lang_code="en", phoneme_type=PhonemeType.UNICODE, alphabet=Alphabet.UNICODE,
                         phonemizer_model=None, add_diacritics=False, keep_punctuation=False)
    voice = TTSVoice(config=config, session=MagicMock(), adapter=MagicMock())
    voice.config.tokenizer = types.SimpleNamespace(tokenize=lambda phonemes: list(range(len(phonemes))))
    if not normalise:
        voice.phonemizer.normalizer = None
    return voice


def _shami_voice():
    config = VoiceConfig(num_symbols=1, num_speakers=1, num_langs=1, sample_rate=22050,
                         lang_code="ar", phoneme_type=PhonemeType.SHAMI, alphabet=Alphabet.IPA,
                         phonemizer_model=None, add_diacritics=False, keep_punctuation=False)
    voice = TTSVoice(config=config, session=MagicMock(), adapter=MagicMock())
    voice.config.tokenizer = types.SimpleNamespace(
        tokenize=lambda phonemes: list(range(len(phonemes))),
        tokenize_with_streams=lambda phonemes, streams: (list(range(len(phonemes))), streams))
    return voice


def _utterances(voice, text, **stages):
    return list(voice.prepare(text, SynthesisConfig(phonetic_spellings=False, stages=Stages(**stages))))


class TestChunkSize:
    """``SynthesisConfig.stages.chunk`` reaches the frontend: one utterance per chunk."""

    # quebra_frases reads a word with no vowel before a period as an
    # abbreviation ("Dr.") and does not end the sentence there
    TEXT = "One, two. Three\n\nFour."

    @pytest.mark.parametrize("chunk, expected", [
        (None, ["One two", "Three", "Four"]),
        ("sentence", ["One two", "Three", "Four"]),
        ("clause", ["One", "two", "Three", "Four"]),
        ("paragraph", ["One two Three", "Four"]),
        ("none", ["One two Three Four"]),
    ])
    def test_one_utterance_per_chunk(self, chunk, expected):
        utterances = _utterances(_unicode_voice(), self.TEXT, chunk=chunk)
        assert ["".join(u.symbols) for u in utterances] == expected

    def test_paragraph_and_none_differ_on_a_voice_that_normalises(self):
        """text2spoken keeps the blank line between paragraphs, so the paragraph cut still sees it.
        The digit read as "three" shows that the normaliser ran."""
        text = "Ab, cd. Ef.\n\nGh 3."
        voice = _unicode_voice(normalise=True)
        paragraph = ["".join(u.symbols) for u in _utterances(voice, text, chunk="paragraph")]
        whole = ["".join(u.symbols) for u in _utterances(voice, text, chunk="none")]
        assert paragraph == ["Ab cd Ef", "Gh three"]
        assert whole == ["Ab cd Ef Gh three"]

    def test_an_unknown_size_raises(self):
        with pytest.raises(ValueError):
            _utterances(_unicode_voice(), self.TEXT, chunk="word")


class TestShamiStageOrder:
    """Shami's language ids follow the contract order: normalise the whole text, then cut."""

    TEXT = "عندي 12 قطة. I have 3 cats."

    def test_the_normaliser_reads_the_whole_text_once(self):
        voice = _shami_voice()
        seen = []
        voice.phonemizer.normalizer = lambda text, lang: seen.append(text) or text2spoken.normalize(text, lang)
        _utterances(voice, self.TEXT)
        assert seen == [self.TEXT]

    def test_language_ids_are_those_of_the_normalised_sentences(self):
        utterances = _utterances(_shami_voice(), self.TEXT)
        expected = list(ShamiPhonemizer().phonemize_with_language_ids_lazy(text2spoken.normalize(self.TEXT, "ar"), "ar"))
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
