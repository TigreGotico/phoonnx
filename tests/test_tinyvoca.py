"""tiny-voca: the adapter's feed and its three model rules, and the dizione token path.

The ids come from tests/fixtures/tinyvoca_config.json, the vocabulary of
n4ze3m/tiny-voca: <bos> 1, <eos> 2, <sp> 3, "," 5, "." 6, "!" 8, '"' 11, "k" 21, "ə" 43.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from scriptconv.phonemizers.dizione import DizionePhonemizer

from phoonnx.config import SynthesisConfig, VoiceConfig
from phoonnx.engines import detect_engine
from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.tinyvoca import MAX_TOKENS, TinyVocaAdapter
from phoonnx.voice import TTSVoice

CONFIG = json.loads((Path(__file__).parent / "fixtures" / "tinyvoca_config.json").read_text())
BOS, EOS, GAP, COMMA, STOP, BANG, QUOTE, K, SCHWA = 1, 2, 3, 5, 6, 8, 11, 21, 43
INPUTS = ("tokens", "stress", "voice", "speed", "pitch_shift", "pitch_range")


def _config():
    return VoiceConfig.from_dict(json.loads(json.dumps(CONFIG)))


def _adapter():
    adapter = TinyVocaAdapter()
    adapter.configure(_config())
    return adapter


def _session(peak=0.5):
    """A stand-in graph: one sample per token at *peak*, and token i lasts i + 1 frames."""
    session = MagicMock()
    session.get_inputs.return_value = [SimpleNamespace(name=n) for n in INPUTS]
    session.get_outputs.return_value = [SimpleNamespace(name="audio"), SimpleNamespace(name="durations")]
    session.run.side_effect = lambda _, feed: [
        np.full((1, feed["tokens"].shape[1]), peak, np.float32),
        np.arange(1, feed["tokens"].shape[1] + 1, dtype=np.int64)[None, :]]
    return session


def _request(ids, stress=None, **params):
    return AdapterSynthesisRequest(
        phoneme_ids=np.array([ids], dtype=np.int64),
        phoneme_lengths=np.array([len(ids)], dtype=np.int64),
        speaker_id=1, stress=None if stress is None else np.array([stress], dtype=np.int64),
        params=params)


def _fed(session):
    return [call.args[1] for call in session.run.call_args_list]


class TestFeed:
    def test_inputs(self):
        feed = _adapter().build_feed_dict(_request([BOS, K, STOP, EOS], [0, 1, 0, 0], length_scale=2.0,
                                                   pitch_shift=0.058, pitch_range=0.5), _session())
        assert sorted(feed) == sorted(INPUTS)
        assert feed["tokens"].tolist() == [[BOS, K, STOP, EOS]]
        assert feed["stress"].tolist() == [[0, 1, 0, 0]]
        assert feed["voice"].tolist() == [1] and feed["voice"].dtype == np.int64
        assert feed["speed"].tolist() == [0.5] and feed["speed"].dtype == np.float32
        assert feed["pitch_shift"].tolist() == pytest.approx([0.058])
        assert feed["pitch_range"].tolist() == [0.5]

    def test_missing_stress_is_zero(self):
        feed = _adapter().build_feed_dict(_request([BOS, K, STOP, EOS]), _session())
        assert feed["stress"].tolist() == [[0, 0, 0, 0]]


class TestFinalMark:
    def test_full_stop_added_when_no_mark(self):
        session = _session()
        _adapter().synthesize(_request([BOS, K, SCHWA, EOS], [0, 1, 0, 0]), session)
        (feed,) = _fed(session)
        assert feed["tokens"].tolist() == [[BOS, K, SCHWA, STOP, EOS]]
        assert feed["stress"].tolist() == [[0, 1, 0, 0, 0]]

    @pytest.mark.parametrize("end", [[STOP], [COMMA], [BANG, QUOTE]])
    def test_marked_sentence_unchanged(self, end):
        ids = [BOS, K, SCHWA] + end + [EOS]
        session = _session()
        _adapter().synthesize(_request(ids), session)
        assert _fed(session)[0]["tokens"].tolist() == [ids]


class TestLongSentence:
    def test_cut_after_last_clause_mark_that_fits(self):
        first = [K, SCHWA, GAP] * 100 + [COMMA, GAP]    # comma at body index 300
        second = [K, SCHWA, GAP] * 100 + [K, STOP]
        ids = [BOS] + first + second + [EOS]
        stress = [0] + [1, 2, 0] * 100 + [0, 0] + [2, 1, 0] * 100 + [1, 0] + [0]
        assert len(ids) > MAX_TOKENS
        session = _session()
        _adapter().synthesize(_request(ids, stress), session)
        fed = [(f["tokens"][0].tolist(), f["stress"][0].tolist()) for f in _fed(session)]
        assert fed == [([BOS] + first[:-1] + [EOS], [0] + [1, 2, 0] * 100 + [0, 0]),
                       ([BOS] + second + [EOS], [0] + [2, 1, 0] * 100 + [1, 0, 0])]
        assert all(len(piece) <= MAX_TOKENS for piece, _ in fed)

    def test_cut_between_words_without_a_clause_mark(self):
        body = [K, SCHWA, GAP] * 200 + [K, STOP]
        body_stress = [1, 2, 0] * 200 + [1, 0]
        session = _session()
        _adapter().synthesize(_request([BOS] + body + [EOS], [0] + body_stress + [0]), session)
        fed = [(f["tokens"][0].tolist(), f["stress"][0].tolist()) for f in _fed(session)]
        assert len(fed) > 1
        assert all(len(piece) <= MAX_TOKENS and piece[0] == BOS and piece[-1] == EOS for piece, _ in fed)
        joined = [t for piece, _ in fed for t in piece[1:-1] + [GAP]][:-1]
        joined_stress = [s for _, st in fed for s in st[1:-1] + [0]][:-1]
        assert joined == body
        assert joined_stress == body_stress

    def test_clause_mark_at_the_last_place_that_fits(self):
        room = MAX_TOKENS - 2
        # commas at body index room - 1 (fits: a 500-token piece) and room (one too many)
        head = [K] * (room - 1) + [COMMA]
        tail = [COMMA, GAP] + [K, SCHWA, GAP] * 5 + [K, STOP]
        session = _session()
        _adapter().synthesize(_request([BOS] + head + tail + [EOS]), session)
        fed = [f["tokens"][0].tolist() for f in _fed(session)]
        assert fed[0] == [BOS] + head + [EOS] and len(fed[0]) == MAX_TOKENS
        assert fed[1] == [BOS] + tail + [EOS]

    @pytest.mark.parametrize("length, calls", [(MAX_TOKENS, 1), (MAX_TOKENS + 1, 2)])
    def test_limit_is_inclusive(self, length, calls):
        body = ([K, SCHWA, GAP] * MAX_TOKENS)[:length - 3] + [STOP]
        session = _session()
        _adapter().synthesize(_request([BOS] + body + [EOS]), session)
        assert len(_fed(session)) == calls

    def test_one_long_word_raises(self):
        with pytest.raises(ValueError, match="one word"):
            _adapter().synthesize(_request([BOS] + [K] * MAX_TOKENS + [STOP, EOS]), _session())


class TestOutput:
    def test_peak_limited(self):
        result = _adapter().synthesize(_request([BOS, K, STOP, EOS]), _session(peak=2.0))
        assert np.abs(result.audio).max() == pytest.approx(0.99)

    def test_quiet_audio_unchanged(self):
        result = _adapter().synthesize(_request([BOS, K, STOP, EOS]), _session(peak=0.5))
        assert result.audio.tolist() == [0.5] * 4

    def test_durations_follow_the_request_tokens(self):
        adapter = _adapter()
        kept = adapter.synthesize(_request([BOS, K, STOP, EOS]), _session())
        assert kept.extras["phoneme_id_samples"].tolist() == [1, 2, 3, 4]
        # the model reads [BOS, K, STOP, EOS] with 1, 2, 3, 4 frames; the stop's 3 go to <eos>
        finished = adapter.synthesize(_request([BOS, K, EOS]), _session())
        assert finished.extras["phoneme_id_samples"].tolist() == [1, 2, 3 + 4]

    def test_cut_sentence_has_no_durations(self):
        body = [K, SCHWA, GAP] * 200 + [K, STOP]
        result = _adapter().synthesize(_request([BOS] + body + [EOS]), _session())
        assert "phoneme_id_samples" not in result.extras


class TestDetect:
    def test_by_config(self):
        assert isinstance(detect_engine(config=CONFIG), TinyVocaAdapter)

    def test_by_graph_inputs(self):
        assert TinyVocaAdapter.detect(session=_session())
        vits = MagicMock()
        vits.get_inputs.return_value = [SimpleNamespace(name=n) for n in ("input", "input_lengths", "scales")]
        assert not TinyVocaAdapter.detect(session=vits)


class TestDizioneTokens:
    """The voice sends dizione's own ids and stress, one engine call per sentence as dizione cuts it."""

    # "Hello" holds the adjacent ə and l that compound folding would merge into əl
    TEXTS = ["Hello! This is tiny voca.",
             "Please record the record before the concert.",
             "Dr. Patel paid $12.99 for the book",
             "I read the paper yesterday, and I will read the next one tomorrow.",
             "On Jan. 1, 2000 we met Dr. Patel. He paid $5."]

    def _voice(self):
        adapter = MagicMock()
        adapter.default_params.return_value = {}
        adapter.synthesize.return_value = SimpleNamespace(audio=np.zeros(4, np.float32), extras={})
        return TTSVoice(session=MagicMock(), config=_config(), adapter=adapter)

    def _sent(self, text, **settings):
        voice = self._voice()
        list(voice.synthesize(text, SynthesisConfig(normalize_audio=False, **settings)))
        return [(tuple(c.args[0].phoneme_ids[0].tolist()), tuple(c.args[0].stress[0].tolist()))
                for c in voice.adapter.synthesize.call_args_list]

    @pytest.mark.parametrize("text", TEXTS)
    def test_one_call_per_dizione_sentence(self, text):
        assert self._sent(text) == [(s.ids, s.stress) for s in DizionePhonemizer().token_segments(text)]

    @pytest.mark.parametrize("text", TEXTS)
    def test_the_whole_text_is_one_call_of_tokens(self, text):
        expected = DizionePhonemizer().tokens(text)
        assert self._sent(text, chunk="none") == [(expected.ids, expected.stress)]

    def test_an_abbreviation_ends_no_sentence(self):
        """A split of the raw text cuts "Jan." from its date; dizione reads one sentence."""
        expected = DizionePhonemizer().tokens("Jan. 1, 2000")
        assert self._sent("Jan. 1, 2000") == [(expected.ids, expected.stress)]

    def test_two_sentences_are_two_calls(self):
        sentences = DizionePhonemizer().token_segments("On Jan. 1, 2000 we met Dr. Patel. He paid $5.")
        assert len(sentences) == 2
        assert len(self._sent("On Jan. 1, 2000 we met Dr. Patel. He paid $5.")) == 2

    def test_phoonnx_normalizer_left_out(self):
        assert self._voice().phonemizer.normalizer is None

    def test_nothing_to_pronounce_is_skipped(self, caplog):
        with caplog.at_level("WARNING", logger="scriptconv.frontend"):
            assert self._sent("مرحبا") == []
        assert "no phonemes for 'مرحبا'" in caplog.text
