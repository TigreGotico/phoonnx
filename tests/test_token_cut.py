"""The base adapter's token-level cut: pieces rejoin to the request and stay within the limit."""
import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest, AdapterSynthesisResult, BaseOnnxAdapter

BOS, EOS, GAP, COMMA, SEMI, WORD_A, WORD_B = 1, 2, 3, 4, 5, 6, 7


class _Adapter(BaseOnnxAdapter):
    def __init__(self, limit):
        self.max_input_tokens = limit
        self.bos, self.eos, self.word_gap, self.clause_marks = BOS, EOS, GAP, {COMMA, SEMI}

    def build_feed_dict(self, request, session):
        return {"ids": request.phoneme_ids}

    def parse_outputs(self, outputs, request, output_names=None):
        return AdapterSynthesisResult(audio=outputs[0], extras={"seen": True})

    def default_params(self):
        return {}


def _session():
    session = MagicMock()
    session.get_outputs.return_value = [SimpleNamespace(name="audio")]
    session.run.side_effect = lambda _, feed: [feed["ids"][0].astype(np.float32)]
    return session


def _request(body, **streams):
    ids = [BOS] + body + [EOS]
    return AdapterSynthesisRequest(
        phoneme_ids=np.array([ids], dtype=np.int64),
        phoneme_lengths=np.array([len(ids)], dtype=np.int64),
        **{name: np.array([values], dtype=np.int64) for name, values in streams.items()})


def _sentence(rng):
    body = []
    for _ in range(rng.randint(20, 120)):
        body += [rng.choice([WORD_A, WORD_B]) for _ in range(rng.randint(1, 9))]
        if rng.random() < 0.25:
            body.append(rng.choice([COMMA, SEMI]))
        body.append(GAP)
    return body[:-1]


def _rejoin(pieces, body):
    """The pieces' bodies in order; a word gap may stand between two of them."""
    joined = []
    for piece in pieces:
        joined += piece[1:-1]
        if len(joined) < len(body) and body[len(joined)] == GAP:
            joined.append(GAP)
    return joined[:len(body)]


@pytest.mark.parametrize("seed", range(60))
def test_pieces_rejoin_and_fit(seed):
    rng = random.Random(seed)
    body = _sentence(rng)
    limit = rng.randint(12, 80)
    language_ids = [rng.randint(0, 3) for _ in range(len(body) + 2)]
    stress = [rng.randint(0, 2) for _ in range(len(body) + 2)]
    request = _request(body, language_ids=language_ids, stress=stress)
    pieces = _Adapter(limit)._cut(request)
    ids = [p.phoneme_ids[0].tolist() for p in pieces]
    assert len(ids) > 1
    assert all(len(i) <= limit and i[0] == BOS and i[-1] == EOS for i in ids)
    assert all(p.phoneme_lengths.tolist() == [len(i)] for p, i in zip(pieces, ids))
    assert _rejoin(ids, body) == body
    for name, values in (("language_ids", language_ids), ("stress", stress)):
        streams = [getattr(p, name)[0].tolist() for p in pieces]
        assert [len(s) for s in streams] == [len(i) for i in ids]
        # a stream is cut where its ids are: the pieces' inner values, taken in order, are the
        # request's inner values minus those of the word gaps dropped at a cut
        inner = [v for s in streams for v in s[1:-1]]
        kept = values[1:-1]
        # the request's own bos and eos keep their values; each one the cut adds is 0
        assert streams[0][0] == values[0] and streams[-1][-1] == values[-1]
        assert all(s[-1] == 0 for s in streams[:-1]) and all(s[0] == 0 for s in streams[1:])
        position, expected = 0, []
        for piece in ids:
            size = len(piece) - 2
            expected += kept[position:position + size]
            position += size
            if position < len(body) and body[position] == GAP:
                position += 1
        assert inner == expected


@pytest.mark.parametrize("seed", range(20))
def test_cut_after_the_last_clause_mark_that_fits(seed):
    rng = random.Random(seed)
    body = _sentence(rng)
    limit = rng.randint(12, 80)
    first = _Adapter(limit)._cut(_request(body))[0].phoneme_ids[0].tolist()[1:-1]
    window = body[:limit - 2]
    marks = [i for i, t in enumerate(window) if t in (COMMA, SEMI)]
    if marks:
        assert first == body[:marks[-1] + 1]
    else:
        assert body[len(first)] == GAP and GAP not in body[len(first) + 1:limit - 1]


@pytest.mark.parametrize("seed", range(20))
def test_engine_without_a_limit_is_not_cut(seed):
    request = _request(_sentence(random.Random(seed)), stress=[0] * 1000)
    assert _Adapter(None)._cut(request) == [request]


def test_request_at_the_limit_is_not_cut():
    request = _request([WORD_A] * 8)
    assert _Adapter(10)._cut(request) == [request]


def test_one_word_longer_than_the_limit_raises():
    with pytest.raises(ValueError, match="one word"):
        _Adapter(10)._cut(_request([WORD_A] * 20))


def test_base_default_is_not_cut():
    assert BaseOnnxAdapter.max_input_tokens is None


def test_synthesize_joins_the_pieces_audio():
    body = _sentence(random.Random(3))
    session = _session()
    result = _Adapter(30).synthesize(_request(body), session)
    fed = [call.args[1]["ids"][0].tolist() for call in session.run.call_args_list]
    assert len(fed) > 1
    assert result.audio.tolist() == [float(t) for piece in fed for t in piece]
    assert result.extras == {}


def test_uncut_synthesize_keeps_the_result():
    result = _Adapter(None).synthesize(_request([WORD_A, GAP, WORD_B]), _session())
    assert result.audio.tolist() == [1.0, 6.0, 3.0, 7.0, 2.0]
    assert result.extras == {"seen": True}
