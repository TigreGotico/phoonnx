"""Orpheus adapter tests.

Everything the adapter does around the model — the served prompt assembly (including the
double BOS), voice resolution, chunking, the SNAC de-interleave, the vLLM sampler order
and the KV-cached decode loop — is observable without the real weights, so the ONNX
sessions here are fakes that record their feeds.
"""
import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.orpheus import (
    AUDIO_TOKEN_BASE, AUDIO_TOKEN_LAST, BOS, CODEBOOK_SIZE, END_OF_HUMAN, END_OF_SPEECH,
    EOT, START_OF_AI, START_OF_HUMAN, START_OF_SPEECH, TOKENS_PER_FRAME, OrpheusAdapter,
    _apply_repetition_penalty, _sample,
)

LAYERS = 2
KV_HEADS = 4
HEAD_DIM = 8
EN_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]


class _IO:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape or []


class _FakeSession:
    """Minimal onnxruntime.InferenceSession stand-in: named IO + a feed log."""

    def __init__(self, fn):
        self._ins = [_IO("input_ids"), _IO("attention_mask")]
        self._outs = [_IO("logits")]
        for i in range(LAYERS):
            for kind in ("key", "value"):
                self._ins.append(_IO(f"past_key_values.{i}.{kind}",
                                     [1, KV_HEADS, "past", HEAD_DIM]))
                self._outs.append(_IO(f"present.{i}.{kind}"))
        self._fn = fn
        self.feeds = []

    def get_inputs(self):
        return self._ins

    def get_outputs(self):
        return self._outs

    def run(self, _none, feed):
        self.feeds.append(feed)
        return self._fn(feed, len(self.feeds) - 1)


class _FakeTokenizer:
    """One id per character, offset into the text range; BOS is prepended on demand."""

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text, add_special_tokens=False):
        ids = [ord(c) % 1000 + 100 for c in text]
        if add_special_tokens:
            ids = [BOS] + ids
        return self._Enc(ids)


class _FakeSnac:
    """Records the three code streams and returns one sample per code in stream 0."""

    def __init__(self):
        self.feeds = []

    def get_inputs(self):
        return [_IO("audio_codes.0"), _IO("audio_codes.1"), _IO("audio_codes.2")]

    def run(self, _none, feed):
        self.feeds.append(feed)
        n = feed["audio_codes.0"].shape[1]
        return [np.full((1, 1, n * 2048), 0.25, np.float32)]


def _adapter(voices=EN_VOICES, default="tara", snac=None):
    a = OrpheusAdapter()
    a.tokenizer = _FakeTokenizer()
    a.voices = list(voices)
    a.default_voice = default
    a.snac = snac if snac is not None else _FakeSnac()
    return a


class _Cfg:
    def __init__(self, **kw):
        self.engine_params = kw.pop("engine_params", {})
        self.speaker_id_map = kw.pop("speaker_id_map", {})


class _Voice:
    def __init__(self, cfg):
        self.config = cfg


class _Syn:
    def __init__(self, extra_params=None, speaker_id=None):
        self.extra_params = extra_params or {}
        self.speaker_id = speaker_id


# ----------------------------------------------------------------------
# prompt assembly — the served (vLLM) form, not the naive HF one
# ----------------------------------------------------------------------

def test_prompt_reproduces_the_served_double_bos():
    """Upstream decodes its ids back to a string and lets vLLM re-tokenize it, which
    prepends a *second* BOS. Dropping it is a different prompt from the served one."""
    ids = _adapter().build_prompt_ids("hi", "tara")
    assert ids[0] == BOS, "vLLM's own add_special_tokens BOS must lead"
    assert ids[1] == START_OF_HUMAN
    assert ids[2] == BOS, "the upstream tokenizer call's BOS must follow start-of-human"
    assert ids[-4:] == [EOT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]


def test_prompt_writes_the_voice_name_into_the_text():
    """The speaker is a name in the prompt text, not an embedding or an id."""
    tok = _FakeTokenizer()
    with_voice = _adapter().build_prompt_ids("hi", "leo")
    body = tok.encode("leo: hi", add_special_tokens=True).ids
    assert with_voice[2:2 + len(body)] == body


def test_prompt_without_a_voice_omits_the_prefix():
    tok = _FakeTokenizer()
    ids = _adapter().build_prompt_ids("hi", None)
    assert ids[2:2 + len(tok.encode("hi", add_special_tokens=True).ids)] == \
        tok.encode("hi", add_special_tokens=True).ids


def test_emotive_tags_survive_into_the_prompt():
    """<laugh> and friends are ordinary text the checkpoint's BPE encodes — nothing may
    strip or phonemize them away."""
    tok = _FakeTokenizer()
    ids = _adapter().build_prompt_ids("well <laugh> ok", "tara")
    assert tok.encode("tara: well <laugh> ok", add_special_tokens=True).ids[1:] == ids[3:-4]


# ----------------------------------------------------------------------
# voice resolution
# ----------------------------------------------------------------------

def test_voice_precedence_call_over_config_over_default():
    a = _adapter()
    cfg = _Cfg(engine_params={"voice": "mia"})
    assert a.resolve_voice(_Voice(cfg), _Syn({"voice": "zac"})) == "zac"
    assert a.resolve_voice(_Voice(cfg), _Syn()) == "mia"
    assert a.resolve_voice(_Voice(_Cfg()), _Syn()) == "tara"


def test_speaker_id_maps_back_to_a_name():
    """A plain speaker_id must reach the model as the *name*, via speaker_id_map."""
    a = _adapter()
    cfg = _Cfg(speaker_id_map={v: i for i, v in enumerate(EN_VOICES)})
    assert a.resolve_voice(_Voice(cfg), _Syn(speaker_id=3)) == "leo"


def test_unknown_voice_is_rejected():
    with pytest.raises(ValueError, match="unknown Orpheus voice"):
        _adapter().resolve_voice(_Voice(_Cfg()), _Syn({"voice": "nobody"}))


# ----------------------------------------------------------------------
# SNAC de-interleave
# ----------------------------------------------------------------------

def test_token_ids_to_codes_deinterleaves_one_frame():
    """Position i carries a (i % 7) * 4096 offset; the seven codes fill the three
    streams at rates 1 / 2 / 4."""
    want = [11, 22, 33, 44, 55, 66, 77]
    toks = [AUDIO_TOKEN_BASE + i * CODEBOOK_SIZE + c for i, c in enumerate(want)]
    s0, s1, s2 = OrpheusAdapter.token_ids_to_codes(toks)
    assert s0 == [want[0]]
    assert s1 == [want[1], want[4]]
    assert s2 == [want[2], want[3], want[5], want[6]]


def test_token_ids_to_codes_drops_a_partial_trailing_frame():
    toks = [AUDIO_TOKEN_BASE + (i % TOKENS_PER_FRAME) * CODEBOOK_SIZE
            for i in range(TOKENS_PER_FRAME + 3)]
    s0, s1, s2 = OrpheusAdapter.token_ids_to_codes(toks)
    assert (len(s0), len(s1), len(s2)) == (1, 2, 4), "SNAC needs whole frames"


def test_token_ids_to_codes_drops_an_out_of_range_frame():
    good = [AUDIO_TOKEN_BASE + i * CODEBOOK_SIZE for i in range(TOKENS_PER_FRAME)]
    bad = list(good)
    bad[3] = AUDIO_TOKEN_BASE          # wrong offset for position 3 -> negative code
    s0, _, _ = OrpheusAdapter.token_ids_to_codes(good + bad)
    assert s0 == [0], "the corrupt frame is dropped, not clamped"


def test_last_audio_token_is_in_range():
    assert AUDIO_TOKEN_LAST == AUDIO_TOKEN_BASE + TOKENS_PER_FRAME * CODEBOOK_SIZE - 1
    codes = OrpheusAdapter.token_ids_to_codes(
        [AUDIO_TOKEN_BASE + i * CODEBOOK_SIZE + CODEBOOK_SIZE - 1
         for i in range(TOKENS_PER_FRAME)])
    assert codes[0] == [CODEBOOK_SIZE - 1]


# ----------------------------------------------------------------------
# sampler — vLLM order and penalty scope
# ----------------------------------------------------------------------

def test_zero_temperature_is_greedy():
    assert _sample(np.array([1.0, 9.0, 3.0], np.float32), 0.0, 0.8,
                   np.random.default_rng(0)) == 1


def test_top_p_truncates_over_the_tempered_distribution():
    """vLLM tempers first and truncates second, so a low temperature shrinks the nucleus.
    Under llama.cpp's order (truncate first) the same scores keep two candidates."""
    scores = np.array([0.0, 1.0, 2.0], np.float32)
    picks = {_sample(scores, 0.1, 0.8, np.random.default_rng(s)) for s in range(50)}
    assert picks == {2}, "tempering first collapses the nucleus onto the top token"


def test_repetition_penalty_spans_prompt_and_generation():
    """vLLM penalises prompt tokens too — not a window, and not generation-only."""
    scores = np.array([5.0, 5.0, 5.0], np.float32)
    out = _apply_repetition_penalty(scores, np.array([0], np.int64), 2.0)
    assert out[0] == pytest.approx(2.5)
    assert out[1] == 5.0


def test_repetition_penalty_multiplies_negative_scores():
    out = _apply_repetition_penalty(np.array([-4.0], np.float32), np.array([0], np.int64), 2.0)
    assert out[0] == pytest.approx(-8.0)


def test_repetition_penalty_of_one_is_a_noop():
    scores = np.array([1.0, 2.0], np.float32)
    assert np.array_equal(_apply_repetition_penalty(scores, np.array([0, 1]), 1.0), scores)


# ----------------------------------------------------------------------
# KV-cached decode loop
# ----------------------------------------------------------------------

def _logits_emitting(sequence):
    """Build a fake LM that emits ``sequence`` one token per step."""
    def fn(feed, step):
        vocab = AUDIO_TOKEN_LAST + 2
        logits = np.zeros((1, feed["input_ids"].shape[1], vocab), np.float32)
        logits[0, -1, sequence[min(step, len(sequence) - 1)]] = 100.0
        past = feed["past_key_values.0.key"].shape[2]
        grown = past + feed["input_ids"].shape[1]
        out = [logits]
        for i in range(LAYERS):
            for _ in ("key", "value"):
                out.append(np.zeros((1, KV_HEADS, grown, HEAD_DIM), np.float32))
        return out
    return fn


def test_decode_loop_grows_kv_cache_and_attention_mask():
    frame = [AUDIO_TOKEN_BASE + i * CODEBOOK_SIZE for i in range(TOKENS_PER_FRAME)]
    sess = _FakeSession(_logits_emitting(frame + [END_OF_SPEECH]))
    a = _adapter()
    prompt = [1, 2, 3, 4, 5]
    toks = a.generate(sess, prompt, {"temperature": 0.0, "top_p": 1.0,
                                     "repetition_penalty": 1.0, "max_new_tokens": 50},
                      np.random.default_rng(0))
    assert toks == frame, "generation stops at end-of-speech, which is not emitted"
    # prefill sees the whole prompt with an empty cache; each step adds exactly one token
    assert sess.feeds[0]["input_ids"].shape == (1, len(prompt))
    assert sess.feeds[0]["past_key_values.0.key"].shape == (1, KV_HEADS, 0, HEAD_DIM)
    for n, feed in enumerate(sess.feeds[1:], start=1):
        assert feed["input_ids"].shape == (1, 1)
        assert feed["attention_mask"].shape == (1, len(prompt) + n)
        assert feed["past_key_values.0.key"].shape[2] == len(prompt) + n - 1


def test_decode_loop_stops_on_eos():
    sess = _FakeSession(_logits_emitting([AUDIO_TOKEN_BASE, EOT]))
    toks = _adapter().generate(sess, [1, 2], {"temperature": 0.0, "top_p": 1.0,
                                              "repetition_penalty": 1.0,
                                              "max_new_tokens": 20},
                               np.random.default_rng(0))
    assert toks == [AUDIO_TOKEN_BASE]


def test_decode_loop_stops_on_a_non_audio_token():
    """Anything outside the audio range means the run has left the codec stream."""
    sess = _FakeSession(_logits_emitting([AUDIO_TOKEN_BASE, 42]))
    toks = _adapter().generate(sess, [1], {"temperature": 0.0, "top_p": 1.0,
                                           "repetition_penalty": 1.0,
                                           "max_new_tokens": 20},
                               np.random.default_rng(0))
    assert toks == [AUDIO_TOKEN_BASE]


def test_decode_loop_honours_the_token_cap():
    sess = _FakeSession(_logits_emitting([AUDIO_TOKEN_BASE]))
    toks = _adapter().generate(sess, [1], {"temperature": 0.0, "top_p": 1.0,
                                           "repetition_penalty": 1.0,
                                           "max_new_tokens": 9},
                               np.random.default_rng(0))
    assert len(toks) == 9


def test_kv_geometry_is_read_from_the_graph():
    sess = _FakeSession(_logits_emitting([END_OF_SPEECH]))
    a = _adapter()
    a.generate(sess, [1], {"temperature": 0.0, "max_new_tokens": 1}, np.random.default_rng(0))
    assert a.num_kv_heads == KV_HEADS and a.head_dim == HEAD_DIM
    assert len(a.past_names) == LAYERS * 2


# ----------------------------------------------------------------------
# synthesis
# ----------------------------------------------------------------------

def test_synthesize_decodes_a_frame_to_audio():
    frame = [AUDIO_TOKEN_BASE + i * CODEBOOK_SIZE + i for i in range(TOKENS_PER_FRAME)]
    snac = _FakeSnac()
    a = _adapter(snac=snac)
    sess = _FakeSession(_logits_emitting(frame + [END_OF_SPEECH]))
    req = AdapterSynthesisRequest(
        phoneme_ids=np.asarray([[1, 2, 3]], np.int64),
        phoneme_lengths=np.asarray([3], np.int64),
        params={"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0,
                "max_new_tokens": 20})
    res = a.synthesize(req, sess)
    assert res.audio.shape == (2048,), "one SNAC frame is 2048 samples at 24 kHz"
    assert res.extras == {"frames": 1, "tokens": TOKENS_PER_FRAME}
    assert snac.feeds[0]["audio_codes.1"].shape == (1, 2)
    assert snac.feeds[0]["audio_codes.2"].shape == (1, 4)


def test_synthesize_rejects_a_bare_reference_clip():
    a = _adapter()
    req = AdapterSynthesisRequest(phoneme_ids=np.asarray([[1]], np.int64),
                                  phoneme_lengths=np.asarray([1], np.int64),
                                  params={"reference_audio": (np.zeros(10), 24000)})
    with pytest.raises(RuntimeError, match="speaker_reference_text"):
        a.synthesize(req, _FakeSession(_logits_emitting([END_OF_SPEECH])))


def test_synthesize_raises_when_no_complete_frame():
    a = _adapter()
    sess = _FakeSession(_logits_emitting([AUDIO_TOKEN_BASE, END_OF_SPEECH]))
    req = AdapterSynthesisRequest(phoneme_ids=np.asarray([[1]], np.int64),
                                  phoneme_lengths=np.asarray([1], np.int64),
                                  params={"temperature": 0.0, "max_new_tokens": 20})
    with pytest.raises(RuntimeError, match="no complete SNAC frame"):
        a.synthesize(req, sess)


def test_missing_snac_decoder_is_reported():
    a = _adapter()
    a.snac = None
    req = AdapterSynthesisRequest(phoneme_ids=np.asarray([[1]], np.int64),
                                  phoneme_lengths=np.asarray([1], np.int64))
    with pytest.raises(RuntimeError, match="snac_decoder_path"):
        a.synthesize(req, _FakeSession(_logits_emitting([END_OF_SPEECH])))


# ----------------------------------------------------------------------
# text plumbing
# ----------------------------------------------------------------------

def test_chunking_packs_whole_sentences():
    a = _adapter()
    chunks = a.chunk_text("One two three. Four five six. Seven eight nine.", 20)
    assert all(len(c) <= 20 for c in chunks)
    assert " ".join(chunks).split() == \
        "One two three. Four five six. Seven eight nine.".split()


def test_chunking_splits_an_over_long_sentence_on_words():
    a = _adapter()
    chunks = a.chunk_text(" ".join(["word"] * 40), 20)
    assert chunks and all(len(c) <= 20 for c in chunks)
    assert " ".join(chunks).split() == ["word"] * 40


def test_encode_text_returns_one_prompt_per_chunk():
    a = _adapter()
    ids = a.encode_text("One two three. Four five six.", _Voice(_Cfg()),
                        _Syn({"voice": "tara", "max_chunk_chars": 20}))
    assert len(ids) == 2
    assert all(p[0] == BOS and p[-1] == START_OF_SPEECH for p in ids)


def test_defaults_are_the_served_values_not_the_generation_config():
    """generation_config.json says top_p 0.9; the vLLM server overrides it to 0.8."""
    p = OrpheusAdapter().default_params()
    assert p["temperature"] == 0.6
    assert p["top_p"] == 0.8
    assert p["repetition_penalty"] == 1.3


def test_detect_matches_only_the_named_engine():
    assert OrpheusAdapter.detect(config={"engine": "orpheus"}) is True
    assert OrpheusAdapter.detect(config={"engine": "neutts"}) is False
    assert OrpheusAdapter.detect(config=None) is False
