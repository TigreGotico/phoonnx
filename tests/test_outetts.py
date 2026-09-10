"""OuteTTS 1.0 adapter tests.

Everything the adapter does around the model — text normalization, chunking, prompt
assembly, speaker resolution, the KV-cached decode loop, the DAC hand-off and the
loudness stage — is observable without the real weights, so the ONNX sessions here are
fakes that record their feeds.
"""
import copy
import json

import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.outetts import (
    AUDIO_END, AUDIO_START, BOS, C1, C2, CODE, CODEBOOK_TOKENS, EOS, FEATURES,
    REPETITION_WINDOW, SAMPLE_RATE, TEXT_END, TEXT_START, WORD_END, WORD_START,
    OuteTTSAdapter, _apply_repetition_penalty, _sample, chunk_text,
    integrated_loudness, normalize_loudness, split_into_sentences, text_normalizations,
)

LAYERS = 2
KV_HEADS = 4
HEAD_DIM = 8
VOCAB = 64

# fake vocabulary: 0..7 are text/special, 8..(8+N) c1, then c2
C1_BASE = 8
C2_BASE = 8 + 8
AUDIO_END_ID = 5
EOS_ID = 6
WORD_START_ID = 7
N_CODES = 8  # codebook size in the fake tokenizer


SPEAKER = {
    "text": "Hello there",
    "global_features": {"energy": 13, "spectral_centroid": 20, "pitch": 28},
    "interface_version": "3",
    "words": [
        {"word": "Hello", "duration": 0.2, "c1": [1, 2], "c2": [3, 4],
         "features": {"energy": 10, "spectral_centroid": 15, "pitch": 45}},
        {"word": "there", "duration": 0.35, "c1": [5], "c2": [6],
         "features": {"energy": 11, "spectral_centroid": 16, "pitch": 46}},
    ],
}


def _req(prompt_ids=(1, 2, 3), **params):
    ids = np.asarray(prompt_ids, np.int64).reshape(1, -1)
    return AdapterSynthesisRequest(phoneme_ids=ids,
                                   phoneme_lengths=np.array([ids.shape[1]], np.int64),
                                   speaker_id=0, language_id=0, params=params)


class _IO:
    def __init__(self, name, shape=None, type_="tensor(float)"):
        self.name = name
        self.shape = shape or []
        self.type = type_


class _FakeSession:
    """Minimal onnxruntime.InferenceSession stand-in: named IO + a feed log."""

    def __init__(self, output_names, fn, input_specs=()):
        self._outs = [_IO(n) for n in output_names]
        self._ins = [_IO(n, s) for n, s in input_specs]
        self._fn = fn
        self.feeds = []

    def get_outputs(self):
        return self._outs

    def get_inputs(self):
        return self._ins

    def run(self, _none, feed):
        self.feeds.append(feed)
        named = self._fn(feed, len(self.feeds) - 1)
        return [named[o.name] for o in self._outs]


class _FakeTokenizer:
    """One id per character for plain text; codec/special tokens map to fixed ids."""

    SPECIALS = {AUDIO_END: AUDIO_END_ID, EOS: EOS_ID, WORD_START: WORD_START_ID}

    def token_to_id(self, token):
        if token in self.SPECIALS:
            return self.SPECIALS[token]
        for template, base in ((C1, C1_BASE), (C2, C2_BASE)):
            for i in range(N_CODES):
                if token == template.format(i):
                    return base + i
        return None

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text, add_special_tokens=False):
        # id 0 stands in for "some text token"; only the count matters to the tests
        return self._Enc([0] * len(text))


def _lm_names():
    return ["logits"] + [f"present.{i}.{k}" for i in range(LAYERS) for k in ("key", "value")]


def _lm_inputs():
    specs = [("input_ids", [1, "seq"]), ("attention_mask", [1, "total"]),
             ("position_ids", [1, "seq"])]
    for i in range(LAYERS):
        for k in ("key", "value"):
            specs.append((f"past_key_values.{i}.{k}", [1, KV_HEADS, "past", HEAD_DIM]))
    return specs


def _lm_session(tokens, end_token=AUDIO_END_ID):
    """A fake LM that emits ``tokens`` one per step, then ``end_token``.

    Its KV outputs carry a per-layer fingerprint and grow by the number of tokens fed,
    so the decode loop's present -> past wiring and cache growth are both observable.
    """
    def fn(feed, i):
        seq = feed["input_ids"].shape[1]
        past = feed["past_key_values.0.key"].shape[2]
        # logits for every position; only the last row is the next-token distribution
        logits = np.full((1, seq, VOCAB), -10.0, np.float32)
        want = tokens[i] if i < len(tokens) else end_token
        logits[0, -1, want] = 10.0
        out = {"logits": logits}
        for j in range(LAYERS):
            for k in ("key", "value"):
                block = np.full((1, KV_HEADS, past + seq, HEAD_DIM),
                                float(j) + (0.5 if k == "value" else 0.0), np.float32)
                out[f"present.{j}.{k}"] = block
        return out
    return _FakeSession(_lm_names(), fn, _lm_inputs())


def _adapter(tokens=(C1_BASE + 1, C2_BASE + 2), speakers=None):
    a = OuteTTSAdapter()
    a.tokenizer = _FakeTokenizer()
    a._build_token_maps()
    a.speakers = speakers if speakers is not None else {"spk": copy.deepcopy(SPEAKER)}
    a.default_speaker = "spk" if speakers is None else None
    return a


# ----------------------------------------------------------------------------------
# text normalization
# ----------------------------------------------------------------------------------
def test_normalization_collapses_ellipsis_and_dashes():
    assert text_normalizations("a… b—c −d") == "a... b-c -d"


def test_normalization_unifies_quotes_and_drops_double_quotes():
    # curly singles become ASCII apostrophes; double quotes are removed entirely, and
    # the trailing-apostrophe rule then pulls the closing quote onto the word
    assert text_normalizations("“le ‘mot’ ”") == "le'mot'"


def test_normalization_repairs_split_contractions():
    assert text_normalizations("can 't stop") == "can't stop"


def test_normalization_spaces_after_punctuation():
    assert text_normalizations("hi ,there.Now") == "hi, there. Now"


def test_normalization_collapses_repeated_terminators():
    assert text_normalizations("what?!!! really???") == "what?! really?"


def test_normalization_strips_control_and_zero_width():
    assert text_normalizations("a​b\x07c") == "abc"


def test_normalization_handles_non_strings():
    assert text_normalizations(None) == ""
    assert text_normalizations(7) == "7"


# ----------------------------------------------------------------------------------
# chunking
# ----------------------------------------------------------------------------------
def test_split_into_sentences_keeps_terminators():
    assert split_into_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_chunk_text_returns_short_text_whole():
    assert chunk_text("Just a few words here.") == ["Just a few words here."]


def test_chunk_text_never_exceeds_max_words():
    text = " ".join(f"word{i}" for i in range(100)) + "."
    chunks = chunk_text(text, max_words=10)
    assert chunks and all(len(c.split()) <= 10 for c in chunks)
    assert " ".join(chunks) == text


def test_chunk_text_preserves_every_word():
    text = ". ".join(" ".join(f"w{i}{j}" for j in range(12)) for i in range(4)) + "."
    joined = " ".join(chunk_text(text, max_words=15))
    for token in text.replace(".", " ").split():
        assert token in joined


def test_chunk_text_packs_short_sentences_together():
    text = "One two three. Four five six. Seven eight nine."
    assert chunk_text(text, min_words=3, max_words=30) == [text]


def test_chunk_text_splits_when_max_reached():
    text = "One two three four five. Six seven eight nine ten."
    chunks = chunk_text(text, min_words=5, max_words=5)
    assert len(chunks) == 2


def test_chunk_text_empty_input():
    assert chunk_text("   ") == []


def test_chunk_text_cjk_splits_on_characters():
    # no MeCab in phoonnx: CJK is counted per character, which is a tighter bound
    chunks = chunk_text("你好世界你好世界你好世界。", max_words=4)
    assert chunks
    assert all(len(c) <= 4 for c in chunks)


# ----------------------------------------------------------------------------------
# speaker text merging
# ----------------------------------------------------------------------------------
def test_merge_adds_period_when_speaker_text_unterminated():
    merged, sep = OuteTTSAdapter().merge_speaker_text("World", "Hello")
    assert merged == "Hello. World"
    assert sep == "."


def test_merge_adds_only_a_space_when_already_terminated():
    merged, sep = OuteTTSAdapter().merge_speaker_text("World", "Hello!")
    assert merged == "Hello! World"
    assert sep == ""


def test_merge_uses_ideographic_period_for_cjk():
    merged, sep = OuteTTSAdapter().merge_speaker_text("世界", "你好")
    assert merged == "你好。世界"
    assert sep == "。"


def test_merge_with_empty_speaker_text():
    merged, sep = OuteTTSAdapter().merge_speaker_text("World", "")
    assert merged == "World"
    assert sep == ""


# ----------------------------------------------------------------------------------
# prompt assembly
# ----------------------------------------------------------------------------------
def test_prompt_without_speaker_is_bare_text_block():
    prompt = OuteTTSAdapter().build_prompt("Hello world")
    assert prompt == f"{BOS}\n{TEXT_START}Hello world{TEXT_END}\n{AUDIO_START}\n"


def test_prompt_with_speaker_ends_on_an_open_word():
    a = _adapter()
    prompt = a.build_prompt("World", copy.deepcopy(SPEAKER))
    assert prompt.startswith(f"{BOS}\n{TEXT_START}Hello there. World{TEXT_END}\n{AUDIO_START}\n")
    assert prompt.endswith("\n" + WORD_START)


def test_prompt_word_block_has_the_exact_upstream_layout():
    a = _adapter()
    prompt = a.build_prompt("World", copy.deepcopy(SPEAKER))
    expected = (WORD_START + "Hello" + FEATURES + "<|t_0.20|>"
                + "<|energy_10|><|spectral_centroid_15|><|pitch_45|>"
                + CODE + C1.format(1) + C2.format(3) + C1.format(2) + C2.format(4)
                + WORD_END)
    assert expected in prompt


def test_prompt_appends_the_separator_to_the_last_speaker_word():
    a = _adapter()
    prompt = a.build_prompt("World", copy.deepcopy(SPEAKER))
    assert WORD_START + "there." + FEATURES in prompt


def test_build_prompt_does_not_mutate_the_profile():
    """Upstream appends the separator in place, corrupting the profile for later calls."""
    a = _adapter()
    profile = copy.deepcopy(SPEAKER)
    a.build_prompt("One", profile)
    a.build_prompt("Two", profile)
    assert profile["words"][-1]["word"] == "there"


def test_repeated_prompts_are_identical():
    a = _adapter()
    first = a.build_prompt("One", a.speakers["spk"])
    second = a.build_prompt("One", a.speakers["spk"])
    assert first == second


def test_prompt_normalizes_the_target_text():
    a = OuteTTSAdapter()
    assert "..." in a.build_prompt("wait…")
    assert "…" not in a.build_prompt("wait…")


# ----------------------------------------------------------------------------------
# speaker resolution
# ----------------------------------------------------------------------------------
class _Cfg:
    def __init__(self, **params):
        self.engine_params = params


class _Voice:
    def __init__(self, **params):
        self.config = _Cfg(**params)


class _Syn:
    def __init__(self, **extra):
        self.extra_params = extra


def test_per_call_voice_wins_over_the_pinned_one():
    a = _adapter(speakers={"a": SPEAKER, "b": SPEAKER})
    a.default_speaker = "a"
    picked = a.resolve_speaker(_Voice(voice="a"), _Syn(voice="b"))
    assert picked is a.speakers["b"]


def test_pinned_voice_wins_over_the_bundle_default():
    a = _adapter(speakers={"a": SPEAKER, "b": SPEAKER})
    a.default_speaker = "a"
    assert a.resolve_speaker(_Voice(voice="b"), _Syn()) is a.speakers["b"]


def test_bundle_default_is_the_last_resort():
    a = _adapter(speakers={"a": SPEAKER})
    a.default_speaker = "a"
    assert a.resolve_speaker(_Voice(), None) is a.speakers["a"]


def test_unknown_speaker_name_raises_and_lists_the_options():
    a = _adapter(speakers={"a": SPEAKER})
    with pytest.raises(ValueError, match="nope"):
        a.resolve_speaker(_Voice(), _Syn(voice="nope"))


def test_no_speaker_configured_returns_none():
    a = _adapter(speakers={})
    assert a.resolve_speaker(_Voice(), None) is None


# ----------------------------------------------------------------------------------
# configure / token maps
# ----------------------------------------------------------------------------------
def test_configure_loads_a_bare_upstream_profile(tmp_path):
    path = tmp_path / "spk.json"
    path.write_text(json.dumps(SPEAKER), encoding="utf-8")
    a = OuteTTSAdapter()
    a.configure(_Cfg(speakers_path=str(path)))
    assert a.default_speaker == "default"
    assert a.speakers["default"]["text"] == "Hello there"


def test_configure_loads_a_phoonnx_bundle(tmp_path):
    path = tmp_path / "voices.json"
    path.write_text(json.dumps({"default_voice": "b", "speakers": {"a": SPEAKER,
                                                                  "b": SPEAKER}}),
                    encoding="utf-8")
    a = OuteTTSAdapter()
    a.configure(_Cfg(speakers_path=str(path)))
    assert a.default_speaker == "b"
    assert sorted(a.speakers) == ["a", "b"]


def test_token_maps_cover_both_codebooks_and_specials():
    a = _adapter()
    assert a.c1 == {C1_BASE + i: i for i in range(N_CODES)}
    assert a.c2 == {C2_BASE + i: i for i in range(N_CODES)}
    assert (a.audio_end_id, a.eos_id, a.word_start_id) == (AUDIO_END_ID, EOS_ID,
                                                           WORD_START_ID)


def test_token_maps_skip_ids_the_tokenizer_does_not_have():
    a = _adapter()
    # the fake tokenizer only knows N_CODES entries out of CODEBOOK_TOKENS
    assert len(a.c1) < CODEBOOK_TOKENS


# ----------------------------------------------------------------------------------
# encode_text
# ----------------------------------------------------------------------------------
def test_encode_text_returns_one_id_list_per_chunk():
    a = _adapter()
    text = ". ".join(" ".join(f"w{i}{j}" for j in range(12)) for i in range(3)) + "."
    ids = a.encode_text(text, _Voice(), _Syn(max_chunk_words=12))
    assert len(ids) >= 3
    assert all(isinstance(seq, list) and seq for seq in ids)


def test_encode_text_honours_the_per_call_word_budget():
    a = _adapter()
    text = " ".join(f"w{i}" for i in range(60)) + "."
    few = a.encode_text(text, _Voice(), _Syn(max_chunk_words=30))
    many = a.encode_text(text, _Voice(), _Syn(max_chunk_words=5))
    assert len(many) > len(few)


def test_encode_text_without_a_tokenizer_raises():
    a = OuteTTSAdapter()
    with pytest.raises(RuntimeError, match="tokenizer_path"):
        a.encode_text("hi", _Voice(), None)


# ----------------------------------------------------------------------------------
# sampling
# ----------------------------------------------------------------------------------
def test_repetition_penalty_divides_positive_and_multiplies_negative():
    scores = np.array([2.0, -2.0, 5.0], np.float32)
    out = _apply_repetition_penalty(scores, np.array([0, 1]), 2.0)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(-4.0)
    assert out[2] == pytest.approx(5.0)


def test_repetition_penalty_of_one_is_a_no_op():
    scores = np.array([2.0, -2.0], np.float32)
    assert np.array_equal(_apply_repetition_penalty(scores, np.array([0, 1]), 1.0), scores)


def test_repetition_penalty_ignores_out_of_range_ids():
    scores = np.array([2.0, 3.0], np.float32)
    out = _apply_repetition_penalty(scores, np.array([-1, 99]), 2.0)
    assert np.array_equal(out, scores)


def test_zero_temperature_is_argmax():
    scores = np.array([0.0, 3.0, 1.0], np.float32)
    assert _sample(scores, 0.0, 0, 1.0, 0.0, None) == 1


def test_top_k_restricts_the_candidate_set():
    scores = np.array([5.0, 4.0, -20.0, -30.0], np.float32)
    rng = np.random.default_rng(0)
    picked = {_sample(scores, 1.0, 2, 1.0, 0.0, rng) for _ in range(200)}
    assert picked == {0, 1}


def test_top_p_keeps_the_token_that_crosses_the_threshold():
    # probabilities are ~0.525 / 0.475 / ~0: the second token is what crosses 0.6
    scores = np.array([10.0, 9.9, -30.0], np.float32)
    rng = np.random.default_rng(0)
    assert {_sample(scores, 1.0, 0, 0.6, 0.0, rng) for _ in range(200)} == {0, 1}
    assert {_sample(scores, 1.0, 0, 0.4, 0.0, rng) for _ in range(200)} == {0}


def test_min_p_drops_tokens_far_below_the_top():
    scores = np.array([10.0, 4.0, 0.0], np.float32)
    rng = np.random.default_rng(0)
    picked = {_sample(scores, 1.0, 0, 1.0, 0.5, rng) for _ in range(200)}
    assert picked == {0}


def test_temperature_is_applied_before_truncation():
    """HuggingFace's order, not llama.cpp's.

    With temperature first, a low temperature sharpens the distribution *before* top-p
    looks at it, so nucleus sampling keeps fewer tokens. Under llama.cpp's order top-p
    would see the raw distribution and the candidate set would not depend on
    temperature at all.
    """
    scores = np.array([2.0, 1.0, 0.0], np.float32)
    rng = np.random.default_rng(0)
    cold = {_sample(scores, 0.1, 0, 0.9, 0.0, rng) for _ in range(300)}
    hot = {_sample(scores, 5.0, 0, 0.9, 0.0, rng) for _ in range(300)}
    assert cold == {0}
    assert len(hot) > len(cold)


def test_sampling_is_reproducible_for_a_seed():
    scores = np.array([1.0, 1.0, 1.0], np.float32)
    first = [_sample(scores, 1.0, 0, 1.0, 0.0, np.random.default_rng(7)) for _ in range(5)]
    second = [_sample(scores, 1.0, 0, 1.0, 0.0, np.random.default_rng(7)) for _ in range(5)]
    assert first == second


# ----------------------------------------------------------------------------------
# the KV-cached decode loop
# ----------------------------------------------------------------------------------
def test_prefill_feeds_empty_caches_and_absolute_positions():
    a = _adapter()
    session = _lm_session([C1_BASE + 1])
    a.generate(session, [1, 2, 3, 4], a.default_params(), np.random.default_rng(0))
    first = session.feeds[0]
    assert first["input_ids"].tolist() == [[1, 2, 3, 4]]
    assert first["attention_mask"].shape == (1, 4)
    assert first["position_ids"].tolist() == [[0, 1, 2, 3]]
    assert first["past_key_values.0.key"].shape == (1, KV_HEADS, 0, HEAD_DIM)


def test_decode_steps_feed_one_token_with_the_next_absolute_position():
    a = _adapter()
    session = _lm_session([C1_BASE + 1, C2_BASE + 2, C1_BASE + 3])
    a.generate(session, [1, 2, 3], a.default_params(), np.random.default_rng(0))
    steps = session.feeds[1:]
    assert [f["input_ids"].tolist() for f in steps[:3]] == [[[C1_BASE + 1]],
                                                            [[C2_BASE + 2]],
                                                            [[C1_BASE + 3]]]
    assert [f["position_ids"].tolist() for f in steps[:3]] == [[[3]], [[4]], [[5]]]


def test_attention_mask_covers_past_plus_current_token():
    a = _adapter()
    session = _lm_session([C1_BASE + 1, C2_BASE + 2])
    a.generate(session, [1, 2, 3], a.default_params(), np.random.default_rng(0))
    assert [f["attention_mask"].shape[1] for f in session.feeds] == [3, 4, 5]


def test_present_caches_are_fed_back_as_past():
    """present.<i>.<k> at step n must arrive as past_key_values.<i>.<k> at step n+1."""
    a = _adapter()
    session = _lm_session([C1_BASE + 1, C2_BASE + 2])
    a.generate(session, [1, 2, 3], a.default_params(), np.random.default_rng(0))
    for i in range(LAYERS):
        # the fake session fingerprints each layer/kind, so a crossed wire is visible
        assert session.feeds[1][f"past_key_values.{i}.key"][0, 0, 0, 0] == float(i)
        assert session.feeds[1][f"past_key_values.{i}.value"][0, 0, 0, 0] == float(i) + 0.5


def test_kv_cache_grows_by_one_token_per_step():
    a = _adapter()
    session = _lm_session([C1_BASE + 1, C2_BASE + 2, C1_BASE + 3])
    a.generate(session, [1, 2, 3], a.default_params(), np.random.default_rng(0))
    grew = [f["past_key_values.0.key"].shape[2] for f in session.feeds[1:]]
    assert grew == [3, 4, 5]


def test_kv_geometry_is_read_off_the_graph():
    a = _adapter()
    session = _lm_session([C1_BASE + 1])
    a.generate(session, [1], a.default_params(), np.random.default_rng(0))
    assert (a.num_layers, a.num_kv_heads, a.head_dim) == (LAYERS, KV_HEADS, HEAD_DIM)


def test_audio_end_stops_generation_and_is_not_emitted():
    a = _adapter()
    session = _lm_session([C1_BASE + 1, C2_BASE + 2])
    out = a.generate(session, [1, 2], a.default_params(), np.random.default_rng(0))
    assert out == [C1_BASE + 1, C2_BASE + 2]


def test_eos_also_stops_generation():
    a = _adapter()
    session = _lm_session([C1_BASE + 1], end_token=EOS_ID)
    out = a.generate(session, [1, 2], a.default_params(), np.random.default_rng(0))
    assert out == [C1_BASE + 1]


def test_length_cap_stops_a_model_that_never_ends():
    a = _adapter()
    session = _lm_session([C1_BASE + 1] * 50)
    params = {**a.default_params(), "max_new_tokens": 5}
    out = a.generate(session, [1, 2], params, np.random.default_rng(0))
    assert len(out) == 5


def test_only_the_last_logit_row_drives_the_next_token():
    """The export returns logits for every prefill position, not just the last one."""
    a = _adapter()

    def fn(feed, i):
        seq = feed["input_ids"].shape[1]
        past = feed["past_key_values.0.key"].shape[2]
        logits = np.full((1, seq, VOCAB), -10.0, np.float32)
        # every earlier row votes for a different token than the last one
        logits[0, :, C2_BASE + 7] = 50.0
        logits[0, -1, :] = -10.0
        logits[0, -1, C1_BASE + 1 if i == 0 else AUDIO_END_ID] = 10.0
        out = {"logits": logits}
        for j in range(LAYERS):
            for k in ("key", "value"):
                out[f"present.{j}.{k}"] = np.zeros((1, KV_HEADS, past + seq, HEAD_DIM),
                                                   np.float32)
        return out

    session = _FakeSession(_lm_names(), fn, _lm_inputs())
    assert a.generate(session, [1, 2, 3], a.default_params(),
                      np.random.default_rng(0)) == [C1_BASE + 1]


def test_a_last_row_only_export_drives_the_same_loop():
    """phoonnx's own 1B export returns ``logits[1, 1, V]``, not every position.

    ``scripts/conversion/outetts/export_outetts_onnx.py`` drops the prefill rows the
    sampler never reads. ``logits[0, -1]`` is the last row either way, so one adapter
    drives both that graph and OuteAI's full-sequence exports.
    """
    a = _adapter()

    def fn(feed, i):
        seq = feed["input_ids"].shape[1]
        past = feed["past_key_values.0.key"].shape[2]
        logits = np.full((1, 1, VOCAB), -10.0, np.float32)
        logits[0, 0, C1_BASE + 1 if i == 0 else AUDIO_END_ID] = 10.0
        out = {"logits": logits}
        for j in range(LAYERS):
            for k in ("key", "value"):
                out[f"present.{j}.{k}"] = np.zeros((1, KV_HEADS, past + seq, HEAD_DIM),
                                                   np.float32)
        return out

    session = _FakeSession(_lm_names(), fn, _lm_inputs())
    assert a.generate(session, [1, 2, 3], a.default_params(),
                      np.random.default_rng(0)) == [C1_BASE + 1]
    # the prefill still fed the whole prompt, only the returned logits are narrower
    assert session.feeds[0]["input_ids"].shape == (1, 3)


def test_repetition_penalty_window_spans_the_prompt():
    """Upstream's patched processor penalises prompt tokens too, within the window."""
    a = _adapter()
    seen = {}

    def fn(feed, i):
        seq = feed["input_ids"].shape[1]
        past = feed["past_key_values.0.key"].shape[2]
        logits = np.zeros((1, seq, VOCAB), np.float32)
        logits[0, -1, AUDIO_END_ID] = 1.0
        logits[0, -1, 1] = 1.0  # token 1 is in the prompt
        out = {"logits": logits}
        for j in range(LAYERS):
            for k in ("key", "value"):
                out[f"present.{j}.{k}"] = np.zeros((1, KV_HEADS, past + seq, HEAD_DIM),
                                                   np.float32)
        return out

    session = _FakeSession(_lm_names(), fn, _lm_inputs())
    params = {**a.default_params(), "temperature": 0.0, "repetition_penalty": 2.0}
    # token 1 sits in the prompt, so its score is halved and AUDIO_END wins outright
    assert a.generate(session, [1, 1, 1], params, np.random.default_rng(0)) == []
    assert REPETITION_WINDOW == 64


# ----------------------------------------------------------------------------------
# codec hand-off
# ----------------------------------------------------------------------------------
def test_token_ids_to_codes_splits_the_two_codebooks():
    a = _adapter()
    ids = [C1_BASE + 1, C2_BASE + 2, C1_BASE + 3, C2_BASE + 4]
    assert a.token_ids_to_codes(ids) == [[1, 3], [2, 4]]


def test_token_ids_to_codes_drops_a_trailing_half_frame():
    a = _adapter()
    ids = [C1_BASE + 1, C2_BASE + 2, C1_BASE + 3]
    assert a.token_ids_to_codes(ids) == [[1], [2]]


def test_token_ids_to_codes_ignores_non_codec_tokens():
    a = _adapter()
    ids = [WORD_START_ID, C1_BASE + 1, 0, C2_BASE + 2, WORD_START_ID]
    assert a.token_ids_to_codes(ids) == [[1], [2]]


def _codec_session(record):
    def fn(feed, _i):
        codes = feed["audio_codes"]
        record.append(codes.shape)
        n = codes.shape[-1]
        return {"audio_values": np.ones((1, 1, n * 512), np.float32)}
    return _FakeSession(["audio_values"], fn, [("audio_codes", [1, 2, "t"])])


def test_decode_codes_feeds_a_two_codebook_int64_tensor():
    a = _adapter()
    shapes = []
    a.codec = _codec_session(shapes)
    a.decode_codes([[1, 2, 3], [4, 5, 6]])
    assert shapes == [(1, 2, 3)]


def test_decode_codes_chunks_long_streams():
    a = _adapter()
    shapes = []
    a.codec = _codec_session(shapes)
    n = a.DECODE_CHUNK + 10
    a.decode_codes([list(range(n)), list(range(n))])
    assert [s[-1] for s in shapes] == [a.DECODE_CHUNK, 10]


def test_decode_codes_returns_mono_float32():
    a = _adapter()
    a.codec = _codec_session([])
    audio = a.decode_codes([[1, 2], [3, 4]])
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert audio.shape[0] == 2 * 512


def test_decode_codes_of_an_empty_stream_is_silence():
    a = _adapter()
    a.codec = _codec_session([])
    assert a.decode_codes([[], []]).shape == (0,)


def test_decode_chunks_are_faded_in_and_out():
    a = _adapter()
    a.codec = _codec_session([])
    audio = a.decode_codes([[1] * 100, [1] * 100])
    fade = int(SAMPLE_RATE * a.FADE_SECONDS)
    assert abs(audio[0]) < abs(audio[fade])
    assert abs(audio[-1]) < abs(audio[-fade])


# ----------------------------------------------------------------------------------
# loudness
# ----------------------------------------------------------------------------------
def _sine(seconds=3.0, amplitude=0.5, freq=1000.0):
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_integrated_loudness_tracks_amplitude_by_the_decibel():
    loud = integrated_loudness(_sine(amplitude=0.5))
    quiet = integrated_loudness(_sine(amplitude=0.25))
    assert loud - quiet == pytest.approx(6.02, abs=0.05)


def test_integrated_loudness_of_silence_is_minus_infinity():
    assert integrated_loudness(np.zeros(SAMPLE_RATE, np.float32)) == -np.inf


def test_integrated_loudness_of_a_too_short_clip_is_minus_infinity():
    assert integrated_loudness(_sine(seconds=0.1)) == -np.inf


def test_normalize_loudness_hits_the_target():
    out = normalize_loudness(_sine(amplitude=0.2), target=-18.0)
    assert integrated_loudness(out) == pytest.approx(-18.0, abs=0.1)


def test_normalize_loudness_respects_the_peak_ceiling():
    out = normalize_loudness(_sine(amplitude=0.01), target=0.0, peak_limit=-1.0)
    assert np.max(np.abs(out)) == pytest.approx(10 ** (-1.0 / 20.0), rel=1e-6)


def test_normalize_loudness_keeps_the_original_length():
    short = _sine(seconds=0.05)
    assert normalize_loudness(short).shape == short.shape


def test_normalize_loudness_passes_silence_through():
    out = normalize_loudness(np.zeros(SAMPLE_RATE, np.float32))
    assert np.all(out == 0.0)


def test_normalize_loudness_of_an_empty_array():
    assert normalize_loudness(np.zeros(0, np.float32)).shape == (0,)


# ----------------------------------------------------------------------------------
# synthesize
# ----------------------------------------------------------------------------------
def test_synthesize_runs_the_loop_and_the_codec():
    a = _adapter()
    a.codec = _codec_session([])
    session = _lm_session([C1_BASE + 1, C2_BASE + 2, C1_BASE + 3, C2_BASE + 4])
    result = a.synthesize(_req((1, 2, 3), temperature=0.0, seed=1), session)
    assert result.extras["frames"] == 2
    assert result.audio.shape[0] == 2 * 512


def test_synthesize_without_a_codec_raises():
    a = _adapter()
    with pytest.raises(RuntimeError, match="codec_decoder_path"):
        a.synthesize(_req(), _lm_session([]))


def test_synthesize_without_a_tokenizer_raises():
    a = OuteTTSAdapter()
    a.codec = _codec_session([])
    with pytest.raises(RuntimeError, match="tokenizer_path"):
        a.synthesize(_req(), _lm_session([]))


def test_synthesize_rejects_a_reference_clip():
    a = _adapter()
    a.codec = _codec_session([])
    with pytest.raises(RuntimeError, match="speaker profiles"):
        a.synthesize(_req(reference_audio=np.zeros(10)), _lm_session([]))


def test_synthesize_without_any_codec_tokens_raises():
    a = _adapter()
    a.codec = _codec_session([])
    session = _lm_session([])  # ends immediately
    with pytest.raises(RuntimeError, match="no codec tokens"):
        a.synthesize(_req(temperature=0.0), session)


def test_synthesize_is_deterministic_for_a_seed():
    def run():
        a = _adapter()
        a.codec = _codec_session([])
        return a.synthesize(_req((1, 2, 3), seed=42),
                            _lm_session([C1_BASE + 1, C2_BASE + 2])).audio
    assert np.array_equal(run(), run())


# ----------------------------------------------------------------------------------
# registration
# ----------------------------------------------------------------------------------
def test_detect_matches_only_a_declared_outetts_voice():
    assert OuteTTSAdapter.detect(config={"engine": "outetts"})
    assert not OuteTTSAdapter.detect(config={"engine": "neutts"})
    assert not OuteTTSAdapter.detect(config=None)


def test_engine_is_registered_and_probed_after_its_siblings():
    from phoonnx.engines import get_adapter, list_engines
    assert "outetts" in list_engines()
    assert isinstance(get_adapter("outetts"), OuteTTSAdapter)


def test_config_selects_the_adapter_for_an_outetts_voice():
    from phoonnx.engines import detect_engine
    assert isinstance(detect_engine(config={"engine": "outetts"}), OuteTTSAdapter)


def test_default_params_match_the_published_generation_config():
    p = OuteTTSAdapter().default_params()
    assert p["temperature"] == 0.4
    assert p["top_k"] == 40.0
    assert p["top_p"] == 0.9
    assert p["min_p"] == 0.05
    assert p["repetition_penalty"] == 1.1


def test_every_default_param_has_a_label():
    a = OuteTTSAdapter()
    assert set(a.default_params()) <= set(a.param_labels())


def test_the_abc_hooks_refuse_the_feed_dict_path():
    a = OuteTTSAdapter()
    with pytest.raises(NotImplementedError):
        a.build_feed_dict(_req(), None)
    with pytest.raises(NotImplementedError):
        a.parse_outputs([], _req())
