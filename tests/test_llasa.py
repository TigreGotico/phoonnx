"""Llasa adapter tests.

Everything the adapter does around the model — chat-template prompt assembly, preset
resolution, chunking, logit masking, the sampler's warper order, the KV-cached decode
loop and the codec hand-off — is observable without the real weights, so the ONNX
sessions here are fakes that record their feeds.
"""
import json

import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.llasa import (
    MAX_REFERENCE_CODES, NUM_SPEECH_TOKENS, PROMPT_TEMPLATE, TEMPLATE_DATE,
    LlasaAdapter, _sample,
)

LAYERS = 2
KV_HEADS = 4
HEAD_DIM = 8

# fake vocabulary: 0..9 are text/special, 10.. are the speech-token block
SPEECH_BASE = 10
END_ID = 5
VOCAB = SPEECH_BASE + 40


def _req(prompt_ids=(1, 2, 3), **params):
    ids = np.asarray(prompt_ids, np.int64).reshape(1, -1)
    return AdapterSynthesisRequest(phoneme_ids=ids,
                                   phoneme_lengths=np.array([ids.shape[1]], np.int64),
                                   speaker_id=0, language_id=0, params=params)


class _IO:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape or []


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
    """One id per character; only the token count and the specials matter here."""

    SPECIALS = {"<|SPEECH_GENERATION_END|>": END_ID, "<|s_0|>": SPEECH_BASE}

    def token_to_id(self, token):
        return self.SPECIALS.get(token)

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text, add_special_tokens=False):
        self.last = text
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


def _lm_session(tokens):
    """A fake LM that wants ``tokens`` one per step, then the end token."""
    def fn(feed, i):
        seq = feed["input_ids"].shape[1]
        past = feed["past_key_values.0.key"].shape[2]
        logits = np.full((1, 1, VOCAB), -10.0, np.float32)
        want = tokens[i] if i < len(tokens) else END_ID
        logits[0, 0, want] = 10.0
        out = {"logits": logits}
        for j in range(LAYERS):
            shape = (1, KV_HEADS, past + seq, HEAD_DIM)
            out[f"present.{j}.key"] = np.zeros(shape, np.float32)
            out[f"present.{j}.value"] = np.zeros(shape, np.float32)
        return out
    return _FakeSession(_lm_names(), fn, _lm_inputs())


def _codec_session(samples_per_code=320):
    def fn(feed, _i):
        n = feed["codes"].shape[-1]
        return {"audio": np.full((1, n * samples_per_code), 0.25, np.float32)}
    return _FakeSession(["audio"], fn, [("codes", [1, 1, "n"])])


VOICES_JSON = {
    "default_voice": "en_female_a",
    "presets": {
        "en_female_a": {"text": "the morning light", "lang": "en",
                        "codes": list(range(600)), "synthetic": True},
        "zh_male_a": {"text": "今天天气很好", "lang": "zh",
                      "codes": [7, 8, 9], "synthetic": True},
    },
}


@pytest.fixture
def adapter(tmp_path):
    ad = LlasaAdapter()
    voices = tmp_path / "voices.json"
    voices.write_text(json.dumps(VOICES_JSON, ensure_ascii=False))

    class VC:
        engine_params = {"voices_path": str(voices)}

    ad.configure(VC())
    ad.tokenizer = _FakeTokenizer()
    ad.speech_end_id = END_ID
    ad.speech_token_base = SPEECH_BASE
    ad.codec = _codec_session()
    return ad


class _SC:
    def __init__(self, **extra):
        self.extra_params = extra


# ----------------------------------------------------------------- registration
def test_registered():
    from phoonnx.engines import list_engines
    assert "llasa" in list_engines()


def test_detect():
    assert LlasaAdapter.detect({"engine": "llasa"})
    assert not LlasaAdapter.detect({"engine": "neutts"})
    assert not LlasaAdapter.detect(None)
    assert not LlasaAdapter.detect({})


def test_detect_priority_is_15():
    from phoonnx.engines import _PRIORITIES
    assert _PRIORITIES["llasa"] == 15


def test_engine_enum_and_config_detection():
    from phoonnx.config import Engine, VoiceConfig
    from scriptconv.phonemizers.enums import Alphabet
    assert Engine.LLASA.value == "llasa"
    cfg = VoiceConfig.from_dict({"engine": "llasa"}, lang_code="en")
    assert cfg.engine == Engine.LLASA
    # GRAPHEMES routes text -> ids through the adapter's own template + BPE path
    assert cfg.alphabet == Alphabet.GRAPHEMES
    assert cfg.sample_rate == 16000


def test_default_params_match_the_model_card():
    p = LlasaAdapter().default_params()
    assert p["temperature"] == 0.9
    assert p["top_p"] == 0.95
    assert set(LlasaAdapter().param_labels()) >= set(p)


# ------------------------------------------------------------------ prompt
def test_prompt_is_the_llama3_chat_template_with_a_frozen_date():
    prompt = LlasaAdapter().build_prompt("hello there")
    assert prompt.startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>")
    assert f"Today Date: {TEMPLATE_DATE}" in prompt
    assert "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>hello there" \
           "<|TEXT_UNDERSTANDING_END|>" in prompt
    assert prompt.endswith("<|SPEECH_GENERATION_START|>")


def test_prompt_is_stable_across_calls():
    a = LlasaAdapter().build_prompt("same text")
    b = LlasaAdapter().build_prompt("same text")
    assert a == b == PROMPT_TEMPLATE.format(date=TEMPLATE_DATE, text="same text")


def test_reference_transcript_is_prepended_inside_the_text_block():
    prompt = LlasaAdapter().build_prompt("target words", "reference words")
    assert "<|TEXT_UNDERSTANDING_START|>reference words target words" \
           "<|TEXT_UNDERSTANDING_END|>" in prompt


def test_encode_text_appends_preset_codes_as_speech_ids(adapter):
    ids = adapter.encode_text("hello", None, _SC(voice="zh_male_a"))
    assert len(ids) == 1
    # the three preset codes 7, 8, 9 land at the end as base+code
    assert ids[0][-3:] == [SPEECH_BASE + 7, SPEECH_BASE + 8, SPEECH_BASE + 9]


def test_encode_text_truncates_a_long_reference(adapter):
    ids = adapter.encode_text("hello", None, _SC(voice="en_female_a"))
    speech = [i for i in ids[0] if i >= SPEECH_BASE]
    assert len(speech) == MAX_REFERENCE_CODES < len(VOICES_JSON["presets"]["en_female_a"]["codes"])


def test_encode_text_without_a_preset_carries_no_speech_prefix():
    ad = LlasaAdapter()
    ad.tokenizer = _FakeTokenizer()
    ad.speech_token_base = SPEECH_BASE
    ids = ad.encode_text("hello", None, None)
    assert all(i < SPEECH_BASE for i in ids[0])


def test_encode_text_requires_a_tokenizer():
    with pytest.raises(RuntimeError, match="tokenizer_path"):
        LlasaAdapter().encode_text("hi", None, None)


# ------------------------------------------------------------------ presets
def test_default_preset_is_used_when_nothing_is_requested(adapter):
    assert adapter.resolve_preset(None, None)["lang"] == "en"


def test_per_call_voice_wins(adapter):
    assert adapter.resolve_preset(None, _SC(voice="zh_male_a"))["lang"] == "zh"


def test_unknown_preset_is_an_error(adapter):
    with pytest.raises(ValueError, match="unknown Llasa voice preset"):
        adapter.resolve_preset(None, _SC(voice="nope"))


# ------------------------------------------------------------------ chunking
def test_short_text_is_a_single_chunk():
    assert LlasaAdapter().chunk_text("one short sentence.", 300) == ["one short sentence."]


def test_long_text_is_split_on_sentence_boundaries():
    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = LlasaAdapter().chunk_text(text, 30)
    assert len(chunks) > 1
    assert all(len(c) <= 30 for c in chunks)
    assert " ".join(chunks).replace("  ", " ") == text


def test_an_oversized_sentence_is_split_on_words_not_dropped():
    text = "word " * 40
    chunks = LlasaAdapter().chunk_text(text, 25)
    assert chunks
    assert sum(c.count("word") for c in chunks) == 40


def test_empty_text_produces_no_chunks():
    assert LlasaAdapter().chunk_text("   ", 300) == []


# ------------------------------------------------------------------ codes
def test_token_ids_map_back_to_codec_indices(adapter):
    assert adapter.token_ids_to_codes([SPEECH_BASE, SPEECH_BASE + 3]) == [0, 3]


def test_non_speech_tokens_are_discarded(adapter):
    assert adapter.token_ids_to_codes([0, END_ID, SPEECH_BASE + 1]) == [1]


def test_codes_round_trip_through_ids(adapter):
    assert adapter.token_ids_to_codes(adapter.codes_to_token_ids([0, 5, 39])) == [0, 5, 39]


def test_decode_codes_feeds_the_codec_a_3d_int64_tensor(adapter):
    audio = adapter.decode_codes([1, 2, 3])
    assert audio.shape == (3 * 320,)
    fed = adapter.codec.feeds[0]["codes"]
    assert fed.shape == (1, 1, 3)
    assert fed.dtype == np.int64


def test_decode_codes_of_nothing_is_silence(adapter):
    assert adapter.decode_codes([]).shape == (0,)


# ------------------------------------------------------------------ logit masking
def test_masking_keeps_only_the_speech_block(adapter):
    logits = np.arange(VOCAB, dtype=np.float32)
    masked = adapter._mask_logits(logits, allow_end=False)
    assert np.isneginf(masked[:SPEECH_BASE]).all()
    assert np.array_equal(masked[SPEECH_BASE:], logits[SPEECH_BASE:])


def test_masking_admits_the_end_token_only_once_a_run_has_started(adapter):
    logits = np.arange(VOCAB, dtype=np.float32)
    assert np.isneginf(adapter._mask_logits(logits, allow_end=False)[END_ID])
    assert adapter._mask_logits(logits, allow_end=True)[END_ID] == logits[END_ID]


def test_masking_never_reads_past_the_vocabulary(adapter):
    # a checkpoint whose vocabulary stops inside the nominal speech block
    logits = np.zeros(SPEECH_BASE + 4, np.float32)
    assert adapter._mask_logits(logits, allow_end=True).shape == logits.shape


# ------------------------------------------------------------------ sampler
def test_zero_temperature_is_greedy():
    logits = np.array([1.0, 5.0, 2.0])
    rng = np.random.default_rng(0)
    assert _sample(logits, 0.0, 0.95, rng) == 1


def test_top_p_keeps_the_token_that_crosses_the_threshold():
    # softmax at temperature 1 is dominated by index 0; top_p=0.5 must still be able
    # to return it, and must never return the negligible tail
    logits = np.array([10.0, 1.0, 0.0])
    rng = np.random.default_rng(3)
    assert {_sample(logits, 1.0, 0.5, rng) for _ in range(50)} == {0}


def test_temperature_is_applied_before_top_p():
    # HuggingFace order: a hot temperature flattens the distribution first, so top_p
    # then admits more candidates. With the llama.cpp order (truncate first) the
    # candidate set could not grow.
    logits = np.array([4.0, 3.0, 2.0, 1.0])
    rng = np.random.default_rng(11)
    cold = {_sample(logits, 0.2, 0.9, rng) for _ in range(200)}
    hot = {_sample(logits, 5.0, 0.9, rng) for _ in range(200)}
    assert len(hot) > len(cold)


# ------------------------------------------------------------------ decode loop
def test_generate_prefills_then_steps_one_token_at_a_time(adapter):
    session = _lm_session([SPEECH_BASE + 1, SPEECH_BASE + 2])
    rng = np.random.default_rng(0)
    out = adapter.generate(session, [1, 2, 3, 4], {"temperature": 0.0, "top_p": 1.0}, rng)
    assert out == [SPEECH_BASE + 1, SPEECH_BASE + 2]
    prefill, first, second = session.feeds[:3]
    assert prefill["input_ids"].shape == (1, 4)
    assert prefill["past_key_values.0.key"].shape == (1, KV_HEADS, 0, HEAD_DIM)
    assert prefill["position_ids"].tolist() == [[0, 1, 2, 3]]
    assert first["input_ids"].tolist() == [[SPEECH_BASE + 1]]
    assert first["position_ids"].tolist() == [[4]]
    assert first["attention_mask"].shape == (1, 5)
    assert second["position_ids"].tolist() == [[5]]


def test_generate_grows_the_kv_cache_every_step(adapter):
    session = _lm_session([SPEECH_BASE + 1, SPEECH_BASE + 2, SPEECH_BASE + 3])
    rng = np.random.default_rng(0)
    adapter.generate(session, [1, 2, 3], {"temperature": 0.0, "top_p": 1.0}, rng)
    pasts = [f["past_key_values.0.key"].shape[2] for f in session.feeds]
    assert pasts == [0, 3, 4, 5]


def test_generate_reads_the_kv_geometry_off_the_graph(adapter):
    session = _lm_session([SPEECH_BASE + 1])
    adapter.generate(session, [1], {"temperature": 0.0, "top_p": 1.0},
                     np.random.default_rng(0))
    assert (adapter.num_layers, adapter.num_kv_heads, adapter.head_dim) == \
           (LAYERS, KV_HEADS, HEAD_DIM)


def test_generate_stops_at_the_end_token(adapter):
    session = _lm_session([SPEECH_BASE + 1])
    out = adapter.generate(session, [1], {"temperature": 0.0, "top_p": 1.0},
                           np.random.default_rng(0))
    assert out == [SPEECH_BASE + 1]
    assert len(session.feeds) == 2


def test_generate_honours_the_token_cap(adapter):
    # a session that never wants to stop
    session = _lm_session([SPEECH_BASE + 1] * 50)
    out = adapter.generate(session, [1], {"temperature": 0.0, "top_p": 1.0,
                                          "max_new_tokens": 4},
                           np.random.default_rng(0))
    assert len(out) == 4


def test_generate_never_emits_a_text_token(adapter):
    # the fake LM wants token 0 (text) at every step; masking must veto it
    session = _lm_session([0, 0, 0])
    out = adapter.generate(session, [1], {"temperature": 0.0, "top_p": 1.0,
                                          "max_new_tokens": 3},
                           np.random.default_rng(0))
    assert out and all(t >= SPEECH_BASE for t in out)


def test_generate_cannot_end_on_the_first_step(adapter):
    session = _lm_session([END_ID, SPEECH_BASE + 2])
    out = adapter.generate(session, [1], {"temperature": 0.0, "top_p": 1.0},
                           np.random.default_rng(0))
    assert out and out[0] != END_ID


# ------------------------------------------------------------------ synthesize
def test_synthesize_returns_audio_and_the_code_count(adapter):
    session = _lm_session([SPEECH_BASE + 1, SPEECH_BASE + 2])
    result = adapter.synthesize(_req(temperature=0.0, top_p=1.0), session)
    assert result.audio.shape == (2 * 320,)
    assert result.extras["codes"] == 2


def test_synthesize_needs_the_codec():
    ad = LlasaAdapter()
    ad.tokenizer = _FakeTokenizer()
    with pytest.raises(RuntimeError, match="codec_decoder_path"):
        ad.synthesize(_req(), _lm_session([]))


def test_synthesize_needs_the_tokenizer():
    ad = LlasaAdapter()
    ad.codec = _codec_session()
    with pytest.raises(RuntimeError, match="tokenizer_path"):
        ad.synthesize(_req(), _lm_session([]))


def test_a_reference_clip_is_refused_with_a_pointer_to_presets(adapter):
    with pytest.raises(RuntimeError, match="pre-encoded voice presets"):
        adapter.synthesize(_req(reference_audio=(np.zeros(16000), 16000)), _lm_session([]))


def test_seed_makes_synthesis_reproducible(adapter):
    def noisy():
        def fn(feed, i):
            seq = feed["input_ids"].shape[1]
            past = feed["past_key_values.0.key"].shape[2]
            rng = np.random.default_rng(i)
            logits = rng.normal(size=(1, 1, VOCAB)).astype(np.float32)
            out = {"logits": logits}
            for j in range(LAYERS):
                shape = (1, KV_HEADS, past + seq, HEAD_DIM)
                out[f"present.{j}.key"] = np.zeros(shape, np.float32)
                out[f"present.{j}.value"] = np.zeros(shape, np.float32)
            return out
        return _FakeSession(_lm_names(), fn, _lm_inputs())

    first = adapter.synthesize(_req(seed=42, max_new_tokens=8), noisy()).audio
    second = adapter.synthesize(_req(seed=42, max_new_tokens=8), noisy()).audio
    assert np.array_equal(first, second)


def test_synthesize_resamples_when_the_run_yields_no_speech_tokens(adapter, monkeypatch):
    """token_ids_to_codes() coming back empty is foreclosed by the step-0 end-token
    mask in generate() (see test_generate_cannot_end_on_the_first_step), but the
    MAX_ATTEMPTS retry in synthesize() is belt-and-braces for anything that reaches
    it anyway. Force it directly so the fallback path itself is under test rather
    than relying on sampler luck.
    """
    calls = []

    def flaky(token_ids):
        calls.append(token_ids)
        if len(calls) < adapter.MAX_ATTEMPTS:
            return []
        return [1, 2]

    monkeypatch.setattr(adapter, "token_ids_to_codes", flaky)
    session = _lm_session([SPEECH_BASE + 1, SPEECH_BASE + 2])
    result = adapter.synthesize(_req(temperature=0.0, top_p=1.0), session)
    assert len(calls) == adapter.MAX_ATTEMPTS
    assert result.extras["codes"] == 2
    assert result.audio.shape == (2 * 320,)


def test_synthesize_raises_after_max_attempts_of_no_speech_tokens(adapter, monkeypatch):
    """If every attempt comes back empty, synthesize() must not silently return
    empty/silent audio — it raises, naming the fallback exhausted."""
    monkeypatch.setattr(adapter, "token_ids_to_codes", lambda token_ids: [])
    session = _lm_session([SPEECH_BASE + 1, SPEECH_BASE + 2])
    with pytest.raises(RuntimeError, match="no speech tokens"):
        adapter.synthesize(_req(temperature=0.0, top_p=1.0), session)


def test_preset_missing_codes_key_degrades_to_no_reference(adapter):
    """A malformed preset entry (e.g. hand-edited voices.json) missing the "codes"
    key must not crash prompt assembly — it degrades to a text-only reference,
    same as a preset with an explicitly empty code list."""
    adapter.presets["broken"] = {"text": "hello there"}  # no "codes" key at all
    ids_list = adapter.encode_text("hi", voice=None, syn_config=_SC(voice="broken"))
    assert ids_list  # still produces at least one chunk of prompt ids
    assert isinstance(ids_list[0], list)


def test_build_feed_dict_and_parse_outputs_are_refused():
    ad = LlasaAdapter()
    with pytest.raises(NotImplementedError):
        ad.build_feed_dict(_req(), None)
    with pytest.raises(NotImplementedError):
        ad.parse_outputs([], _req())


def test_speech_block_size_matches_the_codebook():
    assert NUM_SPEECH_TOKENS == 65536
