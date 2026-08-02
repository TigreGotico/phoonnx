"""NeuTTS adapter tests.

Everything the adapter does around the model — phonemization with punctuation kept,
prompt assembly, preset resolution, chunking, the KV-cached decode loop and the codec
hand-off — is observable without the real weights, so the ONNX sessions here are fakes
that record their feeds.
"""
import json

import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.neutts import (
    ESPEAK_VOICE, MAX_REFERENCE_CODES, SPEECH_GENERATION_START, SPEECH_TOKEN_TEMPLATE,
    TEXT_PROMPT_END, TEXT_PROMPT_START, NeuTTSAdapter, _apply_repetition_penalty,
    _sample, preserve_punctuation_phonemize,
)

LAYERS = 2
KV_HEADS = 4
HEAD_DIM = 8
VOCAB = 64

# fake vocabulary: 0..9 are text/special, 10.. are speech tokens
SPEECH_BASE = 10
END_ID = 5
EOS_ID = 6


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
    """One id per character for plain text; speech/special tokens map to fixed ids."""

    SPECIALS = {TEXT_PROMPT_START: 1, TEXT_PROMPT_END: 2, SPEECH_GENERATION_START: 3,
                "<|SPEECH_GENERATION_END|>": END_ID, "<|endoftext|>": EOS_ID}

    def token_to_id(self, token):
        if token in self.SPECIALS:
            return self.SPECIALS[token]
        if token == SPEECH_TOKEN_TEMPLATE.format(0):
            return SPEECH_BASE
        return None

    def id_to_token(self, tid):
        for name, i in self.SPECIALS.items():
            if i == tid:
                return name
        if tid >= SPEECH_BASE:
            return SPEECH_TOKEN_TEMPLATE.format(tid - SPEECH_BASE)
        return "x"

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text, add_special_tokens=False):
        # id 0 stands in for "some text token"; only the count matters to the tests
        return self._Enc([0] * len(text))


def _lm_names():
    return ["logits"] + [f"present_{k}_{i}" for i in range(LAYERS) for k in ("key", "value")]


def _lm_inputs():
    specs = [("input_ids", [1, "seq"]), ("attention_mask", [1, "total"]),
             ("position_ids", [1, "seq"])]
    for i in range(LAYERS):
        for k in ("key", "value"):
            specs.append((f"past_{k}_{i}", [1, KV_HEADS, "past", HEAD_DIM]))
    return specs


def _lm_session(tokens):
    """A fake LM that emits ``tokens`` one per step, then the end token."""
    def fn(feed, i):
        seq = feed["input_ids"].shape[1]
        past = feed["past_key_0"].shape[2]
        logits = np.full((1, VOCAB), -10.0, np.float32)
        want = tokens[i] if i < len(tokens) else END_ID
        logits[0, want] = 10.0
        out = {"logits": logits}
        for j in range(LAYERS):
            out[f"present_key_{j}"] = np.zeros((1, KV_HEADS, past + seq, HEAD_DIM), np.float32)
            out[f"present_value_{j}"] = np.zeros((1, KV_HEADS, past + seq, HEAD_DIM), np.float32)
        return out
    return _FakeSession(_lm_names(), fn, _lm_inputs())


def _codec_session(samples_per_code=480):
    def fn(feed, _i):
        n = feed["codes"].shape[-1]
        return {"audio": np.full((1, 1, n * samples_per_code), 0.25, np.float32)}
    return _FakeSession(["audio"], fn, [("codes", [1, 1, "n"])])


VOICES_JSON = {
    "meta": {"spec": "vieneu.voice.presets"},
    "default_voice": "ama",
    "presets": {
        "ama": {"text": "meda wo ase", "codes": list(range(300)), "description": "d"},
        "kofi": {"text": "akwaaba", "codes": [7, 8, 9], "description": "d"},
    },
}


@pytest.fixture
def adapter(tmp_path):
    ad = NeuTTSAdapter()
    voices = tmp_path / "voices.json"
    voices.write_text(json.dumps(VOICES_JSON))

    class VC:
        engine_params = {"voices_path": str(voices)}

    ad.configure(VC())
    ad.tokenizer = _FakeTokenizer()
    ad.speech_end_id = END_ID
    ad.eos_id = EOS_ID
    ad.speech_token_base = SPEECH_BASE
    ad.codec = _codec_session()
    # deterministic stand-in for espeak so the tests never shell out
    ad._phonemizer = type("P", (), {
        "phonemize_string": staticmethod(lambda t, lang: f"[{lang}:{t}]")})()
    return ad


# ----------------------------------------------------------------- registration
def test_registered():
    from phoonnx.engines import list_engines
    assert "neutts" in list_engines()


def test_detect():
    assert NeuTTSAdapter.detect({"engine": "neutts"})
    assert not NeuTTSAdapter.detect({"engine": "mosstts"})
    assert not NeuTTSAdapter.detect(None)
    assert not NeuTTSAdapter.detect({})


def test_engine_enum_and_config_detection():
    from phoonnx.config import Engine, VoiceConfig
    from scriptconv.phonemizers.enums import Alphabet
    assert Engine.NEUTTS.value == "neutts"
    cfg = VoiceConfig.from_dict({"engine": "neutts"}, lang_code="tw")
    assert cfg.engine == Engine.NEUTTS
    # GRAPHEMES routes text -> ids through the adapter's own espeak + BPE path
    assert cfg.alphabet == Alphabet.GRAPHEMES
    assert cfg.sample_rate == 24000


def test_default_params_match_upstream_cli_defaults():
    p = NeuTTSAdapter().default_params()
    assert p["temperature"] == 0.4
    assert p["top_p"] == 0.8
    assert p["repetition_penalty"] == 1.3
    assert set(NeuTTSAdapter().param_labels()) >= set(p)


# ------------------------------------------------------------------ phonemization
def test_punctuation_is_preserved_around_phonemes():
    out = preserve_punctuation_phonemize("meda wo ase paa.", lambda t: "P")
    assert out == "P."


def test_internal_punctuation_splits_the_espeak_calls():
    seen = []

    def fake(text):
        seen.append(text)
        return text.upper()

    out = preserve_punctuation_phonemize("anopa yi, eye fe paa!", fake)
    assert seen == ["anopa yi", "eye fe paa"]
    assert out == "ANOPA YI, EYE FE PAA!"


def test_unpunctuated_text_is_a_single_call():
    seen = []
    preserve_punctuation_phonemize("meda wo ase", lambda t: seen.append(t) or "P")
    assert seen == ["meda wo ase"]


def test_espeak_uses_the_voice_the_checkpoint_was_trained_with(adapter):
    assert adapter._espeak("meda") == f"[{ESPEAK_VOICE}:meda]"
    assert ESPEAK_VOICE == "lfn"


# --------------------------------------------------------------------- presets
def test_presets_and_default_are_read_from_voices_json(adapter):
    assert sorted(adapter.presets) == ["ama", "kofi"]
    assert adapter.default_preset == "ama"


def test_preset_falls_back_to_the_json_default(adapter):
    assert adapter.resolve_preset(None, None)["text"] == "meda wo ase"


def test_voice_config_pins_a_preset(adapter):
    voice = type("V", (), {"config": type("C", (), {"engine_params": {"voice": "kofi"}})})
    assert adapter.resolve_preset(voice, None)["text"] == "akwaaba"


def test_per_call_voice_wins_over_the_pinned_one(adapter):
    voice = type("V", (), {"config": type("C", (), {"engine_params": {"voice": "kofi"}})})
    syn = type("S", (), {"extra_params": {"voice": "ama"}})
    assert adapter.resolve_preset(voice, syn)["text"] == "meda wo ase"


def test_unknown_preset_is_rejected(adapter):
    syn = type("S", (), {"extra_params": {"voice": "nobody"}})
    with pytest.raises(ValueError, match="unknown NeuTTS voice preset"):
        adapter.resolve_preset(None, syn)


def test_reference_phones_are_cached(adapter):
    calls = []
    adapter._phonemizer = type("P", (), {
        "phonemize_string": staticmethod(lambda t, lang: calls.append(t) or "P")})()
    adapter.reference_phones("kofi", VOICES_JSON["presets"]["kofi"])
    adapter.reference_phones("kofi", VOICES_JSON["presets"]["kofi"])
    assert len(calls) == 1


# ---------------------------------------------------------------------- prompt
def test_prompt_layout_matches_the_training_format(adapter):
    prompt = adapter.build_prompt("TGT", "REF", [1, 2])
    assert prompt == (f"{TEXT_PROMPT_START}REF TGT{TEXT_PROMPT_END}"
                      f"{SPEECH_GENERATION_START}<|speech_1|><|speech_2|>")


def test_prompt_without_a_reference_omits_the_speech_prefix(adapter):
    assert adapter.build_prompt("TGT") == (
        f"{TEXT_PROMPT_START}TGT{TEXT_PROMPT_END}{SPEECH_GENERATION_START}")


def test_reference_codes_are_truncated(adapter):
    prompt = adapter.build_prompt("T", "R", list(range(500)))
    assert prompt.count("<|speech_") == MAX_REFERENCE_CODES


def test_encode_text_builds_one_prompt_per_chunk(adapter):
    ids = adapter.encode_text("hello", None, None)
    assert len(ids) == 1
    assert len(ids[0]) > 0


def test_encode_text_without_a_tokenizer_is_an_error(adapter):
    adapter.tokenizer = None
    with pytest.raises(RuntimeError, match="tokenizer_path"):
        adapter.encode_text("hello", None, None)


# --------------------------------------------------------------------- chunking
def test_short_text_is_one_chunk(adapter):
    assert adapter.chunk_text("Meda wo ase.", 200) == ["Meda wo ase."]


def test_empty_text_yields_no_chunks(adapter):
    assert adapter.chunk_text("   ", 200) == []


def test_sentences_are_packed_up_to_the_budget(adapter):
    text = "aaaa bbbb. cccc dddd. eeee ffff."
    chunks = adapter.chunk_text(text, 22)
    assert all(len(c) <= 22 for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_an_over_long_sentence_is_split_not_dropped(adapter):
    text = " ".join(["word"] * 40)
    chunks = adapter.chunk_text(text, 30)
    assert chunks and all(len(c) <= 30 for c in chunks)
    assert sum(c.count("word") for c in chunks) == 40


# ------------------------------------------------------------------- sampling
def test_greedy_when_temperature_is_zero():
    scores = np.array([1.0, 5.0, 2.0], np.float32)
    assert _sample(scores, 0.0, 0, 0.0, np.random.default_rng(0)) == 1


def test_top_k_of_one_is_deterministic():
    scores = np.array([1.0, 5.0, 2.0], np.float32)
    assert all(_sample(scores, 1.0, 1, 1.0, np.random.default_rng(s)) == 1
               for s in range(5))


def test_repetition_penalty_lowers_seen_positive_scores():
    scores = np.array([2.0, 2.0], np.float32)
    out = _apply_repetition_penalty(scores, np.array([0]), 2.0)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == 2.0


def test_repetition_penalty_raises_the_magnitude_of_negative_scores():
    out = _apply_repetition_penalty(np.array([-2.0], np.float32), np.array([0]), 2.0)
    assert out[0] == pytest.approx(-4.0)


def test_repetition_penalty_of_one_is_a_no_op():
    scores = np.array([2.0, -2.0], np.float32)
    assert np.array_equal(_apply_repetition_penalty(scores, np.array([0, 1]), 1.0), scores)


def test_repetition_penalty_ignores_out_of_range_tokens():
    scores = np.array([2.0], np.float32)
    assert np.array_equal(_apply_repetition_penalty(scores, np.array([9]), 2.0), scores)


# --------------------------------------------------------------- decode loop
def test_generate_stops_at_the_end_token(adapter):
    session = _lm_session([SPEECH_BASE + 1, SPEECH_BASE + 2])
    tokens = adapter.generate(session, [1, 2, 3], adapter.default_params(),
                              np.random.default_rng(0))
    assert tokens == [SPEECH_BASE + 1, SPEECH_BASE + 2]


def test_generate_stops_at_eos(adapter):
    session = _lm_session([SPEECH_BASE + 1, EOS_ID, SPEECH_BASE + 3])
    tokens = adapter.generate(session, [1, 2, 3], adapter.default_params(),
                              np.random.default_rng(0))
    assert tokens == [SPEECH_BASE + 1]


def test_generate_respects_the_token_cap(adapter):
    session = _lm_session([SPEECH_BASE + i for i in range(50)])
    p = {**adapter.default_params(), "max_new_tokens": 3}
    assert len(adapter.generate(session, [1, 2], p, np.random.default_rng(0))) == 3


def test_prefill_feeds_an_empty_cache_and_full_positions(adapter):
    session = _lm_session([SPEECH_BASE + 1])
    adapter.generate(session, [1, 2, 3, 4], adapter.default_params(), np.random.default_rng(0))
    first = session.feeds[0]
    assert first["input_ids"].shape == (1, 4)
    assert first["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert first["position_ids"].tolist() == [[0, 1, 2, 3]]
    assert first["past_key_0"].shape == (1, KV_HEADS, 0, HEAD_DIM)


def test_decode_steps_thread_the_cache_and_advance_positions(adapter):
    session = _lm_session([SPEECH_BASE + 1, SPEECH_BASE + 2, SPEECH_BASE + 3])
    adapter.generate(session, [1, 2, 3, 4], adapter.default_params(), np.random.default_rng(0))
    step = session.feeds[1]
    assert step["input_ids"].tolist() == [[SPEECH_BASE + 1]]
    assert step["position_ids"].tolist() == [[4]]
    assert step["attention_mask"].shape == (1, 5)
    assert step["past_key_0"].shape == (1, KV_HEADS, 4, HEAD_DIM)
    assert session.feeds[2]["past_key_0"].shape == (1, KV_HEADS, 5, HEAD_DIM)


def test_kv_geometry_is_read_off_the_graph(adapter):
    session = _lm_session([SPEECH_BASE + 1])
    adapter.generate(session, [1], adapter.default_params(), np.random.default_rng(0))
    assert (adapter.num_layers, adapter.num_kv_heads, adapter.head_dim) == (
        LAYERS, KV_HEADS, HEAD_DIM)


# ------------------------------------------------------------------- codec
def test_speech_tokens_map_back_to_codec_indices(adapter):
    assert adapter.token_ids_to_codes([SPEECH_BASE + 5, SPEECH_BASE + 9]) == [5, 9]


def test_non_speech_tokens_are_dropped(adapter):
    assert adapter.token_ids_to_codes([1, SPEECH_BASE + 4, END_ID]) == [4]


def test_decode_codes_feeds_the_codec_a_3d_int32_tensor(adapter):
    audio = adapter.decode_codes([1, 2, 3])
    feed = adapter.codec.feeds[0]["codes"]
    assert feed.shape == (1, 1, 3)
    assert feed.dtype == np.int32
    assert audio.ndim == 1 and audio.dtype == np.float32


def test_decode_codes_of_nothing_is_empty(adapter):
    assert adapter.decode_codes([]).shape == (0,)


# --------------------------------------------------------------- synthesize
def test_synthesize_returns_audio_and_a_code_count(adapter):
    session = _lm_session([SPEECH_BASE + 1, SPEECH_BASE + 2])
    result = adapter.synthesize(_req(), session)
    assert result.extras["codes"] == 2
    assert result.audio.shape[0] == 2 * 480


def test_synthesize_without_a_codec_is_an_error(adapter):
    adapter.codec = None
    with pytest.raises(RuntimeError, match="codec_decoder_path"):
        adapter.synthesize(_req(), _lm_session([SPEECH_BASE + 1]))


def test_synthesize_rejects_a_reference_clip(adapter):
    request = _req(reference_audio=(np.zeros(100, np.float32), 24000))
    with pytest.raises(RuntimeError, match="pre-encoded voice presets"):
        adapter.synthesize(request, _lm_session([SPEECH_BASE + 1]))


def test_synthesize_without_speech_tokens_is_an_error(adapter):
    with pytest.raises(RuntimeError, match="no speech tokens"):
        adapter.synthesize(_req(), _lm_session([]))


def test_the_seed_makes_synthesis_reproducible(adapter):
    session = _lm_session([SPEECH_BASE + 1, SPEECH_BASE + 2])
    a = adapter.synthesize(_req(seed=7), session).audio
    b = adapter.synthesize(_req(seed=7), _lm_session([SPEECH_BASE + 1, SPEECH_BASE + 2])).audio
    assert np.array_equal(a, b)


def test_the_static_graph_entry_points_are_refused(adapter):
    with pytest.raises(NotImplementedError):
        adapter.build_feed_dict(_req(), None)
    with pytest.raises(NotImplementedError):
        adapter.parse_outputs([], _req())


# --------------------------------------------------------------- voice index
def test_voice_index_entries_are_well_formed():
    import json as _json
    from pathlib import Path
    import phoonnx
    path = Path(phoonnx.__file__).parent / "voice_index" / "neutts.json"
    entries = _json.loads(path.read_text())
    assert entries
    for vid, entry in entries.items():
        assert entry["voice_id"] == vid
        assert entry["engine"] == "neutts"
        assert entry["alphabet"] == "graphemes"
        assert entry["lang"] == "tw"
        assert entry["engine_options"]["voice"] == vid.rsplit("/", 1)[-1]
        assert set(entry["aux_model_urls"]) == {
            "codec_decoder_path", "tokenizer_path", "voices_path"}


def test_engine_options_reach_engine_params(tmp_path):
    from phoonnx.model_manager import TTSModelInfo
    info = TTSModelInfo(voice_id="neutts/x", lang="tw", model_url="http://example/m.onnx",
                        engine_options={"voice": "kofi"})
    assert info.engine_params()["voice"] == "kofi"
