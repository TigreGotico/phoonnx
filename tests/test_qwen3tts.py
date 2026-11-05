import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.qwen3tts import (
    CODEC_BOS,
    CODEC_EOS,
    CODEC_NOTHINK,
    CODEC_PAD,
    CODEC_THINK,
    CODEC_THINK_BOS,
    CODEC_THINK_EOS,
    DECODE_CHUNK,
    DECODE_LEFT_CONTEXT,
    LANGUAGE_IDS,
    MIN_NEW_TOKENS,
    NUM_CODE_GROUPS,
    PREDICTOR_LAYERS,
    ROLE_PREFIX_TOKENS,
    SAMPLE_RATE,
    SPEAKER_DIALECT,
    SPEAKER_IDS,
    SUPPRESS_FROM,
    TAIL_TOKENS,
    TALKER_HEAD_DIM,
    TALKER_KV_HEADS,
    TALKER_LAYERS,
    TALKER_VOCAB,
    UPSAMPLE,
    Qwen3TTSAdapter,
    apply_logits_processors,
    chunk_text,
    empty_cache,
    resolve_language_id,
    resolve_language_name,
    roll_cache,
    sample_token,
)

HIDDEN = 1024


# ----------------------------------------------------------------------
# registration / config
# ----------------------------------------------------------------------

def test_qwen3tts_registered():
    from phoonnx.engines import list_engines
    assert "qwen3tts" in list_engines()


def test_qwen3tts_detect():
    assert Qwen3TTSAdapter.detect({"engine": "qwen3tts"})
    assert not Qwen3TTSAdapter.detect({"engine": "sparktts"})
    assert not Qwen3TTSAdapter.detect(None)


def test_engine_enum_and_config_alphabet():
    from phoonnx.config import Alphabet, Engine, VoiceConfig
    cfg = VoiceConfig.from_dict({"engine": "qwen3tts"}, engine=Engine.QWEN3TTS,
                                lang_code="en-US")
    # graphemes routes text->ids through the adapter, never through a phonemizer
    assert cfg.engine == Engine.QWEN3TTS
    assert cfg.alphabet == Alphabet.GRAPHEMES
    assert cfg.sample_rate == SAMPLE_RATE == 24000


def test_default_params_match_upstream_generation_config():
    # Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice generation_config.json
    assert Qwen3TTSAdapter().default_params() == {
        "temperature": 0.9, "top_k": 50.0, "top_p": 1.0, "repetition_penalty": 1.05,
        "subtalker_temperature": 0.9, "subtalker_top_k": 50.0, "subtalker_top_p": 1.0}


def test_param_labels_cover_every_default():
    adapter = Qwen3TTSAdapter()
    assert set(adapter.param_labels()) == set(adapter.default_params())


def test_frame_geometry_matches_the_codec():
    # 12.5 Hz codec at 24 kHz: one frame is 1920 samples
    assert UPSAMPLE == SAMPLE_RATE // 12.5 == 1920


# ----------------------------------------------------------------------
# language and speaker resolution
# ----------------------------------------------------------------------

@pytest.mark.parametrize("tag,expected", [
    ("en-US", "english"), ("en", "english"), ("zh-CN", "chinese"),
    ("pt-BR", "portuguese"), ("ja-JP", "japanese"), ("ko-KR", "korean"),
    ("de-DE", "german"), ("fr-FR", "french"), ("ru-RU", "russian"),
    ("es-ES", "spanish"), ("it-IT", "italian"),
    ("English", "english"), ("auto", "auto"), (None, "auto"), ("", "auto"),
    ("xx-YY", "auto"),
])
def test_resolve_language_name(tag, expected):
    assert resolve_language_name(tag) == expected


def test_every_supported_language_has_a_talker_token():
    for name in set(resolve_language_name(t) for t in
                    ("en", "zh", "de", "it", "pt", "es", "ja", "ko", "fr", "ru")):
        assert name in LANGUAGE_IDS


def test_auto_language_drops_the_language_token():
    assert resolve_language_id("auto", "ryan") is None


def test_language_id_for_a_plain_voice():
    assert resolve_language_id("en-US", "ryan") == LANGUAGE_IDS["english"]
    assert resolve_language_id("zh-CN", "vivian") == LANGUAGE_IDS["chinese"]


@pytest.mark.parametrize("speaker,dialect", sorted(SPEAKER_DIALECT.items()))
def test_dialect_voice_overrides_chinese_and_auto(speaker, dialect):
    assert resolve_language_id("zh-CN", speaker) == LANGUAGE_IDS[dialect]
    assert resolve_language_id("auto", speaker) == LANGUAGE_IDS[dialect]


@pytest.mark.parametrize("speaker", sorted(SPEAKER_DIALECT))
def test_dialect_voice_keeps_a_non_chinese_language(speaker):
    # upstream only overrides when the language is Chinese or unset
    assert resolve_language_id("en-US", speaker) == LANGUAGE_IDS["english"]


def test_speaker_table_is_the_checkpoint_roster():
    assert sorted(SPEAKER_IDS) == [
        "aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee",
        "uncle_fu", "vivian"]
    assert all(0 <= i < TALKER_VOCAB for i in SPEAKER_IDS.values())
    assert len(set(SPEAKER_IDS.values())) == len(SPEAKER_IDS)


def test_dialect_speakers_are_real_speakers():
    assert set(SPEAKER_DIALECT) <= set(SPEAKER_IDS)


# ----------------------------------------------------------------------
# logits processors
# ----------------------------------------------------------------------

def _logits(value=0.0):
    return np.full(TALKER_VOCAB, value, np.float64)


def test_suppressed_range_is_masked():
    scores = apply_logits_processors(_logits(1.0), [], 1.0, step=5)
    assert np.all(np.isneginf(scores[SUPPRESS_FROM:][:CODEC_EOS - SUPPRESS_FROM]))
    assert np.all(np.isneginf(scores[CODEC_EOS + 1:]))
    assert np.isfinite(scores[SUPPRESS_FROM - 1])


def test_end_of_speech_survives_suppression_after_the_floor():
    scores = apply_logits_processors(_logits(1.0), [], 1.0, step=MIN_NEW_TOKENS)
    assert scores[CODEC_EOS] == 1.0


def test_end_of_speech_is_blocked_below_the_floor():
    for step in range(MIN_NEW_TOKENS):
        scores = apply_logits_processors(_logits(1.0), [], 1.0, step=step)
        assert np.isneginf(scores[CODEC_EOS])


def test_repetition_penalty_divides_positive_scores():
    logits = _logits(0.0)
    logits[7] = 2.0
    scores = apply_logits_processors(logits, [7], 1.05, step=5)
    assert scores[7] == pytest.approx(2.0 / 1.05)


def test_repetition_penalty_multiplies_negative_scores():
    logits = _logits(0.0)
    logits[7] = -2.0
    scores = apply_logits_processors(logits, [7], 1.05, step=5)
    assert scores[7] == pytest.approx(-2.0 * 1.05)


def test_repetition_penalty_applies_once_per_repeated_token():
    logits = _logits(0.0)
    logits[7] = 2.0
    once = apply_logits_processors(logits, [7], 1.05, step=5)
    twice = apply_logits_processors(logits, [7, 7, 7], 1.05, step=5)
    assert once[7] == pytest.approx(twice[7])


def test_repetition_penalty_leaves_unseen_tokens_alone():
    logits = _logits(0.0)
    logits[7] = 2.0
    scores = apply_logits_processors(logits, [9], 1.05, step=5)
    assert scores[7] == 2.0


def test_suppression_runs_after_the_penalty():
    # a penalised suppressed token must still be masked, not merely scaled
    logits = _logits(0.0)
    logits[SUPPRESS_FROM + 3] = 9.0
    scores = apply_logits_processors(logits, [SUPPRESS_FROM + 3], 1.05, step=5)
    assert np.isneginf(scores[SUPPRESS_FROM + 3])


def test_processors_do_not_mutate_the_caller_array():
    logits = _logits(1.0)
    apply_logits_processors(logits, [7], 1.05, step=5)
    assert np.all(logits == 1.0)


# ----------------------------------------------------------------------
# sampler
# ----------------------------------------------------------------------

def test_sample_token_greedy_when_sampling_is_off():
    assert sample_token(np.array([1.0, 9.0, 2.0]), 0.9, 50, 1.0,
                        np.random.default_rng(0), do_sample=False) == 1


def test_sample_token_greedy_at_zero_temperature():
    assert sample_token(np.array([1.0, 9.0, 2.0]), 0.0, 50, 1.0,
                        np.random.default_rng(0)) == 1


def test_sample_token_top_k_one_is_argmax():
    logits = np.array([0.1, 5.0, 0.2, 0.3])
    for seed in range(5):
        assert sample_token(logits, 1.0, 1, 1.0, np.random.default_rng(seed)) == 1


def test_sample_token_top_p_keeps_only_the_nucleus():
    logits = np.array([10.0, 0.0, -10.0, -20.0])
    picks = {sample_token(logits, 1.0, 50, 0.5, np.random.default_rng(s))
             for s in range(20)}
    assert picks == {0}


def test_sample_token_is_reproducible_for_a_seed():
    logits = np.random.default_rng(3).normal(size=64)
    a = [sample_token(logits, 0.9, 50, 1.0, np.random.default_rng(11)) for _ in range(3)]
    assert len(set(a)) == 1


def test_sample_token_never_picks_a_masked_token():
    logits = np.array([1.0, 2.0, 3.0])
    logits[2] = -np.inf
    picks = {sample_token(logits, 0.9, 50, 1.0, np.random.default_rng(s))
             for s in range(20)}
    assert 2 not in picks


# ----------------------------------------------------------------------
# text chunking
# ----------------------------------------------------------------------

def test_chunk_text_keeps_short_text_whole():
    assert chunk_text("Hello there.") == ["Hello there."]


def test_chunk_text_respects_the_character_budget():
    text = " ".join(["This is a sentence."] * 60)
    chunks = chunk_text(text, max_len=100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks[:-1])


def test_chunk_text_splits_paragraphs():
    assert len(chunk_text("One.\n\nTwo.")) == 2


def test_chunk_text_on_empty_input():
    assert chunk_text("   ") == []


# ----------------------------------------------------------------------
# KV-cache helpers
# ----------------------------------------------------------------------

def test_empty_cache_shape_and_names():
    cache = empty_cache(TALKER_LAYERS, TALKER_KV_HEADS, TALKER_HEAD_DIM)
    assert len(cache) == 2 * TALKER_LAYERS
    assert cache["past_key_values.0.key"].shape == (1, TALKER_KV_HEADS, 0, TALKER_HEAD_DIM)
    assert cache["past_key_values.27.value"].dtype == np.float32


def test_roll_cache_maps_present_onto_past():
    outputs = {f"present.{i}.{k}": np.full((1, 2, 3, 4), i, np.float32)
               for i in range(PREDICTOR_LAYERS) for k in ("key", "value")}
    rolled = roll_cache(outputs, PREDICTOR_LAYERS)
    assert set(rolled) == {f"past_key_values.{i}.{k}"
                           for i in range(PREDICTOR_LAYERS) for k in ("key", "value")}
    assert rolled["past_key_values.3.key"][0, 0, 0, 0] == 3


# ----------------------------------------------------------------------
# fake sessions: the full two-loop pipeline without any real weights
# ----------------------------------------------------------------------

class FakeSession:
    """Records every feed and returns shaped outputs, like a real ORT session."""

    def __init__(self, output_names, handler):
        self._names = list(output_names)
        self._handler = handler
        self.calls = []

    def get_outputs(self):
        return [type("O", (), {"name": n})() for n in self._names]

    def get_inputs(self):
        return []

    def run(self, _outputs, feed):
        self.calls.append({k: (v.copy() if isinstance(v, np.ndarray) else v)
                           for k, v in feed.items()})
        return self._handler(feed, len(self.calls) - 1)


def _embed_session():
    """An embedding graph: one row of ones per id, scaled by the id."""
    def handler(feed, _call):
        ids = np.asarray(feed["input_ids"]).reshape(1, -1)
        return [np.ones((1, ids.shape[1], HIDDEN), np.float32) * ids.reshape(1, -1, 1)]
    return FakeSession(["hidden"], handler)


def _sub_embed_session():
    def handler(feed, _call):
        ids = np.asarray(feed["input_ids"]).reshape(-1)
        tables = np.asarray(feed["tables"]).reshape(-1)
        assert ids.shape == tables.shape, "token and table streams must line up"
        return [(np.ones((1, ids.size, HIDDEN), np.float32)
                 * (ids + 1000 * tables).reshape(1, -1, 1))]
    return FakeSession(["hidden"], handler)


def _talker_session(tokens, layers=TALKER_LAYERS, heads=TALKER_KV_HEADS,
                    head_dim=TALKER_HEAD_DIM):
    """Emit ``tokens`` in order and grow a KV cache exactly like a real step."""
    names = ["logits", "last_hidden"] + [f"present.{i}.{k}" for i in range(layers)
                                         for k in ("key", "value")]

    def handler(feed, call):
        width = feed["inputs_embeds"].shape[1]
        past = feed["past_key_values.0.key"].shape[2]
        logits = np.full((1, 1, TALKER_VOCAB), -5.0, np.float32)
        logits[0, 0, tokens[min(call, len(tokens) - 1)]] = 50.0
        present = np.zeros((1, heads, past + width, head_dim), np.float32)
        return ([logits, np.full((1, 1, HIDDEN), 0.25, np.float32)]
                + [present.copy() for _ in range(2 * layers)])
    return FakeSession(names, handler)


def _predictor_sessions(group_token=7):
    prefill_names = ["logits"] + [f"present.{i}.{k}" for i in range(PREDICTOR_LAYERS)
                                  for k in ("key", "value")]
    step_names = list(prefill_names)

    def prefill(feed, _call):
        assert feed["inputs_embeds"].shape[1] == 2, "prefill reads exactly two positions"
        logits = np.full((1, 2048), -5.0, np.float32)
        logits[0, group_token] = 9.0
        present = np.zeros((1, PREDICTOR_KV, 2, PREDICTOR_HD), np.float32)
        return [logits] + [present.copy() for _ in range(2 * PREDICTOR_LAYERS)]

    def step(feed, _call):
        past = feed["past_key_values.0.key"].shape[2]
        logits = np.full((1, 2048), -5.0, np.float32)
        logits[0, group_token] = 9.0
        present = np.zeros((1, PREDICTOR_KV, past + 1, PREDICTOR_HD), np.float32)
        return [logits] + [present.copy() for _ in range(2 * PREDICTOR_LAYERS)]

    return FakeSession(prefill_names, prefill), FakeSession(step_names, step)


PREDICTOR_KV, PREDICTOR_HD = 8, 128


def _decoder_session():
    def handler(feed, _call):
        frames = feed["codes"].shape[-1]
        return [np.full((1, 1, frames * UPSAMPLE), 0.5, np.float32)]
    return FakeSession(["wav"], handler)


def _adapter(tokens=(11, 12, CODEC_EOS)):
    adapter = Qwen3TTSAdapter()
    adapter.text_embed = _embed_session()
    adapter.codec_embed = _embed_session()
    adapter.predictor_prefill, adapter.predictor_step = _predictor_sessions()
    adapter.sub_codec_embed = _sub_embed_session()
    adapter.decoder = _decoder_session()
    adapter.speaker = "ryan"
    adapter.language = "english"
    return adapter, _talker_session(list(tokens))


def _ids(n_text=6):
    return np.arange(1, ROLE_PREFIX_TOKENS + n_text + TAIL_TOKENS + 1,
                     dtype=np.int64)[None]


# -- prompt ------------------------------------------------------------

def test_prompt_length_follows_the_text():
    adapter, _ = _adapter()
    for n_text in (1, 6, 30):
        prompt, _ = adapter.build_prompt(_ids(n_text), "ryan", "english")
        # role(3) + think block(4) + speaker + pad + text + eos + codec bos
        assert prompt.shape == (1, ROLE_PREFIX_TOKENS + 6 + n_text + 1 + 1, HIDDEN)


def test_prompt_without_a_language_is_one_position_shorter():
    adapter, _ = _adapter()
    tagged, _ = adapter.build_prompt(_ids(), "ryan", "english")
    auto, _ = adapter.build_prompt(_ids(), "ryan", "auto")
    assert auto.shape[1] == tagged.shape[1] - 1


def test_prompt_carries_the_language_token():
    adapter, _ = _adapter()
    adapter.build_prompt(_ids(), "ryan", "german")
    fed = [c["input_ids"].reshape(-1).tolist() for c in adapter.codec_embed.calls]
    assert [CODEC_THINK, CODEC_THINK_BOS, LANGUAGE_IDS["german"], CODEC_THINK_EOS] in fed


def test_prompt_uses_the_no_language_block_for_auto():
    adapter, _ = _adapter()
    adapter.build_prompt(_ids(), "ryan", "auto")
    fed = [c["input_ids"].reshape(-1).tolist() for c in adapter.codec_embed.calls]
    assert [CODEC_NOTHINK, CODEC_THINK_BOS, CODEC_THINK_EOS] in fed


def test_prompt_carries_the_speaker_token():
    adapter, _ = _adapter()
    adapter.build_prompt(_ids(), "vivian", "chinese")
    fed = [c["input_ids"].reshape(-1).tolist() for c in adapter.codec_embed.calls]
    assert [SPEAKER_IDS["vivian"]] in fed


def test_prompt_pads_the_codec_side_across_the_text():
    adapter, _ = _adapter()
    adapter.build_prompt(_ids(n_text=6), "ryan", "english")
    fed = [c["input_ids"].reshape(-1).tolist() for c in adapter.codec_embed.calls]
    assert [CODEC_PAD] * 7 in fed          # six text tokens plus the text EOS


def test_prompt_ends_on_the_codec_bos():
    adapter, _ = _adapter()
    adapter.build_prompt(_ids(), "ryan", "english")
    assert adapter.codec_embed.calls[-1]["input_ids"].reshape(-1).tolist() == [CODEC_BOS]


def test_prompt_rejects_an_unknown_speaker():
    adapter, _ = _adapter()
    with pytest.raises(ValueError, match="no speaker"):
        adapter.build_prompt(_ids(), "nobody", "english")


def test_prompt_rejects_text_free_input():
    adapter, _ = _adapter()
    short = np.arange(ROLE_PREFIX_TOKENS + TAIL_TOKENS, dtype=np.int64)[None]
    with pytest.raises(ValueError, match="no text"):
        adapter.build_prompt(short, "ryan", "english")


# -- code predictor ----------------------------------------------------

def test_code_predictor_emits_every_remaining_group():
    adapter, _ = _adapter()
    groups = adapter.predict_code_groups(
        np.zeros((1, 1, HIDDEN), np.float32), np.zeros((1, 1, HIDDEN), np.float32),
        np.random.default_rng(0), 0.9, 50, 1.0, do_sample=False)
    assert len(groups) == NUM_CODE_GROUPS - 1 == 15


def test_code_predictor_advances_step_and_position_together():
    adapter, _ = _adapter()
    adapter.predict_code_groups(
        np.zeros((1, 1, HIDDEN), np.float32), np.zeros((1, 1, HIDDEN), np.float32),
        np.random.default_rng(0), 0.9, 50, 1.0, do_sample=False)
    steps = [int(c["step"]) for c in adapter.predictor_step.calls]
    positions = [int(c["position_ids"][0, 0]) for c in adapter.predictor_step.calls]
    assert steps == list(range(1, NUM_CODE_GROUPS - 1))
    # the prefill wrote positions 0 and 1, so step n sits at n + 1
    assert positions == [s + 1 for s in steps]


def test_code_predictor_cache_grows_one_position_per_step():
    adapter, _ = _adapter()
    adapter.predict_code_groups(
        np.zeros((1, 1, HIDDEN), np.float32), np.zeros((1, 1, HIDDEN), np.float32),
        np.random.default_rng(0), 0.9, 50, 1.0, do_sample=False)
    lengths = [c["past_key_values.0.key"].shape[2] for c in adapter.predictor_step.calls]
    assert lengths == list(range(2, 2 + NUM_CODE_GROUPS - 2))


def test_code_predictor_reads_the_previous_group_through_its_own_table():
    adapter, _ = _adapter()
    adapter.predict_code_groups(
        np.zeros((1, 1, HIDDEN), np.float32), np.zeros((1, 1, HIDDEN), np.float32),
        np.random.default_rng(0), 0.9, 50, 1.0, do_sample=False)
    tables = [int(c["tables"][0]) for c in adapter.sub_codec_embed.calls
              if c["tables"].size == 1]
    assert tables == list(range(NUM_CODE_GROUPS - 2))


# -- talker loop -------------------------------------------------------

def test_generate_codes_stops_on_end_of_speech():
    adapter, talker = _adapter(tokens=(11, 12, 13, CODEC_EOS))
    codes = adapter.generate_codes(
        talker, np.zeros((1, 9, HIDDEN), np.float32), np.zeros((1, 1, HIDDEN), np.float32),
        {"do_sample": False}, np.random.default_rng(0))
    assert codes.shape == (3, NUM_CODE_GROUPS)
    assert codes[:, 0].tolist() == [11, 12, 13]


def test_generate_codes_honours_the_length_cap():
    adapter, talker = _adapter(tokens=(11,))
    codes = adapter.generate_codes(
        talker, np.zeros((1, 9, HIDDEN), np.float32), np.zeros((1, 1, HIDDEN), np.float32),
        {"do_sample": False, "max_new_tokens": 4}, np.random.default_rng(0))
    assert codes.shape == (4, NUM_CODE_GROUPS)


def test_generate_codes_feeds_the_prompt_once_then_single_frames():
    adapter, talker = _adapter(tokens=(11, 12, CODEC_EOS))
    adapter.generate_codes(
        talker, np.zeros((1, 9, HIDDEN), np.float32), np.zeros((1, 1, HIDDEN), np.float32),
        {"do_sample": False}, np.random.default_rng(0))
    widths = [c["inputs_embeds"].shape[1] for c in talker.calls]
    assert widths == [9, 1, 1]


def test_generate_codes_advances_positions_past_the_prompt():
    adapter, talker = _adapter(tokens=(11, 12, CODEC_EOS))
    adapter.generate_codes(
        talker, np.zeros((1, 9, HIDDEN), np.float32), np.zeros((1, 1, HIDDEN), np.float32),
        {"do_sample": False}, np.random.default_rng(0))
    positions = [c["position_ids"].reshape(-1).tolist() for c in talker.calls]
    assert positions[0] == list(range(9))
    assert positions[1] == [9]
    assert positions[2] == [10]


def test_generate_codes_grows_the_talker_cache_by_the_fed_width():
    adapter, talker = _adapter(tokens=(11, 12, 13, CODEC_EOS))
    adapter.generate_codes(
        talker, np.zeros((1, 9, HIDDEN), np.float32), np.zeros((1, 1, HIDDEN), np.float32),
        {"do_sample": False}, np.random.default_rng(0))
    lengths = [c["past_key_values.0.key"].shape[2] for c in talker.calls]
    assert lengths == [0, 9, 10, 11]


def test_generate_codes_passes_every_layer_of_cache():
    adapter, talker = _adapter(tokens=(11, CODEC_EOS))
    adapter.generate_codes(
        talker, np.zeros((1, 9, HIDDEN), np.float32), np.zeros((1, 1, HIDDEN), np.float32),
        {"do_sample": False}, np.random.default_rng(0))
    keys = {k for k in talker.calls[-1] if k.startswith("past_key_values.")}
    assert len(keys) == 2 * TALKER_LAYERS


def test_generate_codes_sums_all_sixteen_groups_into_the_next_step():
    adapter, talker = _adapter(tokens=(11, CODEC_EOS))
    adapter.generate_codes(
        talker, np.zeros((1, 9, HIDDEN), np.float32),
        np.full((1, 1, HIDDEN), 3.0, np.float32),
        {"do_sample": False}, np.random.default_rng(0))
    # group 0 embeds to 11; each of the 15 other groups embeds to token + 1000*table
    expected = 11 + sum(7 + 1000 * t for t in range(NUM_CODE_GROUPS - 1)) + 3.0
    assert talker.calls[1]["inputs_embeds"][0, 0, 0] == pytest.approx(expected)


def test_generate_codes_cannot_end_before_the_floor():
    # the fake talker asks to end at every step; the floor holds it off
    adapter, talker = _adapter(tokens=(CODEC_EOS,))
    codes = adapter.generate_codes(
        talker, np.zeros((1, 9, HIDDEN), np.float32), np.zeros((1, 1, HIDDEN), np.float32),
        {"do_sample": False}, np.random.default_rng(0))
    assert codes.shape == (MIN_NEW_TOKENS, NUM_CODE_GROUPS)


def test_generate_codes_returns_an_empty_matrix_when_the_cap_is_zero():
    adapter, talker = _adapter(tokens=(11,))
    codes = adapter.generate_codes(
        talker, np.zeros((1, 9, HIDDEN), np.float32), np.zeros((1, 1, HIDDEN), np.float32),
        {"do_sample": False, "max_new_tokens": 0}, np.random.default_rng(0))
    assert codes.shape == (0, NUM_CODE_GROUPS)


def test_generate_codes_never_emits_a_suppressed_token():
    # the fake talker points every step at a suppressed control token
    adapter, talker = _adapter(tokens=(SUPPRESS_FROM + 4,))
    codes = adapter.generate_codes(
        talker, np.zeros((1, 9, HIDDEN), np.float32), np.zeros((1, 1, HIDDEN), np.float32),
        {"do_sample": False, "max_new_tokens": 3}, np.random.default_rng(0))
    assert np.all(codes[:, 0] < SUPPRESS_FROM)


# -- codec decoder -----------------------------------------------------

def test_decode_codes_transposes_to_the_graph_layout():
    adapter, _ = _adapter()
    adapter.decode_codes(np.zeros((7, NUM_CODE_GROUPS), np.int64))
    assert adapter.decoder.calls[0]["codes"].shape == (1, NUM_CODE_GROUPS, 7)


def test_decode_codes_length_matches_the_frame_count():
    adapter, _ = _adapter()
    wav = adapter.decode_codes(np.zeros((7, NUM_CODE_GROUPS), np.int64))
    assert wav.shape == (7 * UPSAMPLE,)
    assert wav.dtype == np.float32


def test_decode_codes_chunks_long_input_with_left_context():
    adapter, _ = _adapter()
    frames = DECODE_CHUNK + 40
    wav = adapter.decode_codes(np.zeros((frames, NUM_CODE_GROUPS), np.int64))
    widths = [c["codes"].shape[-1] for c in adapter.decoder.calls]
    assert widths == [DECODE_CHUNK, 40 + DECODE_LEFT_CONTEXT]
    # the left context is decoded again and then dropped, so length is unchanged
    assert wav.shape == (frames * UPSAMPLE,)


def test_decode_codes_first_chunk_has_no_left_context():
    adapter, _ = _adapter()
    adapter.decode_codes(np.zeros((10, NUM_CODE_GROUPS), np.int64))
    assert adapter.decoder.calls[0]["codes"].shape[-1] == 10


# -- synthesize --------------------------------------------------------

def _request(**params):
    return AdapterSynthesisRequest(
        phoneme_ids=_ids(), phoneme_lengths=np.array([_ids().size], np.int64),
        params=params)


def test_synthesize_returns_audio_and_extras():
    adapter, talker = _adapter(tokens=(11, 12, CODEC_EOS))
    result = adapter.synthesize(_request(do_sample=False), talker)
    assert result.audio.shape == (2 * UPSAMPLE,)
    assert result.extras["frame_count"] == 2
    assert result.extras["speaker"] == "ryan"
    assert result.extras["language"] == "english"


def test_synthesize_takes_the_speaker_from_the_request():
    adapter, talker = _adapter(tokens=(11, CODEC_EOS))
    result = adapter.synthesize(_request(do_sample=False, speaker="vivian"), talker)
    assert result.extras["speaker"] == "vivian"
    fed = [c["input_ids"].reshape(-1).tolist() for c in adapter.codec_embed.calls]
    assert [SPEAKER_IDS["vivian"]] in fed


def test_synthesize_returns_silence_when_no_frame_is_produced():
    adapter, talker = _adapter(tokens=(11,))
    result = adapter.synthesize(_request(do_sample=False, max_new_tokens=0), talker)
    assert result.audio.shape == (0,)


def test_synthesize_is_reproducible_for_a_seed():
    outputs = []
    for _ in range(2):
        adapter, talker = _adapter(tokens=(11, 12, CODEC_EOS))
        outputs.append(adapter.synthesize(_request(seed=7), talker).audio)
    assert np.array_equal(outputs[0], outputs[1])


@pytest.mark.parametrize("missing,key", [
    ("text_embed", "text_embed_path"),
    ("codec_embed", "codec_embed_path"),
    ("predictor_prefill", "code_predictor_prefill_path"),
    ("predictor_step", "code_predictor_step_path"),
    ("sub_codec_embed", "sub_codec_embed_path"),
    ("decoder", "codec_decoder_path"),
])
def test_synthesize_names_the_missing_graph(missing, key):
    adapter, talker = _adapter()
    setattr(adapter, missing, None)
    with pytest.raises(RuntimeError, match=key):
        adapter.synthesize(_request(), talker)


def test_feed_dict_path_is_refused():
    adapter, talker = _adapter()
    with pytest.raises(NotImplementedError):
        adapter.build_feed_dict(_request(), talker)
    with pytest.raises(NotImplementedError):
        adapter.parse_outputs([], _request())


# -- configure ---------------------------------------------------------

class _Config:
    def __init__(self, engine_params, lang_code=None):
        self.engine_params = engine_params
        self.lang_code = lang_code


def test_configure_rejects_an_unknown_speaker():
    with pytest.raises(ValueError, match="no speaker"):
        Qwen3TTSAdapter().configure(_Config({"speaker": "nobody"}))


def test_configure_takes_the_language_from_the_voice():
    adapter = Qwen3TTSAdapter()
    adapter.configure(_Config({"speaker": "vivian"}, lang_code="zh-CN"))
    assert adapter.language == "chinese"
    assert adapter.speaker == "vivian"


def test_configure_prefers_an_explicit_language():
    adapter = Qwen3TTSAdapter()
    adapter.configure(_Config({"speaker": "ryan", "language": "English"},
                              lang_code="zh-CN"))
    assert adapter.language == "english"


def test_configure_without_a_language_falls_back_to_auto():
    adapter = Qwen3TTSAdapter()
    adapter.configure(_Config({"speaker": "ryan"}))
    assert adapter.language == "auto"


def test_encode_text_needs_a_tokenizer():
    with pytest.raises(RuntimeError, match="bpe_tokenizer_path"):
        Qwen3TTSAdapter().encode_text("hello", None, None)
