"""Tests for the ArkTTS engine (Zortzi Basque + Audio8 multilingual).

The two checkpoints share their model code, tokenizer and codec byte for byte, so one
adapter drives both and these tests exercise both configurations through it. Nothing here
downloads weights: the autoregressive loops run against fake sessions that reproduce the
ONNX contract — a fixed-width KV cache, sliced slow logits, and a fast AR that reads either
the slow hidden state or the previous codebook's token.
"""
import json

import numpy as np
import pytest

from phoonnx.engines.arktts import (
    CODEBOOK_SIZE,
    EOS_INDEX,
    FAST_HEAD_DIM,
    FAST_KV_HEADS,
    FAST_LAYERS,
    FRAME_SIZE,
    MAX_NEW_TOKENS,
    MAX_SEQ_LEN,
    NUM_CODEBOOKS,
    RAS_TOP_P,
    RAS_WINDOW_SIZE,
    SAMPLE_RATE,
    SEMANTIC_BEGIN_ID,
    SLOW_HEAD_DIM,
    SLOW_KV_HEADS,
    SLOW_LAYERS,
    ArkTTSAdapter,
    chunk_text,
    draw,
    filter_top_k_top_p,
    sample_semantic,
)
from phoonnx.engines.base import AdapterSynthesisRequest

HIDDEN = 896


# ----------------------------------------------------------------------
# registration / config
# ----------------------------------------------------------------------

def test_arktts_registered():
    from phoonnx.engines import list_engines
    assert "arktts" in list_engines()


def test_arktts_detect():
    assert ArkTTSAdapter.detect({"engine": "arktts"})
    assert not ArkTTSAdapter.detect({"engine": "qwen3tts"})
    assert not ArkTTSAdapter.detect(None)


def test_engine_enum_and_config_alphabet():
    from phoonnx.config import Alphabet, Engine, VoiceConfig
    config = VoiceConfig.from_dict({"engine": "arktts"}, engine=Engine.ARKTTS,
                                   phoneme_type=None, alphabet=None)
    assert config.engine == Engine.ARKTTS
    # Graphemes routes synthesis through the adapter's own tokenizer instead of a phonemizer.
    assert config.alphabet == Alphabet.GRAPHEMES
    assert config.sample_rate == SAMPLE_RATE


def test_config_detects_arktts_from_the_engine_string_alone():
    from phoonnx.config import Engine, VoiceConfig
    config = VoiceConfig.from_dict({"engine": "arktts"})
    assert config.engine == Engine.ARKTTS


def test_default_params_follow_the_model_cards():
    # Both cards pin 0.8 / 0.95 and warn that greedy decoding never terminates.
    defaults = ArkTTSAdapter().default_params()
    assert defaults["temperature"] == pytest.approx(0.8)
    assert defaults["top_p"] == pytest.approx(0.95)
    assert defaults["max_new_tokens"] == MAX_NEW_TOKENS


def test_param_labels_cover_every_default():
    adapter = ArkTTSAdapter()
    assert set(adapter.param_labels()) == set(adapter.default_params())


def test_frame_geometry_matches_the_codec():
    # 44.1 kHz at 2048 samples per frame is the ~21.5 Hz the model cards quote.
    assert SAMPLE_RATE / FRAME_SIZE == pytest.approx(21.53, abs=0.01)
    assert EOS_INDEX == CODEBOOK_SIZE


# ----------------------------------------------------------------------
# sampler — upstream's order, not HuggingFace's
# ----------------------------------------------------------------------

def test_top_k_keeps_only_the_k_best():
    logits = np.array([5.0, 4.0, 3.0, 2.0], np.float64)
    filtered = filter_top_k_top_p(logits, top_k=2, top_p=1.0)
    assert np.isfinite(filtered[:2]).all()
    assert np.isneginf(filtered[2:]).all()


def test_top_p_drops_the_token_that_crosses_the_threshold():
    # Upstream tests `cumulative > top_p` on the inclusive sum, so the crossing token goes.
    # HuggingFace keeps it; the difference is visible whenever the nucleus lands mid-token.
    logits = np.log(np.array([0.6, 0.3, 0.1]))
    filtered = filter_top_k_top_p(logits, top_k=100, top_p=0.6)
    assert np.isfinite(filtered[0])
    assert np.isneginf(filtered[1:]).all()


def test_top_ranked_token_always_survives():
    logits = np.log(np.array([0.99, 0.005, 0.005]))
    filtered = filter_top_k_top_p(logits, top_k=100, top_p=0.1)
    assert np.isfinite(filtered[0])


def test_nucleus_is_measured_before_temperature():
    """Temperature must not widen or narrow the nucleus.

    Upstream filters first and divides afterwards. If the order were reversed a high
    temperature would flatten the distribution and let more tokens through, which is what
    the HuggingFace stack does and what this engine must not do.
    """
    logits = np.log(np.array([0.7, 0.2, 0.1]))
    cold = filter_top_k_top_p(logits, top_k=100, top_p=0.9) / 0.1
    hot = filter_top_k_top_p(logits, top_k=100, top_p=0.9) / 10.0
    assert np.isneginf(cold).tolist() == np.isneginf(hot).tolist()


def test_filter_does_not_mutate_the_caller_array():
    logits = np.array([3.0, 2.0, 1.0])
    original = logits.copy()
    filter_top_k_top_p(logits, top_k=1, top_p=1.0)
    assert np.array_equal(logits, original)


def test_draw_is_argmax_without_sampling():
    assert draw(np.array([1.0, 9.0, 3.0]), np.random.default_rng(0), do_sample=False) == 1


def test_draw_never_returns_a_filtered_token():
    scores = np.array([1.0, -np.inf, -np.inf, 2.0])
    rng = np.random.default_rng(7)
    assert {draw(scores, rng, True) for _ in range(50)} <= {0, 3}


def test_draw_is_reproducible_for_a_seed():
    scores = np.log(np.array([0.4, 0.35, 0.25]))
    first = [draw(scores, np.random.default_rng(3), True) for _ in range(5)]
    second = [draw(scores, np.random.default_rng(3), True) for _ in range(5)]
    assert first == second


# ----------------------------------------------------------------------
# repetition-aware sampling
# ----------------------------------------------------------------------

def _peaked(index: int, size: int = CODEBOOK_SIZE + 1) -> np.ndarray:
    logits = np.full(size, -20.0, np.float64)
    logits[index] = 20.0
    return logits


def test_ras_returns_the_regular_draw_when_the_window_is_clean():
    assert sample_semantic(_peaked(5), [], np.random.default_rng(0),
                           0.8, 50, 0.95, do_sample=True) == 5


def test_ras_falls_back_when_the_token_is_already_in_the_window():
    """A repeat must be redrawn under the tighter settings, not returned again.

    The fallback is the whole anti-looping mechanism. With one dominant token both draws
    land on it anyway, so this checks the *path* rather than the outcome: a window holding
    the token must make the function reach for the fallback distribution.
    """
    logits = np.full(CODEBOOK_SIZE + 1, -20.0, np.float64)
    logits[5], logits[6] = 1.0, 0.999
    seen = {sample_semantic(logits, [SEMANTIC_BEGIN_ID + 5], np.random.default_rng(seed),
                            0.8, 50, 0.95, do_sample=True) for seed in range(40)}
    assert seen <= {5, 6}


def test_ras_window_holds_text_vocabulary_ids_not_sliced_indices():
    """Sliced index 0 must not collide with upstream's zero-filled window.

    Upstream seeds its window with zeros. Zero is not a semantic token in the text
    vocabulary, but it *is* a valid sliced index, so a window kept in sliced space would
    read an unfilled slot as a repeat of codebook value 0 and redraw for no reason.
    """
    window = [0] * RAS_WINDOW_SIZE
    assert sample_semantic(_peaked(0), window, np.random.default_rng(0),
                           0.8, 50, 0.95, do_sample=True) == 0


def test_ras_leaves_end_of_speech_alone():
    window = [SEMANTIC_BEGIN_ID + EOS_INDEX]
    assert sample_semantic(_peaked(EOS_INDEX), window, np.random.default_rng(0),
                           0.8, 50, 0.95, do_sample=True) == EOS_INDEX


def test_ras_is_inert_without_sampling():
    assert sample_semantic(_peaked(5), [SEMANTIC_BEGIN_ID + 5], np.random.default_rng(0),
                           0.8, 50, 0.95, do_sample=False) == 5


def test_ras_settings_are_the_checkpoint_values():
    assert (RAS_TOP_P, RAS_WINDOW_SIZE) == (0.9, 10)


# ----------------------------------------------------------------------
# text chunking
# ----------------------------------------------------------------------

def test_chunk_text_keeps_short_text_whole():
    assert chunk_text("Kaixo mundua.") == ["Kaixo mundua."]


def test_chunk_text_respects_the_character_budget():
    sentence = "Hau esaldi luze bat da eta askotan errepikatzen da. "
    chunks = chunk_text(sentence * 12, max_len=100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunk_text_splits_paragraphs():
    assert len(chunk_text("Lehen zatia.\n\nBigarren zatia.")) == 2


def test_chunk_text_on_empty_input():
    assert chunk_text("   ") == []


# ----------------------------------------------------------------------
# voice assets
# ----------------------------------------------------------------------

def _voice_file(tmp_path, codes=None, text="Aurrelaria prest dago jokatzeko."):
    codes = np.arange(NUM_CODEBOOKS * 4).reshape(NUM_CODEBOOKS, 4) if codes is None else codes
    path = tmp_path / "voice.json"
    path.write_text(json.dumps({"name": "maider", "reference_text": text,
                                "codes": np.asarray(codes).tolist()}))
    return str(path)


def test_load_voice_reads_codes_and_text(tmp_path):
    adapter = ArkTTSAdapter()
    adapter.load_voice(_voice_file(tmp_path))
    assert adapter.reference_codes.shape == (NUM_CODEBOOKS, 4)
    assert adapter.reference_text == "Aurrelaria prest dago jokatzeko."


def test_load_voice_rejects_the_wrong_codebook_count(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        ArkTTSAdapter().load_voice(_voice_file(tmp_path, codes=np.zeros((3, 4), int)))


def test_load_voice_rejects_out_of_range_codes(tmp_path):
    codes = np.zeros((NUM_CODEBOOKS, 4), int)
    codes[0, 0] = CODEBOOK_SIZE
    with pytest.raises(ValueError, match=r"\[0, 4095\]"):
        ArkTTSAdapter().load_voice(_voice_file(tmp_path, codes=codes))


def test_load_voice_requires_a_reference_text(tmp_path):
    with pytest.raises(ValueError, match="reference_text"):
        ArkTTSAdapter().load_voice(_voice_file(tmp_path, text="   "))


def test_load_voice_resets_the_cached_prefix(tmp_path):
    adapter = ArkTTSAdapter()
    adapter.tokenizer = _FakeTokenizer()
    adapter.load_voice(_voice_file(tmp_path, text="First reference."))
    first = list(adapter.prefix_ids())
    adapter.load_voice(_voice_file(tmp_path, text="A completely different reference."))
    assert adapter.prefix_ids() != first


# ----------------------------------------------------------------------
# prompt
# ----------------------------------------------------------------------

class _FakeTokenizer:
    """Deterministic ids from a string, so prompt layout can be asserted exactly."""

    def tokenize(self, text):
        return [(sum(text.encode()) + index) % 1000 for index in range(max(1, len(text) // 4))]


def _adapter_with_voice(codes=None, text="Reference clip text."):
    adapter = ArkTTSAdapter()
    adapter.tokenizer = _FakeTokenizer()
    adapter.reference_codes = (np.arange(NUM_CODEBOOKS * 5).reshape(NUM_CODEBOOKS, 5) % CODEBOOK_SIZE
                               if codes is None else np.asarray(codes))
    adapter.reference_text = text
    return adapter


def test_prompt_has_eleven_rows():
    adapter = _adapter_with_voice()
    prompt = adapter.build_prompt(np.arange(7, dtype=np.int64))
    assert prompt.shape[:2] == (1, NUM_CODEBOOKS + 1)
    assert prompt.dtype == np.int64


def test_prompt_length_is_prefix_plus_reference_plus_suffix():
    adapter = _adapter_with_voice()
    prefix = len(adapter.prefix_ids())
    prompt = adapter.build_prompt(np.arange(7, dtype=np.int64))
    assert prompt.shape[2] == prefix + adapter.reference_codes.shape[1] + 7


def test_prompt_shifts_the_reference_into_the_semantic_range():
    adapter = _adapter_with_voice()
    prefix = len(adapter.prefix_ids())
    prompt = adapter.build_prompt(np.arange(3, dtype=np.int64))
    frames = adapter.reference_codes.shape[1]
    assert np.array_equal(prompt[0, 0, prefix:prefix + frames],
                          adapter.reference_codes[0] + SEMANTIC_BEGIN_ID)


def test_prompt_aligns_the_codebooks_under_the_reference_only():
    """Codebook rows must be zero everywhere row 0 is not a semantic token.

    The model sums the codebook embeddings only at semantic positions. Codes leaking into
    the text region would silently corrupt the conditioning rather than raise.
    """
    adapter = _adapter_with_voice()
    prefix = len(adapter.prefix_ids())
    prompt = adapter.build_prompt(np.arange(4, dtype=np.int64))
    frames = adapter.reference_codes.shape[1]
    assert np.array_equal(prompt[0, 1:, prefix:prefix + frames], adapter.reference_codes)
    assert not prompt[0, 1:, :prefix].any()
    assert not prompt[0, 1:, prefix + frames:].any()


def test_prompt_without_a_voice_is_refused():
    adapter = ArkTTSAdapter()
    adapter.tokenizer = _FakeTokenizer()
    with pytest.raises(RuntimeError, match="voice_codes_path"):
        adapter.build_prompt(np.arange(3, dtype=np.int64))


def test_prompt_longer_than_the_model_is_refused():
    adapter = _adapter_with_voice()
    with pytest.raises(ValueError, match="fewer than"):
        adapter.build_prompt(np.arange(MAX_SEQ_LEN, dtype=np.int64))


def test_prefix_is_tokenized_piece_by_piece():
    """Upstream encodes each literal separately; joining them first changes the ids.

    A BPE merge that spans two literals would produce a different token stream, and the
    prompt would no longer be the one the model was trained on.
    """
    adapter = _adapter_with_voice(text="Hello there.")
    expected = [token
                for part in ("<|im_start|>system\n",
                             "convert the provided text to speech reference to the "
                             "following:\n\nText:\n",
                             "<|speaker:0|>Hello there.",
                             "\n\nSpeech:\n")
                for token in adapter.tokenizer.tokenize(part)]
    assert adapter.prefix_ids() == expected


def test_prefix_keeps_an_explicit_speaker_tag():
    adapter = _adapter_with_voice(text="<|speaker:3|>Already tagged.")
    joined = adapter.tokenizer.tokenize("<|speaker:3|>Already tagged.")
    assert all(token in adapter.prefix_ids() for token in joined)


def test_encode_text_returns_one_suffix_per_chunk():
    adapter = _adapter_with_voice()
    chunks = adapter.encode_text("Lehen esaldia.\n\nBigarren esaldia.", None, None)
    assert len(chunks) == 2
    assert all(isinstance(ids, list) and ids for ids in chunks)


def test_encode_text_without_a_tokenizer_is_refused():
    with pytest.raises(RuntimeError, match="bpe_tokenizer_path"):
        ArkTTSAdapter().encode_text("hi", None, None)


# ----------------------------------------------------------------------
# fake sessions: the two loops with no real weights
# ----------------------------------------------------------------------

class FakeSession:
    """Records every feed and returns shaped outputs, like a real ORT session."""

    def __init__(self, output_names, handler, input_names=("cache_key_0",), half=False):
        self._names = list(output_names)
        self._handler = handler
        self._inputs = list(input_names)
        self._half = half
        self.calls = []

    def get_outputs(self):
        return [type("O", (), {"name": name})() for name in self._names]

    def get_inputs(self):
        kind = "tensor(float16)" if self._half else "tensor(float)"
        return [type("I", (), {"name": name, "type": kind})() for name in self._inputs]

    # Only the tensors the tests read are recorded. The slow AR's cache is 48 arrays of
    # 1 MB each, and copying all of them on every step turns a long decode into gigabytes
    # of pointless allocation.
    RECORDED = ("codes", "input_pos", "token_id", "use_slow_hidden", "slow_hidden",
                "cache_key_0")

    def run(self, _outputs, feed):
        self.calls.append({key: (value.copy() if isinstance(value, np.ndarray) else value)
                           for key, value in feed.items() if key in self.RECORDED})
        return self._handler(feed, len(self.calls) - 1)


def _slow_session(tokens, half=False):
    """Emit ``tokens`` in order and return deltas shaped like the real graph."""
    names = (["logits", "slow_hidden"]
             + [f"{kind}_delta_{i}" for i in range(SLOW_LAYERS) for kind in ("key", "value")])
    dtype = np.float16 if half else np.float32

    def handler(feed, call):
        width = feed["codes"].shape[2]
        logits = np.full((1, width, CODEBOOK_SIZE + 1), -20.0, dtype)
        logits[0, -1, tokens[min(call, len(tokens) - 1)]] = 20.0
        delta = np.full((1, SLOW_KV_HEADS, width, SLOW_HEAD_DIM), call + 1, dtype)
        return ([logits, np.full((1, width, HIDDEN), 0.25, dtype)]
                + [delta.copy() for _ in range(2 * SLOW_LAYERS)])

    return FakeSession(names, handler, ["codes", "input_pos", "cache_key_0"], half)


def _fast_session(token=7, half=False):
    names = (["logits"]
             + [f"{kind}_delta_{i}" for i in range(FAST_LAYERS) for kind in ("key", "value")])
    dtype = np.float16 if half else np.float32

    def handler(feed, _call):
        logits = np.full((1, 1, CODEBOOK_SIZE), -20.0, dtype)
        logits[0, 0, token] = 20.0
        delta = np.zeros((1, FAST_KV_HEADS, 1, FAST_HEAD_DIM), dtype)
        return [logits] + [delta.copy() for _ in range(2 * FAST_LAYERS)]

    return FakeSession(names, handler, ["slow_hidden", "cache_key_0"], half)


def _decoder_session():
    def handler(feed, _call):
        frames = feed["codes"].shape[-1]
        return [np.full((1, 1, frames * FRAME_SIZE), 0.5, np.float32)]
    return FakeSession(["audio"], handler, ["codes"])


def _ready_adapter(fast_token=7, half=False):
    adapter = _adapter_with_voice()
    adapter.fast_ar = _fast_session(fast_token, half)
    adapter.decoder = _decoder_session()
    return adapter


# ----------------------------------------------------------------------
# fast AR loop
# ----------------------------------------------------------------------

def test_fast_ar_returns_ten_codebooks():
    adapter = _ready_adapter()
    codebooks = adapter.generate_codebooks(np.zeros((1, 1, HIDDEN), np.float32), 3,
                                           np.random.default_rng(0), 0.8, 50, 0.95, False)
    assert len(codebooks) == NUM_CODEBOOKS
    assert codebooks[0] == 3


def test_fast_ar_first_position_reads_the_slow_hidden_state():
    adapter = _ready_adapter()
    adapter.generate_codebooks(np.zeros((1, 1, HIDDEN), np.float32), 3,
                               np.random.default_rng(0), 0.8, 50, 0.95, False)
    first = adapter.fast_ar.calls[0]
    assert bool(first["use_slow_hidden"][0]) is True
    assert int(first["input_pos"][0]) == 0


def test_fast_ar_later_positions_read_the_previous_codebook():
    adapter = _ready_adapter(fast_token=11)
    adapter.generate_codebooks(np.zeros((1, 1, HIDDEN), np.float32), 3,
                               np.random.default_rng(0), 0.8, 50, 0.95, False)
    second = adapter.fast_ar.calls[1]
    assert bool(second["use_slow_hidden"][0]) is False
    assert int(second["token_id"][0, 0]) == 3          # codebook 0, passed in
    assert int(adapter.fast_ar.calls[2]["token_id"][0, 0]) == 11   # then the fast AR's own


def test_fast_ar_walks_positions_zero_to_nine():
    adapter = _ready_adapter()
    adapter.generate_codebooks(np.zeros((1, 1, HIDDEN), np.float32), 0,
                               np.random.default_rng(0), 0.8, 50, 0.95, False)
    assert [int(call["input_pos"][0]) for call in adapter.fast_ar.calls] == list(
        range(NUM_CODEBOOKS))


def test_fast_ar_cache_is_the_codebook_width():
    adapter = _ready_adapter()
    adapter.generate_codebooks(np.zeros((1, 1, HIDDEN), np.float32), 0,
                               np.random.default_rng(0), 0.8, 50, 0.95, False)
    assert adapter.fast_ar.calls[0]["cache_key_0"].shape == (
        1, FAST_KV_HEADS, NUM_CODEBOOKS, FAST_HEAD_DIM)


def test_fast_ar_cache_grows_one_position_per_step():
    """Each step must see exactly its predecessors written, and nothing beyond.

    The graph masks by ``key <= input_pos``, so a delta written at the wrong offset is
    invisible to the mask and produces wrong attention with no error anywhere.
    """
    adapter = _ready_adapter()
    adapter.fast_ar = _fast_session()

    def handler(feed, call):
        written = np.abs(feed["cache_key_0"]).sum(axis=(0, 1, 3)) > 0
        assert int(written.sum()) == call, f"step {call} saw {int(written.sum())} written slots"
        logits = np.full((1, 1, CODEBOOK_SIZE), -20.0, np.float32)
        logits[0, 0, 7] = 20.0
        delta = np.ones((1, FAST_KV_HEADS, 1, FAST_HEAD_DIM), np.float32)
        return [logits] + [delta.copy() for _ in range(2 * FAST_LAYERS)]

    adapter.fast_ar._handler = handler
    adapter.generate_codebooks(np.zeros((1, 1, HIDDEN), np.float32), 0,
                               np.random.default_rng(0), 0.8, 50, 0.95, False)


def test_fast_ar_feeds_half_precision_when_the_graph_is_half():
    adapter = _ready_adapter(half=True)
    adapter.generate_codebooks(np.zeros((1, 1, HIDDEN), np.float32), 0,
                               np.random.default_rng(0), 0.8, 50, 0.95, False)
    assert adapter.fast_ar.calls[0]["slow_hidden"].dtype == np.float16


# ----------------------------------------------------------------------
# slow AR loop
# ----------------------------------------------------------------------

def test_slow_ar_stops_at_end_of_speech():
    adapter = _ready_adapter()
    session = _slow_session([3, 4, EOS_INDEX])
    codes = adapter.generate_codes(session, adapter.build_prompt(np.arange(4, dtype=np.int64)),
                                   {"do_sample": False}, np.random.default_rng(0))
    assert codes.shape == (NUM_CODEBOOKS, 2)


def test_end_of_speech_frame_is_not_emitted():
    adapter = _ready_adapter()
    session = _slow_session([EOS_INDEX])
    codes = adapter.generate_codes(session, adapter.build_prompt(np.arange(4, dtype=np.int64)),
                                   {"do_sample": False}, np.random.default_rng(0))
    assert codes.shape[1] == 0


def test_slow_ar_honours_the_frame_budget():
    adapter = _ready_adapter()
    session = _slow_session([3])            # never emits EOS
    codes = adapter.generate_codes(session, adapter.build_prompt(np.arange(4, dtype=np.int64)),
                                   {"do_sample": False, "max_new_tokens": 5},
                                   np.random.default_rng(0))
    assert codes.shape[1] == 5


def test_frame_budget_is_capped_by_the_sequence_limit():
    """The budget can never push the sequence past the cache width.

    The prompt here nearly fills the 2048-position window, so an unbounded
    ``max_new_tokens`` has to shrink to the handful of positions that are left rather than
    overrun the cache. A prompt this long is also what makes the assertion cheap to run.
    """
    adapter = _ready_adapter()
    adapter.reference_codes = np.zeros((NUM_CODEBOOKS, 8), np.int64)
    prefix = len(adapter.prefix_ids())
    room = MAX_SEQ_LEN - prefix - 8 - 6
    prompt = adapter.build_prompt(np.arange(room, dtype=np.int64))
    session = _slow_session([3])            # never emits EOS
    codes = adapter.generate_codes(session, prompt,
                                   {"do_sample": False, "max_new_tokens": MAX_SEQ_LEN * 2},
                                   np.random.default_rng(0))
    assert codes.shape[1] == MAX_SEQ_LEN - prompt.shape[2] == 6


def test_prefill_reads_the_whole_prompt_then_one_position_per_step():
    adapter = _ready_adapter()
    prompt = adapter.build_prompt(np.arange(4, dtype=np.int64))
    session = _slow_session([3, 4, EOS_INDEX])
    adapter.generate_codes(session, prompt, {"do_sample": False}, np.random.default_rng(0))
    widths = [call["codes"].shape[2] for call in session.calls]
    assert widths[0] == prompt.shape[2]
    assert widths[1:] == [1] * (len(widths) - 1)


def test_positions_continue_from_the_prompt():
    adapter = _ready_adapter()
    prompt = adapter.build_prompt(np.arange(4, dtype=np.int64))
    session = _slow_session([3, 4, 5, EOS_INDEX])
    adapter.generate_codes(session, prompt, {"do_sample": False}, np.random.default_rng(0))
    width = prompt.shape[2]
    assert list(session.calls[0]["input_pos"]) == list(range(width))
    assert [int(call["input_pos"][0]) for call in session.calls[1:]] == [
        width, width + 1, width + 2]


def test_slow_cache_is_the_full_window_from_the_first_call():
    adapter = _ready_adapter()
    session = _slow_session([EOS_INDEX])
    adapter.generate_codes(session, adapter.build_prompt(np.arange(4, dtype=np.int64)),
                           {"do_sample": False}, np.random.default_rng(0))
    assert session.calls[0]["cache_key_0"].shape == (
        1, SLOW_KV_HEADS, MAX_SEQ_LEN, SLOW_HEAD_DIM)


def test_slow_cache_starts_empty_and_fills_the_prompt_prefix():
    adapter = _ready_adapter()
    prompt = adapter.build_prompt(np.arange(4, dtype=np.int64))
    session = _slow_session([3, EOS_INDEX])
    adapter.generate_codes(session, prompt, {"do_sample": False}, np.random.default_rng(0))
    assert not session.calls[0]["cache_key_0"].any()
    written = np.abs(session.calls[1]["cache_key_0"]).sum(axis=(0, 1, 3)) > 0
    assert int(written.sum()) == prompt.shape[2]
    assert written[:prompt.shape[2]].all()


def test_slow_cache_grows_one_slot_per_decode_step():
    adapter = _ready_adapter()
    prompt = adapter.build_prompt(np.arange(4, dtype=np.int64))
    session = _slow_session([3, 4, 5, EOS_INDEX])
    adapter.generate_codes(session, prompt, {"do_sample": False}, np.random.default_rng(0))
    counts = [int((np.abs(call["cache_key_0"]).sum(axis=(0, 1, 3)) > 0).sum())
              for call in session.calls]
    assert counts == [0, prompt.shape[2], prompt.shape[2] + 1, prompt.shape[2] + 2]


def test_next_step_feeds_the_frame_it_just_produced():
    """Row 0 carries the semantic token in text-vocabulary space, rows 1..10 the codebooks."""
    adapter = _ready_adapter(fast_token=11)
    session = _slow_session([3, EOS_INDEX])
    adapter.generate_codes(session, adapter.build_prompt(np.arange(4, dtype=np.int64)),
                           {"do_sample": False}, np.random.default_rng(0))
    column = session.calls[1]["codes"]
    assert column.shape == (1, NUM_CODEBOOKS + 1, 1)
    assert int(column[0, 0, 0]) == SEMANTIC_BEGIN_ID + 3
    assert int(column[0, 1, 0]) == 3
    assert list(column[0, 2:, 0]) == [11] * (NUM_CODEBOOKS - 1)


def test_slow_ar_feeds_half_precision_when_the_graph_is_half():
    adapter = _ready_adapter(half=True)
    session = _slow_session([EOS_INDEX], half=True)
    adapter.generate_codes(session, adapter.build_prompt(np.arange(4, dtype=np.int64)),
                           {"do_sample": False}, np.random.default_rng(0))
    assert session.calls[0]["cache_key_0"].dtype == np.float16


# ----------------------------------------------------------------------
# synthesis
# ----------------------------------------------------------------------

def _request(ids=None, **params):
    ids = np.arange(6, dtype=np.int64) if ids is None else ids
    return AdapterSynthesisRequest(phoneme_ids=ids.reshape(1, -1),
                                   phoneme_lengths=np.asarray([ids.size], np.int64),
                                   params=params)


def test_synthesize_returns_audio_and_frame_count():
    adapter = _ready_adapter()
    session = _slow_session([3, 4, EOS_INDEX])
    result = adapter.synthesize(_request(do_sample=False), session)
    assert result.extras["frame_count"] == 2
    assert result.audio.shape == (2 * FRAME_SIZE,)
    assert result.audio.dtype == np.float32


def test_synthesize_carries_the_reference_text():
    adapter = _ready_adapter()
    result = adapter.synthesize(_request(do_sample=False), _slow_session([3, EOS_INDEX]))
    assert result.extras["reference_text"] == adapter.reference_text


def test_synthesize_returns_empty_audio_when_nothing_is_emitted():
    adapter = _ready_adapter()
    result = adapter.synthesize(_request(do_sample=False), _slow_session([EOS_INDEX]))
    assert result.audio.size == 0


def test_synthesize_is_reproducible_for_a_seed():
    logits_tokens = [3, 4, 5, EOS_INDEX]
    first = _ready_adapter().synthesize(_request(seed=99), _slow_session(logits_tokens))
    second = _ready_adapter().synthesize(_request(seed=99), _slow_session(logits_tokens))
    assert np.array_equal(first.audio, second.audio)


def test_synthesize_without_the_fast_graph_is_refused():
    adapter = _adapter_with_voice()
    adapter.decoder = _decoder_session()
    with pytest.raises(RuntimeError, match="fast_ar_path"):
        adapter.synthesize(_request(), _slow_session([EOS_INDEX]))


def test_synthesize_without_the_decoder_is_refused():
    adapter = _adapter_with_voice()
    adapter.fast_ar = _fast_session()
    with pytest.raises(RuntimeError, match="codec_decoder_path"):
        adapter.synthesize(_request(), _slow_session([EOS_INDEX]))


def test_decode_codes_feeds_the_decoder_one_batch_of_ten_rows():
    adapter = _ready_adapter()
    adapter.decode_codes(np.zeros((NUM_CODEBOOKS, 9), np.int64))
    assert adapter.decoder.calls[0]["codes"].shape == (1, NUM_CODEBOOKS, 9)


def test_single_graph_path_is_not_offered():
    adapter = ArkTTSAdapter()
    with pytest.raises(NotImplementedError):
        adapter.build_feed_dict(_request(), None)
    with pytest.raises(NotImplementedError):
        adapter.parse_outputs([], _request())


# ----------------------------------------------------------------------
# configure() — both checkpoints go through the same code path
# ----------------------------------------------------------------------

class _VoiceConfig:
    def __init__(self, **engine_params):
        self.engine_params = engine_params
        self.lang_code = engine_params.pop("lang_code", None)


def test_configure_loads_the_voice_asset(tmp_path, monkeypatch):
    monkeypatch.setattr("phoonnx.engines.arktts.make_session", lambda path, providers=None: path)
    monkeypatch.setattr("phoonnx.engines.arktts.BPETokenizer", lambda path: path)
    adapter = ArkTTSAdapter()
    adapter.configure(_VoiceConfig(fast_ar_path="fast", codec_decoder_path="codec",
                                   bpe_tokenizer_path="tok",
                                   voice_codes_path=_voice_file(tmp_path)))
    assert adapter.fast_ar == "fast"
    assert adapter.decoder == "codec"
    assert adapter.reference_codes.shape == (NUM_CODEBOOKS, 4)


@pytest.mark.parametrize("lang_code", ["eu-ES", "en-US", "ja-JP"])
def test_configure_is_language_agnostic(tmp_path, monkeypatch, lang_code):
    """Zortzi (Basque) and Audio8 (11 languages) take the identical path.

    ArkTTS has no language token — the language comes from the text and the reference
    clip — so nothing in the adapter may branch on the voice's language.
    """
    monkeypatch.setattr("phoonnx.engines.arktts.make_session", lambda path, providers=None: path)
    monkeypatch.setattr("phoonnx.engines.arktts.BPETokenizer", lambda path: path)
    adapter = ArkTTSAdapter()
    adapter.configure(_VoiceConfig(fast_ar_path="fast", codec_decoder_path="codec",
                                   bpe_tokenizer_path="tok", lang_code=lang_code,
                                   voice_codes_path=_voice_file(tmp_path)))
    assert adapter.reference_text == "Aurrelaria prest dago jokatzeko."


def test_configure_tolerates_a_voice_without_aux_paths():
    adapter = ArkTTSAdapter()
    adapter.configure(_VoiceConfig())
    assert adapter.fast_ar is None and adapter.reference_codes is None


# ----------------------------------------------------------------------
# voice index
# ----------------------------------------------------------------------

def _index():
    from phoonnx.model_manager import TTSModelManager
    path = TTSModelManager.voice_index_path() / "arktts.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_voice_index_is_bundled():
    from phoonnx.model_manager import TTSModelManager
    assert any(path.name == "arktts.json" for path in TTSModelManager.voice_index_files())


def test_voice_index_entries_are_complete():
    for voice_id, entry in _index().items():
        assert entry["engine"] == "arktts", voice_id
        assert entry["voice_id"] == voice_id
        aux = entry["aux_model_urls"]
        for key in ("fast_ar_path", "codec_decoder_path", "bpe_tokenizer_path",
                    "voice_codes_path"):
            assert key in aux, f"{voice_id} is missing {key}"
        assert entry["model_url"].endswith(".onnx")


def test_voice_index_covers_both_checkpoints():
    mirrors = {entry["model_url"].split("/resolve/")[0] for entry in _index().values()}
    assert any("zortzi" in mirror for mirror in mirrors)
    assert any("audio8" in mirror for mirror in mirrors)


def test_voice_index_languages_are_tagged():
    for voice_id, entry in _index().items():
        assert entry["lang"], voice_id
