import json

import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.magpie import (
    AUDIO_EOS,
    BYTE_LANGUAGES,
    BYTE_TOKENIZER_PREFIX,
    CHAR_LANGUAGES,
    MagpieAdapter,
    MagpieTokenizer,
    chunk_text,
)

# A miniature stand-in for the real checkpoint: 2 codebooks over 2 stacked frames,
# a codebook of 10 real tokens plus the 8 special ones, and 2 decoder layers.
CONFIG = {
    "num_audio_codebooks": 2,
    "frame_stacking_factor": 2,
    "codebook_size": 10,
    "num_all_tokens_per_codebook": 18,
    "decoder_n_layers": 2,
    "local_transformer_n_layers": 2,
    "sa_n_heads": 2,
    "sa_d_head": 4,
    "estimate_alignment_from_layers": [1],
    "transcript_decoder_layers": [0, 1],
    "language_to_tokenizer": {
        "en": ["english_phoneme"],
        "fr": ["french_chartokenizer"],
        "ar-MSA": ["arabic_MSA_chartokenizer"],
        "zh": ["mandarin_phoneme"],
    },
    "inference": {
        "max_decoder_steps": 40,
        "attention_prior_epsilon": 0.1,
        "attention_prior_lookahead_window": 2,
        "attention_sink_threshold": 3,
        "eos_detection_method": "argmax_or_multinomial_any",
        "cfg_scale": 2.0,
        "min_generated_frames": 0,
    },
}

STACKED = CONFIG["num_audio_codebooks"] * CONFIG["frame_stacking_factor"]
VOCAB = CONFIG["num_all_tokens_per_codebook"]
EOS_ID = CONFIG["codebook_size"] + AUDIO_EOS


def _tokenizer_asset(tmp_path):
    """A tokenizer export with one byte table and one character table."""
    tokens = ["<pad>", "</s>", "<unk>"] + [chr(i) for i in range(256)]   # byt5 layout
    arabic = [" ", "ا", "ب", "ت"]
    payload = {
        "vocab_size": len(tokens) + len(arabic),
        "tokens": tokens + arabic,
        "tokenizer_offsets": {"french_chartokenizer": 0,
                              "arabic_MSA_chartokenizer": len(tokens)},
        "num_tokens_per_tokenizer": {"french_chartokenizer": len(tokens),
                                     "arabic_MSA_chartokenizer": len(arabic)},
        "tokenizer_pad_ids": {"french_chartokenizer": 0,
                              "arabic_MSA_chartokenizer": len(tokens)},
        "eos_id": len(tokens) + len(arabic),
    }
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _Session:
    """Records every feed dict and replays a scripted list of outputs."""

    def __init__(self, outputs, input_names=()):
        self._outputs = outputs
        self._input_names = list(input_names)
        self.calls = []

    def run(self, _output_names, feed):
        self.calls.append({k: (v.copy() if isinstance(v, np.ndarray) else v)
                           for k, v in feed.items()})
        result = self._outputs
        return result(feed) if callable(result) else result

    def get_inputs(self):
        return [type("I", (), {"name": n})() for n in self._input_names]


def _adapter(tmp_path=None):
    adapter = MagpieAdapter()
    adapter.config = dict(CONFIG)
    adapter.speakers = {"Aria": 0, "Leo": 1}
    adapter.context_embeddings = np.zeros((2, 3, 6), np.float32)
    if tmp_path is not None:
        adapter.tokenizer = MagpieTokenizer(str(_tokenizer_asset(tmp_path)))
    return adapter


# ----------------------------------------------------------------------
# Registration and detection
# ----------------------------------------------------------------------

def test_magpie_registered():
    from phoonnx.engines import list_engines
    assert "magpie" in list_engines()


def test_magpie_registered_below_every_other_engine():
    # Magpie must be probed last: its detect() matches on input names, and it should
    # never be given the chance to steal a voice another adapter claims by config.
    from phoonnx.engines import _PRIORITIES
    assert _PRIORITIES["magpie"] == 19
    others = {n: p for n, p in _PRIORITIES.items() if n != "magpie"}
    assert min(others.values()) > 19


def test_detect_by_engine_name():
    assert MagpieAdapter.detect({"engine": "magpie"})
    assert MagpieAdapter.detect({"model_type": "magpie-tts"})
    assert not MagpieAdapter.detect({"engine": "sparktts"})
    assert not MagpieAdapter.detect(None)


def test_detect_by_decoder_input_signature():
    magpie = _Session(None, ["x", "pos", "self_k", "self_v", "cross_k", "cross_v",
                             "cond_mask", "attn_prior"])
    assert MagpieAdapter.detect(None, magpie)
    # A plain autoregressive decoder has a KV cache but no cross-attention prior.
    other = _Session(None, ["x", "self_k", "self_v"])
    assert not MagpieAdapter.detect(None, other)


def test_default_params_match_the_checkpoint():
    # model_config.yaml inference_parameters: temperature 0.6, topk 80, cfg_scale 2.5
    assert MagpieAdapter().default_params() == {"temperature": 0.6, "top_k": 80.0,
                                                "cfg_scale": 2.5}


# ----------------------------------------------------------------------
# Tokenizer
# ----------------------------------------------------------------------

def test_byte_tokenizer_matches_the_byt5_layout(tmp_path):
    tokenizer = MagpieTokenizer(str(_tokenizer_asset(tmp_path)))
    ids = tokenizer.encode("ab", "fr", "french_chartokenizer")
    assert ids == [BYTE_TOKENIZER_PREFIX + ord("a"),
                   BYTE_TOKENIZER_PREFIX + ord("b"),
                   tokenizer.eos_id]


def test_byte_tokenizer_encodes_multibyte_characters_as_bytes(tmp_path):
    tokenizer = MagpieTokenizer(str(_tokenizer_asset(tmp_path)))
    ids = tokenizer.encode("é", "fr", "french_chartokenizer")
    assert ids[:-1] == [BYTE_TOKENIZER_PREFIX + b for b in "é".encode("utf-8")]
    assert len(ids) == 3   # two bytes plus end-of-sentence


def test_character_tokenizer_uses_the_exported_symbol_table(tmp_path):
    tokenizer = MagpieTokenizer(str(_tokenizer_asset(tmp_path)))
    symbols = tokenizer.symbol_map("arabic_MSA_chartokenizer")
    ids = tokenizer.encode("اب", "ar", "arabic_MSA_chartokenizer")
    assert ids == [symbols["ا"], symbols["ب"], tokenizer.eos_id]


def test_character_tokenizer_drops_symbols_outside_the_table(tmp_path):
    tokenizer = MagpieTokenizer(str(_tokenizer_asset(tmp_path)))
    ids = tokenizer.encode("اZب", "ar", "arabic_MSA_chartokenizer")
    assert len(ids) == 3   # the Latin letter is not in the Arabic table


def test_ipa_languages_are_refused_rather_than_approximated(tmp_path):
    tokenizer = MagpieTokenizer(str(_tokenizer_asset(tmp_path)))
    with pytest.raises(NotImplementedError, match="scriptconv"):
        tokenizer.encode("hello", "en", "english_phoneme")


def test_unknown_sub_tokenizer_is_an_error(tmp_path):
    tokenizer = MagpieTokenizer(str(_tokenizer_asset(tmp_path)))
    with pytest.raises(ValueError, match="sub-tokenizer"):
        tokenizer.encode("bonjour", "fr", "klingon_chartokenizer")


def test_supported_language_sets_do_not_overlap():
    assert not BYTE_LANGUAGES & CHAR_LANGUAGES


def test_tokenizer_name_lookup_handles_region_tags(tmp_path):
    adapter = _adapter(tmp_path)
    assert adapter.tokenizer_name_for("fr") == "french_chartokenizer"
    assert adapter.tokenizer_name_for("fr-FR") == "french_chartokenizer"
    assert adapter.tokenizer_name_for("ar-MSA") == "arabic_MSA_chartokenizer"
    with pytest.raises(ValueError, match="no tokenizer"):
        adapter.tokenizer_name_for("xx")


def test_encode_text_returns_one_id_list_per_chunk(tmp_path):
    adapter = _adapter(tmp_path)

    class _Voice:
        lang = "fr"

    chunks = adapter.encode_text("Bonjour. Ca va?", _Voice(), None)
    assert len(chunks) == 2
    assert all(ids[-1] == adapter.tokenizer.eos_id for ids in chunks)


def test_encode_text_without_a_tokenizer_is_an_error():
    with pytest.raises(RuntimeError, match="tokenizer_path"):
        _adapter().encode_text("hi", None, None)


# ----------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------

def test_chunk_text_splits_on_sentences():
    assert chunk_text("One. Two. Three.") == ["One.", "Two.", "Three."]


def test_chunk_text_breaks_an_oversized_sentence_on_whitespace():
    sentence = " ".join(["word"] * 100)
    chunks = chunk_text(sentence, max_len=40)
    assert len(chunks) > 1
    assert all(len(c) <= 40 for c in chunks)
    assert " ".join(chunks).split() == sentence.split()


def test_chunk_text_of_blank_input_is_empty():
    assert chunk_text("   ") == []


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------

def test_special_tokens_are_forbidden_but_end_of_speech_is_allowed():
    adapter = _adapter()
    forbidden = adapter.forbidden_token_ids(forbid_eos=False)
    assert EOS_ID not in forbidden
    assert len(forbidden) == 7
    assert EOS_ID in adapter.forbidden_token_ids(forbid_eos=True)


def test_sampling_never_returns_a_special_token():
    adapter = _adapter()
    rng = np.random.default_rng(0)
    # Make every special token overwhelmingly likely; they must still be excluded.
    logits = np.zeros(VOCAB, np.float32)
    logits[CONFIG["codebook_size"]:] = 50.0
    logits[EOS_ID] = -50.0
    token = adapter.sample_codebook(logits, 1.0, VOCAB, forbid_eos=True,
                                   force_eos=False, rng=rng)
    assert token < CONFIG["codebook_size"]


def test_zero_temperature_is_greedy():
    adapter = _adapter()
    logits = np.zeros(VOCAB, np.float32)
    logits[4] = 9.0
    token = adapter.sample_codebook(logits, 0.0, VOCAB, False, False,
                                    np.random.default_rng(0))
    assert token == 4


def test_top_k_of_one_is_greedy_at_any_temperature():
    adapter = _adapter()
    logits = np.arange(VOCAB, dtype=np.float32)
    logits[3] = 100.0
    for _ in range(5):
        token = adapter.sample_codebook(logits, 5.0, 1, False, False,
                                        np.random.default_rng(1))
        assert token == 3


def test_forcing_end_of_speech_short_circuits_sampling():
    adapter = _adapter()
    logits = np.zeros(VOCAB, np.float32)
    logits[2] = 99.0
    assert adapter.sample_codebook(logits, 1.0, VOCAB, False, True,
                                   np.random.default_rng(0)) == EOS_ID


def test_frame_sampling_reads_each_codebook_from_its_own_slice():
    adapter = _adapter()
    logits = np.full(STACKED * VOCAB, -10.0, np.float32)
    wanted = [1, 2, 3, 4]
    for slot, token in enumerate(wanted):
        logits[slot * VOCAB + token] = 10.0
    frame = adapter.sample_frame_from_logits(logits, 0.0, 1, False, False,
                                             np.random.default_rng(0))
    assert frame.shape == (CONFIG["num_audio_codebooks"], CONFIG["frame_stacking_factor"])
    # slot index is codebook + num_codebooks * stack_index
    assert frame[0, 0] == 1 and frame[1, 0] == 2
    assert frame[0, 1] == 3 and frame[1, 1] == 4


# ----------------------------------------------------------------------
# Local transformer (refiner) KV loop
# ----------------------------------------------------------------------

def _local_sessions(picks):
    """A refiner that always names ``picks[cb]`` and grows its cache by one."""
    state = {"cb": 0}

    def local(feed):
        cb = int(feed["cb"])
        state["cb"] = cb
        logits = np.full((feed["h"].shape[0], VOCAB), -10.0, np.float32)
        logits[:, picks[cb]] = 10.0
        grown = feed["cache_k"].shape[2] + 1
        shape = (CONFIG["local_transformer_n_layers"], feed["h"].shape[0], grown,
                 CONFIG["sa_n_heads"], CONFIG["sa_d_head"])
        return [logits, np.zeros(shape, np.float32), np.zeros(shape, np.float32)]

    def embed(feed):
        return [np.zeros((feed["tok"].shape[0], 1, 6), np.float32)]

    return _Session(local), _Session(embed)


def test_refiner_walks_every_codebook_once_and_grows_its_cache():
    adapter = _adapter()
    picks = [1, 2, 3, 4]
    adapter.local, adapter.lt_embed = _local_sessions(picks)
    frame = adapter.refine_frame(np.zeros((2, 6), np.float32), 0.0, 1, 2.0,
                                 False, False, np.random.default_rng(0))

    assert len(adapter.local.calls) == STACKED
    # positions are the codebook index, in order
    assert [int(c["pos"][0]) for c in adapter.local.calls] == list(range(STACKED))
    assert [int(c["cb"]) for c in adapter.local.calls] == list(range(STACKED))
    # the cache handed in grows by exactly one row per codebook
    assert [c["cache_k"].shape[2] for c in adapter.local.calls] == list(range(STACKED))
    # the first call starts from an empty cache
    assert adapter.local.calls[0]["cache_k"].shape[2] == 0
    assert frame[0, 0] == 1 and frame[1, 0] == 2
    assert frame[0, 1] == 3 and frame[1, 1] == 4


def test_refiner_feeds_each_chosen_token_back_in():
    adapter = _adapter()
    picks = [5, 6, 7, 8]
    adapter.local, adapter.lt_embed = _local_sessions(picks)
    adapter.refine_frame(np.zeros((2, 6), np.float32), 0.0, 1, 2.0, False, False,
                         np.random.default_rng(0))
    # one embedding lookup per codebook, each carrying the token just chosen
    assert len(adapter.lt_embed.calls) == STACKED
    assert [int(c["tok"][0]) for c in adapter.lt_embed.calls] == picks


def test_refiner_applies_guidance_across_the_batch():
    adapter = _adapter()

    def local(feed):
        batch = feed["h"].shape[0]
        logits = np.zeros((batch, VOCAB), np.float32)
        logits[0, 1] = 1.0        # conditional prefers token 1
        logits[1, 2] = 1.0        # unconditional prefers token 2
        shape = (2, batch, feed["cache_k"].shape[2] + 1, CONFIG["sa_n_heads"],
                 CONFIG["sa_d_head"])
        return [logits, np.zeros(shape, np.float32), np.zeros(shape, np.float32)]

    adapter.local = _Session(local)
    adapter.lt_embed = _Session(lambda f: [np.zeros((f["tok"].shape[0], 1, 6), np.float32)])
    frame = adapter.refine_frame(np.zeros((2, 6), np.float32), 0.0, 1, 2.0,
                                 False, False, np.random.default_rng(0))
    # cfg 2.0: 2*cond - 1*uncond, so token 1 scores +2 and token 2 scores -1
    assert set(np.unique(frame)) == {1}


def test_refiner_output_is_ordered_codebook_by_stack():
    adapter = _adapter()
    adapter.local, adapter.lt_embed = _local_sessions([1, 2, 3, 4])
    frame = adapter.refine_frame(np.zeros((2, 6), np.float32), 0.0, 1, 2.0,
                                 False, False, np.random.default_rng(0))
    # flattening a frame for audio_embed must give back the sampling order
    assert frame.T.reshape(-1).tolist() == [1, 2, 3, 4]


# ----------------------------------------------------------------------
# Alignment and the attention prior
# ----------------------------------------------------------------------

def test_mean_cross_attention_takes_the_newest_step():
    adapter = _adapter()
    # (layers, batch, heads, steps, text)
    probs = np.zeros((2, 1, 2, 3, 4), np.float32)
    probs[:, :, :, -1, 2] = 1.0
    scores = adapter.mean_cross_attention(probs)
    assert scores.shape == (1, 4)
    assert int(np.argmax(scores[0])) == 2


def test_mean_cross_attention_can_restrict_to_chosen_layers():
    adapter = _adapter()
    probs = np.zeros((2, 1, 1, 1, 4), np.float32)
    probs[0, :, :, :, 0] = 1.0     # layer 0 looks at position 0
    probs[1, :, :, :, 3] = 1.0     # layer 1 looks at position 3
    assert int(np.argmax(adapter.mean_cross_attention(probs, [1])[0])) == 3
    assert int(np.argmax(adapter.mean_cross_attention(probs, [0])[0])) == 0


def test_alignment_only_looks_forward():
    adapter = _adapter()
    scores = np.zeros(12, np.float32)
    scores[0] = 1.0        # a strong pull backwards, outside the window
    scores[6] = 0.5
    attended = adapter.most_attended_position(scores, last_attended=5, text_len=12,
                                              counter={})
    assert attended >= 5


def test_alignment_steps_past_an_attention_sink():
    adapter = _adapter()
    scores = np.zeros(12, np.float32)
    counter = {5: CONFIG["inference"]["attention_sink_threshold"]}
    attended = adapter.most_attended_position(scores, last_attended=5, text_len=12,
                                              counter=counter)
    assert attended >= 6


def test_alignment_at_the_end_of_the_text_reports_the_last_position():
    adapter = _adapter()
    scores = np.zeros(8, np.float32)
    attended = adapter.most_attended_position(scores, last_attended=7, text_len=8,
                                              counter={})
    assert attended == 7


def test_alignment_counts_every_visit():
    adapter = _adapter()
    counter = {}
    scores = np.zeros(12, np.float32)
    scores[4] = 1.0
    for _ in range(3):
        adapter.most_attended_position(scores, 4, 12, counter)
    assert sum(counter.values()) == 3


def test_prior_opens_a_window_around_the_spoken_position():
    adapter = _adapter()
    prior = adapter.build_prior(text_len=12, attended=5, counter={}, batch=1,
                                text_positions=12)
    epsilon = CONFIG["inference"]["attention_prior_epsilon"]
    window = CONFIG["inference"]["attention_prior_lookahead_window"]
    assert prior.shape == (1, 1, 12)
    for position in range(4, 5 + window + 1):
        assert prior[0, 0, position] == 1.0
    assert prior[0, 0, 0] == pytest.approx(epsilon)
    assert prior[0, 0, 11] == pytest.approx(epsilon)


def test_prior_is_flat_for_very_short_text():
    adapter = _adapter()
    prior = adapter.build_prior(text_len=4, attended=1, counter={}, batch=2,
                                text_positions=4)
    assert np.all(prior == 1.0)


def test_prior_damps_everything_before_an_attention_sink():
    adapter = _adapter()
    counter = {2: CONFIG["inference"]["attention_sink_threshold"]}
    prior = adapter.build_prior(text_len=12, attended=6, counter=counter, batch=1,
                                text_positions=12)
    epsilon = CONFIG["inference"]["attention_prior_epsilon"]
    assert np.allclose(prior[0, 0, :3], epsilon)


def test_prior_is_shared_by_both_guidance_branches():
    adapter = _adapter()
    prior = adapter.build_prior(text_len=12, attended=5, counter={}, batch=2,
                                text_positions=12)
    assert prior.shape[0] == 2
    assert np.array_equal(prior[0], prior[1])


# ----------------------------------------------------------------------
# End of speech
# ----------------------------------------------------------------------

def test_end_of_speech_reports_the_first_frame_that_carries_it():
    adapter = _adapter()
    codes = np.zeros((2, 2), np.int64)
    codes[1, 1] = EOS_ID
    assert adapter.find_eos_frame(codes) == 1.0


def test_no_end_of_speech_is_infinite():
    adapter = _adapter()
    assert adapter.find_eos_frame(np.zeros((2, 2), np.int64)) == float("inf")


def test_end_of_speech_trusts_whichever_read_ends_first():
    adapter = _adapter()
    sampled = np.zeros((2, 2), np.int64)
    greedy = np.zeros((2, 2), np.int64)
    greedy[0, 1] = EOS_ID
    assert adapter.detect_eos(sampled, greedy) == 1.0
    sampled[0, 0] = EOS_ID
    assert adapter.detect_eos(sampled, greedy) == 0.0


# ----------------------------------------------------------------------
# Speakers
# ----------------------------------------------------------------------

def test_speaker_can_be_named_or_indexed():
    adapter = _adapter()
    assert adapter._resolve_speaker(AdapterSynthesisRequest(
        phoneme_ids=np.zeros((1, 1), np.int64),
        phoneme_lengths=np.array([1]), speaker_id="Leo")) == 1
    assert adapter._resolve_speaker(AdapterSynthesisRequest(
        phoneme_ids=np.zeros((1, 1), np.int64),
        phoneme_lengths=np.array([1]), speaker_id=0)) == 0


def test_unknown_speaker_name_is_an_error():
    adapter = _adapter()
    with pytest.raises(ValueError, match="no voice"):
        adapter._resolve_speaker(AdapterSynthesisRequest(
            phoneme_ids=np.zeros((1, 1), np.int64),
            phoneme_lengths=np.array([1]), speaker_id="Nobody"))


def test_speaker_index_outside_the_baked_embeddings_is_an_error():
    adapter = _adapter()
    with pytest.raises(ValueError, match="outside"):
        adapter._resolve_speaker(AdapterSynthesisRequest(
            phoneme_ids=np.zeros((1, 1), np.int64),
            phoneme_lengths=np.array([1]), speaker_id=9))


# ----------------------------------------------------------------------
# Conditioning
# ----------------------------------------------------------------------

def test_guidance_zeroes_the_unconditional_branch():
    adapter = _adapter()
    adapter.encoder = _Session(lambda f: [np.ones((1, f["text"].shape[1], 6), np.float32)])
    adapter.cross_kv = _Session(lambda f: [np.zeros((2, f["cond"].shape[0], 4, 1, 2), np.float32),
                                           np.zeros((2, f["cond"].shape[0], 4, 1, 2), np.float32)])
    cond, cond_mask, _, _ = adapter.encode_conditioning(np.arange(4, dtype=np.int64), True)
    assert cond.shape[0] == 2
    assert np.all(cond[0] == 1.0)
    assert np.all(cond[1] == 0.0)
    # the unconditional mask keeps only the first text position
    assert np.all(cond_mask[0] == 1.0)
    assert cond_mask[1].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_without_guidance_the_batch_stays_at_one():
    adapter = _adapter()
    adapter.encoder = _Session(lambda f: [np.ones((1, f["text"].shape[1], 6), np.float32)])
    adapter.cross_kv = _Session(lambda f: [np.zeros((2, 1, 4, 1, 2), np.float32),
                                           np.zeros((2, 1, 4, 1, 2), np.float32)])
    cond, cond_mask, _, _ = adapter.encode_conditioning(np.arange(4, dtype=np.int64), False)
    assert cond.shape[0] == 1
    assert cond_mask.shape == (1, 4)


# ----------------------------------------------------------------------
# Decoder KV loop
# ----------------------------------------------------------------------

def _loop_adapter(eos_at_step, exact):
    """Wire an adapter whose decoder ends the utterance at a chosen step."""
    adapter = _adapter()
    adapter._params = {"exact_decode": exact}
    text_positions = 8
    heads, head_dim = CONFIG["sa_n_heads"], CONFIG["sa_d_head"]
    state = {"step": 0}

    adapter.encoder = _Session(
        lambda f: [np.ones((1, f["text"].shape[1], 6), np.float32)])
    adapter.cross_kv = _Session(
        lambda f: [np.zeros((2, f["cond"].shape[0], text_positions, 1, 2), np.float32),
                   np.zeros((2, f["cond"].shape[0], text_positions, 1, 2), np.float32)])
    adapter.audio_embed = _Session(
        lambda f: [np.zeros((f["codes"].shape[0], 1, 6), np.float32)])

    def decoder(feed):
        step = state["step"]
        state["step"] += 1
        batch, steps = feed["x"].shape[0], feed["x"].shape[1]
        logits = np.full((batch, steps, STACKED * VOCAB), -10.0, np.float32)
        if step >= eos_at_step:
            for slot in range(STACKED):
                logits[:, -1, slot * VOCAB + EOS_ID] = 10.0
        else:
            for slot in range(STACKED):
                logits[:, -1, slot * VOCAB + 3] = 10.0
        grown = feed["self_k"].shape[2] + steps
        cache = np.zeros((CONFIG["decoder_n_layers"], batch, grown, heads, head_dim),
                         np.float32)
        attn = np.zeros((CONFIG["decoder_n_layers"], batch, 1, steps, text_positions),
                        np.float32)
        attn[..., min(step, text_positions - 1)] = 1.0
        return [logits, np.zeros((batch, steps, 6), np.float32), cache, cache, attn]

    adapter.decoder = _Session(decoder)
    adapter.local, adapter.lt_embed = _local_sessions([3, 3, 3, 3])
    return adapter


def test_decoder_loop_prefills_context_then_steps_one_frame_at_a_time():
    adapter = _loop_adapter(eos_at_step=99, exact=False)
    adapter.generate_codes(np.arange(8, dtype=np.int64), 0,
                           {"temperature": 0.0, "top_k": 1, "cfg_scale": 2.0})
    calls = adapter.decoder.calls
    # the first pass carries the 3-frame speaker context plus the start frame
    assert calls[0]["x"].shape[1] == adapter.context_embeddings.shape[1] + 1
    assert calls[0]["self_k"].shape[2] == 0
    # every later pass carries exactly one frame, on top of a cache that keeps growing
    assert all(c["x"].shape[1] == 1 for c in calls[1:])
    assert [c["self_k"].shape[2] for c in calls[1:4]] == [4, 5, 6]


def test_decoder_loop_advances_the_position_index_with_the_cache():
    adapter = _loop_adapter(eos_at_step=99, exact=False)
    adapter.generate_codes(np.arange(8, dtype=np.int64), 0,
                           {"temperature": 0.0, "top_k": 1, "cfg_scale": 2.0})
    calls = adapter.decoder.calls
    assert calls[0]["pos"].tolist() == list(range(4))
    assert [int(c["pos"][0]) for c in calls[1:4]] == [4, 5, 6]


def test_exact_mode_recomputes_the_whole_sequence_with_an_empty_cache():
    # NeMo's default re-applies the newest prior over the whole history, so the past
    # changes every step; a KV cache would give a different sample path.
    adapter = _loop_adapter(eos_at_step=99, exact=True)
    adapter.generate_codes(np.arange(8, dtype=np.int64), 0,
                           {"temperature": 0.0, "top_k": 1, "cfg_scale": 2.0})
    calls = adapter.decoder.calls
    assert all(c["self_k"].shape[2] == 0 for c in calls)
    lengths = [c["x"].shape[1] for c in calls]
    assert lengths == list(range(lengths[0], lengths[0] + len(lengths)))


def test_decoder_loop_feeds_the_prior_back_in_after_the_first_pass():
    adapter = _loop_adapter(eos_at_step=99, exact=False)
    adapter.generate_codes(np.arange(8, dtype=np.int64), 0,
                           {"temperature": 0.0, "top_k": 1, "cfg_scale": 2.0})
    calls = adapter.decoder.calls
    # the first pass has no prior yet, so it is flat
    assert np.all(calls[0]["attn_prior"] == 1.0)
    # later passes carry a real prior: a narrow window of ones over damped positions
    later = calls[3]["attn_prior"]
    assert later.shape[1] == 1
    assert later.min() < 1.0 and later.max() == pytest.approx(1.0)


def test_decoder_loop_stops_once_end_of_speech_is_reached():
    adapter = _loop_adapter(eos_at_step=6, exact=False)
    codes = adapter.generate_codes(np.arange(8, dtype=np.int64), 0,
                                   {"temperature": 0.0, "top_k": 1, "cfg_scale": 2.0})
    assert len(adapter.decoder.calls) < CONFIG["inference"]["max_decoder_steps"]
    assert codes.shape[0] == CONFIG["num_audio_codebooks"]
    assert codes.shape[1] >= 4     # the codec needs at least four frames


def test_decoder_loop_respects_the_step_cap_when_nothing_ends():
    adapter = _loop_adapter(eos_at_step=10 ** 6, exact=False)
    codes = adapter.generate_codes(np.arange(8, dtype=np.int64), 0,
                                   {"temperature": 0.0, "top_k": 1, "cfg_scale": 2.0})
    cap = CONFIG["inference"]["max_decoder_steps"] // CONFIG["frame_stacking_factor"]
    assert len(adapter.decoder.calls) == cap
    assert codes.shape[1] == cap * CONFIG["frame_stacking_factor"]


def test_generated_codes_never_contain_a_special_token():
    adapter = _loop_adapter(eos_at_step=8, exact=False)
    codes = adapter.generate_codes(np.arange(8, dtype=np.int64), 0,
                                   {"temperature": 0.0, "top_k": 1, "cfg_scale": 2.0})
    assert codes.max() < CONFIG["codebook_size"]


# ----------------------------------------------------------------------
# Synthesis wiring
# ----------------------------------------------------------------------

def test_synthesize_reports_missing_graphs():
    adapter = _adapter()
    request = AdapterSynthesisRequest(phoneme_ids=np.zeros((1, 3), np.int64),
                                      phoneme_lengths=np.array([3]), speaker_id=0)
    with pytest.raises(RuntimeError, match="missing graphs"):
        adapter.synthesize(request, _Session(None))


def test_synthesize_requires_the_speaker_embeddings():
    adapter = _loop_adapter(eos_at_step=4, exact=False)
    adapter.codec = _Session(lambda f: [np.zeros((1, 64), np.float32)])
    adapter.context_embeddings = None
    request = AdapterSynthesisRequest(phoneme_ids=np.zeros((1, 3), np.int64),
                                      phoneme_lengths=np.array([3]), speaker_id=0)
    with pytest.raises(RuntimeError, match="context_embeddings_path"):
        adapter.synthesize(request, adapter.decoder)


def test_synthesize_returns_a_flat_waveform_and_the_codes():
    adapter = _loop_adapter(eos_at_step=4, exact=False)
    adapter.codec = _Session(lambda f: [np.zeros((1, f["codes"].shape[2] * 8), np.float32)])
    request = AdapterSynthesisRequest(phoneme_ids=np.zeros((1, 8), np.int64),
                                      phoneme_lengths=np.array([8]), speaker_id="Aria",
                                      params={"temperature": 0.0, "top_k": 1})
    result = adapter.synthesize(request, adapter.decoder)
    assert result.audio.ndim == 1
    assert result.audio.dtype == np.float32
    assert result.extras["codes"].shape[0] == CONFIG["num_audio_codebooks"]
    # the codec is handed a batch of one
    assert adapter.codec.calls[0]["codes"].shape[0] == 1


def test_single_graph_hooks_are_refused():
    adapter = _adapter()
    with pytest.raises(NotImplementedError):
        adapter.build_feed_dict(None, None)
    with pytest.raises(NotImplementedError):
        adapter.parse_outputs([], None)


# ----------------------------------------------------------------------
# Voice index
# ----------------------------------------------------------------------

def test_voice_index_entries_are_consistent():
    from pathlib import Path
    path = Path(__file__).parent.parent / "phoonnx" / "voice_index" / "magpie.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    assert index
    for voice_id, entry in index.items():
        assert entry["voice_id"] == voice_id
        assert entry["engine"] == "magpie"
        assert entry["lang"] in (BYTE_LANGUAGES | CHAR_LANGUAGES)
        aux = entry["aux_model_urls"]
        for key in ("text_encoder_path", "cross_kv_path", "local_step_path",
                    "audio_embed_path", "lt_embed_path", "codec_decoder_path",
                    "tokenizer_path", "context_embeddings_path",
                    "magpie_config_path", "speakers_path"):
            assert key in aux, f"{voice_id} missing {key}"
        assert entry["engine_options"]["speaker_id"] in range(5)


def test_voice_index_covers_every_speaker_and_language():
    from pathlib import Path
    path = Path(__file__).parent.parent / "phoonnx" / "voice_index" / "magpie.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    speakers = {e["engine_options"]["speaker"] for e in index.values()}
    langs = {e["lang"] for e in index.values()}
    assert speakers == {"Aria", "Jason", "John", "Leo", "Sofia"}
    assert langs == BYTE_LANGUAGES | CHAR_LANGUAGES
    assert len(index) == len(speakers) * len(langs)


# ----------------------------------------------------------------------
# Config wiring
# ----------------------------------------------------------------------

def test_config_routes_magpie_through_the_adapters_own_tokenizer():
    from phoonnx.config import Alphabet, Engine, VoiceConfig
    cfg = VoiceConfig.from_dict({"engine": "magpie"}, engine=Engine.MAGPIE, lang_code="fr-FR")
    assert cfg.engine == Engine.MAGPIE
    # graphemes routes text -> ids through the adapter, never through a phonemizer
    assert cfg.alphabet == Alphabet.GRAPHEMES
    assert cfg.sample_rate == 22050


def test_config_detects_magpie_from_a_bare_engine_string():
    from phoonnx.config import Engine, VoiceConfig
    cfg = VoiceConfig.from_dict({"engine": "magpie"}, lang_code="ar")
    assert cfg.engine == Engine.MAGPIE
