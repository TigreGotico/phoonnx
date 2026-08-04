"""Tests for the Indic Parler-TTS adapter.

The interesting part of this engine is the encoder-decoder KV loop: the prefill graph
emits *two* caches (self-attention and cross-attention) and the decode graph consumes
both but only refreshes the self-attention half. The fake-session tests below assert
that contract directly, because getting it wrong is silent: cross-attention that is
dropped or recomputed still produces audio, just the wrong audio.
"""
import json

import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.indic_parler import (
    AUDIO_VOCAB_SIZE, AVAILABLE_LANGS, BOS_TOKEN_ID, EOS_TOKEN_ID, NUM_CODEBOOKS,
    DAC_CODEBOOK_SIZE, PAD_TOKEN_ID, IndicParlerAdapter, apply_delay_pattern_mask,
    build_delay_pattern_mask, sample_logits, strip_delay_pattern,
)

N_LAYERS = 2   # fake models are 2-layer; the real checkpoint is 24


def _req(ids=(5, 6, 7), **params):
    arr = np.array([list(ids)], np.int64)
    return AdapterSynthesisRequest(phoneme_ids=arr,
                                   phoneme_lengths=np.array([arr.shape[-1]], np.int64),
                                   params=params)


class _Inp:
    def __init__(self, name):
        self.name = name


class _Out:
    def __init__(self, name):
        self.name = name


class _FakePrefill:
    """Prefill stand-in: emits logits plus self- and cross-attention KV for N_LAYERS."""

    def __init__(self, first_token=42, src_len=4):
        self.first_token = first_token
        self.src_len = src_len
        self.last_feed = None
        self.calls = 0

    def get_outputs(self):
        names = ["logits"]
        for i in range(N_LAYERS):
            names += [f"present.{i}.decoder.key", f"present.{i}.decoder.value",
                      f"present.{i}.encoder.key", f"present.{i}.encoder.value"]
        return [_Out(n) for n in names]

    def get_inputs(self):
        return [_Inp(n) for n in ("codec_input_ids", "prompt_input_ids",
                                   "encoder_hidden_states", "encoder_attention_mask")]

    def run(self, _names, feed):
        self.calls += 1
        self.last_feed = feed
        logits = np.full((NUM_CODEBOOKS, AUDIO_VOCAB_SIZE), -10.0, np.float32)
        logits[:, self.first_token] = 10.0
        out = [logits]
        for i in range(N_LAYERS):
            out += [np.full((1, 2, 1, 4), float(i) + 0.5, np.float32),     # self K
                    np.full((1, 2, 1, 4), float(i) + 0.6, np.float32),     # self V
                    np.full((1, 2, self.src_len, 4), float(i) + 0.7, np.float32),   # cross K
                    np.full((1, 2, self.src_len, 4), float(i) + 0.8, np.float32)]   # cross V
        return out


class _FakeDecode:
    """Decode stand-in: records every feed, emits EOS after ``eos_after`` steps."""

    def __init__(self, eos_after=3, src_len=4):
        self.eos_after = eos_after
        self.src_len = src_len
        self.feeds = []

    def get_inputs(self):
        names = ["codec_input_ids", "encoder_attention_mask"]
        for i in range(N_LAYERS):
            names += [f"past_key_values.{i}.decoder.key", f"past_key_values.{i}.decoder.value",
                      f"past_key_values.{i}.encoder.key", f"past_key_values.{i}.encoder.value"]
        return [_Inp(n) for n in names]

    def get_outputs(self):
        names = ["logits"]
        for i in range(N_LAYERS):
            names += [f"present.{i}.decoder.key", f"present.{i}.decoder.value"]
        return [_Out(n) for n in names]

    def run(self, _names, feed):
        self.feeds.append({k: np.array(v, copy=True) for k, v in feed.items()})
        step = len(self.feeds)
        logits = np.full((NUM_CODEBOOKS, AUDIO_VOCAB_SIZE), -10.0, np.float32)
        target = EOS_TOKEN_ID if step >= self.eos_after else 7
        logits[:, target] = 10.0
        out = [logits]
        past = step + 1
        for i in range(N_LAYERS):
            out += [np.full((1, 2, past, 4), float(i) + 1.5, np.float32),
                    np.full((1, 2, past, 4), float(i) + 1.6, np.float32)]
        return out


class _FakeEncoder:
    def __init__(self, src_len=4):
        self.src_len = src_len
        self.last_feed = None

    def run(self, _names, feed):
        self.last_feed = feed
        n = feed["input_ids"].shape[-1]
        return [np.full((1, n, 8), 0.25, np.float32)]


class _FakeDac:
    def __init__(self):
        self.last_codes = None

    def run(self, _names, feed):
        self.last_codes = feed["audio_codes"]
        frames = feed["audio_codes"].shape[-1]
        return [np.linspace(-0.5, 0.5, max(frames, 1) * 8, dtype=np.float32).reshape(1, 1, -1)]


class _FakeTokenizer:
    """Stands in for a ``tokenizers.Tokenizer``: one id per character."""

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def __init__(self, offset=0):
        self.offset = offset
        self.seen = []

    def encode(self, text):
        self.seen.append(text)
        return self._Enc([ord(c) % 100 + self.offset for c in text])


def _wired(eos_after=3, src_len=4, description="Rohit's voice is clear."):
    ad = IndicParlerAdapter()
    ad.text_encoder = _FakeEncoder(src_len)
    ad.decode_session = _FakeDecode(eos_after, src_len)
    ad._decode_inputs = {i.name for i in ad.decode_session.get_inputs()}
    ad._decode_outputs = [o.name for o in ad.decode_session.get_outputs()]
    ad.dac_decoder = _FakeDac()
    ad.prompt_tokenizer = _FakeTokenizer(offset=0)
    ad.description_tokenizer = _FakeTokenizer(offset=1000)
    ad.description = description
    return ad


# ---------------------------------------------------------------------------
# registration / detection
# ---------------------------------------------------------------------------

def test_indic_parler_registered():
    from phoonnx.engines import list_engines
    assert "indic_parler" in list_engines()


def test_registered_at_priority_18():
    from phoonnx.engines import _PRIORITIES
    assert _PRIORITIES["indic_parler"] == 18


def test_detect_by_config():
    assert IndicParlerAdapter.detect({"engine": "indic_parler"})
    assert not IndicParlerAdapter.detect({"engine": "chatterbox"})
    assert not IndicParlerAdapter.detect(None)


def test_detect_by_session_signature():
    class _Sess:
        def get_inputs(self):
            return [_Inp(n) for n in ("codec_input_ids", "prompt_input_ids",
                                       "encoder_hidden_states", "encoder_attention_mask")]

    assert IndicParlerAdapter.detect(session=_Sess())


def test_detect_rejects_decoder_only_codec_lm():
    """A decoder-only codec LM has codec ids but no separate prompt/encoder stream."""
    class _Sess:
        def get_inputs(self):
            return [_Inp(n) for n in ("codec_input_ids", "attention_mask")]

    assert not IndicParlerAdapter.detect(session=_Sess())


def test_engine_enum_value():
    from phoonnx.config import Engine
    assert Engine("indic_parler") is Engine.INDIC_PARLER


def test_config_branch_uses_graphemes_and_44k():
    from phoonnx.config import Alphabet, Engine, VoiceConfig
    cfg = VoiceConfig.from_dict({"engine": "indic_parler"}, lang_code="hi")
    assert cfg.engine is Engine.INDIC_PARLER
    assert cfg.alphabet is Alphabet.GRAPHEMES
    assert cfg.sample_rate == 44100


# ---------------------------------------------------------------------------
# delay pattern
# ---------------------------------------------------------------------------

def test_delay_pattern_too_short_to_staircase_predicts_everything():
    """Below 2*codebooks-1 positions the staircase cannot close, so nothing is forced."""
    start, pattern = build_delay_pattern_mask(1, 5)
    assert (pattern == -1).all()
    assert start.shape == (NUM_CODEBOOKS, 1)


def test_delay_pattern_shape_and_corners():
    start, pattern = build_delay_pattern_mask(1, 20)
    assert pattern.shape == (NUM_CODEBOOKS, 20)
    # codebook k is forced to BOS for its first k+1 positions
    for k in range(NUM_CODEBOOKS):
        assert (pattern[k, : k + 1] == BOS_TOKEN_ID).all()
        assert pattern[k, k + 1] == -1
    # the tail is PAD, staircased the other way
    assert pattern[NUM_CODEBOOKS - 1, -1] == -1
    assert pattern[0, -1] == PAD_TOKEN_ID
    # only the first column is determined before any prediction
    assert start.shape == (NUM_CODEBOOKS, 1)


def test_apply_delay_pattern_overrides_only_forced_cells():
    _, pattern = build_delay_pattern_mask(1, 20)
    ids = np.full((NUM_CODEBOOKS, 5), 300, np.int64)
    out = apply_delay_pattern_mask(ids, pattern)
    assert (out[0, 1:] == 300).all()          # codebook 0 free from position 1
    assert (out[4, :5] == BOS_TOKEN_ID).all()  # codebook 4 still inside its BOS ramp
    assert out.shape == ids.shape


def test_strip_delay_pattern_recovers_square_codes():
    _, pattern = build_delay_pattern_mask(1, 30)
    raw = np.arange(NUM_CODEBOOKS * 30, dtype=np.int64).reshape(NUM_CODEBOOKS, 30) % 900 + 10
    codes = strip_delay_pattern(raw, pattern)
    assert codes.shape[0] == NUM_CODEBOOKS
    # every codebook contributes the same number of real frames
    assert codes.shape[1] == 30 - NUM_CODEBOOKS
    assert not (codes == BOS_TOKEN_ID).any()


def test_strip_cuts_the_frame_carrying_end_of_speech():
    """A natural stop leaves each codebook's EOS one column before the PAD staircase, so
    the staircase does not mask it. DAC only accepts 0..1023, so that frame must go."""
    _, pattern = build_delay_pattern_mask(1, 40)
    raw = np.full((NUM_CODEBOOKS, 40), 500, np.int64)
    for k in range(NUM_CODEBOOKS):
        raw[k, 30 + k] = EOS_TOKEN_ID
    codes = strip_delay_pattern(raw, pattern)
    assert codes.max() < DAC_CODEBOOK_SIZE
    assert codes.shape[1] == 30 - 1          # one frame shorter than the untruncated strip


def test_strip_leaves_a_ceiling_stop_untouched():
    """No out-of-range column means the run hit the frame ceiling; nothing is cut."""
    _, pattern = build_delay_pattern_mask(1, 40)
    raw = np.full((NUM_CODEBOOKS, 40), 500, np.int64)
    codes = strip_delay_pattern(raw, pattern)
    assert codes.shape[1] == 40 - NUM_CODEBOOKS
    assert codes.max() < DAC_CODEBOOK_SIZE


def test_eos_delay_constraint_walks_one_codebook_at_a_time():
    ad = IndicParlerAdapter()
    logits = np.zeros((NUM_CODEBOOKS, AUDIO_VOCAB_SIZE), np.float32)
    raw = np.full((NUM_CODEBOOKS, 3), 5, np.int64)
    first = ad._eos_delay_constraint(logits, raw, 0)
    assert first == 0
    assert logits[0, EOS_TOKEN_ID] == 0.0                      # codebook 0 may still stop
    assert np.isneginf(logits[1:, EOS_TOKEN_ID]).all()          # the rest may not

    raw[0, -1] = EOS_TOKEN_ID
    logits2 = np.zeros((NUM_CODEBOOKS, AUDIO_VOCAB_SIZE), np.float32)
    second = ad._eos_delay_constraint(logits2, raw, 0)
    assert second == 1
    assert not np.isneginf(logits2[1, EOS_TOKEN_ID])
    assert np.isneginf(logits2[2:, EOS_TOKEN_ID]).all()


def test_eos_constraint_stops_at_last_codebook():
    ad = IndicParlerAdapter()
    raw = np.full((NUM_CODEBOOKS, 2), EOS_TOKEN_ID, np.int64)
    logits = np.zeros((NUM_CODEBOOKS, AUDIO_VOCAB_SIZE), np.float32)
    assert ad._eos_delay_constraint(logits, raw, NUM_CODEBOOKS - 1) == NUM_CODEBOOKS - 1


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def test_sample_logits_respects_top_k():
    logits = np.full((NUM_CODEBOOKS, AUDIO_VOCAB_SIZE), -50.0)
    logits[:, 11] = 1.0
    logits[:, 12] = 0.9
    rng = np.random.default_rng(0)
    draws = {int(x) for _ in range(20) for x in sample_logits(logits, 1.0, 2, rng)}
    assert draws <= {11, 12}


def test_sample_logits_temperature_zero_is_argmax_like():
    logits = np.zeros((NUM_CODEBOOKS, AUDIO_VOCAB_SIZE))
    logits[:, 33] = 5.0
    rng = np.random.default_rng(1)
    assert (sample_logits(logits, 1e-6, 0, rng) == 33).all()


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------

def test_encode_text_splits_sentences_and_uses_prompt_tokenizer():
    ad = _wired()
    out = ad.encode_text("Hello there. How are you?", voice=None, syn_config=None)
    assert len(out) == 2
    assert all(isinstance(x, list) and x for x in out)
    assert ad.prompt_tokenizer.seen == ["Hello there.", "How are you?"]


def test_encode_text_without_tokenizer_raises():
    ad = IndicParlerAdapter()
    with pytest.raises(RuntimeError, match="prompt_tokenizer_path"):
        ad.encode_text("hi", voice=None, syn_config=None)


def test_description_uses_the_other_tokenizer():
    ad = _wired()
    ad.encode_description("Rohit speaks clearly.")
    ids = ad.text_encoder.last_feed["input_ids"]
    assert ids.min() >= 1000            # description tokenizer offset, not the prompt one
    assert "Rohit speaks clearly." in ad.description_tokenizer.seen
    assert "Rohit speaks clearly." not in ad.prompt_tokenizer.seen


def test_encode_description_builds_matching_mask():
    ad = _wired()
    hidden, mask = ad.encode_description("A clear voice.")
    assert hidden.shape[1] == mask.shape[1]
    assert (mask == 1).all()


# ---------------------------------------------------------------------------
# the encoder -> decoder cross-attention KV loop
# ---------------------------------------------------------------------------

def test_prefill_receives_prompt_and_encoder_states():
    ad = _wired()
    res = ad.synthesize(_req(), ad_session := _FakePrefill())
    feed = ad_session.last_feed
    assert set(feed) == {"codec_input_ids", "prompt_input_ids",
                         "encoder_hidden_states", "encoder_attention_mask"}
    assert feed["codec_input_ids"].shape[:2] == (1, NUM_CODEBOOKS)
    assert (feed["prompt_input_ids"] == np.array([[5, 6, 7]])).all()
    assert res.audio.dtype == np.float32


def test_decode_reuses_cross_attention_kv_unchanged():
    """Cross-attention KV comes from the prefill graph and must be fed back byte-identical
    on every step -- the decode graph never re-emits it."""
    ad = _wired(eos_after=4)
    prefill = _FakePrefill()
    ad.synthesize(_req(), prefill)
    feeds = ad.decode_session.feeds
    assert len(feeds) >= 3
    prefill_cross_k = np.full((1, 2, 4, 4), 0.7, np.float32)
    for f in feeds:
        assert np.array_equal(f["past_key_values.0.encoder.key"], prefill_cross_k)
        assert np.array_equal(f["past_key_values.0.encoder.key"],
                              feeds[0]["past_key_values.0.encoder.key"])


def test_decode_self_attention_kv_grows_every_step():
    ad = _wired(eos_after=5)
    ad.synthesize(_req(), _FakePrefill())
    lens = [f["past_key_values.0.decoder.key"].shape[2] for f in ad.decode_session.feeds]
    assert lens == sorted(lens)
    assert lens[0] == 1                     # the prefill's single position
    assert lens[-1] == lens[0] + len(lens) - 1


def test_decode_feeds_exactly_one_frame_per_step():
    ad = _wired(eos_after=4)
    ad.synthesize(_req(), _FakePrefill())
    for f in ad.decode_session.feeds:
        assert f["codec_input_ids"].shape == (1, NUM_CODEBOOKS, 1)


def test_encoder_attention_mask_is_forwarded_to_the_decode_graph():
    ad = _wired(eos_after=3, description="Mary speaks slowly and clearly today.")
    ad.synthesize(_req(), _FakePrefill())
    src = ad.text_encoder.last_feed["input_ids"].shape[-1]
    for f in ad.decode_session.feeds:
        assert f["encoder_attention_mask"].shape == (1, src)


def test_generation_stops_on_eos_and_reaches_the_dac():
    ad = _wired(eos_after=14)
    res = ad.synthesize(_req(), _FakePrefill())
    # EOS may only move one codebook per step, so once the model first wants to stop
    # (step 14) the staircase still needs one step per remaining codebook to unwind.
    assert len(ad.decode_session.feeds) == 14 + NUM_CODEBOOKS - 1
    assert ad.dac_decoder.last_codes.shape[1] == NUM_CODEBOOKS
    assert ad.dac_decoder.last_codes.max() < DAC_CODEBOOK_SIZE
    assert res.extras["codes"].shape[0] == NUM_CODEBOOKS
    assert res.extras["codes"].shape[1] > 0
    assert res.audio.ndim == 1


def test_an_immediate_stop_yields_no_frames_and_no_dac_call():
    """Nothing but end-of-speech means there is no complete frame to decode."""
    ad = _wired(eos_after=1)
    res = ad.synthesize(_req(), _FakePrefill())
    assert res.extras["codes"].shape[1] == 0
    assert ad.dac_decoder.last_codes is None
    assert res.audio.shape == (0,)


def test_max_new_tokens_bounds_the_loop():
    ad = _wired(eos_after=10_000)
    ad.synthesize(_req(max_new_tokens=6), _FakePrefill())
    assert len(ad.decode_session.feeds) == 5          # 6 draws, last one is not fed back


def test_greedy_is_deterministic():
    a = _wired(eos_after=6)
    b = _wired(eos_after=6)
    ra = a.synthesize(_req(do_sample=0), _FakePrefill())
    rb = b.synthesize(_req(do_sample=0), _FakePrefill())
    assert np.array_equal(ra.extras["codes"], rb.extras["codes"])


def test_seeded_sampling_is_reproducible():
    a = _wired(eos_after=6)
    b = _wired(eos_after=6)
    ra = a.synthesize(_req(do_sample=1, seed=7), _FakePrefill())
    rb = b.synthesize(_req(do_sample=1, seed=7), _FakePrefill())
    assert np.array_equal(ra.extras["codes"], rb.extras["codes"])


def test_missing_description_is_an_error():
    ad = _wired(description="")
    with pytest.raises(RuntimeError, match="description"):
        ad.synthesize(_req(), _FakePrefill())


def test_per_request_description_overrides_the_voice_default():
    ad = _wired(description="Rohit's voice is clear.")
    ad.synthesize(_req(description="Divya speaks fast."), _FakePrefill())
    assert "Divya speaks fast." in ad.description_tokenizer.seen
    assert "Rohit's voice is clear." not in ad.description_tokenizer.seen


def test_missing_decode_graph_is_an_error():
    ad = _wired()
    ad.decode_session = None
    with pytest.raises(RuntimeError, match="decoder_decode_path"):
        ad.synthesize(_req(), _FakePrefill())


def test_missing_dac_is_an_error():
    ad = _wired()
    ad.dac_decoder = None
    with pytest.raises(RuntimeError, match="dac_decoder_path"):
        ad.synthesize(_req(), _FakePrefill())


def test_missing_text_encoder_is_an_error():
    ad = _wired()
    ad.text_encoder = None
    with pytest.raises(RuntimeError, match="text_encoder_path"):
        ad.synthesize(_req(), _FakePrefill())


def test_decode_audio_on_empty_codes():
    ad = _wired()
    assert ad.decode_audio(np.zeros((NUM_CODEBOOKS, 0), np.int64)).shape == (0,)


def test_build_feed_dict_and_parse_outputs_refuse():
    ad = IndicParlerAdapter()
    with pytest.raises(NotImplementedError):
        ad.build_feed_dict(_req(), None)
    with pytest.raises(NotImplementedError):
        ad.parse_outputs([], _req())


# ---------------------------------------------------------------------------
# configure / params
# ---------------------------------------------------------------------------

def test_default_params_and_labels():
    ad = IndicParlerAdapter(temperature=0.8, top_k=50, do_sample=True, max_new_tokens=900)
    assert ad.default_params() == {"temperature": 0.8, "top_k": 50.0,
                                   "do_sample": 1.0, "max_new_tokens": 900.0}
    assert set(ad.param_labels()) == set(ad.default_params())


def test_configure_reads_engine_options(tmp_path):
    meta = tmp_path / "config.json"
    meta.write_text(json.dumps({"sample_rate": 44100, "engine": "indic_parler"}))

    class _Cfg:
        engine_params = {"description": "Anu speaks calmly.", "temperature": 0.7,
                         "top_k": 20, "do_sample": 0, "max_new_tokens": 800,
                         "seed": 3, "model_config_path": str(meta)}

    ad = IndicParlerAdapter()
    ad.configure(_Cfg())
    assert ad.description == "Anu speaks calmly."
    assert (ad.temperature, ad.top_k, ad.do_sample, ad.max_new_tokens, ad.seed) == \
           (0.7, 20, False, 800, 3)
    assert ad.sample_rate == 44100


def test_available_langs_cover_the_indexed_voices():
    from phoonnx.model_manager import TTSModelManager
    m = TTSModelManager()
    m.merge_default_voices()
    langs = {v.lang for v in m.voices.values()
             if v.engine and v.engine.value == "indic_parler"}
    assert langs
    # every indexed voice's language is one the model card claims support for,
    # except Chhattisgarhi (hne) which the card lists as a supported *speaker* language
    assert langs - set(AVAILABLE_LANGS) == {"hne"}


# ---------------------------------------------------------------------------
# voice index
# ---------------------------------------------------------------------------

def _indexed():
    from phoonnx.model_manager import TTSModelManager
    m = TTSModelManager()
    m.merge_default_voices()
    return [v for v in m.voices.values() if v.engine and v.engine.value == "indic_parler"]


def test_index_has_every_recommended_speaker():
    voices = _indexed()
    assert len(voices) == 32
    names = {v.engine_options["speaker"] for v in voices}
    assert {"Rohit", "Divya", "Mary", "Thoma", "Jaya", "Aryan", "Amrita"} <= names


def test_every_voice_carries_a_description_and_all_four_graphs():
    for v in _indexed():
        desc = v.engine_options["description"]
        assert v.engine_options["speaker"] in desc
        assert desc.endswith(".")
        assert set(v.aux_model_urls) == {"text_encoder_path", "decoder_decode_path",
                                          "dac_decoder_path", "prompt_tokenizer_path",
                                          "description_tokenizer_path", "model_config_path"}
        assert v.model_url.endswith("decoder_prefill.onnx")


def test_voice_ids_are_unique_and_namespaced():
    ids = [v.voice_id for v in _indexed()]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("indic_parler/") for i in ids)
