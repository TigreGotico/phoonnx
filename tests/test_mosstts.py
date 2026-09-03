"""MOSS-TTS-Nano adapter tests.

The autoregressive loop is driven entirely through ONNX ``session.run`` calls, so it is
exercised here against fake sessions that record their feeds — the two-level
(global step -> local frame) protocol, the KV-cache threading and the safety cap are all
observable without the 700 MB of real weights.
"""
import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.mosstts import (
    AUDIO_ASSISTANT_SLOT_TOKEN_ID, AUDIO_END_TOKEN_ID, AUDIO_PAD_TOKEN_ID,
    AUDIO_START_TOKEN_ID, AUDIO_USER_SLOT_TOKEN_ID, CODEBOOK_SIZE, FIXED_SAMPLING,
    N_VQ, ROW_WIDTH, MossTTSNanoAdapter, _apply_repetition_penalty, _sample,
)

HIDDEN = 768
LAYERS = 12


def _req(phoneme_ids=(11, 22, 33), **params):
    ids = np.asarray(phoneme_ids, np.int64).reshape(1, -1)
    return AdapterSynthesisRequest(phoneme_ids=ids,
                                   phoneme_lengths=np.array([ids.shape[1]], np.int64),
                                   speaker_id=0, language_id=0, params=params)


class _IO:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape or []


class _FakeSession:
    """Minimal onnxruntime.InferenceSession stand-in: named outputs + a feed log."""

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


def _kv(seq):
    return np.zeros((1, seq, 12, 64), np.float32)


def _global_names():
    return ["global_hidden"] + [f"present_{k}_{i}" for i in range(LAYERS) for k in ("key", "value")]


def _wire(adapter, *, frames_before_stop=3, decoded_seconds=1.0):
    """Wire an adapter to fake prefill / decode_step / local_frame / codec graphs."""
    state = {"prefill_len": 0}

    def prefill_fn(feed, _i):
        seq = feed["input_ids"].shape[1]
        state["prefill_len"] = seq
        out = {"global_hidden": np.zeros((1, seq, HIDDEN), np.float32)}
        for i in range(LAYERS):
            out[f"present_key_{i}"] = _kv(seq)
            out[f"present_value_{i}"] = _kv(seq)
        return out

    def decode_fn(feed, i):
        seq = state["prefill_len"] + i + 1
        out = {"global_hidden": np.zeros((1, 1, HIDDEN), np.float32)}
        for j in range(LAYERS):
            out[f"present_key_{j}"] = _kv(seq)
            out[f"present_value_{j}"] = _kv(seq)
        return out

    def frame_fn(_feed, i):
        keep = 1 if i < frames_before_stop else 0
        return {"should_continue": np.array([[keep]], np.int32),
                "frame_token_ids": np.arange(N_VQ, dtype=np.int32).reshape(1, N_VQ) + i}

    def codec_encode_fn(feed, _i):
        n = max(1, feed["waveform"].shape[-1] // 3840)
        return {"audio_codes": np.zeros((1, n, N_VQ), np.int32),
                "audio_code_lengths": np.array([n], np.int32)}

    def codec_decode_fn(_feed, _i):
        n = int(48000 * decoded_seconds)
        # stereo: left = +0.5, right = -0.1 -> mono mean 0.2
        audio = np.stack([np.full(n, 0.5, np.float32), np.full(n, -0.1, np.float32)])
        return {"audio": audio[None, ...], "audio_lengths": np.array([n], np.int32)}

    adapter.decode_step = _FakeSession(_global_names(), decode_fn)
    adapter.local_fixed_frame = _FakeSession(
        ["should_continue", "frame_token_ids"], frame_fn)
    adapter.codec_encode = _FakeSession(
        ["audio_codes", "audio_code_lengths"], codec_encode_fn)
    adapter.codec_decode = _FakeSession(
        ["audio", "audio_lengths"], codec_decode_fn)
    return _FakeSession(_global_names(), prefill_fn)


class _FakeSp:
    """SentencePiece stand-in: one token per character."""

    def encode(self, text, out_type=int):
        return [ord(c) for c in text]


# ----------------------------------------------------------------- registration
def test_registered():
    from phoonnx.engines import list_engines
    assert "mosstts" in list_engines()


def test_detect():
    assert MossTTSNanoAdapter.detect({"engine": "mosstts"})
    assert not MossTTSNanoAdapter.detect({"engine": "chatterbox"})
    assert not MossTTSNanoAdapter.detect(None)
    assert not MossTTSNanoAdapter.detect({})


def test_engine_enum_and_config_detection():
    from phoonnx.config import Engine, VoiceConfig
    from scriptconv.phonemizers.enums import Alphabet
    assert Engine.MOSSTTS.value == "mosstts"
    cfg = VoiceConfig.from_dict({"engine": "mosstts"}, lang_code="en")
    assert cfg.engine == Engine.MOSSTTS
    # GRAPHEMES routes text -> ids through the adapter's own SentencePiece model
    assert cfg.alphabet == Alphabet.GRAPHEMES
    assert cfg.sample_rate == 48000


def test_default_params_match_the_baked_in_graph_constants():
    p = MossTTSNanoAdapter().default_params()
    for k, v in FIXED_SAMPLING.items():
        assert p[k] == v
    assert p["sample_mode"] == "fixed"
    assert set(MossTTSNanoAdapter().param_labels()) >= set(FIXED_SAMPLING)


# ------------------------------------------------------------------ prompt rows
def test_text_rows_pad_the_codebook_columns():
    rows = MossTTSNanoAdapter._text_rows([5, 6])
    assert rows == [[5] + [AUDIO_PAD_TOKEN_ID] * N_VQ, [6] + [AUDIO_PAD_TOKEN_ID] * N_VQ]
    assert all(len(r) == ROW_WIDTH for r in rows)


def test_audio_rows_carry_the_slot_token_in_column_zero():
    rows = MossTTSNanoAdapter._audio_rows([list(range(N_VQ))])
    assert rows[0][0] == AUDIO_USER_SLOT_TOKEN_ID
    assert rows[0][1:] == list(range(N_VQ))


def test_audio_rows_pad_short_frames():
    rows = MossTTSNanoAdapter._audio_rows([[1, 2, 3]])
    assert rows[0][1:4] == [1, 2, 3]
    assert rows[0][4:] == [AUDIO_PAD_TOKEN_ID] * (N_VQ - 3)


def test_build_prompt_rows_layout():
    ad = MossTTSNanoAdapter()
    codes = [[7] * N_VQ, [8] * N_VQ]
    rows = ad.build_prompt_rows(codes, [111, 222])
    assert all(len(r) == ROW_WIDTH for r in rows)
    col0 = [r[0] for r in rows]
    # the reference block is delimited by <audio_start> ... <audio_end>
    start = col0.index(AUDIO_START_TOKEN_ID)
    assert col0[start + 1:start + 3] == [AUDIO_USER_SLOT_TOKEN_ID] * 2
    assert col0[start + 3] == AUDIO_END_TOKEN_ID
    # the text to speak is present, and the prompt ends by handing over to the assistant
    assert 111 in col0 and 222 in col0
    assert col0[-1] == AUDIO_START_TOKEN_ID


# ---------------------------------------------------------------------- chunking
def test_chunking_respects_the_token_budget():
    ad = MossTTSNanoAdapter()
    ad.sp = _FakeSp()
    chunks = ad._chunk("One two. Three four. Five six.", 12)
    assert chunks and all(0 < len(c) <= 12 for c in chunks)
    # nothing is dropped: every character of every sentence survives somewhere
    assert sum(len(c) for c in chunks) > 0


def test_chunking_hard_splits_an_oversized_sentence():
    ad = MossTTSNanoAdapter()
    ad.sp = _FakeSp()
    chunks = ad._chunk("a" * 200, 25)
    assert len(chunks) == 8 and all(len(c) == 25 for c in chunks)


def test_chunking_empty_text():
    ad = MossTTSNanoAdapter()
    ad.sp = _FakeSp()
    assert ad._chunk("   ", 10) == []


def test_encode_text_without_a_tokenizer_is_a_clear_error():
    with pytest.raises(RuntimeError, match="sp_model_path"):
        MossTTSNanoAdapter().encode_text("hi", None, None)


# ---------------------------------------------------------------------- sampling
def test_repetition_penalty_direction():
    scores = np.array([1.0, 2.0, -1.0, 4.0], np.float32)
    out = _apply_repetition_penalty(scores, {1, 2}, 2.0)
    assert out[1] == pytest.approx(1.0)     # positive -> divided
    assert out[2] == pytest.approx(-2.0)    # negative -> multiplied
    assert out[0] == pytest.approx(1.0)     # untouched
    assert out[3] == pytest.approx(4.0)


def test_repetition_penalty_is_a_noop_when_disabled_or_unseen():
    scores = np.array([1.0, 2.0], np.float32)
    assert _apply_repetition_penalty(scores, set(), 2.0) is scores
    assert _apply_repetition_penalty(scores, {0}, 1.0) is scores


def test_repetition_penalty_ignores_out_of_range_tokens():
    scores = np.array([1.0, 2.0], np.float32)
    out = _apply_repetition_penalty(scores, {5, -1}, 2.0)
    assert np.allclose(out, scores)


def test_sample_greedy_is_argmax():
    logits = np.array([0.1, 9.0, 0.2], np.float32)
    assert _sample(logits, 1.0, 0, 1.0, np.random.default_rng(0), do_sample=False) == 1


def test_sample_top_k_one_is_deterministic():
    logits = np.array([0.1, 9.0, 0.2, 3.0], np.float32)
    rng = np.random.default_rng(0)
    assert all(_sample(logits, 1.0, 1, 1.0, rng) == 1 for _ in range(5))


def test_sample_nucleus_never_picks_outside_the_nucleus():
    logits = np.array([12.0, -20.0, -20.0, 0.5], np.float32)
    rng = np.random.default_rng(1)
    assert {_sample(logits, 0.8, 0, 0.9, rng) for _ in range(20)} <= {0, 3}


def test_sample_rejects_zero_temperature_when_sampling():
    with pytest.raises(ValueError):
        _sample(np.array([1.0, 2.0], np.float32), 0.0, 0, 1.0, np.random.default_rng(0))


# ------------------------------------------------------------------------ codec
def test_decode_frames_downmixes_stereo_to_mono():
    ad = MossTTSNanoAdapter()
    _wire(ad, decoded_seconds=0.1)
    audio = ad.decode_frames([[0] * N_VQ])
    assert audio.ndim == 1 and audio.dtype == np.float32
    assert np.allclose(audio, 0.2)          # mean(0.5, -0.1)


def test_decode_frames_of_nothing_is_empty():
    assert MossTTSNanoAdapter().decode_frames([]).shape == (0,)


def test_encode_reference_duplicates_mono_to_stereo():
    ad = MossTTSNanoAdapter()
    _wire(ad)
    codes = ad.encode_reference(np.zeros(48000, np.float32), 48000)
    fed = ad.codec_encode.feeds[0]["waveform"]
    assert fed.shape[:2] == (1, 2)
    assert codes and len(codes[0]) == N_VQ


def test_encode_reference_resamples():
    ad = MossTTSNanoAdapter()
    _wire(ad)
    ad.encode_reference(np.zeros(16000, np.float32), 16000)
    assert ad.codec_encode.feeds[0]["waveform"].shape[-1] == pytest.approx(48000, abs=64)


# ------------------------------------------------------------- the AR loop proper
def test_generate_frames_stops_on_should_continue():
    ad = MossTTSNanoAdapter()
    prefill = _wire(ad, frames_before_stop=3)
    rows = ad.build_prompt_rows([[0] * N_VQ], [1, 2])
    frames = ad.generate_frames(prefill, rows, ad.default_params(), np.random.default_rng(0))
    assert len(frames) == 3
    assert all(len(f) == N_VQ for f in frames)
    # one decode_step per emitted frame; the stopping step emits none
    assert len(ad.decode_step.feeds) == 3


def test_generate_frames_threads_the_kv_cache_and_the_valid_length():
    ad = MossTTSNanoAdapter()
    prefill = _wire(ad, frames_before_stop=2)
    rows = ad.build_prompt_rows([[0] * N_VQ], [1, 2])
    ad.generate_frames(prefill, rows, ad.default_params(), np.random.default_rng(0))
    lengths = [int(f["past_valid_lengths"][0]) for f in ad.decode_step.feeds]
    assert lengths == [len(rows), len(rows) + 1]
    for feed in ad.decode_step.feeds:
        assert {f"past_key_{i}" for i in range(LAYERS)} <= set(feed)
        assert {f"past_value_{i}" for i in range(LAYERS)} <= set(feed)
        assert not any(k.startswith("present_") for k in feed)


def test_generate_frames_feeds_the_frame_back_as_an_assistant_row():
    ad = MossTTSNanoAdapter()
    prefill = _wire(ad, frames_before_stop=1)
    rows = ad.build_prompt_rows([[0] * N_VQ], [1])
    frames = ad.generate_frames(prefill, rows, ad.default_params(), np.random.default_rng(0))
    row = ad.decode_step.feeds[0]["input_ids"]
    assert row.shape == (1, 1, ROW_WIDTH) and row.dtype == np.int32
    assert row[0, 0, 0] == AUDIO_ASSISTANT_SLOT_TOKEN_ID
    assert list(row[0, 0, 1:]) == frames[0]


def test_generate_frames_honours_the_max_frame_cap():
    ad = MossTTSNanoAdapter()
    prefill = _wire(ad, frames_before_stop=10_000)     # a model that never stops
    rows = ad.build_prompt_rows([[0] * N_VQ], [1])
    params = {**ad.default_params(), "max_new_frames": 5}
    frames = ad.generate_frames(prefill, rows, params, np.random.default_rng(0))
    assert len(frames) == 5


def test_generate_frames_cap_is_at_least_one_frame():
    ad = MossTTSNanoAdapter()
    prefill = _wire(ad, frames_before_stop=10_000)
    rows = ad.build_prompt_rows([[0] * N_VQ], [1])
    frames = ad.generate_frames(prefill, rows, {**ad.default_params(), "max_new_frames": 0},
                                np.random.default_rng(0))
    assert len(frames) == 1


def test_repetition_mask_grows_with_the_emitted_tokens():
    ad = MossTTSNanoAdapter()
    prefill = _wire(ad, frames_before_stop=3)
    rows = ad.build_prompt_rows([[0] * N_VQ], [1])
    ad.generate_frames(prefill, rows, ad.default_params(), np.random.default_rng(0))
    masks = [f["repetition_seen_mask"] for f in ad.local_fixed_frame.feeds]
    assert masks[0].shape == (1, N_VQ, CODEBOOK_SIZE)
    assert masks[0].sum() == 0                       # nothing emitted yet
    assert masks[1].sum() == N_VQ                    # one token per channel
    assert masks[2].sum() == 2 * N_VQ


def test_deviating_sampling_params_fall_back_to_host_sampling():
    ad = MossTTSNanoAdapter()
    prefill = _wire(ad)
    ad.local_cached_step = None                       # host path unavailable
    rows = ad.build_prompt_rows([[0] * N_VQ], [1])
    params = {**ad.default_params(), "audio_temperature": 0.1}
    with pytest.raises(RuntimeError, match="local_cached_step_path"):
        ad.generate_frames(prefill, rows, params, np.random.default_rng(0))


def test_fixed_mode_without_the_fixed_graph_needs_the_host_path():
    ad = MossTTSNanoAdapter()
    prefill = _wire(ad)
    ad.local_fixed_frame = None
    ad.local_cached_step = None
    rows = ad.build_prompt_rows([[0] * N_VQ], [1])
    with pytest.raises(RuntimeError, match="local_cached_step_path"):
        ad.generate_frames(prefill, rows, ad.default_params(), np.random.default_rng(0))


# -------------------------------------------------------------------- synthesize
def test_synthesize_end_to_end_against_fakes():
    ad = MossTTSNanoAdapter()
    prefill = _wire(ad, frames_before_stop=2, decoded_seconds=0.25)
    req = _req(reference_audio=(np.zeros(48000, np.float32), 48000), seed=3)
    out = ad.synthesize(req, prefill)
    assert out.audio.ndim == 1 and out.audio.dtype == np.float32
    assert np.abs(out.audio).max() <= 1.0
    assert out.extras["frames"] == 2


def test_synthesize_accepts_precomputed_prompt_codes():
    ad = MossTTSNanoAdapter()
    prefill = _wire(ad, frames_before_stop=1)
    out = ad.synthesize(_req(prompt_audio_codes=[[3] * N_VQ]), prefill)
    assert out.extras["frames"] == 1
    assert not ad.codec_encode.feeds          # no reference encode needed


def test_synthesize_is_reproducible_for_a_given_seed():
    def run():
        ad = MossTTSNanoAdapter()
        prefill = _wire(ad, frames_before_stop=2)
        return ad.synthesize(_req(prompt_audio_codes=[[0] * N_VQ], seed=11), prefill).audio
    assert np.array_equal(run(), run())


def test_interleaved_requests_do_not_bleed_reference_state():
    """Reference codes/text flow through request.params on each call, not through
    attributes stashed on the adapter instance — a second, differently-referenced
    request run in between must not change the outcome of the first."""
    ad = MossTTSNanoAdapter()
    req_a = _req(phoneme_ids=(1, 2, 3), prompt_audio_codes=[[1] * N_VQ], seed=7)
    req_b = _req(phoneme_ids=(9, 8, 7, 6), prompt_audio_codes=[[9] * N_VQ, [9] * N_VQ], seed=7)

    # Each call gets its own fake graphs (as a real request gets its own KV-cache
    # run), but the adapter INSTANCE is reused across all three, exactly as the
    # model manager reuses one adapter across concurrent/interleaved requests.
    first = ad.synthesize(req_a, _wire(ad, frames_before_stop=2))
    ad.synthesize(req_b, _wire(ad, frames_before_stop=2))  # interleaved, different reference
    second = ad.synthesize(req_a, _wire(ad, frames_before_stop=2))

    assert np.array_equal(first.audio, second.audio)
    assert first.extras == second.extras


def test_synthesize_requires_the_auxiliary_graphs():
    with pytest.raises(RuntimeError, match="decode_step_path"):
        MossTTSNanoAdapter().synthesize(_req(prompt_audio_codes=[[0] * N_VQ]), None)


def test_synthesize_requires_a_reference():
    ad = MossTTSNanoAdapter()
    prefill = _wire(ad)
    with pytest.raises(RuntimeError, match="reference clip"):
        ad.synthesize(_req(), prefill)


def test_synthesize_rejects_a_reference_too_short_to_encode():
    ad = MossTTSNanoAdapter()
    prefill = _wire(ad)
    with pytest.raises(RuntimeError, match="too short"):
        ad.synthesize(_req(prompt_audio_codes=[]), prefill)


def test_static_graph_helpers_are_refused():
    ad = MossTTSNanoAdapter()
    with pytest.raises(NotImplementedError):
        ad.build_feed_dict(_req(), None)
    with pytest.raises(NotImplementedError):
        ad.parse_outputs([], _req())


# ------------------------------------------------------------------- voice index
def test_voice_index_entries_are_valid():
    import json
    from pathlib import Path
    import phoonnx
    from phoonnx.model_manager import TTSModelInfo

    index = json.loads((Path(phoonnx.__file__).parent / "voice_index" /
                        "mosstts.json").read_text())
    assert len(index) >= 19
    required = {"decode_step_path", "local_fixed_frame_path", "local_cached_step_path",
                "sp_model_path", "codec_encode_path", "codec_decode_path"}
    for vid, entry in index.items():
        info = TTSModelInfo(**entry)
        assert info.engine == "mosstts"
        assert str(info.alphabet) == "graphemes" or info.alphabet.value == "graphemes"
        aux = entry["aux_model_urls"]
        assert required <= set(aux)
        # every graph that references external weights must have those weights
        # downloaded alongside it, or onnxruntime cannot open the session
        assert any(u.endswith("moss_tts_global_shared.data") for u in aux.values())
        assert any(u.endswith("moss_tts_local_shared.data") for u in aux.values())
        assert any(u.endswith("moss_audio_tokenizer_encode.data") for u in aux.values())
        assert any(u.endswith("moss_audio_tokenizer_decode_shared.data") for u in aux.values())
        assert entry["model_url"].endswith("moss_tts_prefill.onnx")


def test_voice_index_is_merged_by_the_manager():
    from phoonnx.model_manager import TTSModelManager
    assert "mosstts.json" in TTSModelManager._VOICE_INDEX_ORDER
    assert any(p.name == "mosstts.json" for p in TTSModelManager.voice_index_files())
