import json
from pathlib import Path

import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.sparktts import (
    N_GLOBAL_TOKENS,
    REF_SEGMENT_SAMPLES,
    SAMPLE_RATE,
    SEMANTIC_CODEBOOK,
    SparkTTSAdapter,
    chunk_text,
    magnitude_spectrogram,
    reference_clip,
    sample_token,
    volume_normalize,
)

SPECIAL = {
    "<|task_tts|>": 1000,
    "<|start_content|>": 1001,
    "<|end_content|>": 1002,
    "<|start_global_token|>": 1003,
    "<|end_global_token|>": 1004,
    "<|start_semantic_token|>": 1005,
    "<|im_end|>": 1006,
    "<|bicodec_global_0|>": 2000,
    "<|bicodec_semantic_0|>": 3000,
}


def _req(**params):
    return AdapterSynthesisRequest(phoneme_ids=np.array([[7, 8, 9]], np.int64),
                                   phoneme_lengths=np.array([3], np.int64),
                                   speaker_id=0, language_id=0, params=params)


def _adapter():
    ad = SparkTTSAdapter()
    ad.special = dict(SPECIAL)
    return ad


def test_sparktts_registered():
    from phoonnx.engines import list_engines
    assert "sparktts" in list_engines()


def test_sparktts_detect():
    assert SparkTTSAdapter.detect({"engine": "sparktts"})
    assert not SparkTTSAdapter.detect({"engine": "chatterbox"})
    assert not SparkTTSAdapter.detect(None)


def test_default_params_match_upstream():
    # upstream SparkTTS.inference: temperature 0.8, top_k 50, top_p 0.95, no repetition penalty
    assert SparkTTSAdapter().default_params() == {"temperature": 0.8, "top_k": 50.0, "top_p": 0.95}


def test_engine_enum_and_config_alphabet():
    from phoonnx.config import Alphabet, Engine, VoiceConfig
    cfg = VoiceConfig.from_dict({"engine": "sparktts"}, engine=Engine.SPARKTTS, lang_code="en-US")
    # graphemes routes text->ids through the adapter, never through a phonemizer
    assert cfg.engine == Engine.SPARKTTS
    assert cfg.alphabet == Alphabet.GRAPHEMES
    assert cfg.sample_rate == 16000


# ----------------------------------------------------------------------
# sampler
# ----------------------------------------------------------------------

def test_sample_token_greedy_at_zero_temperature():
    logits = np.array([1.0, 9.0, 2.0])
    assert sample_token(logits, 0.0, 50, 0.95, np.random.default_rng(0)) == 1


def test_sample_token_top_k_one_is_argmax():
    logits = np.array([3.0, 1.0, 2.9])
    rng = np.random.default_rng(0)
    assert all(sample_token(logits, 1.0, 1, 1.0, rng) == 0 for _ in range(5))


def test_sample_token_top_p_drops_the_tail():
    # token 0 alone already covers > 0.9 of the mass, so nothing else can be drawn
    logits = np.array([12.0, 0.0, -3.0, 0.5])
    rng = np.random.default_rng(0)
    assert all(sample_token(logits, 1.0, 50, 0.9, rng) == 0 for _ in range(5))


def test_sample_token_top_k_applied_before_top_p():
    # HF order is temperature -> top_k -> top_p. With k=2 the third token is gone even
    # though a top_p-first nucleus of 0.99 would have kept it.
    logits = np.array([2.0, 1.9, 1.8])
    rng = np.random.default_rng(1)
    assert {sample_token(logits, 1.0, 2, 0.99, rng) for _ in range(60)} == {0, 1}


# ----------------------------------------------------------------------
# audio front end
# ----------------------------------------------------------------------

def test_volume_normalize_scales_a_quiet_clip_up():
    quiet = (np.sin(np.linspace(0, 100, 4000)) * 0.02).astype(np.float32)
    out = volume_normalize(quiet)
    assert np.abs(out).max() > np.abs(quiet).max()
    assert np.abs(out).max() <= 1.0


def test_volume_normalize_never_clips():
    loud = (np.sin(np.linspace(0, 100, 4000)) * 4.0).astype(np.float32)
    assert np.abs(volume_normalize(loud)).max() <= 1.0


def test_reference_clip_repeats_short_and_truncates_long():
    short = np.arange(1000, dtype=np.float32)
    assert reference_clip(short).shape == (REF_SEGMENT_SAMPLES,)
    long = np.zeros(REF_SEGMENT_SAMPLES * 2, np.float32)
    assert reference_clip(long).shape == (REF_SEGMENT_SAMPLES,)


def test_reference_clip_rejects_empty():
    with pytest.raises(ValueError):
        reference_clip(np.zeros(0, np.float32))


def test_magnitude_spectrogram_layout():
    wav = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    spec = magnitude_spectrogram(wav)
    # centred STFT: 1 + n // hop frames, 1 + n_fft // 2 bins, real and non-negative
    assert spec.shape == (1, 513, 1 + 16000 // 320)
    assert spec.dtype == np.float32
    assert (spec >= 0).all()


# ----------------------------------------------------------------------
# text
# ----------------------------------------------------------------------

def test_chunk_text_packs_sentences_under_the_cap():
    text = " ".join(["This is a sentence of a reasonable length."] * 20)
    chunks = chunk_text(text, max_len=100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 or " " not in c for c in chunks)


def test_chunk_text_keeps_a_single_short_sentence_whole():
    assert chunk_text("Hello there.") == ["Hello there."]


def test_encode_text_uses_the_models_own_bpe():
    ad = _adapter()

    class _Bpe:
        def tokenize(self, text):
            return [ord(c) for c in text]

    ad.tokenizer = _Bpe()
    assert ad.encode_text("hi!", None, None) == [[ord("h"), ord("i"), ord("!")]]


def test_encode_text_tokenizes_the_reference_transcription_with_the_same_bpe():
    # the transcription is text for this model, so it must not come from the shared
    # phonemizer path (whose prompt_tokens are phoneme ids for ZipVoice-style engines)
    ad = _adapter()

    class _Bpe:
        def tokenize(self, text):
            return [ord(c) for c in text]

    class _Syn:
        speaker_reference_text = "ab"

    ad.tokenizer = _Bpe()
    ad.encode_text("hi", None, _Syn())
    assert ad._reference_text_ids == [ord("a"), ord("b")]
    # a later call without a transcription clears it, so a stale reference never leaks
    ad.encode_text("hi", None, None)
    assert ad._reference_text_ids is None


def test_encode_text_without_a_tokenizer_is_an_error():
    with pytest.raises(RuntimeError):
        SparkTTSAdapter().encode_text("hi", None, None)


# ----------------------------------------------------------------------
# prompt
# ----------------------------------------------------------------------

def test_build_prompt_bare_layout():
    ad = _adapter()
    globals_ = np.arange(N_GLOBAL_TOKENS, dtype=np.int64)
    prompt = ad.build_prompt([7, 8], globals_)[0].tolist()
    assert prompt[:2] == [1000, 1001]
    assert prompt[2:4] == [7, 8]
    assert prompt[4:6] == [1002, 1003]
    assert prompt[6:6 + N_GLOBAL_TOKENS] == [2000 + i for i in range(N_GLOBAL_TOKENS)]
    assert prompt[-1] == 1004
    assert 1005 not in prompt          # no semantic section without a transcription


def test_build_prompt_in_context_layout():
    ad = _adapter()
    globals_ = np.zeros(N_GLOBAL_TOKENS, np.int64)
    prompt = ad.build_prompt([7], globals_, prompt_text_ids=[41, 42],
                             prompt_semantic=np.array([5, 6]))[0].tolist()
    # the reference transcription comes first, then the text to speak
    assert prompt[2:5] == [41, 42, 7]
    assert prompt[-3:] == [1005, 3005, 3006]


def test_build_prompt_ignores_semantics_without_a_transcription():
    ad = _adapter()
    prompt = ad.build_prompt([7], np.zeros(N_GLOBAL_TOKENS, np.int64),
                             prompt_semantic=np.array([5]))[0].tolist()
    assert 1005 not in prompt


# ----------------------------------------------------------------------
# speaker resolution / guards
# ----------------------------------------------------------------------

def test_speaker_tokens_asset_must_hold_32_tokens(tmp_path):
    good = tmp_path / "v.json"
    good.write_text(json.dumps({"global_tokens": list(range(N_GLOBAL_TOKENS))}))
    assert SparkTTSAdapter._load_speaker_tokens(str(good)).tolist() == list(range(N_GLOBAL_TOKENS))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"global_tokens": [1, 2, 3]}))
    with pytest.raises(ValueError):
        SparkTTSAdapter._load_speaker_tokens(str(bad))


def test_speaker_tokens_asset_accepts_a_bare_list(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps(list(range(N_GLOBAL_TOKENS))))
    assert SparkTTSAdapter._load_speaker_tokens(str(path)).size == N_GLOBAL_TOKENS


def test_speaker_tokens_asset_rejects_out_of_range_codes(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"global_tokens": [9999] * N_GLOBAL_TOKENS}))
    with pytest.raises(ValueError):
        SparkTTSAdapter._load_speaker_tokens(str(path))


def test_reference_encoding_is_reused_across_chunks():
    ad = _adapter()
    calls = []

    class _Sess:
        def run(self, _, feed):
            calls.append(1)
            return [np.zeros((1, 1, N_GLOBAL_TOKENS), np.int64)]

    ad.speaker_tokenizer = _Sess()
    audio = np.sin(np.linspace(0, 200, 16000)).astype(np.float32)
    req = _req(reference_audio=(audio, 16000))
    ad._resolve_speaker(req)
    ad._resolve_speaker(req)
    assert len(calls) == 1


def test_synthesize_requires_the_vocoder():
    with pytest.raises(RuntimeError):
        SparkTTSAdapter().synthesize(_req(), None)


def test_synthesize_requires_a_speaker():
    ad = _adapter()
    ad.vocoder = object()
    ad.tokenizer = object()
    with pytest.raises(RuntimeError):      # neither preset nor reference clip
        ad.synthesize(_req(), None)


def test_cloning_without_the_speaker_graph_is_an_error():
    ad = _adapter()
    with pytest.raises(RuntimeError):
        ad.tokenize_reference(np.zeros(16000, np.float32), 16000)


def test_in_context_cloning_needs_the_semantic_graphs():
    ad = _adapter()

    class _Sess:
        def run(self, _, feed):
            return [np.zeros((1, 1, N_GLOBAL_TOKENS), np.int64)]

    ad.speaker_tokenizer = _Sess()
    audio = np.sin(np.linspace(0, 200, 16000)).astype(np.float32)
    # the speaker stream alone works without the wav2vec2 front end
    globals_, semantic = ad.tokenize_reference(audio, 16000)
    assert globals_.size == N_GLOBAL_TOKENS and semantic is None
    with pytest.raises(RuntimeError):
        ad.tokenize_reference(audio, 16000, with_semantic=True)


def test_adapter_is_not_a_single_graph_engine():
    ad = SparkTTSAdapter()
    with pytest.raises(NotImplementedError):
        ad.build_feed_dict(_req(), None)
    with pytest.raises(NotImplementedError):
        ad.parse_outputs([], _req())


# ----------------------------------------------------------------------
# voice index
# ----------------------------------------------------------------------

def test_voice_index_entries():
    import phoonnx
    from phoonnx.model_manager import TTSModelInfo

    index = json.loads((Path(phoonnx.__file__).parent / "voice_index" /
                        "sparktts.json").read_text())
    assert set(index) == {"sparktts/female/en", "sparktts/male/en",
                          "sparktts/female/zh", "sparktts/male/zh"}
    for vid, entry in index.items():
        info = TTSModelInfo(**entry)
        assert info.engine == "sparktts"
        assert info.alphabet == "graphemes"
        aux = entry["aux_model_urls"]
        # every graph the adapter loads in configure() must be published
        assert set(aux) == {"vocoder_path", "wav2vec2_path", "speaker_tokenizer_path",
                            "semantic_tokenizer_path", "bpe_tokenizer_path",
                            "speaker_tokens_path"}
        for url in list(aux.values()) + [entry["model_url"]]:
            assert url.startswith("https://huggingface.co/OpenVoiceOS/phoonnx-spark-tts/")
        # each voice must point at its own speaker asset, not a shared one
        gender, short = vid.split("/")[1:]
        assert aux["speaker_tokens_path"].endswith(f"/voices/{gender}_{short}.json")


def test_voice_index_is_merged_by_the_manager():
    from phoonnx.model_manager import TTSModelManager
    assert any(p.name == "sparktts.json" for p in TTSModelManager.voice_index_files())


# ----------------------------------------------------------------------
# _generate() — the KV-cached autoregressive loop
# ----------------------------------------------------------------------

class _IOSpec:
    """Mimics onnxruntime.NodeArg — just the ``.name``/``.shape`` the adapter reads."""

    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


NUM_KV_HEADS, HEAD_DIM, NUM_LAYERS = 2, 4, 3
VOCAB = 12000


class _FakeLMSession:
    """Fake Qwen2 KV-cached LM session with the real InferenceSession contract.

    Implements ``get_inputs()``/``get_outputs()``/``run()`` with real numpy shapes so it
    drives the adapter's own prefill/decode loop, the same way the neutts/pockettts fakes
    drive their engines. ``tokens_to_emit`` is the fixed script of ids the "model" emits,
    one per decode step; the fake never looks at ``input_ids`` to choose a token, but it
    does assert the feed shapes it is handed are consistent with the KV-cache contract.
    """

    def __init__(self, tokens_to_emit, vocab=VOCAB):
        self.tokens_to_emit = list(tokens_to_emit)
        self.vocab = vocab
        self.step = 0
        self.calls = []  # recorded feed dicts, in call order

    def get_inputs(self):
        ins = [_IOSpec("input_ids", ["batch", "seq"]),
              _IOSpec("attention_mask", ["batch", "seq"]),
              _IOSpec("position_ids", ["batch", "seq"])]
        for i in range(NUM_LAYERS):
            ins.append(_IOSpec(f"past_key_values.{i}.key",
                               ["batch", NUM_KV_HEADS, "past", HEAD_DIM]))
        return ins

    def get_outputs(self):
        outs = [_IOSpec("logits", ["batch", "seq", self.vocab])]
        for i in range(NUM_LAYERS):
            outs.append(_IOSpec(f"present.{i}.key",
                                ["batch", NUM_KV_HEADS, "total", HEAD_DIM]))
        return outs

    def run(self, output_names, feed):
        self.calls.append({k: (v.copy() if isinstance(v, np.ndarray) else v)
                           for k, v in feed.items()})
        input_ids = feed["input_ids"]
        seq = input_ids.shape[1]
        past_len = feed[f"past_key_values.0.key"].shape[2]

        # attention_mask / position_ids must match the running sequence length exactly
        assert feed["attention_mask"].shape == (1, past_len + seq)
        assert (feed["attention_mask"] == 1).all()
        if past_len == 0:
            assert feed["position_ids"].tolist() == [list(range(seq))]
        else:
            assert feed["position_ids"].tolist() == [[past_len]]
            assert seq == 1  # decode steps feed exactly one new token

        token = self.tokens_to_emit[min(self.step, len(self.tokens_to_emit) - 1)]
        self.step += 1

        logits = np.full((1, seq, self.vocab), -1e9, np.float32)
        logits[0, -1, token] = 10.0  # sharp peak: argmax/greedy always resolves to `token`

        new_past_len = past_len + seq
        present = np.zeros((1, NUM_KV_HEADS, new_past_len, HEAD_DIM), np.float32)
        # stamp a per-step, per-layer signature so growth/threading can be asserted
        present[..., :] = new_past_len

        outputs = [logits]
        for i in range(NUM_LAYERS):
            outputs.append(present.copy())
        return outputs


def _fake_lm_adapter():
    ad = _adapter()
    ad.past_names = [f"past_key_values.{i}.key" for i in range(NUM_LAYERS)]
    ad.num_kv_heads = NUM_KV_HEADS
    ad.head_dim = HEAD_DIM
    return ad


def test_generate_prefill_feed_has_zero_length_past():
    ad = _fake_lm_adapter()
    eos = SPECIAL["<|im_end|>"]
    session = _FakeLMSession([100, 101, eos])
    prompt = np.array([[1, 2, 3, 4]], np.int64)
    ad._generate(session, prompt, temperature=0.0, top_k=50, top_p=0.95,
                rng=np.random.default_rng(0))

    first_feed = session.calls[0]
    assert first_feed["input_ids"].shape == (1, 4)          # whole prompt, one shot
    for name in ad.past_names:
        assert first_feed[name].shape == (1, NUM_KV_HEADS, 0, HEAD_DIM)  # empty KV cache


def test_generate_maps_present_to_past_key_values_and_grows_shapes_each_step():
    ad = _fake_lm_adapter()
    eos = SPECIAL["<|im_end|>"]
    session = _FakeLMSession([100, 101, 102, eos])
    prompt = np.array([[1, 2, 3]], np.int64)
    ad._generate(session, prompt, temperature=0.0, top_k=50, top_p=0.95,
                rng=np.random.default_rng(0))

    # call 0: prefill (seq=3, past=0); calls 1..3: decode steps (seq=1, past growing)
    assert len(session.calls) == 4
    for i in range(NUM_LAYERS):
        name = f"past_key_values.{i}.key"
        # step 1 decode feed must carry step-0's `present.N` renamed to `past_key_values.N`
        # (the value the fake stamped == the running length after the prefill step, i.e. 3)
        assert session.calls[1][name].shape == (1, NUM_KV_HEADS, 3, HEAD_DIM)
        assert (session.calls[1][name] == 3).all()
        # each further decode step grows the cache by exactly one position
        assert session.calls[2][name].shape == (1, NUM_KV_HEADS, 4, HEAD_DIM)
        assert session.calls[3][name].shape == (1, NUM_KV_HEADS, 5, HEAD_DIM)


def test_generate_decode_feed_shapes_and_position_ids_advance():
    ad = _fake_lm_adapter()
    eos = SPECIAL["<|im_end|>"]
    session = _FakeLMSession([100, 101, eos])
    prompt = np.array([[1, 2, 3, 4, 5]], np.int64)  # prompt_len = 5
    ad._generate(session, prompt, temperature=0.0, top_k=50, top_p=0.95,
                rng=np.random.default_rng(0))

    decode_feeds = session.calls[1:]
    for step, feed in enumerate(decode_feeds):
        assert feed["input_ids"].shape == (1, 1)                 # decode feeds one token
        assert feed["position_ids"].tolist() == [[5 + step]]     # prompt_len + step
        assert feed["attention_mask"].shape == (1, 5 + step + 1)


def test_generate_stops_on_eos_and_does_not_emit_it():
    ad = _fake_lm_adapter()
    eos = SPECIAL["<|im_end|>"]
    session = _FakeLMSession([111, 222, eos, 333])  # 333 must never be reached
    prompt = np.array([[1, 2]], np.int64)
    emitted = ad._generate(session, prompt, temperature=0.0, top_k=50, top_p=0.95,
                           rng=np.random.default_rng(0))

    assert emitted == [111, 222]
    assert len(session.calls) == 3  # prefill + 2 decode steps, stops before a 3rd decode


def test_generate_respects_max_new_tokens_when_eos_never_comes():
    from phoonnx.engines import sparktts as sparktts_mod
    ad = _fake_lm_adapter()
    session = _FakeLMSession([777])  # repeats forever, never emits eos
    prompt = np.array([[1]], np.int64)
    old_cap = sparktts_mod.MAX_NEW_TOKENS
    sparktts_mod.MAX_NEW_TOKENS = 5
    try:
        emitted = ad._generate(session, prompt, temperature=0.0, top_k=50, top_p=0.95,
                               rng=np.random.default_rng(0))
    finally:
        sparktts_mod.MAX_NEW_TOKENS = old_cap
    assert emitted == [777] * 5


def test_generate_reads_kv_shape_lazily_from_the_session_when_unset():
    ad = _adapter()  # no past_names / num_kv_heads / head_dim pre-seeded
    eos = SPECIAL["<|im_end|>"]
    session = _FakeLMSession([100, eos])
    prompt = np.array([[1, 2]], np.int64)
    ad._generate(session, prompt, temperature=0.0, top_k=50, top_p=0.95,
                rng=np.random.default_rng(0))
    assert ad.past_names == [f"past_key_values.{i}.key" for i in range(NUM_LAYERS)]
    assert ad.num_kv_heads == NUM_KV_HEADS
    assert ad.head_dim == HEAD_DIM


def test_generate_temperature_zero_is_deterministic_argmax_through_the_fake_session():
    # two independent rng streams must produce the identical script, because temperature<=0
    # takes the argmax branch in sample_token() and never touches the rng
    eos = SPECIAL["<|im_end|>"]
    tokens = [50, 3999, 0, eos]  # exercise low/high/edge vocab ids too
    emitted_a = _fake_lm_adapter()._generate(
        _FakeLMSession(tokens), np.array([[1, 2]], np.int64),
        temperature=0.0, top_k=50, top_p=0.95, rng=np.random.default_rng(1))
    emitted_b = _fake_lm_adapter()._generate(
        _FakeLMSession(tokens), np.array([[1, 2]], np.int64),
        temperature=-1.0, top_k=1, top_p=0.01, rng=np.random.default_rng(999))
    assert emitted_a == emitted_b == [50, 3999, 0]


# ----------------------------------------------------------------------
# synthesize() — semantic-token -> vocoder handoff
# ----------------------------------------------------------------------

class _FakeVocoderSession:
    """Records the exact feed dict it receives and returns a deterministic waveform."""

    def __init__(self, n_samples=1600):
        self.n_samples = n_samples
        self.calls = []

    def run(self, output_names, feed):
        self.calls.append(feed)
        return [np.linspace(-0.5, 0.5, self.n_samples, dtype=np.float32)[None]]


def _synth_ready_adapter(lm_tokens, vocoder=None):
    ad = _fake_lm_adapter()
    ad.vocoder = vocoder or _FakeVocoderSession()
    ad.tokenizer = object()
    ad.preset_global_tokens = np.arange(N_GLOBAL_TOKENS, dtype=np.int64)
    return ad, _FakeLMSession(lm_tokens)


def test_synthesize_offsets_semantic_tokens_before_the_vocoder_call():
    base = SPECIAL["<|bicodec_semantic_0|>"]
    eos = SPECIAL["<|im_end|>"]
    # raw emitted ids are base-offset; the vocoder must see them de-offset back to [0, 8192)
    raw = [base + 5, base + 0, base + 8191, eos]
    ad, session = _synth_ready_adapter(raw)
    result = ad.synthesize(_req(), session)

    vocoder_calls = ad.vocoder.calls
    assert len(vocoder_calls) == 1
    feed = vocoder_calls[0]
    assert set(feed) == {"semantic_tokens", "global_tokens"}
    assert feed["semantic_tokens"].dtype == np.int64
    assert feed["semantic_tokens"].tolist() == [[5, 0, 8191]]
    assert feed["global_tokens"].dtype == np.int64
    assert feed["global_tokens"].shape == (1, 1, N_GLOBAL_TOKENS)
    assert feed["global_tokens"].reshape(-1).tolist() == list(range(N_GLOBAL_TOKENS))
    assert result.extras == {"semantic_token_count": 3}


def test_synthesize_waveform_round_trips_into_an_audio_chunk():
    from phoonnx.voice import AudioChunk

    base = SPECIAL["<|bicodec_semantic_0|>"]
    eos = SPECIAL["<|im_end|>"]
    n_samples = 800
    vocoder = _FakeVocoderSession(n_samples=n_samples)
    ad, session = _synth_ready_adapter([base + 1, base + 2, eos], vocoder=vocoder)
    result = ad.synthesize(_req(), session)

    assert result.audio.dtype == np.float32
    assert result.audio.shape == (n_samples,)
    assert result.audio.min() >= -1.0 and result.audio.max() <= 1.0

    chunk = AudioChunk(sample_rate=SAMPLE_RATE, sample_width=2, sample_channels=1,
                       audio_float_array=result.audio)
    assert chunk.audio_int16_array.dtype == np.int16
    assert chunk.audio_int16_array.shape == (n_samples,)
    assert len(chunk.audio_int16_bytes) == n_samples * 2


def test_synthesize_filters_out_non_semantic_token_ids_before_the_vocoder():
    # a malformed/garbage LM output: ids outside the semantic codebook window (control
    # tokens, global-token ids, or plain out-of-range garbage) must never reach the vocoder
    base = SPECIAL["<|bicodec_semantic_0|>"]
    eos = SPECIAL["<|im_end|>"]
    garbage_below = base - 1                       # just under the semantic window
    garbage_above = base + SEMANTIC_CODEBOOK        # just at/over the semantic window
    control_token = SPECIAL["<|start_content|>"]    # a real control id, not semantic at all
    raw = [garbage_below, base + 42, garbage_above, control_token, eos]
    ad, session = _synth_ready_adapter(raw)
    result = ad.synthesize(_req(), session)

    feed = ad.vocoder.calls[0]
    assert feed["semantic_tokens"].tolist() == [[42]]   # only the one valid token survives
    assert result.extras == {"semantic_token_count": 1}


def test_synthesize_returns_silence_when_nothing_survives_the_filter_and_skips_the_vocoder():
    base = SPECIAL["<|bicodec_semantic_0|>"]
    eos = SPECIAL["<|im_end|>"]
    ad, session = _synth_ready_adapter([SPECIAL["<|start_content|>"], eos])  # no semantic ids at all
    result = ad.synthesize(_req(), session)

    assert result.audio.shape == (0,)
    assert result.audio.dtype == np.float32
    assert ad.vocoder.calls == []  # vocoder must never be called with an empty token list
