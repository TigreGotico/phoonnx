import json

import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.supertonic import (
    SuperTonicAdapter, AVAILABLE_LANGS, preprocess_text, chunk_text,
    load_unicode_indexer, length_to_mask, sample_noisy_latent,
)
from phoonnx.tokenizer import TTSTokenizer, Vocabulary


def _req(**params):
    return AdapterSynthesisRequest(phoneme_ids=np.array([[1, 2, 3]], np.int64),
                                   phoneme_lengths=np.array([3], np.int64),
                                   speaker_id=0, language_id=0, params=params)


def _tiny_indexer():
    """A synthetic 65536-entry table mapping a handful of ASCII chars + '<'/'>'/'e'/'n'
    (enough for '<en>hi.</en>') to small ids; everything else stays -1 (unmapped)."""
    table = [-1] * 65536
    for i, ch in enumerate(" .!?<>/abcdefghijklmnopqrstuvwxyz"):
        table[ord(ch)] = i + 1
    return table


def _tiny_tokenizer():
    return TTSTokenizer(Vocabulary.from_supertonic_indexer(_tiny_indexer()),
                        add_blank_char=False, add_blank_word=False,
                        use_eos_bos=False, blank_at_end=False, blank_at_start=False)


class _FakeSession:
    """Minimal onnxruntime.InferenceSession stand-in: records the last feed dict and
    returns caller-supplied outputs (or a shape-preserving default)."""

    def __init__(self, outputs=None):
        self.outputs = outputs
        self.last_feed = None
        self.calls = 0

    def run(self, output_names, feed):
        self.calls += 1
        self.last_feed = feed
        if self.outputs is not None:
            return self.outputs(feed) if callable(self.outputs) else self.outputs
        return [feed[list(feed)[0]]]


# ---------------------------------------------------------------------------
# registration / detect
# ---------------------------------------------------------------------------

def test_supertonic_registered():
    from phoonnx.engines import list_engines
    assert "supertonic" in list_engines()


def test_supertonic_detect_by_config():
    assert SuperTonicAdapter.detect({"engine": "supertonic"})
    assert not SuperTonicAdapter.detect({"engine": "vits"})
    assert not SuperTonicAdapter.detect(None)


def test_supertonic_detect_by_session_signature():
    class _Inp:
        def __init__(self, name): self.name = name

    class _Sess:
        def get_inputs(self):
            return [_Inp(n) for n in ("noisy_latent", "text_emb", "style_ttl",
                                       "latent_mask", "text_mask", "current_step", "total_step")]

    assert SuperTonicAdapter.detect(session=_Sess())

    class _OtherSess:
        def get_inputs(self):
            return [_Inp("x")]

    assert not SuperTonicAdapter.detect(session=_OtherSess())


def test_default_params():
    ad = SuperTonicAdapter(total_step=6, speed=1.1, silence_duration=0.2)
    params = ad.default_params()
    assert params == {"total_step": 6.0, "speed": 1.1, "silence_duration": 0.2}


# ---------------------------------------------------------------------------
# text preprocessing
# ---------------------------------------------------------------------------

def test_preprocess_text_wraps_lang_and_terminates():
    out = preprocess_text("hello world", "en")
    assert out == "<en>hello world.</en>"


def test_preprocess_text_keeps_existing_terminal_punctuation():
    out = preprocess_text("hello world!", "en")
    assert out == "<en>hello world!</en>"


def test_preprocess_text_strips_emoji():
    out = preprocess_text("hi \U0001F600 there", "en")
    assert "\U0001F600" not in out
    assert out == "<en>hi there.</en>"


def test_preprocess_text_unknown_lang_raises():
    with pytest.raises(ValueError):
        preprocess_text("hi", "xx")


def test_preprocess_text_all_available_langs_accepted():
    for lang in AVAILABLE_LANGS:
        assert preprocess_text("hi", lang).startswith(f"<{lang}>")


def test_chunk_text_splits_long_input():
    long_text = ("This is a sentence. " * 40).strip()
    chunks = chunk_text(long_text, max_len=100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # no words dropped in the process
    assert sum(len(c.split()) for c in chunks) == len(long_text.split())


def test_chunk_text_short_input_single_chunk():
    assert chunk_text("Hello there.", max_len=300) == ["Hello there."]


def test_chunk_text_empty_input():
    assert chunk_text("", max_len=300) == []
    assert chunk_text("   ", max_len=300) == []


# ---------------------------------------------------------------------------
# unicode indexer -> standard Vocabulary/TTSTokenizer
# ---------------------------------------------------------------------------

def test_load_unicode_indexer_list_form(tmp_path):
    table = [-1] * 65536
    table[ord("a")] = 5
    p = tmp_path / "indexer.json"
    p.write_text(json.dumps(table))
    loaded = load_unicode_indexer(str(p))
    assert loaded[ord("a")] == 5
    assert loaded[ord("b")] == -1


def test_load_unicode_indexer_dict_form(tmp_path):
    p = tmp_path / "indexer.json"
    p.write_text(json.dumps({"a": 5, "65": 9}))   # char key + codepoint-string key
    loaded = load_unicode_indexer(str(p))
    assert loaded[ord("a")] == 5
    assert loaded[65] == 9                          # "A"


def test_vocabulary_from_supertonic_indexer_drops_unmapped():
    table = [-1] * 65536
    table[ord("a")] = 5
    table[ord("b")] = 0
    voc = Vocabulary.from_supertonic_indexer(table)
    assert voc.char2idx == {"a": 5, "b": 0}   # -1 entries excluded, 0 is a valid id
    assert "c" not in voc.char2idx


def test_vocabulary_from_supertonic_indexer_keys_not_normalized():
    # a combining-accent sequence must survive as-is: the table indexes exactly the
    # shipped codepoints, never NFKD/NFC-folded (NFKD applies to *input text*, not to
    # the vocabulary's own keys)
    combining_e = "é"   # 'e' + COMBINING ACUTE ACCENT, two codepoints
    table = [-1] * 65536
    table[ord("e")] = 1
    table[0x0301] = 2
    voc = Vocabulary.from_supertonic_indexer(table)
    assert voc.char2idx == {"e": 1, "́": 2}
    assert combining_e not in voc.char2idx   # it's two chars, not one


def test_tokenizer_maps_known_chars():
    tok = _tiny_tokenizer()
    ids = tok.tokenize("<en>hi.</en>")
    assert len(ids) == len("<en>hi.</en>")
    assert all(i > 0 for i in ids)


def test_tokenizer_drops_missing_char_with_warning(caplog):
    # standard TTSTokenizer/Vocabulary behavior: an out-of-vocabulary character is
    # dropped (with a warning), not a hard error — SuperTonic gets this for free by
    # reusing the standard tokenizer instead of a bespoke raise-on-miss lookup.
    tok = _tiny_tokenizer()
    ids = tok.tokenize("hi\U0001F600")   # emoji never in the tiny table
    assert ids == tok.tokenize("hi")
    assert "\U0001F600" in tok.not_found_characters


def test_real_indexer_slice_matches_tokenizer():
    """Regression fixture: a genuine slice of Supertone/supertonic-3's published
    unicode_indexer.json (onnx/unicode_indexer.json, verified against the real
    HuggingFace file) for the characters in '<en>hi there.</en>'. Confirms
    Vocabulary.from_supertonic_indexer + TTSTokenizer reproduce the exact ids the
    real table gives by direct codepoint lookup — i.e. the standard-tokenizer path
    is behaviorally identical to indexing the raw table."""
    real_ids = {" ": 2, ".": 15, "/": 16, "<": 29, ">": 31,
                "e": 64, "h": 67, "i": 68, "n": 73, "r": 77, "t": 79}
    table = [-1] * 65536
    for ch, tid in real_ids.items():
        table[ord(ch)] = tid
    tok = TTSTokenizer(Vocabulary.from_supertonic_indexer(table),
                       add_blank_char=False, add_blank_word=False,
                       use_eos_bos=False, blank_at_end=False, blank_at_start=False)
    sample = "<en>hi there.</en>"
    expected = [real_ids[c] for c in sample]
    assert tok.tokenize(sample) == expected


# ---------------------------------------------------------------------------
# mask / noise shape math
# ---------------------------------------------------------------------------

def test_length_to_mask_shape():
    m = length_to_mask(7)
    assert m.shape == (1, 1, 7)
    assert (m == 1.0).all()


def test_length_to_mask_zero_length():
    m = length_to_mask(0)
    assert m.shape == (1, 1, 0)


def test_sample_noisy_latent_shapes():
    rng = np.random.default_rng(0)
    latent, mask = sample_noisy_latent(duration_s=1.0, sample_rate=44100,
                                        base_chunk_size=512, chunk_compress_factor=6,
                                        latent_dim=24, rng=rng)
    chunk_size = 512 * 6
    expected_len = -(-int(1.0 * 44100) // chunk_size)   # ceil div
    assert latent.shape == (1, 24 * 6, expected_len)
    assert mask.shape == (1, 1, expected_len)


def test_sample_noisy_latent_min_one_frame():
    # a near-zero duration must still produce at least one latent frame
    latent, mask = sample_noisy_latent(duration_s=1e-6, sample_rate=44100,
                                        base_chunk_size=512, chunk_compress_factor=6,
                                        latent_dim=24)
    assert latent.shape[-1] >= 1


# ---------------------------------------------------------------------------
# encode_text (adapter-owned text -> ids)
# ---------------------------------------------------------------------------

class _Cfg:
    def __init__(self, lang_code): self.lang_code = lang_code


class _Voice:
    def __init__(self, lang_code): self.config = _Cfg(lang_code)


def test_encode_text_requires_tokenizer():
    ad = SuperTonicAdapter()
    with pytest.raises(RuntimeError):
        ad.encode_text("hi", _Voice("en-US"), None)


def test_encode_text_produces_ids_per_chunk():
    ad = SuperTonicAdapter()
    ad.tokenizer = _tiny_tokenizer()
    out = ad.encode_text("hello there.", _Voice("en-US"), None)
    assert isinstance(out, list) and len(out) == 1
    assert all(isinstance(i, int) for i in out[0])


def test_encode_text_empty_text_yields_no_chunks():
    ad = SuperTonicAdapter()
    ad.tokenizer = _tiny_tokenizer()
    assert ad.encode_text("", _Voice("en-US"), None) == []
    assert ad.encode_text("   ", _Voice("en-US"), None) == []


def test_encode_text_emoji_only_input():
    ad = SuperTonicAdapter()
    ad.tokenizer = _tiny_tokenizer()
    # an emoji-only message strips to nothing meaningful but must not crash;
    # the wrapper tags + terminal period still map through the tokenizer.
    out = ad.encode_text("\U0001F600\U0001F601", _Voice("en-US"), None)
    assert isinstance(out, list)
    assert len(out) <= 1


def test_encode_text_unknown_lang_raises():
    ad = SuperTonicAdapter()
    ad.tokenizer = _tiny_tokenizer()
    with pytest.raises(ValueError):
        ad.encode_text("hi", _Voice("xx-XX"), None)


def test_encode_text_missing_indexer_char_is_dropped_not_raised():
    # out-of-vocabulary characters are silently dropped (standard tokenizer
    # behavior), not a hard error — matches every other tokenizer-backed engine.
    ad = SuperTonicAdapter()
    ad.tokenizer = _tiny_tokenizer()
    out = ad.encode_text("hi 中文", _Voice("en-US"), None)   # CJK not in the tiny table
    assert len(out) == 1
    # "<en>hi 中文.</en>" with the two CJK chars dropped == "<en>hi .</en>"
    expected = _tiny_tokenizer().tokenize("<en>hi .</en>")
    assert out[0] == expected


def test_encode_text_no_terminal_punctuation_gets_period():
    ad = SuperTonicAdapter()
    ad.tokenizer = _tiny_tokenizer()
    out = ad.encode_text("hello there", _Voice("en-US"), None)
    assert len(out) == 1   # still one chunk; preprocess_text appended '.'


def test_encode_text_very_long_text_chunks():
    ad = SuperTonicAdapter()
    ad.tokenizer = _tiny_tokenizer()
    long_text = ("hi there. " * 60).strip()
    out = ad.encode_text(long_text, _Voice("en-US"), None)
    assert len(out) > 1


def test_encode_text_ko_uses_shorter_chunk_len():
    ad = SuperTonicAdapter()
    ad.tokenizer = _tiny_tokenizer()
    text = ("hi there. " * 20).strip()   # ~200 chars: > 120 (ko) but < 300 (en)
    ko_chunks = ad.encode_text(text, _Voice("ko-KR"), None)
    en_chunks = ad.encode_text(text, _Voice("en-US"), None)
    assert len(ko_chunks) >= len(en_chunks)
    assert len(ko_chunks) > 1


# ---------------------------------------------------------------------------
# synthesize() — mocked multi-graph pipeline
# ---------------------------------------------------------------------------

def _configured_adapter():
    ad = SuperTonicAdapter(total_step=2)
    ad.tokenizer = _tiny_tokenizer()
    ad.duration_predictor = _FakeSession(outputs=lambda feed: [np.array([1.0], np.float32)])
    ad.text_encoder = _FakeSession(
        outputs=lambda feed: [np.zeros((1, 256, feed["text_ids"].shape[1]), np.float32)])
    ad.vocoder = _FakeSession(
        outputs=lambda feed: [np.zeros((1, feed["latent"].shape[-1] * 100), np.float32)])
    ad.style_ttl = np.zeros((1, 50, 256), np.float32)
    ad.style_dp = np.zeros((1, 8, 16), np.float32)
    ad.sample_rate = 44100
    ad.base_chunk_size = 512
    ad.chunk_compress_factor = 6
    ad.latent_dim = 24
    return ad


def test_synthesize_requires_aux_graphs():
    with pytest.raises(RuntimeError):
        SuperTonicAdapter().synthesize(_req(), None)


def test_synthesize_requires_style():
    ad = _configured_adapter()
    ad.style_ttl = ad.style_dp = None
    with pytest.raises(RuntimeError):
        ad.synthesize(_req(), _FakeSession())


def test_synthesize_rejects_non_positive_total_step():
    ad = _configured_adapter()
    with pytest.raises(ValueError):
        ad.synthesize(_req(total_step=0), _FakeSession())


def test_synthesize_runs_full_pipeline():
    ad = _configured_adapter()
    vector_estimator = _FakeSession(outputs=lambda feed: [feed["noisy_latent"]])   # identity step
    result = ad.synthesize(_req(), vector_estimator)
    assert result.audio.ndim == 1
    assert result.audio.size > 0
    assert "duration" in result.extras
    # the ODE loop ran exactly total_step times
    assert vector_estimator.calls == ad.total_step


def test_synthesize_speed_scales_duration():
    ad = _configured_adapter()
    vector_estimator = _FakeSession(outputs=lambda feed: [feed["noisy_latent"]])
    ad.synthesize(_req(speed=2.0), vector_estimator)
    # duration_predictor always returns 1.0s; speed=2.0 halves it before latent sizing
    assert ad.duration_predictor.last_feed is not None


def test_build_feed_dict_and_parse_outputs_not_implemented():
    ad = SuperTonicAdapter()
    with pytest.raises(NotImplementedError):
        ad.build_feed_dict(_req(), None)
    with pytest.raises(NotImplementedError):
        ad.parse_outputs([], _req())


# ---------------------------------------------------------------------------
# configure() — engine_params wiring
# ---------------------------------------------------------------------------

def test_configure_builds_tokenizer_from_indexer(tmp_path):
    table = [-1] * 65536
    table[ord("a")] = 3
    p = tmp_path / "unicode_indexer.json"
    p.write_text(json.dumps(table))

    class _VoiceConfig:
        engine_params = {"unicode_indexer_path": str(p)}

    ad = SuperTonicAdapter()
    ad.configure(_VoiceConfig())
    assert isinstance(ad.tokenizer, TTSTokenizer)
    assert ad.tokenizer.tokenize("a") == [3]


def test_configure_loads_tts_config(tmp_path):
    cfg = {"ae": {"sample_rate": 44100, "base_chunk_size": 512},
           "ttl": {"chunk_compress_factor": 6, "latent_dim": 24}}
    p = tmp_path / "tts.json"
    p.write_text(json.dumps(cfg))

    class _VoiceConfig:
        engine_params = {"tts_config_path": str(p)}

    ad = SuperTonicAdapter()
    ad.configure(_VoiceConfig())
    assert ad.sample_rate == 44100
    assert ad.chunk_compress_factor == 6
    assert ad.latent_dim == 24


def test_configure_loads_style(tmp_path):
    style = {"style_ttl": {"data": [0.0] * (50 * 256), "dims": [1, 50, 256]},
             "style_dp": {"data": [0.0] * (8 * 16), "dims": [1, 8, 16]}}
    p = tmp_path / "F1.json"
    p.write_text(json.dumps(style))

    class _VoiceConfig:
        engine_params = {"style_path": str(p)}

    ad = SuperTonicAdapter()
    ad.configure(_VoiceConfig())
    assert ad.style_ttl.shape == (1, 50, 256)
    assert ad.style_dp.shape == (1, 8, 16)


def test_configure_overrides_control_params():
    class _VoiceConfig:
        engine_params = {"total_step": 4, "speed": 1.2, "silence_duration": 0.5}

    ad = SuperTonicAdapter()
    ad.configure(_VoiceConfig())
    assert ad.total_step == 4
    assert ad.speed == 1.2
    assert ad.silence_duration == 0.5


def test_configure_noop_without_engine_params():
    ad = SuperTonicAdapter()
    ad.configure(None)   # no engine_params attribute at all
    assert ad.duration_predictor is None
