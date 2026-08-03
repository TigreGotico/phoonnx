import json
from pathlib import Path

import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.sparktts import (
    N_GLOBAL_TOKENS,
    REF_SEGMENT_SAMPLES,
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
