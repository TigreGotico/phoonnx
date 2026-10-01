import numpy as np
import pytest

from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.engines.chatterbox import ChatterboxAdapter, _apply_repetition_penalty


def _build_tiny_onnx_model(path):
    """Write a minimal but real single-node ONNX model to *path*."""
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "tiny", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_configure_shares_aux_sessions_across_voices(tmp_path):
    # The aux graphs (speech_encoder, embed_tokens, conditional_decoder) are
    # ~1.19GB per voice and identical across every voice that shares a
    # checkpoint; configure() must route them through make_session so a
    # second voice over the same files reuses the sessions the first one
    # built, instead of loading its own private copy of each.
    from types import SimpleNamespace

    speech_encoder = tmp_path / "speech_encoder.onnx"
    embed_tokens = tmp_path / "embed_tokens.onnx"
    cond_decoder = tmp_path / "conditional_decoder.onnx"
    for p in (speech_encoder, embed_tokens, cond_decoder):
        _build_tiny_onnx_model(p)

    def _voice_config():
        return SimpleNamespace(engine_params={
            "speech_encoder_path": str(speech_encoder),
            "embed_tokens_path": str(embed_tokens),
            "conditional_decoder_path": str(cond_decoder),
        })

    first = ChatterboxAdapter()
    first.configure(_voice_config())
    second = ChatterboxAdapter()
    second.configure(_voice_config())

    assert first.speech_encoder is second.speech_encoder
    assert first.embed_tokens is second.embed_tokens
    assert first.cond_decoder is second.cond_decoder


def _req(**params):
    return AdapterSynthesisRequest(phoneme_ids=np.array([[1, 2, 3]], np.int64),
                                   phoneme_lengths=np.array([3], np.int64),
                                   speaker_id=0, language_id=0, params=params)


def test_chatterbox_registered():
    from phoonnx.engines import list_engines
    assert "chatterbox" in list_engines()


def test_chatterbox_detect():
    assert ChatterboxAdapter.detect({"engine": "chatterbox"})
    assert not ChatterboxAdapter.detect({"engine": "vits"})
    assert not ChatterboxAdapter.detect(None)


def test_chatterbox_encode_text_is_bpe():
    # the engine owns text->ids: chatterbox BPEs the raw text (punctuation kept), one chunk
    class _Bpe:
        def tokenize(self, t): return [ord(c) for c in t]

    class _Voice:
        tokenizer = _Bpe()

    out = ChatterboxAdapter().encode_text("hi!", _Voice(), None)
    assert out == [[ord("h"), ord("i"), ord("!")]]


def test_chatterbox_requires_aux_graphs():
    with pytest.raises(RuntimeError):
        ChatterboxAdapter().synthesize(_req(reference_audio=(np.zeros(100, np.float32), 24000)), None)


def test_chatterbox_requires_reference():
    ad = ChatterboxAdapter()
    ad.embed_tokens = ad.speech_encoder = ad.cond_decoder = object()   # pretend configured
    with pytest.raises(RuntimeError):
        ad.synthesize(_req(), None)                                    # no reference_audio


def test_repetition_penalty():
    scores = np.array([[1.0, 2.0, -1.0, 4.0]])
    prev = np.array([[1, 2]])                       # tokens 1 (+) and 2 (-) already emitted
    out = _apply_repetition_penalty(prev, scores, 2.0)
    assert out[0, 1] == pytest.approx(1.0)          # 2.0 / 2  (positive -> divide)
    assert out[0, 2] == pytest.approx(-2.0)         # -1.0 * 2 (negative -> multiply)
    assert out[0, 0] == pytest.approx(1.0)          # untouched


def test_bpe_tokenizer_joins_units():
    from phoonnx.tokenizer import BPETokenizer
    bpe = BPETokenizer.__new__(BPETokenizer)        # bypass __init__ (no tokenizer.json on disk)

    class _Enc:
        def __init__(self, ids): self.ids = ids

    class _Tok:
        def encode(self, text): return _Enc([ord(c) for c in text])

    bpe._tok = _Tok()
    # a list of char units (from the UNICODE-style path) is joined back to text first
    assert bpe.tokenize(["a", "b", "c"]) == [97, 98, 99]
    assert bpe.tokenize("abc") == [97, 98, 99]


def test_sample_top_p_picks_dominant():
    from phoonnx.engines.chatterbox import _sample_top_p
    rng = np.random.default_rng(0)
    logits = np.array([[12.0, 0.0, -5.0, 0.1]])      # token 0 dominates the nucleus
    tok = _sample_top_p(logits, 0.8, 0.9, rng)
    assert tok.shape == (1, 1) and tok[0, 0] == 0


def test_mtl_tokenizer_language_frontend():
    from phoonnx.tokenizer import ChatterboxMTLTokenizer
    tok = ChatterboxMTLTokenizer.__new__(ChatterboxMTLTokenizer)
    seen = {}

    class _Enc:
        def __init__(self, ids): self.ids = ids

    class _Tok:
        def token_to_id(self, t): return 1 if t in ("[pt]", "[SPACE]") else None
        def encode(self, text): seen["text"] = text; return _Enc([1, 2, 3])

    tok._tok = _Tok()
    # known language -> [lang] prefix, lowercase, spaces become [SPACE]
    tok.tokenize("Hello World", language="pt-BR")
    assert seen["text"] == "[pt]hello[SPACE]world"
    # unknown language token -> plain BPE, untouched
    tok.tokenize("Hello World", language="xx")
    assert seen["text"] == "Hello World"


def test_mtl_tokenizer_lang_tokens_override():
    from phoonnx.tokenizer import ChatterboxMTLTokenizer
    tok = ChatterboxMTLTokenizer.__new__(ChatterboxMTLTokenizer)
    seen = {}

    class _Enc:
        def __init__(self, ids): self.ids = ids

    class _Tok:                                   # [ar] + [SPACE] exist; [eg] does NOT
        def token_to_id(self, t): return 1 if t in ("[ar]", "[SPACE]") else None
        def encode(self, text): seen["text"] = text; return _Enc([1])

    tok._tok = _Tok()
    # dialect "hack": lang_tokens maps ar-EG -> literal "eg" and prepends [eg] even though
    # it isn't a single vocab token (it BPE-splits) — sidesteps lang-code normalization too
    tok.tokenize("xy", language="ar-EG", lang_tokens={"ar-EG": "eg"})
    assert seen["text"].startswith("[eg]")
    # without a map, ar-EG derives the base [ar] token (present in the vocab)
    tok.tokenize("xy", language="ar-EG")
    assert seen["text"].startswith("[ar]")


def test_lahgtna_voice_index_entries():
    # the 10 Arabic-dialect lahgtna voices must build valid TTSModelInfo objects:
    # unified model (one URL set), dialect-accurate lang codes, and NO lang_tokens —
    # the MTL tokenizer derives the [ar] token from the lang code by itself
    import json
    from pathlib import Path
    import phoonnx
    from phoonnx.model_manager import TTSModelInfo

    index = json.loads((Path(phoonnx.__file__).parent / "voice_index" /
                        "chatterbox.json").read_text())
    dialects = {"eg": "ar-EG", "sa": "ar-SA", "mo": "ar-MA", "iq": "ar-IQ",
                "lb": "ar-LB", "sd": "ar-SD", "ly": "ar-LY", "sy": "ar-SY",
                "tn": "ar-TN", "ps": "ar-PS"}
    for dia, lang in dialects.items():
        vid = f"chatterbox/lahgtna/{dia}"
        assert vid in index, vid
        entry = index[vid]
        assert "lang_tokens" not in entry
        info = TTSModelInfo(**entry)
        assert info.engine == "chatterbox"
        assert info.lang == lang
        for field in ("model_url", "speech_encoder_url", "embed_tokens_url",
                      "conditional_decoder_url", "tokenizer_config_url"):
            assert entry[field].startswith(
                "https://huggingface.co/OpenVoiceOS/phoonnx-chatterbox-lahgtna/")
