"""Round-trip: foreign config -> VoiceConfig.to_native_dict() -> native config.

Modernizing community mirrors to native phoonnx configs must reproduce the exact
tokenization, including models that differ from the phoonnx defaults (e.g. MMS /
matcha use no BOS/EOS). The native config must carry the tokenizer flags.
"""
from phoonnx.config import VoiceConfig


def _ids(vc):
    keys = [k for k in vc.tokenizer.vocabulary.char2idx if len(k) == 1][2:10]
    return vc.tokenizer.tokenize(keys)


def _roundtrip(vc):
    native = vc.to_native_dict()
    assert "phoonnx_version" in native
    assert native["phoneme_id_map"]
    vc2 = VoiceConfig.from_dict(dict(native))
    return vc, vc2, native


def test_piper_roundtrip():
    cfg = {
        "piper_version": "1.0.0", "phoneme_type": "espeak",
        "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a": [3], "b": [4],
                            "c": [5], "!": [6], " ": [7]},
        "num_symbols": 8, "num_speakers": 1, "audio": {"sample_rate": 22050},
        "espeak": {"voice": "en-us"},
        "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
    }
    vc = VoiceConfig.from_dict(cfg)
    vc, vc2, native = _roundtrip(vc)
    assert _ids(vc) == _ids(vc2)
    assert native["use_eos_bos"] == vc.tokenizer.use_eos_bos


def test_mms_style_roundtrip_no_eos_bos():
    # transformers/MMS path: add_blank, no bos/eos
    vocab = {"_": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, " ": 6}
    tok_cfg = {"add_blank": True, "language": "en", "pad_token": "_"}
    vc = VoiceConfig.from_dict({}, vocab=vocab, tokenizer_config=tok_cfg,
                               phoneme_type="graphemes", alphabet="unicode",
                               lang_code="en")
    assert vc.tokenizer.use_eos_bos is False  # the case the old native path broke
    vc, vc2, native = _roundtrip(vc)
    assert native["use_eos_bos"] is False
    assert vc2.tokenizer.use_eos_bos is False
    assert _ids(vc) == _ids(vc2)


def test_native_carries_tokenizer_flags():
    vocab = {"_": 0, "a": 1, "b": 2, "c": 3}
    vc = VoiceConfig.from_dict({}, vocab=vocab, tokenizer_config={"add_blank": True, "language": "en", "pad_token": "_"},
                               phoneme_type="graphemes", alphabet="unicode", lang_code="en")
    native = vc.to_native_dict()
    for k in ("add_blank_char", "add_blank_word", "use_eos_bos",
              "blank_at_start", "blank_at_end", "pad", "blank", "phoneme_id_map"):
        assert k in native, k
