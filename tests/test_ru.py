"""Tests for the alphacep vosk-tts Russian voices (config + tokenizer).

The Russian G2P itself lives in scriptconv (``VoskPhonemizer``); what phoonnx
owns — and what is tested here — is recognising a vosk voice and turning its
phoneme tokens into the exact id stream the model was trained on.
"""
from phoonnx.config import VoiceConfig, Engine, Alphabet, PhonemeType
from phoonnx.tokenizer import TTSTokenizer

# A minimal vosk phoneme inventory (piper-shaped: id is a 1-element list).
# Order/ids only need to be internally consistent for the tests.
_VOSK_PHONES = [
    "_", "^", "$", " ", ",", ".", "-",
    "a0", "a1", "e0", "e1", "i0", "i1", "o0", "o1", "u0", "u1", "y0", "y1",
    "b", "bj", "c", "ch", "d", "dj", "f", "g", "h", "j", "k", "l", "lj",
    "m", "mj", "n", "nj", "p", "r", "rj", "s", "sch", "sh", "sj", "t", "tj",
    "v", "vj", "z", "zh",
]


def _vosk_config():
    return {
        "audio": {"sample_rate": 22050},
        "espeak": {"voice": "ru"},
        "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
        "num_speakers": 1,
        "phoneme_id_map": {p: [i] for i, p in enumerate(_VOSK_PHONES)},
    }


# --------------------------------------------------------------------------- #
# config detection / construction
# --------------------------------------------------------------------------- #
def test_is_vosk_detects_inventory():
    assert VoiceConfig.is_vosk(_vosk_config()) is True


def test_is_vosk_rejects_non_vosk():
    # piper espeak config: no a0/sch markers
    assert VoiceConfig.is_vosk({"phoneme_id_map": {"a": [0], "b": [1]}}) is False
    assert VoiceConfig.is_vosk({}) is False


def test_from_dict_builds_vosk_voice():
    cfg = VoiceConfig.from_dict(_vosk_config())
    assert cfg.engine == Engine.VOSK
    assert cfg.phoneme_type == PhonemeType.VOSK
    assert cfg.alphabet == Alphabet.VOSK
    assert cfg.sample_rate == 22050
    # tokenizer must not greedily fold neighbouring single-char tokens
    assert cfg.tokenizer.fold_compounds is False


def test_explicit_engine_selects_vosk_without_sniffing():
    # a voice that names its engine is authoritative, even with an inventory
    # the shape-sniffer would not recognise
    cfg = VoiceConfig.from_dict({"engine": "vosk",
                                 "audio": {"sample_rate": 22050},
                                 "phoneme_id_map": {"_": [0], "^": [1], "$": [2],
                                                    "p": [3]}})
    assert cfg.engine == Engine.VOSK
    assert cfg.phoneme_type == PhonemeType.VOSK
    assert cfg.alphabet == Alphabet.VOSK


def test_explicit_non_vosk_engine_is_not_stolen():
    cfg = _vosk_config()
    cfg["engine"] = "piper"
    cfg["phoneme_type"] = "espeak"
    assert VoiceConfig.from_dict(cfg).engine != Engine.VOSK


# --------------------------------------------------------------------------- #
# tokenizer parity: phoonnx must reproduce vosk's id stream exactly
# --------------------------------------------------------------------------- #
def _vosk_reference_ids(tokens, char2idx):
    """Replicate vosk_tts g2p_noembed: ^ 0 p 0 p ... 0 $ (blank id 0)."""
    phones = ["^"] + tokens + ["$"]
    ids = [char2idx[phones[0]]]
    for p in phones[1:]:
        ids += [0, char2idx[p]]
    return ids


def test_tokenizer_matches_vosk_id_stream():
    cfg = VoiceConfig.from_dict(_vosk_config())
    char2idx = {p: i for i, p in enumerate(_VOSK_PHONES)}
    # the cluster case that compound-folding would corrupt
    tokens = ["s", "h", "o0", "dj", "i1", "tj", " ", "s", "ch", "a1"]
    expected = _vosk_reference_ids(tokens, char2idx)
    assert cfg.tokenizer.tokenize(tokens) == expected
    # explicitly: 's' 'h' did not collapse into the 'sh' id
    assert char2idx["sh"] not in cfg.tokenizer.tokenize(tokens)


def test_multi_char_phonemes_still_resolve():
    # folding off must not stop genuine multi-char tokens ('sch', 'a1') from
    # being looked up — they arrive as whole tokens
    cfg = VoiceConfig.from_dict(_vosk_config())
    char2idx = {p: i for i, p in enumerate(_VOSK_PHONES)}
    ids = cfg.tokenizer.tokenize(["sch", "a1"])
    assert char2idx["sch"] in ids and char2idx["a1"] in ids


def test_fold_compounds_flag_changes_behaviour():
    # guard: with folding on (default), adjacent s+h WOULD merge to 'sh'
    folding = TTSTokenizer.from_piper_config(_vosk_config())
    assert folding.fold_compounds is True
    char2idx = {p: i for i, p in enumerate(_VOSK_PHONES)}
    assert char2idx["sh"] in folding.tokenize(["s", "h"])


def test_fold_compounds_round_trips_through_phoonnx_config():
    # a vosk voice exported to a native phoonnx config must not silently
    # regain compound folding when reloaded
    cfg = VoiceConfig.from_dict(_vosk_config())
    native = cfg.to_native_dict()
    assert native["fold_compounds"] is False
    assert TTSTokenizer.from_phoonnx_config(native).fold_compounds is False


# --------------------------------------------------------------------------- #
# bundled voice index
# --------------------------------------------------------------------------- #
def test_voice_index_entries_construct():
    import json
    from phoonnx.model_manager import TTSModelInfo, TTSModelManager

    index = TTSModelManager.voice_index_path() / "vosk.json"
    entries = json.loads(index.read_text(encoding="utf-8"))
    assert entries
    for voice_id, entry in entries.items():
        info = TTSModelInfo(**entry)
        assert info.voice_id == voice_id
        assert info.engine == Engine.VOSK
        assert info.alphabet == Alphabet.VOSK
        assert info.phoneme_type == PhonemeType.VOSK
        assert info.dictionary_url
        assert info.lang.startswith("ru")
