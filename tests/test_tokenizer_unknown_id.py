"""An F5-TTS voice reads a symbol that its vocabulary lacks as id 0, as its upstream code does."""
from phoonnx.config import VoiceConfig

# A row recorded with the upstream code over the F5-TTS v1 Base vocab.txt:
# get_tokenizer reads line i as id i, and list_str_to_idx in
# src/f5_tts/model/utils.py (SWivid/F5-TTS 600e5fd) maps each token with
# vocab_char_map.get(c, 0). The map holds every vocab.txt entry that the row,
# its NFD form or its lowercase form uses, with the vocab.txt id.
F5_BASE_ROW = {
    "tokens": list("Ćao Ɛyɛ؟ ok"),
    "ids": [0, 62, 827, 0, 0, 1236, 1550, 0, 0, 827, 560],
    "phoneme_id_map": {" ": 0, "C": 34, "a": 62, "k": 560, "o": 827, "y": 1236,
                       "ć": 1495, "ɛ": 1550, "́": 1574},
}


def _f5_config(engine="f5tts"):
    return {"phoonnx_version": "1.0", "engine": engine, "phoneme_type": "graphemes",
            "alphabet": "unicode", "lang_code": "multi", "audio": {"sample_rate": 24000},
            "phoneme_id_map": F5_BASE_ROW["phoneme_id_map"], "add_blank_char": False,
            "blank_at_start": False, "blank_at_end": False, "use_eos_bos": False,
            "fold_compounds": False}


def test_f5tts_reads_a_symbol_its_vocabulary_lacks_as_id_0():
    """Ć, Ɛ and ؟ are not in the vocabulary, so upstream gives each id 0, with no
    NFD (C + U+0301) or lowercase (ɛ) fallback."""
    tokenizer = VoiceConfig.from_dict(_f5_config()).tokenizer
    assert tokenizer.tokenize(F5_BASE_ROW["tokens"]) == F5_BASE_ROW["ids"]


def test_another_engine_still_drops_an_unknown_symbol_and_falls_back():
    tokenizer = VoiceConfig.from_dict(_f5_config(engine="phoonnx")).tokenizer
    assert tokenizer.unknown_id is None
    assert tokenizer.tokenize(F5_BASE_ROW["tokens"]) == [34, 1574, 62, 827, 0, 1550, 1236, 1550, 0, 827, 560]


# A row recorded with the upstream code: convert_char_to_pinyin(["我㗑你㕶"])
# (SWivid/F5-TTS 600e5fd, f5-tts 1.1.22) gives the syllables bai and n3, which
# the F5-TTS v1 Base vocab.txt lacks, and list_str_to_idx reads each as id 0.
# The map holds the vocab.txt entries of the row and the two nearest syllables.
F5_BASE_PINYIN_ROW = {
    "tokens": [" ", "wo3", " ", "bai", " ", "ni3", " ", "n3"],
    "ids": [0, 1178, 0, 0, 0, 794, 0, 0],
    "phoneme_id_map": {" ": 0, "wo3": 1178, "ni3": 794, "bai2": 85, "n2": 766},
}


def test_f5tts_reads_a_pinyin_syllable_its_vocabulary_lacks_as_id_0():
    """A missing token of several characters takes id 0 too, as one token."""
    config = dict(_f5_config(), phoneme_id_map=F5_BASE_PINYIN_ROW["phoneme_id_map"])
    tokenizer = VoiceConfig.from_dict(config).tokenizer
    assert tokenizer.tokenize(F5_BASE_PINYIN_ROW["tokens"]) == F5_BASE_PINYIN_ROW["ids"]
