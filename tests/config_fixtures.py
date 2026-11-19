"""Voice configs of each third-party format and the helpers that the config tests share."""


def piper_ids(id_map, phonemes):
    """Piper's own id stream: BOS, PAD, then each phoneme followed by PAD, then EOS."""
    ids = [*id_map["^"], *id_map["_"]]
    for phoneme in phonemes:
        ids += [*id_map[phoneme], *id_map["_"]]
    return ids + id_map["$"]


PIPER = {
    "piper_version": "1.0.0", "phoneme_type": "espeak",
    "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a": [3], "b": [4], "!": [5], " ": [6]},
    "num_symbols": 7, "num_speakers": 1, "audio": {"sample_rate": 22050},
    "espeak": {"voice": "en-us"},
    "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
}
VOSK = {"espeak": {"voice": "ru"},
        "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a0": [10], "sch": [20]}}
COQUI = {"audio": {"sample_rate": 22050}, "add_blank": True, "enable_eos_bos_chars": False,
         "characters": {"characters_class": "TTS.tts.models.vits.VitsCharacters", "pad": "<PAD>",
                        "punctuations": "!,", "characters": "ab", "blank": "<BLNK>"},
         "datasets": [{"language": "en"}]}
CANONICAL = {"engine": "coqui", "phoneme_type": "espeak", "alphabet": "ipa", "lang_code": "ca-ES",
             "num_symbols": 4, "audio": {"sample_rate": 22050},
             "phoneme_id_map": {"_": 0, " ": 1, "a": 2, "b": 3}, "pad": "_", "blank": "_"}
TRANSFORMERS = dict(vocab={"_": 0, "a": 1, "b": 2, " ": 3},
                    tokenizer_config={"add_blank": True, "language": "en", "pad_token": "_"},
                    phoneme_type="graphemes", alphabet="unicode", lang_code="en")


def assert_same_voice(old, new, sentences):
    for sentence in sentences:
        assert new.tokenizer.tokenize(sentence) == old.tokenizer.tokenize(sentence)
    assert new.frontend == old.frontend
    assert (new.engine, new.sample_rate, new.num_symbols) == (old.engine, old.sample_rate, old.num_symbols)
