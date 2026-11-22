"""Piper 1.5 voice configs (piper_version 1.5.0) give the ids that Piper itself gives.

The fixture is pt_PT-tugão-medium from rhasspy/piper-voices, with its phoneme_id_map and
phoneme_counts cut to the symbols the tests read. Each expected id list is read off that map
by hand, in Piper's order: "^", "_", then each phoneme followed by "_", then "$".
"""
import copy
import json
import logging

import pytest
from scriptconv.phonemizers import GraphemePhonemizer

from phoonnx.config import SynthesisConfig, VoiceConfig
from phoonnx.voice import TTSVoice

TUGAO = {
    "audio": {"sample_rate": 22050, "quality": "medium"},
    "espeak": {"voice": "pt"},
    "phoneme_type": "espeak",
    "num_symbols": 256,
    "num_speakers": 1,
    "inference": {"noise_scale": 0.5, "length_scale": 1.0, "noise_w": 0.333},
    "phoneme_id_map": {
        "_": [0], "^": [1], "$": [2], " ": [3], ".": [10], "a": [14], "c": [16], "d": [17], "e": [18],
        "l": [24], "m": [25], "o": [27], "p": [28], "s": [31], "v": [34], "ç": [40], "ɔ": [54],
        "ɪ": [74], "ʊ": [100], "ˈ": [120], "̧": [140],
        "aɪ": [161], "aʊ": [162], "ɔɪ": [163], "eɪ": [164], "oʊ": [165],
    },
    "speaker_id_map": {},
    "hop_length": 256,
    "default_speaker_id": 0,
    "piper_version": "1.5.0",
    "vowel_clusters": [["a", "ɪ"], ["a", "ʊ"], ["e", "ɪ"], ["o", "ʊ"], ["ɔ", "ɪ"]],
    "language": {"code": "pt_PT", "family": "pt", "region": "PT", "name_native": "Português",
                 "name_english": "Portuguese", "country_english": "Portugal"},
    "dataset": "tugão",
    "phoneme_counts": {" ": 11136, "a": 1307, "aɪ": 211, "ɪ": 702},
}


def load(config):
    return VoiceConfig.from_dict(copy.deepcopy(config))


@pytest.mark.parametrize("phonemes, ids", [
    ("pˈaɪ", [1, 0, 28, 0, 120, 0, 161, 0, 2]),  # pai
    ("mˈaʊ", [1, 0, 25, 0, 120, 0, 162, 0, 2]),  # mau
    ("lˈeɪ", [1, 0, 24, 0, 120, 0, 164, 0, 2]),  # lei
    ("vˈoʊ", [1, 0, 34, 0, 120, 0, 165, 0, 2]),  # vou
    ("dˈɔɪ", [1, 0, 17, 0, 120, 0, 163, 0, 2]),  # dói
    ("ˈaɪ ˈa ɪ.", [1, 0, 120, 0, 161, 0, 3, 0, 120, 0, 14, 0, 3, 0, 74, 0, 10, 0, 2]),
])
def test_a_vowel_cluster_reads_as_its_own_id(phonemes, ids):
    assert load(TUGAO).tokenizer.tokenize(phonemes) == ids


@pytest.mark.parametrize("phonemes, ids", [
    ("pˈaɪ", [1, 0, 28, 0, 120, 0, 14, 0, 74, 0, 2]),
    ("mˈaʊ", [1, 0, 25, 0, 120, 0, 14, 0, 100, 0, 2]),
    ("dˈɔɪ", [1, 0, 17, 0, 120, 0, 54, 0, 74, 0, 2]),
])
def test_piper_default_map_without_vowel_clusters_reads_a_diphthong_as_two_ids(phonemes, ids):
    # no_NO-nvcc-medium: Piper merges only the clusters in vowel_clusters, and this config has none
    config = {key: value for key, value in TUGAO.items() if key != "vowel_clusters"}
    assert load(config).tokenizer.tokenize(phonemes) == ids


# ghananlpcommunity/stable-twi-tts: a Piper 1.5.0 config from another trainer, which folded its
# own multi-character keys; the ids are those of its phoneme_id_map
TWI = {**{key: value for key, value in TUGAO.items() if key not in ("vowel_clusters", "language", "dataset")},
       "espeak": {"voice": "lfn"},
       "phoneme_id_map": {"_": [0], "^": [1], "$": [2], " ": [3], ".": [10], "a": [14], "d": [17], "ɪ": [74],
                          "ʒ": [108], "ˈ": [192], "dʒ": [46], "aɪ": [180], "aʊ": [181], "eɪ": [182]},
       "phoneme_counts": {}}


def test_a_config_with_keys_of_its_own_trainer_keeps_the_folding():
    assert load(TWI).tokenizer.tokenize("dʒˈaɪ. ˈa ɪ") == [1, 0, 46, 0, 192, 0, 180, 0, 10, 0, 3, 0, 192, 0, 14, 0,
                                                           3, 0, 74, 0, 2]


def test_a_map_with_some_of_the_default_diphthongs_reads_them_as_two_ids():
    config = {key: value for key, value in TUGAO.items() if key != "vowel_clusters"}
    config["phoneme_id_map"] = {k: v for k, v in TUGAO["phoneme_id_map"].items() if k not in ("aʊ", "ɔɪ", "eɪ", "oʊ")}
    assert load(config).tokenizer.tokenize("pˈaɪ") == [1, 0, 28, 0, 120, 0, 14, 0, 74, 0, 2]


def test_diphthongs_away_from_the_piper_default_ids_keep_the_folding():
    config = {**TWI, "phoneme_id_map": {k: v for k, v in TWI["phoneme_id_map"].items() if k != "dʒ"}}
    assert load(config).tokenizer.tokenize("ˈaɪ") == [1, 0, 192, 0, 180, 0, 2]


def test_a_phoonnx_side_car_without_vowel_clusters_keeps_the_folding():
    config = {key: value for key, value in TUGAO.items() if key != "vowel_clusters"}
    side_car = {**config, "piper_version": "phoonnx-1.0"}
    assert load(side_car).tokenizer.tokenize("pˈaɪ") == [1, 0, 28, 0, 120, 0, 161, 0, 2]


def test_a_precomposed_symbol_reads_as_its_decomposition():
    # Piper reads its phonemes in NFD: "ç" is "c" and U+0327, and id 40 never reaches the model
    assert load(TUGAO).tokenizer.tokenize("ˈɪç") == [1, 0, 120, 0, 74, 0, 16, 0, 140, 0, 2]
    assert load(TUGAO).tokenizer.tokenize("ˈɪç") == [1, 0, 120, 0, 74, 0, 16, 0, 140, 0, 2]


def test_a_decomposition_part_missing_from_the_map_gives_no_id():
    config = copy.deepcopy(TUGAO)
    del config["phoneme_id_map"]["̧"]  # piper 0.2.0 voices map "ç" but not U+0327
    assert load(config).tokenizer.tokenize("ç") == [1, 0, 16, 0, 2]


def test_a_phoonnx_side_car_keeps_the_precomposed_id():
    side_car = {**TUGAO, "piper_version": "phoonnx-1.0"}
    assert load(side_car).tokenizer.tokenize("ç") == [1, 0, 40, 0, 2]


@pytest.mark.parametrize("espeak_voice, code, lang", [("pt", "pt_PT", "pt"), ("es-419", "es_MX", "es-419"),
                                                      ("nb", "no_NO", "nb")])
def test_espeak_voice_is_the_phonemizer_language(espeak_voice, code, lang):
    config = {**TUGAO, "espeak": {"voice": espeak_voice}, "language": {"code": code}}
    assert load(config).lang_code == lang


def test_a_text_model_keeps_the_language_code():
    config = {**TUGAO, "phoneme_type": "text", "espeak": {"voice": "es-419"}, "language": {"code": "es_MX"}}
    assert load(config).lang_code == "es-MX"


def test_the_config_loads_without_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        config = load(TUGAO)
        config.tokenizer.tokenize("pˈaɪ mˈaʊ lˈeɪ vˈoʊ dˈɔɪ ç.")
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert config.tokenizer.not_found_characters == set()
    assert (config.sample_rate, config.hop_length, config.noise_scale, config.noise_w_scale) == (22050, 256, 0.5, 0.333)


def _speaker(config, syn_config=SynthesisConfig()):
    voice = TTSVoice(session=None, config=load(config), phonemizer=GraphemePhonemizer())
    return voice._request([1, 0, 2], syn_config).speaker_id


def test_default_speaker_id_is_the_speaker_of_a_multi_speaker_voice():
    multi = {**TUGAO, "num_speakers": 4, "default_speaker_id": 2}
    assert _speaker(multi) == 2
    assert _speaker(multi, SynthesisConfig(speaker_id=0)) == 0
    assert _speaker({**multi, "num_speakers": 1}) is None
    assert _speaker({k: v for k, v in multi.items() if k != "default_speaker_id"}) is None


def test_the_native_config_keeps_the_ids_and_the_default_speaker():
    voice = load({**TUGAO, "num_speakers": 4, "default_speaker_id": 2})
    native = VoiceConfig.from_dict(json.loads(json.dumps(voice.to_dict(), ensure_ascii=False)))
    for phonemes in ("pˈaɪ mˈaʊ lˈeɪ vˈoʊ dˈɔɪ.", "ˈɪç"):
        assert native.tokenizer.tokenize(phonemes) == voice.tokenizer.tokenize(phonemes)
    assert native.default_speaker_id == 2
