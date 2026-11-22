"""Piper voices whose ``phoneme_type`` names a phonemizer of its own load and give Piper's ids.

``fixtures/piper_phoneme_types.json`` holds, per type, one voice config of
rhasspy/piper-voices and the phonemes and ids that the reference
implementation gives for two FLORES-200 devtest sentences:

    phonemes = PiperVoice.phonemize(text)        # per sentence
    ids = PiperVoice.phonemes_to_ids(phonemes)

with piper-tts 1.8.0 (Lithuanian: OHF-Voice/piper1-gpl main 39ac0a0, the
first code with that type). The IPA types also hold the ids that
``piper.phoneme_ids.phonemes_to_ids`` gives to ``["a", "ɪ"]``: Piper merges
vowel clusters only on its espeak path, so those two symbols keep two ids
although the map holds ``aɪ``.
"""
import json
from pathlib import Path

import pytest

from phoonnx.compat import LoadRequest, PiperLoader

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "piper_phoneme_types.json").read_text(encoding="utf-8"))
TYPES = [kind for kind in FIXTURE if kind != "source"]

# GROUP_END_PHONEMES of src/piper/phonemize_chinese.py (piper1-gpl v1.8.0)
PINYIN_GROUP_END = frozenset(["1", "2", "3", "4", "5", "。", "？", "！", ".", "?", "!",
                              "—", "…", "、", "，", "：", "；", ",", ":", ";", " "])

BACKENDS = {
    "pinyin": ("g2pw", "pinyin"),
    "hebrew": ("hebrew_ipa", "ipa"),
    "japanese": ("openjtalk_prosody", "ipa"),
    "thai": ("tltk", "ipa"),
    "lithuanian": ("espeak_lt_accent", "ipa"),
}


def load(kind: str):
    return PiperLoader.load(LoadRequest(config=json.loads(json.dumps(FIXTURE[kind]["config"]))))


@pytest.mark.parametrize("kind", TYPES)
def test_the_type_maps_to_its_scriptconv_backend(kind):
    loaded = load(kind)
    assert (loaded.phoneme_type, loaded.alphabet.value) == BACKENDS[kind]
    assert {"phoneme_type", "alphabet"} <= loaded.pinned
    assert loaded.tokenizer.fold_compounds is False


@pytest.mark.parametrize("kind", TYPES)
def test_the_voice_config_loads_with_the_installed_scriptconv(kind):
    # The loader gives the backend as a name. VoiceConfig needs a Phonemizer
    # member of that name, so this fails when scriptconv lacks the backend.
    from phoonnx.config import VoiceConfig

    config = VoiceConfig.from_dict(json.loads(json.dumps(FIXTURE[kind]["config"])))
    assert (config.phoneme_type.value, config.alphabet.value) == BACKENDS[kind]


# the phoonnx extra of the language of each type's voices
LANGUAGE_EXTRAS = {"pinyin": "zh", "hebrew": "he", "japanese": "ja", "thai": "th", "lithuanian": "lt"}


@pytest.mark.parametrize("kind", TYPES)
def test_the_language_extra_installs_the_backend(kind):
    import tomllib

    from packaging.requirements import Requirement
    from scriptconv.phonemizers.enums import Phonemizer
    from scriptconv.phonemizers.registry import PHONEMIZER_REGISTRY

    pyproject = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [Requirement(r) for r in pyproject["project"]["optional-dependencies"][LANGUAGE_EXTRAS[kind]]]
    _, _, backend_extra = PHONEMIZER_REGISTRY[Phonemizer(BACKENDS[kind][0])]
    assert any(r.name == "scriptconv" and backend_extra in r.extras for r in requirements)


def test_hebrew_voices_diacritize_with_nakdimon():
    loaded = load("hebrew")
    assert (loaded.add_diacritics, loaded.diacritizer_model) == (True, "nakdimon")


def test_pinyin_voices_put_a_blank_after_a_group_only():
    assert load("pinyin").tokenizer.blank_after == PINYIN_GROUP_END
    assert all(load(kind).tokenizer.blank_after is None for kind in TYPES if kind != "pinyin")


@pytest.mark.parametrize("kind, case", [(kind, case) for kind in TYPES for case in FIXTURE[kind]["cases"]])
def test_the_tokenizer_gives_piper_ids(kind, case):
    tokenizer = load(kind).tokenizer
    assert [tokenizer.tokenize(list(sentence)) for sentence in case["phonemes"]] == case["ids"]


def test_blank_after_round_trips_through_the_native_config():
    from phoonnx.config import VoiceConfig

    config = VoiceConfig.from_dict(json.loads(json.dumps(FIXTURE["pinyin"]["config"])))
    again = VoiceConfig.from_dict(config.to_dict())
    case = FIXTURE["pinyin"]["cases"][0]
    assert again.tokenizer.blank_after == PINYIN_GROUP_END
    assert [again.tokenizer.tokenize(list(s)) for s in case["phonemes"]] == case["ids"]


def test_a_blank_follows_the_last_id_of_a_symbol():
    # "é" is not in the vocabulary, so it reads as "e" and U+0301, two ids of one symbol
    from phoonnx.tokenizer import TTSTokenizer, Vocabulary

    vocabulary = Vocabulary(char2idx={"_": 0, "^": 1, "$": 2, "m": 3, "e": 4, "\u0301": 5},
                            pad="_", bos="^", eos="$", blank="_")
    tokenizer = TTSTokenizer(vocabulary, add_blank_char=True, add_blank_word=False, use_eos_bos=True,
                             blank_at_end=True, blank_at_start=True, blank_after=frozenset("é"))
    assert tokenizer.tokenize(["m", "é", "m"]) == [1, 3, 4, 5, 0, 3, 2]
