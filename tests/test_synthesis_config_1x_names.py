"""``SynthesisConfig`` groups its settings in ``stages``, ``audio`` and ``reference``.

Each 1.x flat name still works as a keyword, as an attribute and in ``from_dict``,
with a ``DeprecationWarning`` that names the new home and the release that removes
the old name. The expected removal release is the next major of ``phoonnx.version``.
"""
import dataclasses
import warnings

import numpy as np
import pytest
from scriptconv.frontend import Chunk, Stages

import phoonnx.version
from phoonnx.config import Alphabet, AudioPost, SpeakerReference, SynthesisConfig, VoiceConfig
from phoonnx.encoders import stages_of
from phoonnx.voice import TTSVoice

REMOVAL = f"{phoonnx.version.VERSION_MAJOR + 1}.0.0"

# old name, value given, new path, value read at the new path
NAMES = [
    ("alphabet", Alphabet.IPA, "input_alphabet", Alphabet.IPA),
    ("enable_phonetic_spellings", False, "phonetic_spellings", False),
    ("speaker_id", 3, "speaker", 3),
    ("lang_id", 2, "language", 2),
    ("extra_params", {"seed": 1}, "extra", {"seed": 1}),
    ("speaker_reference", "me.wav", "reference.audio", "me.wav"),
    ("speaker_reference_text", "olá", "reference.text", "olá"),
    ("speaker_reference_lang", "pt", "reference.lang", "pt"),
    ("add_diacritics", True, "stages.diacritize", True),
    ("diacritizer_model", "rawi", "stages.diacritizer_model", "rawi"),
    ("chunk", "clause", "stages.chunk", Chunk.CLAUSE),
    ("normalize_audio", False, "audio.normalize", False),
    ("target_loudness", -16.0, "audio.target_loudness", -16.0),
    ("volume", 0.5, "audio.volume", 0.5),
    ("super_resolution", True, "audio.super_resolution", True),
    ("super_resolution_model", "lavasr", "audio.super_resolution_model", "lavasr"),
    ("vc_reference", "target.wav", "audio.voice_conversion.audio", "target.wav"),
    ("vc_engine", "knnvc", "audio.voice_conversion_engine", "knnvc"),
]
IDS = [name for name, *_ in NAMES]


def _at(config, path):
    for part in path.split("."):
        config = getattr(config, part)
    return config


def _warning(name, path):
    return rf"SynthesisConfig\.{name} is deprecated and will be removed in phoonnx {REMOVAL}\. " \
           rf"Use SynthesisConfig\.{path}\."


@pytest.mark.parametrize("name, value, path, expected", NAMES, ids=IDS)
def test_a_1x_keyword_lands_in_its_new_home(name, value, path, expected):
    with pytest.warns(DeprecationWarning, match=_warning(name, path)):
        config = SynthesisConfig(**{name: value})
    assert _at(config, path) == expected


@pytest.mark.parametrize("name, value, path, expected", NAMES, ids=IDS)
def test_a_1x_attribute_reads_the_new_home(name, value, path, expected):
    with pytest.warns(DeprecationWarning):
        config = SynthesisConfig(**{name: value})
    with pytest.warns(DeprecationWarning, match=_warning(name, path)):
        assert getattr(config, name) == value


@pytest.mark.parametrize("name, value, path, expected", NAMES, ids=IDS)
def test_from_dict_reads_a_1x_name(name, value, path, expected):
    with pytest.warns(DeprecationWarning, match=_warning(name, path)):
        config = SynthesisConfig.from_dict({name: value})
    assert _at(config, path) == expected


def test_the_removal_release_follows_the_installed_major(monkeypatch):
    monkeypatch.setattr(phoonnx.version, "VERSION_MAJOR", 7)
    with pytest.warns(DeprecationWarning, match=r"removed in phoonnx 8\.0\.0\. Use SynthesisConfig\.speaker\."):
        SynthesisConfig(speaker_id=1)


def test_an_unset_1x_name_reads_its_1x_default():
    config = SynthesisConfig()
    with pytest.warns(DeprecationWarning):
        assert (config.normalize_audio, config.volume, config.super_resolution) == (True, 1.0, False)
    with pytest.warns(DeprecationWarning):
        assert (config.vc_reference, config.speaker_reference, config.speaker_id) == (None, None, None)


def test_the_new_names_give_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = SynthesisConfig(
            stages=Stages(diacritize=True, chunk="clause"), speaker="p225", language="pt",
            length_scale=1.1, reference=SpeakerReference(audio="me.wav", text="olá", lang="pt"),
            extra={"seed": 1}, audio=AudioPost(normalize=False, volume=0.5),
            phonetic_spellings=False, input_alphabet=Alphabet.IPA)
        assert config.stages.chunk == Chunk.CLAUSE
        assert config.reference.text == "olá"


def test_a_1x_name_joins_the_group_given_beside_it():
    with pytest.warns(DeprecationWarning):
        config = SynthesisConfig(audio=AudioPost(volume=2.0), normalize_audio=False)
    assert config.audio == AudioPost(normalize=False, volume=2.0)


def test_replace_takes_a_1x_name():
    base = SynthesisConfig(stages=Stages(chunk="clause"))
    with pytest.warns(DeprecationWarning):
        config = dataclasses.replace(base, add_diacritics=True)
    assert config.stages == Stages(diacritize=True, chunk="clause")


def test_an_unknown_keyword_is_a_type_error():
    with pytest.raises(TypeError, match="'speed'"):
        SynthesisConfig(speed=2)


def test_the_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        SynthesisConfig().speaker = 1


def test_to_dict_and_from_dict_round_trip():
    config = SynthesisConfig(
        stages=Stages(diacritize=False, diacritizer_model="rawi", chunk="paragraph", lang="pt-PT"),
        speaker=2, language="pt", noise_scale=0.5, temperature=0.7,
        reference=SpeakerReference(audio="me.wav", text="olá", lang="pt"), extra={"seed": 3},
        audio=AudioPost(target_loudness=-16.0, voice_conversion=SpeakerReference(audio="t.wav"),
                        voice_conversion_engine="knnvc"),
        phonetic_spellings=False, input_alphabet=Alphabet.IPA)
    data = config.to_dict()
    assert data["stages"]["chunk"] == "paragraph" and data["input_alphabet"] == "ipa"
    assert data["audio"]["voice_conversion"] == {"audio": "t.wav", "text": None, "lang": None}
    assert SynthesisConfig.from_dict(data) == config


def test_a_language_code_sets_the_stage_language():
    assert stages_of(SynthesisConfig(language="pt-PT")).lang == "pt-PT"
    assert stages_of(SynthesisConfig(language="pt-PT", stages=Stages(lang="en"))).lang == "en"
    assert stages_of(SynthesisConfig(language=1)).lang is None


def _voice():
    config = VoiceConfig.from_dict({
        "phoonnx_version": "1.0", "engine": "phoonnx", "phoneme_type": "unicode", "alphabet": "unicode",
        "lang_code": "pt-PT", "audio": {"sample_rate": 16000}, "num_symbols": 8, "num_speakers": 3,
        "speaker_id_map": {"ana": 0, "rui": 2}, "num_langs": 2, "lang_id_map": {"pt": 0, "en": 1},
        "phoneme_id_map": {"_": [0], "^": [1], "$": [2], "a": [3]}})
    voice = TTSVoice.__new__(TTSVoice)
    voice.config, voice.adapter, voice.session = config, None, None
    voice.adapter = type("Adapter", (), {"default_params": lambda self: {},
                                         "length_scale_params": lambda self, scale: {}})()
    return voice


def test_a_speaker_name_and_a_language_code_reach_the_engine_as_ids():
    request = _voice()._request([3], SynthesisConfig(speaker="rui", language="en"))
    assert (request.speaker_id, request.language_id) == (2, 1)
    np.testing.assert_array_equal(request.phoneme_ids, [[3]])


def test_a_speaker_id_and_a_language_id_pass_through():
    request = _voice()._request([3], SynthesisConfig(speaker=1, language=0))
    assert (request.speaker_id, request.language_id) == (1, 0)


def test_an_unknown_speaker_name_fails():
    with pytest.raises(ValueError, match="'zé'.*'ana', 'rui'"):
        _voice()._request([3], SynthesisConfig(speaker="zé"))
