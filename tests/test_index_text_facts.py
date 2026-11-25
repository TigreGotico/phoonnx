"""Bundled voices load with the text path of their own config.json.

Each fixture is the config.json that the voice's index entry names, as the Hub
serves it. A voice with a config.json takes its phoneme type, alphabet and
phonemizer model from that file, and the index copies are catalogue data. These
voices load graph-free through ``TTSModelInfo.load`` and must end with the
config's language, phoneme type and alphabet.
"""
import json
from pathlib import Path
from unittest import mock

import pytest

import phoonnx.providers
import phoonnx.voice
from phoonnx.model_manager import TTSModelInfo, TTSModelManager

FIXTURES = Path(__file__).parent / "fixtures" / "index_text_facts"

# voice id -> config fixture, with the Hub commit it was read at
VOICES = {
    # Phoonnx/phoonnx-glowtts@b796c880 coqui-be-common-voice-glow-tts/config.json:
    # its phoneme_id_map is espeak IPA and holds no Belarusian Cyrillic letter
    "coqui-be-common-voice-glow-tts": "coqui-be-common-voice-glow-tts.json",
    # Phoonnx/phoonnx-vits@c0d756d3 hf_community__HirCoir__piper-sorah-neuronal/config.json:
    # a Piper voice with the standard Piper espeak IPA map
    "hf_community/HirCoir/piper-sorah-neuronal": "piper-sorah-neuronal.json",
    # Phoonnx/phoonnx_es-CO_dii_espeak@d4747c60 dii_es-CO.json and
    # Phoonnx/phoonnx_es-CO_miro_espeak@6a8552a6 miro_es-CO.json: trained on espeak es-419,
    # which has the seseo of Colombian Spanish; espeak reads es-CO as Castilian es
    "OpenVoiceOS/phoonnx_es-CO_dii_espeak": "phoonnx_es-CO_dii_espeak.json",
    "OpenVoiceOS/phoonnx_es-CO_miro_espeak": "phoonnx_es-CO_miro_espeak.json",
    # Phoonnx/phoonnx-vits@c0d756d3 mimic3__uk_UK__m-ailabs_low/config.json: Ukrainian of Ukraine
    "mimic3/uk_UK/m-ailabs_low": "mimic3_uk_UK_m-ailabs_low.json",
}

# The catalogue language of a voice stays the language its users ask for.
CATALOGUE_LANG = {
    "OpenVoiceOS/phoonnx_es-CO_dii_espeak": "es-CO",
    "OpenVoiceOS/phoonnx_es-CO_miro_espeak": "es-CO",
    "mimic3/uk_UK/m-ailabs_low": "uk-UA",
}


@pytest.fixture(scope="module")
def manager(tmp_path_factory):
    manager = TTSModelManager(cache_path=str(tmp_path_factory.mktemp("registry") / "voices.json"))
    manager.merge_default_voices()
    return manager


def graph_free_load(info: TTSModelInfo, config_path: Path):
    """``info.load()`` with the config read from ``config_path`` and no graph opened."""
    def hub_path(self, url):
        return config_path if url == self.config_url else None

    with mock.patch.object(TTSModelInfo, "hub_path", hub_path), \
            mock.patch.object(TTSModelInfo, "download_model", lambda self: Path("graph-not-fetched.onnx")), \
            mock.patch.object(TTSModelInfo, "engine_params", lambda self: {}), \
            mock.patch.object(phoonnx.providers, "_build_session", lambda *args, **kwargs: object()), \
            mock.patch.object(phoonnx.voice, "make_session", lambda *args, **kwargs: None):
        return info.load()


@pytest.mark.parametrize("voice_id", sorted(VOICES))
def test_the_voice_loads_with_the_text_path_of_its_config(manager, voice_id):
    config_path = FIXTURES / VOICES[voice_id]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    voice = graph_free_load(manager.get_voice(voice_id), config_path)
    loaded = voice.config
    # the config's language is the language its phonemizer was trained in
    assert (loaded.phonemizer_lang or loaded.lang_code, loaded.phoneme_type.value, loaded.alphabet.value) == \
           (config["lang_code"], config["phoneme_type"], config["alphabet"])
    assert loaded.engine.value == config.get("engine", "phoonnx")


@pytest.mark.parametrize("voice_id", ["OpenVoiceOS/phoonnx_es-CO_dii_espeak", "OpenVoiceOS/phoonnx_es-CO_miro_espeak"])
def test_an_es_co_voice_reads_its_text_as_es_co_on_an_es_419_phonemizer(manager, voice_id):
    loaded = graph_free_load(manager.get_voice(voice_id), FIXTURES / VOICES[voice_id]).config
    assert (loaded.lang_code, loaded.phonemizer_lang) == ("es-CO", "es-419")


@pytest.mark.parametrize("voice_id", sorted(CATALOGUE_LANG))
def test_the_catalogue_language_is_the_voice_s_own(manager, voice_id):
    assert manager.get_voice(voice_id).lang == CATALOGUE_LANG[voice_id]


# a config.json that states its text path, and an index entry that lists another one
STATED = {"engine": "phoonnx", "phoneme_type": "espeak", "alphabet": "ipa", "lang_code": "be",
          "phoneme_id_map": {"_": [0], "^": [1], "$": [2], " ": [3], "a": [4]}, "num_symbols": 5}


def test_the_config_json_text_path_wins_over_the_index():
    info = TTSModelInfo(voice_id="test/voice", lang="be", engine="vits", model_url="https://x/m.onnx",
                        config_url="https://x/config.json", phoneme_type="graphemes", alphabet="unicode",
                        phonemizer_model="variant")
    with mock.patch.object(TTSModelInfo, "download_config", return_value=dict(STATED)), \
            mock.patch("phoonnx.model_manager.LOG") as log:
        loaded = info.config
    assert (loaded.phoneme_type.value, loaded.alphabet.value, loaded.phonemizer_model) == ("espeak", "ipa", None)
    warnings = " | ".join(call.args[0] for call in log.warning.call_args_list)
    assert "lists phoneme_type graphemes, config.json gives espeak" in warnings
    assert "lists alphabet unicode, config.json gives ipa" in warnings


def test_a_voice_with_no_config_json_reads_its_text_path_from_the_index():
    info = TTSModelInfo(voice_id="test/voice", lang="be", engine="vits", model_url="https://x/m.onnx",
                        phoneme_type="espeak", alphabet="ipa", phonemizer_model="variant")
    loaded = info.config
    assert (loaded.phoneme_type.value, loaded.alphabet.value, loaded.phonemizer_model) == ("espeak", "ipa", "variant")
