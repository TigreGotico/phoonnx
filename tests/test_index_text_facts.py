"""Bundled voices load with the text path of their own config.json.

Each fixture is the config.json that the voice's index entry names, as the Hub
serves it. The index writes its text fields over the config at load, so an
index value that disagrees with a correct config gives the model the wrong
symbols. These voices load graph-free through ``TTSModelInfo.load`` and must
end with the config's language, phoneme type and alphabet.
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
    assert (loaded.lang_code, loaded.phoneme_type.value, loaded.alphabet.value) == \
           (config["lang_code"], config["phoneme_type"], config["alphabet"])
    assert loaded.engine.value == config.get("engine", "phoonnx")


@pytest.mark.parametrize("voice_id", sorted(CATALOGUE_LANG))
def test_the_catalogue_language_is_the_voice_s_own(manager, voice_id):
    assert manager.get_voice(voice_id).lang == CATALOGUE_LANG[voice_id]
