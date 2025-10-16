import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Iterable, List

import requests
from json_database import JsonStorageXDG, JsonStorage
from ovos_utils.lang import standardize_lang_tag
from ovos_utils.xdg_utils import xdg_cache_home

from phoonnx.config import PhonemeType, get_phonemizer, VoiceConfig, Engine, Alphabet
from phoonnx.util import match_lang
from phoonnx.voice import TTSVoice


@dataclass
class TTSModelInfo:
    voice_id: str
    lang: str  # not always present in config.json and often wrong if present
    model_url: str
    config_url: str
    tokens_url: Optional[str] = None  # mimic3/sherpa provide phoneme_map in this format
    phoneme_map_url: Optional[str] = None  # json lookup table for phoneme replacement
    config: Optional[VoiceConfig] = None
    phoneme_type: Optional[PhonemeType] = None

    def __post_init__(self):
        os.makedirs(self.voice_path, exist_ok=True)
        if not self.config:
            config_path = self.voice_path / "model.json"
            if not config_path.is_file():
                self.download_config()
            with open(config_path, "r") as f:
                config = json.load(f)

            # HACK: seen in some published piper voices
            # "es_MX-ald-medium"
            if config.get('phoneme_type', "") == "PhonemeType.ESPEAK":
                config["phoneme_type"] = "espeak"
            #####
            if self.tokens_url:
                self.download_phoneme_map()
                self.config = VoiceConfig.from_dict(config, phonemes_txt=str(self.voice_path / "tokens.txt"))
            else:
                self.config = VoiceConfig.from_dict(config)

            self.config.lang_code = self.lang  # sometimes the config is wrong

        if not self.phoneme_type:
            self.phoneme_type = self.config.phoneme_type
        else:
            self.config.phoneme_type = self.phoneme_type

    @property
    def alphabet(self) -> Alphabet:
        return self.config.alphabet

    @property
    def engine(self) -> Engine:
        return self.config.engine

    @property
    def voice_path(self) -> Path:
        return xdg_cache_home() / "phoonnx" / "voices" / self.voice_id

    def download_config(self):
        config_path = self.voice_path / "model.json"
        if not config_path.is_file():
            r = requests.get(self.config_url)
            cfg = r.json()  # validate received json
            with open(config_path, "w") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=4)

    def download_phoneme_map(self):
        tokens_path = self.voice_path / "tokens.txt"
        if self.tokens_url and not tokens_path.is_file():
            tokens = requests.get(self.tokens_url).text
            with open(tokens_path, "w") as f:
                f.write(tokens)

    def download_model(self):
        model_path = self.voice_path / "model.onnx"
        if not model_path.is_file():
            r = requests.get(self.model_url)
            with open(model_path, "wb") as f:
                f.write(r.content)

    def load(self) -> TTSVoice:
        model_path = self.voice_path / "model.onnx"
        config_path = self.voice_path / "model.json"
        tokens_path = self.voice_path / "tokens.txt"
        self.download_model()

        voice = TTSVoice.load(model_path=model_path,
                              config_path=config_path,
                              phonemes_txt=str(tokens_path) if self.tokens_url else None)

        # override phoneme_type, if config.json is wrong
        voice.phoneme_type = self.phoneme_type
        voice.phonemizer = get_phonemizer(self.phoneme_type)
        return voice


class TTSModelManager:
    def __init__(self, cache_path: Optional[str] = None):
        self.voices: Dict[str, TTSModelInfo] = {}
        if cache_path:
            self.cache = JsonStorage(cache_path)
        else:
            self.cache = JsonStorageXDG("voices", subfolder="phoonnx")

    def clear(self):
        self.cache.clear()
        self.voices = {}

    def load(self):
        self.cache.reload()
        self.voices = {voice_id: TTSModelInfo(**voice_dict)
                       for voice_id, voice_dict in self.cache.items()}

    def save(self):
        self.cache.clear()
        for voice_id, voice_info in self.voices.items():
            self.cache[voice_id] = {"voice_id": voice_info.voice_id,
                                    "model_url": voice_info.model_url,
                                    "phoneme_type": voice_info.phoneme_type,
                                    "lang": voice_info.lang,
                                    "tokens_url": voice_info.tokens_url,
                                    "phoneme_map_url": voice_info.phoneme_map_url,
                                    "config_url": voice_info.config_url}
        self.cache.store()

    def add_voice(self, voice_info: TTSModelInfo):
        self.voices[voice_info.voice_id] = voice_info
        self.cache[voice_info.voice_id] = {"voice_id": voice_info.voice_id,
                                           "model_url": voice_info.model_url,
                                           "tokens_url": voice_info.tokens_url,
                                           "phoneme_type": voice_info.phoneme_type,
                                           "phoneme_map_url": voice_info.phoneme_map_url,
                                           "lang": voice_info.lang,
                                           "config_url": voice_info.config_url}

    def get_lang_voices(self, lang: str) -> List[TTSModelInfo]:
        voices = sorted(
            [
                (voice_info, match_lang(voice_info.lang, lang)[-1])
                for voice_info in self.voices.values()
            ], key=lambda k: k[1])
        return [v[0] for v in voices if v[1] < 10]

    # helpers to get official voice models
    def get_proxectonos_voice_list(self):
        self.add_voice(TTSModelInfo(
            voice_id="proxectonos/sabela",
            lang="gl-ES",
            model_url="https://huggingface.co/OpenVoiceOS/proxectonos-sabela-vits-phonemes-onnx/resolve/main/model.onnx",
            config_url="https://huggingface.co/OpenVoiceOS/proxectonos-sabela-vits-phonemes-onnx/resolve/main/config.json",
            phoneme_type=PhonemeType.COTOVIA
        ))
        self.add_voice(TTSModelInfo(
            voice_id="proxectonos/celtia",
            lang="gl-ES",
            model_url="https://huggingface.co/OpenVoiceOS/proxectonos-celtia-vits-graphemes-onnx/resolve/main/model.onnx",
            config_url="https://huggingface.co/OpenVoiceOS/proxectonos-celtia-vits-graphemes-onnx/resolve/main/config.json"
        ))

    def get_piper_voice_list(self):
        base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        voice_list = "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json"
        piper_voices = requests.get(voice_list).json()

        for v in piper_voices.values():
            try:
                voice = TTSModelInfo(
                    voice_id="piper_" + v["key"],
                    lang=standardize_lang_tag(v["key"].split("-")[0]),
                    model_url=base + [a for a in v["files"] if a.endswith(".onnx")][0],
                    config_url=base + [a for a in v["files"] if a.endswith(".json")][0],
                )
                self.add_voice(voice)
            except Exception:
                print(f"Failed to get voice info for {v['key']}")

    def get_mimic3_voice_list(self):
        voice_list = "https://raw.githubusercontent.com/MycroftAI/mimic3/refs/heads/master/mimic3_tts/voices.json"
        mimic3_voices = requests.get(voice_list).json()
        for k, v in mimic3_voices.items():
            try:
                lang = standardize_lang_tag(k.split("/")[0])
                speaker_map = {s: idx for idx, s in enumerate(v["speakers"])}
                config_url = f"https://huggingface.co/mukowaty/mimic3-voices/resolve/main/voices/{k}/config.json"
                model_url = f"https://huggingface.co/mukowaty/mimic3-voices/resolve/main/voices/{k}/generator.onnx"
                tokens_url = f"https://huggingface.co/mukowaty/mimic3-voices/resolve/main/voices/{k}/phonemes.txt"
                phoneme_map_url = f"https://huggingface.co/mukowaty/mimic3-voices/resolve/main/voices/{k}/phoneme_map.txt"
                voice_info = TTSModelInfo(
                    voice_id="mimic3_" + k,
                    lang=lang,
                    config_url=config_url,
                    tokens_url=tokens_url,
                    model_url=model_url,
                    phoneme_map_url=phoneme_map_url
                )
                voice_info.config.lang = lang
                voice_info.config.speaker_id_map = speaker_map
                self.add_voice(voice_info)
            except Exception:
                print(f"Failed to get voice info for {k}")

    def get_ovos_voice_list(self):
        phoonnx = [
            "OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone",
            "OpenVoiceOS/phoonnx_pt-PT_dii_tugaphone",
            "OpenVoiceOS/phoonnx_eu-ES_miro_espeak",
            "OpenVoiceOS/phoonnx_eu-ES_dii_espeak",
            "OpenVoiceOS/phoonnx_ar-SA_miro_espeak_V2",
            "OpenVoiceOS/phoonnx_ar-SA_dii_espeak",
            "OpenVoiceOS/phoonnx_sv-SE_miro_espeak",
            "OpenVoiceOS/phoonnx_da-DK_miro_espeak",
            "OpenVoiceOS/phoonnx_es-ES_dii_espeak"
        ]
        for repo in phoonnx:
            lang = repo.split("phoonnx_")[-1].split("_")[0]
            voice = f"miro_{lang}" if "miro" in repo else f"dii_{lang}"
            self.add_voice(TTSModelInfo(
                lang=lang,
                voice_id=repo,
                model_url=f"https://huggingface.co/{repo}/resolve/main/{voice}.onnx",
                config_url=f"https://huggingface.co/{repo}/resolve/main/{voice}.json",
            ))

        piper_ovos = [
            "en-GB", "pt-BR", "pt-PT", "es-ES", "it-IT",
            "nl-NL", "de-DE", "fr-FR", "en-US"
        ]
        for lang in piper_ovos:
            for voice in ["miro", "dii"]:
                repo = f"OpenVoiceOS/pipertts_{lang}_{voice}"
                try:
                    self.add_voice(TTSModelInfo(
                        lang=lang,
                        voice_id=repo,
                        model_url=f"https://huggingface.co/{repo}/resolve/main/{voice}_{lang}.onnx",
                        config_url=f"https://huggingface.co/{repo}/resolve/main/{voice}_{lang}.onnx.json",
                    ))
                except Exception:
                    continue  # not all langs have male + female

    def get_phonikud_voice_list(self):
        self.add_voice(
            TTSModelInfo(
                voice_id="thewh1teagle/phonikud",
                lang="he",
                model_url="https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.onnx",
                config_url="https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.config.json",
                phoneme_type=PhonemeType.PHONIKUD
            ))

    @property
    def all_voices(self) -> Iterable[TTSModelInfo]:
        return self.voices.values()

    @property
    def supported_langs(self) -> Iterable[str]:
        return sorted(set(l.lang for l in self.all_voices))


if __name__ == "__main__":
    manager = TTSModelManager()
    manager.clear()
    # manager.load()
    manager.get_ovos_voice_list()
    manager.get_proxectonos_voice_list()
    manager.get_phonikud_voice_list()
    manager.get_piper_voice_list()
    manager.get_mimic3_voice_list()
    manager.save()

    print(f"Total voices: {len(manager.all_voices)}")
    print(f"Total langs: {len(manager.supported_langs)}")

    # Total voices: 214
    # Total langs: 60

    for voice in manager.get_lang_voices('pt-PT'):
        print(voice)

    print(manager.supported_langs)
    # ['af-ZA', 'ar-JO', 'bn', 'ca-ES', 'cs-CZ', 'cy-GB', 'da-DK', 'de-DE', 'el-GR', 'en-GB', 'en-US', 'es-AR',
    # 'es-ES', 'es-MX', 'fa', 'fa-IR', 'fi-FI', 'fr-FR', 'gl-ES', 'gu-IN', 'ha-NE', 'hi-IN', 'hu-HU', 'id-ID', 'is-IS',
    # 'it-IT', 'jv-ID', 'ka-GE', 'kk-KZ', 'ko-KO', 'lb-LU', 'lv-LV', 'ml-IN', 'ne-NP', 'nl', 'nl-BE', 'nl-NL',
    # 'no-NO', 'pl-PL', 'pt-BR', 'pt-PT', 'ro-RO', 'ru-RU', 'sk-SK', 'sl-SI', 'sr-RS', 'sv-SE', 'sw', 'sw-CD',
    # 'te-IN', 'tn-ZA', 'tr-TR', 'uk-GB', 'uk-UA', 'vi-VN', 'yo', 'zh-CN']
