import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import requests
from json_database import JsonStorageXDG, JsonStorage
from langcodes import standardize_tag

from phoonnx.config import PhonemeType, get_phonemizer, VoiceConfig, Engine, Alphabet
from phoonnx.util import LOG
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
        return Path(os.path.expanduser("~")) / ".cache" / "phoonnx" / "voices" / self.voice_id

    def download_config(self):
        config_path = self.voice_path / "model.json"
        if not config_path.is_file():
            r = requests.get(self.config_url, timeout=30)
            r.raise_for_status()
            cfg = r.json()  # validate received json
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=4)

    def download_phoneme_map(self):
        tokens_path = self.voice_path / "tokens.txt"
        if self.tokens_url and not tokens_path.is_file():
            r = requests.get(self.tokens_url, timeout=30)
            r.raise_for_status()
            tokens = r.text
            with open(tokens_path, "w", encoding="utf-8") as f:
                f.write(tokens)

    def download_model(self):
        model_path = self.voice_path / "model.onnx"
        if not model_path.is_file():
            with requests.get(self.model_url, timeout=120, stream=True) as r:
                r.raise_for_status()
                with open(model_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

    def load(self) -> TTSVoice:
        model_path = self.voice_path / "model.onnx"
        config_path = self.voice_path / "model.json"
        tokens_path = self.voice_path / "tokens.txt"
        self.download_model()

        voice = TTSVoice.load(model_path=model_path,
                              config_path=config_path,
                              phonemes_txt=str(tokens_path) if self.tokens_url else None)

        # override phoneme_type, if config.json is wrong
        if self.phoneme_type != voice.config.phoneme_type:
            voice.phoneme_type = self.phoneme_type
            voice.phonemizer = get_phonemizer(self.phoneme_type,
                                              alphabet=voice.config.alphabet,
                                              model=voice.config.phonemizer_model)
        return voice


class TTSModelManager:
    def __init__(self, cache_path: Optional[str] = None):
        self.voices: Dict[str, TTSModelInfo] = {}
        if cache_path:
            self.cache = JsonStorage(cache_path)
        else:
            self.cache = JsonStorageXDG("voices", subfolder="phoonnx")

    @property
    def all_voices(self) -> List[TTSModelInfo]:
        return list(self.voices.values())

    @property
    def supported_langs(self) -> List[str]:
        return sorted(set(l.lang for l in self.all_voices))

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

    def refresh_voices(self):
        self.get_ovos_voice_list()
        self.get_proxectonos_voice_list()
        self.get_phonikud_voice_list()
        self.get_piper_voice_list()
        self.get_mimic3_voice_list()
        self.cache.store()

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
                    lang=standardize_tag(v["key"].split("-")[0]),
                    model_url=base + [a for a in v["files"] if a.endswith(".onnx")][0],
                    config_url=base + [a for a in v["files"] if a.endswith(".json")][0],
                )
                self.add_voice(voice)
            except Exception:
                print(f"Failed to get voice info for {v['key']}")

    def get_mimic3_voice_list(self):
        voice_list = "https://raw.githubusercontent.com/MycroftAI/mimic3/refs/heads/master/mimic3_tts/voices.json"
        r = requests.get(voice_list, timeout=30)
        r.raise_for_status()
        mimic3_voices = r.json()
        for k, v in mimic3_voices.items():
            try:
                lang = standardize_tag(k.split("/")[0])
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
            except Exception as e:
                LOG.error(f"Failed to get voice info for {k}: {e}")

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


if __name__ == "__main__":
    manager = TTSModelManager()
    manager.clear()
    # manager.load()
    manager.refresh_voices()
    manager.save()

    print(f"Total voices: {len(manager.all_voices)}")
    print(f"Total langs: {len(manager.supported_langs)}")

    # Total voices: 214
    # Total langs: 60

    for voice in manager.get_lang_voices('pt-PT'):
        print(voice)
    # TTSModelInfo(voice_id='OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone', lang='pt-PT', model_url='https://huggingface.co/OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone/resolve/main/miro_pt-PT.onnx', config_url='https://huggingface.co/OpenVoiceOS/phoonnx_pt-PT_miro_tugaphone/resolve/main/miro_pt-PT.json', tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=1, num_langs=1, sample_rate=22050, lang_code='pt-PT', phoneme_id_map={' ': 3, '!': 4, '"': 150, '#': 149, '$': 2, "'": 5, '(': 6, ')': 7, ',': 8, '-': 9, '.': 10, '0': 130, '1': 131, '2': 132, '3': 133, '4': 134, '5': 135, '6': 136, '7': 137, '8': 138, '9': 139, ':': 11, ';': 12, '?': 13, 'X': 156, '^': 1, '_': 0, 'a': 14, 'b': 15, 'c': 16, 'd': 17, 'e': 18, 'f': 19, 'g': 154, 'h': 20, 'i': 21, 'j': 22, 'k': 23, 'l': 24, 'm': 25, 'n': 26, 'o': 27, 'p': 28, 'q': 29, 'r': 30, 's': 31, 't': 32, 'u': 33, 'v': 34, 'w': 35, 'x': 36, 'y': 37, 'z': 38, 'æ': 39, 'ç': 40, 'ð': 41, 'ø': 42, 'ħ': 43, 'ŋ': 44, 'œ': 45, 'ǀ': 46, 'ǁ': 47, 'ǂ': 48, 'ǃ': 49, 'ɐ': 50, 'ɑ': 51, 'ɒ': 52, 'ɓ': 53, 'ɔ': 54, 'ɕ': 55, 'ɖ': 56, 'ɗ': 57, 'ɘ': 58, 'ə': 59, 'ɚ': 60, 'ɛ': 61, 'ɜ': 62, 'ɞ': 63, 'ɟ': 64, 'ɠ': 65, 'ɡ': 66, 'ɢ': 67, 'ɣ': 68, 'ɤ': 69, 'ɥ': 70, 'ɦ': 71, 'ɧ': 72, 'ɨ': 73, 'ɪ': 74, 'ɫ': 75, 'ɬ': 76, 'ɭ': 77, 'ɮ': 78, 'ɯ': 79, 'ɰ': 80, 'ɱ': 81, 'ɲ': 82, 'ɳ': 83, 'ɴ': 84, 'ɵ': 85, 'ɶ': 86, 'ɸ': 87, 'ɹ': 88, 'ɺ': 89, 'ɻ': 90, 'ɽ': 91, 'ɾ': 92, 'ʀ': 93, 'ʁ': 94, 'ʂ': 95, 'ʃ': 96, 'ʄ': 97, 'ʈ': 98, 'ʉ': 99, 'ʊ': 100, 'ʋ': 101, 'ʌ': 102, 'ʍ': 103, 'ʎ': 104, 'ʏ': 105, 'ʐ': 106, 'ʑ': 107, 'ʒ': 108, 'ʔ': 109, 'ʕ': 110, 'ʘ': 111, 'ʙ': 112, 'ʛ': 113, 'ʜ': 114, 'ʝ': 115, 'ʟ': 116, 'ʡ': 117, 'ʢ': 118, 'ʦ': 155, 'ʰ': 145, 'ʲ': 119, 'ˈ': 120, 'ˌ': 121, 'ː': 122, 'ˑ': 123, '˞': 124, 'ˤ': 146, '̃': 141, '̊': 158, '̝': 157, '̧': 140, '̩': 144, '̪': 142, '̯': 143, '̺': 152, '̻': 153, 'β': 125, 'ε': 147, 'θ': 126, 'χ': 127, 'ᵻ': 128, '↑': 151, '↓': 148, 'ⱱ': 129}, phoneme_type=<PhonemeType.TUGAPHONE: 'tugaphone'>, alphabet='ipa', phonemizer_model='', speaker_id_map={}, lang_id_map={}, engine=<Engine.PHOONNX: 'phoonnx'>, length_scale=1, noise_scale=0.667, noise_w_scale=0.8, blank_at_start=True, blank_at_end=True, include_whitespace=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.TUGAPHONE: 'tugaphone'>)
    # TTSModelInfo(voice_id='OpenVoiceOS/phoonnx_pt-PT_dii_tugaphone', lang='pt-PT', model_url='https://huggingface.co/OpenVoiceOS/phoonnx_pt-PT_dii_tugaphone/resolve/main/dii_pt-PT.onnx', config_url='https://huggingface.co/OpenVoiceOS/phoonnx_pt-PT_dii_tugaphone/resolve/main/dii_pt-PT.json', tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=1, num_langs=1, sample_rate=22050, lang_code='pt-PT', phoneme_id_map={' ': 3, '!': 4, '"': 150, '#': 149, '$': 2, "'": 5, '(': 6, ')': 7, ',': 8, '-': 9, '.': 10, '0': 130, '1': 131, '2': 132, '3': 133, '4': 134, '5': 135, '6': 136, '7': 137, '8': 138, '9': 139, ':': 11, ';': 12, '?': 13, 'X': 156, '^': 1, '_': 0, 'a': 14, 'b': 15, 'c': 16, 'd': 17, 'e': 18, 'f': 19, 'g': 154, 'h': 20, 'i': 21, 'j': 22, 'k': 23, 'l': 24, 'm': 25, 'n': 26, 'o': 27, 'p': 28, 'q': 29, 'r': 30, 's': 31, 't': 32, 'u': 33, 'v': 34, 'w': 35, 'x': 36, 'y': 37, 'z': 38, 'æ': 39, 'ç': 40, 'ð': 41, 'ø': 42, 'ħ': 43, 'ŋ': 44, 'œ': 45, 'ǀ': 46, 'ǁ': 47, 'ǂ': 48, 'ǃ': 49, 'ɐ': 50, 'ɑ': 51, 'ɒ': 52, 'ɓ': 53, 'ɔ': 54, 'ɕ': 55, 'ɖ': 56, 'ɗ': 57, 'ɘ': 58, 'ə': 59, 'ɚ': 60, 'ɛ': 61, 'ɜ': 62, 'ɞ': 63, 'ɟ': 64, 'ɠ': 65, 'ɡ': 66, 'ɢ': 67, 'ɣ': 68, 'ɤ': 69, 'ɥ': 70, 'ɦ': 71, 'ɧ': 72, 'ɨ': 73, 'ɪ': 74, 'ɫ': 75, 'ɬ': 76, 'ɭ': 77, 'ɮ': 78, 'ɯ': 79, 'ɰ': 80, 'ɱ': 81, 'ɲ': 82, 'ɳ': 83, 'ɴ': 84, 'ɵ': 85, 'ɶ': 86, 'ɸ': 87, 'ɹ': 88, 'ɺ': 89, 'ɻ': 90, 'ɽ': 91, 'ɾ': 92, 'ʀ': 93, 'ʁ': 94, 'ʂ': 95, 'ʃ': 96, 'ʄ': 97, 'ʈ': 98, 'ʉ': 99, 'ʊ': 100, 'ʋ': 101, 'ʌ': 102, 'ʍ': 103, 'ʎ': 104, 'ʏ': 105, 'ʐ': 106, 'ʑ': 107, 'ʒ': 108, 'ʔ': 109, 'ʕ': 110, 'ʘ': 111, 'ʙ': 112, 'ʛ': 113, 'ʜ': 114, 'ʝ': 115, 'ʟ': 116, 'ʡ': 117, 'ʢ': 118, 'ʦ': 155, 'ʰ': 145, 'ʲ': 119, 'ˈ': 120, 'ˌ': 121, 'ː': 122, 'ˑ': 123, '˞': 124, 'ˤ': 146, '̃': 141, '̊': 158, '̝': 157, '̧': 140, '̩': 144, '̪': 142, '̯': 143, '̺': 152, '̻': 153, 'β': 125, 'ε': 147, 'θ': 126, 'χ': 127, 'ᵻ': 128, '↑': 151, '↓': 148, 'ⱱ': 129}, phoneme_type=<PhonemeType.TUGAPHONE: 'tugaphone'>, alphabet='ipa', phonemizer_model='', speaker_id_map={}, lang_id_map={}, engine=<Engine.PHOONNX: 'phoonnx'>, length_scale=1, noise_scale=0.667, noise_w_scale=0.8, blank_at_start=True, blank_at_end=True, include_whitespace=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.TUGAPHONE: 'tugaphone'>)
    # TTSModelInfo(voice_id='OpenVoiceOS/pipertts_pt-PT_miro', lang='pt-PT', model_url='https://huggingface.co/OpenVoiceOS/pipertts_pt-PT_miro/resolve/main/miro_pt-PT.onnx', config_url='https://huggingface.co/OpenVoiceOS/pipertts_pt-PT_miro/resolve/main/miro_pt-PT.onnx.json', tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=1, num_langs=1, sample_rate=22050, lang_code='pt-PT', phoneme_id_map={' ': [3], '!': [4], '"': [150], '#': [149], '$': [2], "'": [5], '(': [6], ')': [7], ',': [8], '-': [9], '.': [10], '0': [130], '1': [131], '2': [132], '3': [133], '4': [134], '5': [135], '6': [136], '7': [137], '8': [138], '9': [139], ':': [11], ';': [12], '?': [13], 'X': [156], '^': [1], '_': [0], 'a': [14], 'b': [15], 'c': [16], 'd': [17], 'e': [18], 'f': [19], 'g': [154], 'h': [20], 'i': [21], 'j': [22], 'k': [23], 'l': [24], 'm': [25], 'n': [26], 'o': [27], 'p': [28], 'q': [29], 'r': [30], 's': [31], 't': [32], 'u': [33], 'v': [34], 'w': [35], 'x': [36], 'y': [37], 'z': [38], 'æ': [39], 'ç': [40], 'ð': [41], 'ø': [42], 'ħ': [43], 'ŋ': [44], 'œ': [45], 'ǀ': [46], 'ǁ': [47], 'ǂ': [48], 'ǃ': [49], 'ɐ': [50], 'ɑ': [51], 'ɒ': [52], 'ɓ': [53], 'ɔ': [54], 'ɕ': [55], 'ɖ': [56], 'ɗ': [57], 'ɘ': [58], 'ə': [59], 'ɚ': [60], 'ɛ': [61], 'ɜ': [62], 'ɞ': [63], 'ɟ': [64], 'ɠ': [65], 'ɡ': [66], 'ɢ': [67], 'ɣ': [68], 'ɤ': [69], 'ɥ': [70], 'ɦ': [71], 'ɧ': [72], 'ɨ': [73], 'ɪ': [74], 'ɫ': [75], 'ɬ': [76], 'ɭ': [77], 'ɮ': [78], 'ɯ': [79], 'ɰ': [80], 'ɱ': [81], 'ɲ': [82], 'ɳ': [83], 'ɴ': [84], 'ɵ': [85], 'ɶ': [86], 'ɸ': [87], 'ɹ': [88], 'ɺ': [89], 'ɻ': [90], 'ɽ': [91], 'ɾ': [92], 'ʀ': [93], 'ʁ': [94], 'ʂ': [95], 'ʃ': [96], 'ʄ': [97], 'ʈ': [98], 'ʉ': [99], 'ʊ': [100], 'ʋ': [101], 'ʌ': [102], 'ʍ': [103], 'ʎ': [104], 'ʏ': [105], 'ʐ': [106], 'ʑ': [107], 'ʒ': [108], 'ʔ': [109], 'ʕ': [110], 'ʘ': [111], 'ʙ': [112], 'ʛ': [113], 'ʜ': [114], 'ʝ': [115], 'ʟ': [116], 'ʡ': [117], 'ʢ': [118], 'ʦ': [155], 'ʰ': [145], 'ʲ': [119], 'ˈ': [120], 'ˌ': [121], 'ː': [122], 'ˑ': [123], '˞': [124], 'ˤ': [146], '̃': [141], '̊': [158], '̝': [157], '̧': [140], '̩': [144], '̪': [142], '̯': [143], '̺': [152], '̻': [153], 'β': [125], 'ε': [147], 'θ': [126], 'χ': [127], 'ᵻ': [128], '↑': [151], '↓': [148], 'ⱱ': [129]}, phoneme_type=<PhonemeType.ESPEAK: 'espeak'>, alphabet=<Alphabet.IPA: 'ipa'>, phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine=<Engine.PIPER: 'piper'>, length_scale=1, noise_scale=0.667, noise_w_scale=0.8, blank_at_start=True, blank_at_end=True, include_whitespace=True, pad_token='_', blank_token='_', bos_token='^', eos_token='$', word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.ESPEAK: 'espeak'>)
    # TTSModelInfo(voice_id='OpenVoiceOS/pipertts_pt-PT_dii', lang='pt-PT', model_url='https://huggingface.co/OpenVoiceOS/pipertts_pt-PT_dii/resolve/main/dii_pt-PT.onnx', config_url='https://huggingface.co/OpenVoiceOS/pipertts_pt-PT_dii/resolve/main/dii_pt-PT.onnx.json', tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=1, num_langs=1, sample_rate=22050, lang_code='pt-PT', phoneme_id_map={' ': [3], '!': [4], '"': [150], '#': [149], '$': [2], "'": [5], '(': [6], ')': [7], ',': [8], '-': [9], '.': [10], '0': [130], '1': [131], '2': [132], '3': [133], '4': [134], '5': [135], '6': [136], '7': [137], '8': [138], '9': [139], ':': [11], ';': [12], '?': [13], 'X': [156], '^': [1], '_': [0], 'a': [14], 'b': [15], 'c': [16], 'd': [17], 'e': [18], 'f': [19], 'g': [154], 'h': [20], 'i': [21], 'j': [22], 'k': [23], 'l': [24], 'm': [25], 'n': [26], 'o': [27], 'p': [28], 'q': [29], 'r': [30], 's': [31], 't': [32], 'u': [33], 'v': [34], 'w': [35], 'x': [36], 'y': [37], 'z': [38], 'æ': [39], 'ç': [40], 'ð': [41], 'ø': [42], 'ħ': [43], 'ŋ': [44], 'œ': [45], 'ǀ': [46], 'ǁ': [47], 'ǂ': [48], 'ǃ': [49], 'ɐ': [50], 'ɑ': [51], 'ɒ': [52], 'ɓ': [53], 'ɔ': [54], 'ɕ': [55], 'ɖ': [56], 'ɗ': [57], 'ɘ': [58], 'ə': [59], 'ɚ': [60], 'ɛ': [61], 'ɜ': [62], 'ɞ': [63], 'ɟ': [64], 'ɠ': [65], 'ɡ': [66], 'ɢ': [67], 'ɣ': [68], 'ɤ': [69], 'ɥ': [70], 'ɦ': [71], 'ɧ': [72], 'ɨ': [73], 'ɪ': [74], 'ɫ': [75], 'ɬ': [76], 'ɭ': [77], 'ɮ': [78], 'ɯ': [79], 'ɰ': [80], 'ɱ': [81], 'ɲ': [82], 'ɳ': [83], 'ɴ': [84], 'ɵ': [85], 'ɶ': [86], 'ɸ': [87], 'ɹ': [88], 'ɺ': [89], 'ɻ': [90], 'ɽ': [91], 'ɾ': [92], 'ʀ': [93], 'ʁ': [94], 'ʂ': [95], 'ʃ': [96], 'ʄ': [97], 'ʈ': [98], 'ʉ': [99], 'ʊ': [100], 'ʋ': [101], 'ʌ': [102], 'ʍ': [103], 'ʎ': [104], 'ʏ': [105], 'ʐ': [106], 'ʑ': [107], 'ʒ': [108], 'ʔ': [109], 'ʕ': [110], 'ʘ': [111], 'ʙ': [112], 'ʛ': [113], 'ʜ': [114], 'ʝ': [115], 'ʟ': [116], 'ʡ': [117], 'ʢ': [118], 'ʦ': [155], 'ʰ': [145], 'ʲ': [119], 'ˈ': [120], 'ˌ': [121], 'ː': [122], 'ˑ': [123], '˞': [124], 'ˤ': [146], '̃': [141], '̊': [158], '̝': [157], '̧': [140], '̩': [144], '̪': [142], '̯': [143], '̺': [152], '̻': [153], 'β': [125], 'ε': [147], 'θ': [126], 'χ': [127], 'ᵻ': [128], '↑': [151], '↓': [148], 'ⱱ': [129]}, phoneme_type=<PhonemeType.ESPEAK: 'espeak'>, alphabet=<Alphabet.IPA: 'ipa'>, phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine=<Engine.PIPER: 'piper'>, length_scale=1, noise_scale=0.667, noise_w_scale=0.8, blank_at_start=True, blank_at_end=True, include_whitespace=True, pad_token='_', blank_token='_', bos_token='^', eos_token='$', word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.ESPEAK: 'espeak'>)
    # TTSModelInfo(voice_id='piper_pt_PT-tugão-medium', lang='pt-PT', model_url='https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_PT/tugão/medium/pt_PT-tugão-medium.onnx', config_url='https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_PT/tugão/medium/pt_PT-tugão-medium.onnx.json', tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=1, num_langs=1, sample_rate=22050, lang_code='pt-PT', phoneme_id_map={' ': [3], '!': [4], '"': [150], '#': [149], '$': [2], "'": [5], '(': [6], ')': [7], ',': [8], '-': [9], '.': [10], '0': [130], '1': [131], '2': [132], '3': [133], '4': [134], '5': [135], '6': [136], '7': [137], '8': [138], '9': [139], ':': [11], ';': [12], '?': [13], 'X': [156], '^': [1], '_': [0], 'a': [14], 'b': [15], 'c': [16], 'd': [17], 'e': [18], 'f': [19], 'g': [154], 'h': [20], 'i': [21], 'j': [22], 'k': [23], 'l': [24], 'm': [25], 'n': [26], 'o': [27], 'p': [28], 'q': [29], 'r': [30], 's': [31], 't': [32], 'u': [33], 'v': [34], 'w': [35], 'x': [36], 'y': [37], 'z': [38], 'æ': [39], 'ç': [40], 'ð': [41], 'ø': [42], 'ħ': [43], 'ŋ': [44], 'œ': [45], 'ǀ': [46], 'ǁ': [47], 'ǂ': [48], 'ǃ': [49], 'ɐ': [50], 'ɑ': [51], 'ɒ': [52], 'ɓ': [53], 'ɔ': [54], 'ɕ': [55], 'ɖ': [56], 'ɗ': [57], 'ɘ': [58], 'ə': [59], 'ɚ': [60], 'ɛ': [61], 'ɜ': [62], 'ɞ': [63], 'ɟ': [64], 'ɠ': [65], 'ɡ': [66], 'ɢ': [67], 'ɣ': [68], 'ɤ': [69], 'ɥ': [70], 'ɦ': [71], 'ɧ': [72], 'ɨ': [73], 'ɪ': [74], 'ɫ': [75], 'ɬ': [76], 'ɭ': [77], 'ɮ': [78], 'ɯ': [79], 'ɰ': [80], 'ɱ': [81], 'ɲ': [82], 'ɳ': [83], 'ɴ': [84], 'ɵ': [85], 'ɶ': [86], 'ɸ': [87], 'ɹ': [88], 'ɺ': [89], 'ɻ': [90], 'ɽ': [91], 'ɾ': [92], 'ʀ': [93], 'ʁ': [94], 'ʂ': [95], 'ʃ': [96], 'ʄ': [97], 'ʈ': [98], 'ʉ': [99], 'ʊ': [100], 'ʋ': [101], 'ʌ': [102], 'ʍ': [103], 'ʎ': [104], 'ʏ': [105], 'ʐ': [106], 'ʑ': [107], 'ʒ': [108], 'ʔ': [109], 'ʕ': [110], 'ʘ': [111], 'ʙ': [112], 'ʛ': [113], 'ʜ': [114], 'ʝ': [115], 'ʟ': [116], 'ʡ': [117], 'ʢ': [118], 'ʦ': [155], 'ʰ': [145], 'ʲ': [119], 'ˈ': [120], 'ˌ': [121], 'ː': [122], 'ˑ': [123], '˞': [124], 'ˤ': [146], '̃': [141], '̊': [158], '̝': [157], '̧': [140], '̩': [144], '̪': [142], '̯': [143], '̺': [152], '̻': [153], 'β': [125], 'ε': [147], 'θ': [126], 'χ': [127], 'ᵻ': [128], '↑': [151], '↓': [148], 'ⱱ': [129]}, phoneme_type=<PhonemeType.ESPEAK: 'espeak'>, alphabet=<Alphabet.IPA: 'ipa'>, phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine=<Engine.PIPER: 'piper'>, length_scale=1, noise_scale=0.667, noise_w_scale=0.8, blank_at_start=True, blank_at_end=True, include_whitespace=True, pad_token='_', blank_token='_', bos_token='^', eos_token='$', word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.ESPEAK: 'espeak'>)
    # TTSModelInfo(voice_id='OpenVoiceOS/pipertts_pt-BR_miro', lang='pt-BR', model_url='https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_miro/resolve/main/miro_pt-BR.onnx', config_url='https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_miro/resolve/main/miro_pt-BR.onnx.json', tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=1, num_langs=1, sample_rate=22050, lang_code='pt-BR', phoneme_id_map={' ': [3], '!': [4], '"': [150], '#': [149], '$': [2], "'": [5], '(': [6], ')': [7], ',': [8], '-': [9], '.': [10], '0': [130], '1': [131], '2': [132], '3': [133], '4': [134], '5': [135], '6': [136], '7': [137], '8': [138], '9': [139], ':': [11], ';': [12], '?': [13], 'X': [156], '^': [1], '_': [0], 'a': [14], 'b': [15], 'c': [16], 'd': [17], 'e': [18], 'f': [19], 'g': [154], 'h': [20], 'i': [21], 'j': [22], 'k': [23], 'l': [24], 'm': [25], 'n': [26], 'o': [27], 'p': [28], 'q': [29], 'r': [30], 's': [31], 't': [32], 'u': [33], 'v': [34], 'w': [35], 'x': [36], 'y': [37], 'z': [38], 'æ': [39], 'ç': [40], 'ð': [41], 'ø': [42], 'ħ': [43], 'ŋ': [44], 'œ': [45], 'ǀ': [46], 'ǁ': [47], 'ǂ': [48], 'ǃ': [49], 'ɐ': [50], 'ɑ': [51], 'ɒ': [52], 'ɓ': [53], 'ɔ': [54], 'ɕ': [55], 'ɖ': [56], 'ɗ': [57], 'ɘ': [58], 'ə': [59], 'ɚ': [60], 'ɛ': [61], 'ɜ': [62], 'ɞ': [63], 'ɟ': [64], 'ɠ': [65], 'ɡ': [66], 'ɢ': [67], 'ɣ': [68], 'ɤ': [69], 'ɥ': [70], 'ɦ': [71], 'ɧ': [72], 'ɨ': [73], 'ɪ': [74], 'ɫ': [75], 'ɬ': [76], 'ɭ': [77], 'ɮ': [78], 'ɯ': [79], 'ɰ': [80], 'ɱ': [81], 'ɲ': [82], 'ɳ': [83], 'ɴ': [84], 'ɵ': [85], 'ɶ': [86], 'ɸ': [87], 'ɹ': [88], 'ɺ': [89], 'ɻ': [90], 'ɽ': [91], 'ɾ': [92], 'ʀ': [93], 'ʁ': [94], 'ʂ': [95], 'ʃ': [96], 'ʄ': [97], 'ʈ': [98], 'ʉ': [99], 'ʊ': [100], 'ʋ': [101], 'ʌ': [102], 'ʍ': [103], 'ʎ': [104], 'ʏ': [105], 'ʐ': [106], 'ʑ': [107], 'ʒ': [108], 'ʔ': [109], 'ʕ': [110], 'ʘ': [111], 'ʙ': [112], 'ʛ': [113], 'ʜ': [114], 'ʝ': [115], 'ʟ': [116], 'ʡ': [117], 'ʢ': [118], 'ʦ': [155], 'ʰ': [145], 'ʲ': [119], 'ˈ': [120], 'ˌ': [121], 'ː': [122], 'ˑ': [123], '˞': [124], 'ˤ': [146], '̃': [141], '̊': [158], '̝': [157], '̧': [140], '̩': [144], '̪': [142], '̯': [143], '̺': [152], '̻': [153], 'β': [125], 'ε': [147], 'θ': [126], 'χ': [127], 'ᵻ': [128], '↑': [151], '↓': [148], 'ⱱ': [129]}, phoneme_type=<PhonemeType.ESPEAK: 'espeak'>, alphabet=<Alphabet.IPA: 'ipa'>, phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine=<Engine.PIPER: 'piper'>, length_scale=1, noise_scale=0.667, noise_w_scale=0.8, blank_at_start=True, blank_at_end=True, include_whitespace=True, pad_token='_', blank_token='_', bos_token='^', eos_token='$', word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.ESPEAK: 'espeak'>)
    # TTSModelInfo(voice_id='OpenVoiceOS/pipertts_pt-BR_dii', lang='pt-BR', model_url='https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_dii/resolve/main/dii_pt-BR.onnx', config_url='https://huggingface.co/OpenVoiceOS/pipertts_pt-BR_dii/resolve/main/dii_pt-BR.onnx.json', tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=1, num_langs=1, sample_rate=22050, lang_code='pt-BR', phoneme_id_map={' ': [3], '!': [4], '"': [150], '#': [149], '$': [2], "'": [5], '(': [6], ')': [7], ',': [8], '-': [9], '.': [10], '0': [130], '1': [131], '2': [132], '3': [133], '4': [134], '5': [135], '6': [136], '7': [137], '8': [138], '9': [139], ':': [11], ';': [12], '?': [13], 'X': [156], '^': [1], '_': [0], 'a': [14], 'b': [15], 'c': [16], 'd': [17], 'e': [18], 'f': [19], 'g': [154], 'h': [20], 'i': [21], 'j': [22], 'k': [23], 'l': [24], 'm': [25], 'n': [26], 'o': [27], 'p': [28], 'q': [29], 'r': [30], 's': [31], 't': [32], 'u': [33], 'v': [34], 'w': [35], 'x': [36], 'y': [37], 'z': [38], 'æ': [39], 'ç': [40], 'ð': [41], 'ø': [42], 'ħ': [43], 'ŋ': [44], 'œ': [45], 'ǀ': [46], 'ǁ': [47], 'ǂ': [48], 'ǃ': [49], 'ɐ': [50], 'ɑ': [51], 'ɒ': [52], 'ɓ': [53], 'ɔ': [54], 'ɕ': [55], 'ɖ': [56], 'ɗ': [57], 'ɘ': [58], 'ə': [59], 'ɚ': [60], 'ɛ': [61], 'ɜ': [62], 'ɞ': [63], 'ɟ': [64], 'ɠ': [65], 'ɡ': [66], 'ɢ': [67], 'ɣ': [68], 'ɤ': [69], 'ɥ': [70], 'ɦ': [71], 'ɧ': [72], 'ɨ': [73], 'ɪ': [74], 'ɫ': [75], 'ɬ': [76], 'ɭ': [77], 'ɮ': [78], 'ɯ': [79], 'ɰ': [80], 'ɱ': [81], 'ɲ': [82], 'ɳ': [83], 'ɴ': [84], 'ɵ': [85], 'ɶ': [86], 'ɸ': [87], 'ɹ': [88], 'ɺ': [89], 'ɻ': [90], 'ɽ': [91], 'ɾ': [92], 'ʀ': [93], 'ʁ': [94], 'ʂ': [95], 'ʃ': [96], 'ʄ': [97], 'ʈ': [98], 'ʉ': [99], 'ʊ': [100], 'ʋ': [101], 'ʌ': [102], 'ʍ': [103], 'ʎ': [104], 'ʏ': [105], 'ʐ': [106], 'ʑ': [107], 'ʒ': [108], 'ʔ': [109], 'ʕ': [110], 'ʘ': [111], 'ʙ': [112], 'ʛ': [113], 'ʜ': [114], 'ʝ': [115], 'ʟ': [116], 'ʡ': [117], 'ʢ': [118], 'ʦ': [155], 'ʰ': [145], 'ʲ': [119], 'ˈ': [120], 'ˌ': [121], 'ː': [122], 'ˑ': [123], '˞': [124], 'ˤ': [146], '̃': [141], '̊': [158], '̝': [157], '̧': [140], '̩': [144], '̪': [142], '̯': [143], '̺': [152], '̻': [153], 'β': [125], 'ε': [147], 'θ': [126], 'χ': [127], 'ᵻ': [128], '↑': [151], '↓': [148], 'ⱱ': [129]}, phoneme_type=<PhonemeType.ESPEAK: 'espeak'>, alphabet=<Alphabet.IPA: 'ipa'>, phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine=<Engine.PIPER: 'piper'>, length_scale=1, noise_scale=0.667, noise_w_scale=0.8, blank_at_start=True, blank_at_end=True, include_whitespace=True, pad_token='_', blank_token='_', bos_token='^', eos_token='$', word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.ESPEAK: 'espeak'>)
    # TTSModelInfo(voice_id='piper_pt_BR-cadu-medium', lang='pt-BR', model_url='https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/cadu/medium/pt_BR-cadu-medium.onnx', config_url='https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/cadu/medium/pt_BR-cadu-medium.onnx.json', tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=1, num_langs=1, sample_rate=22050, lang_code='pt-BR', phoneme_id_map={'_': [0], '^': [1], '$': [2], ' ': [3], '!': [4], "'": [5], '(': [6], ')': [7], ',': [8], '-': [9], '.': [10], ':': [11], ';': [12], '?': [13], 'a': [14], 'b': [15], 'c': [16], 'd': [17], 'e': [18], 'f': [19], 'h': [20], 'i': [21], 'j': [22], 'k': [23], 'l': [24], 'm': [25], 'n': [26], 'o': [27], 'p': [28], 'q': [29], 'r': [30], 's': [31], 't': [32], 'u': [33], 'v': [34], 'w': [35], 'x': [36], 'y': [37], 'z': [38], 'æ': [39], 'ç': [40], 'ð': [41], 'ø': [42], 'ħ': [43], 'ŋ': [44], 'œ': [45], 'ǀ': [46], 'ǁ': [47], 'ǂ': [48], 'ǃ': [49], 'ɐ': [50], 'ɑ': [51], 'ɒ': [52], 'ɓ': [53], 'ɔ': [54], 'ɕ': [55], 'ɖ': [56], 'ɗ': [57], 'ɘ': [58], 'ə': [59], 'ɚ': [60], 'ɛ': [61], 'ɜ': [62], 'ɞ': [63], 'ɟ': [64], 'ɠ': [65], 'ɡ': [66], 'ɢ': [67], 'ɣ': [68], 'ɤ': [69], 'ɥ': [70], 'ɦ': [71], 'ɧ': [72], 'ɨ': [73], 'ɪ': [74], 'ɫ': [75], 'ɬ': [76], 'ɭ': [77], 'ɮ': [78], 'ɯ': [79], 'ɰ': [80], 'ɱ': [81], 'ɲ': [82], 'ɳ': [83], 'ɴ': [84], 'ɵ': [85], 'ɶ': [86], 'ɸ': [87], 'ɹ': [88], 'ɺ': [89], 'ɻ': [90], 'ɽ': [91], 'ɾ': [92], 'ʀ': [93], 'ʁ': [94], 'ʂ': [95], 'ʃ': [96], 'ʄ': [97], 'ʈ': [98], 'ʉ': [99], 'ʊ': [100], 'ʋ': [101], 'ʌ': [102], 'ʍ': [103], 'ʎ': [104], 'ʏ': [105], 'ʐ': [106], 'ʑ': [107], 'ʒ': [108], 'ʔ': [109], 'ʕ': [110], 'ʘ': [111], 'ʙ': [112], 'ʛ': [113], 'ʜ': [114], 'ʝ': [115], 'ʟ': [116], 'ʡ': [117], 'ʢ': [118], 'ʲ': [119], 'ˈ': [120], 'ˌ': [121], 'ː': [122], 'ˑ': [123], '˞': [124], 'β': [125], 'θ': [126], 'χ': [127], 'ᵻ': [128], 'ⱱ': [129], '0': [130], '1': [131], '2': [132], '3': [133], '4': [134], '5': [135], '6': [136], '7': [137], '8': [138], '9': [139], '̧': [140], '̃': [141], '̪': [142], '̯': [143], '̩': [144], 'ʰ': [145], 'ˤ': [146], 'ε': [147], '↓': [148], '#': [149], '"': [150], '↑': [151], '̺': [152], '̻': [153], 'g': [154], 'ʦ': [155], 'X': [156], '̝': [157], '̊': [158], 'ɝ': [159], 'ʷ': [160]}, phoneme_type=<PhonemeType.ESPEAK: 'espeak'>, alphabet=<Alphabet.IPA: 'ipa'>, phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine=<Engine.PIPER: 'piper'>, length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, blank_at_start=True, blank_at_end=True, include_whitespace=True, pad_token='_', blank_token='_', bos_token='^', eos_token='$', word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.ESPEAK: 'espeak'>)
    # TTSModelInfo(voice_id='piper_pt_BR-edresson-low', lang='pt-BR', model_url='https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/edresson/low/pt_BR-edresson-low.onnx', config_url='https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/edresson/low/pt_BR-edresson-low.onnx.json', tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=130, num_speakers=1, num_langs=1, sample_rate=16000, lang_code='pt-BR', phoneme_id_map={'_': [0], '^': [1], '$': [2], ' ': [3], '!': [4], "'": [5], '(': [6], ')': [7], ',': [8], '-': [9], '.': [10], ':': [11], ';': [12], '?': [13], 'a': [14], 'b': [15], 'c': [16], 'd': [17], 'e': [18], 'f': [19], 'h': [20], 'i': [21], 'j': [22], 'k': [23], 'l': [24], 'm': [25], 'n': [26], 'o': [27], 'p': [28], 'q': [29], 'r': [30], 's': [31], 't': [32], 'u': [33], 'v': [34], 'w': [35], 'x': [36], 'y': [37], 'z': [38], 'æ': [39], 'ç': [40], 'ð': [41], 'ø': [42], 'ħ': [43], 'ŋ': [44], 'œ': [45], 'ǀ': [46], 'ǁ': [47], 'ǂ': [48], 'ǃ': [49], 'ɐ': [50], 'ɑ': [51], 'ɒ': [52], 'ɓ': [53], 'ɔ': [54], 'ɕ': [55], 'ɖ': [56], 'ɗ': [57], 'ɘ': [58], 'ə': [59], 'ɚ': [60], 'ɛ': [61], 'ɜ': [62], 'ɞ': [63], 'ɟ': [64], 'ɠ': [65], 'ɡ': [66], 'ɢ': [67], 'ɣ': [68], 'ɤ': [69], 'ɥ': [70], 'ɦ': [71], 'ɧ': [72], 'ɨ': [73], 'ɪ': [74], 'ɫ': [75], 'ɬ': [76], 'ɭ': [77], 'ɮ': [78], 'ɯ': [79], 'ɰ': [80], 'ɱ': [81], 'ɲ': [82], 'ɳ': [83], 'ɴ': [84], 'ɵ': [85], 'ɶ': [86], 'ɸ': [87], 'ɹ': [88], 'ɺ': [89], 'ɻ': [90], 'ɽ': [91], 'ɾ': [92], 'ʀ': [93], 'ʁ': [94], 'ʂ': [95], 'ʃ': [96], 'ʄ': [97], 'ʈ': [98], 'ʉ': [99], 'ʊ': [100], 'ʋ': [101], 'ʌ': [102], 'ʍ': [103], 'ʎ': [104], 'ʏ': [105], 'ʐ': [106], 'ʑ': [107], 'ʒ': [108], 'ʔ': [109], 'ʕ': [110], 'ʘ': [111], 'ʙ': [112], 'ʛ': [113], 'ʜ': [114], 'ʝ': [115], 'ʟ': [116], 'ʡ': [117], 'ʢ': [118], 'ʲ': [119], 'ˈ': [120], 'ˌ': [121], 'ː': [122], 'ˑ': [123], '˞': [124], 'β': [125], 'θ': [126], 'χ': [127], 'ᵻ': [128], 'ⱱ': [129]}, phoneme_type=<PhonemeType.ESPEAK: 'espeak'>, alphabet=<Alphabet.IPA: 'ipa'>, phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine=<Engine.PIPER: 'piper'>, length_scale=1, noise_scale=0.667, noise_w_scale=0.8, blank_at_start=True, blank_at_end=True, include_whitespace=True, pad_token='_', blank_token='_', bos_token='^', eos_token='$', word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.ESPEAK: 'espeak'>)
    # TTSModelInfo(voice_id='piper_pt_BR-faber-medium', lang='pt-BR', model_url='https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx', config_url='https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx.json', tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=1, num_langs=1, sample_rate=22050, lang_code='pt-BR', phoneme_id_map={'_': [0], '^': [1], '$': [2], ' ': [3], '!': [4], "'": [5], '(': [6], ')': [7], ',': [8], '-': [9], '.': [10], ':': [11], ';': [12], '?': [13], 'a': [14], 'b': [15], 'c': [16], 'd': [17], 'e': [18], 'f': [19], 'h': [20], 'i': [21], 'j': [22], 'k': [23], 'l': [24], 'm': [25], 'n': [26], 'o': [27], 'p': [28], 'q': [29], 'r': [30], 's': [31], 't': [32], 'u': [33], 'v': [34], 'w': [35], 'x': [36], 'y': [37], 'z': [38], 'æ': [39], 'ç': [40], 'ð': [41], 'ø': [42], 'ħ': [43], 'ŋ': [44], 'œ': [45], 'ǀ': [46], 'ǁ': [47], 'ǂ': [48], 'ǃ': [49], 'ɐ': [50], 'ɑ': [51], 'ɒ': [52], 'ɓ': [53], 'ɔ': [54], 'ɕ': [55], 'ɖ': [56], 'ɗ': [57], 'ɘ': [58], 'ə': [59], 'ɚ': [60], 'ɛ': [61], 'ɜ': [62], 'ɞ': [63], 'ɟ': [64], 'ɠ': [65], 'ɡ': [66], 'ɢ': [67], 'ɣ': [68], 'ɤ': [69], 'ɥ': [70], 'ɦ': [71], 'ɧ': [72], 'ɨ': [73], 'ɪ': [74], 'ɫ': [75], 'ɬ': [76], 'ɭ': [77], 'ɮ': [78], 'ɯ': [79], 'ɰ': [80], 'ɱ': [81], 'ɲ': [82], 'ɳ': [83], 'ɴ': [84], 'ɵ': [85], 'ɶ': [86], 'ɸ': [87], 'ɹ': [88], 'ɺ': [89], 'ɻ': [90], 'ɽ': [91], 'ɾ': [92], 'ʀ': [93], 'ʁ': [94], 'ʂ': [95], 'ʃ': [96], 'ʄ': [97], 'ʈ': [98], 'ʉ': [99], 'ʊ': [100], 'ʋ': [101], 'ʌ': [102], 'ʍ': [103], 'ʎ': [104], 'ʏ': [105], 'ʐ': [106], 'ʑ': [107], 'ʒ': [108], 'ʔ': [109], 'ʕ': [110], 'ʘ': [111], 'ʙ': [112], 'ʛ': [113], 'ʜ': [114], 'ʝ': [115], 'ʟ': [116], 'ʡ': [117], 'ʢ': [118], 'ʲ': [119], 'ˈ': [120], 'ˌ': [121], 'ː': [122], 'ˑ': [123], '˞': [124], 'β': [125], 'θ': [126], 'χ': [127], 'ᵻ': [128], 'ⱱ': [129], '0': [130], '1': [131], '2': [132], '3': [133], '4': [134], '5': [135], '6': [136], '7': [137], '8': [138], '9': [139], '̧': [140], '̃': [141], '̪': [142], '̯': [143], '̩': [144], 'ʰ': [145], 'ˤ': [146], 'ε': [147], '↓': [148], '#': [149], '"': [150], '↑': [151]}, phoneme_type=<PhonemeType.ESPEAK: 'espeak'>, alphabet=<Alphabet.IPA: 'ipa'>, phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine=<Engine.PIPER: 'piper'>, length_scale=1, noise_scale=0.667, noise_w_scale=0.8, blank_at_start=True, blank_at_end=True, include_whitespace=True, pad_token='_', blank_token='_', bos_token='^', eos_token='$', word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.ESPEAK: 'espeak'>)
    # TTSModelInfo(voice_id='piper_pt_BR-jeff-medium', lang='pt-BR', model_url='https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/jeff/medium/pt_BR-jeff-medium.onnx', config_url='https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/jeff/medium/pt_BR-jeff-medium.onnx.json', tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=1, num_langs=1, sample_rate=22050, lang_code='pt-BR', phoneme_id_map={'_': [0], '^': [1], '$': [2], ' ': [3], '!': [4], "'": [5], '(': [6], ')': [7], ',': [8], '-': [9], '.': [10], ':': [11], ';': [12], '?': [13], 'a': [14], 'b': [15], 'c': [16], 'd': [17], 'e': [18], 'f': [19], 'h': [20], 'i': [21], 'j': [22], 'k': [23], 'l': [24], 'm': [25], 'n': [26], 'o': [27], 'p': [28], 'q': [29], 'r': [30], 's': [31], 't': [32], 'u': [33], 'v': [34], 'w': [35], 'x': [36], 'y': [37], 'z': [38], 'æ': [39], 'ç': [40], 'ð': [41], 'ø': [42], 'ħ': [43], 'ŋ': [44], 'œ': [45], 'ǀ': [46], 'ǁ': [47], 'ǂ': [48], 'ǃ': [49], 'ɐ': [50], 'ɑ': [51], 'ɒ': [52], 'ɓ': [53], 'ɔ': [54], 'ɕ': [55], 'ɖ': [56], 'ɗ': [57], 'ɘ': [58], 'ə': [59], 'ɚ': [60], 'ɛ': [61], 'ɜ': [62], 'ɞ': [63], 'ɟ': [64], 'ɠ': [65], 'ɡ': [66], 'ɢ': [67], 'ɣ': [68], 'ɤ': [69], 'ɥ': [70], 'ɦ': [71], 'ɧ': [72], 'ɨ': [73], 'ɪ': [74], 'ɫ': [75], 'ɬ': [76], 'ɭ': [77], 'ɮ': [78], 'ɯ': [79], 'ɰ': [80], 'ɱ': [81], 'ɲ': [82], 'ɳ': [83], 'ɴ': [84], 'ɵ': [85], 'ɶ': [86], 'ɸ': [87], 'ɹ': [88], 'ɺ': [89], 'ɻ': [90], 'ɽ': [91], 'ɾ': [92], 'ʀ': [93], 'ʁ': [94], 'ʂ': [95], 'ʃ': [96], 'ʄ': [97], 'ʈ': [98], 'ʉ': [99], 'ʊ': [100], 'ʋ': [101], 'ʌ': [102], 'ʍ': [103], 'ʎ': [104], 'ʏ': [105], 'ʐ': [106], 'ʑ': [107], 'ʒ': [108], 'ʔ': [109], 'ʕ': [110], 'ʘ': [111], 'ʙ': [112], 'ʛ': [113], 'ʜ': [114], 'ʝ': [115], 'ʟ': [116], 'ʡ': [117], 'ʢ': [118], 'ʲ': [119], 'ˈ': [120], 'ˌ': [121], 'ː': [122], 'ˑ': [123], '˞': [124], 'β': [125], 'θ': [126], 'χ': [127], 'ᵻ': [128], 'ⱱ': [129], '0': [130], '1': [131], '2': [132], '3': [133], '4': [134], '5': [135], '6': [136], '7': [137], '8': [138], '9': [139], '̧': [140], '̃': [141], '̪': [142], '̯': [143], '̩': [144], 'ʰ': [145], 'ˤ': [146], 'ε': [147], '↓': [148], '#': [149], '"': [150], '↑': [151], '̺': [152], '̻': [153], 'g': [154], 'ʦ': [155], 'X': [156], '̝': [157], '̊': [158], 'ɝ': [159], 'ʷ': [160]}, phoneme_type=<PhonemeType.ESPEAK: 'espeak'>, alphabet=<Alphabet.IPA: 'ipa'>, phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine=<Engine.PIPER: 'piper'>, length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, blank_at_start=True, blank_at_end=True, include_whitespace=True, pad_token='_', blank_token='_', bos_token='^', eos_token='$', word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.ESPEAK: 'espeak'>)

    print(manager.supported_langs)
    # ['af-ZA', 'ar-JO', 'bn', 'ca-ES', 'cs-CZ', 'cy-GB', 'da-DK', 'de-DE', 'el-GR', 'en-GB', 'en-US', 'es-AR',
    # 'es-ES', 'es-MX', 'fa', 'fa-IR', 'fi-FI', 'fr-FR', 'gl-ES', 'gu-IN', 'ha-NE', 'he', 'hi-IN', 'hu-HU', 'id-ID',
    # 'is-IS', 'it-IT', 'jv-ID', 'ka-GE', 'kk-KZ', 'ko-KO', 'lb-LU', 'lv-LV', 'ml-IN', 'ne-NP', 'nl', 'nl-BE', 'nl-NL',
    # 'no-NO', 'pl-PL', 'pt-BR', 'pt-PT', 'ro-RO', 'ru-RU', 'sk-SK', 'sl-SI', 'sr-RS', 'sv-SE', 'sw', 'sw-CD',
    # 'te-IN', 'tn-ZA', 'tr-TR', 'uk-GB', 'uk-UA', 'vi-VN', 'yo', 'zh-CN']

    manager.all_voices[0].load()
