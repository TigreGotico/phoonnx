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
    alphabet: Optional[Alphabet] = None

    def __post_init__(self):
        """
        Initialize the TTSModelInfo instance by ensuring local cache files exist and synchronizing its configuration, alphabet, and phoneme type.
        
        If no VoiceConfig was provided, ensure the voice cache directory exists, download and load the model config (model.json), apply a known phoneme-type compatibility fix, and—when a tokens URL is present—download the tokens file and construct the VoiceConfig using it. Always set the loaded config's language code from this instance's `lang`. After loading (or when a config was provided), ensure `alphabet` and `phoneme_type` on the dataclass and on the loaded config are consistent by propagating values from whichever side is present.
        """
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

        if not self.alphabet:
            self.alphabet = self.config.alphabet
        else:
            self.config.alphabet = self.alphabet

        if not self.phoneme_type:
            self.phoneme_type = self.config.phoneme_type
        else:
            self.config.phoneme_type = self.phoneme_type

    @property
    def engine(self) -> Engine:
        """
        Return the Engine type used by this voice's configuration.
        
        Returns:
            Engine: The engine configured for this voice.
        """
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
        """
        Load and return a TTSVoice for this model, ensuring the ONNX model is downloaded and the voice configuration is applied.
        
        Loads a TTSVoice from the cached model and config files (and tokens file if available). If this TTSModelInfo specifies a different phoneme type or alphabet than the loaded voice, updates the loaded voice's phoneme_type and alphabet and rebuilds its phonemizer accordingly.
        
        Returns:
            TTSVoice: The configured TTSVoice instance ready for synthesis.
        """
        model_path = self.voice_path / "model.onnx"
        config_path = self.voice_path / "model.json"
        tokens_path = self.voice_path / "tokens.txt"
        self.download_model()

        voice = TTSVoice.load(model_path=model_path,
                              config_path=config_path,
                              lang_code=self.config.lang_code,
                              phoneme_type_str=self.config.phoneme_type.value,
                              alphabet_str=self.config.alphabet.value,
                              phonemes_txt=str(tokens_path) if self.tokens_url else None)
        # override phoneme_type, if config.json is wrong
        if self.phoneme_type != voice.config.phoneme_type or self.alphabet != voice.config.alphabet:
            voice.phoneme_type = self.phoneme_type
            voice.config.alphabet = self.alphabet
            voice.phonemizer = get_phonemizer(self.phoneme_type,
                                              alphabet=self.alphabet,
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
        """
        Persist current in-memory voice metadata to the configured cache storage.
        
        Clears the cache, writes each managed voice's public metadata (voice_id, model_url,
        phoneme_type, lang, tokens_url, phoneme_map_url, alphabet, config_url) into the cache,
        and then stores the cache to disk.
        """
        self.cache.clear()
        for voice_id, voice_info in self.voices.items():
            self.cache[voice_id] = {"voice_id": voice_info.voice_id,
                                    "model_url": voice_info.model_url,
                                    "phoneme_type": voice_info.phoneme_type,
                                    "lang": voice_info.lang,
                                    "tokens_url": voice_info.tokens_url,
                                    "phoneme_map_url": voice_info.phoneme_map_url,
                                    "alphabet": voice_info.alphabet,
                                    "config_url": voice_info.config_url}
        self.cache.store()

    def add_voice(self, voice_info: TTSModelInfo):
        """
        Add or update a TTS voice in the manager's in-memory registry and persist its public metadata to the cache.
        
        This stores the given TTSModelInfo under its voice_id in memory and writes a curated subset of its fields (voice_id, model_url, tokens_url, phoneme_type, phoneme_map_url, alphabet, lang, config_url) into the persistent cache, overwriting any existing entry for the same voice_id.
        
        Parameters:
            voice_info (TTSModelInfo): The voice metadata to add or update.
        """
        self.voices[voice_info.voice_id] = voice_info
        self.cache[voice_info.voice_id] = {"voice_id": voice_info.voice_id,
                                           "model_url": voice_info.model_url,
                                           "tokens_url": voice_info.tokens_url,
                                           "phoneme_type": voice_info.phoneme_type,
                                           "phoneme_map_url": voice_info.phoneme_map_url,
                                           "alphabet": voice_info.alphabet,
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
        """
        Refresh the in-memory voice catalog from all known sources and persist the updated cache.
        
        This repopulates the manager's voices by fetching entries from the configured sources (official and community manifests) and then stores the resulting voice metadata to the persistent cache.
        """
        self.get_ovos_voice_list()
        self.get_proxectonos_voice_list()
        self.get_piper_voice_list()
        self.get_mimic3_voice_list()
        self.get_phonikud_voice_list()
        self.get_neurlang_voice_list()
        self.get_piper_community_voice_list()
        self.get_coqui_community_voice_list()
        self.cache.store()

    # helpers to get official voice models
    def get_ovos_voice_list(self):
        """
        Register OpenVoiceOS phoonnx and Piper TTS voices into the manager's voice catalog.
        
        Adds TTSModelInfo entries for a hardcoded set of phoonnx Hugging Face repositories and for a set of common Piper languages. For each entry the method constructs model and config URLs pointing to the repository's main branch on Hugging Face and calls add_voice to register the voice. Missing voice variants are skipped silently.
        """
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

    def get_proxectonos_voice_list(self):
        # NOTE: these are models trained with coqui
        #  we need to explicitly assign phonemizer
        """
        Add Proxectonos (Galician) TTS model entries to the manager.
        
        Adds two grapheme-based voices ("brais", "celtia") with PhonemeType.GRAPHEMES and Alphabet.UNICODE, and four phoneme-based voices ("sabela", "icia", "paulo", "iago") with PhonemeType.COTOVIA and Alphabet.COTOVIA. Each entry includes model and config URLs pointing to the corresponding OpenVoiceOS Proxectonos Hugging Face repositories.
        """
        for voice in ["brais", "celtia"]:
            self.add_voice(TTSModelInfo(
                voice_id=f"proxectonos/{voice}",
                lang="gl-ES",
                model_url=f"https://huggingface.co/OpenVoiceOS/proxectonos-{voice}-vits-graphemes-onnx/resolve/main/model.onnx",
                config_url=f"https://huggingface.co/OpenVoiceOS/proxectonos-{voice}-vits-graphemes-onnx/resolve/main/config.json",
                phoneme_type=PhonemeType.GRAPHEMES,
                alphabet=Alphabet.UNICODE
            ))
        for voice in ["sabela", "icia", "paulo", "iago"]:
            self.add_voice(TTSModelInfo(
                voice_id=f"proxectonos/{voice}",
                lang="gl-ES",
                model_url=f"https://huggingface.co/OpenVoiceOS/proxectonos-{voice}-vits-phonemes-onnx/resolve/main/model.onnx",
                config_url=f"https://huggingface.co/OpenVoiceOS/proxectonos-{voice}-vits-phonemes-onnx/resolve/main/config.json",
                phoneme_type=PhonemeType.COTOVIA,
                alphabet=Alphabet.COTOVIA

            ))

    def get_piper_voice_list(self):
        """
        Fetches the Piper voices manifest from the Rhasspy piper-voices repository and registers each voice in the manager.
        
        Downloads the voices.json manifest, creates a TTSModelInfo for each entry (deriving a voice_id prefixed with "piper_", a standardized language tag, and the first ONNX and JSON file URLs from the entry), and calls add_voice to store it. If an entry cannot be processed, prints a failure message for that voice.
        """
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

    def get_neurlang_voice_list(self):
        """
        Populate the manager with a fixed set of NeurLang Piper voices.
        
        Adds four NeurLang Piper TTSModelInfo entries (Arabic, British English, Slovak, Korean), each configured to use the `GORUUT` phoneme type and stored under voice IDs prefixed with `piper_neurlang/`.
        """
        for repo, lang in [
            ("piper-onnx-zayd0-arabic-diacritized", "ar"),
            ("piper-onnx-jane-eyre-english-british", "en-GB"),
            ("piper-onnx-slovakspeech-female-slovak", "sl-SI"),
            ("piper-onnx-kss-korean", "ko-KO"),
        ]:
            model = repo.replace('-onnx-kss', '-kss')
            url = f"https://huggingface.co/neurlang/{repo}/resolve/main/{model}.onnx"
            voice = TTSModelInfo(
                voice_id="piper_" + f"neurlang/{lang}_{repo.replace('piper-onnx-', '')}",
                lang=lang,
                model_url=url,
                config_url=url + ".json",
                phoneme_type=PhonemeType.GORUUT
            )
            self.add_voice(voice)

    def get_mimic3_voice_list(self):
        """
        Fetch and register Mimic3 TTS voices from Mycroft's voices manifest.
        
        Fetches the remote Mimic3 voices manifest, constructs TTSModelInfo entries for each voice (including config, model, tokens, and phoneme map URLs), sets the voice's language and speaker_id_map, and adds the voice to the manager. Individual voice failures are logged and do not interrupt processing.
        """
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

    def get_phonikud_voice_list(self):
        # NOTE: trained with piper + raw phonemes
        #  we need to explicitly assign phonemizer
        """
        Register Phonikud-trained Hebrew Piper voices in the manager's catalog.
        
        Adds two TTSModelInfo entries for Phonikud-based Hebrew voices and marks them with PhonemeType.PHONIKUD so the phonemizer is assigned explicitly.
        """
        self.add_voice(
            TTSModelInfo(
                voice_id="thewh1teagle/phonikud",
                lang="he",
                model_url="https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.onnx",
                config_url="https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.config.json",
                phoneme_type=PhonemeType.PHONIKUD
            )
        )
        self.add_voice(
            TTSModelInfo(
                voice_id="thewh1teagle/phonikud-shaul",
                lang="he",
                model_url="https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/shaul.onnx",
                config_url="https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.config.json",
                phoneme_type=PhonemeType.PHONIKUD
            )
        )

    # community models sourced from around the web
    def get_piper_community_voice_list(self):
        """
        Register a collection of community-sourced Piper TTS voices into the manager.
        
        Adds hardcoded Piper community voice entries (voice_id, lang, model_url, config_url) to the manager by calling self.add_voice. Some entries may require Hugging Face authentication or special handling (archives, nested archives), and duplicate models can appear because Piper models are sometimes merged upstream.
        """
        # https://huggingface.co/mbarnig/lb_rhasspy_piper_tts
        for voice in ["androgynous", "femaleLOD", "marylux"]:
            url = f"https://huggingface.co/mbarnig/lb_rhasspy_piper_tts/resolve/main/lb/lb_LU/{voice}/medium/lb_LU-{voice}-medium.onnx"
            voice = TTSModelInfo(
                voice_id="piper_" + f"mbarnig/lb-LU_{voice}",
                lang="lb-LU",
                model_url=url,
                config_url=url + ".json",
            )
            self.add_voice(voice)

        # https://huggingface.co/superkeka/piper-tts-luka
        url = f"https://huggingface.co/superkeka/piper-tts-luka/resolve/main/ru/ru_RU/luka/medium/ru_RU-luka-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "superkeka/ru-RU_luka",
            lang="ru-RU",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/davit312/piper-TTS-Armenian
        url = f"https://huggingface.co/davit312/piper-TTS-Armenian/resolve/main/v3/hy_AM-gor-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "davit312/hy-AM_gor",
            lang="hy-AM",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/raphaelmerx/piper-voices
        url = f"https://huggingface.co/raphaelmerx/piper-voices/resolve/main/tdt/tdt_TL/joao/medium/tdt_TL-joao-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "raphaelmerx/tdt-TL_joao",
            lang="tdt-TL",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/wezzmeister/piper-voices
        url = f"https://huggingface.co/wezzmeister/piper-voices/resolve/main/sv_SE-lisa-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "wezzmeister/sv-SE_lisa",
            lang="sv-SE",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/SubZeroAI/piper-swedish-tts-multispeaker
        # TODO - 401 - needs login and approval in HF
        # url = f"https://huggingface.co/SubZeroAI/piper-swedish-tts-multispeaker/resolve/main/piper-swedish-tts-multispeaker.onnx"
        # voice = TTSModelInfo(
        #    voice_id="piper_" + "SubZeroAI/sv-SE_multispeaker",
        #    lang="sv-SE",
        #    model_url=url,
        #    config_url=url + ".json",
        # )
        # self.add_voice(voice)

        # https://huggingface.co/larcanio/piper-voices
        url = f"https://huggingface.co/larcanio/piper-voices/resolve/main/es_AR-daniela-high.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "larcanio/es-AR_daniela",
            lang="es-AR",
            model_url=url,
            config_url=url.replace(".onnx", ".json")
        )
        self.add_voice(voice)

        # https://huggingface.co/friyin/vits-piper-es_ES-friyin-high
        url = f"https://huggingface.co/friyin/vits-piper-es_ES-carlfm-high/resolve/main/es_ES-carlfm-high.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "friyin/es-ES_friyin",
            lang="es-ES",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/Wiseyak/piper_tts
        url = f"https://huggingface.co/Wiseyak/piper_tts/resolve/main/ne-seto_bagh-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "Wiseyak/ne-NP_seto_bagh",
            lang="ne-NP",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/colafly/piper_zh_tw
        url = f"https://huggingface.co/colafly/piper_zh_tw/resolve/main/yt-chinese_female.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "colafly/zh-TW_yt-chinese_female",
            lang="zh-TW",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/ppisljar/piper_si_artur
        url = f"https://huggingface.co/ppisljar/piper_si_artur/resolve/main/model.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "ppisljar/sl-SI_artur",
            lang="sl-SI",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/giganticlab/piper-id_ID-news_tts-medium
        url = f"https://huggingface.co/giganticlab/piper-id_ID-news_tts-medium/resolve/main/model.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "giganticlab/id-ID_news",
            lang="id-ID",
            model_url=url,
            config_url=url.replace("model.onnx", "config.json"),
        )
        self.add_voice(voice)

        # https://huggingface.co/phcatan9921/piper_tts
        url = f"https://huggingface.co/phcatan9921/piper_tts/resolve/main/vi_VN-vais1000-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "phcatan9921/vi-VN_vais1000",
            lang="vi-VN",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://github.com/phatjkk/vits-tts-vietnamese
        url = f"https://github.com/phatjkk/vits-tts-vietnamese/raw/refs/heads/main/pretrained_vi.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "phatjkk/vi-VN_InfoRe",
            lang="vi-VN",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/RaivisDejus/Piper-lv_LV-Aivars-medium
        url = f"https://huggingface.co/RaivisDejus/Piper-lv_LV-Aivars-medium/resolve/main/lv_LV-aivars-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "RaivisDejus/lv-LV_Aivars",
            lang="lv-LV",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/PravalX/piper-voices
        for voice in ["priyamvada", "pratham"]:
            url = f"https://huggingface.co/PravalX/piper-voices/resolve/main/hi/hi_IN/{voice}/medium/hi_IN-{voice}-medium.onnx"
            voice = TTSModelInfo(
                voice_id="piper_" + f"PravalX/hi-IN_{voice}",
                lang="hi-IN",
                model_url=url,
                config_url=url + ".json",
            )
            self.add_voice(voice)

        # https://huggingface.co/WitoldG/polish_piper_models
        for voice in ["jarvis", "justyna", "meski", "zenski"]:
            url = f"https://huggingface.co/WitoldG/polish_piper_models/resolve/main/pl_PL-{voice}_wg_glos-medium.onnx"
            voice = TTSModelInfo(
                voice_id="piper_" + f"WitoldG/pl-PL_{voice}",
                lang="pl-PL",
                model_url=url,
                config_url=url + ".json",
            )
            self.add_voice(voice)

        # https://huggingface.co/srxz/sage-voice-pt-br
        url = "https://huggingface.co/srxz/sage-voice-pt-br/resolve/main/pt_BR-sage_13364-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "srxz/pt-BR_sage",
            lang="pt-BR",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/Thomcles/Piper-TTS-Czech
        for qual in ["medium", "high"]:
            url = f"https://huggingface.co/Thomcles/Piper-TTS-Czech/resolve/main/{qual}/model.onnx"
            voice = TTSModelInfo(
                voice_id="piper_" + f"Thomcles/cs-CZ_honza_{qual}",
                lang="cs-CZ",
                model_url=url,
                config_url=url + ".json",
            )
            self.add_voice(voice)

        # https://huggingface.co/AsmoKoskinen/Piper_Finnish_Model
        url = "https://huggingface.co/AsmoKoskinen/Piper_Finnish_Model/resolve/main/fi_FI-asmo-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "AsmoKoskinen/fi-FI_asmo",
            lang="fi-FI",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/gyroing/Persian-Piper-Model-gyro
        url = "https://huggingface.co/gyroing/Persian-Piper-Model-gyro/resolve/main/fa_IR-gyro-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "gyroing/fa-IR_gyro",
            lang="fa-IR",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/mah92/Reza-And-Ibrahim-FA_EN-Piper-TTS-Model
        url = "https://huggingface.co/mah92/Reza-And-Ibrahim-FA_EN-Piper-TTS-Model/resolve/main/fa_en-rezahedayatfar-ibrahimwalk-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "mah92/fa-IR_Reza-And-Ibrahim",
            lang="fa-IR",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/Einrich99/PiperTTS-UGO-Italian
        url = "https://huggingface.co/Einrich99/PiperTTS-UGO-Italian/resolve/main/medium/it_IT-ugo-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "Einrich99/it-IT_ugo",
            lang="it-IT",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/paolapersico1/Piper-TTS-Italian
        url = "https://huggingface.co/paolapersico1/Piper-TTS-Italian/resolve/main/paola/medium/it_IT-paola-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "paolapersico1/it-IT_paola",
            lang="it-IT",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/kirys79/piper_italiano
        url = f"https://huggingface.co/kirys79/piper_italiano/resolve/main/Aurora/it_IT-aurora-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "kirys79/it-IT_Aurora",
            lang="it-IT",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)
        url = "https://huggingface.co/kirys79/piper_italiano/resolve/main/Giorgio/giorgio-epoch%3D5028-step%3D1098436.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "kirys79/it-IT_Giorgio",
            lang="it-IT",
            model_url=url,
            config_url=url.replace(".onnx", ".json"),
        )
        self.add_voice(voice)
        url = f"https://huggingface.co/kirys79/piper_italiano/resolve/main/Leonardo/leonardo-epoch%3D2024-step%3D996300.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "kirys79/it-IT_Leonardo",
            lang="it-IT",
            model_url=url,
            config_url=url.replace(".onnx", ".json"),
        )
        self.add_voice(voice)

        # https://huggingface.co/nardocolin/nardocolin-pipertts
        url = "https://huggingface.co/nardocolin/nardocolin-pipertts/resolve/main/high/colin-voice_high.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "nardocolin/en-GB_Colin",
            lang="en-GB",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/Da-Bob/piper-mikev3
        url = "https://huggingface.co/Da-Bob/piper-mikev3/resolve/main/mikev3.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "Da-Bob/en-US_mikev3",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/agentvibes/piper-custom-voices
        for voice, lang in [("kristin", "en-US"), ("jenny", "en-IE"), ("16Speakers", "en")]:
            url = f"https://huggingface.co/agentvibes/piper-custom-voices/resolve/main/{voice}.onnx"
            voice = TTSModelInfo(
                voice_id="piper_" + f"agentvibe/{lang}_{voice}",
                lang=lang,
                model_url=url,
                config_url=url + ".json",
            )
            self.add_voice(voice)

        # HAV0X1014/KF-PiperTTS-voices
        for voice, qual in [("Cheetah", "high"), ("KingCheetah", "medium"), ("silverfox", "medium")]:
            url = f"https://huggingface.co/HAV0X1014/KF-PiperTTS-voices/resolve/main/{voice}/en_US-{voice.lower()}-{qual}.onnx"
            voice = TTSModelInfo(
                voice_id="piper_" + f"agentvibe/en-US_{voice}",
                lang="en-US",
                model_url=url,
                config_url=url + ".json",
            )
            self.add_voice(voice)

        # https://huggingface.co/campwill/HAL-9000-Piper-TTS
        url = "https://huggingface.co/campwill/HAL-9000-Piper-TTS/resolve/main/hal.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "campwill/en-US_HAL-9000",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/redromnon/piper-tts-elise
        url = "https://huggingface.co/redromnon/piper-tts-elise/resolve/main/en_US-elisa-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "redromnon/en-US_elise",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/poisson-fish/piper-vasco
        url = "https://huggingface.co/poisson-fish/piper-vasco/resolve/main/onnx/vasco.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "poisson-fish/en-US_vasco",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/rokeya71/VITS-Piper-GlaDOS-en-onnx
        url = "https://huggingface.co/rokeya71/VITS-Piper-GlaDOS-en-onnx/resolve/main/glados.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "rokeya71/en-US_GlaDOS",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/Aquaaa123/piper-tts-pda-subnautica
        url = "https://huggingface.co/Aquaaa123/piper-tts-pda-subnautica/resolve/main/pda.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "Aquaaa123/en-US_pda-subnautica",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/drewThomasson/piper_tts_finetune_death_from_puss_and_boots
        url = "https://huggingface.co/drewThomasson/piper_tts_finetune_death_from_puss_and_boots/resolve/main/en_US-death-high_onnx/en_US-death-high.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + "drewThomasson/en-US_death_from_puss_and_boots",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/samarthshrivas/piper-finetune-Andrew-Huberman
        url = "https://huggingface.co/samarthshrivas/piper-finetune-Andrew-Huberman/resolve/main/lightning_logs/version_2/checkpoints/epoch%3D2609-step%3D1364440.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + f"samarthshrivas/en-US_Andrew-Huberman",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/swqg-messiah/kusaal_chitti_piper
        url = "https://huggingface.co/swqg-messiah/kusaal_chitti_piper/resolve/main/chitti.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + f"swqg-messiah/en-US_chitti",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/jstlntch/Scaramouche_or_Wanderer_voice_model_for_piper
        url = "https://huggingface.co/jstlntch/Scaramouche_or_Wanderer_voice_model_for_piper/resolve/main/model.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + f"jstlntchh/en_Scaramouche",
            lang="en",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/Rikels/piper-dutch
        url = "https://huggingface.co/Rikels/piper-dutch/resolve/main/anna.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + f"Rikels/nl-NL_anna",
            lang="nl-NL",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/systemofapwne/piper-de-glados
        for qual in ["high", "medium", "low"]:
            url = f"https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados/{qual}/de_DE-glados-{qual}.onnx"
            voice = TTSModelInfo(
                voice_id="piper_" + f"systemofapwne/de-DE_glados_{qual}",
                lang="de-DE",
                model_url=url,
                config_url=url + ".json",
            )
            self.add_voice(voice)

            url = f"https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados-turret/{qual}/de_DE-glados-turret-{qual}.onnx"
            voice = TTSModelInfo(
                voice_id="piper_" + f"systemofapwne/de-DE_glados-turret_{qual}",
                lang="de-DE",
                model_url=url,
                config_url=url + ".json",
            )
            self.add_voice(voice)

        # https://huggingface.co/nullnullvier/kantodel
        url = "https://huggingface.co/nullnullvier/kantodel/resolve/main/kantodel.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + f"nullnullvier/de-DE_kantodel",
            lang="de-DE",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # https://huggingface.co/domoskanonos/piper-tts-models
        for voice, qual in [("domoskanonos", "high"), ("sebastian100", "medium"), ("sebastian121", "medium")]:
            url = f"https://huggingface.co/domoskanonos/piper-tts-models/resolve/main/de-{voice}-{qual}.onnx"
            voice = TTSModelInfo(
                voice_id="piper_" + f"domoskanonos/de-DE_{voice}",
                lang="de-DE",
                model_url=url,
                config_url=url + ".json",
            )
            self.add_voice(voice)

        # TODO - unknown phonemizer type?
        # https://huggingface.co/tiennguyenbnbk/male_vivoice_piper_viphone

        # TODO - these models are inside a .tar.gz/.zip and will need special handling
        # https://huggingface.co/MysticonsLover/PiperWillowbrook
        # https://huggingface.co/BibEBobberson/Piper
        # https://huggingface.co/Beesa/Piper_brawlstars
        # https://huggingface.co/BornSaint/piper-TTS

        # https://huggingface.co/HirCoir/Piper-TTS-Laura
        url = f"https://huggingface.co/HirCoir/Piper-TTS-Laura/resolve/main/es_MX-laura-high.onnx"
        voice = TTSModelInfo(
            voice_id="piper_" + f"HirCoir/es-MX_Laura",
            lang="es-MX",
            model_url=url,
            config_url=url + ".json",
        )
        self.add_voice(voice)

        # TODO - 401 - needs auth and approval in hugging face
        # https://huggingface.co/HirCoir/HirCoir/piper-emma-neuronal
        # https://huggingface.co/HirCoir/piper-sorah-neuronal
        # https://huggingface.co/HirCoir/piper-voice-es-mx-lucas-melor
        # https://huggingface.co/HirCoir/piper-voice-es-mx-veritasium
        # https://huggingface.co/HirCoir/piper-voice-es-mx-1peso-de-salsa
        # https://huggingface.co/HirCoir/piper-checkpoint-es-mx-sorah-v2
        # https://huggingface.co/HirCoir/piper-checkpoint-es-mx-sorahv2
        # https://huggingface.co/HirCoir/piper-checkpoint-es-ar-elena
        # https://huggingface.co/HirCoir/piper-checkpoint-yiseni
        # https://huggingface.co/HirCoir/piper-checkpoint-es-mx-dark
        # https://huggingface.co/HirCoir/piper-checkpoint-es-mx-maney
        # https://huggingface.co/HirCoir/piper-checkpoint-es-mx-yahir
        # https://huggingface.co/HirCoir/piper-checkpoint-es-mx-1peso-de-salsa
        # https://huggingface.co/HirCoir/piper-checkpoint-es-mx-laurav2
        # https://huggingface.co/HirCoir/piper-checkpoint-es-mx-veritsasium
        # https://huggingface.co/HirCoir/piper-checkpoint-es-mx-lilith
        # https://huggingface.co/HirCoir/piper-checkpoint-es-mx-towi
        # https://huggingface.co/HirCoir/piper-checkpoint-es-mx-cortana-ce-legacy
        # https://huggingface.co/HirCoir/piper-voice-es_MX-Cortana-CE-Legacy

    def get_coqui_community_voice_list(self):
        """
        Add Coqui community voice entries to the manager.
        
        This method is a placeholder and currently performs no action. Intended to discover Coqui community TTS model manifests and add corresponding TTSModelInfo entries to self.voices when implemented.
        """
        pass  # placeholder


if __name__ == "__main__":
    manager = TTSModelManager()
    manager.clear()
    # manager.load()
    manager.refresh_voices()
    manager.save()

    print(f"Total voices: {len(manager.all_voices)}")
    print(f"Total langs: {len(manager.supported_langs)}")

    # Total voices: 284
    # Total langs: 67

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
    # ['af-ZA', 'ar', 'ar-JO', 'ar-SA', 'bg-BG', 'bn', 'ca-ES', 'cs-CZ', 'cy-GB', 'da-DK', 'de-DE', 'el-GR', 'en',
    # 'en-GB', 'en-IE', 'en-US', 'es-AR', 'es-ES', 'es-MX', 'eu-ES', 'fa', 'fa-IR', 'fi-FI', 'fr-FR', 'gl-ES', 'gu-IN',
    # 'ha-NE', 'he', 'hi-IN', 'hu-HU', 'hy-AM', 'id-ID', 'is-IS', 'it-IT', 'jv-ID', 'ka-GE', 'kk-KZ', 'ko-KO', 'lb-LU',
    # 'lv-LV', 'ml-IN', 'ne-NP', 'nl', 'nl-BE', 'nl-NL', 'no-NO', 'pl-PL', 'pt-BR', 'pt-PT', 'ro-RO', 'ru-RU', 'sk-SK',
    # 'sl-SI', 'sr-RS', 'sv-SE', 'sw', 'sw-CD', 'tdt-TL', 'te-IN', 'tn-ZA', 'tr-TR', 'uk-GB', 'uk-UA', 'vi-VN', 'yo',
    # 'zh-CN', 'zh-TW']

    manager.all_voices[0].load()