from pathlib import Path
from typing import Dict, List
import unicodedata

import requests
from json_database import JsonStorage
from langcodes import standardize_tag
from phoonnx.config import PhonemeType, Engine, Alphabet
from phoonnx.model_manager import TTSModelInfo
from phoonnx.util import LOG


class TTSModelManager:
    def __init__(self):
        self.voices: Dict[str, TTSModelInfo] = {}

    @property
    def all_voices(self) -> List[TTSModelInfo]:
        return list(self.voices.values())

    @property
    def supported_langs(self) -> List[str]:
        return sorted(set(l.lang for l in self.all_voices))

    def clear(self):
        self.voices = {}

    def save(self, json_name):
        """
        Persist current in-memory voice metadata to the configured cache storage.
        
        Clears the cache, writes each managed voice's public metadata (voice_id, model_url,
        phoneme_type, lang, tokens_url, phoneme_map_url, alphabet, config_url) into the cache,
        and then stores the cache to disk.
        """
        path = Path(__file__).parent.parent / "phoonnx" / "voice_index" / json_name
        cache = JsonStorage(str(path))
        for voice_id, voice_info in self.voices.items():
            cache[voice_id] = {"voice_id": voice_info.voice_id,
                                    "model_url": voice_info.model_url,
                                    "phoneme_type": voice_info.phoneme_type,
                                    "lang": voice_info.lang,
                                    "tokens_url": voice_info.tokens_url,
                                    "tokenizer_config_url": voice_info.tokenizer_config_url,
                                    "vocab_url": voice_info.vocab_url,
                                    "phoneme_map_url": voice_info.phoneme_map_url,
                                    "alphabet": voice_info.alphabet,
                                    "engine": voice_info.engine,
                                    "config_url": voice_info.config_url}
        cache.store()

    def add_voice(self, voice_info: TTSModelInfo):
        """
        Add or update a TTS voice in the manager's in-memory registry and persist its public metadata to the cache.
        
        This stores the given TTSModelInfo under its voice_id in memory and writes a curated subset of its fields (voice_id, model_url, tokens_url, phoneme_type, phoneme_map_url, alphabet, lang, config_url) into the persistent cache, overwriting any existing entry for the same voice_id.
        
        Parameters:
            voice_info (TTSModelInfo): The voice metadata to add or update.
        """
        self.voices[voice_info.voice_id] = voice_info

    # helpers to get official voice models
    def get_ovos_voice_list(self):
        """
        Register OpenVoiceOS phoonnx and Piper TTS voices into the manager's voice catalog.

        Adds TTSModelInfo entries for a hardcoded set of phoonnx Hugging Face repositories and for a set of common Piper languages. For each entry the method constructs model and config URLs pointing to the repository's main branch on Hugging Face and calls add_voice to register the voice. Missing voice variants are skipped silently.
        """
        phoonnx = [
            "OpenVoiceOS/phoonnx_oc_miro_unicode",
            "OpenVoiceOS/phoonnx_oc_dii_unicode",
            "OpenVoiceOS/phoonnx_an_miro_unicode",
            "OpenVoiceOS/phoonnx_an_dii_unicode",
            "OpenVoiceOS/phoonnx_ast_miro_unicode",
            "OpenVoiceOS/phoonnx_ast_dii_unicode",
            "OpenVoiceOS/phoonnx_ca_miro_espeak",
            "OpenVoiceOS/phoonnx_gl-ES_miro_unicode",
            "OpenVoiceOS/phoonnx_pt-PT_miro_unicode",
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
                engine=Engine.PHOONNX
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
                        engine=Engine.PIPER
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
                alphabet=Alphabet.UNICODE,
                engine=Engine.COQUI
            ))
        for voice in ["brais", "celtia", "sabela", "icia", "paulo", "iago"]:
            self.add_voice(TTSModelInfo(
                voice_id=f"proxectonos/{voice}-cotovia",
                lang="gl-ES",
                model_url=f"https://huggingface.co/OpenVoiceOS/proxectonos-{voice}-vits-phonemes-onnx/resolve/main/model.onnx",
                config_url=f"https://huggingface.co/OpenVoiceOS/proxectonos-{voice}-vits-phonemes-onnx/resolve/main/config.json",
                phoneme_type=PhonemeType.COTOVIA,
                alphabet=Alphabet.COTOVIA,
                engine=Engine.COQUI

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
                    voice_id="piper/" + v["key"],
                    lang=standardize_tag(v["key"].split("-")[0]),
                    model_url=base + [a for a in v["files"] if a.endswith(".onnx")][0],
                    config_url=base + [a for a in v["files"] if a.endswith(".json")][0],
                    engine=Engine.PIPER
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
                phoneme_type=PhonemeType.GORUUT,
                engine=Engine.PIPER
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
                    voice_id="mimic3/" + k,
                    lang=lang,
                    config_url=config_url,
                    tokens_url=tokens_url,
                    model_url=model_url,
                    phoneme_map_url=phoneme_map_url,
                    engine=Engine.MIMIC3
                )
                voice_info.config.lang_code = lang
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
                voice_id="phonikud/phonikud",
                lang="he-IL",
                model_url="https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.onnx",
                config_url="https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.config.json",
                phoneme_type=PhonemeType.PHONIKUD,
                engine=Engine.PIPER
            )
        )
        self.add_voice(
            TTSModelInfo(
                voice_id="phonikud/shaul",
                lang="he-IL",
                model_url="https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/shaul.onnx",
                config_url="https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.config.json",
                phoneme_type=PhonemeType.PHONIKUD,
                engine=Engine.PIPER
            )
        )

    def get_mms_voice_list(self):
        url = "https://huggingface.co/willwade/mms-tts-multilingual-models-onnx/raw/main/languages-supported.json"
        for data in requests.get(url).json():
            lang = data["Iso Code"]

            try:
                std_lang = standardize_tag(lang)
            except Exception as e:
                std_lang = lang

            ascii_lang = unicodedata.normalize('NFKD', lang).encode('ascii', 'ignore').decode('ascii')
            if lang in ["ubu", "ubl", "tzo-dialect_chenalhó"]:
                voice = TTSModelInfo(
                    voice_id=f"facebook/mms-tts-{lang}-{data['Language Name']}",
                    lang=std_lang,
                    model_url=f"https://huggingface.co/willwade/mms-tts-multilingual-models-onnx/resolve/main/{lang}/model.onnx",
                    tokens_url=f"https://huggingface.co/facebook/mms-tts/blob/main/full_models/{lang}/vocab.txt",
                    phoneme_type=PhonemeType.GRAPHEMES,
                    alphabet=Alphabet.UNICODE,
                    engine=Engine.TRANSFORMERS
                )
            else:
                voice = TTSModelInfo(
                    voice_id=f"facebook/mms-tts-{lang}-{data['Language Name']}",
                    lang=std_lang,
                    model_url=f"https://huggingface.co/willwade/mms-tts-multilingual-models-onnx/resolve/main/{lang}/model.onnx",
                    vocab_url=f"https://huggingface.co/facebook/mms-tts-{ascii_lang}/resolve/main/vocab.json",
                    tokenizer_config_url=f"https://huggingface.co/facebook/mms-tts-{ascii_lang}/resolve/main/tokenizer_config.json",
                    phoneme_type=PhonemeType.GRAPHEMES,
                    alphabet=Alphabet.UNICODE,
                    engine=Engine.TRANSFORMERS
                )

            self.add_voice(voice)

    # community models sourced from around the web
    def get_piper_community_voice_list(self):
        """
        Register a collection of community-sourced Piper TTS voices into the manager.
        
        Adds hardcoded Piper community voice entries (voice_id, lang, model_url, config_url) to the manager by calling self.add_voice. Some entries may require Hugging Face authentication or special handling (archives, nested archives), and duplicate models can appear because Piper models are sometimes merged upstream.
        """
        # https://huggingface.co/ISTNetworks/piper-qatar-tts
        voice = TTSModelInfo(
            voice_id="piper_community/" + "ISTNetworks/piper-qatar-tts",
            lang="ar-QA",
            model_url="https://huggingface.co/ISTNetworks/piper-qatar-tts/resolve/main/models/qatar_spk4_epoch2472.onnx",
            config_url="https://huggingface.co/ISTNetworks/piper-qatar-tts/resolve/main/models/qatar_spk4_epoch2472.onnx.json",
            engine=Engine.PIPER,
        )
        self.add_voice(voice)

        # https://huggingface.co/vadimbelsky/arabic-emirati-female-piper
        voice = TTSModelInfo(
            voice_id="piper_community/" + "vadimbelsky/arabic-emirati-female-piper",
            lang="ar-AE",
            model_url="https://huggingface.co/vadimbelsky/arabic-emirati-female-piper/resolve/main/arabic-emirati-female-model.onnx",
            config_url="https://huggingface.co/vadimbelsky/arabic-emirati-female-piper/resolve/main/arabic-emirati-female-model.onnx.json",
            engine=Engine.PIPER,
        )
        self.add_voice(voice)

        # https://huggingface.co/mbarnig/lb_rhasspy_piper_tts
        for voice in ["androgynous", "femaleLOD", "marylux"]:
            url = f"https://huggingface.co/mbarnig/lb_rhasspy_piper_tts/resolve/main/lb/lb_LU/{voice}/medium/lb_LU-{voice}-medium.onnx"
            voice = TTSModelInfo(
                voice_id="piper_community/" + f"mbarnig/lb-LU_{voice}",
                lang="lb-LU",
                model_url=url,
                config_url=url + ".json",
                engine=Engine.PIPER
            )
            self.add_voice(voice)

        # https://huggingface.co/superkeka/piper-tts-luka
        url = f"https://huggingface.co/superkeka/piper-tts-luka/resolve/main/ru/ru_RU/luka/medium/ru_RU-luka-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"superkeka/ru-RU_luka",
            lang="ru-RU",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/davit312/piper-TTS-Armenian
        url = f"https://huggingface.co/davit312/piper-TTS-Armenian/resolve/main/v3/hy_AM-gor-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"davit312/hy-AM_gor",
            lang="hy-AM",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/raphaelmerx/piper-voices
        url = f"https://huggingface.co/raphaelmerx/piper-voices/resolve/main/tdt/tdt_TL/joao/medium/tdt_TL-joao-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"raphaelmerx/tdt-TL_joao",
            lang="tdt-TL",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/wezzmeister/piper-voices
        url = f"https://huggingface.co/wezzmeister/piper-voices/resolve/main/sv_SE-lisa-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"wezzmeister/sv-SE_lisa",
            lang="sv-SE",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/SubZeroAI/piper-swedish-tts-multispeaker
        # TODO - 401 - needs login and approval in HF
        # url = f"https://huggingface.co/SubZeroAI/piper-swedish-tts-multispeaker/resolve/main/piper-swedish-tts-multispeaker.onnx"
        # voice = TTSModelInfo(
        #    voice_id="piper_community/" +"SubZeroAI/sv-SE_multispeaker",
        #    lang="sv-SE",
        #    model_url=url,
        #    config_url=url + ".json",
        # )
        # self.add_voice(voice)

        # https://huggingface.co/larcanio/piper-voices
        url = f"https://huggingface.co/larcanio/piper-voices/resolve/main/es_AR-daniela-high.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"larcanio/es-AR_daniela",
            lang="es-AR",
            model_url=url,
            config_url=url.replace(".onnx", ".json"),
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/friyin/vits-piper-es_ES-friyin-high
        url = f"https://huggingface.co/friyin/vits-piper-es_ES-carlfm-high/resolve/main/es_ES-carlfm-high.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"friyin/es-ES_friyin",
            lang="es-ES",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/Wiseyak/piper_tts
        url = f"https://huggingface.co/Wiseyak/piper_tts/resolve/main/ne-seto_bagh-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"Wiseyak/ne-NP_seto_bagh",
            lang="ne-NP",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/colafly/piper_zh_tw
        url = f"https://huggingface.co/colafly/piper_zh_tw/resolve/main/yt-chinese_female.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"colafly/zh-TW_yt-chinese_female",
            lang="zh-TW",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/ppisljar/piper_si_artur
        url = f"https://huggingface.co/ppisljar/piper_si_artur/resolve/main/model.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"ppisljar/sl-SI_artur",
            lang="sl-SI",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/giganticlab/piper-id_ID-news_tts-medium
        url = f"https://huggingface.co/giganticlab/piper-id_ID-news_tts-medium/resolve/main/model.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"giganticlab/id-ID_news",
            lang="id-ID",
            model_url=url,
            config_url=url.replace("model.onnx", "config.json"),
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/phcatan9921/piper_tts
        url = f"https://huggingface.co/phcatan9921/piper_tts/resolve/main/vi_VN-vais1000-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"phcatan9921/vi-VN_vais1000",
            lang="vi-VN",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://github.com/phatjkk/vits-tts-vietnamese
        url = f"https://github.com/phatjkk/vits-tts-vietnamese/raw/refs/heads/main/pretrained_vi.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"phatjkk/vi-VN_InfoRe",
            lang="vi-VN",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/RaivisDejus/Piper-lv_LV-Aivars-medium
        url = f"https://huggingface.co/RaivisDejus/Piper-lv_LV-Aivars-medium/resolve/main/lv_LV-aivars-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"RaivisDejus/lv-LV_Aivars",
            lang="lv-LV",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/PravalX/piper-voices
        for voice in ["priyamvada", "pratham"]:
            url = f"https://huggingface.co/PravalX/piper-voices/resolve/main/hi/hi_IN/{voice}/medium/hi_IN-{voice}-medium.onnx"
            voice = TTSModelInfo(
                voice_id="piper_community/" +f"PravalX/hi-IN_{voice}",
                lang="hi-IN",
                model_url=url,
                config_url=url + ".json",
                engine=Engine.PIPER
            )
            self.add_voice(voice)

        # https://huggingface.co/WitoldG/polish_piper_models
        for voice in ["jarvis", "justyna", "meski", "zenski"]:
            url = f"https://huggingface.co/WitoldG/polish_piper_models/resolve/main/pl_PL-{voice}_wg_glos-medium.onnx"
            voice = TTSModelInfo(
                voice_id="piper_community/" +f"WitoldG/pl-PL_{voice}",
                lang="pl-PL",
                model_url=url,
                config_url=url + ".json",
                engine=Engine.PIPER
            )
            self.add_voice(voice)

        # https://huggingface.co/srxz/sage-voice-pt-br
        url = "https://huggingface.co/srxz/sage-voice-pt-br/resolve/main/pt_BR-sage_13364-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"srxz/pt-BR_sage",
            lang="pt-BR",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/Thomcles/Piper-TTS-Czech
        for qual in ["medium", "high"]:
            url = f"https://huggingface.co/Thomcles/Piper-TTS-Czech/resolve/main/{qual}/model.onnx"
            voice = TTSModelInfo(
                voice_id="piper_community/" +f"Thomcles/cs-CZ_honza_{qual}",
                lang="cs-CZ",
                model_url=url,
                config_url=url + ".json",
                engine=Engine.PIPER
            )
            self.add_voice(voice)

        # https://huggingface.co/AsmoKoskinen/Piper_Finnish_Model
        url = "https://huggingface.co/AsmoKoskinen/Piper_Finnish_Model/resolve/main/fi_FI-asmo-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"AsmoKoskinen/fi-FI_asmo",
            lang="fi-FI",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/gyroing/Persian-Piper-Model-gyro
        url = "https://huggingface.co/gyroing/Persian-Piper-Model-gyro/resolve/main/fa_IR-gyro-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"gyroing/fa-IR_gyro",
            lang="fa-IR",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/mah92/Reza-And-Ibrahim-FA_EN-Piper-TTS-Model
        url = "https://huggingface.co/mah92/Reza-And-Ibrahim-FA_EN-Piper-TTS-Model/resolve/main/fa_en-rezahedayatfar-ibrahimwalk-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"mah92/fa-IR_Reza-And-Ibrahim",
            lang="fa-IR",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/Einrich99/PiperTTS-UGO-Italian
        url = "https://huggingface.co/Einrich99/PiperTTS-UGO-Italian/resolve/main/medium/it_IT-ugo-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"Einrich99/it-IT_ugo",
            lang="it-IT",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/paolapersico1/Piper-TTS-Italian
        url = "https://huggingface.co/paolapersico1/Piper-TTS-Italian/resolve/main/paola/medium/it_IT-paola-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"paolapersico1/it-IT_paola",
            lang="it-IT",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/kirys79/piper_italiano
        url = f"https://huggingface.co/kirys79/piper_italiano/resolve/main/Aurora/it_IT-aurora-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"kirys79/it-IT_Aurora",
            lang="it-IT",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)
        url = "https://huggingface.co/kirys79/piper_italiano/resolve/main/Giorgio/giorgio-epoch%3D5028-step%3D1098436.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"kirys79/it-IT_Giorgio",
            lang="it-IT",
            model_url=url,
            config_url=url.replace(".onnx", ".json"),
            engine=Engine.PIPER
        )
        self.add_voice(voice)
        url = f"https://huggingface.co/kirys79/piper_italiano/resolve/main/Leonardo/leonardo-epoch%3D2024-step%3D996300.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"kirys79/it-IT_Leonardo",
            lang="it-IT",
            model_url=url,
            config_url=url.replace(".onnx", ".json"),
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/nardocolin/nardocolin-pipertts
        url = "https://huggingface.co/nardocolin/nardocolin-pipertts/resolve/main/high/colin-voice_high.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"nardocolin/en-GB_Colin",
            lang="en-GB",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/Da-Bob/piper-mikev3
        url = "https://huggingface.co/Da-Bob/piper-mikev3/resolve/main/mikev3.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"Da-Bob/en-US_mikev3",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/agentvibes/piper-custom-voices
        for voice, lang in [("kristin", "en-US"), ("jenny", "en-IE"), ("16Speakers", "en")]:
            url = f"https://huggingface.co/agentvibes/piper-custom-voices/resolve/main/{voice}.onnx"
            voice = TTSModelInfo(
                voice_id="piper_community/" +f"agentvibe/{lang}_{voice}",
                lang=lang,
                model_url=url,
                config_url=url + ".json",
                engine=Engine.PIPER
            )
            self.add_voice(voice)

        # HAV0X1014/KF-PiperTTS-voices
        for voice, qual in [("Cheetah", "high"), ("KingCheetah", "medium"), ("silverfox", "medium")]:
            url = f"https://huggingface.co/HAV0X1014/KF-PiperTTS-voices/resolve/main/{voice}/en_US-{voice.lower()}-{qual}.onnx"
            voice = TTSModelInfo(
                voice_id="piper_community/" +f"agentvibe/en-US_{voice}",
                lang="en-US",
                model_url=url,
                config_url=url + ".json",
                engine=Engine.PIPER
            )
            self.add_voice(voice)

        # https://huggingface.co/campwill/HAL-9000-Piper-TTS
        url = "https://huggingface.co/campwill/HAL-9000-Piper-TTS/resolve/main/hal.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"campwill/en-US_HAL-9000",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/redromnon/piper-tts-elise
        url = "https://huggingface.co/redromnon/piper-tts-elise/resolve/main/en_US-elisa-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"redromnon/en-US_elise",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/poisson-fish/piper-vasco
        url = "https://huggingface.co/poisson-fish/piper-vasco/resolve/main/onnx/vasco.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"poisson-fish/en-US_vasco",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/rokeya71/VITS-Piper-GlaDOS-en-onnx
        url = "https://huggingface.co/rokeya71/VITS-Piper-GlaDOS-en-onnx/resolve/main/glados.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"rokeya71/en-US_GlaDOS",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/Aquaaa123/piper-tts-pda-subnautica
        url = "https://huggingface.co/Aquaaa123/piper-tts-pda-subnautica/resolve/main/pda.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"Aquaaa123/en-US_pda-subnautica",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/drewThomasson/piper_tts_finetune_death_from_puss_and_boots
        url = "https://huggingface.co/drewThomasson/piper_tts_finetune_death_from_puss_and_boots/resolve/main/en_US-death-high_onnx/en_US-death-high.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +"drewThomasson/en-US_death_from_puss_and_boots",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/samarthshrivas/piper-finetune-Andrew-Huberman
        url = "https://huggingface.co/samarthshrivas/piper-finetune-Andrew-Huberman/resolve/main/lightning_logs/version_2/checkpoints/epoch%3D2609-step%3D1364440.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +f"samarthshrivas/en-US_Andrew-Huberman",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/swqg-messiah/kusaal_chitti_piper
        url = "https://huggingface.co/swqg-messiah/kusaal_chitti_piper/resolve/main/chitti.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +f"swqg-messiah/en-US_chitti",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://brycebeattie.com/files/tts
        for voice, model, lang in [
            ("LJSpeech-medium", "lj-med", "en-US"),
            ("LJSpeech-high", "ljspeech", "en-US"),
            ("Jenny-Dioco", "jenny", "en-GB"),
            ("Clean100", "clean100", "en-US"),
            ("Cori-high", "cori-high", "en-GB"),
            ("Cori-medium", "cori-med", "en-GB"),
            ("Kristin", "kristin", "en-US"),
            ("John", "john", "en-US"),
            ("Bryce", "bryce", "en-US"),
            ("Norman", "norman", "en-US"),
            ("ManyVoice", "mv2", "en"),
        ]:
            url = f"https://sfo3.digitaloceanspaces.com/bkmdls/{model}.onnx"
            voice = TTSModelInfo(
                voice_id="piper_community/" +f"brycebeattie/{lang}_{voice}",
                lang=lang,
                model_url=url,
                config_url=url + ".json",
                engine=Engine.PIPER
            )
            self.add_voice(voice)

        # https://github.com/simoniz0r/piper-voice-models
        for voice in ["bobby", "carl", "eminem", "patrick"]:
            url = f"https://github.com/simoniz0r/piper-voice-models/releases/download/{voice}/en_US-{voice}-medium.onnx"
            voice = TTSModelInfo(
                voice_id="piper_community/" +f"simoniz0r/en-US_{voice}",
                lang="en-US",
                model_url=url,
                config_url=url + ".json",
                engine=Engine.PIPER
            )
            self.add_voice(voice)

        # https://github.com/dividebysandwich/piper-voice-models
        for voice, model in [("Data", "en_US-data_7024-medium"),
                             ("Picard", "en_US-picard_7399-medium"),
                             ("HAL9000-denoised", "en_US-hal_6409-medium"),
                             ("HAL9000-no-denoise", "en_US-hal_12894-medium")]:
            url = f"https://github.com/dividebysandwich/piper-voice-models/raw/refs/heads/main/{voice}/{model}.onnx"
            voice = TTSModelInfo(
                voice_id="piper_community/" +f"dividebysandwich/en-US_{voice}",
                lang="en-US",
                model_url=url,
                config_url=url + ".json",
                engine=Engine.PIPER
            )
            self.add_voice(voice)

        # https://huggingface.co/russdill/kronk
        url = "https://huggingface.co/russdill/kronk/resolve/main/en/en_US/kronk/medium/kronk-medium.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +f"russdill/en-US_kronk",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/davet2001/cave_johnson1
        url = "https://huggingface.co/davet2001/cave_johnson1/resolve/main/cave_johnson1.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +f"davet2001/en-US_cave_johnson",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/davet2001/wheatley1
        url = "https://huggingface.co/davet2001/wheatley1/resolve/main/wheatley1.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +f"davet2001/en-US_wheatley",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://github.com/robit-man/combine_overwatch_onnx
        url = "https://github.com/robit-man/combine_overwatch_onnx/raw/refs/heads/main/overwatch.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +f"robit-man/en-US_overwatch",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://github.com/DJMalachite/PiperVoiceModels
        url = "https://github.com/DJMalachite/PiperVoiceModels/raw/refs/heads/main/Titanfall2/BT7274/BT7274.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +f"DJMalachite/en-US_BT7274",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://github.com/hopkira/k9_piper_voice
        url = "https://github.com/hopkira/k9_piper_voice/raw/refs/heads/main/k9_model.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +f"hopkira/en-US_k9",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)
        url = "https://github.com/hopkira/k9_piper_voice/raw/refs/heads/main/k9_2449_model.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +f"hopkira/en-US_k9_2449",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://github.com/1liminal1/xiaozhi-esphome
        url = "https://github.com/1liminal1/xiaozhi-esphome/raw/refs/heads/main/piper-voices/en_US-bmo_voice.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +f"1liminal1/en-US_bmo",
            lang="en-US",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/jstlntch/Scaramouche_or_Wanderer_voice_model_for_piper
        url = "https://huggingface.co/jstlntch/Scaramouche_or_Wanderer_voice_model_for_piper/resolve/main/model.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +f"jstlntchh/en_Scaramouche",
            lang="en",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/Rikels/piper-dutch
        url = "https://huggingface.co/Rikels/piper-dutch/resolve/main/anna.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +f"Rikels/nl-NL_anna",
            lang="nl-NL",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/systemofapwne/piper-de-glados
        for qual in ["high", "medium", "low"]:
            url = f"https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados/{qual}/de_DE-glados-{qual}.onnx"
            voice = TTSModelInfo(
                voice_id="piper_community/" +f"systemofapwne/de-DE_glados_{qual}",
                lang="de-DE",
                model_url=url,
                config_url=url + ".json",
                engine=Engine.PIPER
            )
            self.add_voice(voice)

            url = f"https://huggingface.co/systemofapwne/piper-de-glados/resolve/main/de/de_DE/glados-turret/{qual}/de_DE-glados-turret-{qual}.onnx"
            voice = TTSModelInfo(
                voice_id="piper_community/" +f"systemofapwne/de-DE_glados-turret_{qual}",
                lang="de-DE",
                model_url=url,
                config_url=url + ".json",
                engine=Engine.PIPER
            )
            self.add_voice(voice)

        # https://huggingface.co/nullnullvier/kantodel
        url = "https://huggingface.co/nullnullvier/kantodel/resolve/main/kantodel.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +f"nullnullvier/de-DE_kantodel",
            lang="de-DE",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
        )
        self.add_voice(voice)

        # https://huggingface.co/domoskanonos/piper-tts-models
        for voice, qual in [("domoskanonos", "high"), ("sebastian100", "medium"), ("sebastian121", "medium")]:
            url = f"https://huggingface.co/domoskanonos/piper-tts-models/resolve/main/de-{voice}-{qual}.onnx"
            voice = TTSModelInfo(
                voice_id="piper_community/" +f"domoskanonos/de-DE_{voice}",
                lang="de-DE",
                model_url=url,
                config_url=url + ".json",
                engine=Engine.PIPER
            )
            self.add_voice(voice)

        # TODO - unknown phonemizer type?
        # https://huggingface.co/tiennguyenbnbk/male_vivoice_piper_viphone

        # TODO - these models are inside a .tar.gz/.zip and will need special handling
        # https://github.com/GraceDabbieri/piper-tts-voices
        # https://huggingface.co/MysticonsLover/PiperWillowbrook
        # https://huggingface.co/BibEBobberson/Piper
        # https://huggingface.co/Beesa/Piper_brawlstars
        # https://huggingface.co/BornSaint/piper-TTS

        # https://huggingface.co/HirCoir/Piper-TTS-Laura
        url = f"https://huggingface.co/HirCoir/Piper-TTS-Laura/resolve/main/es_MX-laura-high.onnx"
        voice = TTSModelInfo(
            voice_id="piper_community/" +f"HirCoir/es-MX_Laura",
            lang="es-MX",
            model_url=url,
            config_url=url + ".json",
            engine=Engine.PIPER
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

        # TODO - these will need to be mirrored somewhere else to allow download
        # https://www.nexusmods.com/skyrimspecialedition/mods/98631
        # https://www.nexusmods.com/fallout4/mods/79747

    def get_coqui_community_voice_list(self):
        """
        Add Coqui community voice entries to the manager.

        This method is a placeholder and currently performs no action. Intended to discover Coqui community TTS model manifests and add corresponding TTSModelInfo entries to self.voices when implemented.
        """
        # https://huggingface.co/z-uo/vits-male-it/  TODO
        # https://huggingface.co/z-uo/vits-female-it/  TODO
        # https://huggingface.co/z-uo/vits-commonvoice9.0/tree/main TODO
        # https://huggingface.co/SmartGitiCorp/persian_tts_vits/tree/main

        # projecte-aina/tts-ca-coqui-vits-multispeaker
        voice = TTSModelInfo(
            voice_id=f"hf_community/projecte-aina/tts-ca-coqui-vits-multispeaker",
            lang="ca-ES",
            model_url="https://huggingface.co/projecte-aina/tts-ca-coqui-vits-multispeaker/resolve/main/model/vits_ca.onnx",
            config_url="https://huggingface.co/projecte-aina/tts-ca-coqui-vits-multispeaker/resolve/main/model/config.json",
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.UNICODE,
            engine=Engine.COQUI
        )
        self.add_voice(voice)

        # denZLS/luxembourgish-male-vits-tts
        voice = TTSModelInfo(
            voice_id=f"hf_community/denZLS/luxembourgish-male-vits-tts",
            lang="lb-LU",
            model_url="https://huggingface.co/Jarbas/luxembourgish-male-vits-tts-onnx/resolve/main/checkpoint_53442.pth.onnx",
            config_url="https://huggingface.co/Jarbas/luxembourgish-male-vits-tts-onnx/resolve/main/config.json",
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.UNICODE,
            engine=Engine.COQUI
        )
        self.add_voice(voice)

        # denZLS/luxembourgish-female-vits-tts
        voice = TTSModelInfo(
            voice_id=f"hf_community/denZLS/luxembourgish-female-vits-tts",
            lang="lb-LU",
            model_url="https://huggingface.co/Jarbas/luxembourgish-female-vits-tts-onnx/resolve/main/checkpoint_53442.pth.onnx",
            config_url="https://huggingface.co/Jarbas/luxembourgish-female-vits-tts-onnx/resolve/main/config.json",
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.UNICODE,
            engine=Engine.COQUI
        )
        self.add_voice(voice)

        # anzorq/kbd-vits-tts-female
        base = "https://huggingface.co/anzorq/kbd-vits-tts-female/resolve/main"
        voice = TTSModelInfo(
            voice_id=f"hf_community/anzorq/kbd-vits-tts-female",
            lang="kbd",
            model_url=f"{base}/onnx/kbd_vits_female.onnx",
            config_url=f"{base}/config.json",
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.UNICODE,
            engine=Engine.COQUI
        )
        self.add_voice(voice)

        # anzorq/kbd-vits-tts-male
        base = "https://huggingface.co/anzorq/kbd-vits-tts-male/resolve/main"
        voice = TTSModelInfo(
            voice_id=f"hf_community/anzorq/kbd-vits-tts-male",
            lang="kbd",
            model_url=f"{base}/onnx/kbd_vits_male.onnx",
            config_url=f"{base}/config.json",
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.UNICODE,
            engine=Engine.COQUI
        )
        self.add_voice(voice)


        # jerichosiahaya/vits-tts-id
        # TODO - no onnx model, convert and upload to hf
        #voice = TTSModelInfo(
        #    voice_id=f"hf_community/jerichosiahaya/vits-tts-id",
        #    lang="id-ID",
        #    model_url="https://huggingface.co/jerichosiahaya/vits-tts-id/resolve/main/model.onnx",
        #    config_url="https://huggingface.co/jerichosiahaya/vits-tts-id/resolve/main/config.json",
        #    phoneme_type=PhonemeType.GRAPHEMES,
        #    alphabet=Alphabet.UNICODE,
        #    engine=Engine.COQUI
        #)
        #self.add_voice(voice)

    def get_transformers_community_voice_list(self):
        # TODO - models that need to be converted to onnx
        # https://huggingface.co/ylacombe/models?search=vits
        # https://huggingface.co/ylacombe/models?search=mms
        # https://huggingface.co/mahwizzzz/vits-ur
        # https://huggingface.co/wasmdashai/vits-ar
        # https://huggingface.co/wasmdashai/vits-ar-sa-A
        # https://huggingface.co/wasmdashai/vits-ar-sa-huba-v2
        # https://huggingface.co/wasmdashai/vits-en-v1
        # https://huggingface.co/wasmdashai/vits-ar-ye-sa
        # https://huggingface.co/wasmdashai/vits-eng-us-ljs

        # https://huggingface.co/Jarbas/NorHsangPha-mms-tts-shn-onnx
        lang = "shn"
        base = "https://huggingface.co/Jarbas/NorHsangPha-mms-tts-shn-onnx/resolve/main"
        voice = TTSModelInfo(
            voice_id=f"hf_community/NorHsangPha/mms-tts-{lang}-Shan",
            lang=lang,
            model_url=f"{base}/onnx/model.onnx",
            vocab_url=f"{base}/vocab.json",
            tokenizer_config_url=f"{base}/tokenizer_config.json",
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.UNICODE,
            engine=Engine.TRANSFORMERS
        )
        self.add_voice(voice)


        # https://huggingface.co/ylacombe/vits_ljs_welsh_female_monospeaker
        base = "https://huggingface.co/BricksDisplay/vits-eng-welsh-female/resolve/main"
        voice = TTSModelInfo(
            voice_id=f"hf_community/ylacombe/vits_ljs_welsh_female_monospeaker",
            lang="en-cy",  # TODO - is this model welsh, or english with welsh accent?
            model_url=f"{base}/onnx/model.onnx",
            vocab_url=f"{base}/vocab.json",
            tokenizer_config_url=f"{base}/tokenizer_config.json",
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.UNICODE,
            engine=Engine.TRANSFORMERS
        )
        self.add_voice(voice)

        # https://huggingface.co/kakao-enterprise/vits-ljs
        base = "https://huggingface.co/BricksDisplay/vits-eng/resolve/main"
        voice = TTSModelInfo(
            voice_id=f"hf_community/kakao-enterprise/vits-ljs-eng",
            lang="en-US",
            model_url=f"{base}/onnx/model.onnx",
            vocab_url=f"{base}/vocab.json",
            tokenizer_config_url=f"{base}/tokenizer_config.json",
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.UNICODE,
            engine=Engine.TRANSFORMERS
        )
        self.add_voice(voice)

        # https://huggingface.co/BricksDisplay/vits-cmn
        base = "https://huggingface.co/BricksDisplay/vits-cmn/resolve/main"
        voice = TTSModelInfo(
            voice_id=f"hf_community/BricksDisplay/vits-cmn",
            lang="zh-CN",
            model_url=f"{base}/onnx/model.onnx",
            vocab_url=f"{base}/vocab.json",
            tokenizer_config_url=f"{base}/tokenizer_config.json",
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.PINYIN,
            engine=Engine.TRANSFORMERS
        )
        self.add_voice(voice)

        # https://huggingface.co/joefox/tts_vits_ru_hf
        # TODO - no onnx
        #base = f"https://huggingface.co/joefox/tts_vits_ru_hf/resolve/main"
        #voice = TTSModelInfo(
        #    voice_id=f"hf_community/joefox/tts_vits_ru_hf",
        #    lang="ru-RU",
        #    model_url=f"{base}/model.onnx",
        #    vocab_url=f"{base}/vocab.json",
        #    tokenizer_config_url=f"{base}/tokenizer_config.json",
        #    phoneme_type=PhonemeType.GRAPHEMES,
        #    alphabet=Alphabet.UNICODE,
        #    engine=Engine.TRANSFORMERS
        #)
        #self.add_voice(voice)

        # https://huggingface.co/utrobinmv/tts_ru_free_hf_vits_low_multispeaker
        for qual in ["low"]: # , "high"  # TODO - high has no onnx
            base = f"https://huggingface.co/utrobinmv/tts_ru_free_hf_vits_{qual}_multispeaker/resolve/main"
            voice = TTSModelInfo(
                voice_id=f"hf_community/utrobinmv/tts_ru_free_hf_vits_{qual}_multispeaker",
                lang="ru-RU",
                model_url=f"{base}/model.onnx",
                vocab_url=f"{base}/vocab.json",
                tokenizer_config_url=f"{base}/tokenizer_config.json",
                phoneme_type=PhonemeType.GRAPHEMES,
                alphabet=Alphabet.UNICODE,
                engine=Engine.TRANSFORMERS
            )
            self.add_voice(voice)

        # https://huggingface.co/utrobinmv/tts_ru_free_hf_vits_low_multispeaker
        base = "https://huggingface.co/utrobinmv/tts_ru_free_hf_vits_low_multispeaker/resolve/main"
        voice = TTSModelInfo(
            voice_id=f"hf_community/utrobinmv/tts_ru_free_hf_vits_low_multispeaker",
            lang="ru-RU",
            model_url=f"{base}/model.onnx",
            vocab_url=f"{base}/vocab.json",
            tokenizer_config_url=f"{base}/tokenizer_config.json",
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.UNICODE,
            engine=Engine.TRANSFORMERS
        )
        self.add_voice(voice)

        # https://huggingface.co/ylacombe/mms-mar-finetuned-monospeaker
        base = "https://huggingface.co/ylacombe/mms-mar-finetuned-monospeaker/resolve/main"
        voice = TTSModelInfo(
            voice_id=f"hf_community/ylacombe/mms-mar-finetuned-monospeaker",
            lang="mar",
            model_url=f"{base}/onnx/model.onnx",
            vocab_url=f"{base}/vocab.json",
            tokenizer_config_url=f"{base}/tokenizer_config.json",
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.UNICODE,
            engine=Engine.TRANSFORMERS
        )
        self.add_voice(voice)

        # https://huggingface.co/ylacombe/mms-guj-finetuned-monospeaker
        base = "https://huggingface.co/ylacombe/mms-guj-finetuned-monospeaker/resolve/main"
        voice = TTSModelInfo(
            voice_id=f"hf_community/ylacombe/mms-guj-finetuned-monospeaker",
            lang="guj",
            model_url=f"{base}/onnx/model.onnx",
            vocab_url=f"{base}/vocab.json",
            tokenizer_config_url=f"{base}/tokenizer_config.json",
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.UNICODE,
            engine=Engine.TRANSFORMERS
        )
        self.add_voice(voice)

        # https://huggingface.co/ylacombe/mms-tam-finetuned-monospeaker
        base = "https://huggingface.co/ylacombe/mms-tam-finetuned-monospeaker/resolve/main"
        voice = TTSModelInfo(
            voice_id=f"hf_community/ylacombe/mms-tam-finetuned-monospeaker",
            lang="tam",
            model_url=f"{base}/onnx/model.onnx",
            vocab_url=f"{base}/vocab.json",
            tokenizer_config_url=f"{base}/tokenizer_config.json",
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.UNICODE,
            engine=Engine.TRANSFORMERS
        )
        self.add_voice(voice)

        # https://huggingface.co/ylacombe/mms-spa-finetuned-colombian-monospeaker
        base = "https://huggingface.co/ylacombe/mms-spa-finetuned-colombian-monospeaker/resolve/main"
        voice = TTSModelInfo(
            voice_id=f"hf_community/ylacombe/mms-spa-finetuned-colombian-monospeaker",
            lang="es-CO",
            model_url=f"{base}/onnx/model.onnx",
            vocab_url=f"{base}/vocab.json",
            tokenizer_config_url=f"{base}/tokenizer_config.json",
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.UNICODE,
            engine=Engine.TRANSFORMERS
        )
        self.add_voice(voice)

        # https://huggingface.co/ylacombe/mms-spa-finetuned-argentinian-monospeaker
        base = "https://huggingface.co/ylacombe/mms-spa-finetuned-argentinian-monospeaker/resolve/main"
        voice = TTSModelInfo(
            voice_id=f"hf_community/ylacombe/mms-spa-finetuned-argentinian-monospeaker",
            lang="es-AR",
            model_url=f"{base}/onnx/model.onnx",
            vocab_url=f"{base}/vocab.json",
            tokenizer_config_url=f"{base}/tokenizer_config.json",
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.UNICODE,
            engine=Engine.TRANSFORMERS
        )
        self.add_voice(voice)

        # https://huggingface.co/ylacombe/mms-spa-finetuned-chilean-monospeaker
        base = "https://huggingface.co/ylacombe/mms-spa-finetuned-chilean-monospeaker/resolve/main"
        voice = TTSModelInfo(
            voice_id=f"hf_community/ylacombe/mms-spa-finetuned-chilean-monospeaker",
            lang="es-CL",
            model_url=f"{base}/onnx/model.onnx",
            vocab_url=f"{base}/vocab.json",
            tokenizer_config_url=f"{base}/tokenizer_config.json",
            phoneme_type=PhonemeType.GRAPHEMES,
            alphabet=Alphabet.UNICODE,
            engine=Engine.TRANSFORMERS
        )
        self.add_voice(voice)


        # https://huggingface.co/ai4bharat/vits_rasa_13
        # https://huggingface.co/Sigurdur/vits_icelandic_rosa_female_monospeaker/tree/main


if __name__ == "__main__":
    manager = TTSModelManager()

    def clear():
        print(f"Total voices: {len(manager.all_voices)}")
        print(f"Total langs: {len(manager.supported_langs)}")
        manager.clear()

    def sync_ovos():
        manager.get_ovos_voice_list()
        manager.save("OVOS.json")
        clear()


    def sync_nos():
        manager.get_proxectonos_voice_list()
        manager.save("proxectonos.json")
        clear()


    def sync_piper():
        manager.get_piper_voice_list()
        manager.save("piper.json")
        clear()


    def sync_neurlang():
        manager.get_neurlang_voice_list()
        manager.save("neurlang.json")
        clear()


    def sync_mimic3():
        manager.get_mimic3_voice_list()
        manager.save("mimic3.json")
        clear()


    def sync_phonikud():
        manager.get_phonikud_voice_list()
        manager.save("phonikud.json")
        clear()


    def sync_transformers():
        manager.get_transformers_community_voice_list()
        manager.save("transformers_community.json")
        clear()


    def sync_piper_community():
        manager.get_piper_community_voice_list()
        manager.save("piper_community.json")
        clear()


    def sync_coqui():
        manager.get_coqui_community_voice_list()
        manager.save("coqui_community.json")
        clear()


    def sync_mms():
        manager.get_mms_voice_list()
        manager.save("MMS.json")
        clear()

    sync_ovos()
    sync_mms()
    sync_nos()
    sync_coqui()
    sync_piper()
    sync_mimic3()
    sync_neurlang()
    sync_transformers()
    sync_phonikud()
    sync_piper_community()
