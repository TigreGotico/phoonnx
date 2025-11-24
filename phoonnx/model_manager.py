import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any
import requests
from json_database import JsonStorageXDG, JsonStorage

from phoonnx.config import PhonemeType, get_phonemizer, VoiceConfig, Engine, Alphabet
from phoonnx.util import match_lang, normalize_lang
from phoonnx.voice import TTSVoice


@dataclass
class TTSModelInfo:
    voice_id: str
    lang: str  # not always present in config.json and often wrong if present
    model_url: str
    config_url: Optional[str] = None  # some models only provide tokens.txt
    vocab_url: Optional[str] = None  # transformers provides vocab.json with tokens
    tokenizer_config_url: Optional[str] = None  # transformers provides tokenizer_config.json with metadata
    tokens_url: Optional[str] = None  # mimic3/sherpa provide phoneme_map in this format
    phoneme_map_url: Optional[str] = None  # json lookup table for phoneme replacement
    config: Optional[VoiceConfig] = None
    phoneme_type: Optional[PhonemeType] = None
    alphabet: Optional[Alphabet] = None
    engine: Optional[Engine] = None
    vocab_override: Optional[Dict[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        """
        Initialize the TTSModelInfo instance by ensuring local cache files exist and synchronizing its configuration, alphabet, and phoneme type.
        
        If no VoiceConfig was provided, ensure the voice cache directory exists, download and load the model config (model.json), apply a known phoneme-type compatibility fix, and—when a tokens URL is present—download the tokens file and construct the VoiceConfig using it. Always set the loaded config's language code from this instance's `lang`. After loading (or when a config was provided), ensure `alphabet` and `phoneme_type` on the dataclass and on the loaded config are consistent by propagating values from whichever side is present.
        """
        os.makedirs(self.voice_path, exist_ok=True)
        if not self.config:
            if self.config_url:
                config = self.download_config()
                # HACK: seen in some published piper voices
                # "es_MX-ald-medium"
                if config.get('phoneme_type', "") == "PhonemeType.ESPEAK":
                    config["phoneme_type"] = "espeak"
            else:
                config = {"phoneme_type": "graphemes", "alphabet": "unicode"}

            if self.vocab_override:
                self.config = VoiceConfig.from_dict(config, vocab=self.vocab_override)
            elif self.vocab_url:
                vocab = self.download_vocab()
                if self.tokenizer_config_url:
                    tokenizer_config = self.download_tokenizer_config()
                else:
                    tokenizer_config = {}
                self.config = VoiceConfig.from_dict(config, vocab=vocab, tokenizer_config=tokenizer_config)
            if self.tokens_url:
                self.download_tokens_txt()
                self.config = VoiceConfig.from_dict(config, tokens_txt=str(self.voice_path / "tokens.txt"))

            if self.phoneme_type:
                config["phoneme_type"] = self.phoneme_type

            self.config = self.config or VoiceConfig.from_dict(config)
            self.config.lang_code = self.lang  # sometimes the config is wrong

        self.config.lang_code = self.lang = normalize_lang(self.config.lang_code)

        if not self.alphabet:
            self.alphabet = self.config.alphabet
        else:
            self.config.alphabet = self.alphabet

        if not self.phoneme_type:
            self.phoneme_type = self.config.phoneme_type
        else:
            self.config.phoneme_type = self.phoneme_type

        if not self.engine:
            self.engine = self.config.engine
        else:
            self.config.engine = self.engine

        # cast strings to enum for consistency
        if not isinstance(self.engine, Engine) and isinstance(self.engine, str):
            self.engine = Engine(self.engine)
        if not isinstance(self.alphabet, Alphabet) and isinstance(self.alphabet, str):
            self.alphabet = Alphabet(self.alphabet)
        if not isinstance(self.phoneme_type, PhonemeType) and isinstance(self.phoneme_type, str):
            self.phoneme_type = PhonemeType(self.phoneme_type)

    @property
    def voice_path(self) -> Path:
        return Path(os.path.expanduser("~")) / ".cache" / "phoonnx" / "voices" / self.voice_id

    def download_config(self) -> Dict[str, Any]:
        config_path = self.voice_path / "model.json"
        if not config_path.is_file():
            r = requests.get(self.config_url, timeout=30)
            r.raise_for_status()
            cfg = r.json()  # validate received json
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=4)
            return cfg
        with open(config_path, "r") as f:
            return json.load(f)

    def download_tokenizer_config(self) -> Dict[str, Any]:
        config_path = self.voice_path / "tokenizer_config.json"
        if not config_path.is_file():
            r = requests.get(self.tokenizer_config_url, timeout=30)
            r.raise_for_status()
            cfg = r.json()  # validate received json
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=4)
            return cfg
        with open(config_path, "r") as f:
            return json.load(f)

    def download_vocab(self) -> Dict[str, Any]:
        vocab_path = self.voice_path / "vocab.json"
        if self.vocab_url and not vocab_path.is_file():
            r = requests.get(self.vocab_url, timeout=30)
            r.raise_for_status()
            cfg = r.json()
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            return cfg
        with open(vocab_path, "r") as f:
            return json.load(f)

    def download_tokens_txt(self) -> str:
        tokens_path = self.voice_path / "tokens.txt"
        if self.tokens_url and not tokens_path.is_file():
            r = requests.get(self.tokens_url, timeout=30)
            r.raise_for_status()
            tokens = r.text
            with open(tokens_path, "w", encoding="utf-8") as f:
                f.write(tokens)
            return tokens
        with open(tokens_path, "r") as f:
            return f.read()

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
        vocab_path = self.voice_path / "vocab.json"
        tokenizer_config_path = self.voice_path / "tokenizer_config.json"
        tokens_path = self.voice_path / "tokens.txt"
        self.download_model()

        voice = TTSVoice.load(model_path=model_path,
                              config_path=config_path,
                              vocab_path=vocab_path,
                              tokenizer_config_path=tokenizer_config_path,
                              lang_code=self.config.lang_code,
                              phoneme_type_str=self.config.phoneme_type,
                              alphabet_str=self.config.alphabet,
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
                                    "tokenizer_config_url": voice_info.tokenizer_config_url,
                                    "vocab_url": voice_info.vocab_url,
                                    "phoneme_map_url": voice_info.phoneme_map_url,
                                    "alphabet": voice_info.alphabet,
                                    "engine": voice_info.engine,
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
                                           "tokenizer_config_url": voice_info.tokenizer_config_url,
                                           "vocab_url": voice_info.vocab_url,
                                           "phoneme_type": voice_info.phoneme_type,
                                           "phoneme_map_url": voice_info.phoneme_map_url,
                                           "alphabet": voice_info.alphabet,
                                           "engine": voice_info.engine,
                                           "lang": voice_info.lang,
                                           "config_url": voice_info.config_url}

    def get_lang_voices(self, lang: str) -> List[TTSModelInfo]:
        voices = sorted(
            [
                (voice_info, match_lang(voice_info.lang, lang)[-1])
                for voice_info in self.voices.values()
            ], key=lambda k: k[1])
        return [v[0] for v in voices if v[1] < 10]

    def merge_default_voices(self, store=False):
        base_path = Path(os.path.dirname(__file__)) / "voice_index"
        self.cache.update(JsonStorage(str(base_path / "OVOS.json")))
        self.cache.update(JsonStorage(str(base_path / "MMS.json")))
        self.cache.update(JsonStorage(str(base_path / "proxectonos.json")))
        self.cache.update(JsonStorage(str(base_path / "piper.json")))
        self.cache.update(JsonStorage(str(base_path / "phonikud.json")))
        self.cache.update(JsonStorage(str(base_path / "neurlang.json")))
        self.cache.update(JsonStorage(str(base_path / "mimic3.json")))
        self.cache.update(JsonStorage(str(base_path / "transformers_community.json")))
        self.cache.update(JsonStorage(str(base_path / "piper_community.json")))
        self.voices = {voice_id: TTSModelInfo(**voice_dict)
                       for voice_id, voice_dict in self.cache.items()}
        if store:
            self.cache.store()



if __name__ == "__main__":
    manager = TTSModelManager()
    manager.cache.clear()
    manager.merge_default_voices(store=True)  # load and cache known voice models

    print(f"Total voices: {len(manager.all_voices)}")
    print(f"Total langs: {len(manager.supported_langs)}")

    # Total voices: 1462
    # Total langs: 1205

    for voice in manager.get_lang_voices("gl"):
        print(voice)
        # TTSModelInfo(voice_id='proxectonos/brais', lang='gl-ES', model_url='https://huggingface.co/OpenVoiceOS/proxectonos-brais-vits-graphemes-onnx/resolve/main/model.onnx', config_url='https://huggingface.co/OpenVoiceOS/proxectonos-brais-vits-graphemes-onnx/resolve/main/config.json', vocab_url=None, tokenizer_config_url=None, tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=0, num_langs=1, sample_rate=16000, lang_code='gl-ES', phoneme_type='graphemes', alphabet='unicode', phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine='coqui', length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, add_diacritics=False, tokenizer=TTSTokenizer(vocabulary=Vocabulary(char2idx={'<PAD>': 0, '!': 1, '¡': 2, "'": 3, '(': 4, ')': 5, ',': 6, '-': 7, '.': 8, ':': 9, ';': 10, '¿': 11, '?': 12, ' ': 13, '"': 14, '\n': 15, 'A': 16, 'B': 17, 'C': 18, 'D': 19, 'E': 20, 'F': 21, 'G': 22, 'H': 23, 'I': 24, 'J': 25, 'K': 26, 'L': 27, 'M': 28, 'N': 29, 'O': 30, 'P': 31, 'Q': 32, 'R': 33, 'S': 34, 'T': 35, 'U': 36, 'V': 37, 'W': 38, 'X': 39, 'Y': 40, 'Z': 41, 'Ç': 42, 'Á': 43, 'É': 44, 'Í': 45, 'Ï': 46, 'Ó': 47, 'Ú': 48, 'Ü': 49, 'a': 50, 'b': 51, 'c': 52, 'd': 53, 'e': 54, 'f': 55, 'g': 56, 'h': 57, 'i': 58, 'j': 59, 'k': 60, 'l': 61, 'm': 62, 'n': 63, 'o': 64, 'p': 65, 'q': 66, 'r': 67, 's': 68, 't': 69, 'u': 70, 'v': 71, 'w': 72, 'x': 73, 'y': 74, 'z': 75, 'ñ': 76, 'á': 77, 'é': 78, 'í': 79, 'ï': 80, 'ó': 81, 'ú': 82, 'ü': 83, '<BLNK>': 84}, pad='<PAD>', eos='<EOS>', bos='<BOS>', blank='<BLNK>', blank_word=None), add_blank_char=True, add_blank_word=False, use_eos_bos=False, blank_at_end=True, blank_at_start=True), blank_at_start=True, blank_at_end=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.GRAPHEMES: 'graphemes'>, alphabet=<Alphabet.UNICODE: 'unicode'>, engine=<Engine.COQUI: 'coqui'>, vocab_override={})
        # TTSModelInfo(voice_id='proxectonos/celtia', lang='gl-ES', model_url='https://huggingface.co/OpenVoiceOS/proxectonos-celtia-vits-graphemes-onnx/resolve/main/model.onnx', config_url='https://huggingface.co/OpenVoiceOS/proxectonos-celtia-vits-graphemes-onnx/resolve/main/config.json', vocab_url=None, tokenizer_config_url=None, tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=0, num_langs=1, sample_rate=16000, lang_code='gl-ES', phoneme_type='graphemes', alphabet='unicode', phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine='coqui', length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, add_diacritics=False, tokenizer=TTSTokenizer(vocabulary=Vocabulary(char2idx={'_': 0, '!': 1, '"': 2, '(': 3, ')': 4, ',': 5, '-': 6, '.': 7, ':': 8, ';': 9, '?': 10, '¡': 11, '¿': 12, ' ': 13, 'A': 14, 'B': 15, 'C': 16, 'D': 17, 'E': 18, 'F': 19, 'G': 20, 'H': 21, 'I': 22, 'J': 23, 'K': 24, 'L': 25, 'M': 26, 'N': 27, 'O': 28, 'P': 29, 'Q': 30, 'R': 31, 'S': 32, 'T': 33, 'U': 34, 'V': 35, 'X': 36, 'Y': 37, 'Z': 38, 'a': 39, 'b': 40, 'c': 41, 'd': 42, 'e': 43, 'f': 44, 'g': 45, 'h': 46, 'i': 47, 'j': 48, 'k': 49, 'l': 50, 'm': 51, 'n': 52, 'o': 53, 'p': 54, 'q': 55, 'r': 56, 's': 57, 't': 58, 'u': 59, 'v': 60, 'w': 61, 'x': 62, 'y': 63, 'z': 64, 'Á': 65, 'É': 66, 'Í': 67, 'Ó': 68, 'Ú': 69, 'á': 70, 'é': 71, 'í': 72, 'ñ': 73, 'ó': 74, 'ú': 75, 'ü': 76, '<BLNK>': 77}, pad='_', eos='*', bos='^', blank='<BLNK>', blank_word=None), add_blank_char=True, add_blank_word=False, use_eos_bos=False, blank_at_end=True, blank_at_start=True), blank_at_start=True, blank_at_end=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.GRAPHEMES: 'graphemes'>, alphabet=<Alphabet.UNICODE: 'unicode'>, engine=<Engine.COQUI: 'coqui'>, vocab_override={})
        # TTSModelInfo(voice_id='proxectonos/brais-cotovia', lang='gl-ES', model_url='https://huggingface.co/OpenVoiceOS/proxectonos-brais-vits-phonemes-onnx/resolve/main/model.onnx', config_url='https://huggingface.co/OpenVoiceOS/proxectonos-brais-vits-phonemes-onnx/resolve/main/config.json', vocab_url=None, tokenizer_config_url=None, tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=0, num_langs=1, sample_rate=16000, lang_code='gl-ES', phoneme_type='cotovia', alphabet='cotovia', phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine='coqui', length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, add_diacritics=False, tokenizer=TTSTokenizer(vocabulary=Vocabulary(char2idx={'<PAD>': 0, '!': 1, '¡': 2, "'": 3, '(': 4, ')': 5, ',': 6, '-': 7, '.': 8, ':': 9, ';': 10, '¿': 11, '?': 12, ' ': 13, '"': 14, '\n': 15, 'A': 16, 'B': 17, 'C': 18, 'D': 19, 'E': 20, 'F': 21, 'G': 22, 'H': 23, 'I': 24, 'J': 25, 'K': 26, 'L': 27, 'M': 28, 'N': 29, 'O': 30, 'P': 31, 'Q': 32, 'R': 33, 'S': 34, 'T': 35, 'U': 36, 'V': 37, 'W': 38, 'X': 39, 'Y': 40, 'Z': 41, 'Ç': 42, 'Á': 43, 'É': 44, 'Í': 45, 'Ó': 46, 'Ú': 47, 'Ü': 48, 'a': 49, 'b': 50, 'c': 51, 'd': 52, 'e': 53, 'f': 54, 'g': 55, 'h': 56, 'i': 57, 'j': 58, 'k': 59, 'l': 60, 'm': 61, 'n': 62, 'o': 63, 'p': 64, 'q': 65, 'r': 66, 's': 67, 't': 68, 'u': 69, 'v': 70, 'w': 71, 'x': 72, 'y': 73, 'z': 74, 'ñ': 75, 'á': 76, 'é': 77, 'í': 78, 'ó': 79, 'ú': 80, 'ü': 81, '<BLNK>': 82}, pad='<PAD>', eos='<EOS>', bos='<BOS>', blank='<BLNK>', blank_word=None), add_blank_char=True, add_blank_word=False, use_eos_bos=False, blank_at_end=True, blank_at_start=True), blank_at_start=True, blank_at_end=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.COTOVIA: 'cotovia'>, alphabet=<Alphabet.COTOVIA: 'cotovia'>, engine=<Engine.COQUI: 'coqui'>, vocab_override={})
        # TTSModelInfo(voice_id='proxectonos/celtia-cotovia', lang='gl-ES', model_url='https://huggingface.co/OpenVoiceOS/proxectonos-celtia-vits-phonemes-onnx/resolve/main/model.onnx', config_url='https://huggingface.co/OpenVoiceOS/proxectonos-celtia-vits-phonemes-onnx/resolve/main/config.json', vocab_url=None, tokenizer_config_url=None, tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=0, num_langs=1, sample_rate=16000, lang_code='gl-ES', phoneme_type='cotovia', alphabet='cotovia', phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine='coqui', length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, add_diacritics=False, tokenizer=TTSTokenizer(vocabulary=Vocabulary(char2idx={'<PAD>': 0, '!': 1, '¡': 2, "'": 3, '(': 4, ')': 5, ',': 6, '-': 7, '.': 8, ':': 9, ';': 10, '¿': 11, '?': 12, ' ': 13, '"': 14, '\n': 15, 'A': 16, 'B': 17, 'C': 18, 'D': 19, 'E': 20, 'F': 21, 'G': 22, 'H': 23, 'I': 24, 'J': 25, 'K': 26, 'L': 27, 'M': 28, 'N': 29, 'O': 30, 'P': 31, 'Q': 32, 'R': 33, 'S': 34, 'T': 35, 'U': 36, 'V': 37, 'W': 38, 'X': 39, 'Y': 40, 'Z': 41, 'Ç': 42, 'Á': 43, 'É': 44, 'Í': 45, 'Ó': 46, 'Ú': 47, 'Ü': 48, 'a': 49, 'b': 50, 'c': 51, 'd': 52, 'e': 53, 'f': 54, 'g': 55, 'h': 56, 'i': 57, 'j': 58, 'k': 59, 'l': 60, 'm': 61, 'n': 62, 'o': 63, 'p': 64, 'q': 65, 'r': 66, 's': 67, 't': 68, 'u': 69, 'v': 70, 'w': 71, 'x': 72, 'y': 73, 'z': 74, 'ñ': 75, 'á': 76, 'é': 77, 'í': 78, 'ó': 79, 'ú': 80, 'ü': 81, '<BLNK>': 82}, pad='<PAD>', eos='<EOS>', bos='<BOS>', blank='<BLNK>', blank_word=None), add_blank_char=True, add_blank_word=False, use_eos_bos=False, blank_at_end=True, blank_at_start=True), blank_at_start=True, blank_at_end=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.COTOVIA: 'cotovia'>, alphabet=<Alphabet.COTOVIA: 'cotovia'>, engine=<Engine.COQUI: 'coqui'>, vocab_override={})
        # TTSModelInfo(voice_id='proxectonos/sabela-cotovia', lang='gl-ES', model_url='https://huggingface.co/OpenVoiceOS/proxectonos-sabela-vits-phonemes-onnx/resolve/main/model.onnx', config_url='https://huggingface.co/OpenVoiceOS/proxectonos-sabela-vits-phonemes-onnx/resolve/main/config.json', vocab_url=None, tokenizer_config_url=None, tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=0, num_langs=1, sample_rate=16000, lang_code='gl-ES', phoneme_type='cotovia', alphabet='cotovia', phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine='coqui', length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, add_diacritics=False, tokenizer=TTSTokenizer(vocabulary=Vocabulary(char2idx={'<PAD>': 0, '!': 1, '¡': 2, "'": 3, '(': 4, ')': 5, ',': 6, '-': 7, '.': 8, ':': 9, ';': 10, '¿': 11, '?': 12, ' ': 13, '"': 14, '\n': 15, 'A': 16, 'B': 17, 'C': 18, 'D': 19, 'E': 20, 'F': 21, 'G': 22, 'H': 23, 'I': 24, 'J': 25, 'K': 26, 'L': 27, 'M': 28, 'N': 29, 'O': 30, 'P': 31, 'Q': 32, 'R': 33, 'S': 34, 'T': 35, 'U': 36, 'V': 37, 'W': 38, 'X': 39, 'Y': 40, 'Z': 41, 'Ç': 42, 'Á': 43, 'É': 44, 'Í': 45, 'Ó': 46, 'Ú': 47, 'Ü': 48, 'a': 49, 'b': 50, 'c': 51, 'd': 52, 'e': 53, 'f': 54, 'g': 55, 'h': 56, 'i': 57, 'j': 58, 'k': 59, 'l': 60, 'm': 61, 'n': 62, 'o': 63, 'p': 64, 'q': 65, 'r': 66, 's': 67, 't': 68, 'u': 69, 'v': 70, 'w': 71, 'x': 72, 'y': 73, 'z': 74, 'ñ': 75, 'á': 76, 'é': 77, 'í': 78, 'ó': 79, 'ú': 80, 'ü': 81, '<BLNK>': 82}, pad='<PAD>', eos='<EOS>', bos='<BOS>', blank='<BLNK>', blank_word=None), add_blank_char=True, add_blank_word=False, use_eos_bos=False, blank_at_end=True, blank_at_start=True), blank_at_start=True, blank_at_end=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.COTOVIA: 'cotovia'>, alphabet=<Alphabet.COTOVIA: 'cotovia'>, engine=<Engine.COQUI: 'coqui'>, vocab_override={})
        # TTSModelInfo(voice_id='proxectonos/icia-cotovia', lang='gl-ES', model_url='https://huggingface.co/OpenVoiceOS/proxectonos-icia-vits-phonemes-onnx/resolve/main/model.onnx', config_url='https://huggingface.co/OpenVoiceOS/proxectonos-icia-vits-phonemes-onnx/resolve/main/config.json', vocab_url=None, tokenizer_config_url=None, tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=0, num_langs=1, sample_rate=16000, lang_code='gl-ES', phoneme_type='cotovia', alphabet='cotovia', phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine='coqui', length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, add_diacritics=False, tokenizer=TTSTokenizer(vocabulary=Vocabulary(char2idx={'<PAD>': 0, '<EOS>': 1, '<BOS>': 2, '<BLNK>': 3, '\n': 4, '"': 5, 'A': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10, 'F': 11, 'G': 12, 'H': 13, 'I': 14, 'J': 15, 'K': 16, 'L': 17, 'M': 18, 'N': 19, 'O': 20, 'P': 21, 'Q': 22, 'R': 23, 'S': 24, 'T': 25, 'U': 26, 'V': 27, 'W': 28, 'X': 29, 'Y': 30, 'Z': 31, 'a': 32, 'b': 33, 'c': 34, 'd': 35, 'e': 36, 'f': 37, 'g': 38, 'h': 39, 'i': 40, 'j': 41, 'k': 42, 'l': 43, 'm': 44, 'n': 45, 'o': 46, 'p': 47, 'q': 48, 'r': 49, 's': 50, 't': 51, 'u': 52, 'v': 53, 'w': 54, 'x': 55, 'y': 56, 'z': 57, 'Á': 58, 'Ç': 59, 'É': 60, 'Í': 61, 'Ó': 62, 'Ú': 63, 'Ü': 64, 'á': 65, 'é': 66, 'í': 67, 'ñ': 68, 'ó': 69, 'ú': 70, 'ü': 71, '!': 72, '¡': 73, "'": 74, '(': 75, ')': 76, ',': 77, '-': 78, '.': 79, ':': 80, ';': 81, '¿': 82, '?': 83, ' ': 84}, pad='<PAD>', eos='<EOS>', bos='<BOS>', blank='<BLNK>', blank_word=None), add_blank_char=True, add_blank_word=False, use_eos_bos=False, blank_at_end=True, blank_at_start=True), blank_at_start=True, blank_at_end=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.COTOVIA: 'cotovia'>, alphabet=<Alphabet.COTOVIA: 'cotovia'>, engine=<Engine.COQUI: 'coqui'>, vocab_override={})
        # TTSModelInfo(voice_id='proxectonos/paulo-cotovia', lang='gl-ES', model_url='https://huggingface.co/OpenVoiceOS/proxectonos-paulo-vits-phonemes-onnx/resolve/main/model.onnx', config_url='https://huggingface.co/OpenVoiceOS/proxectonos-paulo-vits-phonemes-onnx/resolve/main/config.json', vocab_url=None, tokenizer_config_url=None, tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=0, num_langs=1, sample_rate=16000, lang_code='gl-ES', phoneme_type='cotovia', alphabet='cotovia', phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine='coqui', length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, add_diacritics=False, tokenizer=TTSTokenizer(vocabulary=Vocabulary(char2idx={'<PAD>': 0, '<EOS>': 1, '<BOS>': 2, '<BLNK>': 3, 'A': 4, 'B': 5, 'C': 6, 'D': 7, 'E': 8, 'F': 9, 'G': 10, 'H': 11, 'I': 12, 'J': 13, 'K': 14, 'L': 15, 'M': 16, 'N': 17, 'O': 18, 'P': 19, 'Q': 20, 'R': 21, 'S': 22, 'T': 23, 'U': 24, 'V': 25, 'W': 26, 'X': 27, 'Y': 28, 'Z': 29, 'a': 30, 'b': 31, 'c': 32, 'd': 33, 'e': 34, 'f': 35, 'g': 36, 'h': 37, 'i': 38, 'j': 39, 'k': 40, 'l': 41, 'm': 42, 'n': 43, 'o': 44, 'p': 45, 'q': 46, 'r': 47, 's': 48, 't': 49, 'u': 50, 'v': 51, 'w': 52, 'x': 53, 'y': 54, 'z': 55, 'Á': 56, 'Ç': 57, 'É': 58, 'Í': 59, 'Ó': 60, 'Ú': 61, 'Ü': 62, 'á': 63, 'é': 64, 'í': 65, 'ñ': 66, 'ó': 67, 'ú': 68, 'ü': 69, '!': 70, '¡': 71, "'": 72, '(': 73, ')': 74, ',': 75, '-': 76, '.': 77, ':': 78, ';': 79, '¿': 80, '?': 81, ' ': 82, '"': 83, '\n': 84}, pad='<PAD>', eos='<EOS>', bos='<BOS>', blank='<BLNK>', blank_word=None), add_blank_char=True, add_blank_word=False, use_eos_bos=False, blank_at_end=True, blank_at_start=True), blank_at_start=True, blank_at_end=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.COTOVIA: 'cotovia'>, alphabet=<Alphabet.COTOVIA: 'cotovia'>, engine=<Engine.COQUI: 'coqui'>, vocab_override={})
        # TTSModelInfo(voice_id='proxectonos/iago-cotovia', lang='gl-ES', model_url='https://huggingface.co/OpenVoiceOS/proxectonos-iago-vits-phonemes-onnx/resolve/main/model.onnx', config_url='https://huggingface.co/OpenVoiceOS/proxectonos-iago-vits-phonemes-onnx/resolve/main/config.json', vocab_url=None, tokenizer_config_url=None, tokens_url=None, phoneme_map_url=None, config=VoiceConfig(num_symbols=256, num_speakers=0, num_langs=1, sample_rate=16000, lang_code='gl-ES', phoneme_type='cotovia', alphabet='cotovia', phonemizer_model=None, speaker_id_map={}, lang_id_map={}, engine='coqui', length_scale=1.0, noise_scale=0.667, noise_w_scale=0.8, add_diacritics=False, tokenizer=TTSTokenizer(vocabulary=Vocabulary(char2idx={'<PAD>': 0, '<EOS>': 1, '<BOS>': 2, '<BLNK>': 3, 'A': 4, 'B': 5, 'C': 6, 'D': 7, 'E': 8, 'F': 9, 'G': 10, 'H': 11, 'I': 12, 'J': 13, 'K': 14, 'L': 15, 'M': 16, 'N': 17, 'O': 18, 'P': 19, 'Q': 20, 'R': 21, 'S': 22, 'T': 23, 'U': 24, 'V': 25, 'W': 26, 'X': 27, 'Y': 28, 'Z': 29, 'a': 30, 'b': 31, 'c': 32, 'd': 33, 'e': 34, 'f': 35, 'g': 36, 'h': 37, 'i': 38, 'j': 39, 'k': 40, 'l': 41, 'm': 42, 'n': 43, 'o': 44, 'p': 45, 'q': 46, 'r': 47, 's': 48, 't': 49, 'u': 50, 'v': 51, 'w': 52, 'x': 53, 'y': 54, 'z': 55, 'Á': 56, 'Ç': 57, 'É': 58, 'Í': 59, 'Ó': 60, 'Ú': 61, 'Ü': 62, 'á': 63, 'é': 64, 'í': 65, 'ñ': 66, 'ó': 67, 'ú': 68, 'ü': 69, '!': 70, '¡': 71, "'": 72, '(': 73, ')': 74, ',': 75, '-': 76, '.': 77, ':': 78, ';': 79, '¿': 80, '?': 81, ' ': 82, '"': 83, '\n': 84}, pad='<PAD>', eos='<EOS>', bos='<BOS>', blank='<BLNK>', blank_word=None), add_blank_char=True, add_blank_word=False, use_eos_bos=False, blank_at_end=True, blank_at_start=True), blank_at_start=True, blank_at_end=True, pad_token=None, blank_token=None, bos_token=None, eos_token=None, word_sep_token=' ', blank_between=<BlankBetween.TOKENS_AND_WORDS: 'tokens_and_words'>), phoneme_type=<PhonemeType.COTOVIA: 'cotovia'>, alphabet=<Alphabet.COTOVIA: 'cotovia'>, engine=<Engine.COQUI: 'coqui'>, vocab_override={})

    print(manager.supported_langs)
    # ['abi', 'abp', 'aca', 'acd', 'ace', 'acf', 'ach', 'acn', 'acr', 'acu', 'ade', 'adh', 'adj', 'adx', 'aeu',
    # 'af-ZA', 'agd', 'agg', 'agn', 'agr', 'agu', 'agx', 'aha', 'ahk', 'aia', 'aka', 'akb', 'ake', 'akp', 'alj',
    # 'alp', 'alt', 'alz', 'ame', 'amf', 'amh', 'ami', 'amk', 'ann', 'any', 'aoz', 'apb', 'apr', 'ar', 'ar-JO',
    # 'ar-SA', 'ara', 'arl', 'asa', 'asg', 'asm', 'ata', 'atb', 'atg', 'ati', 'atq', 'ava', 'avn', 'avu', 'awa', 'awb',
    # 'ayo', 'ayr', 'ayz', 'azb', 'azg', 'azj-script_cyrillic', 'azj-script_latin', 'azz', 'bak', 'bam', 'ban', 'bao',
    # 'bav', 'bba', 'bbb', 'bbc', 'bbo', 'bcc-script_arabic', 'bcc-script_latin', 'bcl', 'bcw', 'bdg', 'bdh', 'bdq',
    # 'bdu', 'bdv', 'beh', 'bem', 'ben', 'bep', 'bex', 'bfa', 'bfo', 'bfy', 'bfz', 'bg-BG', 'bgc', 'bgq', 'bgr', 'bgt',
    # 'bgw', 'bha', 'bht', 'bhz', 'bib', 'bim', 'bis', 'biv', 'bjr', 'bjv', 'bjw', 'bjz', 'bkd', 'bkv', 'blh', 'blt',
    # 'blx', 'blz', 'bmq', 'bmr', 'bmu', 'bmv', 'bn', 'bng', 'bno', 'bnp', 'boa', 'bod', 'boj', 'bom', 'bor', 'bov',
    # 'box', 'bpr', 'bps', 'bqc', 'bqi', 'bqj', 'bqp', 'bru', 'bsc', 'bsq', 'bss', 'btd', 'bts', 'btt', 'btx', 'bud',
    # 'bul', 'bus', 'bvc', 'bvz', 'bwq', 'bwu', 'byr', 'bzh', 'bzi', 'bzj', 'ca-ES', 'caa', 'cab',
    # 'cak-dialect_central', 'cak-dialect_santodomingoxenacoj', 'cak-dialect_southcentral', 'cak-dialect_western',
    # 'cak-dialect_yepocapa', 'cap', 'car', 'cas', 'cat', 'cax', 'cbc', 'cbi', 'cbr', 'cbs', 'cbt', 'cbu', 'cbv', 'cce',
    # 'cco', 'cdj', 'ceb', 'ceg', 'cek', 'cfm', 'cgc', 'che', 'chf', 'chv', 'chz', 'cjo', 'cjp', 'cjs', 'cko', 'ckt',
    # 'cla', 'cle', 'cly', 'cme', 'cmo-script_khmer', 'cmo-script_latin', 'cmr', 'cnh', 'cni', 'cnl', 'cnt', 'coe',
    # 'cof', 'cok', 'con', 'cot', 'cou', 'cpa', 'cpb', 'cpu', 'crh', 'crk-script_latin', 'crk-script_syllabics', 'crn',
    # 'crq', 'crs', 'crt', 'cs-CZ', 'csk', 'cso', 'ctd', 'ctg', 'cto', 'ctu', 'cuc', 'cui', 'cuk', 'cul', 'cwa', 'cwe',
    # 'cwt', 'cy-GB', 'cya', 'cym', 'da-DK', 'daa', 'dah', 'dar', 'dbj', 'dbq', 'ddn', 'de-DE', 'ded', 'des', 'deu',
    # 'dga', 'dgi', 'dgk', 'dgo', 'dgr', 'dhi', 'did', 'dig', 'dik', 'dip', 'div', 'djk', 'dnj-dialect_blowowest',
    # 'dnj-dialect_gweetaawueast', 'dnt', 'dnw', 'dop', 'dos', 'dsh', 'dso', 'dtp', 'dts', 'dug', 'dwr', 'dyi', 'dyo',
    # 'dyu', 'dzo', 'eip', 'eka', 'el-GR', 'ell', 'emp', 'en', 'en-GB', 'en-IE', 'en-US', 'en-cy', 'enb', 'eng', 'enx',
    # 'es-AR', 'es-CL', 'es-CO', 'es-ES', 'es-MX', 'ese', 'ess', 'eu-ES', 'eus', 'evn', 'ewe', 'eza', 'fa', 'fa-IR',
    # 'fal', 'fao', 'far', 'fas', 'fi-FI', 'fij', 'fin', 'flr', 'fmu', 'fon', 'fr-FR', 'fra', 'frd', 'ful',
    # 'gag-script_cyrillic', 'gag-script_latin', 'gai', 'gam', 'gau', 'gbi', 'gbk', 'gbm', 'gbo', 'gde', 'geb', 'gej',
    # 'gil', 'gjn', 'gkn', 'gl-ES', 'gld', 'glk', 'gmv', 'gna', 'gnd', 'gng', 'gof-script_latin', 'gog', 'gor', 'gqr',
    # 'grc', 'gri', 'grn', 'grt', 'gso', 'gu-IN', 'gub', 'guc', 'gud', 'guh', 'guj', 'guk', 'gum', 'guo', 'guq', 'guu',
    # 'gux', 'gvc', 'gvl', 'gwi', 'gwr', 'gym', 'gyr', 'ha-NE', 'had', 'hag', 'hak', 'hap', 'hat', 'hau', 'hay', 'he',
    # 'heb', 'heh', 'hi-IN', 'hif', 'hig', 'hil', 'hin', 'hlb', 'hlt', 'hne', 'hnn', 'hns', 'hoc', 'hoy', 'hto', 'hu-HU',
    # 'hub', 'hui', 'hun', 'hus-dialect_centralveracruz', 'hus-dialect_westernpotosino', 'huu', 'huv', 'hvn', 'hwc',
    # 'hy-AM', 'hyw', 'iba', 'icr', 'id-ID', 'idd', 'ifa', 'ifb', 'ife', 'ifk', 'ifu', 'ify', 'ign', 'ikk', 'ilb', 'ilo',
    # 'imo', 'inb', 'ind', 'iou', 'ipi', 'iqw', 'iri', 'irk', 'is-IS', 'isl', 'it-IT', 'itl', 'itv',
    # 'ixl-dialect_sangasparchajul', 'ixl-dialect_sanjuancotzal', 'ixl-dialect_santamarianebaj', 'izr', 'izz', 'jac',
    # 'jam', 'jav', 'jbu', 'jen', 'jic', 'jiv', 'jmc', 'jmd', 'jun', 'juy', 'jv-ID', 'jvn', 'ka-GE', 'kaa', 'kab',
    # 'kac', 'kak', 'kan', 'kao', 'kaq', 'kay', 'kaz', 'kbo', 'kbp', 'kbq', 'kbr', 'kby', 'kca', 'kcg', 'kdc', 'kde',
    # 'kdh', 'kdi', 'kdj', 'kdl', 'kdn', 'kdt', 'kek', 'ken', 'keo', 'ker', 'key', 'kez', 'kfb', 'kff-script_telugu',
    # 'kfw', 'kfx', 'khg', 'khm', 'khq', 'kia', 'kij', 'kik', 'kin', 'kir', 'kjb', 'kje', 'kjg', 'kjh', 'kk-KZ', 'kki',
    # 'kkj', 'kle', 'klu', 'klv', 'klw', 'kma', 'kmd', 'kml', 'kmr-script_arabic', 'kmr-script_cyrillic',
    # 'kmr-script_latin', 'kmu', 'knb', 'kne', 'knf', 'knj', 'knk', 'kno', 'ko-KO', 'kog', 'kor', 'kpq', 'kps', 'kpv',
    # 'kpy', 'kpz', 'kqe', 'kqp', 'kqr', 'kqy', 'krc', 'kri', 'krj', 'krl', 'krr', 'krs', 'kru', 'ksb', 'ksr', 'kss',
    # 'ktb', 'ktj', 'kub', 'kue', 'kum', 'kus', 'kvn', 'kvw', 'kwd', 'kwf', 'kwi', 'kxc', 'kxf', 'kxm', 'kxv', 'kyb',
    # 'kyc', 'kyf', 'kyg', 'kyo', 'kyq', 'kyu', 'kyz', 'kzf', 'lac', 'laj', 'lam', 'lao', 'las', 'lat', 'lav', 'law',
    # 'lb-LU', 'lbj', 'lbw', 'lcp', 'lee', 'lef', 'lem', 'lew', 'lex', 'lgg', 'lgl', 'lhu', 'lia', 'lid', 'lif', 'lip',
    # 'lis', 'lje', 'ljp', 'llg', 'lln', 'lme', 'lnd', 'lns', 'lob', 'lok', 'lom', 'lon', 'loq', 'lsi', 'lsm', 'luc',
    # 'lug', 'lv-LV', 'lwo', 'lww', 'lzz', 'maa-dialect_sanantonio', 'mad', 'mag', 'mah', 'mai', 'maj', 'mak', 'mal',
    # 'mam-dialect_central', 'mam-dialect_northern', 'mam-dialect_southern', 'mam-dialect_western', 'maq', 'mar', 'maw',
    # 'maz', 'mbb', 'mbc', 'mbh', 'mbj', 'mbt', 'mbu', 'mbz', 'mca', 'mcb', 'mcd', 'mco', 'mcp', 'mcq', 'mcu', 'mda',
    # 'mdv', 'mdy', 'med', 'mee', 'mej', 'men', 'meq', 'met', 'mev', 'mfe', 'mfh', 'mfi', 'mfk', 'mfq', 'mfy', 'mfz',
    # 'mgd', 'mge', 'mgh', 'mgo', 'mhi', 'mhr', 'mhu', 'mhx', 'mhy', 'mib', 'mie', 'mif', 'mih', 'mil', 'mim', 'min',
    # 'mio', 'mip', 'miq', 'mit', 'miy', 'miz', 'mjl', 'mjv', 'mkl', 'mkn', 'ml-IN', 'mlg', 'mmg', 'mnb', 'mnf', 'mnk',
    # 'mnw', 'mnx', 'moa', 'mog', 'mon', 'mop', 'mor', 'mos', 'mox', 'moz', 'mpg', 'mpm', 'mpp', 'mpx', 'mqb', 'mqf',
    # 'mqj', 'mqn', 'mrw', 'msy', 'mtd', 'mtj', 'mto', 'muh', 'mup', 'mur', 'muv', 'muy', 'mvp', 'mwq', 'mwv', 'mxb',
    # 'mxq', 'mxt', 'mxv', 'mya', 'myb', 'myk', 'myl', 'myv', 'myx', 'myy', 'mza', 'mzi', 'mzj', 'mzk', 'mzm', 'mzw',
    # 'nab', 'nag', 'nan', 'nas', 'naw', 'nca', 'nch', 'ncj', 'ncl', 'ncu', 'ndj', 'ndp', 'ndv', 'ndy', 'ndz', 'ne-NP',
    # 'neb', 'new', 'nfa', 'nfr', 'nga', 'ngl', 'ngp', 'ngu', 'nhe', 'nhi', 'nhu', 'nhw', 'nhx', 'nhy', 'nia', 'nij',
    # 'nim', 'nin', 'nko', 'nl', 'nl-BE', 'nl-NL', 'nlc', 'nld', 'nlg', 'nlk', 'nmz', 'nnb', 'nnq', 'nnw', 'no-NO',
    # 'noa', 'nod', 'nog', 'not', 'npl', 'npy', 'nst', 'nsu', 'ntm', 'ntr', 'nuj', 'nus', 'nuz', 'nwb', 'nxq', 'nya',
    # 'nyf', 'nyn', 'nyo', 'nyy', 'nzi', 'obo', 'ojb-script_latin', 'ojb-script_syllabics', 'oku', 'old', 'omw', 'onb',
    # 'ood', 'orm', 'ory', 'oss', 'ote', 'otq', 'ozm', 'pab', 'pad', 'pag', 'pam', 'pan', 'pao', 'pap', 'pau', 'pbb',
    # 'pbc', 'pbi', 'pce', 'pcm', 'peg', 'pez', 'pib', 'pil', 'pir', 'pis', 'pjt', 'pkb', 'pl-PL', 'pls', 'plw', 'pmf',
    # 'pny', 'poh-dialect_eastern', 'poh-dialect_western', 'poi', 'pol', 'por', 'poy', 'ppk', 'pps', 'prf', 'prk', 'prt',
    # 'pse', 'pss', 'pt-BR', 'pt-PT', 'ptu', 'pui', 'pwg', 'pww', 'pxm', 'qub', 'quc-dialect_central', 'quc-dialect_east',
    # 'quc-dialect_north', 'quf', 'quh', 'qul', 'quw', 'quy', 'quz', 'qvc', 'qve', 'qvh', 'qvm', 'qvn', 'qvo', 'qvs',
    # 'qvw', 'qvz', 'qwh', 'qxh', 'qxl', 'qxn', 'qxo', 'qxr', 'rah', 'rai', 'rap', 'rav', 'raw', 'rej', 'rel', 'rgu',
    # 'rhg', 'rif-script_arabic', 'rif-script_latin', 'ril', 'rim', 'rjs', 'rkt', 'rmc-script_cyrillic',
    # 'rmc-script_latin', 'rmo', 'rmy-script_cyrillic', 'rmy-script_latin', 'rng', 'rnl', 'ro-RO', 'rol', 'ron', 'rop',
    # 'rro', 'ru-RU', 'rub', 'ruf', 'rug', 'run', 'rus', 'sab', 'sag', 'sah', 'saj', 'saq', 'sas', 'sba', 'sbd', 'sbl',
    # 'sbp', 'sch', 'sck', 'sda', 'sea', 'seh', 'ses', 'sey', 'sgb', 'sgj', 'sgw', 'shi', 'shk', 'shn', 'sho', 'shp',
    # 'sid', 'sig', 'sil', 'sja', 'sjm', 'sk-SK', 'sl-SI', 'sld', 'slu', 'sml', 'smo', 'sna', 'sne', 'snn', 'snp',
    # 'snw', 'som', 'soy', 'spa', 'spp', 'spy', 'sqi', 'sr-RS', 'sri', 'srm', 'srn', 'srx', 'stn', 'stp', 'suc', 'suk',
    # 'sun', 'sur', 'sus', 'suv', 'suz', 'sv-SE', 'sw', 'sw-CD', 'swe', 'swh', 'sxb', 'sxn', 'sya', 'syl', 'sza', 'tac',
    # 'taj', 'tam', 'tao', 'tap', 'taq', 'tat', 'tav', 'tbc', 'tbg', 'tbk', 'tbl', 'tby', 'tbz', 'tca', 'tcc', 'tcs',
    # 'tcz', 'tdj', 'tdt-TL', 'te-IN', 'ted', 'tee', 'tel', 'tem', 'teo', 'ter', 'tes', 'tew', 'tex', 'tfr', 'tgj',
    # 'tgk', 'tgl', 'tgo', 'tgp', 'tha', 'thk', 'thl', 'tih', 'tik', 'tir', 'tkr', 'tlb', 'tlj', 'tly', 'tmc', 'tmf',
    # 'tn-ZA', 'tna', 'tng', 'tnk', 'tnn', 'tnp', 'tnr', 'tnt', 'tob', 'toc', 'toh', 'tom', 'tos', 'tpi', 'tpm', 'tpp',
    # 'tpt', 'tr-TR', 'trc', 'tri', 'trn', 'trs', 'tso', 'tsz', 'ttc', 'tte', 'ttq-script_tifinagh', 'tue', 'tuf',
    # 'tuk-script_arabic', 'tuk-script_latin', 'tuo', 'tur', 'tvw', 'twb', 'twe', 'twu', 'txa', 'txq', 'txu', 'tye',
    # 'tzh-dialect_tenejapa', 'tzj-dialect_eastern', 'tzj-dialect_western', 'tzo-dialect_chamula', 'udm', 'udu',
    # 'uig-script_arabic', 'uig-script_cyrillic', 'uk-GB', 'uk-UA', 'ukr', 'unr', 'upv', 'ura', 'urb',
    # 'urd-script_arabic', 'urd-script_devanagari', 'urd-script_latin', 'urk', 'urt', 'ury', 'usp',
    # 'uzb-script_cyrillic', 'vag', 'vi-VN', 'vid', 'vie', 'vif', 'vmw', 'vmy', 'vun', 'vut', 'wal-script_ethiopic',
    # 'wal-script_latin', 'wap', 'war', 'waw', 'way', 'wba', 'wlo', 'wlx', 'wmw', 'wob', 'wsg', 'wwa', 'xal', 'xdy',
    # 'xed', 'xer', 'xmm', 'xnj', 'xnr', 'xog', 'xon', 'xrb', 'xsb', 'xsm', 'xsr', 'xsu', 'xta', 'xtd', 'xte', 'xtm',
    # 'xtn', 'xua', 'xuo', 'yaa', 'yad', 'yal', 'yam', 'yao', 'yas', 'yat', 'yaz', 'yba', 'ybb', 'ycl', 'ycn', 'yea',
    # 'yka', 'yli', 'yo', 'yor', 'yre', 'yua', 'yuz', 'yva', 'zaa', 'zab', 'zac', 'zad', 'zae', 'zai', 'zam', 'zao',
    # 'zaq', 'zar', 'zas', 'zav', 'zaw', 'zca', 'zga', 'zh-CN', 'zh-TW', 'zim', 'ziw', 'zlm', 'zmz', 'zne', 'zos',
    # 'zpc', 'zpg', 'zpi', 'zpl', 'zpm', 'zpo', 'zpt', 'zpu', 'zpz', 'ztq', 'zty', 'zyb', 'zyp', 'zza']


    def generate_voices_markdown(manager: TTSModelManager, output_file: str = "../VOICES.md"):
        """
        Generates a Markdown table of all supported voices and saves it to a file.

        Args:
            manager (TTSModelManager): The manager with loaded voices.
            output_file (str): The name of the file to save the markdown table to.
        """

        # Sort voices by language code (lang) then by voice ID
        sorted_voices = sorted(
            manager.all_voices,
            key=lambda v: (v.lang.lower(), v.voice_id.lower())
        )

        markdown_output = [
            "## Supported Voices",
            "",
            f"**Total Voices:** {len(sorted_voices)}",
            f"**Total Languages:** {len(manager.supported_langs)}",
            "",
            "> ⚠️ some languages are duplicated, either using a different script or less specific language code (eg. Kurdish is available in latin, cyrillic and arabic scripts)",
            "",
            "> ⚠️ some models are duplicated, piper is known to mirror community voices and those will often show up twice (under piper and under original author)",
            "",
            "| Voice ID | Language Code | Engine | Phoneme Type |",
            "| :--- | :--- | :--- | :--- |"
        ]

        for voice in sorted_voices:
            # Get the value from the Enum object, handling None/string if post_init didn't run
            engine = voice.engine.value if hasattr(voice.engine, 'value') else str(voice.engine)
            phoneme_type = voice.phoneme_type.value if hasattr(voice.phoneme_type, 'value') else str(voice.phoneme_type)

            # Format the row for the markdown table
            row = (
                f"| `{voice.voice_id}` | `{voice.lang}` | "
                f"`{engine}` | `{phoneme_type}` |"
            )
            markdown_output.append(row)

        # Write the output to the specified file
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(markdown_output))
            print(f"\n✅ Successfully generated and saved voice table to: **{output_file}**")
        except IOError as e:
            print(f"\n❌ Error writing to file {output_file}: {e}")
            print("\n--- Start of Markdown Output ---")
            print("\n".join(markdown_output))
            print("--- End of Markdown Output ---")


    generate_voices_markdown(manager, output_file="../VOICES.md")

