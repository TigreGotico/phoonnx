"""multilingual phonemizers"""

import json
import os
import subprocess
from typing import List, Dict, Optional

import numpy as np
import onnxruntime
import requests

from phoonnx.config import Alphabet
from phoonnx.phonemizers.base import BasePhonemizer


class EspeakError(Exception):
    """Custom exception for espeak-ng related errors."""
    pass


class ByT5Phonemizer(BasePhonemizer):
    """
    A phonemizer class that uses a ByT5 ONNX model to convert text into phonemes.
    """
    MODEL2URL = {
        "OpenVoiceOS/g2p-mbyt5-12l-ipa-childes-espeak-onnx": "https://huggingface.co/OpenVoiceOS/g2p-mbyt5-12l-ipa-childes-espeak-onnx/resolve/main/fdemelo_g2p-mbyt5-12l-ipa-childes-espeak.onnx",
    #    "OpenVoiceOS/g2p-multilingual-byt5-tiny-8l-ipa-childes-onnx": "https://huggingface.co/OpenVoiceOS/g2p-multilingual-byt5-tiny-8l-ipa-childes-onnx/resolve/main/byt5_g2p_model.onnx"
    }
    TOKENIZER_CONFIG_URL = "https://huggingface.co/OpenVoiceOS/g2p-multilingual-byt5-tiny-8l-ipa-childes-onnx/resolve/main/tokenizer_config.json"

    BYT5_LANGS =['ca-ES', 'cy-GB', 'da-DK', 'de-DE', 'en-GB', 'en-US', 'es-ES', 'et-EE', 'eu-ES', 'fa-IR', 'fr-FR',
                 'ga-IE', 'hr-HR', 'hu-HU', 'id-ID', 'is-IS', 'it-IT', 'ja-JP', 'ko-KR', 'nb-NO', 'nl-NL', 'pl-PL',
                 'pt-BR', 'pt-PT', 'qu-PE', 'ro-RO', 'sr-RS', 'sv-SE', 'tr-TR', 'yue-CN', 'zh-CN']

    _LEGACY_MODELS = ["g2p-multilingual-byt5-tiny-8l-ipa-childes-onnx"]
    _LEGACY_LANGS = ['ca', 'cy', 'da', 'de', 'en-na', 'en-uk', 'es', 'et', 'eu', 'fa', 'fr', 'ga', 'hr', 'hu', 'id', 'is',
                   'it', 'ja', 'ko', 'nl', 'no', 'pl', 'pt', 'pt-br', 'qu', 'ro', 'sr', 'sv', 'tr', 'zh', 'zh-yue']

    def __init__(self, model: Optional[str] = None, tokenizer_config: Optional[str] = None,
                 use_cuda=bool(os.environ.get("CUDA", False))):
        """
        Initializes the ByT5Phonemizer with the ONNX model and tokenizer configuration.
        If paths are not provided, it attempts to download them to a local directory.

        Args:
            model (str, optional): Path to the ONNX model file. If None, it will be downloaded.
            tokenizer_config (str, optional): Path to the tokenizer configuration JSON file. If None, it will be downloaded.
        """
        super().__init__(Alphabet.IPA)
        model = model or "OpenVoiceOS/g2p-mbyt5-12l-ipa-childes-espeak-onnx"
        # Define the local data path for models and configs
        data_path = os.path.expanduser("~/.local/share/phoonnx")
        os.makedirs(data_path, exist_ok=True)  # Ensure the directory exists

        # Determine the actual paths for the model and tokenizer config
        if model in self.MODEL2URL:
            base = os.path.join(data_path, model)
            os.makedirs(base, exist_ok=True)
            self.onnx_model_path = os.path.join(base, self.MODEL2URL[model].split("/")[-1])
        else:
            self.onnx_model_path = model

        if tokenizer_config is None:
            self.tokenizer_config = os.path.join(data_path, "tokenizer_config.json")
        else:
            self.tokenizer_config = tokenizer_config

        # Download model if it doesn't exist
        if not os.path.exists(self.onnx_model_path):
            if model not in self.MODEL2URL:
                raise ValueError("unknown model")
            print(f"Downloading ONNX model from {self.MODEL2URL[model]} to {self.onnx_model_path}...")
            try:
                response = requests.get(self.MODEL2URL[model], stream=True)
                response.raise_for_status()  # Raise an exception for HTTP errors
                with open(self.onnx_model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("ONNX model downloaded successfully.")
            except requests.exceptions.RequestException as e:
                raise IOError(f"Failed to download ONNX model: {e}")

        # Download tokenizer config if it doesn't exist
        if not os.path.exists(self.tokenizer_config):
            print(f"Downloading tokenizer config from {self.TOKENIZER_CONFIG_URL} to {self.tokenizer_config}...")
            try:
                response = requests.get(self.TOKENIZER_CONFIG_URL, stream=True)
                response.raise_for_status()  # Raise an exception for HTTP errors
                with open(self.tokenizer_config, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Tokenizer config downloaded successfully.")
            except requests.exceptions.RequestException as e:
                raise IOError(f"Failed to download tokenizer config: {e}")

        if use_cuda:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {"cudnn_conv_algo_search": "HEURISTIC"},
                )
            ]
            #LOG.debug("Using CUDA")
        else:
            providers = ["CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(self.onnx_model_path, providers=providers)
        with open(self.tokenizer_config, "r") as f:
            self.tokens: Dict[str, int] = json.load(f).get("added_tokens_decoder", {})

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        """
        Validates and returns the closest supported language code.

        Args:
            target_lang (str): The language code to validate.

        Returns:
            str: The validated language code.

        Raises:
            ValueError: If the language code is unsupported.
        """
        # Find the closest match
        return cls.match_lang(target_lang, cls.BYT5_LANGS)

    def _decode_phones(self, preds: List[int]) -> str:
        """
        Decodes predicted token IDs back into phonemes.

        Args:
            preds (list): A list of predicted token IDs from the ONNX model.

        Returns:
            str: The decoded phoneme string.
        """
        # Convert token IDs back to bytes, excluding special/added tokens
        phone_bytes = [
            bytes([token - 3]) for token in preds
            if str(token) not in self.tokens
        ]
        # Join bytes and decode to UTF-8, ignoring errors
        phones = b''.join(phone_bytes).decode("utf-8", errors="ignore")
        return phones

    @staticmethod
    def _encode_text(text: str, lang: str) -> np.ndarray:
        """
        Encodes input text and language into a numpy array suitable for the model.
        This function replaces the Hugging Face tokenizer for input preparation.

        Args:
            text (str): The input text to encode.
            lang (str): The language code for the text.

        Returns:
            numpy.ndarray: A numpy array of encoded input IDs.
        """
        lang = ByT5Phonemizer.get_lang(lang)  # match lang code
        # Prepend language tag and encode the string to bytes
        encoded_bytes = f"<{lang}>: {text}".encode("utf-8")
        # Convert bytes to a list of integers, adding a shift to account for special tokens
        # (<pad>, </s>, <unk> are typically 0, 1, 2, so we shift by 3 to avoid collision)
        model_inputs = np.array([list(byte + 3 for byte in encoded_bytes)], dtype=np.int64)
        return model_inputs

    def _infer_onnx(self, text: str, lang: str) -> str:
        """
        Performs inference using ONNX Runtime without relying on Hugging Face Tokenizer.

        Args:
            text (str): The input text for G2P conversion.
            lang (str): The language of the input text.

        Returns:
            str: The predicted phoneme string. Returns an empty string if the input text is empty.
        """
        if not text.strip():
            return ""

        # Get the names of the model's output tensors
        onnx_output_names: List[str] = [out.name for out in self.session.get_outputs()]

        # Use the custom _encode_text function to prepare input_ids
        input_ids_np: np.ndarray = self._encode_text(text, lang)

        # Manually create attention_mask (all ones for ByT5, indicating all tokens are attended to)
        attention_mask_np: np.ndarray = np.ones_like(input_ids_np, dtype=np.int64)

        # Hardcode decoder_start_token_id for ByT5 (typically 0 for pad_token_id)
        # This is the initial token fed to the decoder to start generation.
        decoder_start_token_id: int = 0  # Corresponds to <pad> for ByT5

        generated_ids: List[int] = []
        # Initialize the decoder input with the start token
        decoder_input_ids_np: np.ndarray = np.array([[decoder_start_token_id]], dtype=np.int64)

        max_length: int = 512  # Maximum length for the generated sequence

        # Greedy decoding loop
        for _ in range(max_length):
            # Prepare inputs for the ONNX session
            onnx_inputs: Dict[str, np.ndarray] = {
                "input_ids": input_ids_np,
                "attention_mask": attention_mask_np,
                "decoder_input_ids": decoder_input_ids_np
            }

            # Run inference
            outputs: List[np.ndarray] = self.session.run(onnx_output_names, onnx_inputs)
            logits: np.ndarray = outputs[0]  # Get the logits from the model output

            # Get the logits for the last token in the sequence
            next_token_logits: np.ndarray = logits[0, -1, :]
            # Predict the next token by taking the argmax of the logits
            next_token_id: int = np.argmax(next_token_logits).item()  # .item() to get scalar from numpy array
            generated_ids.append(next_token_id)

            # Assuming EOS token ID for ByT5 is 1 (corresponds to </s>)
            # This is a common convention for T5 models.
            eos_token_id: int = 1
            # If the EOS token is generated, stop decoding
            if next_token_id == eos_token_id:
                break

            # Append the newly generated token to the decoder input for the next step
            decoder_input_ids_np = np.concatenate((decoder_input_ids_np,
                                                   np.array([[next_token_id]],
                                                            dtype=np.int64)),
                                                  axis=1)

        # Decode the generated token IDs into phonemes
        return self._decode_phones(generated_ids)

    def phonemize_string(self, text: str, lang: str) -> str:
        return self._infer_onnx(text, lang)


class CharsiuPhonemizer(ByT5Phonemizer):
    """
    A phonemizer class that uses a Charsiu ByT5 ONNX model to convert text into phonemes.
    """
    # TODO - more models
    MODEL2URL = {
        "Jarbas/charsiu_g2p_multilingual_byT5_tiny_16_layers_100_onnx": "https://huggingface.co/Jarbas/charsiu_g2p_multilingual_byT5_tiny_16_layers_100_onnx/resolve/main/charsiu_g2p_multilingual_byT5_tiny_16_layers_100.onnx"
    }
    BYT5_LANGS = ['ady', 'afr', 'sqi', 'amh', 'ara', 'arg', 'arm-e', 'arm-w', 'aze', 'bak', 'eus', 'bel', 'ben', 'bos',
                  'bul', 'bur', 'cat', 'yue', 'zho-t', 'zho-s', 'min', 'cze', 'dan', 'dut', 'eng-uk', 'eng-us', 'epo',
                  'est', 'fin', 'fra', 'fra-qu', 'gla', 'geo', 'ger', 'gre', 'grc', 'grn', 'guj', 'hin', 'hun', 'ido',
                  'ind', 'ina', 'ita', 'jam', 'jpn', 'kaz', 'khm', 'kor', 'kur', 'lat-clas', 'lat-eccl', 'lit', 'ltz',
                  'mac', 'mlt', 'tts', 'nob', 'ori', 'pap', 'fas', 'pol', 'por-po', 'por-bz', 'ron', 'rus', 'san',
                  'srp', 'hbs-latn', 'hbs-cyrl', 'snd', 'slo', 'slv', 'spa', 'spa-latin', 'spa-me', 'swa', 'swe', 'tgl',
                  'tam', 'tat', 'tha', 'tur', 'tuk', 'ukr', 'vie-n', 'vie-c', 'vie-s', 'wel-nw', 'wel-sw', 'ice', 'ang',
                  'gle', 'enm', 'syc', 'glg', 'sme', 'egy']

    def __init__(self, model: Optional[str] = None, tokenizer_config: Optional[str] = None,
                 use_cuda=bool(os.environ.get("CUDA", False))):
        """
        Initializes the ByT5Phonemizer with the ONNX model and tokenizer configuration.
        If paths are not provided, it attempts to download them to a local directory.

        Args:
            model (str, optional): Path to the ONNX model file. If None, it will be downloaded.
            tokenizer_config (str, optional): Path to the tokenizer configuration JSON file. If None, it will be downloaded.
        """
        model = model or "Jarbas/charsiu_g2p_multilingual_byT5_tiny_16_layers_100_onnx"
        super().__init__(model, tokenizer_config, use_cuda)

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        """
        Validates and returns the closest supported language code.

        Args:
            target_lang (str): The language code to validate.

        Returns:
            str: The validated language code.

        Raises:
            ValueError: If the language code is unsupported.
        """
        # Find the closest match
        return cls.match_lang(target_lang, cls.BYT5_LANGS)

    def phonemize_string(self, text: str, lang: str) -> str:
        # charsiu models can't handle whitespace, need to be phonemized word by word
        return " ".join([self._infer_onnx(w, lang) for w in text.split()])


class EspeakPhonemizer(BasePhonemizer):
    """
    A phonemizer class that uses the espeak-ng command-line tool to convert text into phonemes.
    It segments the input text heuristically based on punctuation to mimic clause-by-clause processing.
    """
    ESPEAK_LANGS = ['es-419', 'ca', 'qya', 'ga', 'et', 'ky', 'io', 'fa-latn', 'en-gb', 'fo', 'haw', 'kl',
                    'ta', 'ml', 'gd', 'sd', 'es', 'hy', 'ur', 'ro', 'hi', 'or', 'ti', 'ca-va', 'om', 'tr', 'pa',
                    'smj', 'mk', 'bg', 'cv', "fr", 'fi', 'en-gb-x-rp', 'ru', 'mt', 'an', 'mr', 'pap', 'vi', 'id',
                    'fr-be', 'ltg', 'my', 'nl', 'shn', 'ba', 'az', 'cmn', 'da', 'as', 'sw',
                    'piqd', 'en-us', 'hr', 'it', 'ug', 'th', 'mi', 'cy', 'ru-lv', 'ia', 'tt', 'hu', 'xex', 'te', 'ne',
                    'eu', 'ja', 'bpy', 'hak', 'cs', 'en-gb-scotland', 'hyw', 'uk', 'pt', 'bn', 'mto', 'yue',
                    'be', 'gu', 'sv', 'sl', 'cmn-latn-pinyin', 'lfn', 'lv', 'fa', 'sjn', 'nog', 'ms',
                    'vi-vn-x-central', 'lt', 'kn', 'he', 'qu', 'ca-ba', 'quc', 'nb', 'sk', 'tn', 'py', 'si', 'de',
                    'ar', 'en-gb-x-gbcwmd', 'bs', 'qdb', 'sq', 'sr', 'tk', 'en-029', 'ht', 'ru-cl', 'af', 'pt-br',
                    'fr-ch', 'ka', 'en-gb-x-gbclan', 'ko', 'is', 'ca-nw', 'gn', 'kok', 'la', 'lb', 'am', 'kk', 'ku',
                    'kaa', 'jbo', 'eo', 'uz', 'nci', 'vi-vn-x-south', 'el', 'pl', 'grc', ]

    def __init__(self):
        super().__init__(Alphabet.IPA)

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        """
        Validates and returns the closest supported language code.

        Args:
            target_lang (str): The language code to validate.

        Returns:
            str: The validated language code.

        Raises:
            ValueError: If the language code is unsupported.
        """
        if target_lang.lower() == "en-gb":
            return "en-gb-x-rp"
        if target_lang in cls.ESPEAK_LANGS:
            return target_lang
        if target_lang.lower().split("-")[0] in cls.ESPEAK_LANGS:
            return target_lang.lower().split("-")[0]
        return cls.match_lang(target_lang, cls.ESPEAK_LANGS)

    @staticmethod
    def _run_espeak_command(args: List[str], input_text: str = None, check: bool = True) -> str:
        """
        Helper function to run espeak-ng commands via subprocess.
        Executes 'espeak-ng' with the given arguments and input text.
        Captures stdout and stderr, and raises EspeakError on failure.

        Args:
            args (List[str]): A list of command-line arguments for espeak-ng.
            input_text (str, optional): The text to pass to espeak-ng's stdin. Defaults to None.
            check (bool, optional): If True, raises a CalledProcessError if the command returns a non-zero exit code. Defaults to True.

        Returns:
            str: The stripped standard output from the espeak-ng command.

        Raises:
            EspeakError: If espeak-ng command is not found, or if the subprocess call fails.
        """
        command: List[str] = ['espeak-ng'] + args
        try:
            process: subprocess.CompletedProcess = subprocess.run(
                command,
                input=input_text,
                capture_output=True,
                text=True,
                check=check,
                encoding='utf-8',
                errors='replace'  # Replaces unencodable characters with a placeholder
            )
            return process.stdout.strip()
        except FileNotFoundError:
            raise EspeakError(
                "espeak-ng command not found. Please ensure espeak-ng is installed "
                "and available in your system's PATH."
            )
        except subprocess.CalledProcessError as e:
            raise EspeakError(
                f"espeak-ng command failed with error code {e.returncode}:\n"
                f"STDOUT: {e.stdout}\n"
                f"STDERR: {e.stderr}"
            )
        except Exception as e:
            raise EspeakError(f"An unexpected error occurred while running espeak-ng: {e}")

    def phonemize_string(self, text: str, lang: str) -> str:
        lang = self.get_lang(lang)
        return self._run_espeak_command(
            ['-q', '-x', '--ipa', '-v', lang],
            input_text=text
        )


class GruutPhonemizer(BasePhonemizer):
    """
    A phonemizer class that uses the Gruut library to convert text into phonemes.
    Note: Gruut's internal segmentation is sentence-based
    """
    GRUUT_LANGS = ["en", "ar", "ca", "cs", "de", "es", "fa", "fr", "it",
                   "lb", "nl", "pt", "ru", "sv", "sw"]

    def __init__(self):
        super().__init__(Alphabet.IPA)

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        """
        Validates and returns the closest supported language code.

        Args:
            target_lang (str): The language code to validate.

        Returns:
            str: The validated language code.

        Raises:
            ValueError: If the language code is unsupported.
        """
        return cls.match_lang(target_lang, cls.GRUUT_LANGS)

    def _text_to_phonemes(self, text: str, lang: Optional[str] = None):
        """
        Generates phonemes for text using Gruut's sentence processing.
        Yields lists of word phonemes for each sentence.
        """
        lang = self.get_lang(lang)
        import gruut
        for sentence in gruut.sentences(text, lang=lang):
            sent_phonemes = [w.phonemes for w in sentence if w.phonemes]
            if sentence and not sent_phonemes:
                raise RuntimeError(f"did you install gruut[{lang}] ?")
            if sentence.text.endswith("?"):
                sent_phonemes[-1] = ["?"]
            elif sentence.text.endswith("!"):
                sent_phonemes[-1] = ["!"]
            elif sentence.text.endswith(".") or sent_phonemes[-1] == ["‖"]:
                sent_phonemes[-1] = ["."]
            if sent_phonemes:
                yield sent_phonemes

    def phonemize_string(self, text: str, lang: str) -> str:
        pho = ""
        for sent_phonemes in self._text_to_phonemes(text, lang):
            pho += " ".join(["".join(w) for w in sent_phonemes]) + " "
        return pho.strip()


class GoruutPhonemizer(BasePhonemizer):
    """
    A phonemizer class that uses the pygoruut library to convert text into phonemes.
    https://github.com/neurlang/pygoruut/
    """
    GORUUT_LANGS_NON_STD = [
        'BengaliDhaka', 'BengaliRahr', 'MalayArab', 'VietnameseCentral', 'VietnameseSouthern',
        'EnglishAmerican', 'EnglishBritish', 'NahuatlClassical', 'Hebrew2', 'Hebrew3',
        'MinnanTawianese', 'MinnanHokkien', 'MinnanTawianese2', 'MinnanHokkien2']
    ISO639 = {
        "af": "Afrikaans",
        "am": "Amharic",
        "ar": "Arabic",
        "az": "Azerbaijani",
        "be": "Belarusian",
        "bn": "Bengali",
        "my": "Burmese",
        "ceb": "Cebuano",
        "ce": "Chechen",
        "zh": "ChineseMandarin",
        "cs": "Czech",
        "da": "Danish",
        "nl": "Dutch",
        "dz": "Dzongkha",
        "en": "English",
        "eo": "Esperanto",
        "fa": "Farsi",
        "fi": "Finnish",
        "fr": "French",
        "de": "German",
        "el": "Greek",
        "gu": "Gujarati",
        "ha": "Hausa",
        "he": "Hebrew",
        "hi": "Hindi",
        "hu": "Hungarian",
        "is": "Icelandic",
        "id": "Indonesian",
        "tts": "Isan",
        "it": "Italian",
        "jam": "Jamaican",
        "ja": "Japanese",
        "jv": "Javanese",
        "kk": "Kazakh",
        "ko": "Korean",
        "lb": "Luxembourgish",
        "mk": "Macedonian",
        "ml": "Malayalam",
        "ms": "MalayLatin",
        "mt": "Maltese",
        "mr": "Marathi",
        "mn": "Mongolian",
        "ne": "Nepali",
        "no": "Norwegian",
        "ps": "Pashto",
        "pl": "Polish",
        "pt": "Portuguese",
        "pa": "Punjabi",
        "ro": "Romanian",
        "ru": "Russian",
        "sk": "Slovak",
        "es": "Spanish",
        "sw": "Swahili",
        "sv": "Swedish",
        "ta": "Tamil",
        "te": "Telugu",
        "th": "Thai",
        "bo": "Tibetan",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "ug": "Uyghur",
        "vi": "VietnameseNorthern",
        "zu": "Zulu",
        "hy": "Armenian",
        "eu": "Basque",
        "bg": "Bulgarian",
        "ca": "Catalan",
        "ny": "Chichewa",
        "hr": "Croatian",
        "et": "Estonian",
        "gl": "Galician",
        "ka": "Georgian",
        "km": "KhmerCentral",
        "lo": "Lao",
        "lv": "Latvian",
        "lt": "Lithuanian",
        "sr": "Serbian",
        "tl": "Tagalog",
        "yo": "Yoruba",
        "sq": "Albanian",
        "an": "Aragonese",
        "as": "Assamese",
        "ba": "Bashkir",
        "bpy": "BishnupriyaManipuri",
        "bs": "Bosnian",
        "chr": "Cherokee",
        "cu": "Chuvash",
        "gla": "GaelicScottish",
        "gle": "GaelicIrish",
        "kl": "Greenlandic",
        "gn": "Guarani",
        "ht": "HaitianCreole",
        "haw": "Hawaiian",
        "io": "Ido",
        "ia": "Interlingua",
        "kn": "Kannada",
        "quc": "Kiche",
        "kok": "Konkani",
        "ku": "Kurdish",
        "ky": "Kyrgyz",
        "qdb": "LangBelta",
        "ltg": "Latgalian",
        "la": "LatinClassical",
        "lat": "LatinEcclesiastical",
        "lfn": "LinguaFrancaNova",
        "jbo": "Lojban",
        "smj": "LuleSaami",
        "mi": "Maori",
        "nah": "NahuatlCentral",
        "nci": "NahuatlMecayapan",
        "ncz": "NahuatlTetelcingo",
        "nog": "Nogai",
        "om": "Oromo",
        "pap": "Papiamento",
        "qu": "Quechua",
        "qya": "Quenya",
        "tn": "Setswana",
        "shn": "ShanTaiYai",
        "sjn": "Sindarin",
        "sd": "Sindhi",
        "si": "Sinhala",
        "sl": "Slovenian",
        "tt": "Tatar",
        "tk": "Turkmen",
        "uz": "Uzbek",
        "cyw": "WelshNorth",
        "cys": "WelshSouth",
        "yue": "Cantonese"
    }

    def __init__(self, remote_url=None):
        super().__init__(Alphabet.IPA)
        from pygoruut.pygoruut import Pygoruut
        from pygoruut.pygoruut_languages import PygoruutLanguages

        self.pygoruut_langs = PygoruutLanguages()
        if remote_url is not None:
            # 'https://hashtron.cloud'
            self.pygoruut = Pygoruut(api=remote_url)
        else:
            self.pygoruut = Pygoruut()

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        """
        Validates and returns the closest supported language code.

        Args:
            target_lang (str): The language code to validate.

        Returns:
            str: The validated language code.

        Raises:
            ValueError: If the language code is unsupported.
        """
        if target_lang in cls.GORUUT_LANGS_NON_STD:
            return target_lang
        if target_lang.lower() == "en-us":
            return 'EnglishAmerican'
        if target_lang.lower() == "en-gb" or target_lang.lower() == "en-uk":
            return 'EnglishBritish'
        lang = cls.match_lang(target_lang, list(cls.ISO639))
        return cls.ISO639[lang]

    def phonemize_string(self, text: str, lang: str) -> str:
        lang = self.get_lang(lang)
        return str(self.pygoruut.phonemize(language=lang, sentence=text))


class EpitranPhonemizer(BasePhonemizer):
    """
    """
    EPITRAN_LANGS = ['hsn-Latn', 'ful-Latn', 'jpn-Ktkn-red', 'tel-Telu', 'nld-Latn', 'aze-Latn', 'amh-Ethi-pp',
                     'msa-Latn', 'spa-Latn-eu', 'ori-Orya', 'bxk-Latn', 'spa-Latn', 'kir-Cyrl', 'lij-Latn', 'kin-Latn',
                     'ces-Latn', 'sin-Sinh', 'urd-Arab', 'vie-Latn', 'gan-Latn', 'fra-Latn', 'nan-Latn', 'kaz-Latn',
                     'swe-Latn', 'jpn-Ktkn', 'tam-Taml', 'sag-Latn', 'csb-Latn', 'pii-latn_Holopainen2019', 'yue-Latn',
                     'got-Latn', 'tur-Latn', 'aar-Latn', 'jav-Latn', 'ita-Latn', 'sna-Latn', 'ilo-Latn', 'tam-Taml-red',
                     'kmr-Latn-red', 'uzb-Cyrl', 'amh-Ethi', 'mya-Mymr', 'aii-Syrc', 'lit-Latn', 'kmr-Latn',
                     'hat-Latn-bab', 'ltc-Latn-bax', 'Goth2Latn', 'quy-Latn', 'hau-Latn', 'ood-Latn-alv', 'vie-Latn-so',
                     'run-Latn', 'orm-Latn', 'ind-Latn', 'kir-Latn', 'mal-Mlym', 'ben-Beng-red', 'hun-Latn', 'uew',
                     'sqi-Latn', 'jpn-Hrgn', 'deu-Latn-np', 'xho-Latn', 'fra-Latn-rev', 'fra-Latn-np', 'kaz-Cyrl-bab',
                     'jpn-Hrgn-red', 'Latn2Goth', 'glg-Latn', 'uig-Arab', 'amh-Ethi-red', 'zul-Latn', 'hin-Deva',
                     'uzb-Latn', 'tir-Ethi-red', 'kaz-Cyrl', 'mlt-Latn', 'deu-Latn-nar', 'est-Latn', 'eng-Latn',
                     'pii-latn_Wiktionary', 'ckb-Arab', 'nya-Latn', 'mon-Cyrl-bab', 'fra-Latn-p', 'ood-Latn-sax',
                     'ukr-Cyrl', 'tgl-Latn-red', 'lsm-Latn', 'kor-Hang', 'lav-Latn', 'generic-Latn', 'tur-Latn-red',
                     'srp-Latn', 'tir-Ethi', 'kbd-Cyrl', 'hrv-Latn', 'srp-Cyrl', 'tpi-Latn', 'khm-Khmr', 'jam-Latn',
                     'ben-Beng-east', 'por-Latn', 'cmn-Latn', 'cat-Latn', 'tha-Thai', 'ara-Arab', 'ben-Beng',
                     'fin-Latn', 'hmn-Latn', 'lez-Cyrl', 'fas-Arab', 'lao-Laoo-prereform', 'mar-Deva', 'yor-Latn',
                     'ron-Latn', 'tgl-Latn', 'lao-Laoo', 'deu-Latn', 'pan-Guru', 'tuk-Latn', 'tir-Ethi-pp', 'rus-Cyrl',
                     'swa-Latn-red', 'ceb-Latn', 'wuu-Latn', 'hak-Latn', 'mri-Latn', 'epo-Latn', 'pol-Latn',
                     'tur-Latn-bab', 'kat-Geor', 'tgk-Cyrl', 'aze-Cyrl', 'vie-Latn-ce', 'swa-Latn', 'tuk-Cyrl',
                     'vie-Latn-no', 'nan-Latn-tl', 'zha-Latn', 'cjy-Latn', 'ava-Cyrl', 'som-Latn', 'kir-Arab']

    def __init__(self):
        super().__init__(Alphabet.IPA)
        import epitran
        self.epitran = epitran
        self._epis: Dict[str, epitran.Epitran] = {}

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        """
        Validates and returns the closest supported language code.

        Args:
            target_lang (str): The language code to validate.

        Returns:
            str: The validated language code.

        Raises:
            ValueError: If the language code is unsupported.
        """
        return cls.match_lang(target_lang, cls.EPITRAN_LANGS)

    def phonemize_string(self, text: str, lang: str) -> str:
        lang = self.get_lang(lang)
        epi = self._epis.get(lang)
        if epi is None:
            epi = self.epitran.Epitran(lang)
            self._epis[lang] = epi
        return epi.transliterate(text)


class MisakiPhonemizer(BasePhonemizer):
    """
    https://github.com/hexgrad/misaki
    """
    MISAKI_LANGS = ['en-US', 'en-GB', 'ko', 'ja', 'vi', 'zh']

    def __init__(self):
        super().__init__(Alphabet.IPA)
        self.g2p_en = self.g2p_zh = self.g2p_ko = self.g2p_vi = self.g2p_ja = None

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        """
        Validates and returns the closest supported language code.

        Args:
            target_lang (str): The language code to validate.

        Returns:
            str: The validated language code.

        Raises:
            ValueError: If the language code is unsupported.
        """
        return cls.match_lang(target_lang, cls.MISAKI_LANGS)

    def _get_phonemizer(self, lang: str):
        """lazy load language specific phonemizer on first usage
        NOTE: this can be slow
        """
        lang = self.get_lang(lang)

        if lang == "zh":
            if self.g2p_zh is None:
                from misaki.zh import ZHG2P
                self.g2p_zh = ZHG2P()
            return self.g2p_zh
        elif lang == "ko":
            if self.g2p_ko is None:
                from misaki.ko import KOG2P
                self.g2p_ko = KOG2P()
            return self.g2p_ko
        elif lang == "vi":
            if self.g2p_vi is None:
                from misaki.vi import VIG2P
                self.g2p_vi = VIG2P()
            return self.g2p_vi
        elif lang == "ja":
            if self.g2p_ja is None:
                from misaki.ja import JAG2P
                self.g2p_ja = JAG2P()
            return self.g2p_ja
        else:
            if self.g2p_en is None:
                from misaki import en
                self.g2p_en = en.G2P()
            if lang == "en-GB":
                self.g2p_en.british = True
            elif lang == "en-US":
                self.g2p_en.british = False
            return self.g2p_en

    def phonemize_string(self, text: str, lang: str) -> str:
        pho = self._get_phonemizer(lang)
        phonemes, tokens = pho(text)
        return phonemes


class TransphonePhonemizer(BasePhonemizer):
    """
    It provides approximiated phoneme tokenizers and G2P model for 7546 languages registered in the Glottolog database.
    """
    TRANSPHONE_LANGS = ['aaa', 'aab', 'aac', 'aad', 'aae', 'aaf', 'aag', 'aah', 'aai', 'aak', 'aal', 'aan', 'aao',
                        'aap', 'aaq', 'aar', 'aas', 'aat', 'aau', 'aaw', 'aax', 'aaz', 'aba', 'abb', 'abc', 'abd',
                        'abe', 'abf', 'abg', 'abh', 'abi', 'abj', 'abk', 'abl', 'abm', 'abn', 'abo', 'abp', 'abq',
                        'abr', 'abs', 'abt', 'abu', 'abv', 'abw', 'abx', 'aby', 'abz', 'aca', 'acd', 'ace', 'acf',
                        'ach', 'aci', 'ack', 'acl', 'acm', 'acn', 'acp', 'acq', 'acr', 'acs', 'act', 'acu', 'acv',
                        'acw', 'acx', 'acy', 'acz', 'ada', 'add', 'ade', 'adf', 'adg', 'adh', 'adi', 'adj', 'adl',
                        'adn', 'ado', 'adq', 'adr', 'ads', 'adt', 'adw', 'adx', 'ady', 'adz', 'aea', 'aeb', 'aec',
                        'aed', 'aee', 'aek', 'ael', 'aem', 'aen', 'aeq', 'aer', 'aes', 'aeu', 'aew', 'aey', 'aez',
                        'afb', 'afd', 'afe', 'afg', 'afi', 'afk', 'afn', 'afo', 'afp', 'afr', 'afs', 'aft', 'afu',
                        'afz', 'agb', 'agc', 'agd', 'age', 'agf', 'agg', 'agh', 'agj', 'agk', 'agl', 'agm', 'agn',
                        'ago', 'agq', 'agr', 'ags', 'agt', 'agu', 'agv', 'agw', 'agx', 'agy', 'agz', 'aha', 'ahb',
                        'ahg', 'ahh', 'ahi', 'ahk', 'ahl', 'ahm', 'ahn', 'aho', 'ahp', 'ahs', 'aht', 'aia', 'aib',
                        'aic', 'aid', 'aie', 'aif', 'aig', 'aih', 'aii', 'aij', 'aik', 'ail', 'aim', 'ain', 'aio',
                        'aip', 'aiq', 'air', 'ais', 'ait', 'aiw', 'aix', 'aiy', 'aja', 'ajg', 'aji', 'ajn', 'ajp',
                        'ajt', 'aju', 'ajw', 'ajz', 'aka', 'akb', 'akc', 'akd', 'ake', 'akf', 'akg', 'akh', 'aki',
                        'akj', 'akk', 'akl', 'akm', 'ako', 'akp', 'akq', 'akr', 'aks', 'akt', 'aku', 'akv', 'akw',
                        'akx', 'aky', 'akz', 'ala', 'alc', 'ald', 'ale', 'alf', 'alh', 'ali', 'alj', 'alk', 'all',
                        'alm', 'aln', 'alo', 'alp', 'alq', 'alr', 'als', 'alt', 'alu', 'alw', 'alx', 'aly', 'alz',
                        'ama', 'amb', 'amc', 'ame', 'amf', 'amg', 'amh', 'ami', 'amj', 'amk', 'aml', 'amm', 'amn',
                        'amo', 'amp', 'amq', 'amr', 'ams', 'amt', 'amu', 'amv', 'amw', 'amx', 'amy', 'amz', 'ana',
                        'anb', 'anc', 'and', 'ane', 'anf', 'ang', 'anh', 'ani', 'anj', 'ank', 'anl', 'anm', 'ann',
                        'ano', 'anp', 'anq', 'anr', 'ans', 'ant', 'anu', 'anv', 'anw', 'anx', 'any', 'anz', 'aoa',
                        'aob', 'aoc', 'aod', 'aoe', 'aof', 'aog', 'aoi', 'aoj', 'aok', 'aol', 'aom', 'aon', 'aor',
                        'aos', 'aot', 'aou', 'aox', 'aoz', 'apb', 'apc', 'apd', 'ape', 'apf', 'apg', 'aph', 'api',
                        'apj', 'apk', 'apl', 'apm', 'apn', 'apo', 'app', 'apq', 'apr', 'aps', 'apt', 'apu', 'apw',
                        'apx', 'apy', 'apz', 'aqc', 'aqd', 'aqg', 'aqm', 'aqn', 'aqp', 'aqr', 'aqt', 'aqz', 'arb',
                        'arc', 'ard', 'are', 'arg', 'arh', 'ari', 'arj', 'ark', 'arl', 'arn', 'aro', 'arp', 'arq',
                        'arr', 'ars', 'aru', 'arv', 'arw', 'arx', 'ary', 'arz', 'asa', 'asb', 'asc', 'asd', 'ase',
                        'asf', 'asg', 'ash', 'asi', 'asj', 'ask', 'asl', 'asm', 'asn', 'aso', 'asp', 'asq', 'asr',
                        'ass', 'ast', 'asu', 'asv', 'asw', 'asx', 'asy', 'asz', 'ata', 'atb', 'atc', 'atd', 'ate',
                        'atg', 'ati', 'atj', 'atk', 'atl', 'atm', 'atn', 'ato', 'atp', 'atq', 'atr', 'ats', 'att',
                        'atu', 'atv', 'atw', 'atx', 'aty', 'atz', 'aua', 'aub', 'auc', 'aud', 'aug', 'auh', 'aui',
                        'auj', 'auk', 'aul', 'aum', 'aun', 'auo', 'aup', 'auq', 'aur', 'aut', 'auu', 'auw', 'aux',
                        'auy', 'auz', 'ava', 'avb', 'avd', 'ave', 'avi', 'avl', 'avm', 'avn', 'avs', 'avt', 'avu',
                        'avv', 'awa', 'awb', 'awc', 'awe', 'awg', 'awh', 'awi', 'awk', 'awm', 'awn', 'awo', 'awr',
                        'aws', 'awt', 'awu', 'awv', 'aww', 'awx', 'awy', 'axb', 'axe', 'axg', 'axk', 'axl', 'axx',
                        'aya', 'ayb', 'ayc', 'ayd', 'aye', 'ayg', 'ayh', 'ayi', 'ayk', 'ayl', 'aym', 'ayn', 'ayo',
                        'ayp', 'ayq', 'ayr', 'ayt', 'ayu', 'ayz', 'aza', 'azb', 'azd', 'azg', 'azj', 'azm', 'azn',
                        'azo', 'azt', 'azz', 'baa', 'bab', 'bac', 'bae', 'baf', 'bag', 'bah', 'baj', 'bak', 'bam',
                        'ban', 'bao', 'bap', 'bar', 'bas', 'bau', 'bav', 'baw', 'bax', 'bay', 'bba', 'bbb', 'bbc',
                        'bbd', 'bbe', 'bbf', 'bbg', 'bbh', 'bbi', 'bbj', 'bbk', 'bbl', 'bbm', 'bbn', 'bbo', 'bbp',
                        'bbq', 'bbr', 'bbs', 'bbt', 'bbu', 'bbv', 'bbw', 'bby', 'bca', 'bcc', 'bcd', 'bce', 'bcf',
                        'bcg', 'bch', 'bci', 'bcj', 'bck', 'bcl', 'bcm', 'bcn', 'bco', 'bcp', 'bcq', 'bcr', 'bcs',
                        'bct', 'bcu', 'bcv', 'bcw', 'bcy', 'bcz', 'bda', 'bdb', 'bdc', 'bdd', 'bde', 'bdf', 'bdg',
                        'bdh', 'bdi', 'bdj', 'bdk', 'bdl', 'bdm', 'bdn', 'bdo', 'bdp', 'bdq', 'bdr', 'bds', 'bdt',
                        'bdu', 'bdv', 'bdw', 'bdx', 'bdy', 'bea', 'beb', 'bec', 'bed', 'bee', 'bef', 'beg', 'beh',
                        'bei', 'bej', 'bek', 'bel', 'bem', 'ben', 'beo', 'bep', 'beq', 'bes', 'bet', 'beu', 'bev',
                        'bew', 'bex', 'bey', 'bez', 'bfa', 'bfb', 'bfc', 'bfd', 'bfe', 'bff', 'bfg', 'bfh', 'bfi',
                        'bfj', 'bfk', 'bfl', 'bfm', 'bfn', 'bfo', 'bfp', 'bfq', 'bfs', 'bft', 'bfu', 'bfw', 'bfx',
                        'bfy', 'bfz', 'bga', 'bgb', 'bgc', 'bgd', 'bge', 'bgf', 'bgg', 'bgi', 'bgj', 'bgk', 'bgl',
                        'bgm', 'bgn', 'bgo', 'bgp', 'bgq', 'bgr', 'bgs', 'bgt', 'bgu', 'bgv', 'bgw', 'bgx', 'bgy',
                        'bgz', 'bha', 'bhb', 'bhc', 'bhd', 'bhe', 'bhf', 'bhg', 'bhh', 'bhi', 'bhj', 'bhl', 'bhm',
                        'bhn', 'bho', 'bhp', 'bhq', 'bhr', 'bhs', 'bht', 'bhu', 'bhv', 'bhw', 'bhy', 'bhz', 'bia',
                        'bib', 'bid', 'bie', 'bif', 'big', 'bij', 'bil', 'bim', 'bin', 'bio', 'bip', 'biq', 'bir',
                        'bis', 'bit', 'biu', 'biv', 'biw', 'bix', 'biy', 'biz', 'bja', 'bjb', 'bjc', 'bje', 'bjf',
                        'bjg', 'bjh', 'bji', 'bjj', 'bjk', 'bjl', 'bjm', 'bjn', 'bjo', 'bjp', 'bjr', 'bjs', 'bjt',
                        'bju', 'bjv', 'bjw', 'bjx', 'bjy', 'bjz', 'bka', 'bkc', 'bkd', 'bkf', 'bkh', 'bki', 'bkj',
                        'bkk', 'bkl', 'bkm', 'bkn', 'bko', 'bkp', 'bkq', 'bkr', 'bks', 'bkt', 'bku', 'bkv', 'bkw',
                        'bkx', 'bky', 'bkz', 'bla', 'blb', 'blc', 'bld', 'ble', 'blf', 'blh', 'bli', 'blj', 'blk',
                        'bll', 'blm', 'bln', 'blo', 'blp', 'blq', 'blr', 'bls', 'blt', 'blv', 'blw', 'blx', 'bly',
                        'blz', 'bma', 'bmb', 'bmc', 'bmd', 'bme', 'bmf', 'bmg', 'bmh', 'bmi', 'bmj', 'bmk', 'bml',
                        'bmm', 'bmn', 'bmo', 'bmp', 'bmq', 'bmr', 'bms', 'bmt', 'bmu', 'bmv', 'bmw', 'bmx', 'bmz',
                        'bna', 'bnb', 'bnc', 'bnd', 'bne', 'bnf', 'bng', 'bni', 'bnj', 'bnk', 'bnl', 'bnm', 'bnn',
                        'bno', 'bnp', 'bnq', 'bnr', 'bns', 'bnu', 'bnv', 'bnw', 'bnx', 'bny', 'bnz', 'boa', 'bob',
                        'bod', 'boe', 'bof', 'bog', 'boh', 'boi', 'boj', 'bok', 'bol', 'bom', 'bon', 'boo', 'bop',
                        'boq', 'bor', 'bos', 'bot', 'bou', 'bov', 'bow', 'box', 'boy', 'boz', 'bpa', 'bpd', 'bpg',
                        'bph', 'bpi', 'bpj', 'bpk', 'bpm', 'bpn', 'bpp', 'bpq', 'bpr', 'bps', 'bpt', 'bpu', 'bpv',
                        'bpw', 'bpx', 'bpy', 'bpz', 'bqa', 'bqb', 'bqc', 'bqd', 'bqg', 'bqh', 'bqi', 'bqj', 'bqk',
                        'bql', 'bqm', 'bqn', 'bqo', 'bqp', 'bqq', 'bqr', 'bqs', 'bqt', 'bqu', 'bqv', 'bqw', 'bqx',
                        'bqy', 'bqz', 'bra', 'brb', 'brc', 'brd', 'bre', 'brf', 'brg', 'brh', 'bri', 'brj', 'brk',
                        'brl', 'brm', 'brn', 'bro', 'brp', 'brq', 'brr', 'brs', 'brt', 'bru', 'brv', 'brw', 'brx',
                        'bry', 'brz', 'bsa', 'bsb', 'bsc', 'bse', 'bsf', 'bsg', 'bsh', 'bsi', 'bsj', 'bsk', 'bsl',
                        'bsm', 'bsn', 'bsp', 'bsq', 'bsr', 'bss', 'bst', 'bsu', 'bsw', 'bsx', 'bsy', 'bta', 'btc',
                        'btd', 'bte', 'btf', 'btg', 'bth', 'bti', 'btj', 'btm', 'btn', 'bto', 'btp', 'btq', 'btr',
                        'bts', 'btt', 'btu', 'btv', 'btw', 'btx', 'bty', 'btz', 'bua', 'bub', 'buc', 'bud', 'bue',
                        'buf', 'bug', 'buh', 'bui', 'buj', 'buk', 'bul', 'bum', 'bun', 'buo', 'bup', 'buq', 'bus',
                        'but', 'buu', 'buv', 'buw', 'bux', 'buy', 'buz', 'bva', 'bvb', 'bvc', 'bvd', 'bve', 'bvf',
                        'bvg', 'bvh', 'bvi', 'bvj', 'bvk', 'bvl', 'bvm', 'bvn', 'bvo', 'bvq', 'bvr', 'bvt', 'bvu',
                        'bvv', 'bvw', 'bvx', 'bvy', 'bvz', 'bwa', 'bwb', 'bwc', 'bwd', 'bwe', 'bwf', 'bwg', 'bwh',
                        'bwi', 'bwj', 'bwk', 'bwl', 'bwm', 'bwn', 'bwo', 'bwp', 'bwq', 'bwr', 'bws', 'bwt', 'bwu',
                        'bww', 'bwx', 'bwy', 'bwz', 'bxa', 'bxb', 'bxc', 'bxd', 'bxe', 'bxf', 'bxg', 'bxh', 'bxi',
                        'bxj', 'bxk', 'bxl', 'bxm', 'bxn', 'bxp', 'bxq', 'bxr', 'bxs', 'bxu', 'bxv', 'bxw', 'bxz',
                        'bya', 'byb', 'byc', 'byd', 'bye', 'byf', 'byg', 'byh', 'byi', 'byj', 'byk', 'byl', 'bym',
                        'byn', 'byo', 'byp', 'byq', 'byr', 'bys', 'byt', 'byv', 'byw', 'byx', 'byz', 'bza', 'bzb',
                        'bzc', 'bzd', 'bze', 'bzf', 'bzg', 'bzh', 'bzi', 'bzj', 'bzk', 'bzl', 'bzm', 'bzn', 'bzo',
                        'bzp', 'bzq', 'bzr', 'bzs', 'bzu', 'bzv', 'bzw', 'bzx', 'bzy', 'bzz', 'caa', 'cab', 'cac',
                        'cad', 'cae', 'caf', 'cag', 'cah', 'caj', 'cak', 'cal', 'cam', 'can', 'cao', 'cap', 'caq',
                        'car', 'cas', 'cat', 'cav', 'cax', 'cay', 'caz', 'cbb', 'cbc', 'cbd', 'cbg', 'cbi', 'cbj',
                        'cbk', 'cbl', 'cbn', 'cbo', 'cbq', 'cbr', 'cbs', 'cbt', 'cbu', 'cbv', 'cbw', 'ccc', 'ccd',
                        'cce', 'ccg', 'cch', 'ccj', 'ccl', 'ccm', 'cco', 'ccp', 'ccr', 'cda', 'cde', 'cdf', 'cdh',
                        'cdi', 'cdj', 'cdm', 'cdn', 'cdo', 'cdr', 'cds', 'cdy', 'cdz', 'cea', 'ceb', 'ceg', 'cek',
                        'cen', 'ces', 'cet', 'cfa', 'cfd', 'cfg', 'cfm', 'cga', 'cgc', 'cgg', 'cgk', 'cha', 'chb',
                        'chc', 'chd', 'che', 'chf', 'chg', 'chh', 'chj', 'chk', 'chl', 'chm', 'chn', 'cho', 'chp',
                        'chq', 'chr', 'cht', 'chu', 'chv', 'chw', 'chx', 'chy', 'chz', 'cia', 'cib', 'cic', 'cid',
                        'cie', 'cih', 'cik', 'cim', 'cin', 'cip', 'cir', 'ciw', 'ciy', 'cja', 'cje', 'cjh', 'cji',
                        'cjk', 'cjm', 'cjn', 'cjo', 'cjp', 'cjs', 'cjv', 'cjy', 'ckb', 'ckh', 'ckl', 'cko', 'ckq',
                        'ckr', 'cks', 'ckt', 'cku', 'ckv', 'ckx', 'cky', 'cla', 'clc', 'cld', 'cle', 'clh', 'cli',
                        'clk', 'cll', 'clm', 'clo', 'clt', 'clu', 'clw', 'cly', 'cma', 'cme', 'cmi', 'cml', 'cmn',
                        'cmo', 'cmr', 'cms', 'cna', 'cnb', 'cnc', 'cng', 'cnh', 'cni', 'cnk', 'cnl', 'cns', 'cnt',
                        'cnu', 'cnw', 'coa', 'cob', 'coc', 'cod', 'coe', 'cof', 'cog', 'coh', 'coj', 'cok', 'col',
                        'com', 'con', 'coo', 'cop', 'coq', 'cor', 'cos', 'cot', 'cou', 'cov', 'cow', 'cox', 'coz',
                        'cpa', 'cpb', 'cpc', 'cpg', 'cpn', 'cpo', 'cps', 'cpu', 'cpx', 'cpy', 'cra', 'crb', 'crc',
                        'crd', 'cre', 'crf', 'crg', 'crh', 'cri', 'crj', 'crk', 'crl', 'crm', 'crn', 'cro', 'crq',
                        'crr', 'crs', 'crt', 'crv', 'crw', 'crx', 'cry', 'crz', 'csa', 'csb', 'csc', 'csd', 'cse',
                        'csf', 'csg', 'csh', 'csi', 'csk', 'csl', 'csm', 'csn', 'cso', 'csq', 'csr', 'css', 'cst',
                        'csv', 'csw', 'csy', 'csz', 'cta', 'ctd', 'cte', 'ctg', 'ctl', 'ctm', 'ctn', 'cto', 'ctp',
                        'cts', 'ctt', 'ctu', 'ctz', 'cua', 'cub', 'cuc', 'cug', 'cuh', 'cui', 'cuj', 'cuk', 'cul',
                        'cuo', 'cup', 'cuq', 'cur', 'cut', 'cuu', 'cuv', 'cuw', 'cux', 'cvg', 'cvn', 'cwa', 'cwb',
                        'cwd', 'cwe', 'cwg', 'cwt', 'cya', 'cyb', 'cym', 'cyo', 'czh', 'czn', 'czo', 'czt', 'daa',
                        'dac', 'dad', 'dae', 'dag', 'dah', 'dai', 'daj', 'dak', 'dal', 'dam', 'dan', 'dao', 'daq',
                        'dar', 'das', 'dau', 'dav', 'daw', 'dax', 'daz', 'dba', 'dbb', 'dbd', 'dbe', 'dbf', 'dbg',
                        'dbi', 'dbj', 'dbl', 'dbm', 'dbn', 'dbo', 'dbp', 'dbq', 'dbr', 'dbt', 'dbu', 'dbw', 'dby',
                        'dcc', 'dcr', 'dda', 'ddd', 'dde', 'ddg', 'ddi', 'ddj', 'ddn', 'ddo', 'ddr', 'dds', 'ddw',
                        'dec', 'ded', 'dee', 'def', 'deg', 'deh', 'dei', 'del', 'dem', 'den', 'deq', 'der', 'des',
                        'deu', 'dev', 'dez', 'dga', 'dgb', 'dgc', 'dgd', 'dge', 'dgg', 'dgh', 'dgi', 'dgk', 'dgl',
                        'dgn', 'dgo', 'dgr', 'dgs', 'dgt', 'dgw', 'dgx', 'dgz', 'dhd', 'dhg', 'dhi', 'dhl', 'dhm',
                        'dhn', 'dho', 'dhr', 'dhs', 'dhu', 'dhv', 'dhw', 'dhx', 'dia', 'dib', 'dic', 'did', 'dif',
                        'dig', 'dih', 'dii', 'dij', 'dik', 'dil', 'dim', 'din', 'dio', 'dip', 'diq', 'dir', 'dis',
                        'dit', 'diu', 'div', 'diw', 'dix', 'diy', 'diz', 'djb', 'djc', 'djd', 'dje', 'djf', 'dji',
                        'djj', 'djk', 'djm', 'djn', 'djo', 'djr', 'dju', 'djw', 'dka', 'dkk', 'dkr', 'dks', 'dkx',
                        'dlg', 'dlk', 'dlm', 'dln', 'dma', 'dmb', 'dmc', 'dmd', 'dme', 'dmg', 'dmk', 'dml', 'dmm',
                        'dmo', 'dmr', 'dms', 'dmu', 'dmv', 'dmx', 'dmy', 'dna', 'dnd', 'dne', 'dng', 'dni', 'dnj',
                        'dnk', 'dnn', 'dnr', 'dnt', 'dnu', 'dnv', 'dnw', 'dny', 'doa', 'dob', 'doc', 'doe', 'dof',
                        'doh', 'doi', 'dok', 'dol', 'don', 'doo', 'dop', 'doq', 'dor', 'dos', 'dot', 'dov', 'dow',
                        'dox', 'doy', 'doz', 'dpp', 'drb', 'drc', 'drd', 'dre', 'drg', 'dri', 'drl', 'drn', 'dro',
                        'drq', 'drs', 'drt', 'dru', 'dry', 'dsb', 'dse', 'dsh', 'dsi', 'dsl', 'dsn', 'dsq', 'dta',
                        'dtb', 'dtd', 'dth', 'dti', 'dtk', 'dtm', 'dto', 'dtp', 'dtr', 'dts', 'dtt', 'dtu', 'dty',
                        'dua', 'dub', 'duc', 'dud', 'due', 'duf', 'dug', 'duh', 'dui', 'duj', 'duk', 'dul', 'dum',
                        'dun', 'duo', 'dup', 'duq', 'dur', 'dus', 'duu', 'duv', 'duw', 'dux', 'duy', 'duz', 'dva',
                        'dwa', 'dwr', 'dww', 'dya', 'dyb', 'dyd', 'dyi', 'dym', 'dyn', 'dyo', 'dyu', 'dyy', 'dza',
                        'dze', 'dzg', 'dzl', 'dzn', 'dzo', 'ebg', 'ebo', 'ebr', 'ebu', 'ecs', 'eee', 'efa', 'efe',
                        'efi', 'ega', 'egl', 'ego', 'egy', 'ehu', 'eip', 'eit', 'eiv', 'eja', 'eka', 'eke', 'ekg',
                        'eki', 'ekk', 'ekl', 'ekm', 'eko', 'ekp', 'ekr', 'eky', 'ele', 'elh', 'eli', 'elk', 'ell',
                        'elm', 'elo', 'elu', 'elx', 'ema', 'emb', 'eme', 'emg', 'emi', 'emk', 'emn', 'emp', 'ems',
                        'emu', 'emw', 'emy', 'ena', 'enb', 'enc', 'end', 'enf', 'eng', 'enh', 'enl', 'enm', 'enn',
                        'eno', 'enq', 'enr', 'enu', 'env', 'enw', 'enx', 'eot', 'epi', 'era', 'erg', 'erh', 'eri',
                        'erk', 'ero', 'err', 'ers', 'ert', 'erw', 'ese', 'esh', 'esi', 'esk', 'esl', 'esn', 'eso',
                        'esq', 'ess', 'esu', 'etb', 'eth', 'etn', 'eto', 'etr', 'ets', 'ett', 'etu', 'etx', 'etz',
                        'eus', 'eve', 'evh', 'evn', 'ewe', 'ewo', 'ext', 'eya', 'eyo', 'eza', 'eze', 'faa', 'fab',
                        'fad', 'faf', 'fag', 'fah', 'fai', 'faj', 'fak', 'fal', 'fam', 'fan', 'fao', 'fap', 'far',
                        'fat', 'fau', 'fax', 'fay', 'fcs', 'fer', 'ffm', 'fgr', 'fia', 'fie', 'fij', 'fil', 'fin',
                        'fip', 'fir', 'fit', 'fiw', 'fkk', 'fkv', 'fla', 'flh', 'fli', 'fll', 'fln', 'flr', 'fmp',
                        'fmu', 'fni', 'fod', 'foi', 'fon', 'for', 'fos', 'fpe', 'fqs', 'fra', 'frc', 'frd', 'frk',
                        'frm', 'fro', 'frp', 'frq', 'frr', 'frs', 'frt', 'fry', 'fse', 'fsl', 'fss', 'fub', 'fuc',
                        'fud', 'fue', 'fuf', 'fuh', 'fui', 'fuj', 'ful', 'fun', 'fuq', 'fur', 'fut', 'fuu', 'fuv',
                        'fuy', 'fvr', 'fwa', 'fwe', 'gaa', 'gab', 'gac', 'gad', 'gae', 'gaf', 'gag', 'gah', 'gai',
                        'gaj', 'gak', 'gal', 'gam', 'gan', 'gao', 'gap', 'gaq', 'gar', 'gas', 'gat', 'gau', 'gaw',
                        'gax', 'gay', 'gaz', 'gbb', 'gbd', 'gbe', 'gbf', 'gbg', 'gbh', 'gbi', 'gbj', 'gbk', 'gbl',
                        'gbm', 'gbn', 'gbo', 'gbp', 'gbq', 'gbr', 'gbs', 'gbu', 'gbv', 'gbw', 'gbx', 'gby', 'gbz',
                        'gcc', 'gcd', 'gce', 'gcf', 'gcl', 'gcn', 'gcr', 'gct', 'gda', 'gdb', 'gdc', 'gdd', 'gde',
                        'gdf', 'gdg', 'gdh', 'gdi', 'gdj', 'gdk', 'gdl', 'gdm', 'gdn', 'gdo', 'gdq', 'gdr', 'gds',
                        'gdu', 'gdx', 'gea', 'geb', 'gec', 'ged', 'geh', 'gei', 'gej', 'gek', 'gel', 'geq', 'ges',
                        'gev', 'gew', 'gex', 'gey', 'gez', 'gfk', 'gft', 'gga', 'ggb', 'ggd', 'gge', 'ggg', 'ggk',
                        'ggl', 'ggn', 'ggo', 'ggt', 'ggu', 'ggw', 'gha', 'ghc', 'ghe', 'ghh', 'ghk', 'ghl', 'ghn',
                        'gho', 'ghr', 'ghs', 'ght', 'gia', 'gid', 'gig', 'gih', 'gil', 'gim', 'gin', 'gip', 'giq',
                        'gir', 'gis', 'git', 'giu', 'giw', 'gix', 'giz', 'gjk', 'gjm', 'gjn', 'gju', 'gka', 'gke',
                        'gkn', 'gko', 'gkp', 'gku', 'gla', 'glc', 'gld', 'gle', 'glg', 'glh', 'glj', 'glk', 'gll',
                        'glo', 'glr', 'glu', 'glv', 'glw', 'gly', 'gma', 'gmb', 'gmd', 'gmg', 'gmh', 'gml', 'gmm',
                        'gmn', 'gmu', 'gmv', 'gmx', 'gmy', 'gmz', 'gna', 'gnb', 'gnc', 'gnd', 'gne', 'gng', 'gnh',
                        'gni', 'gnk', 'gnl', 'gnm', 'gnn', 'gno', 'gnq', 'gnr', 'gnt', 'gnu', 'gnw', 'gnz', 'goa',
                        'gob', 'goc', 'god', 'goe', 'gof', 'gog', 'goh', 'goi', 'gol', 'gom', 'gon', 'goo', 'gop',
                        'goq', 'gor', 'gos', 'got', 'gou', 'gow', 'gox', 'goy', 'goz', 'gpa', 'gpe', 'gpn', 'gqa',
                        'gqi', 'gqn', 'gqr', 'gqu', 'gra', 'grb', 'grc', 'grd', 'grg', 'grh', 'gri', 'grj', 'grm',
                        'gro', 'grq', 'grr', 'grs', 'grt', 'gru', 'grv', 'grw', 'grx', 'gry', 'grz', 'gse', 'gsg',
                        'gsl', 'gsm', 'gsn', 'gso', 'gsp', 'gss', 'gsw', 'gta', 'gua', 'gub', 'guc', 'gud', 'gue',
                        'guf', 'gug', 'guh', 'gui', 'guj', 'guk', 'gul', 'gum', 'gun', 'guo', 'gup', 'guq', 'gur',
                        'gus', 'gut', 'guu', 'guw', 'gux', 'guz', 'gva', 'gvc', 'gve', 'gvf', 'gvj', 'gvl', 'gvm',
                        'gvn', 'gvo', 'gvp', 'gvr', 'gvs', 'gvy', 'gwa', 'gwb', 'gwc', 'gwd', 'gwe', 'gwf', 'gwg',
                        'gwi', 'gwj', 'gwm', 'gwn', 'gwr', 'gwt', 'gwu', 'gww', 'gwx', 'gxx', 'gya', 'gyb', 'gyd',
                        'gye', 'gyf', 'gyg', 'gyi', 'gyl', 'gym', 'gyn', 'gyr', 'gyy', 'gza', 'gzi', 'gzn', 'haa',
                        'hab', 'hac', 'had', 'hae', 'haf', 'hag', 'hah', 'hai', 'haj', 'hak', 'hal', 'ham', 'han',
                        'hao', 'hap', 'haq', 'har', 'has', 'hat', 'hau', 'hav', 'haw', 'hax', 'hay', 'haz', 'hba',
                        'hbb', 'hbn', 'hbo', 'hbs', 'hbu', 'hca', 'hch', 'hdn', 'hds', 'hdy', 'hea', 'heb', 'hed',
                        'heg', 'heh', 'hei', 'hem', 'her', 'hgm', 'hgw', 'hhi', 'hhr', 'hhy', 'hia', 'hib', 'hid',
                        'hif', 'hig', 'hih', 'hii', 'hij', 'hik', 'hil', 'hin', 'hio', 'hit', 'hiw', 'hix', 'hji',
                        'hka', 'hke', 'hkk', 'hks', 'hla', 'hlb', 'hld', 'hle', 'hlt', 'hlu', 'hma', 'hmb', 'hmc',
                        'hmd', 'hme', 'hmf', 'hmg', 'hmh', 'hmi', 'hmj', 'hml', 'hmm', 'hmp', 'hmq', 'hmr', 'hms',
                        'hmt', 'hmu', 'hmv', 'hmw', 'hmy', 'hmz', 'hna', 'hnd', 'hne', 'hnh', 'hni', 'hnj', 'hnn',
                        'hno', 'hns', 'hnu', 'hoa', 'hob', 'hoc', 'hod', 'hoe', 'hoh', 'hoi', 'hoj', 'hol', 'hom',
                        'hoo', 'hop', 'hor', 'hos', 'hot', 'hov', 'how', 'hoy', 'hoz', 'hpo', 'hps', 'hra', 'hrc',
                        'hre', 'hrk', 'hrm', 'hro', 'hrp', 'hrt', 'hru', 'hrv', 'hrw', 'hrx', 'hrz', 'hsb', 'hsh',
                        'hsl', 'hsn', 'hss', 'hti', 'hto', 'hts', 'htu', 'hub', 'huc', 'hud', 'hue', 'huf', 'hug',
                        'huh', 'hui', 'huj', 'huk', 'hul', 'hum', 'hun', 'huo', 'hup', 'huq', 'hur', 'hus', 'hut',
                        'huu', 'huv', 'huw', 'hux', 'huy', 'huz', 'hve', 'hvk', 'hvn', 'hvv', 'hwa', 'hwc', 'hwo',
                        'hya', 'hye', 'iai', 'ian', 'iar', 'iba', 'ibb', 'ibd', 'ibe', 'ibg', 'ibl', 'ibm', 'ibn',
                        'ibo', 'ibr', 'ibu', 'iby', 'ica', 'ich', 'icl', 'icr', 'ida', 'idb', 'idc', 'idd', 'ide',
                        'idi', 'idr', 'idt', 'idu', 'ifa', 'ifb', 'ife', 'iff', 'ifk', 'ifm', 'ifu', 'ify', 'igb',
                        'ige', 'igg', 'igl', 'igm', 'ign', 'igo', 'igw', 'ihp', 'ihw', 'iii', 'iin', 'ijc', 'ije',
                        'ijj', 'ijn', 'ijs', 'ike', 'iki', 'ikk', 'ikl', 'iko', 'ikp', 'ikr', 'iks', 'ikt', 'ikv',
                        'ikw', 'ikx', 'ikz', 'ila', 'ilb', 'ilg', 'ili', 'ilk', 'ill', 'ilo', 'ils', 'ilu', 'ilv',
                        'ima', 'imi', 'iml', 'imn', 'imo', 'imr', 'imy', 'inb', 'ind', 'ing', 'inh', 'inj', 'inl',
                        'inm', 'inn', 'ino', 'inp', 'ins', 'int', 'inz', 'ior', 'iou', 'iow', 'ipi', 'ipk', 'ipo',
                        'iqu', 'iqw', 'ire', 'irh', 'iri', 'irk', 'irn', 'iru', 'irx', 'iry', 'isa', 'isc', 'isd',
                        'ise', 'isg', 'ish', 'isi', 'isk', 'isl', 'ism', 'isn', 'iso', 'isr', 'ist', 'isu', 'ita',
                        'itb', 'ite', 'iti', 'itk', 'itl', 'itm', 'ito', 'itr', 'its', 'itt', 'itv', 'itw', 'itx',
                        'ity', 'itz', 'ium', 'ivb', 'ivv', 'iwk', 'iwm', 'iwo', 'iws', 'ixc', 'ixl', 'iya', 'iyo',
                        'iyx', 'izh', 'izr', 'izz', 'jaa', 'jab', 'jac', 'jad', 'jae', 'jaf', 'jah', 'jaj', 'jak',
                        'jal', 'jam', 'jan', 'jao', 'jaq', 'jas', 'jat', 'jau', 'jav', 'jax', 'jay', 'jaz', 'jbi',
                        'jbj', 'jbk', 'jbn', 'jbr', 'jbt', 'jbu', 'jbw', 'jcs', 'jct', 'jda', 'jdg', 'jdt', 'jeb',
                        'jee', 'jeg', 'jeh', 'jei', 'jek', 'jel', 'jen', 'jer', 'jet', 'jeu', 'jgb', 'jge', 'jgo',
                        'jhi', 'jhs', 'jia', 'jib', 'jic', 'jid', 'jie', 'jig', 'jih', 'jii', 'jil', 'jim', 'jio',
                        'jiq', 'jit', 'jiu', 'jiv', 'jiy', 'jje', 'jkm', 'jko', 'jkr', 'jku', 'jle', 'jls', 'jma',
                        'jmb', 'jmc', 'jmd', 'jmi', 'jml', 'jmn', 'jmr', 'jms', 'jmw', 'jmx', 'jna', 'jnd', 'jng',
                        'jni', 'jnj', 'jnl', 'jns', 'job', 'jod', 'jor', 'jos', 'jow', 'jpa', 'jpn', 'jpr', 'jqr',
                        'jra', 'jrr', 'jru', 'jsl', 'jua', 'jub', 'juc', 'jud', 'juh', 'jui', 'juk', 'jul', 'jum',
                        'jun', 'juo', 'jup', 'jur', 'jus', 'jut', 'juu', 'juw', 'juy', 'jvd', 'jvn', 'jwi', 'jya',
                        'jye', 'jyy', 'kaa', 'kab', 'kac', 'kad', 'kae', 'kaf', 'kag', 'kah', 'kai', 'kaj', 'kak',
                        'kal', 'kam', 'kan', 'kao', 'kap', 'kaq', 'kas', 'kat', 'kaw', 'kax', 'kay', 'kaz', 'kba',
                        'kbb', 'kbc', 'kbd', 'kbe', 'kbg', 'kbh', 'kbi', 'kbj', 'kbk', 'kbl', 'kbm', 'kbn', 'kbo',
                        'kbp', 'kbq', 'kbr', 'kbs', 'kbt', 'kbu', 'kbv', 'kbw', 'kbx', 'kby', 'kbz', 'kca', 'kcb',
                        'kcc', 'kcd', 'kcf', 'kcg', 'kci', 'kcj', 'kck', 'kcl', 'kcm', 'kcn', 'kco', 'kcp', 'kcq',
                        'kcr', 'kcs', 'kct', 'kcu', 'kcv', 'kcw', 'kcx', 'kcy', 'kcz', 'kda', 'kdc', 'kdd', 'kde',
                        'kdf', 'kdg', 'kdh', 'kdi', 'kdj', 'kdk', 'kdl', 'kdm', 'kdn', 'kdp', 'kdq', 'kdr', 'kdt',
                        'kdu', 'kdw', 'kdx', 'kdy', 'kdz', 'kea', 'keb', 'kec', 'ked', 'kee', 'kef', 'keg', 'keh',
                        'kei', 'kej', 'kek', 'kel', 'kem', 'ken', 'keo', 'kep', 'keq', 'ker', 'kes', 'ket', 'keu',
                        'kev', 'kew', 'key', 'kez', 'kfa', 'kfb', 'kfc', 'kfd', 'kfe', 'kff', 'kfg', 'kfh', 'kfk',
                        'kfl', 'kfm', 'kfn', 'kfo', 'kfp', 'kfq', 'kfr', 'kfs', 'kft', 'kfu', 'kfv', 'kfw', 'kfx',
                        'kfy', 'kfz', 'kga', 'kgb', 'kgd', 'kge', 'kgf', 'kgg', 'kgi', 'kgj', 'kgk', 'kgl', 'kgn',
                        'kgo', 'kgp', 'kgq', 'kgr', 'kgs', 'kgt', 'kgu', 'kgv', 'kgw', 'kgx', 'kgy', 'kha', 'khb',
                        'khc', 'khd', 'khe', 'khf', 'khg', 'khh', 'khj', 'khk', 'khl', 'khm', 'khn', 'kho', 'khp',
                        'khq', 'khr', 'khs', 'kht', 'khu', 'khv', 'khw', 'khx', 'khy', 'khz', 'kia', 'kib', 'kic',
                        'kid', 'kie', 'kif', 'kig', 'kih', 'kii', 'kij', 'kik', 'kil', 'kim', 'kin', 'kio', 'kip',
                        'kiq', 'kir', 'kis', 'kit', 'kiu', 'kiv', 'kiw', 'kix', 'kiy', 'kiz', 'kja', 'kjb', 'kjc',
                        'kjd', 'kje', 'kjg', 'kjh', 'kji', 'kjj', 'kjk', 'kjl', 'kjm', 'kjn', 'kjo', 'kjp', 'kjq',
                        'kjr', 'kjs', 'kjt', 'kju', 'kjv', 'kjx', 'kjy', 'kjz', 'kka', 'kkb', 'kkc', 'kkd', 'kke',
                        'kkf', 'kkg', 'kkh', 'kki', 'kkj', 'kkk', 'kkl', 'kkm', 'kko', 'kkp', 'kkq', 'kkr', 'kks',
                        'kkt', 'kkv', 'kkw', 'kkx', 'kky', 'kkz', 'kla', 'klb', 'klc', 'kld', 'kle', 'klf', 'klg',
                        'klh', 'kli', 'klj', 'klk', 'kll', 'klm', 'kln', 'klo', 'klp', 'klq', 'klr', 'kls', 'klt',
                        'klu', 'klv', 'klw', 'klx', 'kly', 'klz', 'kma', 'kmb', 'kmc', 'kmd', 'kme', 'kmf', 'kmg',
                        'kmh', 'kmi', 'kmj', 'kmk', 'kml', 'kmm', 'kmn', 'kmo', 'kmp', 'kmq', 'kmr', 'kms', 'kmt',
                        'kmu', 'kmv', 'kmw', 'kmx', 'kmy', 'kmz', 'kna', 'knb', 'knc', 'knd', 'kne', 'knf', 'kng',
                        'kni', 'knj', 'knk', 'knl', 'knm', 'knn', 'kno', 'knp', 'knq', 'knr', 'kns', 'knt', 'knu',
                        'knv', 'knw', 'knx', 'kny', 'knz', 'koa', 'koc', 'kod', 'koe', 'kof', 'kog', 'koh', 'koi',
                        'kol', 'kom', 'koo', 'kop', 'koq', 'kor', 'kos', 'kot', 'kou', 'kov', 'kow', 'koy', 'koz',
                        'kpa', 'kpb', 'kpc', 'kpd', 'kpf', 'kpg', 'kph', 'kpi', 'kpj', 'kpk', 'kpl', 'kpm', 'kpn',
                        'kpo', 'kpq', 'kpr', 'kps', 'kpt', 'kpu', 'kpv', 'kpw', 'kpx', 'kpy', 'kpz', 'kqa', 'kqb',
                        'kqc', 'kqd', 'kqe', 'kqf', 'kqg', 'kqi', 'kqj', 'kqk', 'kql', 'kqm', 'kqn', 'kqo', 'kqp',
                        'kqq', 'kqr', 'kqs', 'kqt', 'kqu', 'kqv', 'kqw', 'kqx', 'kqy', 'kqz', 'kra', 'krb', 'krc',
                        'krd', 'kre', 'krf', 'krh', 'kri', 'krj', 'krk', 'krl', 'krm', 'krn', 'krp', 'krr', 'krs',
                        'krt', 'kru', 'krv', 'krw', 'krx', 'kry', 'krz', 'ksb', 'ksc', 'ksd', 'kse', 'ksf', 'ksg',
                        'ksh', 'ksi', 'ksj', 'ksk', 'ksl', 'ksm', 'ksn', 'ksp', 'ksq', 'ksr', 'kss', 'kst', 'ksu',
                        'ksv', 'ksw', 'ksx', 'ksy', 'ksz', 'kta', 'ktb', 'ktc', 'ktd', 'kte', 'ktf', 'ktg', 'kth',
                        'kti', 'ktj', 'ktk', 'ktl', 'ktm', 'ktn', 'kto', 'ktp', 'ktr', 'kts', 'ktt', 'ktu', 'ktv',
                        'ktw', 'ktx', 'kty', 'ktz', 'kua', 'kub', 'kuc', 'kud', 'kue', 'kuf', 'kug', 'kuh', 'kui',
                        'kuj', 'kuk', 'kul', 'kum', 'kun', 'kuo', 'kup', 'kuq', 'kur', 'kus', 'kut', 'kuu', 'kuv',
                        'kuw', 'kux', 'kuy', 'kuz', 'kva', 'kvb', 'kvc', 'kvd', 'kve', 'kvf', 'kvg', 'kvh', 'kvi',
                        'kvj', 'kvk', 'kvl', 'kvm', 'kvn', 'kvo', 'kvp', 'kvq', 'kvr', 'kvu', 'kvv', 'kvw', 'kvx',
                        'kvy', 'kvz', 'kwa', 'kwb', 'kwc', 'kwd', 'kwe', 'kwf', 'kwg', 'kwh', 'kwi', 'kwj', 'kwk',
                        'kwl', 'kwm', 'kwn', 'kwo', 'kwp', 'kwr', 'kws', 'kwt', 'kwu', 'kwv', 'kww', 'kwx', 'kwy',
                        'kwz', 'kxa', 'kxb', 'kxc', 'kxd', 'kxf', 'kxh', 'kxi', 'kxj', 'kxk', 'kxl', 'kxm', 'kxn',
                        'kxo', 'kxp', 'kxq', 'kxr', 'kxs', 'kxt', 'kxu', 'kxv', 'kxw', 'kxx', 'kxy', 'kxz', 'kya',
                        'kyb', 'kyc', 'kyd', 'kye', 'kyf', 'kyg', 'kyh', 'kyi', 'kyj', 'kyk', 'kyl', 'kyn', 'kyo',
                        'kyq', 'kyr', 'kys', 'kyt', 'kyu', 'kyw', 'kyx', 'kyy', 'kyz', 'kza', 'kzb', 'kzc', 'kzd',
                        'kzf', 'kzg', 'kzi', 'kzj', 'kzk', 'kzl', 'kzm', 'kzn', 'kzo', 'kzp', 'kzq', 'kzr', 'kzs',
                        'kzt', 'kzu', 'kzv', 'kzx', 'kzy', 'kzz', 'laa', 'lac', 'lad', 'lae', 'laf', 'lag', 'lai',
                        'laj', 'lak', 'lal', 'lam', 'lan', 'lao', 'lap', 'laq', 'lar', 'las', 'lat', 'lav', 'law',
                        'lax', 'laz', 'lbb', 'lbc', 'lbe', 'lbf', 'lbj', 'lbk', 'lbl', 'lbm', 'lbn', 'lbo', 'lbq',
                        'lbr', 'lbs', 'lbt', 'lbu', 'lbv', 'lbw', 'lbx', 'lby', 'lbz', 'lcc', 'lcd', 'lce', 'lcf',
                        'lch', 'lcl', 'lcm', 'lcp', 'lcq', 'lcs', 'lda', 'ldb', 'ldd', 'ldg', 'ldh', 'ldi', 'ldj',
                        'ldk', 'ldl', 'ldm', 'ldo', 'ldp', 'ldq', 'lea', 'leb', 'lec', 'led', 'lee', 'lef', 'leh',
                        'lei', 'lej', 'lek', 'lel', 'lem', 'leo', 'lep', 'leq', 'ler', 'les', 'let', 'leu', 'lev',
                        'lew', 'lex', 'ley', 'lez', 'lfa', 'lga', 'lgb', 'lgg', 'lgh', 'lgi', 'lgk', 'lgl', 'lgm',
                        'lgn', 'lgq', 'lgr', 'lgt', 'lgu', 'lgz', 'lha', 'lhh', 'lhi', 'lhl', 'lhm', 'lhn', 'lhp',
                        'lhs', 'lht', 'lhu', 'lia', 'lib', 'lic', 'lid', 'lie', 'lif', 'lig', 'lih', 'lij', 'lik',
                        'lil', 'lim', 'lin', 'lio', 'lip', 'liq', 'lis', 'lit', 'liu', 'liv', 'liw', 'lix', 'liy',
                        'liz', 'lja', 'lje', 'lji', 'ljl', 'ljp', 'ljw', 'ljx', 'lka', 'lkb', 'lkc', 'lkd', 'lke',
                        'lkh', 'lki', 'lkj', 'lkl', 'lkm', 'lkn', 'lko', 'lkr', 'lks', 'lkt', 'lku', 'lky', 'lla',
                        'llb', 'llc', 'lld', 'lle', 'llf', 'llg', 'llh', 'lli', 'llj', 'llk', 'lll', 'llm', 'lln',
                        'llp', 'llq', 'lls', 'llu', 'llx', 'lma', 'lmb', 'lmc', 'lmd', 'lme', 'lmf', 'lmg', 'lmi',
                        'lmj', 'lmk', 'lml', 'lmn', 'lmo', 'lmp', 'lmq', 'lmr', 'lmu', 'lmv', 'lmw', 'lmx', 'lmy',
                        'lna', 'lnb', 'lnd', 'lnh', 'lni', 'lnj', 'lnl', 'lnm', 'lnn', 'lno', 'lns', 'lnu', 'lnz',
                        'loa', 'lob', 'loc', 'loe', 'lof', 'log', 'loh', 'loi', 'loj', 'lok', 'lol', 'lom', 'lon',
                        'loo', 'lop', 'loq', 'lor', 'los', 'lot', 'lou', 'low', 'lox', 'loy', 'loz', 'lpa', 'lpe',
                        'lpn', 'lpo', 'lpx', 'lra', 'lrc', 'lre', 'lrg', 'lri', 'lrl', 'lrm', 'lrn', 'lro', 'lrr',
                        'lrt', 'lrv', 'lrz', 'lsa', 'lsd', 'lse', 'lsh', 'lsi', 'lsl', 'lsm', 'lsp', 'lsr', 'lss',
                        'lst', 'lsy', 'ltc', 'ltg', 'lti', 'ltn', 'lto', 'lts', 'ltu', 'ltz', 'lua', 'lub', 'luc',
                        'lud', 'lue', 'luf', 'lug', 'lui', 'luj', 'luk', 'lul', 'lum', 'lun', 'luo', 'lup', 'luq',
                        'lur', 'lus', 'lut', 'luv', 'luw', 'luz', 'lva', 'lvk', 'lvs', 'lvu', 'lwa', 'lwe', 'lwg',
                        'lwh', 'lwl', 'lwm', 'lwo', 'lwt', 'lwu', 'lww', 'lya', 'lyg', 'lyn', 'lzh', 'lzl', 'lzn',
                        'lzz', 'maa', 'mab', 'mad', 'mae', 'maf', 'mag', 'mah', 'mai', 'maj', 'mak', 'mal', 'mam',
                        'maq', 'mar', 'mas', 'mat', 'mau', 'mav', 'maw', 'max', 'maz', 'mba', 'mbb', 'mbc', 'mbd',
                        'mbe', 'mbf', 'mbh', 'mbi', 'mbj', 'mbk', 'mbl', 'mbm', 'mbn', 'mbo', 'mbp', 'mbq', 'mbr',
                        'mbs', 'mbt', 'mbu', 'mbv', 'mbw', 'mbx', 'mby', 'mbz', 'mca', 'mcb', 'mcc', 'mcd', 'mce',
                        'mcf', 'mcg', 'mch', 'mci', 'mcj', 'mck', 'mcl', 'mcm', 'mcn', 'mco', 'mcp', 'mcq', 'mcr',
                        'mcs', 'mct', 'mcu', 'mcv', 'mcw', 'mcx', 'mcy', 'mcz', 'mda', 'mdb', 'mdc', 'mdd', 'mde',
                        'mdf', 'mdg', 'mdh', 'mdi', 'mdj', 'mdk', 'mdl', 'mdm', 'mdn', 'mdp', 'mdq', 'mdr', 'mds',
                        'mdt', 'mdu', 'mdv', 'mdw', 'mdx', 'mdy', 'mdz', 'mea', 'meb', 'mec', 'med', 'mee', 'mef',
                        'meh', 'mei', 'mej', 'mek', 'mel', 'mem', 'men', 'meo', 'mep', 'meq', 'mer', 'mes', 'met',
                        'meu', 'mev', 'mew', 'mey', 'mez', 'mfa', 'mfb', 'mfc', 'mfd', 'mfe', 'mff', 'mfg', 'mfh',
                        'mfi', 'mfj', 'mfk', 'mfl', 'mfm', 'mfn', 'mfo', 'mfp', 'mfq', 'mfr', 'mfs', 'mft', 'mfu',
                        'mfv', 'mfw', 'mfx', 'mfy', 'mfz', 'mga', 'mgb', 'mgc', 'mgd', 'mge', 'mgf', 'mgg', 'mgh',
                        'mgi', 'mgj', 'mgk', 'mgl', 'mgm', 'mgn', 'mgo', 'mgp', 'mgq', 'mgr', 'mgs', 'mgt', 'mgu',
                        'mgv', 'mgw', 'mgy', 'mgz', 'mha', 'mhb', 'mhc', 'mhd', 'mhe', 'mhf', 'mhg', 'mhi', 'mhj',
                        'mhk', 'mhl', 'mhm', 'mhn', 'mho', 'mhp', 'mhq', 'mhr', 'mhs', 'mht', 'mhu', 'mhw', 'mhx',
                        'mhy', 'mhz', 'mia', 'mib', 'mic', 'mid', 'mie', 'mif', 'mig', 'mih', 'mii', 'mij', 'mik',
                        'mil', 'mim', 'min', 'mio', 'mip', 'miq', 'mir', 'mit', 'miu', 'miw', 'mix', 'miy', 'miz',
                        'mjc', 'mjd', 'mje', 'mjg', 'mjh', 'mji', 'mjj', 'mjk', 'mjl', 'mjm', 'mjn', 'mjo', 'mjp',
                        'mjq', 'mjr', 'mjs', 'mjt', 'mju', 'mjv', 'mjw', 'mjx', 'mjy', 'mjz', 'mka', 'mkb', 'mkc',
                        'mkd', 'mke', 'mkf', 'mkg', 'mki', 'mkj', 'mkk', 'mkl', 'mkm', 'mkn', 'mko', 'mkp', 'mkq',
                        'mkr', 'mks', 'mkt', 'mku', 'mkv', 'mkw', 'mkx', 'mky', 'mkz', 'mla', 'mlb', 'mlc', 'mle',
                        'mlf', 'mlh', 'mli', 'mlj', 'mlk', 'mll', 'mlm', 'mln', 'mlo', 'mlp', 'mlq', 'mlr', 'mls',
                        'mlt', 'mlu', 'mlv', 'mlw', 'mlx', 'mlz', 'mma', 'mmb', 'mmc', 'mmd', 'mme', 'mmf', 'mmg',
                        'mmh', 'mmi', 'mmj', 'mmk', 'mml', 'mmm', 'mmn', 'mmo', 'mmp', 'mmq', 'mmr', 'mmt', 'mmu',
                        'mmv', 'mmw', 'mmx', 'mmy', 'mmz', 'mna', 'mnb', 'mnc', 'mnd', 'mne', 'mnf', 'mng', 'mnh',
                        'mni', 'mnj', 'mnk', 'mnl', 'mnm', 'mnn', 'mnp', 'mnq', 'mnr', 'mns', 'mnu', 'mnv', 'mnw',
                        'mnx', 'mny', 'mnz', 'moa', 'moc', 'moe', 'mog', 'moh', 'moi', 'moj', 'mok', 'mom', 'mon',
                        'moo', 'mop', 'moq', 'mor', 'mos', 'mot', 'mou', 'mov', 'mow', 'mox', 'moy', 'moz', 'mpa',
                        'mpb', 'mpc', 'mpd', 'mpe', 'mpg', 'mph', 'mpi', 'mpj', 'mpk', 'mpl', 'mpm', 'mpn', 'mpo',
                        'mpp', 'mpq', 'mpr', 'mps', 'mpt', 'mpu', 'mpv', 'mpw', 'mpx', 'mpy', 'mpz', 'mqa', 'mqb',
                        'mqc', 'mqe', 'mqf', 'mqg', 'mqh', 'mqi', 'mqj', 'mqk', 'mql', 'mqm', 'mqn', 'mqo', 'mqp',
                        'mqq', 'mqr', 'mqs', 'mqt', 'mqu', 'mqv', 'mqw', 'mqx', 'mqy', 'mqz', 'mra', 'mrb', 'mrc',
                        'mrd', 'mre', 'mrf', 'mrg', 'mrh', 'mri', 'mrj', 'mrk', 'mrl', 'mrm', 'mrn', 'mro', 'mrp',
                        'mrq', 'mrr', 'mrs', 'mrt', 'mru', 'mrv', 'mrw', 'mrx', 'mry', 'mrz', 'msb', 'msc', 'msd',
                        'mse', 'msf', 'msg', 'msh', 'msi', 'msj', 'msk', 'msl', 'msm', 'msn', 'mso', 'msp', 'msq',
                        'msr', 'mss', 'msu', 'msv', 'msw', 'msx', 'msy', 'msz', 'mta', 'mtb', 'mtc', 'mtd', 'mte',
                        'mtf', 'mtg', 'mth', 'mti', 'mtj', 'mtk', 'mtl', 'mtm', 'mtn', 'mto', 'mtp', 'mtq', 'mtr',
                        'mts', 'mtt', 'mtu', 'mtv', 'mtw', 'mtx', 'mty', 'mua', 'mub', 'muc', 'mud', 'mug', 'muh',
                        'mui', 'muj', 'muk', 'mum', 'muo', 'mup', 'muq', 'mur', 'mus', 'mut', 'muu', 'muv', 'mux',
                        'muy', 'muz', 'mva', 'mvb', 'mvd', 'mve', 'mvf', 'mvg', 'mvh', 'mvi', 'mvk', 'mvl', 'mvn',
                        'mvo', 'mvp', 'mvq', 'mvr', 'mvs', 'mvt', 'mvu', 'mvv', 'mvw', 'mvx', 'mvy', 'mvz', 'mwa',
                        'mwb', 'mwc', 'mwe', 'mwf', 'mwg', 'mwh', 'mwi', 'mwk', 'mwl', 'mwm', 'mwn', 'mwo', 'mwp',
                        'mwq', 'mws', 'mwt', 'mwu', 'mwv', 'mww', 'mwy', 'mwz', 'mxa', 'mxb', 'mxc', 'mxd', 'mxe',
                        'mxf', 'mxg', 'mxh', 'mxi', 'mxj', 'mxk', 'mxl', 'mxm', 'mxn', 'mxo', 'mxp', 'mxq', 'mxr',
                        'mxs', 'mxt', 'mxu', 'mxv', 'mxw', 'mxx', 'mxy', 'mxz', 'mya', 'myb', 'myc', 'mye', 'myf',
                        'myg', 'myh', 'myj', 'myk', 'myl', 'mym', 'myo', 'myp', 'myr', 'mys', 'myu', 'myv', 'myw',
                        'myx', 'myy', 'myz', 'mza', 'mzb', 'mzc', 'mzd', 'mze', 'mzg', 'mzh', 'mzi', 'mzj', 'mzk',
                        'mzl', 'mzm', 'mzn', 'mzo', 'mzp', 'mzq', 'mzr', 'mzs', 'mzt', 'mzu', 'mzv', 'mzw', 'mzy',
                        'mzz', 'naa', 'nab', 'nac', 'nae', 'naf', 'nag', 'naj', 'nak', 'nal', 'nam', 'nan', 'nao',
                        'nap', 'naq', 'nar', 'nas', 'nat', 'nau', 'nav', 'naw', 'nax', 'nay', 'naz', 'nba', 'nbb',
                        'nbc', 'nbd', 'nbe', 'nbh', 'nbi', 'nbj', 'nbk', 'nbl', 'nbm', 'nbn', 'nbo', 'nbp', 'nbq',
                        'nbr', 'nbs', 'nbt', 'nbu', 'nbv', 'nbw', 'nby', 'nca', 'ncb', 'ncc', 'ncd', 'nce', 'ncf',
                        'ncg', 'nch', 'nci', 'ncj', 'nck', 'ncl', 'ncm', 'ncn', 'nco', 'ncp', 'ncr', 'ncs', 'nct',
                        'ncu', 'ncx', 'ncz', 'nda', 'ndb', 'ndc', 'ndd', 'nde', 'ndg', 'ndh', 'ndi', 'ndj', 'ndk',
                        'ndl', 'ndm', 'ndn', 'ndo', 'ndp', 'ndq', 'ndr', 'nds', 'ndt', 'ndu', 'ndv', 'ndw', 'ndx',
                        'ndy', 'ndz', 'nea', 'neb', 'nec', 'nee', 'neg', 'neh', 'nej', 'nek', 'nem', 'nen', 'nep',
                        'neq', 'ner', 'nes', 'net', 'nev', 'new', 'nex', 'ney', 'nez', 'nfa', 'nfd', 'nfl', 'nfr',
                        'nfu', 'nga', 'ngb', 'ngc', 'ngd', 'nge', 'ngg', 'ngh', 'ngi', 'ngj', 'ngk', 'ngl', 'ngn',
                        'ngp', 'ngq', 'ngr', 'ngs', 'ngt', 'ngu', 'ngv', 'ngw', 'ngx', 'ngy', 'ngz', 'nha', 'nhb',
                        'nhc', 'nhd', 'nhe', 'nhf', 'nhg', 'nhh', 'nhi', 'nhk', 'nhm', 'nhn', 'nho', 'nhp', 'nhq',
                        'nhr', 'nht', 'nhu', 'nhv', 'nhw', 'nhx', 'nhy', 'nhz', 'nia', 'nib', 'nid', 'nie', 'nif',
                        'nig', 'nih', 'nii', 'nij', 'nik', 'nil', 'nim', 'nin', 'nio', 'niq', 'nir', 'nis', 'nit',
                        'niu', 'niv', 'niw', 'nix', 'niy', 'niz', 'nja', 'njb', 'njh', 'nji', 'njj', 'njl', 'njm',
                        'njn', 'njo', 'njr', 'njs', 'nju', 'njx', 'njy', 'njz', 'nka', 'nkb', 'nkc', 'nkd', 'nke',
                        'nkg', 'nkh', 'nki', 'nkj', 'nkk', 'nkm', 'nkn', 'nko', 'nkp', 'nkq', 'nkr', 'nks', 'nkt',
                        'nku', 'nkv', 'nkw', 'nkx', 'nkz', 'nla', 'nlc', 'nld', 'nle', 'nlg', 'nli', 'nlj', 'nlk',
                        'nll', 'nlo', 'nlu', 'nlv', 'nlx', 'nly', 'nlz', 'nma', 'nmb', 'nmc', 'nmd', 'nme', 'nmf',
                        'nmg', 'nmh', 'nmi', 'nmk', 'nml', 'nmm', 'nmn', 'nmo', 'nmp', 'nmq', 'nmr', 'nms', 'nmt',
                        'nmu', 'nmv', 'nmw', 'nmx', 'nmy', 'nmz', 'nna', 'nnb', 'nnc', 'nnd', 'nne', 'nnf', 'nng',
                        'nnh', 'nni', 'nnj', 'nnk', 'nnl', 'nnm', 'nnn', 'nno', 'nnp', 'nnq', 'nnr', 'nns', 'nnt',
                        'nnu', 'nnv', 'nnw', 'nny', 'nnz', 'noa', 'nob', 'noc', 'nod', 'noe', 'nof', 'nog', 'noh',
                        'noi', 'noj', 'nok', 'nol', 'non', 'nop', 'noq', 'nor', 'nos', 'not', 'nou', 'now', 'noy',
                        'noz', 'npa', 'nph', 'npi', 'npl', 'npn', 'npo', 'nps', 'npy', 'nqg', 'nqk', 'nqm', 'nqn',
                        'nra', 'nrb', 'nrc', 'nre', 'nrf', 'nrg', 'nri', 'nrk', 'nrl', 'nrm', 'nrn', 'nrt', 'nru',
                        'nrz', 'nsa', 'nsd', 'nse', 'nsf', 'nsg', 'nsh', 'nsi', 'nsk', 'nsl', 'nsm', 'nsn', 'nso',
                        'nsp', 'nsq', 'nsr', 'nss', 'nst', 'nsu', 'nsw', 'nsx', 'nsy', 'nsz', 'nte', 'nti', 'ntj',
                        'ntk', 'ntm', 'nto', 'ntp', 'ntr', 'ntu', 'ntw', 'nty', 'ntz', 'nua', 'nuc', 'nud', 'nue',
                        'nuf', 'nug', 'nuh', 'nui', 'nuj', 'nuk', 'nul', 'num', 'nun', 'nuo', 'nup', 'nuq', 'nur',
                        'nus', 'nut', 'nuu', 'nuv', 'nuw', 'nux', 'nuy', 'nuz', 'nvh', 'nvm', 'nvo', 'nwa', 'nwb',
                        'nwe', 'nwi', 'nwm', 'nwo', 'nwr', 'nxa', 'nxd', 'nxe', 'nxg', 'nxi', 'nxl', 'nxn', 'nxo',
                        'nxq', 'nxr', 'nxx', 'nya', 'nyb', 'nyc', 'nyd', 'nye', 'nyf', 'nyg', 'nyh', 'nyi', 'nyj',
                        'nyk', 'nyl', 'nym', 'nyn', 'nyo', 'nyp', 'nyq', 'nyr', 'nys', 'nyt', 'nyu', 'nyv', 'nyw',
                        'nyx', 'nyy', 'nza', 'nzb', 'nzi', 'nzk', 'nzm', 'nzs', 'nzu', 'nzy', 'nzz', 'oaa', 'oac',
                        'oar', 'obi', 'obl', 'obm', 'obo', 'obr', 'obu', 'oca', 'och', 'oci', 'oco', 'ocu', 'odk',
                        'odt', 'odu', 'ofo', 'ofs', 'ofu', 'ogb', 'ogc', 'oge', 'ogg', 'ogo', 'ogu', 'oia', 'oin',
                        'ojb', 'ojc', 'ojg', 'ojp', 'ojs', 'ojv', 'ojw', 'oka', 'okb', 'okd', 'oke', 'okh', 'oki',
                        'okj', 'okk', 'okl', 'okn', 'okr', 'oks', 'oku', 'okv', 'okx', 'ola', 'old', 'ole', 'olm',
                        'olo', 'olr', 'olt', 'oma', 'omb', 'omc', 'omg', 'omi', 'omk', 'oml', 'omo', 'omr', 'omt',
                        'omu', 'omw', 'omx', 'ona', 'onb', 'one', 'ong', 'oni', 'onj', 'onk', 'onn', 'ono', 'onp',
                        'onr', 'ons', 'onu', 'onw', 'ood', 'oog', 'oon', 'oor', 'oos', 'opa', 'opk', 'opm', 'opo',
                        'opt', 'opy', 'ora', 'orc', 'ore', 'org', 'orh', 'orn', 'oro', 'orr', 'ors', 'ort', 'oru',
                        'orv', 'orw', 'orx', 'ory', 'orz', 'osa', 'osc', 'osi', 'oso', 'osp', 'oss', 'ost', 'osu',
                        'osx', 'ota', 'otd', 'ote', 'oti', 'otl', 'otm', 'otn', 'otq', 'otr', 'ots', 'ott', 'otu',
                        'otw', 'otx', 'oty', 'otz', 'oua', 'oub', 'oue', 'oui', 'oum', 'owi', 'owl', 'oyb', 'oyd',
                        'oym', 'oyy', 'ozm', 'pab', 'pac', 'pad', 'pae', 'paf', 'pag', 'pah', 'pai', 'pak', 'pal',
                        'pam', 'pan', 'pao', 'pap', 'paq', 'par', 'pas', 'pat', 'pau', 'pav', 'paw', 'pay', 'paz',
                        'pbb', 'pbc', 'pbe', 'pbf', 'pbg', 'pbh', 'pbi', 'pbl', 'pbn', 'pbo', 'pbp', 'pbr', 'pbs',
                        'pbt', 'pbu', 'pbv', 'pby', 'pca', 'pcb', 'pcc', 'pcd', 'pce', 'pcf', 'pcg', 'pci', 'pcj',
                        'pck', 'pcl', 'pcm', 'pcn', 'pcp', 'pcw', 'pda', 'pdc', 'pdi', 'pdn', 'pdo', 'pdt', 'pdu',
                        'pea', 'peb', 'ped', 'pee', 'pef', 'peg', 'peh', 'pei', 'pej', 'pek', 'pel', 'pem', 'peo',
                        'pep', 'peq', 'pes', 'pev', 'pex', 'pey', 'pez', 'pfa', 'pfe', 'pfl', 'pga', 'pgg', 'pgi',
                        'pgk', 'pgs', 'pgu', 'pha', 'phd', 'phg', 'phh', 'phk', 'phl', 'phm', 'phn', 'pho', 'phq',
                        'phr', 'pht', 'phu', 'phv', 'pia', 'pib', 'pic', 'pid', 'pie', 'pif', 'pih', 'pii', 'pil',
                        'pim', 'pin', 'pio', 'pip', 'pir', 'pis', 'pit', 'piu', 'piv', 'piw', 'pix', 'piy', 'piz',
                        'pjt', 'pkb', 'pkg', 'pkh', 'pkn', 'pko', 'pkp', 'pkr', 'pks', 'pkt', 'pku', 'pla', 'plb',
                        'plc', 'ple', 'plg', 'plh', 'pli', 'plj', 'plk', 'pll', 'pln', 'plo', 'plp', 'plq', 'plr',
                        'pls', 'plt', 'plu', 'plv', 'plw', 'ply', 'plz', 'pma', 'pmb', 'pmd', 'pme', 'pmf', 'pmh',
                        'pmi', 'pmj', 'pmm', 'pmn', 'pmo', 'pmq', 'pmr', 'pms', 'pmt', 'pmw', 'pmx', 'pmy', 'pmz',
                        'pna', 'pnb', 'pnc', 'pne', 'png', 'pnh', 'pni', 'pnj', 'pnk', 'pnl', 'pnm', 'pnn', 'pno',
                        'pnp', 'pnq', 'pnr', 'pns', 'pnt', 'pnu', 'pnv', 'pnw', 'pnx', 'pny', 'pnz', 'poc', 'poe',
                        'pof', 'pog', 'poh', 'poi', 'pol', 'pom', 'pon', 'poo', 'pop', 'poq', 'por', 'pos', 'pot',
                        'pov', 'pow', 'pox', 'poy', 'ppe', 'ppi', 'ppk', 'ppl', 'ppm', 'ppn', 'ppo', 'ppp', 'ppq',
                        'pps', 'ppt', 'ppu', 'pqa', 'pqm', 'prc', 'pre', 'prf', 'prg', 'prh', 'pri', 'prk', 'prl',
                        'prm', 'prn', 'pro', 'prq', 'prr', 'prs', 'prt', 'pru', 'prw', 'prx', 'prz', 'psa', 'psc',
                        'psd', 'pse', 'psg', 'psh', 'psi', 'psl', 'psm', 'psn', 'pso', 'psp', 'psq', 'psr', 'pss',
                        'pst', 'psu', 'psw', 'psy', 'pta', 'pth', 'pti', 'ptn', 'pto', 'ptp', 'ptr', 'ptt', 'ptu',
                        'ptv', 'ptw', 'pty', 'pua', 'pub', 'pud', 'pue', 'puf', 'pug', 'pui', 'puj', 'pum', 'puo',
                        'pup', 'puq', 'pur', 'pus', 'puu', 'puw', 'pux', 'puy', 'pwa', 'pwb', 'pwg', 'pwi', 'pwm',
                        'pwn', 'pwo', 'pwr', 'pww', 'pye', 'pym', 'pyn', 'pys', 'pyu', 'pyx', 'pyy', 'pzn', 'qua',
                        'qub', 'quc', 'qud', 'quf', 'qug', 'quh', 'qui', 'quk', 'qul', 'qum', 'qun', 'qup', 'qur',
                        'qus', 'quv', 'quw', 'qux', 'quy', 'quz', 'qva', 'qvc', 'qve', 'qvh', 'qvi', 'qvj', 'qvl',
                        'qvm', 'qvn', 'qvo', 'qvp', 'qvs', 'qvw', 'qvy', 'qvz', 'qwa', 'qwc', 'qwh', 'qws', 'qwt',
                        'qxa', 'qxc', 'qxh', 'qxl', 'qxn', 'qxo', 'qxp', 'qxq', 'qxr', 'qxs', 'qxt', 'qxu', 'qxw',
                        'qyp', 'raa', 'rab', 'rac', 'rad', 'raf', 'rag', 'rah', 'rai', 'rak', 'ral', 'ram', 'ran',
                        'rao', 'rap', 'raq', 'rar', 'ras', 'rat', 'rau', 'rav', 'raw', 'rax', 'ray', 'raz', 'rbb',
                        'rbp', 'rcf', 'rdb', 'rea', 'reb', 'ree', 'reg', 'rei', 'rej', 'rel', 'rem', 'ren', 'res',
                        'ret', 'rey', 'rga', 'rge', 'rgk', 'rgn', 'rgr', 'rgs', 'rgu', 'rhg', 'rhp', 'ria', 'rif',
                        'ril', 'rim', 'rin', 'rir', 'rit', 'riu', 'rjg', 'rji', 'rjs', 'rka', 'rkb', 'rkh', 'rki',
                        'rkm', 'rkt', 'rma', 'rmb', 'rmc', 'rme', 'rmf', 'rmh', 'rmi', 'rmk', 'rml', 'rmm', 'rmn',
                        'rmo', 'rmp', 'rmq', 'rms', 'rmt', 'rmw', 'rmx', 'rmy', 'rmz', 'rnd', 'rng', 'rnl', 'rnn',
                        'rnp', 'rnr', 'rnw', 'rob', 'roc', 'rod', 'roe', 'rof', 'rog', 'roh', 'rol', 'rom', 'ron',
                        'roo', 'rop', 'ror', 'rou', 'row', 'rpn', 'rpt', 'rri', 'rro', 'rsl', 'rth', 'rtm', 'rtw',
                        'rub', 'ruc', 'rue', 'ruf', 'rug', 'ruh', 'ruk', 'run', 'ruo', 'rup', 'ruq', 'rus', 'rut',
                        'ruu', 'rwa', 'rwk', 'rwm', 'rwo', 'rwr', 'rxd', 'rxw', 'ryn', 'rys', 'ryu', 'saa', 'sab',
                        'sac', 'sad', 'sae', 'saf', 'sag', 'sah', 'saj', 'sak', 'sam', 'san', 'sao', 'saq', 'sar',
                        'sas', 'sat', 'sau', 'sav', 'saw', 'sax', 'say', 'saz', 'sba', 'sbb', 'sbc', 'sbd', 'sbe',
                        'sbf', 'sbg', 'sbh', 'sbi', 'sbj', 'sbk', 'sbl', 'sbm', 'sbn', 'sbo', 'sbp', 'sbq', 'sbr',
                        'sbs', 'sbt', 'sbu', 'sbv', 'sbw', 'sbx', 'sby', 'sbz', 'scb', 'sce', 'scf', 'scg', 'sch',
                        'sci', 'sck', 'scl', 'scn', 'sco', 'scp', 'scq', 'scs', 'scu', 'scv', 'scw', 'sda', 'sdb',
                        'sdc', 'sde', 'sdf', 'sdg', 'sdh', 'sdj', 'sdk', 'sdl', 'sdm', 'sdn', 'sdo', 'sdp', 'sdr',
                        'sds', 'sdu', 'sdx', 'sdz', 'sea', 'seb', 'sec', 'sed', 'see', 'sef', 'seg', 'seh', 'sei',
                        'sej', 'sek', 'sel', 'sen', 'seo', 'sep', 'seq', 'ser', 'ses', 'set', 'seu', 'sev', 'sew',
                        'sey', 'sez', 'sfb', 'sfe', 'sfm', 'sfs', 'sfw', 'sga', 'sgb', 'sgc', 'sgd', 'sge', 'sgg',
                        'sgh', 'sgi', 'sgj', 'sgk', 'sgm', 'sgp', 'sgr', 'sgs', 'sgt', 'sgu', 'sgw', 'sgx', 'sgy',
                        'sgz', 'sha', 'shb', 'shc', 'shd', 'she', 'shg', 'shh', 'shi', 'shj', 'shk', 'shl', 'shm',
                        'shn', 'sho', 'shp', 'shq', 'shr', 'shs', 'sht', 'shu', 'shv', 'shw', 'shx', 'shy', 'shz',
                        'sia', 'sib', 'sid', 'sie', 'sif', 'sig', 'sih', 'sii', 'sij', 'sil', 'sim', 'sin', 'sip',
                        'siq', 'sir', 'sis', 'siu', 'siv', 'siw', 'six', 'siy', 'siz', 'sja', 'sjb', 'sjd', 'sje',
                        'sjg', 'sjk', 'sjl', 'sjm', 'sjo', 'sjp', 'sjr', 'sjs', 'sjt', 'sju', 'sjw', 'ska', 'skb',
                        'skc', 'skd', 'ske', 'skf', 'skg', 'skh', 'ski', 'skj', 'skk', 'skm', 'skn', 'sko', 'skp',
                        'skq', 'skr', 'sks', 'skt', 'sku', 'skv', 'skw', 'skx', 'sky', 'skz', 'slc', 'sld', 'sle',
                        'slf', 'slg', 'slh', 'sli', 'slk', 'sll', 'slm', 'sln', 'slp', 'slq', 'slr', 'slt', 'slu',
                        'slv', 'slw', 'slx', 'sly', 'slz', 'sma', 'smb', 'smc', 'smd', 'sme', 'smf', 'smg', 'smh',
                        'smj', 'smk', 'sml', 'smm', 'smn', 'smo', 'smp', 'smq', 'smr', 'sms', 'smt', 'smu', 'smv',
                        'smw', 'smx', 'smy', 'smz', 'sna', 'snb', 'snc', 'snd', 'sne', 'snf', 'sng', 'sni', 'snj',
                        'snk', 'snl', 'snm', 'snn', 'sno', 'snp', 'snq', 'snr', 'sns', 'snu', 'snv', 'snw', 'snx',
                        'sny', 'snz', 'soa', 'sob', 'soc', 'sod', 'soe', 'sog', 'soh', 'soi', 'soj', 'sok', 'sol',
                        'som', 'soo', 'sop', 'soq', 'sor', 'sos', 'sot', 'sou', 'sov', 'sow', 'sox', 'soy', 'soz',
                        'spa', 'spb', 'spc', 'spd', 'spe', 'spg', 'spi', 'spk', 'spl', 'spm', 'spn', 'spo', 'spp',
                        'spq', 'spr', 'sps', 'spt', 'spu', 'spv', 'spx', 'spy', 'sqa', 'sqh', 'sqi', 'sqk', 'sqm',
                        'sqn', 'sqo', 'sqq', 'sqs', 'sqt', 'squ', 'sra', 'srb', 'src', 'sre', 'srf', 'srg', 'srh',
                        'sri', 'srk', 'srl', 'srm', 'srn', 'sro', 'srp', 'srq', 'srr', 'srs', 'srt', 'sru', 'srv',
                        'srw', 'srx', 'sry', 'srz', 'ssb', 'ssc', 'ssd', 'sse', 'ssf', 'ssg', 'ssh', 'ssi', 'ssj',
                        'ssk', 'ssl', 'ssm', 'ssn', 'sso', 'ssp', 'ssq', 'ssr', 'sss', 'sst', 'ssu', 'ssv', 'ssw',
                        'ssx', 'ssy', 'ssz', 'stb', 'ste', 'stf', 'stg', 'sti', 'stj', 'stk', 'stl', 'stm', 'stn',
                        'sto', 'stp', 'stq', 'str', 'sts', 'stt', 'stu', 'stv', 'stw', 'sty', 'sua', 'sub', 'suc',
                        'sue', 'sug', 'sui', 'suj', 'suk', 'sun', 'suq', 'sur', 'sus', 'sut', 'suv', 'suw', 'sux',
                        'suy', 'suz', 'sva', 'svb', 'svc', 'sve', 'svk', 'svm', 'svs', 'swa', 'swb', 'swc', 'swe',
                        'swf', 'swg', 'swh', 'swi', 'swj', 'swk', 'swl', 'swm', 'swn', 'swo', 'swp', 'swq', 'swr',
                        'sws', 'swt', 'swu', 'swv', 'sww', 'swx', 'swy', 'sxb', 'sxe', 'sxg', 'sxk', 'sxn', 'sxr',
                        'sxs', 'sxu', 'sxw', 'sya', 'syb', 'syc', 'syi', 'syk', 'syl', 'sym', 'syn', 'syo', 'sys',
                        'syw', 'syx', 'syy', 'sza', 'szb', 'szc', 'sze', 'szg', 'szl', 'szn', 'szp', 'szv', 'szw',
                        'taa', 'tab', 'tac', 'tad', 'tae', 'taf', 'tag', 'tah', 'taj', 'tak', 'tal', 'tam', 'tan',
                        'tao', 'tap', 'taq', 'tar', 'tat', 'tau', 'tav', 'taw', 'tax', 'tay', 'taz', 'tba', 'tbc',
                        'tbd', 'tbe', 'tbf', 'tbg', 'tbh', 'tbi', 'tbj', 'tbk', 'tbl', 'tbm', 'tbn', 'tbo', 'tbp',
                        'tbr', 'tbs', 'tbt', 'tbu', 'tbv', 'tbw', 'tbx', 'tby', 'tbz', 'tca', 'tcb', 'tcc', 'tcd',
                        'tce', 'tcf', 'tcg', 'tch', 'tci', 'tck', 'tcl', 'tcm', 'tcn', 'tco', 'tcp', 'tcq', 'tcs',
                        'tct', 'tcu', 'tcw', 'tcx', 'tcy', 'tcz', 'tda', 'tdb', 'tdc', 'tdd', 'tde', 'tdf', 'tdg',
                        'tdh', 'tdi', 'tdj', 'tdk', 'tdl', 'tdn', 'tdo', 'tdq', 'tdr', 'tds', 'tdt', 'tdu', 'tdv',
                        'tdx', 'tdy', 'tea', 'tec', 'ted', 'tee', 'tef', 'teg', 'teh', 'tei', 'tek', 'tel', 'tem',
                        'ten', 'teo', 'tep', 'teq', 'ter', 'tes', 'tet', 'teu', 'tev', 'tew', 'tex', 'tey', 'tfi',
                        'tfn', 'tfo', 'tfr', 'tft', 'tga', 'tgb', 'tgc', 'tgd', 'tge', 'tgf', 'tgh', 'tgi', 'tgj',
                        'tgk', 'tgl', 'tgn', 'tgo', 'tgp', 'tgq', 'tgs', 'tgt', 'tgu', 'tgw', 'tgx', 'tgy', 'tgz',
                        'tha', 'thd', 'the', 'thf', 'thh', 'thi', 'thk', 'thl', 'thm', 'thn', 'thp', 'thq', 'thr',
                        'ths', 'tht', 'thu', 'thv', 'thy', 'thz', 'tia', 'tic', 'tid', 'tif', 'tig', 'tih', 'tii',
                        'tij', 'tik', 'til', 'tim', 'tin', 'tio', 'tip', 'tiq', 'tir', 'tis', 'tit', 'tiu', 'tiv',
                        'tiw', 'tix', 'tiy', 'tiz', 'tja', 'tjg', 'tji', 'tjl', 'tjm', 'tjn', 'tjo', 'tjs', 'tju',
                        'tjw', 'tkb', 'tkd', 'tke', 'tkg', 'tkl', 'tkm', 'tkn', 'tkp', 'tkq', 'tkr', 'tks', 'tkt',
                        'tku', 'tkv', 'tkw', 'tkx', 'tkz', 'tla', 'tlb', 'tlc', 'tld', 'tlf', 'tlg', 'tli', 'tlj',
                        'tlk', 'tll', 'tlm', 'tln', 'tlo', 'tlp', 'tlq', 'tlr', 'tls', 'tlt', 'tlu', 'tlv', 'tlx',
                        'tly', 'tma', 'tmb', 'tmc', 'tmd', 'tmf', 'tmg', 'tmh', 'tmi', 'tmj', 'tml', 'tmm', 'tmn',
                        'tmo', 'tmp', 'tmq', 'tmr', 'tms', 'tmt', 'tmu', 'tmv', 'tmw', 'tmy', 'tmz', 'tna', 'tnb',
                        'tnc', 'tnd', 'tne', 'tng', 'tnh', 'tni', 'tnk', 'tnl', 'tnm', 'tnn', 'tno', 'tnp', 'tnq',
                        'tnr', 'tns', 'tnt', 'tnu', 'tnv', 'tnw', 'tnx', 'tny', 'tnz', 'tob', 'toc', 'tod', 'tof',
                        'tog', 'toh', 'toi', 'toj', 'tol', 'tom', 'ton', 'too', 'top', 'toq', 'tor', 'tos', 'tou',
                        'tov', 'tow', 'tox', 'toy', 'toz', 'tpa', 'tpc', 'tpe', 'tpf', 'tpg', 'tpi', 'tpj', 'tpk',
                        'tpl', 'tpm', 'tpn', 'tpo', 'tpp', 'tpq', 'tpr', 'tpt', 'tpu', 'tpv', 'tpw', 'tpx', 'tpy',
                        'tpz', 'tqb', 'tql', 'tqm', 'tqn', 'tqo', 'tqp', 'tqq', 'tqr', 'tqt', 'tqu', 'tqw', 'tra',
                        'trb', 'trc', 'trd', 'tre', 'trf', 'trg', 'trh', 'tri', 'trj', 'trm', 'trn', 'tro', 'trp',
                        'trq', 'trr', 'trs', 'trt', 'tru', 'trv', 'trw', 'trx', 'try', 'trz', 'tsa', 'tsb', 'tsc',
                        'tsd', 'tse', 'tsg', 'tsh', 'tsi', 'tsj', 'tsk', 'tsl', 'tsm', 'tsn', 'tso', 'tsp', 'tsq',
                        'tsr', 'tss', 'tst', 'tsu', 'tsv', 'tsw', 'tsx', 'tsy', 'tsz', 'tta', 'ttb', 'ttc', 'ttd',
                        'tte', 'ttf', 'ttg', 'tth', 'tti', 'ttj', 'ttk', 'ttl', 'ttm', 'ttn', 'tto', 'ttp', 'ttq',
                        'ttr', 'tts', 'ttt', 'ttu', 'ttv', 'ttw', 'tty', 'ttz', 'tua', 'tub', 'tuc', 'tud', 'tue',
                        'tuf', 'tug', 'tuh', 'tui', 'tuj', 'tuk', 'tul', 'tum', 'tun', 'tuo', 'tuq', 'tur', 'tus',
                        'tuu', 'tuv', 'tux', 'tuy', 'tuz', 'tva', 'tvd', 'tve', 'tvk', 'tvl', 'tvm', 'tvn', 'tvo',
                        'tvs', 'tvt', 'tvu', 'tvw', 'tvy', 'twa', 'twb', 'twc', 'twd', 'twe', 'twf', 'twg', 'twh',
                        'twi', 'twl', 'twn', 'two', 'twp', 'twq', 'twr', 'twt', 'twu', 'tww', 'twx', 'twy', 'txa',
                        'txb', 'txc', 'txe', 'txg', 'txh', 'txi', 'txj', 'txm', 'txn', 'txo', 'txq', 'txs', 'txt',
                        'txu', 'txx', 'txy', 'tya', 'tye', 'tyh', 'tyi', 'tyj', 'tyn', 'typ', 'tyr', 'tys', 'tyt',
                        'tyu', 'tyv', 'tyx', 'tyz', 'tza', 'tzh', 'tzj', 'tzm', 'tzn', 'tzo', 'tzx', 'uan', 'uar',
                        'uba', 'ubi', 'ubl', 'ubr', 'ubu', 'uby', 'uda', 'ude', 'udg', 'udi', 'udj', 'udl', 'udm',
                        'udu', 'ues', 'ufi', 'uga', 'ugb', 'uge', 'ugn', 'ugo', 'ugy', 'uha', 'uhn', 'uig', 'uis',
                        'uiv', 'uji', 'uka', 'ukg', 'ukh', 'ukl', 'ukp', 'ukq', 'ukr', 'uks', 'uku', 'ukw', 'uky',
                        'ula', 'ulb', 'ulc', 'ule', 'ulf', 'uli', 'ulk', 'ull', 'ulm', 'uln', 'ulu', 'ulw', 'uma',
                        'umb', 'umd', 'umg', 'umi', 'umm', 'umn', 'umo', 'ump', 'umr', 'ums', 'umu', 'una', 'une',
                        'ung', 'unk', 'unm', 'unn', 'unr', 'unu', 'unz', 'upi', 'upv', 'ura', 'urb', 'urc', 'urd',
                        'ure', 'urg', 'urh', 'uri', 'urk', 'url', 'urm', 'urn', 'uro', 'urr', 'urt', 'uru', 'urv',
                        'urw', 'urx', 'ury', 'urz', 'usa', 'ush', 'usi', 'usk', 'usp', 'usu', 'uta', 'ute', 'utp',
                        'utr', 'utu', 'uum', 'uun', 'uur', 'uuu', 'uve', 'uvh', 'uvl', 'uwa', 'uya', 'uzb', 'uzn',
                        'uzs', 'vaa', 'vae', 'vaf', 'vag', 'vah', 'vai', 'vaj', 'val', 'vam', 'van', 'vao', 'vap',
                        'var', 'vas', 'vau', 'vav', 'vay', 'vbb', 'vec', 'ved', 'vel', 'vem', 'ven', 'veo', 'vep',
                        'ver', 'vgr', 'vgt', 'vic', 'vid', 'vie', 'vif', 'vig', 'vil', 'vin', 'vis', 'vit', 'viv',
                        'vka', 'vki', 'vkj', 'vkk', 'vkl', 'vkm', 'vko', 'vkp', 'vkt', 'vku', 'vlp', 'vls', 'vma',
                        'vmb', 'vmc', 'vmd', 'vme', 'vmf', 'vmg', 'vmh', 'vmi', 'vmj', 'vmk', 'vml', 'vmm', 'vmp',
                        'vmq', 'vmr', 'vmu', 'vmv', 'vmw', 'vmx', 'vmy', 'vmz', 'vnk', 'vnm', 'vnp', 'vor', 'vot',
                        'vra', 'vro', 'vrs', 'vrt', 'vsi', 'vsl', 'vsv', 'vto', 'vum', 'vun', 'vut', 'vwa', 'waa',
                        'wab', 'wac', 'wad', 'wae', 'wag', 'wah', 'waj', 'wal', 'wam', 'wan', 'wao', 'wap', 'waq',
                        'war', 'was', 'wat', 'wau', 'wav', 'waw', 'wax', 'way', 'waz', 'wba', 'wbb', 'wbe', 'wbf',
                        'wbh', 'wbi', 'wbj', 'wbk', 'wbl', 'wbm', 'wbp', 'wbq', 'wbr', 'wbt', 'wbv', 'wbw', 'wca',
                        'wci', 'wdd', 'wdg', 'wdj', 'wdk', 'wdu', 'wea', 'wec', 'wed', 'weg', 'weh', 'wei', 'wem',
                        'weo', 'wep', 'wer', 'wes', 'wet', 'wew', 'wfg', 'wga', 'wgb', 'wgg', 'wgi', 'wgo', 'wgu',
                        'wgy', 'wha', 'whg', 'whk', 'wib', 'wic', 'wie', 'wig', 'wih', 'wii', 'wij', 'wik', 'wil',
                        'wim', 'win', 'wir', 'wiu', 'wiv', 'wiy', 'wja', 'wji', 'wka', 'wkd', 'wkl', 'wku', 'wkw',
                        'wla', 'wlc', 'wle', 'wlg', 'wli', 'wlk', 'wll', 'wln', 'wlo', 'wlr', 'wls', 'wlu', 'wlv',
                        'wlw', 'wlx', 'wly', 'wmb', 'wmc', 'wmd', 'wme', 'wmh', 'wmi', 'wmm', 'wmn', 'wmo', 'wms',
                        'wmt', 'wmw', 'wmx', 'wnb', 'wnc', 'wnd', 'wne', 'wng', 'wni', 'wnk', 'wnm', 'wno', 'wnp',
                        'wnu', 'wnw', 'wny', 'woa', 'wob', 'woc', 'wod', 'woe', 'wof', 'wog', 'woi', 'wok', 'wol',
                        'wom', 'won', 'woo', 'wor', 'wos', 'wow', 'wpc', 'wrb', 'wrg', 'wrh', 'wri', 'wrk', 'wrl',
                        'wrm', 'wrn', 'wro', 'wrp', 'wrr', 'wrs', 'wru', 'wrv', 'wrw', 'wrx', 'wry', 'wrz', 'wsa',
                        'wsi', 'wsk', 'wsr', 'wss', 'wsv', 'wtf', 'wth', 'wti', 'wtk', 'wtm', 'wtw', 'wua', 'wub',
                        'wud', 'wuh', 'wul', 'wum', 'wun', 'wur', 'wut', 'wuu', 'wuv', 'wux', 'wuy', 'wwa', 'wwo',
                        'wwr', 'www', 'wxa', 'wxw', 'wya', 'wyb', 'wyi', 'wym', 'wyr', 'wyy', 'xaa', 'xab', 'xac',
                        'xad', 'xag', 'xal', 'xam', 'xan', 'xap', 'xar', 'xas', 'xat', 'xau', 'xav', 'xaw', 'xay',
                        'xbc', 'xbe', 'xbg', 'xbi', 'xbj', 'xbn', 'xbo', 'xbp', 'xbr', 'xby', 'xce', 'xcg', 'xch',
                        'xcl', 'xcm', 'xcn', 'xco', 'xcr', 'xct', 'xcv', 'xcw', 'xcy', 'xda', 'xdc', 'xdk', 'xdm',
                        'xdy', 'xeb', 'xed', 'xeg', 'xel', 'xem', 'xer', 'xes', 'xet', 'xeu', 'xfa', 'xga', 'xgb',
                        'xgd', 'xgf', 'xgg', 'xgm', 'xgr', 'xgu', 'xgw', 'xhd', 'xhe', 'xho', 'xhr', 'xht', 'xhu',
                        'xib', 'xii', 'xir', 'xis', 'xiy', 'xjb', 'xjt', 'xka', 'xkb', 'xkc', 'xkd', 'xke', 'xkf',
                        'xkg', 'xki', 'xkj', 'xkk', 'xkl', 'xkn', 'xkp', 'xkq', 'xkr', 'xks', 'xkt', 'xku', 'xkv',
                        'xkw', 'xkx', 'xky', 'xkz', 'xla', 'xlc', 'xld', 'xlo', 'xlp', 'xls', 'xlu', 'xma', 'xmb',
                        'xmc', 'xmd', 'xmf', 'xmg', 'xmh', 'xmj', 'xml', 'xmm', 'xmp', 'xmr', 'xms', 'xmt', 'xmu',
                        'xmv', 'xmw', 'xmx', 'xmy', 'xmz', 'xna', 'xnb', 'xng', 'xni', 'xnn', 'xno', 'xnr', 'xns',
                        'xnt', 'xny', 'xnz', 'xoc', 'xod', 'xog', 'xoi', 'xok', 'xom', 'xon', 'xoo', 'xop', 'xor',
                        'xow', 'xpa', 'xpc', 'xpe', 'xpg', 'xpj', 'xpk', 'xpm', 'xpo', 'xpq', 'xpr', 'xps', 'xpt',
                        'xpu', 'xqt', 'xra', 'xrb', 'xrd', 'xre', 'xri', 'xrn', 'xru', 'xrw', 'xsa', 'xsb', 'xsd',
                        'xse', 'xsh', 'xsi', 'xsl', 'xsm', 'xsn', 'xsp', 'xsq', 'xsr', 'xsu', 'xsy', 'xta', 'xtb',
                        'xtc', 'xtd', 'xte', 'xtg', 'xth', 'xti', 'xtj', 'xtl', 'xtm', 'xtn', 'xto', 'xtp', 'xtq',
                        'xts', 'xtt', 'xtu', 'xtv', 'xtw', 'xty', 'xua', 'xub', 'xud', 'xug', 'xuj', 'xul', 'xum',
                        'xun', 'xuo', 'xup', 'xur', 'xut', 'xuu', 'xve', 'xvi', 'xvo', 'xwa', 'xwc', 'xwd', 'xwe',
                        'xwg', 'xwj', 'xwk', 'xwl', 'xwr', 'xwt', 'xww', 'xxb', 'xxk', 'xxm', 'xxr', 'xxt', 'xya',
                        'xyb', 'xyj', 'xyk', 'xyt', 'xyy', 'xzh', 'yaa', 'yab', 'yac', 'yad', 'yae', 'yaf', 'yag',
                        'yah', 'yai', 'yaj', 'yak', 'yal', 'yam', 'yan', 'yao', 'yap', 'yaq', 'yar', 'yas', 'yat',
                        'yau', 'yav', 'yaw', 'yay', 'yaz', 'yba', 'ybb', 'ybe', 'ybh', 'ybi', 'ybj', 'ybk', 'ybl',
                        'ybm', 'ybn', 'ybo', 'ybx', 'yby', 'ych', 'ycl', 'ycn', 'ycp', 'yda', 'ydd', 'yde', 'ydg',
                        'ydk', 'yea', 'yee', 'yei', 'yej', 'yel', 'yer', 'yes', 'yet', 'yeu', 'yev', 'yey', 'yga',
                        'ygl', 'ygm', 'ygp', 'ygr', 'ygs', 'ygw', 'yha', 'yhd', 'yhl', 'yia', 'yid', 'yif', 'yig',
                        'yih', 'yii', 'yij', 'yik', 'yil', 'yim', 'yin', 'yip', 'yiq', 'yir', 'yis', 'yit', 'yiu',
                        'yiv', 'yix', 'yiz', 'yka', 'ykg', 'yki', 'ykk', 'ykl', 'ykm', 'ykn', 'yko', 'ykr', 'ykt',
                        'yku', 'yky', 'yla', 'yle', 'ylg', 'yli', 'yll', 'ylm', 'yln', 'ylo', 'ylr', 'ylu', 'yly',
                        'ymb', 'ymc', 'ymd', 'yme', 'ymh', 'ymi', 'ymk', 'yml', 'ymm', 'ymn', 'ymo', 'ymp', 'ymq',
                        'ymr', 'yms', 'ymx', 'ymz', 'yna', 'ynd', 'yng', 'ynk', 'ynl', 'ynn', 'yno', 'ynq', 'yns',
                        'ynu', 'yob', 'yog', 'yoi', 'yok', 'yol', 'yom', 'yon', 'yor', 'yot', 'yox', 'yoy', 'ypa',
                        'ypb', 'ypg', 'yph', 'ypm', 'ypn', 'ypo', 'ypp', 'ypz', 'yra', 'yrb', 'yre', 'yrk', 'yrl',
                        'yrm', 'yrn', 'yrw', 'yry', 'ysd', 'ysg', 'ysl', 'ysn', 'yso', 'ysr', 'yss', 'ysy', 'yta',
                        'ytl', 'ytp', 'ytw', 'yua', 'yub', 'yuc', 'yud', 'yue', 'yuf', 'yug', 'yui', 'yuj', 'yuk',
                        'yul', 'yum', 'yun', 'yup', 'yuq', 'yur', 'yut', 'yuw', 'yux', 'yuy', 'yuz', 'yva', 'yvt',
                        'ywa', 'ywg', 'ywl', 'ywn', 'ywq', 'ywr', 'ywt', 'ywu', 'yww', 'yxg', 'yxl', 'yxm', 'yxu',
                        'yxy', 'yyr', 'yyu', 'yyz', 'yzg', 'yzk', 'zaa', 'zab', 'zac', 'zad', 'zae', 'zaf', 'zag',
                        'zah', 'zai', 'zaj', 'zak', 'zal', 'zam', 'zao', 'zap', 'zaq', 'zar', 'zas', 'zat', 'zau',
                        'zav', 'zaw', 'zax', 'zay', 'zaz', 'zbc', 'zbe', 'zbt', 'zbw', 'zca', 'zch', 'zdj', 'zea',
                        'zeg', 'zeh', 'zen', 'zga', 'zgb', 'zgh', 'zgm', 'zgn', 'zgr', 'zha', 'zhb', 'zhd', 'zhi',
                        'zhn', 'zhw', 'zia', 'zib', 'zik', 'zil', 'zim', 'zin', 'zir', 'ziw', 'ziz', 'zka', 'zkb',
                        'zkd', 'zkk', 'zkn', 'zko', 'zkp', 'zkr', 'zkt', 'zku', 'zlj', 'zlm', 'zln', 'zlq', 'zma',
                        'zmb', 'zmc', 'zmd', 'zme', 'zmf', 'zmg', 'zmh', 'zmi', 'zmj', 'zmk', 'zml', 'zmm', 'zmn',
                        'zmo', 'zmp', 'zmq', 'zmr', 'zms', 'zmt', 'zmu', 'zmv', 'zmw', 'zmx', 'zmy', 'zmz', 'zna',
                        'zne', 'zng', 'zns', 'zoc', 'zoh', 'zom', 'zoo', 'zoq', 'zor', 'zos', 'zpa', 'zpb', 'zpc',
                        'zpd', 'zpe', 'zpf', 'zpg', 'zph', 'zpi', 'zpj', 'zpk', 'zpl', 'zpm', 'zpn', 'zpo', 'zpp',
                        'zpq', 'zpr', 'zps', 'zpt', 'zpu', 'zpv', 'zpw', 'zpx', 'zpy', 'zpz', 'zqe', 'zrn', 'zro',
                        'zrs', 'zsa', 'zsl', 'zsm', 'zsr', 'zsu', 'zte', 'ztg', 'ztl', 'ztm', 'ztn', 'ztp', 'ztq',
                        'zts', 'ztt', 'ztu', 'ztx', 'zty', 'zua', 'zuh', 'zul', 'zum', 'zun', 'zuy', 'zwa', 'zyb',
                        'zyg', 'zyj', 'zyn', 'zyp', 'zza', 'zzj'
                        ]

    def __init__(self):
        super().__init__(Alphabet.IPA)
        from transphone import read_tokenizer
        self.read_tokenizer = read_tokenizer
        self._models = {}

    @classmethod
    def get_lang(cls, target_lang: str) -> str:
        """
        Validates and returns the closest supported language code.

        Args:
            target_lang (str): The language code to validate.

        Returns:
            str: The validated language code.

        Raises:
            ValueError: If the language code is unsupported.
        """
        if target_lang.lower() in cls.TRANSPHONE_LANGS:
            return target_lang
        return cls.match_lang(target_lang, cls.TRANSPHONE_LANGS)

    def phonemize_string(self, text: str, lang: str) -> str:
        lang = self.get_lang(lang)
        pho = self._models.get(lang)
        if pho is None:
            self._models[lang] = pho = self.read_tokenizer(lang)
        return "".join(
            [p if p != "<SPACE>" else " "
             for p in pho.tokenize(text, use_space=True)]
        ).strip()


if __name__ == "__main__":
    # for comparison

    tphone = TransphonePhonemizer()
    byt5 = ByT5Phonemizer()
    espeak = EspeakPhonemizer()
    gruut = GruutPhonemizer()
    goruut = GoruutPhonemizer(remote_url='https://hashtron.cloud')
    epitr = EpitranPhonemizer()
    charsiu = CharsiuPhonemizer()
    misaki = MisakiPhonemizer()

    lang = "en-gb"

    text1 = "Hello, world. How are you?"

    print("\n--- Getting phonemes for 'Hello, world. How are you?' ---")
    phonemes1 = espeak.phonemize(text1, lang)
    phonemes1b = gruut.phonemize(text1, lang)
    phonemes1c = byt5.phonemize(text1, lang)
    phonemes1d = epitr.phonemize(text1, lang)
    phonemes1e = charsiu.phonemize(text1, lang)
    phonemes1f = misaki.phonemize(text1, lang)
    phonemes1g = tphone.phonemize(text1, lang)
    phonemes1h = goruut.phonemize(text1, lang)
    print(f" Espeak         Phonemes: {phonemes1}")
    print(f" Gruut          Phonemes: {phonemes1b}")
    print(f" byt5           Phonemes: {phonemes1c}")
    print(f" Epitran        Phonemes: {phonemes1d}")
    print(f" Charsiu        Phonemes: {phonemes1e}")
    print(f" Misaki         Phonemes: {phonemes1f}")
    print(f" Transphone     Phonemes: {phonemes1g}")
    print(f" Goruut         Phonemes: {phonemes1h}")

    lang = "nl"
    sentence = "DJ's en bezoekers van Tomorrowland waren woensdagavond dolblij toen het paradepaardje van het festival alsnog opende in Oostenrijk op de Mainstage.\nWant het optreden van Metallica, waar iedereen zo blij mee was, zou hoe dan ook doorgaan, aldus de DJ die het nieuws aankondigde."
    sentence = "Een regenboog is een gekleurde cirkelboog die aan de hemel waargenomen kan worden als de, laagstaande, zon tegen een nevel van waterdruppeltjes aan schijnt en de zon zich achter de waarnemer bevindt. Het is een optisch effect dat wordt veroorzaakt door de breking en weerspiegeling van licht in de waterdruppels."
    print(f"\n--- Getting phonemes for '{sentence}' ---")
    text1 = sentence
    phonemes1 = espeak.phonemize(text1, lang)
    phonemes1b = gruut.phonemize(text1, lang)
    phonemes1c = byt5.phonemize(text1, lang)
    phonemes1d = epitr.phonemize(text1, lang)
    phonemes1e = charsiu.phonemize(text1, lang)
    print(f" Espeak  Phonemes: {phonemes1}")
    print(f" Gruut   Phonemes: {phonemes1b}")
    print(f" byt5    Phonemes: {phonemes1c}")
    print(f" Epitran Phonemes: {phonemes1d}")
    print(f" Charsiu Phonemes: {phonemes1e}")
