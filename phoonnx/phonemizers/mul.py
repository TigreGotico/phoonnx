"""multilingual phonemizers"""

import json
import os
import re
import subprocess
from typing import List, Dict, Optional

import numpy as np
import onnxruntime
import requests

from phoonnx.phonemizers.base import BasePhonemizer


class EspeakError(Exception):
    """Custom exception for espeak-ng related errors."""
    pass


class ByT5Phonemizer(BasePhonemizer):
    """
    A phonemizer class that uses a ByT5 ONNX model to convert text into phonemes.
    """
    # TODO - support multiple models
    TOKENIZER_CONFIG_URL = "https://huggingface.co/OpenVoiceOS/g2p-multilingual-byt5-tiny-8l-ipa-childes-onnx/resolve/main/tokenizer_config.json"
    MODEL_URL = "https://huggingface.co/OpenVoiceOS/g2p-multilingual-byt5-tiny-8l-ipa-childes-onnx/resolve/main/byt5_g2p_model.onnx"
    BYT5_LANGS = ['ca', 'cy', 'da', 'de', 'en-na', 'en-uk', 'es', 'et', 'eu', 'fa', 'fr', 'ga', 'hr', 'hu', 'id', 'is',
                  'it', 'ja', 'ko', 'nl', 'no', 'pl', 'pt', 'pt-br', 'qu', 'ro', 'sr', 'sv', 'tr', 'zh', 'zh-yue']

    def __init__(self, onnx_model_path: Optional[str] = None, tokenizer_config: Optional[str] = None):
        """
        Initializes the ByT5Phonemizer with the ONNX model and tokenizer configuration.
        If paths are not provided, it attempts to download them to a local directory.

        Args:
            onnx_model_path (str, optional): Path to the ONNX model file. If None, it will be downloaded.
            tokenizer_config (str, optional): Path to the tokenizer configuration JSON file. If None, it will be downloaded.
        """
        # Define the local data path for models and configs
        data_path = os.path.expanduser("~/.local/share/byt5_phonemizer")
        os.makedirs(data_path, exist_ok=True)  # Ensure the directory exists

        # Determine the actual paths for the model and tokenizer config
        if onnx_model_path is None:
            self.onnx_model_path = os.path.join(data_path, "byt5_g2p_model.onnx")
        else:
            self.onnx_model_path = onnx_model_path

        if tokenizer_config is None:
            self.tokenizer_config = os.path.join(data_path, "tokenizer_config.json")
        else:
            self.tokenizer_config = tokenizer_config

        # Download model if it doesn't exist
        if not os.path.exists(self.onnx_model_path):
            print(f"Downloading ONNX model from {self.MODEL_URL} to {self.onnx_model_path}...")
            try:
                response = requests.get(self.MODEL_URL, stream=True)
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

        # TODO - GPU support (optional dependencies only)
        self.session = onnxruntime.InferenceSession(self.onnx_model_path, providers=['CPUExecutionProvider'])
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
        if target_lang.lower() in ["en-us", "en_us"]:
            return "en-na"  # english north america

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
    # TODO - support multiple models
    MODEL_URL = "https://huggingface.co/Jarbas/charsiu_g2p_multilingual_byT5_tiny_16_layers_100_onnx/resolve/main/charsiu_g2p_multilingual_byT5_tiny_16_layers_100.onnx"
    BYT5_LANGS = ['ady', 'afr', 'sqi', 'amh', 'ara', 'arg', 'arm-e', 'arm-w', 'aze', 'bak', 'eus', 'bel', 'ben', 'bos',
                  'bul', 'bur', 'cat', 'yue', 'zho-t', 'zho-s', 'min', 'cze', 'dan', 'dut', 'eng-uk', 'eng-us', 'epo',
                  'est', 'fin', 'fra', 'fra-qu', 'gla', 'geo', 'ger', 'gre', 'grc', 'grn', 'guj', 'hin', 'hun', 'ido',
                  'ind', 'ina', 'ita', 'jam', 'jpn', 'kaz', 'khm', 'kor', 'kur', 'lat-clas', 'lat-eccl', 'lit', 'ltz',
                  'mac', 'mlt', 'tts', 'nob', 'ori', 'pap', 'fas', 'pol', 'por-po', 'por-bz', 'ron', 'rus', 'san',
                  'srp', 'hbs-latn', 'hbs-cyrl', 'snd', 'slo', 'slv', 'spa', 'spa-latin', 'spa-me', 'swa', 'swe', 'tgl',
                  'tam', 'tat', 'tha', 'tur', 'tuk', 'ukr', 'vie-n', 'vie-c', 'vie-s', 'wel-nw', 'wel-sw', 'ice', 'ang',
                  'gle', 'enm', 'syc', 'glg', 'sme', 'egy']

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
        if target_lang.lower() in ["en-us", "en_us"]:
            return "en-na"  # english north america

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
    GRUUT_LANGS = ["en", "ar", "ca", "cs", "de", "es", "fa", "fr", "it", "lb", "nl", "pt", "ru", "sv", "sw"]

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


class EpitranPhonemizer(BasePhonemizer):
    """
    A phonemizer class that uses the Gruut library to convert text into phonemes.
    Note: Gruut's internal segmentation is sentence-based
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


class GraphemePhonemizer(BasePhonemizer):
    """
    A phonemizer class that treats input text as graphemes (characters).
    It performs text normalization and returns the normalized text as a string
    of characters.
    """
    # Regular expression matching whitespace:
    whitespace_re = re.compile(r"\s+")

    def phonemize_string(self, text: str, lang: str) -> str:
        """
        Normalizes input text by applying a series of transformations
        and returns it as a sequence of graphemes.

        Parameters:
            text (str): Input text to be converted to graphemes.
            lang (str): The language code (ignored for grapheme phonemization,
                        but required by BasePhonemizer).

        Returns:
            str: A normalized string of graphemes.
        """
        text = text.lower()
        text = text.replace(";", ",")
        text = text.replace("-", " ")
        text = text.replace(":", ",")
        text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
        text = re.sub(self.whitespace_re, " ", text).strip()
        return text


class MisakiPhonemizer(BasePhonemizer):
    """
    https://github.com/hexgrad/misaki
    """
    MISAKI_LANGS = ['en', 'ko', 'ja', 'vi', 'zh', 'he']

    def __init__(self):
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
        return cls.match_lang(target_lang, cls.MISAKI_LANGS)

    def phonemize_string(self, text: str, lang: str) -> str:
        lang = self.get_lang(lang)
        epi = self._epis.get(lang)
        if epi is None:
            epi = self.epitran.Epitran(lang)
            self._epis[lang] = epi
        return epi.transliterate(text)


if __name__ == "__main__":
    # for comparison

    byt5 = ByT5Phonemizer()
    espeak = EspeakPhonemizer()
    gruut = GruutPhonemizer()
    epitr = EpitranPhonemizer()
    charsiu = CharsiuPhonemizer()
    grapheme_ph = GraphemePhonemizer()

    lang = "en-gb"

    text1 = "Hello, world. How are you?"

    print(f"\n--- Getting graphemes for '{text1}' (GraphemePhonemizer) ---")
    graphemes1 = grapheme_ph.phonemize(text1, lang)
    print(f"  Graphemes: {graphemes1}")


    print("\n--- Getting phonemes for 'Hello, world. How are you?' ---")
    phonemes1 = espeak.phonemize(text1, lang)
    phonemes1b = gruut.phonemize(text1, lang)
    phonemes1c = byt5.phonemize(text1, lang)
    phonemes1d = epitr.phonemize(text1, lang)
    phonemes1e = charsiu.phonemize(text1, lang)
    print(f" Espeak         Phonemes: {phonemes1}")
    print(f" Gruut          Phonemes: {phonemes1b}")
    print(f" byt5           Phonemes: {phonemes1c}")
    print(f" Epitran        Phonemes: {phonemes1d}")
    print(f" Charsiu        Phonemes: {phonemes1e}")


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
