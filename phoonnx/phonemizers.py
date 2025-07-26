import abc
import json
import os
import re
import string
import subprocess
import platform
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import onnxruntime
import requests
from langcodes import tag_distance
from quebra_frases import sentence_tokenize

# list of (substring, terminator, end_of_sentence) tuples.
TextChunks = List[Tuple[str, str, bool]]
# list of (phonemes, terminator, end_of_sentence) tuples.
RawPhonemizedChunks = List[Tuple[str, str, bool]]

class BasePhonemizer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def phonemize_string(self, text: str, lang: str) -> str:
        raise NotImplementedError

    def phonemize(self, text: str, lang: str) -> RawPhonemizedChunks:
        if not text:
            return [('', '', True)]
        results: RawPhonemizedChunks = []
        for chunk, punct, eos in self.chunk_text(text):
            phoneme_str = self.phonemize_string(chunk, lang)
            results += [(self.remove_punctuation(phoneme_str), punct, True)]
        return results

    @staticmethod
    def match_lang(target_lang: str, valid_langs: List[str]) -> str:
        """
        Validates and returns the closest supported language code.

        Args:
            target_lang (str): The language code to validate.

        Returns:
            str: The validated language code.

        Raises:
            ValueError: If the language code is unsupported.
        """
        if target_lang in valid_langs:
            return target_lang
        best_lang = "und"
        best_distance = 10000000
        for l in valid_langs:
            try:
                distance: int = tag_distance(l, target_lang)
            except:
                try:
                    l = f"{l.split('-')[0]}-{l.split('-')[1]}"
                    distance: int = tag_distance(l, target_lang)
                except:
                    try:
                        distance: int = tag_distance(l.split('-')[0], target_lang)
                    except:
                        continue
            if distance < best_distance:
                best_lang, best_distance = l, distance

        # If the score is low (meaning a good match), return the language
        if best_distance <= 10:
            return best_lang
        # Otherwise, raise an error for unsupported language
        raise ValueError(f"unsupported language code: {target_lang}")

    @staticmethod
    def remove_punctuation(text):
        """
        Removes all punctuation characters from a string.
        Punctuation characters are defined by string.punctuation.
        """
        # Create a regex pattern that matches any character in string.punctuation
        punctuation_pattern = r"[" + re.escape(string.punctuation) + r"]"
        return re.sub(punctuation_pattern, '', text).strip()

    @staticmethod
    def chunk_text(text: str, delimiters: Optional[List[str]] = None) -> TextChunks:
        if not text:
            return [('', '', True)]

        results: TextChunks = []
        delimiters = delimiters or [", ", ":", ";", "...", "|"]

        # Create a regex pattern that matches any of the delimiters
        delimiter_pattern = re.escape(delimiters[0])
        for delimiter in delimiters[1:]:
            delimiter_pattern += f"|{re.escape(delimiter)}"

        for sentence in sentence_tokenize(text):
            # Default punctuation if no specific punctuation found
            default_punc = sentence[-1] if sentence[-1] in string.punctuation else "."

            # Use regex to split the sentence by any of the delimiters
            parts = re.split(f'({delimiter_pattern})', sentence)

            # Group parts into chunks (text + delimiter)
            chunks = []
            for i in range(0, len(parts), 2):
                # If there's a delimiter after the text, use it
                delimiter = parts[i + 1] if i + 1 < len(parts) else default_punc

                # Last chunk is marked as complete
                is_last = (i + 2 >= len(parts))

                chunks.append((parts[i].strip(), delimiter.strip(), is_last))

            results.extend(chunks)

        return results


class EspeakError(Exception):
    """Custom exception for espeak-ng related errors."""
    pass


class ByT5Phonemizer(BasePhonemizer):
    """
    A phonemizer class that uses a ByT5 ONNX model to convert text into phonemes.
    It mimics the clause-by-clause segmentation behavior of the piper TTS implementation
    """
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


class CotoviaError(Exception):
    """Custom exception for cotovia related errors."""
    pass


class CotoviaPhonemizer(BasePhonemizer):
    """
    A phonemizer class that uses the Cotovia TTS binary to convert text into phonemes.
    It processes the input sentence through a command-line phonemization tool, applying multiple
    regular expression transformations to clean and normalize the phonetic representation.
    """

    def __init__(self, cotovia_bin_path: Optional[str] = None):
        """
        Initializes the CotoviaPhonemizer.

        Args:
            cotovia_bin_path (str, optional): Path to the Cotovia TTS binary.
                                              If None, it will try to find it in common locations.
        """
        self.cotovia_bin = cotovia_bin_path or self.find_cotovia()
        if not os.path.exists(self.cotovia_bin):
            raise FileNotFoundError(f"Cotovia binary not found at {self.cotovia_bin}. "
                                    "Please ensure it's installed or provide the correct path.")

    @staticmethod
    def find_cotovia() -> str:
        """
        Attempts to find the cotovia binary in common locations.
        """
        path = subprocess.run(["which", "cotovia"], capture_output=True, text=True).stdout.strip()
        if path and os.path.isfile(path):
            return path

        # Fallback to bundled binaries
        local_path = f"{os.path.dirname(__file__)}/cotovia/cotovia_{platform.machine()}"
        if os.path.isfile(local_path):
            return local_path

        # Last resort common system path
        if os.path.isfile("/usr/bin/cotovia"):
            return "/usr/bin/cotovia"

        return "cotovia"  # Return "cotovia" to let subprocess raise FileNotFoundError if not found in PATH

    def phonemize_string(self, text: str, lang: str) -> str:
        """
        Converts a given sentence into phonemes using the Cotovia TTS binary.

        Processes the input sentence through a command-line phonemization tool, applying multiple regular expression transformations to clean and normalize the phonetic representation.

        Parameters:
            text (str): The input text to be phonemized
            lang (str): The language code (ignored by Cotovia, but required by BasePhonemizer)

        Returns:
            str: A cleaned and normalized phonetic representation of the input sentence

        Notes:
            - Uses subprocess to execute the Cotovia TTS binary
            - Applies multiple regex substitutions to improve punctuation and spacing
            - Converts text from ISO-8859-1 to UTF-8 encoding
        """
        cmd = f'echo "{text}" | {self.cotovia_bin} -t -n -S | iconv -f iso88591 -t utf8'
        str_ext = subprocess.check_output(cmd, shell=True).decode("utf-8")

        ## fix punctuation in cotovia output - from official inference script

        # substitute ' ·\n' by ...
        str_ext = re.sub(r" ·", r"...", str_ext)

        # remove spaces before , . ! ? ; : ) ] of the extended string
        str_ext = re.sub(r"\s+([.,!?;:)\]])", r"\1", str_ext)

        # remove spaces after ( [ ¡ ¿ of the extended string
        str_ext = re.sub(r"([\(\[¡¿])\s+", r"\1", str_ext)

        # remove unwanted spaces between quotations marks
        str_ext = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', str_ext)

        # substitute '- text -' to '-text-'
        str_ext = re.sub(r"-\s*([^-]*?)\s*-", r"-\1-", str_ext)

        # remove initial question marks
        str_ext = re.sub(r"[¿¡]", r"", str_ext)

        # eliminate extra spaces
        str_ext = re.sub(r"\s+", r" ", str_ext)

        str_ext = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1 \2", str_ext)

        ### - , ' and () by commas
        # substitute '- text -' to ', text,'
        str_ext = re.sub(r"(\w+)\s+-([^-]*?)-\s+([^-]*?)", r"\1, \\2, ", str_ext)

        # substitute ' - ' by ', '
        str_ext = re.sub(r"(\w+[!\?]?)\s+-\s*", r"\1, ", str_ext)

        # substitute ' ( text )' to ', text,'
        str_ext = re.sub(r"(\w+)\s*\(\s*([^\(\)]*?)\s*\)", r"\1, \\2,", str_ext)

        return str_ext


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


Phonemizer = Union[ByT5Phonemizer, EspeakPhonemizer, GruutPhonemizer, EpitranPhonemizer, CotoviaPhonemizer, GraphemePhonemizer]


if __name__ == "__main__":
    # for comparison

    byt5 = ByT5Phonemizer()
    espeak = EspeakPhonemizer()
    gruut = GruutPhonemizer()
    epitr = EpitranPhonemizer()
    cotovia = CotoviaPhonemizer()
    grapheme_ph = GraphemePhonemizer()

    lang = "nl"
    sentence = "DJ's en bezoekers van Tomorrowland waren woensdagavond dolblij toen het paradepaardje van het festival alsnog opende in Oostenrijk op de Mainstage.\nWant het optreden van Metallica, waar iedereen zo blij mee was, zou hoe dan ook doorgaan, aldus de DJ die het nieuws aankondigde."
    sentence = "Een regenboog is een gekleurde cirkelboog die aan de hemel waargenomen kan worden als de, laagstaande, zon tegen een nevel van waterdruppeltjes aan schijnt en de zon zich achter de waarnemer bevindt. Het is een optisch effect dat wordt veroorzaakt door de breking en weerspiegeling van licht in de waterdruppels."
    print(f"\n--- Getting phonemes for '{sentence}' ---")
    text1 = sentence
    phonemes1 = espeak.phonemize(text1, lang)
    phonemes1b = gruut.phonemize(text1, lang)
    phonemes1c = byt5.phonemize(text1, lang)
    phonemes1d = epitr.phonemize(text1, lang)
    print(f" Espeak  Phonemes: {phonemes1}")
    print(f" Gruut   Phonemes: {phonemes1b}")
    print(f" byt5    Phonemes: {phonemes1c}")
    print(f" Epitran Phonemes: {phonemes1d}")

    lang = "en-gb"

    print("\n--- Getting phonemes for 'Hello, world. How are you?' ---")
    text1 = "Hello, world. How are you?"
    phonemes1 = espeak.phonemize(text1, lang)
    phonemes1b = gruut.phonemize(text1, lang)
    phonemes1c = byt5.phonemize(text1, lang)
    phonemes1d = epitr.phonemize(text1, lang)
    print(f" Espeak  Phonemes: {phonemes1}")
    print(f" Gruut   Phonemes: {phonemes1b}")
    print(f" byt5    Phonemes: {phonemes1c}")
    print(f" Epitran Phonemes: {phonemes1d}")

    print("\n--- Getting phonemes for 'This is a test: a quick one; and done!' ---")
    text2 = "This is a test: a quick one; and done!"
    phonemes2 = espeak.phonemize(text2, lang)
    phonemes2b = gruut.phonemize(text2, lang)
    phonemes2c = byt5.phonemize(text2, lang)
    phonemes2d = epitr.phonemize(text2, lang)
    print(f"  Espeak Phonemes: {phonemes2}")
    print(f"  Gruut Phonemes: {phonemes2b}")
    print(f"   byt5  Phonemes: {phonemes2c}")
    print(f" Epitran Phonemes: {phonemes2d}")

    print("\n--- Getting phonemes for 'Just a phrase without punctuation' ---")
    text3 = "Just a phrase without punctuation"
    phonemes3 = espeak.phonemize(text3, lang)
    phonemes3b = gruut.phonemize(text3, lang)
    phonemes3c = byt5.phonemize(text3, lang)
    phonemes3d = epitr.phonemize(text3, lang)
    print(f"  Espeak Phonemes: {phonemes3}")
    print(f"  Gruut Phonemes: {phonemes3b}")
    print(f"   byt5  Phonemes: {phonemes3c}")
    print(f" Epitran Phonemes: {phonemes3d}")

    lang = "gl"
    text_gl = "Este é un sistema de conversión de texto a voz en lingua galega baseado en redes neuronais artificiais. Ten en conta que as funcionalidades incluídas nesta páxina ofrécense unicamente con fins de demostración. Se tes algún comentario, suxestión ou detectas algún problema durante a demostración, ponte en contacto connosco."
    print(f"\n--- Getting phonemes for '{text_gl}' (Cotovia) ---")
    phonemes_cotovia = cotovia.phonemize(text_gl, lang)
    print(f"  Cotovia Phonemes: {phonemes_cotovia}")

    print(f"\n--- Getting graphemes for '{text1}' (GraphemePhonemizer) ---")
    graphemes1 = grapheme_ph.phonemize(text_gl, lang)
    print(f"  Graphemes: {graphemes1}")
