import json
import os
import re
import subprocess
from typing import List, Tuple, Dict, Optional

import numpy as np
import onnxruntime
import requests
from langcodes import closest_match

# Convert input text to a list of (phonemes, terminator, end_of_sentence) tuples.
RawPhonemizedChunks = List[Tuple[str, str, bool]]


class EspeakError(Exception):
    """Custom exception for espeak-ng related errors."""
    pass


class ByT5Phonemizer:
    """
    A phonemizer class that uses a ByT5 ONNX model to convert text into phonemes.
    It mimics the clause-by-clause segmentation behavior of the piper TTS implementation
    """
    TOKENIZER_CONFIG_URL = "https://huggingface.co/OpenVoiceOS/g2p-multilingual-byt5-tiny-8l-ipa-childes-onnx/resolve/main/tokenizer_config.json"
    MODEL_URL = "https://huggingface.co/OpenVoiceOS/g2p-multilingual-byt5-tiny-8l-ipa-childes-onnx/resolve/main/byt5_g2p_model.onnx"

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

    @staticmethod
    def get_lang(target_lang: str) -> str:
        """
        Validates and returns the closest supported language code.

        Args:
            target_lang (str): The language code to validate.

        Returns:
            str: The validated language code.

        Raises:
            ValueError: If the language code is unsupported.
        """
        # Define the list of supported language codes
        LANGS = "ca,cy,da,de,en-na,en-uk,es,et,eu,fa,fr,ga,hr,hu,id,is,it,ja,ko,nl,no,pl,pt,pt-br,qu,ro,sr,sv,tr,zh,zh-yue".split(
            ",")
        # Find the closest match and its score
        lang, score = closest_match(target_lang, LANGS)
        # If the score is low (meaning a good match), return the language
        if score <= 10:
            return lang
        # Otherwise, raise an error for unsupported language
        raise ValueError(f"unsupported language code: {target_lang}")

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

    def phonemize(self, text: str, lang: str) -> RawPhonemizedChunks:
        """
        Generates phonemes for the given text using the ByT5 ONNX model.
        This function attempts to mimic the clause-by-clause segmentation
        behavior of the C extension by heuristically splitting the input text
        based on common punctuation marks.

        Args:
            text (str): The input text to convert to phonemes.
            lang (str): The language of the input text.

        Returns:
            list: A list of tuples, where each tuple contains:
                  (phonemes_string, terminator_character, is_sentence_end_boolean).
                  - phonemes_string: The IPA phonemes for the text segment.
                  - terminator_character: The punctuation mark that ended the segment (e.g., '.', ',').
                                          Empty string if the segment is the end of the input text
                                          and does not end with a recognized terminator.
                  - is_sentence_end_boolean: True if the segment ends with a sentence-ending
                                             punctuation (., ?, !), False otherwise.
        """
        if not text:
            return [('', '', True)]

        # Regex to split text by common sentence/clause terminators, keeping the delimiters.
        # This creates a list like ['text_part_1', 'delimiter_1', 'text_part_2', 'delimiter_2', ..., 'text_part_N', '']
        # The last empty string handles cases where the text ends with a delimiter.
        # We use re.split with a capturing group for the delimiter.
        parts: List[str] = re.split(r'([.?!,:;])', text)

        results: RawPhonemizedChunks = []
        current_segment_buffer: List[str] = []  # Buffer to accumulate text parts before a delimiter

        # Determine if the last part of the original text ends with punctuation
        # This helps in deciding the 'is_sentence_end' for the final segment
        ends_with_punctuation: bool = bool(text and re.search(r'[.?!,:;]\s*$', text))

        for i, part in enumerate(parts):
            if part is None:
                continue  # Should not happen with re.split, but for safety

            # Strip leading/trailing whitespace from the part
            stripped_part: str = part.strip()

            if re.match(r'[.?!,:;]', stripped_part):  # If the stripped_part is a delimiter
                terminator_str: str = stripped_part
                is_sentence_end: bool = (terminator_str in ['.', '?', '!'])

                # Join the buffered text parts to form the segment text
                segment_text: str = " ".join(current_segment_buffer).strip()

                if segment_text:
                    # Get phonemes for the accumulated text before this delimiter
                    phonemes: str = self._infer_onnx(segment_text, lang)
                    results.append((phonemes, terminator_str, is_sentence_end))
                else:
                    # If the segment_text is empty, it means we encountered consecutive delimiters
                    # or a delimiter at the very beginning.
                    # In such cases, we'll get phonemes for the punctuation itself.
                    phonemes_for_punc: str = self._infer_onnx(terminator_str, lang)
                    if not phonemes_for_punc:
                        phonemes_for_punc = terminator_str  # Fallback if model returns empty for punctuation
                    results.append((phonemes_for_punc, terminator_str, is_sentence_end))

                current_segment_buffer = []  # Reset buffer for the next segment
            else:
                # If the part is text, add it to the buffer if not empty
                if stripped_part:
                    current_segment_buffer.append(stripped_part)

        # Handle any remaining text in the buffer that didn't end with a delimiter
        if current_segment_buffer:
            remaining_text: str = " ".join(current_segment_buffer).strip()
            if remaining_text:
                phonemes: str = self._infer_onnx(remaining_text, lang)

                # For the very last segment without an explicit terminator,
                # assume it's a sentence end to match Piper's behavior.
                is_sentence_end_for_last_segment: bool = not ends_with_punctuation

                results.append((phonemes, "", is_sentence_end_for_last_segment))

        return results


class EspeakPhonemizer:
    """
    A phonemizer class that uses the espeak-ng command-line tool to convert text into phonemes.
    It segments the input text heuristically based on punctuation to mimic clause-by-clause processing.
    """

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

    @classmethod
    def phonemize(cls, text: str, lang: str) -> RawPhonemizedChunks:
        """
        Generates phonemes for the given text using espeak-ng CLI.
        This function attempts to mimic the clause-by-clause segmentation
        behavior of the piper TTS implementation

        Args:
            text (str): The input text to convert to phonemes.
            lang (str): The language code for the text (e.g., "en-gb").

        Returns:
            list: A list of tuples, where each tuple contains:
                  (phonemes_string, terminator_character, is_sentence_end_boolean).
                  - phonemes_string: The IPA phonemes for the text segment.
                  - terminator_character: The punctuation mark that ended the segment (e.g., '.', ',').
                                          Empty string if the segment is the end of the input text
                                          and does not end with a recognized terminator.
                  - is_sentence_end_boolean: True if the segment ends with a sentence-ending
                                             punctuation (., ?, !), False otherwise.

        Raises:
            EspeakError: If espeak-ng is not initialized or if a subprocess call fails.
        """
        if not text:
            return [('', '', True)]

        # Regex to split text by common sentence/clause terminators, keeping the delimiters.
        # This creates a list like ['text_part_1', 'delimiter_1', 'text_part_2', 'delimiter_2', ..., 'text_part_N', '']
        # The last empty string handles cases where the text ends with a delimiter.
        # We use re.split with a capturing group for the delimiter.
        parts: List[str] = re.split(r'([.?!,:;])', text)

        results: RawPhonemizedChunks = []
        current_segment_buffer: List[str] = []  # Buffer to accumulate text parts before a delimiter

        # Determine if the last part of the original text ends with punctuation
        # This helps in deciding the 'is_sentence_end' for the final segment
        ends_with_punctuation: bool = bool(text and re.search(r'[.?!,:;]\s*$', text))

        for i, part in enumerate(parts):
            if part is None:
                continue  # Should not happen with re.split, but for safety

            # Strip leading/trailing whitespace from the part
            stripped_part: str = part.strip()

            if re.match(r'[.?!,:;]', stripped_part):  # If the stripped_part is a delimiter
                terminator_str: str = stripped_part
                is_sentence_end: bool = (terminator_str in ['.', '?', '!'])

                # Join the buffered text parts to form the segment text
                segment_text: str = " ".join(current_segment_buffer).strip()

                if segment_text:
                    # Get phonemes for the accumulated text before this delimiter
                    raw_phonemes_output: str = cls._run_espeak_command(
                        ['-q', '-x', '--ipa', '-v', lang],
                        input_text=segment_text
                    )
                    # Extract only the last line, which should be the pure phonemes
                    phonemes: str = raw_phonemes_output.splitlines()[-1].strip() if raw_phonemes_output else ""
                    results.append((phonemes, terminator_str, is_sentence_end))
                else:
                    # If the segment_text is empty, it means we encountered consecutive delimiters
                    # or a delimiter at the very beginning.
                    # In such cases, we'll use the punctuation itself as phonemes as a fallback.
                    raw_phonemes_output: str = cls._run_espeak_command(
                        ['-q', '-x', '--ipa', '-v', lang],
                        input_text=terminator_str
                    )
                    phonemes_for_punc: str = raw_phonemes_output.splitlines()[-1].strip() if raw_phonemes_output else ""
                    if not phonemes_for_punc:  # Fallback if espeak-ng returns empty or only diagnostic for punctuation
                        phonemes_for_punc = terminator_str
                    results.append((phonemes_for_punc, terminator_str, is_sentence_end))

                current_segment_buffer = []  # Reset buffer for the next segment
            else:
                # If the part is text, add it to the buffer if not empty
                if stripped_part:
                    current_segment_buffer.append(stripped_part)

        # Handle any remaining text in the buffer that didn't end with a delimiter
        if current_segment_buffer:
            remaining_text: str = " ".join(current_segment_buffer).strip()
            if remaining_text:
                raw_phonemes_output: str = cls._run_espeak_command(
                    ['-q', '-x', '--ipa', '-v', lang],
                    input_text=remaining_text
                )
                phonemes: str = raw_phonemes_output.splitlines()[-1].strip() if raw_phonemes_output else ""

                # For the very last segment without an explicit terminator,
                # assume it's a sentence end to match Piper's behavior.
                is_sentence_end_for_last_segment: bool = not ends_with_punctuation

                results.append((phonemes, "", is_sentence_end_for_last_segment))

        return results


if __name__ == "__main__":
    # for comparison

    byt5 = ByT5Phonemizer()
    espeak = EspeakPhonemizer()

    lang = "en-gb"

    print("\n--- Getting phonemes for 'Hello, world. How are you?' ---")
    text1 = "Hello, world. How are you?"
    phonemes1 = espeak.phonemize(text1, lang)
    phonemes1c = byt5.phonemize(text1, lang)
    print(f"  Espeak Phonemes: {phonemes1}")
    print(f"   byt5  Phonemes: {phonemes1c}")

    print("\n--- Getting phonemes for 'This is a test: a quick one; and done!' ---")
    text2 = "This is a test: a quick one; and done!"
    phonemes2 = espeak.phonemize(text2, lang)
    phonemes2c = byt5.phonemize(text2, lang)
    print(f"  Espeak Phonemes: {phonemes2}")
    print(f"   byt5  Phonemes: {phonemes2c}")

    print("\n--- Getting phonemes for 'Just a phrase without punctuation' ---")
    text3 = "Just a phrase without punctuation"
    phonemes3 = espeak.phonemize(text3, lang)
    phonemes3c = byt5.phonemize(text3, lang)
    print(f"  Espeak Phonemes: {phonemes3}")
    print(f"   byt5  Phonemes: {phonemes3c}")

    """
    --- Getting phonemes for 'Hello, world. How are you?' ---
      Espeak Phonemes: [('həlˈəʊ', ',', False), ('wˈɜːld', '.', True), ('hˈaʊ ɑː juː', '?', True)]
       byt5  Phonemes: [('hələʊ', ',', False), ('wɜːld', '.', True), ('haʊ ɑː juː', '?', True)]
    
    --- Getting phonemes for 'This is a test: a quick one; and done!' ---
      Espeak Phonemes: [('ðɪs ɪz ɐ tˈɛst', ':', False), ('ɐ kwˈɪk wˌɒn', ';', False), ('and dˈʌn', '!', True)]
       byt5  Phonemes: [('ðɪs ɪz ɐ tʰestʰ', ':', False), ('ɐ kʰwɪkʰ wɒn', ';', False), ('ænd dʌn', '!', True)]
    
    --- Getting phonemes for 'Just a phrase without punctuation' ---
      Espeak Phonemes: [('dʒˈʌst ɐ fɹˈeɪz wɪðˌaʊt pˌʌŋktʃuːˈeɪʃən', '', True)]
       byt5  Phonemes: [('d̠ʒʌstʰ ɐ fɹeɪz wɪðaʊtʰ pʰʌŋkʰt̠ʃuːeɪʃən', '', True)]
    """
