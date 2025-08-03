import os
import platform
import re
import subprocess
from typing import Optional

from phoonnx.phonemizers.base import BasePhonemizer
from phoonnx.config import Alphabet

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
        super().__init__(Alphabet.COTOVIA)

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
        # this check is here only to throw an exception if invalid language is provided
        return cls.match_lang(target_lang, ["gl-ES"])

    @staticmethod
    def find_cotovia() -> str:
        """
        Attempts to find the cotovia binary in common locations.
        """
        path = subprocess.run(["which", "cotovia"], capture_output=True, text=True).stdout.strip()
        if path and os.path.isfile(path):
            return path

        # Fallback to bundled binaries
        local_path = f"{os.path.dirname(os.path.dirname(__file__))}/thirdparty/cotovia/cotovia_{platform.machine()}"
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
        lang = self.get_lang(lang)
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



if __name__ == "__main__":

    cotovia = CotoviaPhonemizer()

    lang = "gl"
    text_gl = "Este é un sistema de conversión de texto a voz en lingua galega baseado en redes neuronais artificiais. Ten en conta que as funcionalidades incluídas nesta páxina ofrécense unicamente con fins de demostración. Se tes algún comentario, suxestión ou detectas algún problema durante a demostración, ponte en contacto connosco."
    print(f"\n--- Getting phonemes for '{text_gl}' (Cotovia) ---")
    phonemes_cotovia = cotovia.phonemize(text_gl, lang)
    print(f"  Cotovia Phonemes: {phonemes_cotovia}")
