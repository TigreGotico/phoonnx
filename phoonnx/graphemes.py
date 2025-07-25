import logging
import os
import platform
import re
import subprocess
from typing import List, Optional

_LOGGER = logging.getLogger(__name__)

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")


class Graphemes:
    """
    Manages character vocabulary and provides utilities for converting text to token IDs.

    Characters are ordered as follows ```[PAD, EOS, BOS, BLANK, CHARACTERS, PUNCTUATIONS]```.

    Args:
        characters (str):
            Main set of characters to be used in the vocabulary.

        punctuations (str):
            Characters to be treated as punctuation.

        pad (str):
            Special padding character that would be ignored by the model.

        eos (str):
            End of the sentence character.

        bos (str):
            Beginning of the sentence character.

        blank (str):
            Optional character used between characters by some models for better prosody.

        is_unique (bool):
            Remove duplicates from the provided characters. Defaults to True.

        is_sorted (bool):
            Sort the characters in alphabetical order. Only applies to `self.characters`. Defaults to True.
    """

    def __init__(
            self,
            characters: str = None,
            punctuations: str = None,
            pad: str = None,
            eos: str = None,
            bos: str = None,
            blank: str = "<BLNK>",
            is_unique: bool = False,
            is_sorted: bool = True,
    ) -> None:
        """
        Initializes a Graphemes instance with character sets and vocabulary configuration.
        """
        self._characters = characters
        self._punctuations = punctuations
        self._pad = pad
        self._eos = eos
        self._bos = bos
        self._blank = blank
        self.is_unique = is_unique
        self.is_sorted = is_sorted
        self._create_vocab()

    @property
    def pad_id(self) -> int:
        """Returns the ID for the padding character."""
        return self.char_to_id(self.pad) if self.pad else len(self.vocab)

    @property
    def blank_id(self) -> int:
        """Returns the ID of the blank token in the vocabulary."""
        return self.char_to_id(self.blank) if self.blank else len(self.vocab)

    @property
    def eos_id(self) -> int:
        """Returns the ID for the end-of-sentence (EOS) token."""
        return self.char_to_id(self.eos) if self.eos else len(self.vocab)

    @property
    def bos_id(self) -> int:
        """Returns the ID for the beginning-of-sequence (BOS) token."""
        return self.char_to_id(self.bos) if self.bos else len(self.vocab)

    @property
    def characters(self):
        """Get the set of characters in the vocabulary."""
        return self._characters

    @characters.setter
    def characters(self, characters):
        """Set the characters for the vocabulary and regenerate the vocabulary."""
        self._characters = characters
        self._create_vocab()

    @property
    def punctuations(self):
        """Get the set of punctuation characters used in the vocabulary."""
        return self._punctuations

    @punctuations.setter
    def punctuations(self, punctuations):
        """Set the punctuation characters for the vocabulary and recreate the vocabulary."""
        self._punctuations = punctuations
        self._create_vocab()

    @property
    def pad(self):
        """Returns the padding token ID for the grapheme vocabulary."""
        return self._pad

    @pad.setter
    def pad(self, pad):
        """Set the padding character and recreate the vocabulary."""
        self._pad = pad
        self._create_vocab()

    @property
    def eos(self):
        """Returns the end-of-sentence (EOS) token ID."""
        return self._eos

    @eos.setter
    def eos(self, eos):
        """Set the end-of-sentence (EOS) token and recreate the vocabulary."""
        self._eos = eos
        self._create_vocab()

    @property
    def bos(self):
        """Returns the beginning-of-sequence (BOS) token ID."""
        return self._bos

    @bos.setter
    def bos(self, bos):
        """Set the beginning-of-sentence (BOS) token and recreate the vocabulary."""
        self._bos = bos
        self._create_vocab()

    @property
    def blank(self):
        """Returns the blank token used in the grapheme vocabulary."""
        return self._blank

    @blank.setter
    def blank(self, blank):
        """Set the blank token and recreate the vocabulary."""
        self._blank = blank
        self._create_vocab()

    @property
    def vocab(self):
        """Returns the vocabulary dictionary mapping characters to their unique integer IDs."""
        return self._vocab

    @vocab.setter
    def vocab(self, vocab):
        """Create vocabulary mappings from a given list of characters."""
        self._vocab = vocab
        self._char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self._id_to_char = {
            idx: char for idx, char in enumerate(self.vocab)
        }

    @property
    def num_chars(self):
        """Returns the total number of characters in the vocabulary."""
        return len(self._vocab)

    def _create_vocab(self):
        """Create the vocabulary and character-to-ID mappings for the Graphemes class."""
        characters = self._characters if self._characters is not None else ""
        punctuations = self._punctuations if self._punctuations is not None else ""

        self._vocab = []
        if self._pad is not None:
            self._vocab.append(self._pad)
        self._vocab.extend(list(punctuations))
        self._vocab.extend(list(characters))
        if self._blank is not None:
            self._vocab.append(self._blank)

        self._char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self._id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    def char_to_id(self, char: str) -> int:
        """Convert a character to its corresponding integer ID in the vocabulary."""
        try:
            return self._char_to_id[char]
        except KeyError as e:
            raise KeyError(f" [!] {repr(char)} is not in the vocabulary.") from e

    def id_to_char(self, idx: int) -> str:
        """Convert a token ID to its corresponding character."""
        return self._id_to_char[idx]

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize input text by applying a series of transformations.
        This is a static method now part of Graphemes, previously in TTSTokenizer.
        """
        text = text.lower()
        text = text.replace(";", ",")
        text = text.replace("-", " ")
        text = text.replace(":", ",")
        text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
        text = re.sub(_whitespace_re, " ", text).strip()
        return text

    @staticmethod
    def _intersperse_blank_char(char_sequence: List[int], blank_id: int) -> List[int]:
        """
        Intersperses the blank character between characters in a sequence.
        This is a static method now part of Graphemes, previously in TTSTokenizer.
        """
        result = [blank_id] * (len(char_sequence) * 2 + 1)
        result[1::2] = char_sequence
        return result

    @staticmethod
    def _pad_with_bos_eos(char_sequence: List[int], bos_id: int, eos_id: int) -> List[int]:
        """
        Pad a character sequence with beginning-of-sequence (BOS) and end-of-sequence (EOS) tokens.
        This is a static method now part of Graphemes, previously in TTSTokenizer.
        """
        return [bos_id] + list(char_sequence) + [eos_id]

    def text_to_ids(self, text: str, add_blank: bool = False, use_eos_bos: bool = False) -> List[int]:
        """
        Convert text to a sequence of token IDs with optional preprocessing and special token handling.

        This method integrates the functionality previously found in TTSTokenizer.

        Parameters:
            text (str): Input text to be converted to token IDs.
            add_blank (bool): Whether to intersperse blank tokens between character tokens.
            use_eos_bos (bool): Whether to add beginning and end of sequence tokens.

        Returns:
            List[int]: A sequence of token IDs after applying configured transformations.
        """
        # Apply normalization
        text = self._normalize_text(text)

        # Encode characters to IDs
        token_ids = []
        not_found_characters = []  # Local tracking for this call
        for char in text:
            try:
                idx = self.char_to_id(char)
                token_ids.append(idx)
            except KeyError:
                if char not in not_found_characters:
                    not_found_characters.append(char)
                    _LOGGER.debug(text)
                    _LOGGER.warning(f" [!] Character {repr(char)} not found in the vocabulary. Discarding it.")

        if add_blank:
            token_ids = self._intersperse_blank_char(token_ids, self.blank_id)
        if use_eos_bos:
            token_ids = self._pad_with_bos_eos(token_ids, self.bos_id, self.eos_id)

        return token_ids


class CotoviaError(Exception):
    """Custom exception for espeak-ng related errors."""
    pass


# NOTE: cotovia phonemizer is used as a pre-processing step for text since it does not output IPA
class CotoviaGraphemes:
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

    def phonemize(self, sentence: str) -> str:
        """
        Converts a given sentence into phonemes using the Cotovia TTS binary.

        Processes the input sentence through a command-line phonemization tool, applying multiple regular expression transformations to clean and normalize the phonetic representation.

        Parameters:
            sentence (str): The input text to be phonemized

        Returns:
            str: A cleaned and normalized phonetic representation of the input sentence

        Notes:
            - Uses subprocess to execute the Cotovia TTS binary
            - Applies multiple regex substitutions to improve punctuation and spacing
            - Converts text from ISO-8859-1 to UTF-8 encoding
        """
        cmd = f'echo "{sentence}" | {self.cotovia_bin} -t -n -S | iconv -f iso88591 -t utf8'
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
        str_ext = re.sub(r"(\w+)\s+-([^-]*?)-\s+([^-]*?)", r"\1, \2, ", str_ext)

        # substitute ' - ' by ', '
        str_ext = re.sub(r"(\w+[!\?]?)\s+-\s*", r"\1, ", str_ext)

        # substitute ' ( text )' to ', text,'
        str_ext = re.sub(r"(\w+)\s*\(\s*([^\(\)]*?)\s*\)", r"\1, \2,", str_ext)

        return str_ext


if __name__ == "__main__":
    coto = CotoviaGraphemes()
    print(coto.find_cotovia())

    text = "Este é un sistema de conversión de texto a voz en lingua galega baseado en redes neuronais artificiais. Ten en conta que as funcionalidades incluídas nesta páxina ofrécense unicamente con fins de demostración. Se tes algún comentario, suxestión ou detectas algún problema durante a demostración, ponte en contacto connosco."

    print(coto.phonemize(text))