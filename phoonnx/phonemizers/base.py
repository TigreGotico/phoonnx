import abc
import re
import string
import unicodedata
from typing import List, Tuple, Optional, Literal

from langcodes import tag_distance
from quebra_frases import sentence_tokenize
from phoonnx.config import Alphabet
from phoonnx.util import normalize, match_lang
from phoonnx.thirdparty.phonikud import PhonikudDiacritizer

# list of (substring, terminator, end_of_sentence) tuples.
TextChunks = List[Tuple[str, str, bool]]
# list of (phonemes, terminator, end_of_sentence) tuples.
RawPhonemizedChunks = List[Tuple[str, str, bool]]

PhonemizedChunks = list[list[str]]


class BasePhonemizer(metaclass=abc.ABCMeta):
    def __init__(self, alphabet: Alphabet = Alphabet.UNICODE,
                 diacritizer_model: str = "rawi-ensemble"):
        super().__init__()
        self.alphabet = alphabet

        # diacritizer model name, for languages that need one. Arabic uses
        # text2tashkeel; the default "rawi-ensemble" restores hamza and the dagger
        # alef in addition to the standard marks.
        self.diacritizer_model = diacritizer_model
        self._phonikud: Optional[PhonikudDiacritizer] = None # hebrew only
        self._tashkeel: dict = {}  # model name -> text2tashkeel Diacritizer

    @property
    def phonikud(self) -> PhonikudDiacritizer:
        if self._phonikud is None:
            self._phonikud = PhonikudDiacritizer()
        return self._phonikud

    def tashkeel(self, model: Optional[str] = None):
        """Lazily build (and cache) the text2tashkeel Diacritizer used for Arabic.

        text2tashkeel is a dependency of the ``[ar]`` extra; it restores hamza and the
        dagger alef in addition to the standard marks. Install with
        ``pip install phoonnx[ar]`` (or ``pip install text2tashkeel``)."""
        model = model or self.diacritizer_model
        if model not in self._tashkeel:
            try:
                from text2tashkeel import Diacritizer
            except ImportError as e:
                raise ImportError(
                    "Arabic diacritization requires the text2tashkeel package: "
                    "pip install phoonnx[ar]  (or pip install text2tashkeel)"
                ) from e
            self._tashkeel[model] = Diacritizer(model)
        return self._tashkeel[model]

    @abc.abstractmethod
    def phonemize_string(self, text: str, lang: str) -> str:
        raise NotImplementedError

    def phonemize_to_list(self, text: str, lang: str) -> List[str]:
        return list(self.phonemize_string(text, lang))

    def add_diacritics(self, text: str, lang: str,
                       model: Optional[str] = None) -> str:
        if lang.startswith("he"):
            return self.phonikud.diacritize(text)
        elif lang.startswith("ar"):
            return self.tashkeel(model).diacritize(text)
        return text

    def phonemize(self, text: str, lang: str) -> PhonemizedChunks:
        if not text:
            # PhonemizedChunks is list[list[str]]; empty text yields no
            # sentences. (Returning the raw (str, str, bool) tuple form here
            # corrupted the type and broke callers that mutate each sentence,
            # e.g. inline ``[[phoneme]]`` blocks in TTSVoice.phonemize.)
            return []
        results: RawPhonemizedChunks = []
        text = normalize(text, lang)
        for chunk, punct, eos in self.chunk_text(text):
            phoneme_str = self.phonemize_string(self.remove_punctuation(chunk), lang)
            results += [(phoneme_str, punct, True)]
        return self._process_phones(results)

    @staticmethod
    def _process_phones(raw_phones: RawPhonemizedChunks) -> PhonemizedChunks:
        """Text to phonemes grouped by sentence."""
        all_phonemes: list[list[str]] = []
        sentence_phonemes: list[str] = []
        for phonemes_str, terminator_str, end_of_sentence in raw_phones:
            # Filter out (lang) switch (flags).
            # These surround words from languages other than the current voice.
            phonemes_str = re.sub(r"\([^)]+\)", "", phonemes_str)
            sentence_phonemes.extend(list(phonemes_str))
            if end_of_sentence:
                all_phonemes.append(sentence_phonemes)
                sentence_phonemes = []
        if sentence_phonemes:
            all_phonemes.append(sentence_phonemes)
        return all_phonemes

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
        lang, score = match_lang(target_lang, valid_langs)
        if score > 10:
            # raise an error for unsupported language
            raise ValueError(f"unsupported language code: {target_lang}")
        return lang

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
        """
        Split input text into sentence-aware chunks using sentence tokenization and optional intra-sentence delimiters.

        Parameters:
            text (str): Input text to split. If empty, returns a single empty chunk.
            delimiters (Optional[List[str]]): List of substring delimiters to split sentences by (defaults to [":", ";", "...", "|"]).

        Returns:
            TextChunks: A list of tuples (substring, terminator, end_of_sentence) where:
                - `substring` is the chunk text with surrounding whitespace removed,
                - `terminator` is the delimiter that followed the substring or the sentence-final punctuation if none matched,
                - `end_of_sentence` is `True` for the final chunk of a tokenized sentence, `False` otherwise.
        """
        if not text:
            return [('', '', True)]

        results: TextChunks = []
        delimiters = delimiters or [":", ";", "...", "|"]

        # Create a regex pattern that matches any of the delimiters
        delimiter_pattern = re.escape(delimiters[0])
        for delimiter in delimiters[1:]:
            delimiter_pattern += f"|{re.escape(delimiter)}"

        for sentence in sentence_tokenize(text):
            # Default punctuation if no specific punctuation found
            default_punc = sentence[-1] if sentence and sentence[-1] in string.punctuation else "."

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


class UnicodeCodepointPhonemizer(BasePhonemizer):
    """Phonemes = codepoints
    normalization also splits accents and punctuation into it's own codepoints
    """

    def __init__(self, form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFD"):
        self.form = form
        super().__init__(Alphabet.UNICODE)

    def phonemize_string(self, text: str, lang: str) -> str:
        # Phonemes = codepoints
        """
        Normalize the input text to Unicode NFD so phonemes correspond to individual Unicode codepoints.

        Parameters:
            text (str): The input string to normalize.
            lang (str): Language code (ignored by this implementation).

        Returns:
            str: The input text normalized to Unicode NFD, with combining marks separated into distinct codepoints.
        """
        return unicodedata.normalize(self.form, text)


if __name__ == "__main__":
    grap = GraphemePhonemizer()
    uni = UnicodeCodepointPhonemizer()

    text = "olá, quem são vocês?"
    lang = "pt"
    print(grap.phonemize(text, lang))
    print(uni.phonemize(text, lang))

    print(grap.phonemize_string(text, lang))
    print(uni.phonemize_string(text, lang))

    print(grap.phonemize_to_list(text, lang))
    print(uni.phonemize_to_list(text, lang))