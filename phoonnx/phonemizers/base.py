import abc
import re
import string
import unicodedata
from typing import List, Tuple, Optional, Literal

from langcodes import tag_distance
from quebra_frases import sentence_tokenize
from phoonnx.config import Alphabet
from phoonnx.util import normalize
from phoonnx.thirdparty.phonikud import PhonikudDiacritizer
from phoonnx.thirdparty.tashkeel import TashkeelDiacritizer

# list of (substring, terminator, end_of_sentence) tuples.
TextChunks = List[Tuple[str, str, bool]]
# list of (phonemes, terminator, end_of_sentence) tuples.
RawPhonemizedChunks = List[Tuple[str, str, bool]]

PhonemizedChunks = list[list[str]]


class BasePhonemizer(metaclass=abc.ABCMeta):
    def __init__(self, alphabet: Alphabet = Alphabet.UNICODE,
                 taskeen_threshold: Optional[float] = 0.8):
        super().__init__()
        self.alphabet = alphabet

        self.taskeen_threshold = taskeen_threshold  # arabic only
        self._tashkeel: Optional[TashkeelDiacritizer] = None
        self._phonikud: Optional[PhonikudDiacritizer] = None # hebrew only

    @property
    def phonikud(self) -> PhonikudDiacritizer:
        if self._phonikud is None:
            self._phonikud = PhonikudDiacritizer()
        return self._phonikud

    @property
    def tashkeel(self) -> TashkeelDiacritizer:
        if self._tashkeel is None:
            self._tashkeel = TashkeelDiacritizer()
        return self._tashkeel

    @abc.abstractmethod
    def phonemize_string(self, text: str, lang: str) -> str:
        raise NotImplementedError

    def phonemize_to_list(self, text: str, lang: str) -> List[str]:
        return list(self.phonemize_string(text, lang))

    def add_diacritics(self, text: str, lang: str) -> str:
        if lang.startswith("he"):
            return self.phonikud.diacritize(text)
        elif lang.startswith("ar"):
            return self.tashkeel.diacritize(text, self.taskeen_threshold)
        return text

    def phonemize(self, text: str, lang: str) -> PhonemizedChunks:
        if not text:
            return [('', '', True)]
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


### all the 3 below are essentially the same thing
# no phonemization really happens

class RawPhonemes(BasePhonemizer):
    """no phonemization, text is phonemes already"""

    def phonemize_string(self, text: str, lang: str) -> str:
        return text


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
        return unicodedata.normalize(self.form, text)


if __name__ == "__main__":
    raw = RawPhonemes()
    grap = GraphemePhonemizer()
    uni = UnicodeCodepointPhonemizer()

    text = "olá, quem são vocês?"
    lang = "pt"
    print(raw.phonemize(text, lang))
    print(grap.phonemize(text, lang))
    print(uni.phonemize(text, lang))

    print(raw.phonemize_string(text, lang))
    print(grap.phonemize_string(text, lang))
    print(uni.phonemize_string(text, lang))

    print(raw.phonemize_to_list(text, lang))
    print(grap.phonemize_to_list(text, lang))
    print(uni.phonemize_to_list(text, lang))
