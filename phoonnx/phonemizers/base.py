import abc
import re
import string
from typing import List, Tuple, Optional

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

