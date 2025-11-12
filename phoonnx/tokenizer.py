from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, List, Dict, Optional, Any, Set, Union

from phoonnx.util import LOG


class BlankBetween(str, Enum):
    """Placement of blank tokens"""

    TOKENS = "tokens"
    """Blank between every token/phoneme"""

    WORDS = "words"
    """Blank between every word"""

    TOKENS_AND_WORDS = "tokens_and_words"
    """Blank between every token/phoneme and every word (may be different symbols)"""


PHONEME_ID_LIST = List[int]
PHONEME_ID_MAP = Dict[str, int]
PHONEME_LIST = List[str]
PHONEME_WORD_LIST = List[PHONEME_LIST]

DEFAULT_IPA_PHONEME_ID_MAP: Dict[str, PHONEME_ID_LIST] = {
    "_": [0],
    "^": [1],
    "$": [2],
    " ": [3],
    "!": [4],
    "'": [5],
    "(": [6],
    ")": [7],
    ",": [8],
    "-": [9],
    ".": [10],
    ":": [11],
    ";": [12],
    "?": [13],
    "a": [14],
    "b": [15],
    "c": [16],
    "d": [17],
    "e": [18],
    "f": [19],
    "h": [20],
    "i": [21],
    "j": [22],
    "k": [23],
    "l": [24],
    "m": [25],
    "n": [26],
    "o": [27],
    "p": [28],
    "q": [29],
    "r": [30],
    "s": [31],
    "t": [32],
    "u": [33],
    "v": [34],
    "w": [35],
    "x": [36],
    "y": [37],
    "z": [38],
    "æ": [39],
    "ç": [40],
    "ð": [41],
    "ø": [42],
    "ħ": [43],
    "ŋ": [44],
    "œ": [45],
    "ǀ": [46],
    "ǁ": [47],
    "ǂ": [48],
    "ǃ": [49],
    "ɐ": [50],
    "ɑ": [51],
    "ɒ": [52],
    "ɓ": [53],
    "ɔ": [54],
    "ɕ": [55],
    "ɖ": [56],
    "ɗ": [57],
    "ɘ": [58],
    "ə": [59],
    "ɚ": [60],
    "ɛ": [61],
    "ɜ": [62],
    "ɞ": [63],
    "ɟ": [64],
    "ɠ": [65],
    "ɡ": [66],
    "ɢ": [67],
    "ɣ": [68],
    "ɤ": [69],
    "ɥ": [70],
    "ɦ": [71],
    "ɧ": [72],
    "ɨ": [73],
    "ɪ": [74],
    "ɫ": [75],
    "ɬ": [76],
    "ɭ": [77],
    "ɮ": [78],
    "ɯ": [79],
    "ɰ": [80],
    "ɱ": [81],
    "ɲ": [82],
    "ɳ": [83],
    "ɴ": [84],
    "ɵ": [85],
    "ɶ": [86],
    "ɸ": [87],
    "ɹ": [88],
    "ɺ": [89],
    "ɻ": [90],
    "ɽ": [91],
    "ɾ": [92],
    "ʀ": [93],
    "ʁ": [94],
    "ʂ": [95],
    "ʃ": [96],
    "ʄ": [97],
    "ʈ": [98],
    "ʉ": [99],
    "ʊ": [100],
    "ʋ": [101],
    "ʌ": [102],
    "ʍ": [103],
    "ʎ": [104],
    "ʏ": [105],
    "ʐ": [106],
    "ʑ": [107],
    "ʒ": [108],
    "ʔ": [109],
    "ʕ": [110],
    "ʘ": [111],
    "ʙ": [112],
    "ʛ": [113],
    "ʜ": [114],
    "ʝ": [115],
    "ʟ": [116],
    "ʡ": [117],
    "ʢ": [118],
    "ʲ": [119],
    "ˈ": [120],
    "ˌ": [121],
    "ː": [122],
    "ˑ": [123],
    "˞": [124],
    "β": [125],
    "θ": [126],
    "χ": [127],
    "ᵻ": [128],
    "ⱱ": [129],
    "0": [130],
    "1": [131],
    "2": [132],
    "3": [133],
    "4": [134],
    "5": [135],
    "6": [136],
    "7": [137],
    "8": [138],
    "9": [139],
    "̧": [140],
    "̃": [141],
    "̪": [142],
    "̯": [143],
    "̩": [144],
    "ʰ": [145],
    "ˤ": [146],
    "ε": [147],
    "↓": [148],
    "#": [149],
    '"': [150],
    "↑": [151],
    "̺": [152],
    "̻": [153],
    "g": [154],
    "ʦ": [155],
    "X": [156],
    "̝": [157],
    "̊": [158],
    "ɝ": [159],
    "ʷ": [160],
}

DEFAULT_PAD_TOKEN = DEFAULT_BLANK_TOKEN = "_"  # padding (0)
DEFAULT_BOS_TOKEN = "^"  # beginning of sentence
DEFAULT_EOS_TOKEN = "$"  # end of sentence
DEFAULT_BLANK_WORD_TOKEN = " "  # padding between words

STRESS: Set[str] = {"ˈ", "ˌ"}

PUNCTUATION_MAP: Mapping[str, str] = {";": ",", ":": ",", "?": ".", "!": "."}
"""Default punctuation simplification into short (,) and long (.) pauses"""


@dataclass
class Vocabulary:
    """
    A dataclass to store the mapping between characters/phonemes and their integer IDs,
    along with special tokens used in Text-to-Speech (TTS) models.
    """
    char2idx: Dict[str, int]
    pad: Optional[str] = None
    eos: Optional[str] = None
    bos: Optional[str] = None
    blank: Optional[str] = None
    blank_word: Optional[str] = None

    @staticmethod
    def from_phoonnx_config(cfg: Dict[str, Any]) -> 'Vocabulary':
        """
        Creates a Vocabulary instance from a phoonnx configuration dictionary.

        Parameters:
            cfg: The phoonnx configuration dictionary.

        Returns:
            A Vocabulary instance.
        """
        char2idx: Dict[str, int] = cfg.get("phoneme_id_map", {})
        pad: Optional[str] = cfg.get("pad") or DEFAULT_PAD_TOKEN
        eos: Optional[str] = cfg.get("eos") or DEFAULT_EOS_TOKEN
        bos: Optional[str] = cfg.get("bos") or DEFAULT_BOS_TOKEN
        blank: Optional[str] = cfg.get("blank") or DEFAULT_BLANK_TOKEN
        return Vocabulary(char2idx=char2idx, pad=pad, eos=eos, bos=bos, blank=blank)

    @staticmethod
    def from_piper_config(cfg: Dict[str, Any]) -> 'Vocabulary':
        """
        Creates a Vocabulary instance from a Piper configuration dictionary.

        The Piper config format assumes `phoneme_id_map` values are lists where the ID is the first element.

        Parameters:
            cfg: The Piper configuration dictionary.

        Returns:
            A Vocabulary instance.
        """
        # Piper format has value as list of [id, ...]
        char2idx: Dict[str, int] = {char: idx[0] for char, idx in cfg.get("phoneme_id_map", {}).items()}
        pad: Optional[str] = cfg.get("pad") or DEFAULT_PAD_TOKEN
        eos: Optional[str] = cfg.get("eos") or DEFAULT_EOS_TOKEN
        bos: Optional[str] = cfg.get("bos") or DEFAULT_BOS_TOKEN
        blank: Optional[str] = cfg.get("blank") or DEFAULT_BLANK_TOKEN
        return Vocabulary(char2idx=char2idx, pad=pad, eos=eos, bos=bos, blank=blank)

    @staticmethod
    def from_mimic3_config(cfg: Dict[str, Any], tokens_txt: str) -> 'Vocabulary':
        """
        Creates a Vocabulary instance from a Mimic3 configuration dictionary and a tokens.txt content.

        Parameters:
            cfg: The Mimic3 configuration dictionary.
            tokens_txt: The content of the tokens.txt file, mapping IDs to tokens.

        Returns:
            A Vocabulary instance.
        """
        voc: 'Vocabulary' = Vocabulary.from_tokens_txt(tokens_txt)
        voc.pad = cfg.get("phonemes", {}).get("pad") or cfg.get("phonemes", {}).get("phoneme_separator")
        voc.eos = cfg.get("phonemes", {}).get("eos")
        voc.bos = cfg.get("phonemes", {}).get("bos")
        voc.blank = cfg.get("phonemes", {}).get("blank")
        voc.blank_word = cfg.get("phonemes", {}).get("blank_word") or cfg.get("phonemes", {}).get("word_separator")
        return voc

    @staticmethod
    def from_tokens_txt(tokens_txt: str) -> 'Vocabulary':
        """
        Creates a Vocabulary instance by parsing a tokens.txt style string (ID token per line).

        Parameters:
            tokens_txt: A string where each line is formatted as "ID token" (e.g., "0 <pad>").

        Returns:
            A Vocabulary instance with character-to-index mapping.
        """
        char2idx: Dict[str, int] = {}
        for line in tokens_txt.split("\n"):
            try:
                idx_str, token = line.split(" ", 1)
                char2idx[token] = int(idx_str)
            except ValueError:
                # Skip empty lines or malformed lines
                pass
        return Vocabulary(char2idx=char2idx)

    @staticmethod
    def from_coqui_config(cfg: Dict[str, Any]) -> 'Vocabulary':
        """
        Creates a Vocabulary instance from a Coqui TTS configuration dictionary.

        This method handles different character class formats used in Coqui (VitsCharacters, Graphemes).

        Parameters:
            cfg: The Coqui configuration dictionary.

        Returns:
            A Vocabulary instance.

        Raises:
            ValueError: If an unsupported Coqui tokenizer class is found.
        """
        characters_cfg: Dict[str, Any] = cfg.get("characters", {})
        pad: Optional[str] = characters_cfg.get("pad")
        eos: Optional[str] = characters_cfg.get("eos")
        bos: Optional[str] = characters_cfg.get("bos")
        blank: Optional[str] = characters_cfg.get("blank")
        punctuations: Optional[str] = characters_cfg.get("punctuations")
        characters: Optional[str] = characters_cfg.get("characters")
        clazz: str = characters_cfg.get("characters_class", "N/A")
        sort: bool = characters_cfg.get("is_sorted", False)
        unique: bool = characters_cfg.get("is_unique", False)
        vocab: List[str]

        if clazz == "TTS.tts.models.vits.VitsCharacters":
            vocab = list(punctuations) + list(characters)
            if pad:
                vocab.insert(0, pad)
            if cfg.get("add_blank"):
                blank = blank or "<BLNK>"
            if blank:
                vocab.append(blank)
        elif clazz == "TTS.tts.utils.text.characters.Graphemes":
            vocab = list(characters)
            if unique:
                # NOTE: deduplication in coqui does not preserve order
                # MUST be used together with is_sorted
                vocab = list(set(vocab))
            if sort:
                vocab = sorted(vocab)
            vocab = [blank, *vocab] if blank is not None and len(blank) > 0 else vocab
            vocab = [bos, *vocab] if bos is not None and len(bos) > 0 else vocab
            vocab = [eos, *vocab] if eos is not None and len(eos) > 0 else vocab
            vocab = [pad, *vocab] if pad is not None and len(pad) > 0 else vocab
            vocab = vocab + list(punctuations)
        else:
            raise ValueError(f"unsupported coqui tokenizer: {clazz}")

        return Vocabulary(char2idx={char: idx for idx, char in enumerate(vocab)},
                          pad=pad,
                          eos=eos,
                          bos=bos,
                          blank=blank)

    @property
    def idx2char(self) -> Dict[int, str]:
        """Returns the inverse mapping of ID to character."""
        return {idx: char for char, idx in self.char2idx.items()}

    @property
    def pad_id(self) -> Optional[int]:
        """
        Returns the ID for the **padding character**.

        If a padding character is defined, returns its corresponding ID from the vocabulary.
        Returns `None` if the padding token is not defined or not in the vocabulary.

        Returns:
            Optional[int]: The ID of the padding character or None.
        """
        return self.char2idx.get(self.pad) if self.pad else None

    @property
    def blank_id(self) -> Optional[int]:
        """
        Returns the ID of the **blank token** (inter-phoneme blank) in the vocabulary.

        Returns `None` if the blank token is not defined or not in the vocabulary.

        Returns:
            Optional[int]: The ID of the blank token or None.
        """
        return self.char2idx.get(self.blank) if self.blank else None

    @property
    def blank_word_id(self) -> Optional[int]:
        """
        Returns the ID of the **word-level blank token** (separator between words) in the vocabulary.

        Returns `None` if the word blank token is not defined or not in the vocabulary.

        Returns:
            Optional[int]: The ID of the word blank token or None.
        """
        return self.char2idx.get(self.blank_word) if self.blank_word else None

    @property
    def eos_id(self) -> Optional[int]:
        """
        Returns the ID for the **end-of-sequence (EOS) token**.

        Returns `None` if the EOS token is not defined or not in the vocabulary.

        Returns:
            Optional[int]: The ID of the end-of-sequence token or None.
        """
        return self.char2idx.get(self.eos) if self.eos else None

    @property
    def bos_id(self) -> Optional[int]:
        """
        Returns the ID for the **beginning-of-sequence (BOS) token**.

        Returns `None` if the BOS token is not defined or not in the vocabulary.

        Returns:
            Optional[int]: The vocabulary ID for the beginning-of-sequence token or None.
        """
        return self.char2idx.get(self.bos) if self.bos else None

    @property
    def num_chars(self) -> int:
        """
        Returns the total number of characters in the vocabulary.

        Returns:
            int: The number of unique characters in the vocabulary.
        """
        return len(self.char2idx)


@dataclass
class TTSTokenizer:
    """
    TTS tokenizer to convert input characters or phonemes to token IDs, applying
    special token insertions (BOS/EOS) and blank insertions (inter-phoneme/inter-word).
    """
    vocabulary: Vocabulary
    add_blank_char: bool
    add_blank_word: bool
    use_eos_bos: bool
    blank_at_end: bool
    blank_at_start: bool
    not_found_characters: Set[str] = field(default_factory=set)

    @property
    def pad_id(self) -> Optional[int]:
        """Returns the ID for the padding character from the vocabulary, or None."""
        return self.vocabulary.pad_id

    @property
    def blank_id(self) -> Optional[int]:
        """Returns the ID for the inter-phoneme blank token from the vocabulary, or None."""
        return self.vocabulary.blank_id

    @property
    def blank_word_id(self) -> Optional[int]:
        """Returns the ID for the inter-word blank token from the vocabulary, or None."""
        return self.vocabulary.blank_word_id

    def encode(self, text: Union[str, List[str]]) -> List[int]:
        """
        Encode a string of text into a sequence of token IDs based on the character vocabulary.

        This method converts each character in the input text to its corresponding token ID.
        If `add_blank_word` is enabled and the character is a space (" "), it is mapped to
        the `blank_word_id`. Characters not found in the vocabulary are mapped to `None`
        and logged as a warning for the first occurrence.

        Parameters:
            text (str): The input text to be tokenized.

        Returns:
            List[int]: A list of token IDs representing the input text.

        Notes:
            - Out-of-vocabulary characters are silently discarded.
            - Unique out-of-vocabulary characters are tracked and logged with a debug message.
        """
        token_ids: List[Optional[int]] = []
        for char in text:
            idx: Optional[int] = None
            if self.add_blank_word and char == " ":
                idx = self.blank_word_id
            else:
                idx = self.vocabulary.char2idx.get(char)

            if idx is not None:
                token_ids.append(idx)
            else:
                token_ids.append(None)  # Append None for later filtering
                # discard but store not found characters
                if char not in self.not_found_characters:
                    self.not_found_characters.add(char)
                    # LOG.warning(f" [!] Character {repr(char)} not found in the vocabulary. Discarding it.")

        # NOTE: mimic3 adds an extra word_blank at end, so we match that behaviour here
        #  instead of ending [..., BLANK, EOS] it ends with [..., BLANK, BLANK_WORD, BLANK, EOS]
        if self.add_blank_word and self.blank_at_end and self.blank_word_id is not None:
            token_ids.append(self.blank_word_id)

        # Filter out None values (out-of-vocabulary characters)
        return [t for t in token_ids if t is not None]

    def tokenize(self, text: Union[str, List[str]]) -> List[int]:
        """
        Convert text (phonemes or graphemes) to a sequence of token IDs.

        Applies a series of transformations to the input text:
        1. **Encoding**: Converts text characters/phonemes to base token IDs (`self.encode`).
        2. **Inter-character Blank Insertion**: Optionally inserts the blank character (`blank_id`) between tokens.
        3. **Start/End Blank Insertion**: Optionally prepends/appends a blank character.
        4. **BOS/EOS Padding**: Optionally adds beginning-of-sequence (BOS) and end-of-sequence (EOS) tokens.

        Parameters:
            text (str): Input text (phonemes or graphemes) to be converted to token IDs.

        Returns:
            List[int]: A sequence of token IDs after applying configured transformations.
        """
        token_ids: List[int] = self.encode(text)

        # 2. Inter-character Blank Insertion
        if self.add_blank_char and self.blank_id is not None:
            token_ids = self.intersperse_blank_char(token_ids)
        # 3. Start Blank Insertion (only if intersperse wasn't done, as intersperse already handles start/end)
        elif self.blank_at_start and self.blank_id is not None:
            token_ids.insert(0, self.blank_id)

        # 4. BOS/EOS Padding
        if self.use_eos_bos and self.vocabulary.bos_id is not None and self.vocabulary.eos_id is not None:
            token_ids = self.pad_with_bos_eos(token_ids)
        return token_ids

    def pad_with_bos_eos(self, token_sequence: List[int]) -> List[int]:
        """
        Pad a character sequence with beginning-of-sequence (BOS) and end-of-sequence (EOS) tokens.

        Parameters:
            token_sequence (List[int]): A list of character token IDs to be padded.

        Returns:
            List[int]: A new list with BOS token prepended and EOS token appended to the original sequence.
        """
        bos_id = self.vocabulary.bos_id
        eos_id = self.vocabulary.eos_id

        # This check is redundant due to the calling method's check, but added for safety
        if bos_id is None or eos_id is None:
            LOG.warning("BOS or EOS ID is None, skipping padding.")
            return token_sequence

        return [bos_id] + list(token_sequence) + [eos_id]

    def intersperse_blank_char(self, token_sequence: List[int]) -> List[int]:
        """
        Intersperses the blank character between characters in a sequence.

        This method creates a new sequence where the blank character is inserted between each
        original character, with optional blank tokens at the beginning and end of the sequence.

        Parameters:
            token_sequence (List[int]): A list of character IDs to be interspersed with blank tokens.

        Returns:
            List[int]: A new sequence with blank tokens inserted according to configuration.
        """
        blank_id = self.vocabulary.blank_id
        if blank_id is None:
            return token_sequence

        result: List[int] = [blank_id] * (len(token_sequence) * 2 + 1)
        result[1::2] = token_sequence

        # Remove starting/ending blank if configured not to be present
        if not self.blank_at_start and result:
            result = result[1:]
        if not self.blank_at_end and result:
            result = result[:-1]

        # Ensure a final blank is present if blank_at_end is True (mimic3 compatibility)
        if self.blank_at_end and result and result[-1] != blank_id:
            result.append(blank_id)

        return result

    @staticmethod
    def from_phoonnx_config(cfg: Dict[str, Any]) -> 'TTSTokenizer':
        """
        Factory method to create a TTSTokenizer from a phoonnx configuration.

        Parameters:
            cfg: The phoonnx configuration dictionary.

        Returns:
            A configured TTSTokenizer instance.
        """
        voc: Vocabulary = Vocabulary.from_phoonnx_config(cfg)
        # Default settings for phoonnx
        add_blank: bool = True
        blank_at_end: bool = True
        blank_at_start: bool = True
        use_eos_bos: bool = True
        add_blank_word: bool = False
        return TTSTokenizer(voc, add_blank_char=add_blank, add_blank_word=add_blank_word,
                            blank_at_end=blank_at_end, blank_at_start=blank_at_start,
                            use_eos_bos=use_eos_bos)

    @staticmethod
    def from_piper_config(cfg: Dict[str, Any]) -> 'TTSTokenizer':
        """
        Factory method to create a TTSTokenizer from a Piper configuration.

        Parameters:
            cfg: The Piper configuration dictionary.

        Returns:
            A configured TTSTokenizer instance.
        """
        voc: Vocabulary = Vocabulary.from_piper_config(cfg)
        # Default settings for Piper
        add_blank: bool = True
        blank_at_end: bool = True
        blank_at_start: bool = True
        use_eos_bos: bool = True
        add_blank_word: bool = False
        return TTSTokenizer(voc, add_blank_char=add_blank, add_blank_word=add_blank_word,
                            blank_at_end=blank_at_end, blank_at_start=blank_at_start,
                            use_eos_bos=use_eos_bos)

    @staticmethod
    def from_mimic3_config(cfg: Dict[str, Any], tokens_txt: str) -> 'TTSTokenizer':
        """
        Factory method to create a TTSTokenizer from a Mimic3 configuration and tokens file.

        Parameters:
            cfg: The Mimic3 configuration dictionary.
            tokens_txt: The content of the tokens.txt file.

        Returns:
            A configured TTSTokenizer instance.
        """
        voc: Vocabulary = Vocabulary.from_mimic3_config(cfg, tokens_txt)
        phonemes_cfg: Dict[str, Any] = cfg.get("phonemes", {})
        blank_between: str = phonemes_cfg.get("blank_between", BlankBetween.TOKENS_AND_WORDS)
        blank_at_end: bool = phonemes_cfg.get("blank_at_end", True)
        blank_at_start: bool = phonemes_cfg.get("blank_at_start", True)
        use_eos_bos: bool = phonemes_cfg.get("auto_bos_eos", True)

        add_blank: bool = True  # intersperse blank char
        add_blank_word: bool = True  # treat space as blank word token

        if blank_between == BlankBetween.WORDS:
            add_blank = False
        elif blank_between == BlankBetween.TOKENS:
            add_blank_word = False

        return TTSTokenizer(voc, add_blank_char=add_blank, add_blank_word=add_blank_word,
                            blank_at_end=blank_at_end, blank_at_start=blank_at_start,
                            use_eos_bos=use_eos_bos)

    @staticmethod
    def from_tokens_txt(tokens_txt: str) -> 'TTSTokenizer':
        """
        Factory method to create a TTSTokenizer only from a tokens.txt file content,
        using a set of default settings.

        Parameters:
            tokens_txt: The content of the tokens.txt file.

        Returns:
            A configured TTSTokenizer instance.
        """
        voc: Vocabulary = Vocabulary.from_tokens_txt(tokens_txt)
        # Conservative defaults if only tokens.txt is available
        add_blank_word: bool = True
        add_blank: bool = True
        blank_at_end: bool = True
        blank_at_start: bool = True
        use_eos_bos: bool = True
        return TTSTokenizer(voc, add_blank_char=add_blank, add_blank_word=add_blank_word,
                            blank_at_end=blank_at_end, blank_at_start=blank_at_start,
                            use_eos_bos=use_eos_bos)

    @staticmethod
    def from_coqui_config(cfg: Dict[str, Any]) -> 'TTSTokenizer':
        """
        Factory method to create a TTSTokenizer from a Coqui configuration.

        Parameters:
            cfg: The Coqui configuration dictionary.

        Returns:
            A configured TTSTokenizer instance.
        """
        voc: Vocabulary = Vocabulary.from_coqui_config(cfg)
        add_blank_word: bool = False
        # Coqui typically controls blank insertion via 'add_blank' flag
        add_blank: bool = cfg.get("add_blank", False)
        blank_at_end: bool = cfg.get("add_blank", False)
        blank_at_start: bool = cfg.get("add_blank", False)
        use_eos_bos: bool = cfg.get("enable_eos_bos_chars", False)
        return TTSTokenizer(voc, add_blank_char=add_blank, add_blank_word=add_blank_word,
                            blank_at_end=blank_at_end, blank_at_start=blank_at_start,
                            use_eos_bos=use_eos_bos)


if __name__ == "__main__":
    import json


    def _test_mimic3_compat(phone_str: str, cfg_path: str, tokens_path: str) -> None:
        print("\n## Testing mimic3 compat")
        # test original mimic3 code
        from phonemes2ids import phonemes2ids as mimic3_phonemes2ids

        with open(cfg_path, "r") as f:
            cfg = json.load(f)
            with open(tokens_path, "r") as f2:
                toks = f2.read()
            voc = Vocabulary.from_mimic3_config(cfg, toks)

        phone_words = [list(w) for w in
                       phone_str.split()]  # [['h', 'ə', 'l', 'ˈ', 'o', 'ʊ'], ['w', 'ˈ', 'ɜ', 'ː', 'l', 'd']]

        for blank_between in [BlankBetween.WORDS, BlankBetween.TOKENS, BlankBetween.TOKENS_AND_WORDS]:
            for blank_at_end in [True, False]:
                for blank_at_start in [True, False]:
                    for use_eos_bos in [True, False]:
                        add_blank = True
                        add_blank_word = True
                        if blank_between == BlankBetween.WORDS:
                            add_blank = False
                        elif blank_between == BlankBetween.TOKENS:
                            add_blank_word = False
                        print(
                            f"# blank_at_start={blank_at_start}, blank_at_end={blank_at_end}, add_blank={add_blank}, add_blank_word={add_blank_word}, use_eos_bos={use_eos_bos}")
                        tok = TTSTokenizer(voc, add_blank_char=add_blank, add_blank_word=add_blank_word,
                                           blank_at_end=blank_at_end, blank_at_start=blank_at_start,
                                           use_eos_bos=use_eos_bos)
                        print(tok.tokenize(phone_str))
                        print(mimic3_phonemes2ids(phone_words, tok.vocabulary.char2idx, pad=tok.vocabulary.pad,
                                                  bos=tok.vocabulary.bos, eos=tok.vocabulary.eos,
                                                  blank=tok.vocabulary.blank,
                                                  blank_word=tok.vocabulary.blank_word, blank_at_end=blank_at_end,
                                                  blank_at_start=blank_at_start, blank_between=blank_between,
                                                  auto_bos_eos=use_eos_bos))


    def _test_piper_compat(phone_str: str, cfg_path: str):
        print("\n## Testing piper compat")
        from piper_phonemize import phoneme_ids_espeak
        phones = list(phone_str)  # ['h', 'ə', 'l', 'ˈ', 'o', 'ʊ', ' ', 'w', 'ˈ', 'ɜ', 'ː', 'l', 'd']
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
            voc = Vocabulary.from_piper_config(cfg)

        add_blank = blank_at_end = blank_at_start = use_eos_bos = True
        add_blank_word = False
        tok = TTSTokenizer(voc, add_blank_char=add_blank, add_blank_word=add_blank_word,
                           blank_at_end=blank_at_end, blank_at_start=blank_at_start,
                           use_eos_bos=use_eos_bos)
        print(
            f"# blank_at_start={blank_at_start}, blank_at_end={blank_at_end}, add_blank={add_blank}, add_blank_word={add_blank_word}, use_eos_bos={use_eos_bos}")
        print(tok.tokenize(phone_str))
        print(phoneme_ids_espeak(phones))


    def _test_coqui_compat(phone_str: str, cfg_path: str):
        print("\n## Testing coqui compat")

        from TTS.tts.configs.vits_config import VitsConfig
        from TTS.tts.models.vits import Vits

        config = VitsConfig()
        config.load_json(cfg_path)
        vits = Vits.init_from_config(config)

        with open(cfg_path, "r") as f:
            cfg = json.load(f)
            voc = Vocabulary.from_coqui_config(cfg)
            add_blank = blank_at_end = blank_at_start = cfg.get("add_blank")
            use_eos_bos = cfg.get("enable_eos_bos_chars")

        add_blank_word = False
        tok = TTSTokenizer(voc, add_blank_char=add_blank, add_blank_word=add_blank_word,
                           blank_at_end=blank_at_end, blank_at_start=blank_at_start,
                           use_eos_bos=use_eos_bos)
        print(
            f"# blank_at_start={blank_at_start}, blank_at_end={blank_at_end}, add_blank={add_blank}, add_blank_word={add_blank_word}, use_eos_bos={use_eos_bos}")
        print(tok.tokenize(phone_str))
        print(vits.tokenizer.text_to_ids(phone_str, language=None))
        print(vits.tokenizer.characters.vocab)


    phone_str = "həlˈoʊ wˈɜːld"

    piper = "/home/miro/Transferências/miro_eu-ES.piper.json"
    _test_piper_compat(phone_str, piper)

    mimic3 = "/home/miro/Transferências/config.json"
    tokens_txt = "/home/miro/Transferências/phonemes.txt"
    _test_mimic3_compat(phone_str, mimic3, tokens_txt)

    # graphemes
    for v in ["celtia", "brais"]:
        text = "redes neuronais artificiais"
        coqui = f"/home/miro/.cache/phoonnx/voices/proxectonos/{v}/model.json"
        _test_coqui_compat(text, coqui)
    # cotovia
    for v in ["sabela", "iago", "icia", "paulo"]:
        phone_coto = "rreDes newronajs artifiTjajs"
        coqui = f"/home/miro/.cache/phoonnx/voices/proxectonos/{v}/model.json"
        _test_coqui_compat(phone_coto, coqui)
