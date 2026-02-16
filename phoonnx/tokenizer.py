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
        Builds a Vocabulary from a phoonnx configuration dictionary.
        
        Parameters:
            cfg (Dict[str, Any]): Configuration mapping expected to contain "phoneme_id_map"
                (mapping phoneme strings to integer IDs). Optional keys "pad", "eos", "bos",
                and "blank" override the corresponding special token names; defaults are used when absent.
        
        Returns:
            Vocabulary: Vocabulary populated with the parsed char-to-index map and configured special tokens.
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
        Build a Vocabulary from a Mimic3 configuration dictionary and the contents of a tokens.txt file.
        
        Parameters:
            cfg (Dict[str, Any]): Mimic3 configuration dict; special token names are read from the "phonemes" section.
            tokens_txt (str): Raw text of a tokens.txt file that maps token strings to numeric IDs.
        
        Returns:
            Vocabulary: A Vocabulary populated from the tokens file with special tokens (pad, bos, eos, blank, blank_word) set from the configuration when present.
        """
        voc: 'Vocabulary' = Vocabulary.from_tokens_txt(tokens_txt, id_first=True)
        voc.pad = cfg.get("phonemes", {}).get("pad") or cfg.get("phonemes", {}).get("phoneme_separator")
        voc.eos = cfg.get("phonemes", {}).get("eos")
        voc.bos = cfg.get("phonemes", {}).get("bos")
        voc.blank = cfg.get("phonemes", {}).get("blank")
        voc.blank_word = cfg.get("phonemes", {}).get("blank_word") or cfg.get("phonemes", {}).get("word_separator")
        return voc

    @staticmethod
    def from_tokens_txt(tokens_txt: str, id_first: bool = False) -> 'Vocabulary':
        """
        Creates a Vocabulary instance by parsing a tokens.txt style string (ID token per line).

        Parameters:
            tokens_txt: A string where each line is formatted as "ID token" (e.g., "0 <pad>").

        Returns:
            A Vocabulary instance with character-to-index mapping.
        """
        char2idx: Dict[str, int] = {}
        blank_tok = None
        for line in tokens_txt.split("\n"):
            try:
                if id_first:
                    idx_str, token = line.split(" ", 1)
                else:
                    token, idx_str = line.split(" ", 1)
                char2idx[token] = int(idx_str)
                if int(idx_str) == 0:
                    blank_tok = token
            except ValueError:
                # Skip empty lines or malformed lines
                pass

        return Vocabulary(char2idx=char2idx, blank=blank_tok)

    @staticmethod
    def from_coqui_config(cfg: Dict[str, Any]) -> 'Vocabulary':
        """
        Build a Vocabulary from a Coqui TTS configuration dictionary.
        
        Supports Coqui character classes `TTS.tts.models.vits.VitsCharacters` and
        `TTS.tts.utils.text.characters.Graphemes`; maps configured characters, punctuations,
        and special tokens (pad, bos, eos, blank) to integer IDs according to Coqui's settings.
        
        Parameters:
            cfg (Dict[str, Any]): Coqui TTS configuration dictionary containing a "characters" section.
        
        Returns:
            Vocabulary: A Vocabulary whose char2idx maps each token to its index and whose
            special-token fields (pad, eos, bos, blank) are set from the configuration.
        
        Raises:
            ValueError: If the configuration specifies an unsupported Coqui tokenizer class.
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
        """
        Map token IDs to their corresponding characters.
        
        Returns:
            idx2char (Dict[int, str]): A dictionary mapping each token ID to its character.
        """
        return {idx: char for char, idx in self.char2idx.items()}

    @property
    def pad_id(self) -> Optional[int]:
        """
        Get the vocabulary ID for the padding token.
        
        Returns:
            Optional[int]: The padding token's ID if a padding token is defined and present in the vocabulary, `None` otherwise.
        """
        return self.char2idx.get(self.pad) if self.pad else None

    @property
    def blank_id(self) -> Optional[int]:
        """
        Get the vocabulary ID for the blank (inter-phoneme) token.
        
        Returns:
            The ID of the blank token, or `None` if the blank token is not defined or not present in the vocabulary.
        """
        return self.char2idx.get(self.blank) if self.blank else None

    @property
    def blank_word_id(self) -> Optional[int]:
        """
        Return the ID of the word-level blank token used to separate words.
        
        Returns:
            Optional[int]: The ID of the word blank token, or `None` if it is not defined or not present in the vocabulary.
        """
        return self.char2idx.get(self.blank_word) if self.blank_word else None

    @property
    def eos_id(self) -> Optional[int]:
        """
        Get the ID of the end-of-sequence (EOS) token.
        
        Returns:
            Optional[int]: The EOS token ID, or `None` if the EOS token is not set or not found in the vocabulary.
        """
        return self.char2idx.get(self.eos) if self.eos else None

    @property
    def bos_id(self) -> Optional[int]:
        """
        Get the vocabulary ID for the beginning-of-sequence (BOS) token.
        
        Returns:
            bos_id (Optional[int]): The ID for the BOS token if defined and present in the vocabulary, `None` otherwise.
        """
        return self.char2idx.get(self.bos) if self.bos else None

    @property
    def num_chars(self) -> int:
        """
        Return the number of entries in the vocabulary.
        
        Returns:
            int: Number of mapped characters (length of char2idx).
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

    @property
    def pad_id(self) -> Optional[int]:
        """
        Get the padding token ID from the vocabulary.
        
        Returns:
            int or None: ID of the padding token if defined, otherwise None.
        """
        return self.vocabulary.pad_id

    @property
    def blank_id(self) -> Optional[int]:
        """
        Get the vocabulary ID used as the inter-phoneme blank token.
        
        Returns:
            int: ID of the inter-phoneme blank token, or `None` if the token is not defined.
        """
        return self.vocabulary.blank_id

    @property
    def blank_word_id(self) -> Optional[int]:
        """
        Get the token ID used for word-level blanks from the tokenizer's vocabulary.
        
        Returns:
            blank_word_id (Optional[int]): The ID of the inter-word blank token, or None if not defined.
        """
        return self.vocabulary.blank_word_id

    def encode(self, text: Union[str, List[str]]) -> List[int]:
        """
        Map input characters (string or list of single-character strings) to their vocabulary token IDs.
        
        If `add_blank_word` is enabled, space characters are mapped to the vocabulary's blank-word token. Unknown characters are discarded from the returned sequence but their first occurrences are recorded in `self.not_found_characters`. If `add_blank_word` and `blank_at_end` are enabled and a blank-word token exists, a trailing blank-word token is appended.
        
        Parameters:
            text (str | List[str]): Input text as a string or a list of single-character strings to encode.
        
        Returns:
            List[int]: Sequence of token IDs corresponding to the input characters, with unknown characters removed.
        """
        # first pre-process phoneme_map to check for dipthongs having their own phoneme_id
        # common in mimic3 models
        compound_toks = sorted((k for k in self.vocabulary.char2idx
                                if len(k) > 1), key=len, reverse=True)

        token_ids: List[Optional[int]] = []

        compound_idxs: List[int] = []

        for i, char in enumerate(text):
            if i in compound_idxs:
                idx = None
            elif self.add_blank_word and char == " ":
                idx = self.blank_word_id
            else:
                idx = self.vocabulary.char2idx.get(char)

                # Try to match compound phonemes starting at index i
                for compound in compound_toks:
                    n = len(compound)
                    joined = ''.join(text[i:i + n])
                    if joined == compound:
                        idx = self.vocabulary.char2idx[compound]
                        compound_idxs += [i for i in range(i, i+n)]
                        break

            token_ids.append(idx)

        # NOTE: mimic3 adds an extra word_blank at end, so we match that behaviour here
        #  instead of ending [..., BLANK, EOS] it ends with [..., BLANK, BLANK_WORD, BLANK, EOS]
        if self.add_blank_word and self.blank_at_end and self.blank_word_id is not None:
            token_ids.append(self.blank_word_id)

        # Filter out None values (out-of-vocabulary characters)
        return [t for t in token_ids if t is not None]

    def tokenize(self, text: Union[str, List[str]]) -> List[int]:
        """
        Map input phonemes or graphemes to a sequence of vocabulary token IDs.
        
        Applies optional inter-character blank insertion, optional leading/trailing blank insertion, and optional BOS/EOS padding according to the tokenizer's configuration.
        
        Parameters:
            text (str | List[str]): Input text (phonemes or graphemes) or a list of token strings to convert.
        
        Returns:
            List[int]: Sequence of token IDs after applying the configured transformations.
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
        Pad a token sequence with the vocabulary's BOS (beginning-of-sequence) and EOS (end-of-sequence) tokens.
        
        Parameters:
            token_sequence (List[int]): Sequence of token IDs to wrap.
        
        Returns:
            List[int]: A new list with the vocabulary's BOS prepended and EOS appended. If the vocabulary does not define BOS or EOS, returns the original sequence unchanged.
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
        Insert blank tokens between token IDs, optionally adding a leading and/or trailing blank.
        
        If the tokenizer's vocabulary has no blank token defined, the original sequence is returned unchanged.
        The method respects the tokenizer flags `blank_at_start` and `blank_at_end`. When `blank_at_end` is
        True, a trailing blank will be ensured even if the interleaving logic would not produce one.
        
        Parameters:
            token_sequence (List[int]): Sequence of token IDs to intersperse with blank tokens.
        
        Returns:
            List[int]: New sequence with blank token IDs interleaved according to tokenizer configuration,
            or the original sequence if no blank token is available.
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
        Create a TTSTokenizer configured from a phoonnx configuration.
        
        Parameters:
            cfg (Dict[str, Any]): Phoonnx configuration dictionary used to construct the vocabulary
                and tokenizer options.
        
        Returns:
            TTSTokenizer: A tokenizer configured with phoonnx defaults (inter-character blanks enabled,
            blanks at start and end enabled, BOS/EOS wrapping enabled, and word-blank mapping disabled).
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
        Create a TTSTokenizer configured from a Piper configuration.
        
        Parameters:
            cfg (Dict[str, Any]): Piper configuration dictionary containing phoneme/token mappings and optional special token names.
        
        Returns:
            TTSTokenizer: Tokenizer configured with the Vocabulary derived from `cfg` and Piper-oriented defaults for blank and BOS/EOS handling.
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

        add_blank: bool = blank_between != BlankBetween.WORDS  # intersperse blank char
        add_blank_word: bool = blank_between != BlankBetween.TOKENS

        return TTSTokenizer(voc, add_blank_char=add_blank, add_blank_word=add_blank_word,
                            blank_at_end=blank_at_end, blank_at_start=blank_at_start,
                            use_eos_bos=use_eos_bos)

    @staticmethod
    def from_tokens_txt(tokens_txt: str, id_first=False) -> 'TTSTokenizer':
        """
        Create a TTSTokenizer from the contents of a tokens.txt file using conservative defaults.
        
        Parameters:
            tokens_txt (str): Contents of a tokens.txt file that maps token strings to IDs.
        
        Returns:
            TTSTokenizer: A tokenizer built from the parsed vocabulary configured with
            add_blank_char=True, add_blank_word=True, blank_at_end=True, blank_at_start=True,
            and use_eos_bos=True.
        """
        voc: Vocabulary = Vocabulary.from_tokens_txt(tokens_txt, id_first)
        add_blank_word: bool = False
        add_blank: bool = True
        blank_at_end: bool = True
        blank_at_start: bool = True
        use_eos_bos: bool = False
        return TTSTokenizer(voc, add_blank_char=add_blank, add_blank_word=add_blank_word,
                            blank_at_end=blank_at_end, blank_at_start=blank_at_start,
                            use_eos_bos=use_eos_bos)

    @staticmethod
    def from_coqui_config(cfg: Dict[str, Any]) -> 'TTSTokenizer':
        """
        Create a TTSTokenizer configured from a Coqui TTS configuration.
        
        Interprets the following Coqui config keys:
        - "add_blank": enable inter-character blank insertion and control blank placement at start/end.
        - "enable_eos_bos_chars": enable wrapping token sequences with BOS/EOS.
        This factory does not enable word-level blank mapping (blank_word is False).
        
        Parameters:
            cfg (Dict[str, Any]): Coqui configuration dictionary.
        
        Returns:
            TTSTokenizer: Tokenizer instance configured according to the provided Coqui config.
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
        """
        Run compatibility checks against Mimic3's phonemes2ids and print tokenization comparisons.
        
        Builds a Vocabulary from the provided Mimic3 config and tokens file, then for multiple combinations of blank placement and BOS/EOS usage constructs a TTSTokenizer and prints both the tokenizer's output and the result of Mimic3's phonemes2ids for comparison. Outputs are printed to stdout.
        
        Parameters:
            phone_str (str): Space-separated phoneme string to test (words separated by spaces).
            cfg_path (str): Path to the Mimic3 JSON configuration file.
            tokens_path (str): Path to the Mimic3 tokens.txt content file.
        """
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