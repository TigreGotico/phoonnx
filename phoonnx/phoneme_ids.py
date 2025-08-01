"""Utilities for converting phonemes to ids."""

from collections.abc import Sequence
from enum import Enum
from typing import Optional, TextIO, Dict, List, Union, Set, Mapping

try:
    from ovos_utils.log import LOG
except ImportError:
    import logging

    LOG = logging.getLogger(__name__)
    LOG.setLevel("DEBUG")

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
DEFAULT_BLANK_WORD_TOKEN = "#"  # padding between words

STRESS: Set[str] = {"ˈ", "ˌ"}
"""Default stress characters"""

PUNCTUATION_MAP: Mapping[str, str] = {";": ",", ":": ",", "?": ".", "!": "."}
"""Default punctuation simplification into short (,) and long (.) pauses"""


class BlankBetween(str, Enum):
    """Placement of blank tokens"""

    TOKENS = "tokens"
    """Blank between every token/phoneme"""

    WORDS = "words"
    """Blank between every word"""

    TOKENS_AND_WORDS = "tokens_and_words"
    """Blank between every token/phoneme and every word (may be different symbols)"""


def phonemes_to_ids(
        phonemes: PHONEME_LIST,
        id_map: Optional[Mapping[str, Union[int, Sequence[int]]]] = None,
        blank_token: Optional[str] = DEFAULT_BLANK_TOKEN,
        bos_token: Optional[str] = DEFAULT_BOS_TOKEN,
        eos_token: Optional[str] = DEFAULT_EOS_TOKEN,
        word_sep_token: Optional[str] = DEFAULT_BLANK_WORD_TOKEN,
        include_whitespace: Optional[bool] = True,
        blank_at_start: bool = True,
        blank_at_end: bool = True,
        blank_between: BlankBetween = BlankBetween.TOKENS_AND_WORDS,
) -> PHONEME_ID_LIST:
    """Phonemes to ids."""
    if not phonemes:
        return []
    if not id_map:
        id_map = DEFAULT_IPA_PHONEME_ID_MAP

    # compat with piper-style mapping that uses lists
    id_map = {k: v if isinstance(v, list) else [v]
              for k, v in id_map.items()}

    ids: list[int] = []
    blank_id = blank_token if isinstance(blank_token, int) \
        else id_map.get(blank_token, [len(id_map)]) if blank_token \
        else [len(id_map)]
    eos_id = eos_token if isinstance(eos_token, int) \
        else id_map.get(eos_token, [len(id_map)]) if eos_token \
        else [len(id_map)]
    bos_id = eos_token if isinstance(bos_token, int) \
        else id_map.get(bos_token, [len(id_map)]) if bos_token \
        else [len(id_map)]

    if bos_token is not None:
        ids.extend(bos_id)
    if blank_token is not None and blank_at_start:
        ids.extend(blank_id)

    blank_between_tokns = (blank_token is not None and
                           blank_between in [BlankBetween.TOKENS, BlankBetween.TOKENS_AND_WORDS])
    blank_between_words = (blank_token is not None and
                           blank_between in [BlankBetween.WORDS, BlankBetween.TOKENS_AND_WORDS])

    # first pre-process phoneme_map to check for dipthongs having their own phoneme_id
    # common in mimic3 models
    compound_phonemes = sorted((k for k in id_map if len(k) > 1), key=len, reverse=True)
    i = 0
    while i < len(phonemes):
        matched = False

        # Try to match compound phonemes starting at index i
        for compound in compound_phonemes:
            n = len(compound)
            joined = ''.join(phonemes[i:i + n])
            if joined == compound:
                ids.extend(id_map[compound])
                if blank_between_tokns and i + n < len(phonemes):
                    ids.extend(blank_id)
                i += n
                matched = True
                break

        if matched:
            continue

        phoneme = phonemes[i]
        if phoneme not in id_map:
            if phoneme == " " and not include_whitespace:
                i += 1
                continue
            LOG.warning("Missing phoneme from id map: %s", phoneme)
            i += 1
            continue

        if phoneme == " ":
            if include_whitespace:
                ids.extend(id_map[phoneme])
                if blank_between_tokns:
                    ids.extend(blank_id)
            elif blank_between_words:
                ids.extend(id_map[word_sep_token])
                if blank_between_tokns:
                    ids.extend(blank_id)
        else:
            ids.extend(id_map[phoneme])
            if blank_between_tokns and i < len(phonemes) - 1:
                ids.extend(blank_id)
        i += 1

    if blank_token is not None and blank_at_end:
        if not include_whitespace and word_sep_token and blank_between_words:
            if blank_between_tokns:
                ids.extend(blank_id)
            ids.extend(id_map[word_sep_token])
            if blank_between_tokns:
                ids.extend(blank_id)
        else:
            ids.extend(blank_id)
    if eos_token is not None:
        ids.extend(eos_id)

    return ids

def load_phoneme_ids(phonemes_file: TextIO) -> PHONEME_ID_MAP:
    """
    Load phoneme id mapping from a text file.
    Format is ID<space>PHONEME
    Comments start with #

    Args:
        phonemes_file: text file

    Returns:
        dict with phoneme -> id
    """
    phoneme_to_id = {}
    for line in phonemes_file:
        line = line.strip("\r\n")
        if (not line) or line.startswith("#") or (" " not in line):
            # Exclude blank lines, comments, or malformed lines
            continue

        if line.strip().isdigit(): # phoneme is whitespace
            phoneme_str = " "
            phoneme_id = int(line)
        else:
            phoneme_id, phoneme_str = line.split(" ", maxsplit=1)
            if phoneme_str.isdigit():
                phoneme_id, phoneme_str = phoneme_str, phoneme_id

        phoneme_to_id[phoneme_str] = int(phoneme_id)

    return phoneme_to_id


def load_phoneme_map(phoneme_map_file: TextIO) -> Dict[str, List[str]]:
    """
    Load phoneme/phoneme mapping from a text file.
    Format is FROM_PHONEME<space>TO_PHONEME[<space>TO_PHONEME...]
    Comments start with #

    Args:
        phoneme_map_file: text file

    Returns:
        dict with from_phoneme -> [to_phoneme, to_phoneme, ...]
    """
    phoneme_map = {}
    for line in phoneme_map_file:
        line = line.strip("\r\n")
        if (not line) or line.startswith("#") or (" " not in line):
            # Exclude blank lines, comments, or malformed lines
            continue

        from_phoneme, to_phonemes_str = line.split(" ", maxsplit=1)
        if not to_phonemes_str.strip():
            # To whitespace
            phoneme_map[from_phoneme] = [" "]
        else:
            # To one or more non-whitespace phonemes
            phoneme_map[from_phoneme] = to_phonemes_str.split()

    return phoneme_map


if __name__ == "__main__":
    phoneme_ids_path = "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/phonemes.txt"
    with open(phoneme_ids_path, "r", encoding="utf-8") as ids_file:
        phoneme_to_id = load_phoneme_ids(ids_file)
    print(phoneme_to_id)

    phoneme_map_path = "/home/miro/PycharmProjects/phoonnx_tts/mimic3_ap/phoneme_map.txt"
    with open(phoneme_map_path, "r", encoding="utf-8") as map_file:
        phoneme_map = load_phoneme_map(map_file)
    # print(phoneme_map)

    from phoonnx.phonemizers import EspeakPhonemizer

    # test original mimic3 code
    from phonemes2ids import phonemes2ids as mimic3_phonemes2ids

    # test original piper code
    from piper.phoneme_ids import phonemes_to_ids as piper_phonemes_to_ids

    espeak = EspeakPhonemizer()
    phone_str: str = espeak.phonemize_string("hello world", "en")

    phones: PHONEME_LIST = list(phone_str)
    phone_words: PHONEME_WORD_LIST = [list(w) for w in phone_str.split()]
    print(phone_str)
    print(phones) # piper style
    print(phone_words) # mimic3 style

    mapping = {k: v[0] for k, v in DEFAULT_IPA_PHONEME_ID_MAP.items()}
    print("\n#### piper  (tokens_and_words + include_whitespace)")
    print("reference", piper_phonemes_to_ids(phones))
    print("phonnx   ", phonemes_to_ids(phones,
                                       id_map=mapping, include_whitespace=True))

    print("\n#### mimic3  (words)")
    print("reference", mimic3_phonemes2ids(phone_words,
                                           mapping,
                                           bos=DEFAULT_BOS_TOKEN,
                                           eos=DEFAULT_EOS_TOKEN,
                                           blank=DEFAULT_PAD_TOKEN,
                                           blank_at_end=True,
                                           blank_at_start=True,
                                           blank_word=DEFAULT_BLANK_WORD_TOKEN,
                                           blank_between=BlankBetween.WORDS,
                                           auto_bos_eos=True))
    print("phonnx   ", phonemes_to_ids(phones,
                                       id_map=mapping,
                                       include_whitespace=False,
                                       blank_between=BlankBetween.WORDS))

    print("\n#### mimic3  (tokens)")
    print("reference", mimic3_phonemes2ids(phone_words,
                                           mapping,
                                           bos=DEFAULT_BOS_TOKEN,
                                           eos=DEFAULT_EOS_TOKEN,
                                           blank=DEFAULT_PAD_TOKEN,
                                           blank_at_end=True,
                                           blank_at_start=True,
                                           blank_word=DEFAULT_BLANK_WORD_TOKEN,
                                           blank_between=BlankBetween.TOKENS,
                                           auto_bos_eos=True))
    print("phonnx   ", phonemes_to_ids(phones,
                                       id_map=mapping,
                                       include_whitespace=False,
                                       blank_between=BlankBetween.TOKENS))
    print("\n#### mimic3  (tokens_and_words)")
    print("reference", mimic3_phonemes2ids(phone_words,
                                           mapping,
                                           bos=DEFAULT_BOS_TOKEN,
                                           eos=DEFAULT_EOS_TOKEN,
                                           blank=DEFAULT_PAD_TOKEN,
                                           blank_at_end=True,
                                           blank_at_start=True,
                                           blank_word=DEFAULT_BLANK_WORD_TOKEN,
                                           blank_between=BlankBetween.TOKENS_AND_WORDS,
                                           auto_bos_eos=True))
    print("phonnx   ", phonemes_to_ids(phones,
                                       id_map=mapping,
                                       include_whitespace=False,
                                       blank_between=BlankBetween.TOKENS_AND_WORDS))
