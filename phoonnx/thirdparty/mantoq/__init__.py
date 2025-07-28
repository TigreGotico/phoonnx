from phoonnx.thirdparty.mantoq.lib.buck import symbols
from phoonnx.thirdparty.mantoq.lib.buck.tokenization import (arabic_to_phonemes, phon_to_id_,
                                    phonemes_to_tokens, simplify_phonemes)
from phoonnx.thirdparty.mantoq.lib.buck.tokenization import tokens_to_ids as _tokens_to_id
from phoonnx.thirdparty.mantoq.num2words import num2words
import warnings
from phoonnx.thirdparty.tashkeel import TashkeelDiacritizer
try:
    import onnxruntime

    _TASHKEEL_AVAILABLE = True
except ImportError:
    _TASHKEEL_AVAILABLE = False

_DIACRITIZER_INST = None

MANTOQ_SYMBOLS = dict(phon_to_id_)
MANTOQ_SPECIAL_SYMBOLS = dict(
    pad=phon_to_id_[symbols.PADDING_TOKEN],
    eos=phon_to_id_[symbols.EOS_TOKEN],
)
# Maps Arabic-specific puncs with their English equivlents
AR_SPECIAL_PUNCS_TABLE = str.maketrans("،؟؛", ",?;")
OMITTED_SYMBOLS = str.maketrans("", "", "+=<>")

# Quotes
QUOTES = '“”„«»'
QUOTES_TABLE = str.maketrans(QUOTES, '"' * len(QUOTES))
BRACKETS_TABLE = str.maketrans("[]{}", "()()")




def tashkeel(text: str) -> str:
    global _DIACRITIZER_INST
    if not _TASHKEEL_AVAILABLE:
        warnings.warn(
            "Warning: The Tashkeel feature will not be available. Please re-install with the `libtashkeel` extra.",
            UserWarning,
        )
        return text
    if _DIACRITIZER_INST is None:
        _DIACRITIZER_INST = TashkeelDiacritizer()
    return _DIACRITIZER_INST.diacritize(text)

def g2p(
    text: str,
    add_tashkeel: bool = True,
    process_numbers: bool = True,
    append_eos: bool = False,
) -> list[str]:
    text = text.translate(AR_SPECIAL_PUNCS_TABLE).translate(QUOTES_TABLE).translate(BRACKETS_TABLE)
    if add_tashkeel:
        text = tashkeel(text)
    if process_numbers:
        text = num2words(text)
    normalized_text = text
    phones = arabic_to_phonemes(text)
    phones = simplify_phonemes(phones)
    tokens = phonemes_to_tokens(phones)
    if not append_eos:
        tokens = tokens[:-1]
    return normalized_text, tokens


def tokens2ids(tokens: list[str]) -> list[int]:
    return _tokens_to_id(tokens)
