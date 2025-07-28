import re
from functools import partial

from phoonnx.thirdparty.mantoq.lib.pyarabic import araby
from phoonnx.thirdparty.mantoq.lib.pyarabic import number as arnum
from phoonnx.thirdparty.mantoq.lib.pyarabic.trans import normalize_digits

NUM_REGEX = re.compile(r"\d+")
PERCENT_NO_DIAC = "بالمئة"
PERCENT_DIAC = "بِالْمِئَة"


def _convert_num2words(m: re.Match, *, apply_tashkeel):
    number = m.group(0)
    word_representation = arnum.number2text(number)
    if apply_tashkeel:
        return " ".join(arnum.pre_tashkeel_number(word_representation.split(" ")))
    return word_representation


def num2words(text: str, handle_percent=True, apply_tashkeel: bool = True) -> str:
    """
    Converts numbers in `text` to Arabic words.
    Simple conversion. Does not check if the number is date/currency...etc.

    Args:
        text: input text that may contain numbers
        apply_tashkeel: diacritize added words
    """
    text = normalize_digits(text)
    output = NUM_REGEX.sub(
        partial(_convert_num2words, apply_tashkeel=apply_tashkeel), text
    )
    if handle_percent:
        replacement = PERCENT_DIAC if apply_tashkeel else PERCENT_NO_DIAC
        output = output.replace("%", f" {replacement}")
    return araby.fix_spaces(output)
