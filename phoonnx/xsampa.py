"""X-SAMPA to IPA, for ``<phoneme alphabet="x-sampa" ph="...">``.

phoonnx voices take IPA (espeak's alphabet), so an X-SAMPA transcription is converted before it
is spoken. The table is the standard X-SAMPA chart; the longest symbol that matches wins
(``tS`` before ``t``), and anything unknown is passed through unchanged.
"""
from __future__ import annotations

# Symbols and values: the X-SAMPA chart, https://en.wikipedia.org/wiki/X-SAMPA
_XSAMPA = {
    # consonants
    "p": "p", "b": "b", "t": "t", "d": "d", "c": "c", "J\\": "ɟ", "k": "k", "g": "ɡ", "q": "q",
    "G\\": "ɢ", "?": "ʔ", "m": "m", "F": "ɱ", "n": "n", "J": "ɲ", "N": "ŋ", "N\\": "ɴ",
    "B\\": "ʙ", "r": "r", "R\\": "ʀ", "4": "ɾ", "r\\`": "ɻ", "p\\": "ɸ", "B": "β", "f": "f",
    "v": "v", "T": "θ", "D": "ð", "s": "s", "z": "z", "S": "ʃ", "Z": "ʒ", "s\\`": "ʂ",
    "z\\`": "ʐ", "s\\": "ɕ", "z\\": "ʑ", "C": "ç", "j\\": "ʝ", "x": "x", "x\\": "ɧ", "G": "ɣ",
    "X": "χ", "R": "ʁ", "X\\": "ħ", "?\\": "ʕ", "H\\": "ʜ", "h": "h", "h\\": "ɦ", "K": "ɬ",
    "K\\": "ɮ", "P": "ʋ", "v\\": "ʋ", "r\\": "ɹ", "j": "j", "M": "ɯ", "M\\": "ɰ", "L": "ʎ",
    "L\\": "ʟ", "l": "l", "l\\": "ɺ", "w": "w", "H": "ɥ", "W": "ʍ", "t`": "ʈ",
    "d`": "ɖ", "n`": "ɳ", "l`": "ɭ", "r`": "ɽ", "s`": "ʂ", "z`": "ʐ", "tS": "tʃ", "dZ": "dʒ",
    "ts": "ts", "dz": "dz", "O\\": "ʘ", "|\\": "ǀ", "!\\": "ǃ", "=\\": "ǂ", "|\\|\\": "ǁ",
    # vowels
    "i": "i", "y": "y", "1": "ɨ", "}": "ʉ", "M": "ɯ", "u": "u", "I": "ɪ", "I\\": "ᵻ", "Y": "ʏ",
    "U": "ʊ", "U\\": "ᵿ",
    "e": "e", "2": "ø", "@\\": "ɘ", "8": "ɵ", "7": "ɤ", "o": "o", "@": "ə", "E": "ɛ", "9": "œ",
    "3": "ɜ", "3\\": "ɞ", "V": "ʌ", "O": "ɔ", "{": "æ", "6": "ɐ", "a": "a", "&": "ɶ",
    "A": "ɑ", "Q": "ɒ",
    # suprasegmentals and diacritics
    '"': "ˈ", "%": "ˌ", ":": "ː", ":\\": "ˑ", "-": "", ".": ".", "_j": "ʲ", "_w": "ʷ",
    "_h": "ʰ", "_?\\": "ˤ", "~": "̃", "_~": "̃", "=": "̩", "_=": "̩", "_0": "̥", "_v": "̬",
    "_t": "̤",
    "_k": "̰", "_a": "̺", "_m": "̻", "_+": "̟", "_-": "̠", "_\"": "̈", "_X": "̆", "_^": "̯",
    "_o": "̞", "_r": "̝", "_}": "̚", "_G": "ˠ", "_N": "̼", "_d": "̪", "_e": "̴", "`": "˞",
    "<\\": "ʢ", ">\\": "ʡ", "b_<": "ɓ", "d_<": "ɗ", "g_<": "ɠ", "J\\_<": "ʄ", "G\\_<": "ʛ",
}
_SYMBOLS = sorted(_XSAMPA, key=len, reverse=True)


def xsampa_to_ipa(text: str) -> str:
    """The IPA for an X-SAMPA transcription (unknown characters pass through)."""
    out: list[str] = []
    i = 0
    while i < len(text):
        for symbol in _SYMBOLS:
            if text.startswith(symbol, i):
                out.append(_XSAMPA[symbol])
                i += len(symbol)
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)
