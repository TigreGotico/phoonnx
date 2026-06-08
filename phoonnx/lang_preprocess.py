"""Per-language script transforms for TTS front ends.

Some voices are trained on a normalised form of the input script rather than raw
Unicode.  This module provides the transforms that bridge caller text to the
model-expected representation.

Hangul→Jamo is pure Python and always available.  Japanese, Chinese and Russian
each require an optional dependency; when the dependency is absent the function
warns once and returns the input unchanged (graceful identity).
"""
from unicodedata import category, normalize

from phoonnx.util import LOG

# lazy singletons for the optional backends (created on first use)
_kakasi = None
_cangjie = None


def is_kanji(c: str) -> bool:
    return "一" <= c <= "鿿"


def is_katakana(c: str) -> bool:
    return "ァ" <= c <= "ヺ"


def hangul_to_jamo(text: str) -> str:
    """Decompose precomposed Hangul syllables into conjoining Jamo (NFD). Pure Python.

    Models trained on Jamo sequences require this step; the syllable codepoints are
    out-of-vocabulary for them.  Non-Hangul characters are passed through unchanged.
    """
    def decompose(ch: str) -> str:
        if not ("가" <= ch <= "힯"):
            return ch
        base = ord(ch) - 0xAC00
        initial = chr(0x1100 + base // (21 * 28))
        medial = chr(0x1161 + (base % (21 * 28)) // 28)
        final = chr(0x11A7 + base % 28) if base % 28 else ""
        return initial + medial + final

    return "".join(decompose(c) for c in text).strip()


def japanese_to_hiragana(text: str) -> str:
    """Convert kanji to hiragana (katakana kept) and NFKD-normalise. Needs ``pykakasi``.

    Returns *text* unchanged if ``pykakasi`` is not installed.
    """
    global _kakasi
    try:
        if _kakasi is None:
            import pykakasi
            _kakasi = pykakasi.kakasi()
    except ImportError:
        LOG.warning("pykakasi not installed — Japanese text left unprocessed")
        return text
    out = []
    for r in _kakasi.convert(text):
        inp, hira = r["orig"], r["hira"]
        if any(is_kanji(c) for c in inp):
            if hira and hira[0] in ("は", "へ"):  # は / へ
                hira = " " + hira
            out.append(hira)
        elif inp and all(is_katakana(c) for c in inp):
            out.append(inp)
        else:
            out.append(inp)
    return normalize("NFKD", "".join(out))


class ChineseCangjieConverter:
    """Convert Chinese characters to Cangjie-code tokens (``[cj_...]``).

    Needs ``spacy_pkuseg`` for word segmentation and a ``Cangjie5_TC.json``
    glyph→code mapping from HuggingFace (``ResembleAI/chatterbox``).  Without the
    segmenter it still encodes individual glyphs; without the mapping it returns the
    input unchanged with a warning.
    """

    def __init__(self, repo_id: str = "ResembleAI/chatterbox", cache_dir=None):
        self.word2cj: dict = {}
        self.cj2word: dict = {}
        self.segmenter = None
        try:
            import json
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=repo_id, filename="Cangjie5_TC.json", cache_dir=cache_dir
            )
            with open(path, encoding="utf-8") as fp:
                for entry in json.load(fp):
                    word, code = entry.split("\t")[:2]
                    self.word2cj[word] = code
                    self.cj2word.setdefault(code, []).append(word)
        except Exception as e:
            LOG.warning("Could not load Cangjie mapping: %s", e)
        try:
            from spacy_pkuseg import pkuseg
            self.segmenter = pkuseg()
        except ImportError:
            LOG.warning("spacy_pkuseg not installed — Chinese segmentation skipped")

    def _encode_glyph(self, glyph: str):
        code = self.word2cj.get(glyph)
        if code is None:
            return None
        idx = self.cj2word[code].index(glyph)
        return code + (str(idx) if idx > 0 else "")

    def __call__(self, text: str) -> str:
        full = " ".join(self.segmenter.cut(text)) if self.segmenter else text
        out = []
        for t in full:
            if category(t) == "Lo":
                cj = self._encode_glyph(t)
                if cj is None:
                    out.append(t)
                    continue
                out.append("".join(f"[cj_{c}]" for c in cj) + "[cj_.]")
            else:
                out.append(t)
        return "".join(out)


def chinese_to_cangjie(text: str) -> str:
    """Convert Chinese text to Cangjie tokens, reusing a module-level instance."""
    global _cangjie
    if _cangjie is None:
        _cangjie = ChineseCangjieConverter()
    return _cangjie(text)
