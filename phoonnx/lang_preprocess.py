"""Per-language script transforms for multilingual TTS front ends.

Some scripts are ambiguous about pronunciation, so the model is trained on a transformed
form of the text that must be reproduced at inference. Korean (Hangul → Jamo) is pure
Python and always available; Japanese (scriptconv[ja]) and Chinese (spacy-pkuseg) need an
optional dependency; Russian stress comes from ``stressonnx`` (pure onnxruntime). Each
degrades to the raw text (with a warning) when its backend is missing. Hebrew/Arabic
vocalization is the universal ``add_diacritics`` flag, not here.

Ported from Resemble AI's Chatterbox multilingual tokenizer. Currently consumed by
``phoonnx.tokenizer.ChatterboxMTLTokenizer``; kept standalone so other engines can reuse
the transforms (a later pass will fold text preprocessing into a shared config).
"""
from unicodedata import category, normalize

from phoonnx.util import LOG

# lazy singletons for the optional backends (created on first use)
_russian_stresser = None
_cangjie = None


def is_kanji(c: str) -> bool:
    return "一" <= c <= "鿿"


def is_katakana(c: str) -> bool:
    return "ァ" <= c <= "ヺ"


def hangul_to_jamo(text: str) -> str:
    """Decompose Korean syllables into conjoining Jamo (scriptconv)."""
    from scriptconv import decompose_hangul
    return decompose_hangul(text, form="conjoining").strip()


def japanese_to_hiragana(text: str) -> str:
    """Convert kanji to hiragana (katakana kept) and NFKD-normalise.

    Readings come from scriptconv's token API; this function only applies
    phoonnx policy on top: a space is inserted before kanji readings starting
    は/へ so the phonemizer sees them word-initially (pronounced ha/he) instead
    of mistaking them for the topic/direction particles (wa/e).
    """
    try:
        from scriptconv.readings import tokens
        converted = list(tokens(text))
    except ImportError:
        LOG.warning("scriptconv[ja] not installed — Japanese text left unprocessed")
        return text
    except Exception as e:
        LOG.warning("Japanese conversion failed: %s", e)
        return text
    out = []
    for tok in converted:
        if any(is_kanji(c) for c in tok.orig):
            hira = tok.hira
            if hira and hira[0] in ("は", "へ"):   # は / へ
                hira = " " + hira
            out.append(hira)
        else:
            out.append(tok.orig)
    return normalize("NFKD", "".join(out))


def russian_add_stress(text: str) -> str:
    """Add stress marks (so the model knows vowel reduction) via ``stressonnx`` — a
    pure-onnxruntime stressor (no torch at runtime)."""
    global _russian_stresser
    try:
        if _russian_stresser is None:
            from stressonnx import stress
            _russian_stresser = stress
        return _russian_stresser(text, "ru")
    except ImportError:
        LOG.warning("stressonnx not installed — Russian stress skipped")
        return text
    except Exception as e:
        LOG.warning("Russian stress labeling failed: %s", e)
        return text


class ChineseCangjieConverter:
    """Convert Chinese characters to Cangjie-code tokens (``[cj_...]``).

    Needs ``spacy_pkuseg`` for word segmentation and a ``Cangjie5_TC.json`` glyph→code
    mapping (downloaded from HuggingFace). Without the segmenter it still encodes glyphs,
    just without word boundaries.
    """

    def __init__(self, repo_id: str = "ResembleAI/chatterbox", cache_dir=None):
        self.word2cj = {}
        self.cj2word = {}
        self.segmenter = None
        try:
            import json
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=repo_id, filename="Cangjie5_TC.json", cache_dir=cache_dir)
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
        if code is None:                              # e.g. a non-mapped (Japanese) glyph
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
    """Convert Chinese text to Cangjie tokens, reusing a single converter instance."""
    global _cangjie
    if _cangjie is None:
        try:
            _cangjie = ChineseCangjieConverter()
        except Exception as e:
            LOG.warning("Could not initialize Cangjie converter: %s", e)
            return text
    try:
        return _cangjie(text)
    except Exception as e:
        LOG.warning("Cangjie conversion failed: %s", e)
        return text


# language code -> transform; languages absent here need no script transform.
# Hebrew/Arabic diacritization is NOT here — it is the universal ``add_diacritics``
# SynthesisConfig flag, applied before encode_text (set it on those voices' configs).
SCRIPT_TRANSFORMS = {
    "ko": hangul_to_jamo,
    "ja": japanese_to_hiragana,
    "ru": russian_add_stress,
    "zh": chinese_to_cangjie,
}
