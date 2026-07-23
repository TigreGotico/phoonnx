"""
Unified alphabet conversion for phoonnx.

``convert(text, lang, src, tgt, *, phoneme_type=None)`` is the single
entry point for every text-to-model-input transformation.

Phonemization is a special case of alphabet conversion:
  graphemes → IPA/ARPA/SAMPA/…  ==  ``convert(text, lang, GRAPHEMES, IPA)``

The registry maps ``(src_alphabet, tgt_alphabet)`` edges to callables.
Dispatch first checks for a direct edge, then does a BFS over the graph
to compose a chain.  When no path exists the input is returned unchanged
(with a debug log).

Script-conversion edges (HANGUL, HIRA/KANA, CANGJIE) use optional
third-party libraries (pykakasi, spacy-pkuseg).  They are imported lazily
so that the core package remains lightweight.
"""
from __future__ import annotations

import inspect
import logging
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

from phoonnx.config import Alphabet, PhonemeType
from scriptconv.phonemizers.base import PhonemizedChunks

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry type
# ---------------------------------------------------------------------------

# An edge is a callable that accepts (text, lang) and returns a str OR a
# PhonemizedChunks (list[list[str]]).  The phonemization edges return the
# structured list so that the tokenisation contract with existing engines is
# preserved exactly.  All other edges return str.
EdgeFn = Callable  # (text: str, lang: str) -> str | PhonemizedChunks

ALPHABET_CONVERTERS: Dict[Tuple[Alphabet, Alphabet], EdgeFn] = {}


def register_converter(src: Alphabet, tgt: Alphabet, fn: EdgeFn) -> None:
    """Register a direct conversion edge in the graph."""
    ALPHABET_CONVERTERS[(src, tgt)] = fn


# ---------------------------------------------------------------------------
# BFS path finder
# ---------------------------------------------------------------------------

def _find_path(src: Alphabet, tgt: Alphabet) -> Optional[List[Tuple[Alphabet, Alphabet]]]:
    """Return a list of edge-tuples forming the shortest path, or None."""
    if src == tgt:
        return []
    visited = {src}
    queue: deque[Tuple[Alphabet, List]] = deque([(src, [])])
    while queue:
        node, path = queue.popleft()
        for (s, t) in ALPHABET_CONVERTERS:
            if s != node or t in visited:
                continue
            new_path = path + [(s, t)]
            if t == tgt:
                return new_path
            visited.add(t)
            queue.append((t, new_path))
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def convert(text: str,
            lang: str,
            src: Alphabet,
            tgt: Alphabet,
            *,
            phoneme_type: Optional[PhonemeType] = None):
    """
    Convert *text* from alphabet *src* to alphabet *tgt*.

    Parameters
    ----------
    text:
        Input text (or phoneme string, depending on *src*).
    lang:
        BCP-47 language code used by phonemizers and script converters.
    src:
        Alphabet of the incoming text.  Pass ``Alphabet.GRAPHEMES`` for
        normal transcription input.
    tgt:
        Alphabet expected by the model.
    phoneme_type:
        Selects the backend AND tokenisation recipe for the
        graphemes→phoneme family of edges (e.g. ``PhonemeType.ESPEAK``).
        Required when the path traverses a GRAPHEMES→phoneme edge.

    Returns
    -------
    str | PhonemizedChunks
        The converted representation.  Graphemes→phoneme edges return a
        ``PhonemizedChunks`` (``list[list[str]]``) to match the shape
        that ``TTSVoice.synthesize`` already expects.  All other edges
        return a plain ``str``.
    """
    if src == tgt:
        return text

    # Direct edge
    edge_key = (src, tgt)
    if edge_key in ALPHABET_CONVERTERS:
        fn = ALPHABET_CONVERTERS[edge_key]
        return _call_edge(fn, text, lang, phoneme_type)

    # BFS for a multi-hop path
    path = _find_path(src, tgt)
    if path is None:
        LOG.debug(
            "alphabet_convert: no path from %s to %s – returning input unchanged",
            src, tgt,
        )
        return text

    result = text
    for edge_key in path:
        fn = ALPHABET_CONVERTERS[edge_key]
        result = _call_edge(fn, result, lang, phoneme_type)
    return result


def _edge_arity(fn: EdgeFn) -> int:
    """Resolve how many positional arguments *fn* accepts.

    Resolved once per call from the function's signature (not by probing
    with a trial call-and-catch), so a genuine ``TypeError`` raised inside
    the edge body always propagates instead of being swallowed and
    retried.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        # Builtins / C callables without an introspectable signature:
        # assume the full (text, lang, phoneme_type) contract.
        return 3
    params = list(sig.parameters.values())
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        return 3
    positional = [p for p in params
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return len(positional)


def _call_edge(fn: EdgeFn, text, lang: str, phoneme_type: Optional[PhonemeType]):
    """Invoke an edge function with the arity it actually accepts."""
    if _edge_arity(fn) >= 3:
        return fn(text, lang, phoneme_type)
    return fn(text, lang)


# ---------------------------------------------------------------------------
# Phonemization edges  —  THIN FOLD over the existing phonemizer machinery
# ---------------------------------------------------------------------------
# Each edge delegates to get_phonemizer() + phonemizer.phonemize() exactly
# as TTSVoice.phonemize() did before this module existed.  The result shape
# (PhonemizedChunks = list[list[str]]) is preserved so the tokenisation
# contract is unbroken.

def _make_phonemize_edge(pt: PhonemeType,
                         alphabet: Alphabet) -> EdgeFn:
    """
    Build a ``(GRAPHEMES, <phoneme_alphabet>)`` edge that calls the
    existing phonemizer machinery.
    """
    def _edge(text, lang: str, phoneme_type: Optional[PhonemeType] = None) -> PhonemizedChunks:
        # caller may override the phoneme_type at call-time
        effective_pt = phoneme_type if phoneme_type is not None else pt
        from phoonnx.config import get_phonemizer
        phonemizer = get_phonemizer(effective_pt, alphabet)
        return phonemizer.phonemize(text, lang)
    return _edge


# Register every PhonemeType that maps to a phoneme alphabet.
# These are the "graphemes → phoneme alphabet" edges.
_PHONEME_EDGES: List[Tuple[PhonemeType, Alphabet]] = [
    (PhonemeType.ESPEAK, Alphabet.IPA),
    (PhonemeType.GRUUT, Alphabet.IPA),
    (PhonemeType.GORUUT, Alphabet.IPA),
    (PhonemeType.EPITRAN, Alphabet.IPA),
    (PhonemeType.MISAKI, Alphabet.IPA),
    (PhonemeType.MISAKI_EN, Alphabet.IPA),
    (PhonemeType.MISAKI_JA, Alphabet.IPA),
    (PhonemeType.MISAKI_ZH, Alphabet.IPA),
    (PhonemeType.MISAKI_KO, Alphabet.IPA),
    (PhonemeType.MISAKI_VI, Alphabet.IPA),
    (PhonemeType.TRANSPHONE, Alphabet.IPA),
    (PhonemeType.MIRANDESE, Alphabet.IPA),
    (PhonemeType.DEEPPHONEMIZER, Alphabet.IPA),
    (PhonemeType.OPENPHONEMIZER, Alphabet.IPA),
    (PhonemeType.G2PEN, Alphabet.ARPA),
    (PhonemeType.BYT5, Alphabet.IPA),
    (PhonemeType.CHARSIU, Alphabet.IPA),
    (PhonemeType.TUGAPHONE, Alphabet.IPA),
    (PhonemeType.G2PFA, Alphabet.IPA),
    (PhonemeType.PHONIKUD, Alphabet.IPA),
    (PhonemeType.MANTOQ, Alphabet.IPA),
    (PhonemeType.VIPHONEME, Alphabet.IPA),
    (PhonemeType.G2PK, Alphabet.IPA),
    (PhonemeType.KOG2PK, Alphabet.IPA),
    (PhonemeType.G2PC, Alphabet.IPA),
    (PhonemeType.G2PM, Alphabet.IPA),
    (PhonemeType.PYPINYIN, Alphabet.PINYIN),
    (PhonemeType.XPINYIN, Alphabet.PINYIN),
    (PhonemeType.JIEBA, Alphabet.HANZI),
    (PhonemeType.OPENJTALK, Alphabet.IPA),
    (PhonemeType.PYKAKASI, Alphabet.IPA),
    (PhonemeType.CUTLET, Alphabet.IPA),
    (PhonemeType.COTOVIA, Alphabet.COTOVIA),
    (PhonemeType.GRAPHEMES, Alphabet.UNICODE),
    (PhonemeType.UNICODE, Alphabet.UNICODE),
]

# The primary GRAPHEMES→IPA edge uses espeak (the most common backend).
# When `phoneme_type` is supplied at call time, that overrides the backend.
# We register ONE edge per *target alphabet*; the phoneme_type parameter
# at call-time selects which backend to use.
_registered_tgt_alphabets: set = set()
for _pt, _alpha in _PHONEME_EDGES:
    if _alpha not in _registered_tgt_alphabets:
        register_converter(Alphabet.GRAPHEMES, _alpha,
                           _make_phonemize_edge(_pt, _alpha))
        _registered_tgt_alphabets.add(_alpha)


# ---------------------------------------------------------------------------
# Script-conversion edges
# ---------------------------------------------------------------------------

def _graphemes_to_hangul(text: str, lang: str, _pt=None) -> str:
    """Korean: pass through – Hangul input is already Hangul."""
    return text


def _graphemes_to_hiragana(text: str, lang: str, _pt=None) -> str:
    """Japanese: convert kanji/katakana → hiragana via pykakasi."""
    try:
        import pykakasi
        kks = pykakasi.kakasi()
        result = kks.convert(text)
        return "".join(item["hira"] for item in result)
    except ImportError:
        LOG.debug("pykakasi not available; returning text unchanged")
        return text
    except Exception as e:
        LOG.debug("pykakasi conversion failed: %s", e)
        return text


def _graphemes_to_kana(text: str, lang: str, _pt=None) -> str:
    """Japanese: convert to katakana via pykakasi."""
    try:
        import pykakasi
        kks = pykakasi.kakasi()
        result = kks.convert(text)
        return "".join(item["kana"] for item in result)
    except ImportError:
        LOG.debug("pykakasi not available; returning text unchanged")
        return text
    except Exception as e:
        LOG.debug("pykakasi conversion failed: %s", e)
        return text


def _graphemes_to_cangjie(text: str, lang: str, _pt=None) -> str:
    """Chinese: convert to Cangjie codes via spacy-pkuseg (best-effort)."""
    try:
        import pkuseg
        seg = pkuseg.pkuseg()
        words = seg.cut(text)
        # Cangjie encoding is not directly provided by pkuseg; return segmented.
        # A full Cangjie lookup table would be needed for complete conversion.
        return " ".join(words)
    except ImportError:
        LOG.debug("spacy-pkuseg not available; returning text unchanged")
        return text
    except Exception as e:
        LOG.debug("Cangjie conversion failed: %s", e)
        return text


register_converter(Alphabet.GRAPHEMES, Alphabet.HANGUL, _graphemes_to_hangul)
register_converter(Alphabet.GRAPHEMES, Alphabet.HIRA, _graphemes_to_hiragana)
register_converter(Alphabet.GRAPHEMES, Alphabet.KANA, _graphemes_to_kana)
register_converter(Alphabet.GRAPHEMES, Alphabet.CANGJIE, _graphemes_to_cangjie)


# ---------------------------------------------------------------------------
# Phoneme-notation edges  —  delegated to scriptconv
# ---------------------------------------------------------------------------
# These are pure symbol transcodes between phonemic notations (no language,
# no phonemization). scriptconv is the single source of truth; each notation
# converts to and from IPA, so the BFS above can chain them (e.g.
# X-SAMPA → IPA → ARPA). Alphabet .value strings match scriptconv's Notation
# values exactly. Grapheme input never reaches these edges — it is phonemized
# by the model's own phonemizer (see TTSVoice.synthesize).

def _make_scriptconv_edge(src_value: str, tgt_value: str) -> EdgeFn:
    def _edge(text, lang: str = "", _pt: Optional[PhonemeType] = None) -> str:
        from scriptconv.notation import convert as _sc_convert
        return _sc_convert(text, src_value, tgt_value)
    return _edge


for _alpha in (Alphabet.ARPA, Alphabet.XSAMPA, Alphabet.RFE, Alphabet.COTOVIA):
    register_converter(Alphabet.IPA, _alpha, _make_scriptconv_edge("ipa", _alpha.value))
    register_converter(_alpha, Alphabet.IPA, _make_scriptconv_edge(_alpha.value, "ipa"))


# Kana transliteration (orthographic, no phonemization) via scriptconv.
def _hira_to_kana_edge(text, lang: str = "", _pt=None) -> str:
    from scriptconv.translit import hira_to_kana
    return hira_to_kana(text)


def _kana_to_hira_edge(text, lang: str = "", _pt=None) -> str:
    from scriptconv.translit import kana_to_hira
    return kana_to_hira(text)


register_converter(Alphabet.HIRA, Alphabet.KANA, _hira_to_kana_edge)
register_converter(Alphabet.KANA, Alphabet.HIRA, _kana_to_hira_edge)
