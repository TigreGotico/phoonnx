"""Unified IPA phoneme inventory for Levantine-Arabic / English code-switching TTS.

Ported from hams_tts.text.phoneme_inventory (Apache-2.0) with adaptations for
phoonnx.  The inventory, language IDs and tokenisation logic are identical to the
upstream implementation so exported ShamiVITS checkpoints remain compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple


class Lang(IntEnum):
    """Language-ID stream values (fed to the model as an embedding index)."""

    PAD = 0
    AR = 1  # Levantine Arabic
    EN = 2  # English
    NEUTRAL = 3  # punctuation, silence, boundaries, digits handled upstream


# --------------------------------------------------------------------------------------
# Special / structural symbols
# --------------------------------------------------------------------------------------
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
WORD_SEP = " "  # white space = word boundary (kept, it conditions prosody)
SYL_SEP = "."  # syllable boundary (optional, emitted by syllabifier)
# Punctuation that the acoustic model is allowed to "see" (drives pausing/prosody).
PUNCT = [",", ".", "?", "!", ";", ":", "…", "—", "(", ")", '"', "«", "»"]

SPECIALS = [PAD, BOS, EOS, UNK, WORD_SEP, SYL_SEP]

# --------------------------------------------------------------------------------------
# Suprasegmentals (order-independent, but must be in the inventory to be tokenisable)
# --------------------------------------------------------------------------------------
SUPRASEGMENTALS = [
    "ˈ",  # primary stress (U+02C8)
    "ˌ",  # secondary stress (U+02CC)
    "ː",  # length mark (U+02D0)
]

# --------------------------------------------------------------------------------------
# Consonants -- union of the Levantine-Arabic and English inventories, in IPA.
# Multi-codepoint symbols (tie bars, pharyngealisation) are listed explicitly so the
# longest-match tokeniser can find them.
# --------------------------------------------------------------------------------------
CONSONANTS = [
    # --- shared / English-leaning ---
    "p", "b", "t", "d", "k", "ɡ",          # plosives
    "t͡ʃ", "d͡ʒ",                            # affricates
    "f", "v", "θ", "ð", "s", "z",          # fricatives
    "ʃ", "ʒ", "h", "x", "ɣ",               # more fricatives
    "m", "n", "ŋ",                          # nasals
    "l", "ɫ",                               # laterals
    "r", "ɾ", "ɹ",                          # rhotics
    "w", "j",                               # glides
    # --- Arabic-specific (Levantine) ---
    "ʔ",   # glottal stop
    "q",   # uvular plosive
    "ħ",   # voiceless pharyngeal fricative
    "ʕ",   # voiced pharyngeal fricative
    "sˤ",  # emphatic s
    "dˤ",  # emphatic d
    "tˤ",  # emphatic t
    "ðˤ",  # emphatic interdental
    "zˤ",  # emphatic z
]

# --------------------------------------------------------------------------------------
# Vowels
# --------------------------------------------------------------------------------------
VOWELS = [
    # short
    "a", "i", "u", "e", "o", "ə",
    # English lax / extra qualities
    "ɪ", "ʊ", "ɛ", "æ", "ʌ", "ɑ", "ɒ", "ɔ", "ɜ", "ɐ",
    # long
    "aː", "iː", "uː", "eː", "oː", "ɑː", "ɔː", "ɜː",
    # diphthongs
    "a͡ɪ", "a͡ʊ", "e͡ɪ", "o͡ʊ", "ɔ͡ɪ",
]

# --------------------------------------------------------------------------------------
# Final ordered symbol table.  ORDER IS FROZEN: the model's embedding rows are indexed
# by this order.
# --------------------------------------------------------------------------------------
SYMBOLS: List[str] = (
    SPECIALS
    + PUNCT
    + SUPRASEGMENTALS
    + CONSONANTS
    + VOWELS
)

_seen: set = set()
_ordered: List[str] = []
for _s in SYMBOLS:
    if _s not in _seen:
        _seen.add(_s)
        _ordered.append(_s)
SYMBOLS = _ordered

SYMBOL_TO_ID: Dict[str, int] = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL: Dict[int, str] = {i: s for s, i in SYMBOL_TO_ID.items()}
VOCAB_SIZE: int = len(SYMBOLS)

_MULTI_FIRST: List[str] = sorted(
    [s for s in SYMBOLS if s not in (PAD, BOS, EOS, UNK)],
    key=lambda s: (-len(s), s),
)

FOLD_MAP: Dict[str, str] = {
    "g": "ɡ",
    "ɡ": "ɡ",
    "c": "k",
    "y": "j",
    "ʤ": "d͡ʒ",
    "ʧ": "t͡ʃ",
    "dʒ": "d͡ʒ",
    "tʃ": "t͡ʃ",
    "aɪ": "a͡ɪ",
    "aʊ": "a͡ʊ",
    "eɪ": "e͡ɪ",
    "oʊ": "o͡ʊ",
    "ɔɪ": "ɔ͡ɪ",
    "ɝ": "ɜː",
    "ɚ": "ə",
    "ɡ̃": "ŋ",
    "ʁ": "ɣ",
    "χ": "x",
    "ɹ̩": "ɹ",
    "ɫ̩": "ɫ",
    "ɐ": "ə",
}


@dataclass
class TokenizedPhonemes:
    """Result of tokenising an IPA string against the inventory."""

    symbols: List[str]
    ids: List[int]

    def __len__(self) -> int:
        return len(self.ids)


def fold_to_inventory(symbol: str) -> str:
    """Map an arbitrary symbol to the nearest in-inventory symbol (or UNK)."""
    if symbol in SYMBOL_TO_ID:
        return symbol
    if symbol in FOLD_MAP:
        return FOLD_MAP[symbol]
    stripped = symbol.replace("͡", "").replace("ː", "")
    if stripped in SYMBOL_TO_ID:
        return stripped
    if stripped in FOLD_MAP:
        return FOLD_MAP[stripped]
    return UNK


def tokenize_ipa(ipa: str) -> TokenizedPhonemes:
    """Greedy longest-match tokenisation of an IPA string into inventory symbols."""
    out_symbols: List[str] = []
    i = 0
    n = len(ipa)
    while i < n:
        ch = ipa[i]
        if ch.isspace():
            if out_symbols and out_symbols[-1] != WORD_SEP:
                out_symbols.append(WORD_SEP)
            i += 1
            continue
        matched = None
        for cand in _MULTI_FIRST:
            if cand and ipa.startswith(cand, i):
                matched = cand
                break
        if matched is not None:
            out_symbols.append(matched)
            i += len(matched)
        else:
            out_symbols.append(fold_to_inventory(ch))
            i += 1
    cleaned: List[str] = []
    for s in out_symbols:
        if s == WORD_SEP and (not cleaned or cleaned[-1] == WORD_SEP):
            continue
        cleaned.append(s)
    while cleaned and cleaned[-1] == WORD_SEP:
        cleaned.pop()
    ids = [SYMBOL_TO_ID[s] for s in cleaned]
    return TokenizedPhonemes(symbols=cleaned, ids=ids)


def encode(ipa: str, add_bos_eos: bool = True) -> Tuple[List[int], List[str]]:
    """Tokenise + (optionally) wrap with BOS/EOS."""
    tok = tokenize_ipa(ipa)
    symbols = tok.symbols
    if add_bos_eos:
        symbols = [BOS] + symbols + [EOS]
    ids = [SYMBOL_TO_ID[s] for s in symbols]
    return ids, symbols


def decode(ids: List[int]) -> str:
    """Inverse of :func:`encode` for debugging."""
    out = []
    for i in ids:
        s = ID_TO_SYMBOL.get(i, UNK)
        if s in (BOS, EOS, PAD):
            continue
        out.append(s)
    return "".join(out)
