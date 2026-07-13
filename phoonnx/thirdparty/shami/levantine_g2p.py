"""Rule-based grapheme-to-phoneme (G2P) for Levantine Arabic.

Ported from hams_tts.text.levantine_g2p (Apache-2.0).
"""

from __future__ import annotations

import re
from typing import List, Tuple

from . import espeak
from .phoneme_inventory import fold_to_inventory, tokenize_ipa

FATHA, KASRA, DAMMA = "َ", "ِ", "ُ"
SUKUN, SHADDA = "ْ", "ّ"
TANWIN_F, TANWIN_K, TANWIN_D = "ً", "ٍ", "ٌ"
DAGGER_ALEF = "ٰ"
HARAKAT = {FATHA, KASRA, DAMMA, SUKUN, SHADDA, TANWIN_F, TANWIN_K, TANWIN_D, DAGGER_ALEF}

ALEF, ALEF_MAQSURA, ALEF_MADDA = "ا", "ى", "آ"
ALEF_HAMZA_ABOVE, ALEF_HAMZA_BELOW = "أ", "إ"
HAMZA, WAW_HAMZA, YEH_HAMZA = "ء", "ؤ", "ئ"
TEH_MARBUTA, HAMZAT_WASL = "ة", "ٱ"
WAW, YEH, LAM = "و", "ي", "ل"

CONS = {
    "ب": "b",
    "ت": "t",
    "ث": "t",
    "ج": "ʒ",
    "ح": "ħ",
    "خ": "x",
    "د": "d",
    "ذ": "d",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "ʃ",
    "ص": "sˤ",
    "ض": "dˤ",
    "ط": "tˤ",
    "ظ": "zˤ",
    "ع": "ʕ",
    "غ": "ɣ",
    "ف": "f",
    "ق": "ʔ",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "ة": "t",
    HAMZA: "ʔ", WAW_HAMZA: "ʔ", YEH_HAMZA: "ʔ",
}

SUN_LETTERS = set("تثدذرزسشصضطظلن")
EMPHATICS_IPA = {"sˤ", "dˤ", "tˤ", "zˤ"}
GUTTURALS_IPA = {"ħ", "ʕ", "x", "ɣ", "ʔ", "q", "h"}
CONSONANT_IPA = {
    "b", "t", "d", "k", "ɡ", "q", "ʔ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
    "x", "ɣ", "ħ", "ʕ", "m", "n", "ŋ", "l", "ɫ", "r", "ɾ", "ɹ", "w", "j",
    "sˤ", "dˤ", "tˤ", "zˤ", "t͡ʃ", "d͡ʒ",
}

LEXICON = {
    "الله": "ʔaɫɫa",
    "هذا": "haːda", "هاد": "haːd", "هيدا": "heːda",
    "هذه": "haːde", "هاي": "haj", "هيدي": "heːde",
    "ذلك": "haˈdaːk", "هداك": "haˈdaːk",
    "الذي": "ʔilli", "اللي": "ʔilli",
    "شو": "ʃuː", "ليش": "leːʃ", "كيف": "kiːf", "وين": "weːn",
    "هلق": "hallaʔ", "هلأ": "hallaʔ", "هسا": "hassa",
    "مش": "miʃ", "مو": "muː",
    "بدي": "ˈbiddi", "بدك": "ˈbiddak", "بدنا": "ˈbidna",
    "كتير": "ktiːr", "هيك": "heːk", "هون": "hoːn",
    "إيه": "ʔeː", "أيوا": "ʔaˈjwa", "لأ": "laʔ",
    "عشان": "ʕaˈʃaːn", "منشان": "minˈʃaːn", "بس": "bas",
    "في": "fiː", "مين": "miːn", "إمتى": "ʔemta", "قديش": "ʔaˈdeːʃ", "أديش": "ʔaˈdeːʃ",
}

_AR_LETTER_RE = re.compile(r"[ء-يٱ]")


def _strip_haraka(s: str) -> str:
    return "".join(c for c in s if c not in HARAKAT and c != "ـ")


def _apply_article(word: str) -> Tuple[List[str], str, bool]:
    bare = _strip_haraka(word)
    if bare.startswith("ال") or word.startswith(HAMZAT_WASL + LAM) or bare.startswith("ٱل"):
        li = word.find(LAM)
        if li == -1:
            return [], word, False
        rest = word[li + 1:]
        j = 0
        while j < len(rest) and rest[j] in HARAKAT:
            j += 1
        rest = rest[j:]
        m = _AR_LETTER_RE.search(rest)
        if not m:
            return ["ʔ", "i", "l"], rest, False
        first = rest[m.start()]
        if first in SUN_LETTERS:
            return ["ʔ", "i"], rest, True
        return ["ʔ", "i", "l"], rest, False
    return [], word, False


def _g2p_word(word: str) -> List[str]:
    bare = _strip_haraka(word)
    if bare in LEXICON:
        return tokenize_ipa(LEXICON[bare]).symbols

    prefix, word, geminate_first = _apply_article(word)
    phones: List[str] = list(prefix)
    chars = list(word)
    n = len(chars)
    i = 0
    first_cons_emitted = not geminate_first

    def _emit_cons(c_ipa: str) -> None:
        nonlocal first_cons_emitted
        phones.append(c_ipa)
        if not first_cons_emitted:
            phones.append(c_ipa)
            first_cons_emitted = True

    while i < n:
        ch = chars[i]
        nxt = chars[i + 1] if i + 1 < n else ""

        if ch == ALEF_MADDA:
            phones += ["ʔ", "aː"]
            i += 1
            continue
        if ch in (ALEF_HAMZA_ABOVE, ALEF_HAMZA_BELOW):
            phones.append("ʔ")
            v = _vowel_for(nxt)
            if v:
                phones.append(v)
                i += 2
                continue
            phones.append("i" if ch == ALEF_HAMZA_BELOW else "a")
            i += 1
            continue
        if ch == ALEF or ch == HAMZAT_WASL:
            if i == 0:
                v = _vowel_for(nxt)
                if v:
                    phones += ["ʔ", v]
                    i += 2
                    continue
                i += 1
                continue
            phones.append("aː")
            i += 1
            continue
        if ch == ALEF_MAQSURA:
            phones.append("aː")
            i += 1
            continue

        if ch == TEH_MARBUTA:
            if i == n - 1 or (i == n - 2 and chars[i + 1] in HARAKAT):
                prev = _last_phone(phones)
                phones.append("a" if (prev in EMPHATICS_IPA or prev in {"ʕ", "ħ", "x", "ɣ", "q", "r"}) else "e")
            else:
                phones.append("t")
            i += 1
            continue

        if ch == WAW:
            phones.append("w")
            i += 1
            i = _consume_haraka(chars, i, phones)
            continue
        if ch == YEH:
            phones.append("j")
            i += 1
            i = _consume_haraka(chars, i, phones)
            continue

        if ch in CONS:
            c_ipa = CONS[ch]
            _emit_cons(c_ipa)
            i += 1
            if i < n and chars[i] == SHADDA:
                phones.append(c_ipa)
                i += 1
            i = _consume_haraka(chars, i, phones)
            continue

        i += 1

    return phones


def _vowel_for(mark: str) -> str:
    return {FATHA: "a", KASRA: "i", DAMMA: "u"}.get(mark, "")


def _last_phone(phones: List[str]) -> str:
    return phones[-1] if phones else ""


def _consume_haraka(chars: List[str], i: int, phones: List[str]) -> int:
    n = len(chars)
    if i >= n:
        return i
    mark = chars[i]
    nxt = chars[i + 1] if i + 1 < n else ""

    if mark == FATHA:
        if nxt == WAW and _is_glide(chars, i + 1):
            phones.append("oː")
            return i + 2
        if nxt == YEH and _is_glide(chars, i + 1):
            phones.append("eː")
            return i + 2
        if nxt in (ALEF, ALEF_MAQSURA, ALEF_MADDA):
            phones.append("aː")
            return i + 2
        phones.append("a")
        return i + 1
    if mark == KASRA:
        if nxt == YEH and _is_glide(chars, i + 1):
            phones.append("iː")
            return i + 2
        phones.append("i")
        return i + 1
    if mark == DAMMA:
        if nxt == WAW and _is_glide(chars, i + 1):
            phones.append("uː")
            return i + 2
        phones.append("u")
        return i + 1
    if mark == SUKUN:
        return i + 1
    if mark == DAGGER_ALEF:
        phones.append("aː")
        return i + 1
    if mark in (TANWIN_F, TANWIN_K, TANWIN_D):
        phones.append({TANWIN_F: "a", TANWIN_K: "i", TANWIN_D: "u"}[mark])
        return i + 1
    return i


def _is_glide(chars: List[str], idx: int) -> bool:
    nxt = chars[idx + 1] if idx + 1 < len(chars) else ""
    return nxt in ("", SUKUN) or nxt not in (FATHA, KASRA, DAMMA)


def _emphatic_backing(phones: List[str]) -> List[str]:
    out = list(phones)
    for idx, p in enumerate(out):
        if p in ("a", "aː"):
            window = out[max(0, idx - 2): idx + 3]
            if any(w in EMPHATICS_IPA or w == "q" for w in window):
                out[idx] = "ɑː" if p == "aː" else "ɑ"
    return out


def _epenthesis(phones: List[str]) -> List[str]:
    out: List[str] = []
    run = 0
    for p in phones:
        is_cons = p in CONSONANT_IPA
        if is_cons:
            run += 1
            if run >= 3:
                out.insert(len(out), "e")
                run = 1
        else:
            run = 0
        out.append(p)
    return out


def levantine_g2p(text: str, back_emphatics: bool = True, epenthesize: bool = True) -> str:
    """Convert diacritised Levantine Arabic text to an IPA string."""
    words = text.split()
    rendered: List[str] = []
    for w in words:
        phones = _g2p_word(w)
        if back_emphatics:
            phones = _emphatic_backing(phones)
        if epenthesize:
            phones = _epenthesis(phones)
        rendered.append("".join(phones))
    return " ".join(r for r in rendered if r)


_MSA_TO_LEV = [
    ("d͡ʒ", "ʒ"), ("dʒ", "ʒ"),
    ("q", "ʔ"),
    ("ðˤ", "zˤ"), ("ð", "d"), ("θ", "t"),
]


def msa_ipa_to_levantine(ipa: str) -> str:
    for a, b in _MSA_TO_LEV:
        ipa = ipa.replace(a, b)
    return ipa


def arabic_fallback_ipa(text: str, back_emphatics: bool = True) -> str:
    """Phonemise undiacritised Arabic via espeak-ng, then remap toward Levantine."""
    if not espeak.available():
        return levantine_g2p(text, back_emphatics=back_emphatics)
    raw = espeak.phonemize(text, voice="ar")
    raw = msa_ipa_to_levantine(raw)
    out_words = []
    for word in raw.split(" "):
        toks = tokenize_ipa(word).symbols
        out_words.append("".join(fold_to_inventory(t) for t in toks))
    ipa = " ".join(w for w in out_words if w)
    return ipa
