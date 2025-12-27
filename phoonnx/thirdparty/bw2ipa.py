# translate.py
# Deterministic Mantoq → IPA converter
# Assumes the input is already tokenized using the Mantoq inventory.

# ---------------------------------------------------------------------------
# Token → IPA maps
# ---------------------------------------------------------------------------

CONSONANTS = {
    "b":  "b",
    "t":  "t",
    "^":  "θ",
    "j":  "d͡ʒ",
    "H":  "ħ",
    "x":  "x",
    "d":  "d",
    "*":  "ð",
    "r":  "r",
    "z":  "z",
    "s":  "s",
    "$":  "ʃ",
    "S":  "sˤ",
    "D":  "dˤ",
    "T":  "tˤ",
    "Z":  "ðˤ",
    "E":  "ʕ",
    "g":  "ɣ",
    "f":  "f",
    "q":  "q",
    "k":  "k",
    "l":  "l",
    "m":  "m",
    "n":  "n",
    "h":  "h",
    "w":  "w",
    "y":  "j",
    "v":  "v"
}

VOWELS = {
    "a":    "a",
    "aa":   "aː",
    "aaaa": "aːː",
    "i":    "i",
    "ii":   "iː",
    "u":    "u",
    "uu":   "uː",
}


# Punctuation is passed through unchanged:
PUNCTUATION = set(list(".,;:!?()[]{}\"'"))

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize_mantoq(text):
    """
    Tokenize Mantoq string deterministically.

    Priority:
        1. _dbl_
        2. aaaa
        3. ii / uu / aa
        4. single char tokens
        5. punctuation and raw chars are passed through
    """
    tokens = []
    i = 0
    L = len(text)

    while i < L:

        # doubling marker
        if text.startswith("_dbl_", i):
            tokens.append("_dbl_")
            i += 5
            continue

        # word separator
        if text.startswith("_+_", i):
            tokens.append("_+_")
            i += 3
            continue

        # longest vowel first
        if text.startswith("aaaa", i):
            tokens.append("aaaa")
            i += 4
            continue

        if text.startswith("aa", i):
            tokens.append("aa")
            i += 2
            continue

        if text.startswith("ii", i):
            tokens.append("ii")
            i += 2
            continue

        if text.startswith("uu", i):
            tokens.append("uu")
            i += 2
            continue

        # single-character consonant or vowel
        ch = text[i]

        if ch in CONSONANTS or ch in VOWELS:
            tokens.append(ch)
            i += 1
            continue
        if ch == "<":
            tokens.append("ʔ")
            i += 1
            continue

        # punctuation
        if ch in PUNCTUATION:
            tokens.append(ch)
            i += 1
            continue

        # fallback: pass through unknown characters
        tokens.append(ch)
        i += 1

    return tokens

# ---------------------------------------------------------------------------
# IPA Assembly
# ---------------------------------------------------------------------------

def apply_doubling(prev_token, prev_ipa):
    """
    Mantoq doubling rule:
       - If previous token is a vowel token: lengthen it.
       - If previous token is a consonant: mark gemination using ː.
    """
    if prev_token in VOWELS:
        # ensure single long marker; long tokens already contain ː
        if prev_ipa.endswith("ː"):
            return prev_ipa + "ː"
        return prev_ipa + "ː"

    if prev_token in CONSONANTS:
        # consonant gemination: use length mark, not duplication
        if prev_ipa.endswith("ː"):
            return prev_ipa  # already geminated
        return prev_ipa + "ː"

    return prev_ipa


def mantoq_to_ipa(text):
    tokens = tokenize_mantoq(text)

    ipa_out = []
    last_token = None
    last_ipa = None

    for tok in tokens:

        # doubling applies to the previous symbol
        if tok == "_dbl_":
            if last_token is None:
                continue
            new_ipa = apply_doubling(last_token, last_ipa)
            ipa_out[-1] = new_ipa
            last_ipa = new_ipa
            continue

        # explicit word separation
        if tok == "_+_":
            ipa_out.append(" ")
            last_token = tok
            last_ipa = " "
            continue

        # vowels
        if tok in VOWELS:
            ipa_val = VOWELS[tok]
            ipa_out.append(ipa_val)
            last_token = tok
            last_ipa = ipa_val
            continue

        # consonants
        if tok in CONSONANTS:
            ipa_val = CONSONANTS[tok]
            ipa_out.append(ipa_val)
            last_token = tok
            last_ipa = ipa_val
            continue

        # punctuation and fallthrough
        ipa_out.append(tok)
        last_token = tok
        last_ipa = tok

    return "".join(ipa_out)


# backwards compat alias
bw2ipa = mantoq_to_ipa