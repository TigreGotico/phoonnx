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
    Deterministically tokenizes a Mantoq string into Mantoq tokens and punctuation.
    
    Recognized tokens (in order of priority): the doubling marker "_dbl_", the word separator "_+_", the long vowel "aaaa", the long/lengthened vowels "aa", "ii", "uu", single-character consonants and vowels (from the module's CONSONANTS and VOWELS mappings), the glottal-stop marker "<" (returned as "ʔ"), and punctuation characters (from PUNCTUATION). Any unrecognized character is returned as a single-character token.
    
    Parameters:
        text (str): Input Mantoq-formatted text.
    
    Returns:
        list[str]: Ordered list of tokens representing the input sequence.
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
    Apply the Mantoq doubling rule to the IPA for a previously processed token.
    
    Parameters:
        prev_token (str): The original Mantoq token that was converted to `prev_ipa`.
        prev_ipa (str): The IPA string produced for `prev_token` before doubling.
    
    Returns:
        str: The IPA string after applying doubling:
            - If `prev_token` is a vowel token: append the length marker "ː" (an additional "ː" is appended even if one is already present).
            - If `prev_token` is a consonant token: append "ː" unless `prev_ipa` already ends with "ː".
            - Otherwise: return `prev_ipa` unchanged.
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


def mantoq_to_ipa(text, add_stress=True):
    """
    Convert a Mantoq-formatted string into its IPA transcription.
    
    Tokenizes the input deterministically, applies the Mantoq doubling rule for the "_dbl_" token, treats "_+_" as an explicit word separator, maps vowel and consonant tokens via the module's VOWELS and CONSONANTS tables, and passes punctuation or unknown characters through unchanged.
    
    Parameters:
        text (str): Input string in Mantoq orthography.
        add_stress (bool): If true, include stress-related markers in the output when available (default True).
    
    Returns:
        str: The assembled IPA transcription of the input.
    """
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