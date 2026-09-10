import unicodedata

from phoonnx.lang_preprocess import hangul_to_jamo, SCRIPT_TRANSFORMS

# a precomposed Hangul syllable (U+D55C) and one without a final consonant (U+AC00)
HAN = unicodedata.normalize("NFC", "한")
GA = unicodedata.normalize("NFC", "가")


def test_hangul_to_jamo_decomposes():
    # decomposing a Hangul syllable yields its conjoining-jamo (NFD) form
    jamo = hangul_to_jamo(HAN)
    assert jamo == unicodedata.normalize("NFD", HAN)
    assert all("ᄀ" <= c <= "ᇿ" for c in jamo)      # all conjoining jamo
    assert hangul_to_jamo("abc") == "abc"                     # non-Hangul untouched


def test_script_transform_table():
    # script transforms (Hebrew/Arabic diacritics are the universal add_diacritics flag,
    # not a tokenizer transform); other languages need none
    assert set(SCRIPT_TRANSFORMS) == {"ko", "ja", "ru", "zh"}
    assert SCRIPT_TRANSFORMS["ko"](GA) == unicodedata.normalize("NFD", GA)


def test_mtl_tokenizer_applies_korean_transform():
    from phoonnx.tokenizer import ChatterboxMTLTokenizer
    tok = ChatterboxMTLTokenizer.__new__(ChatterboxMTLTokenizer)
    seen = {}

    class _Enc:
        def __init__(self, ids): self.ids = ids

    class _Tok:
        def token_to_id(self, t): return 1 if t in ("[ko]", "[SPACE]") else None
        def encode(self, text): seen["text"] = text; return _Enc([1])

    tok._tok = _Tok()
    tok.tokenize(GA, language="ko")             # NFKD splits the syllable; ko transform is then a no-op
    assert seen["text"] == "[ko]" + unicodedata.normalize("NFD", GA)
