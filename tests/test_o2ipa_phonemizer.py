"""
Tests for the orthography2ipa-backed phonemizer (Orthography2IPAPhonemizer).
"""
import pytest

pytest.importorskip("orthography2ipa")


# ---------------------------------------------------------------------------
# Basic transcription — pt / gl / es / ar / he
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang,text", [
    ("pt", "olá"),
    ("gl", "lingua galega"),
    ("es", "hola mundo"),
    ("ar", "مرحبا"),
    ("he", "שלום"),
])
def test_transcribe_non_empty_ipa(lang, text):
    """transcribe() for known languages should return a non-empty IPA string."""
    from phoonnx.phonemizers.o2ipa import Orthography2IPAPhonemizer
    p = Orthography2IPAPhonemizer()
    result = p.phonemize_string(text, lang)
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert len(result.strip()) > 0, f"Empty result for {lang!r}: {result!r}"


def test_pt_ipa_contains_vowels():
    from phoonnx.phonemizers.o2ipa import Orthography2IPAPhonemizer
    p = Orthography2IPAPhonemizer()
    result = p.phonemize_string("olá", "pt")
    assert any(c in result for c in "aeiouɐɛɔ")


def test_es_ipa_hola():
    from phoonnx.phonemizers.o2ipa import Orthography2IPAPhonemizer
    p = Orthography2IPAPhonemizer()
    result = p.phonemize_string("hola", "es")
    # basic smoke: should contain 'o' and 'l'
    assert "o" in result
    assert "l" in result


# ---------------------------------------------------------------------------
# Alphabet
# ---------------------------------------------------------------------------

def test_alphabet_is_ipa():
    from phoonnx.phonemizers.o2ipa import Orthography2IPAPhonemizer
    from phoonnx.config import Alphabet
    p = Orthography2IPAPhonemizer()
    assert p.alphabet == Alphabet.IPA


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

def test_factory_dispatch():
    from phoonnx.config import PhonemeType, get_phonemizer
    from phoonnx.phonemizers.o2ipa import Orthography2IPAPhonemizer
    p = get_phonemizer(PhonemeType.ORTHOGRAPHY2IPA)
    assert isinstance(p, Orthography2IPAPhonemizer)


# ---------------------------------------------------------------------------
# Unknown language
# ---------------------------------------------------------------------------

def test_unknown_lang_raises():
    from phoonnx.phonemizers.o2ipa import Orthography2IPAPhonemizer
    p = Orthography2IPAPhonemizer()
    with pytest.raises((ValueError, KeyError)):
        p.phonemize_string("test", "xyz-completely-unknown-lang-code")


# ---------------------------------------------------------------------------
# Engine cache — calling twice with same lang reuses cached engine
# ---------------------------------------------------------------------------

def test_engine_cache():
    from phoonnx.phonemizers.o2ipa import Orthography2IPAPhonemizer
    p = Orthography2IPAPhonemizer()
    _ = p.phonemize_string("hola", "es")
    _ = p.phonemize_string("mundo", "es")
    # resolve("es") → "es-ES"; the cache stores resolved codes
    resolved = Orthography2IPAPhonemizer.get_lang("es")
    assert resolved in p._cache


# ---------------------------------------------------------------------------
# supported_langs
# ---------------------------------------------------------------------------

def test_supported_langs_count():
    from phoonnx.phonemizers.o2ipa import Orthography2IPAPhonemizer
    langs = Orthography2IPAPhonemizer.supported_langs()
    assert len(langs) >= 300


def test_supported_langs_includes_common():
    from phoonnx.phonemizers.o2ipa import Orthography2IPAPhonemizer
    langs = Orthography2IPAPhonemizer.supported_langs()
    # orthography2ipa stores fully-qualified codes (gl, gl-ES, es-ES, pt-PT, …);
    # check a representative set that are known to be directly in the list.
    assert "gl" in langs, f"Expected 'gl' in supported_langs"
    assert "ar" in langs, f"Expected 'ar' in supported_langs"
    assert "he" in langs, f"Expected 'he' in supported_langs"
    # es / pt are stored as es-ES / pt-PT — ensure at least one variant exists
    assert any(c.startswith("es") for c in langs), "Expected 'es*' in supported_langs"
    assert any(c.startswith("pt") for c in langs), "Expected 'pt*' in supported_langs"


# ---------------------------------------------------------------------------
# get_lang resolution
# ---------------------------------------------------------------------------

def test_get_lang_exact():
    from phoonnx.phonemizers.o2ipa import Orthography2IPAPhonemizer
    assert Orthography2IPAPhonemizer.get_lang("gl") == "gl"


def test_get_lang_bare_fallback():
    """gl-ES → 'gl' if 'gl-ES' not in codes but 'gl' is."""
    from phoonnx.phonemizers.o2ipa import Orthography2IPAPhonemizer
    import orthography2ipa
    codes = orthography2ipa.available_codes()
    if "gl-ES" not in codes and "gl" in codes:
        assert Orthography2IPAPhonemizer.get_lang("gl-ES") == "gl"
    else:
        result = Orthography2IPAPhonemizer.get_lang("gl-ES")
        assert result in codes
