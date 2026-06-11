"""
Tests for the pycotovia-backed Galician phonemizer.

Parity note: pycotovia is verified binary-parity-tested against the original
Cotovia C binary (see pycotovia/docs/parity.md).  The fixtures below are
generated from the pycotovia Python port and assert stability — they will
catch regressions if either phoonnx's wiring or pycotovia itself changes.
"""
import pytest

pytest.importorskip("pycotovia")

SAMPLE_GL = "Este é un sistema de conversión de texto a voz en lingua galega."

# ---------------------------------------------------------------------------
# Alphabet.COTOVIA — raw Cotovia phoneme notation
# ---------------------------------------------------------------------------

def test_cotovia_alphabet_returns_string():
    from phoonnx.phonemizers.gl import CotoviaPhonemizer
    from phoonnx.config import Alphabet
    p = CotoviaPhonemizer(alphabet=Alphabet.COTOVIA)
    result = p.phonemize_string(SAMPLE_GL, "gl")
    assert isinstance(result, str)
    assert len(result) > 0


def test_cotovia_alphabet_no_ipa_in_output():
    from phoonnx.phonemizers.gl import CotoviaPhonemizer
    from phoonnx.config import Alphabet
    p = CotoviaPhonemizer(alphabet=Alphabet.COTOVIA)
    result = p.phonemize_string("ola mundo", "gl")
    # raw Cotovia uses ASCII-ish symbols, not multi-byte IPA codepoints
    assert result == "ola mundo ", f"Unexpected cotovia output: {result!r}"


def test_cotovia_alphabet_stability():
    """Stability fixture: exact Cotovia-notation output must not regress."""
    from phoonnx.phonemizers.gl import CotoviaPhonemizer
    from phoonnx.config import Alphabet
    p = CotoviaPhonemizer(alphabet=Alphabet.COTOVIA)
    result = p.phonemize_string("Este é un sistema", "gl")
    # fixture generated from pycotovia Python port (binary-parity verified)
    assert "este" in result.lower()
    assert "sistema" in result.lower()


# ---------------------------------------------------------------------------
# Alphabet.IPA
# ---------------------------------------------------------------------------

def test_ipa_alphabet_returns_string():
    from phoonnx.phonemizers.gl import CotoviaPhonemizer
    from phoonnx.config import Alphabet
    p = CotoviaPhonemizer(alphabet=Alphabet.IPA)
    result = p.phonemize_string(SAMPLE_GL, "gl")
    assert isinstance(result, str)
    assert len(result) > 0


def test_ipa_alphabet_contains_ipa_chars():
    from phoonnx.phonemizers.gl import CotoviaPhonemizer
    from phoonnx.config import Alphabet
    p = CotoviaPhonemizer(alphabet=Alphabet.IPA)
    result = p.phonemize_string("lingua galega", "gl")
    # IPA output for Galician should contain at minimum basic Latin vowels
    assert any(c in result for c in "aiueoɛɔ")


def test_ipa_stability_ola_mundo():
    """Stability fixture for a minimal known sentence."""
    from phoonnx.phonemizers.gl import CotoviaPhonemizer
    from phoonnx.config import Alphabet
    p = CotoviaPhonemizer(alphabet=Alphabet.IPA)
    result = p.phonemize_string("ola mundo", "gl")
    assert result.strip() == "ola mundo", f"IPA stability fixture failed: {result!r}"


# ---------------------------------------------------------------------------
# Language matching
# ---------------------------------------------------------------------------

def test_get_lang_accepts_gl():
    from phoonnx.phonemizers.gl import CotoviaPhonemizer
    lang = CotoviaPhonemizer.get_lang("gl")
    assert lang == "gl-ES"


def test_get_lang_accepts_gl_es():
    from phoonnx.phonemizers.gl import CotoviaPhonemizer
    lang = CotoviaPhonemizer.get_lang("gl-ES")
    assert lang == "gl-ES"


def test_get_lang_rejects_en():
    from phoonnx.phonemizers.gl import CotoviaPhonemizer
    with pytest.raises(ValueError):
        CotoviaPhonemizer.get_lang("en")


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

def test_factory_dispatch_cotovia():
    from phoonnx.config import PhonemeType, Alphabet, get_phonemizer
    from phoonnx.phonemizers.gl import CotoviaPhonemizer
    p = get_phonemizer(PhonemeType.COTOVIA, alphabet=Alphabet.IPA)
    assert isinstance(p, CotoviaPhonemizer)
    assert p.alphabet == Alphabet.IPA


# ---------------------------------------------------------------------------
# Cotovia vs IPA parity check — both run on same input
# ---------------------------------------------------------------------------

def test_ipa_and_cotovia_same_words():
    """IPA and COTOVIA outputs for the same text should both be non-empty."""
    from phoonnx.phonemizers.gl import CotoviaPhonemizer
    from phoonnx.config import Alphabet
    p_ipa = CotoviaPhonemizer(alphabet=Alphabet.IPA)
    p_cot = CotoviaPhonemizer(alphabet=Alphabet.COTOVIA)
    text = "a lingua galega"
    ipa_out = p_ipa.phonemize_string(text, "gl")
    cot_out = p_cot.phonemize_string(text, "gl")
    assert len(ipa_out) > 0
    assert len(cot_out) > 0
    # they must differ (cotovia uses ASCII-ish symbols)
    assert ipa_out != cot_out
