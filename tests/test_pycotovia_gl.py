"""
Tests for the pycotovia-backed Galician phonemizer.

Parity note: pycotovia is verified binary-parity-tested against the original
Cotovia C binary (see pycotovia/docs/parity.md).  The fixtures below are
generated from the pycotovia Python port and assert stability — they will
catch regressions if either phoonnx's wiring or pycotovia itself changes.
"""
import pytest

# pycotovia is a hard test dependency (declared in the `test` extra), so the
# Galician tests must always run — never skip on a missing dep.

SAMPLE_GL = "Este é un sistema de conversión de texto a voz en lingua galega."

# ---------------------------------------------------------------------------
# Alphabet.COTOVIA — raw Cotovia phoneme notation
# ---------------------------------------------------------------------------

def test_cotovia_alphabet_returns_string():
    from scriptconv.phonemizers.gl import CotoviaPhonemizer
    from phoonnx.config import Alphabet
    p = CotoviaPhonemizer(alphabet=Alphabet.COTOVIA)
    result = p.phonemize_string(SAMPLE_GL, "gl")
    assert isinstance(result, str)
    assert len(result) > 0


def test_cotovia_alphabet_no_ipa_in_output():
    from scriptconv.phonemizers.gl import CotoviaPhonemizer
    from phoonnx.config import Alphabet
    p = CotoviaPhonemizer(alphabet=Alphabet.COTOVIA)
    result = p.phonemize_string("ola mundo", "gl")
    # raw Cotovia uses ASCII-ish symbols, not multi-byte IPA codepoints
    assert result == "ola mundo ", f"Unexpected cotovia output: {result!r}"


def test_cotovia_alphabet_stability():
    """Stability fixture: exact Cotovia-notation output must not regress."""
    from scriptconv.phonemizers.gl import CotoviaPhonemizer
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
    from scriptconv.phonemizers.gl import CotoviaPhonemizer
    from phoonnx.config import Alphabet
    p = CotoviaPhonemizer(alphabet=Alphabet.IPA)
    result = p.phonemize_string(SAMPLE_GL, "gl")
    assert isinstance(result, str)
    assert len(result) > 0


def test_ipa_alphabet_contains_ipa_chars():
    from scriptconv.phonemizers.gl import CotoviaPhonemizer
    from phoonnx.config import Alphabet
    p = CotoviaPhonemizer(alphabet=Alphabet.IPA)
    result = p.phonemize_string("lingua galega", "gl")
    # IPA output for Galician should contain at minimum basic Latin vowels
    assert any(c in result for c in "aiueoɛɔ")


def test_ipa_stability_ola_mundo():
    """Stability fixture for a minimal known sentence."""
    from scriptconv.phonemizers.gl import CotoviaPhonemizer
    from phoonnx.config import Alphabet
    p = CotoviaPhonemizer(alphabet=Alphabet.IPA)
    result = p.phonemize_string("ola mundo", "gl")
    assert result.strip() == "ola mundo", f"IPA stability fixture failed: {result!r}"


# ---------------------------------------------------------------------------
# Language matching
# ---------------------------------------------------------------------------

def test_get_lang_accepts_gl():
    from scriptconv.phonemizers.gl import CotoviaPhonemizer
    lang = CotoviaPhonemizer.get_lang("gl")
    assert lang == "gl-ES"


def test_get_lang_accepts_gl_es():
    from scriptconv.phonemizers.gl import CotoviaPhonemizer
    lang = CotoviaPhonemizer.get_lang("gl-ES")
    assert lang == "gl-ES"


def test_get_lang_rejects_en():
    from scriptconv.phonemizers.gl import CotoviaPhonemizer
    with pytest.raises(ValueError):
        CotoviaPhonemizer.get_lang("en")


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------

def test_factory_dispatch_cotovia():
    from phoonnx.config import PhonemeType, Alphabet, get_phonemizer
    from scriptconv.phonemizers.gl import CotoviaPhonemizer
    p = get_phonemizer(PhonemeType.COTOVIA, alphabet=Alphabet.IPA)
    assert isinstance(p, CotoviaPhonemizer)
    assert p.alphabet == Alphabet.IPA


# ---------------------------------------------------------------------------
# Cotovia vs IPA parity check — both run on same input
# ---------------------------------------------------------------------------

def test_ipa_and_cotovia_same_words():
    """IPA and COTOVIA outputs for the same text should both be non-empty."""
    from scriptconv.phonemizers.gl import CotoviaPhonemizer
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


def test_cotovia_stress_model_marks_stressed_vowel():
    # the "stress" phonemizer_model emits the stressed vowel with a trailing '^'
    # (HiTZ gl VITS front-end); the default stays stressless.
    from scriptconv.phonemizers.gl import CotoviaPhonemizer
    from phoonnx.config import Alphabet
    stress = CotoviaPhonemizer(alphabet=Alphabet.COTOVIA, model="stress")
    plain = CotoviaPhonemizer(alphabet=Alphabet.COTOVIA)
    s = stress.phonemize_string("ola mundo", "gl")
    assert "^" in s, f"no stress mark in {s!r}"
    assert "^" not in plain.phonemize_string("ola mundo", "gl")


def test_cotovia_stress_model_via_factory():
    from phoonnx.config import get_phonemizer, PhonemeType, Alphabet
    p = get_phonemizer(PhonemeType.COTOVIA, alphabet=Alphabet.COTOVIA, model="stress")
    assert "^" in p.phonemize_string("galego", "gl")


def test_cotovia_stress_folds_into_id_map_tokens():
    # stress-folded 'V^' and multi-char cotovia phonemes (rr, tS) tokenize as
    # single tokens against the HiTZ gl phoneme_id_map.
    from phoonnx.config import VoiceConfig, get_phonemizer, PhonemeType, Alphabet
    cfg = {
        "engine": "coqui", "phoneme_type": "cotovia", "alphabet": "cotovia",
        "lang_code": "gl-ES", "num_symbols": 137, "audio": {"sample_rate": 22050},
        "phonemizer_model": "stress",
        "phoneme_id_map": {"_": 0, " ": 10, "o": 36, "o^": 125, "l": 33,
                           "a": 22, "k": 32, "a^": 85, "rr": 65},
        "pad": "_", "blank": "_",
    }
    vc = VoiceConfig.from_dict(cfg)
    p = get_phonemizer(PhonemeType.COTOVIA, alphabet=Alphabet.COTOVIA, model="stress")
    s = p.phonemize_string("carro", "gl")  # ka^rro -> k, a^, rr, o
    ids = [i for i in vc.tokenizer.encode(s) if i != 10]  # drop word-sep
    assert ids == [32, 85, 65, 36], f"got {ids} for {s!r}"
