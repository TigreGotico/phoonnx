"""
Integration tests for the scriptconv delegation layer.

Covers:
- ARPABET → IPA via scriptconv.notation (arpa_to_ipa / arpa_to_ipa_lookup)
- Buckwalter ↔ Arabic script via scriptconv.notation (used inside the Mantoq pipeline)
- normalize_lang MMS script-tag cases (crk-script_syllabics, cmo-khmer-script)
"""
import pytest


# ---------------------------------------------------------------------------
# ARPABET → IPA
# ---------------------------------------------------------------------------

def test_arpa_to_ipa_lookup_importable():
    """arpa_to_ipa_lookup must remain importable from its historic location."""
    from phoonnx.thirdparty.arpa2ipa import arpa_to_ipa_lookup
    assert isinstance(arpa_to_ipa_lookup, dict)
    assert len(arpa_to_ipa_lookup) > 50


def test_arpa_to_ipa_lookup_basic_phonemes():
    from phoonnx.thirdparty.arpa2ipa import arpa_to_ipa_lookup
    assert arpa_to_ipa_lookup["B"] == "b"
    assert arpa_to_ipa_lookup["IY1"] == "i"
    assert arpa_to_ipa_lookup["AH0"] == "ə"


def test_arpa_to_ipa_function():
    """arpa_to_ipa should produce a non-empty IPA string for a sample word."""
    from phoonnx.thirdparty.arpa2ipa import arpa_to_ipa
    result = arpa_to_ipa("B IY1 T")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "b" in result
    assert "i" in result


def test_arpa_to_ipa_stress_digits_stripped():
    """Stress-digit variants (AH0, AH1) should both map to IPA without KeyError."""
    from phoonnx.thirdparty.arpa2ipa import arpa_to_ipa_lookup
    assert "AH0" in arpa_to_ipa_lookup
    assert "AH1" in arpa_to_ipa_lookup
    assert arpa_to_ipa_lookup["AH0"] == "ə"
    assert arpa_to_ipa_lookup["AH1"] == "ʌ"


def test_en_phonemizer_uses_arpa_lookup():
    """
    G2PEnPhonemizer imports arpa_to_ipa_lookup directly — ensure the import
    still resolves after the scriptconv delegation.
    """
    from phoonnx.phonemizers.en import G2PEnPhonemizer  # noqa: F401 — import-only check


# ---------------------------------------------------------------------------
# Buckwalter ↔ Arabic script (via Mantoq pipeline)
# ---------------------------------------------------------------------------

def test_arabic_to_buckwalter_roundtrip():
    """arabic_to_buckwalter → buckwalter_to_arabic should recover the original."""
    from phoonnx.thirdparty.mantoq.buck.phonetise_buckwalter import (
        arabic_to_buckwalter,
        buckwalter_to_arabic,
    )
    samples = ["مرحبا", "الشمس", "بيت", "كتاب"]
    for word in samples:
        bw = arabic_to_buckwalter(word)
        recovered = buckwalter_to_arabic(bw)
        assert recovered == word, f"Roundtrip failed for {word!r}: got {recovered!r}"


def test_arabic_to_buckwalter_known_values():
    from phoonnx.thirdparty.mantoq.buck.phonetise_buckwalter import arabic_to_buckwalter
    assert arabic_to_buckwalter("مرحبا") == "mrHbA"
    assert arabic_to_buckwalter("الشمس") == "Al$ms"


def test_mantoq_phonemizer_arabic_ipa_smoke():
    """MantoqPhonemizer with IPA alphabet should produce a non-empty string."""
    from phoonnx.phonemizers.ar import MantoqPhonemizer
    from phoonnx.config import Alphabet
    p = MantoqPhonemizer(alphabet=Alphabet.IPA)
    result = p.phonemize_string("مرحبا", lang="ar")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# normalize_lang MMS script-tag cases
# ---------------------------------------------------------------------------

def test_normalize_lang_mms_syllabics():
    """crk-script_syllabics should resolve to a BCP-47 tag with Cans script."""
    from phoonnx.util import normalize_lang
    result = normalize_lang("crk-script_syllabics")
    assert "Cans" in result, f"Expected 'Cans' in {result!r}"


def test_normalize_lang_mms_khmer():
    """cmo-khmer-script should resolve to a BCP-47 tag with Khmr script."""
    from phoonnx.util import normalize_lang
    result = normalize_lang("cmo-khmer-script")
    assert "Khmr" in result, f"Expected 'Khmr' in {result!r}"


def test_normalize_lang_plain_lang_unchanged():
    """A plain language code should still pass through normalize_lang."""
    from phoonnx.util import normalize_lang
    result = normalize_lang("en")
    assert result.startswith("en")


def test_normalize_lang_mms_arabic_script():
    """MMS 'arabic' script word should yield Arab subtag."""
    from phoonnx.util import normalize_lang
    result = normalize_lang("arb-script_arabic")
    assert "Arab" in result, f"Expected 'Arab' in {result!r}"
