"""Tests for the g2p_barranquenho Barranquenho phonemizer.

g2p_barranquenho is a hard test dependency (declared in the ``test`` and ``pt``
extras), so it is imported unconditionally -- no importorskip.
"""
import pytest

from phoonnx.config import PhonemeType, Alphabet, get_phonemizer
from scriptconv.phonemizers.pt import BarranquenhoPhonemizer

LANG = "ext-PT-x-barrancos"


def test_factory_dispatch():
    p = get_phonemizer(PhonemeType.BARRANQUENHO)
    assert isinstance(p, BarranquenhoPhonemizer)
    assert p.alphabet == Alphabet.IPA


def test_phonemize_sentence_returns_ipa():
    p = BarranquenhoPhonemizer()
    out = p.phonemize_string("Boca de la casa.", LANG)
    assert isinstance(out, str)
    assert out.strip()
    # word-level engine joined per word, one space between words
    assert len(out.split()) == 4


def test_nasal_vowel_emitted():
    p = BarranquenhoPhonemizer()
    out = p.phonemize_string("cantá manhán", LANG)
    # coda nasalization surfaces as a nasal vowel (combining tilde U+0303)
    assert "̃" in out


def test_get_lang_accepts_barrancos():
    assert BarranquenhoPhonemizer.get_lang(LANG) == LANG


def test_get_lang_rejects_unrelated():
    with pytest.raises(ValueError):
        BarranquenhoPhonemizer.get_lang("en")


def test_non_ipa_alphabet_rejected():
    with pytest.raises(ValueError):
        BarranquenhoPhonemizer(alphabet=Alphabet.BUCKWALTER)
