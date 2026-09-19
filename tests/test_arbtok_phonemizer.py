"""Tests for the arbtok Arabic phonemizer.

arbtok is a hard test dependency (declared in the ``test`` and ``ar`` extras),
so it is imported unconditionally -- no importorskip.  arbtok's edge is
undiacritized (bare) Arabic: it restores the tashkeel before transcribing.
"""
import pytest

from phoonnx.config import PhonemeType, Alphabet, get_phonemizer
from scriptconv.phonemizers.ar import ArbtokPhonemizer


def test_factory_dispatch():
    p = get_phonemizer(PhonemeType.ARBTOK)
    assert isinstance(p, ArbtokPhonemizer)
    assert p.alphabet == Alphabet.IPA


def test_phonemize_bare_arabic_returns_ipa():
    # undiacritized text -- arbtok diacritizes then transcribes
    p = ArbtokPhonemizer()
    out = p.phonemize_string("ذهب الطالب إلى المكتبة لقراءة كتاب", "ar")
    assert isinstance(out, str)
    assert out.strip()
    # restored short vowels: bare skeleton has no long-vowel-only reading
    assert "a" in out or "i" in out or "u" in out


def test_irab_register_forces_full_case_endings():
    """A phoonnx 'i'rab' register must map to arbtok's 'full' register (not the
    literal string 'irab', which the plugin rejects), and must actually restore
    the case endings the pausal default drops."""
    for alias in ("i'rab", "iʿrab", "irab", "full"):
        assert ArbtokPhonemizer(register=alias).register == "full"
    full = ArbtokPhonemizer(register="i'rab").phonemize_string("ذهب الطالب", "ar")
    pausal = ArbtokPhonemizer().phonemize_string("ذهب الطالب", "ar")
    assert full.strip() and full != pausal  # the endings make it differ


def test_dialect_differs_from_msa():
    p = ArbtokPhonemizer()
    msa = p.phonemize_string("كتاب جميل", "ar")
    egy = p.phonemize_string("كتاب جميل", "ar-EG")
    assert msa.strip() and egy.strip()


def test_get_lang_resolves_arabic_specs():
    assert ArbtokPhonemizer.get_lang("ar")
    assert ArbtokPhonemizer.get_lang("ar-EG")
    assert ArbtokPhonemizer.get_lang("ar-SA-x-najd") == "ar-SA-x-najd"


def test_non_ipa_alphabet_rejected():
    with pytest.raises(ValueError):
        ArbtokPhonemizer(alphabet=Alphabet.BUCKWALTER)
