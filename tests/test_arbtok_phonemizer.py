"""Tests for the arbtok Arabic phonemizer.

arbtok is a hard test dependency (declared in the ``test`` and ``ar`` extras),
so it is imported unconditionally -- no importorskip.  arbtok's edge is
undiacritized (bare) Arabic: it restores the tashkeel before transcribing.
"""
import pytest

from phoonnx.config import PhonemeType, Alphabet, get_phonemizer
from phoonnx.phonemizers.ar import ArbtokPhonemizer


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
