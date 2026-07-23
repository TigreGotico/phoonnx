"""Tests for the orthography2ipa-backed multilingual phonemizer.

orthography2ipa is a hard test dependency (declared in the ``test`` and ``o2i``
extras), so it is imported unconditionally -- no importorskip.
"""
import pytest

from phoonnx.config import PhonemeType, Alphabet, get_phonemizer
from scriptconv.phonemizers.o2ipa import Orthography2IPAPhonemizer


def test_factory_dispatch():
    p = get_phonemizer(PhonemeType.ORTHOGRAPHY2IPA)
    assert isinstance(p, Orthography2IPAPhonemizer)
    assert p.alphabet == Alphabet.IPA


@pytest.mark.parametrize("lang,text", [
    ("ar", "ذهب الطالب إلى المكتبة"),
    ("eu", "Kaixo, mundua."),
    ("gl", "Boas, o mundo."),
    ("pt", "Olá, quem são vocês?"),
    ("mwl", "Buonos dies, mundo."),
])
def test_phonemize_returns_ipa(lang, text):
    p = Orthography2IPAPhonemizer()
    out = p.phonemize_string(text, lang)
    assert isinstance(out, str)
    assert out.strip(), f"empty phonemization for {lang!r}"


def test_get_lang_resolves_bcp47():
    # a regional tag resolves to a supported spec code
    assert Orthography2IPAPhonemizer.get_lang("pt-PT")
    assert Orthography2IPAPhonemizer.get_lang("eu")


def test_get_lang_rejects_gibberish():
    with pytest.raises(ValueError):
        Orthography2IPAPhonemizer.get_lang("zz-nonsense")


def test_supported_langs_is_a_large_family():
    langs = Orthography2IPAPhonemizer.supported_langs()
    assert isinstance(langs, list)
    # one backend, many languages
    assert len(langs) > 100


def test_instance_caches_engines():
    p = Orthography2IPAPhonemizer()
    p.phonemize_string("Kaixo", "eu")
    p.phonemize_string("mundua", "eu")
    resolved = Orthography2IPAPhonemizer.get_lang("eu")
    assert resolved in p._cache
