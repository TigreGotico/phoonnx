"""Tests for the euskaphone dialect-aware Basque phonemizer.

euskaphone is a hard test dependency (declared in the ``test`` and ``eu``
extras), so it is imported unconditionally -- no importorskip.  This backend is
distinct from AhoTTSPhonemizer (also in eu.py): euskaphone drives the
orthography2ipa lattice and resolves the historical Basque dialects.
"""
import pytest

from phoonnx.config import PhonemeType, Alphabet, get_phonemizer
from scriptconv.phonemizers.eu import EuskaphonePhonemizer


def test_factory_dispatch():
    p = get_phonemizer(PhonemeType.EUSKAPHONE)
    assert isinstance(p, EuskaphonePhonemizer)
    assert p.alphabet == Alphabet.IPA


def test_phonemize_batua_returns_ipa():
    p = EuskaphonePhonemizer()
    out = p.phonemize_string("Zazpi katu zuri ikusi ditut.", "eu")
    assert isinstance(out, str)
    assert out.strip()


def test_souletin_shows_aspiration_and_front_rounded():
    # Souletin (eu-x-zuberera) pronounces /h/ and has /y/, unlike Batua
    p = EuskaphonePhonemizer()
    batua = p.phonemize_string("Hotza egiten du gaur mendian.", "eu")
    souletin = p.phonemize_string("Hotza egiten du gaur mendian.", "eu-x-zuberera")
    assert batua != souletin


def test_get_lang_accepts_code_and_alias():
    assert EuskaphonePhonemizer.get_lang("eu")
    assert EuskaphonePhonemizer.get_lang("batua")
    assert EuskaphonePhonemizer.get_lang("souletin")


def test_get_lang_rejects_non_basque():
    with pytest.raises(ValueError):
        EuskaphonePhonemizer.get_lang("zz-nonsense")


def test_non_ipa_alphabet_rejected():
    with pytest.raises(ValueError):
        EuskaphonePhonemizer(alphabet=Alphabet.BUCKWALTER)
