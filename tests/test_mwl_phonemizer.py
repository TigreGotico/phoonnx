"""Tests for the Mirandese phonemizer.

scriptconv runs it on the orthography2ipa specs, so it needs no package of its
own.  It is dialect-aware: central Mirandese (``mwl``, the default), Sendinese
(``mwl-x-sendim``) and the Raiano/Ifanês border variety (``mwl-x-ifanes``).
Every test blocks the ``mwl_phonemizer`` module, so an import of it raises
``ImportError``.
"""
import sys

import pytest

from phoonnx.config import PhonemeType, Alphabet, get_phonemizer
from scriptconv.phonemizers.mwl import MirandesePhonemizer, DIALECTS


@pytest.fixture(autouse=True)
def mwl_phonemizer_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "mwl_phonemizer", None)


def test_factory_dispatch():
    p = get_phonemizer(PhonemeType.MIRANDESE)
    assert isinstance(p, MirandesePhonemizer)
    assert p.alphabet == Alphabet.IPA


@pytest.mark.parametrize("lang", list(DIALECTS))
def test_each_dialect_returns_ipa(lang):
    p = MirandesePhonemizer()
    out = p.phonemize_string("Buonos dies mundo", lang)
    assert isinstance(out, str)
    assert out.strip(), f"empty phonemization for {lang!r}"


def test_default_lang_is_central():
    p = MirandesePhonemizer()
    out = p.phonemize_string("Buonos dies mundo")
    assert out.strip()


def test_sendinese_differs_from_central():
    p = MirandesePhonemizer()
    central = p.phonemize_string("Buonos dies mundo", "mwl")
    sendinese = p.phonemize_string("Buonos dies mundo", "mwl-x-sendim")
    # Sendinese monophthongizes the rising diphthongs (bwo -> bu, dje -> di)
    assert central != sendinese


def test_get_lang_rejects_unsupported():
    with pytest.raises(ValueError):
        MirandesePhonemizer.get_lang("en")
