"""Tests for phoonnx.alphabet_convert — convert, convert_to_alphabet, ALPHABET_CONVERTERS."""
import unicodedata

import pytest

from phoonnx.alphabet_convert import ALPHABET_CONVERTERS, convert, convert_to_alphabet
from phoonnx.config import Alphabet

GA = unicodedata.normalize("NFC", "가")  # precomposed Hangul syllable


# ---------------------------------------------------------------------------
# Generic convert() — registered pair
# ---------------------------------------------------------------------------

def test_convert_hangul_decomposes_to_jamo():
    result = convert(GA, Alphabet.UNICODE, Alphabet.HANGUL)
    assert result == unicodedata.normalize("NFD", GA)
    assert all("ᄀ" <= c <= "ᇿ" for c in result)


def test_convert_hira_registered():
    # Just confirm it runs without error on ASCII (hiragana converter is a passthrough for non-CJK)
    result = convert("hello", Alphabet.UNICODE, Alphabet.HIRA)
    assert isinstance(result, str)


def test_convert_cangjie_registered():
    result = convert("hello", Alphabet.UNICODE, Alphabet.CANGJIE)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Generic convert() — identity and unregistered pairs
# ---------------------------------------------------------------------------

def test_convert_src_eq_dst_is_identity():
    text = "hello world"
    assert convert(text, Alphabet.UNICODE, Alphabet.UNICODE) == text


def test_convert_ipa_to_ipa_is_identity():
    text = "hɛloʊ"
    assert convert(text, Alphabet.IPA, Alphabet.IPA) == text


def test_convert_unregistered_pair_is_identity(caplog):
    import logging
    text = "some text"
    with caplog.at_level(logging.DEBUG, logger="phoonnx.alphabet_convert"):
        result = convert(text, Alphabet.ARPA, Alphabet.HANGUL)
    assert result == text


def test_convert_unregistered_pair_logs_debug(caplog):
    import logging
    with caplog.at_level(logging.DEBUG, logger="phoonnx.alphabet_convert"):
        convert("x", Alphabet.IPA, Alphabet.CANGJIE)
    assert any("no converter registered" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------

def test_registry_keyed_by_pairs():
    for key in ALPHABET_CONVERTERS:
        assert isinstance(key, tuple) and len(key) == 2
        src, dst = key
        assert isinstance(src, Alphabet)
        assert isinstance(dst, Alphabet)


def test_unicode_hangul_in_registry():
    assert (Alphabet.UNICODE, Alphabet.HANGUL) in ALPHABET_CONVERTERS


def test_unicode_hira_in_registry():
    assert (Alphabet.UNICODE, Alphabet.HIRA) in ALPHABET_CONVERTERS


def test_unicode_cangjie_in_registry():
    assert (Alphabet.UNICODE, Alphabet.CANGJIE) in ALPHABET_CONVERTERS


# ---------------------------------------------------------------------------
# Back-compat convert_to_alphabet shim
# ---------------------------------------------------------------------------

def test_shim_hangul_decomposes_to_jamo():
    result = convert_to_alphabet(GA, Alphabet.HANGUL)
    assert result == unicodedata.normalize("NFD", GA)
    assert all("ᄀ" <= c <= "ᇿ" for c in result)


def test_shim_unicode_passthrough():
    text = "hello world"
    assert convert_to_alphabet(text, Alphabet.UNICODE) == text


def test_shim_ipa_passthrough():
    text = "hɛloʊ"
    assert convert_to_alphabet(text, Alphabet.IPA) == text


def test_shim_arpa_passthrough():
    text = "HH AH0 L OW1"
    assert convert_to_alphabet(text, Alphabet.ARPA) == text


def test_shim_cangjie_is_mapped():
    result = convert_to_alphabet("hello", Alphabet.CANGJIE)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Voice preprocessing integration test
# (uses a minimal stub to avoid loading a real ONNX model)
# ---------------------------------------------------------------------------

def test_voice_synthesize_applies_alphabet_conversion(monkeypatch):
    """synthesize() preprocessing converts text when alphabet=HANGUL."""
    from unittest.mock import MagicMock
    from phoonnx.config import VoiceConfig, SynthesisConfig, PhonemeType, Alphabet, Engine
    from phoonnx.tokenizer import TTSTokenizer, Vocabulary
    from phoonnx.voice import TTSVoice

    vocab = Vocabulary(char2idx={c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")})
    tokenizer = TTSTokenizer(vocab, add_blank_char=False, add_blank_word=False,
                              use_eos_bos=False, blank_at_end=False, blank_at_start=False)

    config = VoiceConfig(
        num_symbols=30,
        num_speakers=1,
        num_langs=1,
        sample_rate=22050,
        lang_code="ko",
        phoneme_type=PhonemeType.GRAPHEMES,
        alphabet=Alphabet.HANGUL,
        tokenizer=tokenizer,
        engine=Engine.PIPER,
        phonemizer_model=None,
    )

    captured = {}

    mock_adapter = MagicMock()
    mock_adapter.encode_text.side_effect = lambda text, voice, sc: captured.update({"text": text}) or [[1, 2, 3]]
    mock_adapter.synthesize.return_value = MagicMock(audio=__import__("numpy").zeros(100, dtype="float32"))
    mock_adapter.default_params.return_value = {}

    mock_session = MagicMock()

    voice = TTSVoice.__new__(TTSVoice)
    voice.config = config
    voice.phonetic_spellings = None
    voice.phonemizer = MagicMock()
    voice.adapter = mock_adapter
    voice.session = mock_session

    syn_config = SynthesisConfig(add_diacritics=False, add_stress=False)
    list(voice.synthesize(GA, syn_config=syn_config))

    assert captured["text"] == unicodedata.normalize("NFD", GA)
