"""Tests for phoonnx.alphabet_convert — convert, ALPHABET_CONVERTERS.

Covers:
- Registered pair actually converts (HANGUL decomposition, HIRA/CANGJIE best-effort).
- src == dst → identity (no conversion).
- Unregistered pair → identity + DEBUG log.
- Registry is keyed by (Alphabet, Alphabet) pairs.
- synthesize() wiring: HANGUL-alphabet voice receives Jamo before encode_text.
- SynthesisConfig.alphabet as explicit src overrides the model's alphabet for convert.
"""
import logging
import unicodedata

import pytest

from phoonnx.alphabet_convert import ALPHABET_CONVERTERS, convert
from phoonnx.config import Alphabet

# Precomposed Hangul syllable '가' (U+AC00)
GA = unicodedata.normalize("NFC", "가")


# ---------------------------------------------------------------------------
# Registered pairs
# ---------------------------------------------------------------------------

def test_convert_hangul_decomposes_to_jamo():
    """(UNICODE, HANGUL) decomposes precomposed syllables into conjoining Jamo."""
    result = convert(GA, Alphabet.UNICODE, Alphabet.HANGUL)
    assert result == unicodedata.normalize("NFD", GA)
    # every character should be a Jamo codepoint
    assert all("ᄀ" <= c <= "ᇿ" for c in result)


def test_convert_hira_registered():
    """(UNICODE, HIRA) converter is registered and callable on ASCII (passthrough)."""
    result = convert("hello", Alphabet.UNICODE, Alphabet.HIRA)
    assert isinstance(result, str)


def test_convert_cangjie_registered():
    """(UNICODE, CANGJIE) converter is registered and callable on ASCII (passthrough)."""
    result = convert("hello", Alphabet.UNICODE, Alphabet.CANGJIE)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Identity: src == dst
# ---------------------------------------------------------------------------

def test_convert_src_eq_dst_unicode_identity():
    text = "hello world"
    assert convert(text, Alphabet.UNICODE, Alphabet.UNICODE) == text


def test_convert_src_eq_dst_ipa_identity():
    text = "hɛloʊ"
    assert convert(text, Alphabet.IPA, Alphabet.IPA) == text


def test_convert_src_eq_dst_arpa_identity():
    text = "HH AH0 L OW1"
    assert convert(text, Alphabet.ARPA, Alphabet.ARPA) == text


# ---------------------------------------------------------------------------
# Unregistered pair → identity + debug log
# ---------------------------------------------------------------------------

def test_convert_unregistered_pair_is_identity(caplog):
    text = "some text"
    with caplog.at_level(logging.DEBUG, logger="phoonnx.alphabet_convert"):
        result = convert(text, Alphabet.ARPA, Alphabet.HANGUL)
    assert result == text


def test_convert_unregistered_pair_logs_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger="phoonnx.alphabet_convert"):
        convert("x", Alphabet.IPA, Alphabet.CANGJIE)
    assert any("no converter registered" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------

def test_registry_keyed_by_alphabet_pairs():
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
# Hangul decomposition correctness
# ---------------------------------------------------------------------------

def test_hangul_multi_syllable_all_jamo():
    """Multi-syllable word: all output codepoints are Jamo."""
    text = "안녕"  # two syllables
    result = convert(text, Alphabet.UNICODE, Alphabet.HANGUL)
    assert all("ᄀ" <= c <= "ᇿ" for c in result)
    assert len(result) > len(text)  # decomposed form is longer


def test_hangul_ascii_passthrough():
    """Non-Hangul characters are passed through by the Jamo converter."""
    text = "hello"
    result = convert(text, Alphabet.UNICODE, Alphabet.HANGUL)
    assert result.strip() == text


# ---------------------------------------------------------------------------
# synthesize() wiring — uses minimal stubs, no ONNX model loaded
# ---------------------------------------------------------------------------

def _make_hangul_voice():
    """Return a minimal TTSVoice stub with alphabet=HANGUL and captured phonemize input."""
    from unittest.mock import MagicMock
    import numpy as np

    from phoonnx.config import VoiceConfig, PhonemeType, Alphabet, Engine
    from phoonnx.tokenizer import TTSTokenizer, Vocabulary
    from phoonnx.voice import TTSVoice

    vocab = Vocabulary(
        char2idx={c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}
    )
    tokenizer = TTSTokenizer(
        vocab,
        add_blank_char=False,
        add_blank_word=False,
        use_eos_bos=False,
        blank_at_end=False,
        blank_at_start=False,
    )

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

    captured: dict = {}

    voice = TTSVoice.__new__(TTSVoice)
    voice.config = config
    voice.phonetic_spellings = None

    mock_phonemizer = MagicMock()
    mock_phonemizer.add_diacritics.side_effect = lambda text, lang: text
    voice.phonemizer = mock_phonemizer

    # phonemize captures the text it receives, then returns a single token list
    voice.phonemize = lambda text: captured.update({"text": text}) or [["ᄀ"]]
    voice.phonemes_to_ids = lambda phonemes: [1, 2, 3]
    voice.phoneme_ids_to_audio = lambda ids, sc: np.zeros(100, dtype="float32")

    return voice, captured


def test_voice_synthesize_applies_alphabet_conversion():
    """TTSVoice.synthesize converts UNICODE Hangul → Jamo before phonemize."""
    from phoonnx.config import SynthesisConfig, Alphabet

    voice, captured = _make_hangul_voice()

    # Caller passes raw Hangul; explicit alphabet=UNICODE signals that
    syn_config = SynthesisConfig(add_diacritics=False, alphabet=Alphabet.UNICODE)
    list(voice.synthesize(GA, syn_config=syn_config))

    # phonemize must have received Jamo, not the precomposed syllable
    assert captured["text"] == unicodedata.normalize("NFD", GA)


def test_voice_synthesize_no_alphabet_field_is_noop():
    """SynthesisConfig.alphabet=None → src defaults to model alphabet → no-op."""
    from phoonnx.config import SynthesisConfig

    voice, captured = _make_hangul_voice()

    # Input is already Jamo (pre-decomposed); alphabet=None means "I'm in the model's alphabet"
    pre_jamo = unicodedata.normalize("NFD", GA)
    syn_config = SynthesisConfig(add_diacritics=False, alphabet=None)
    list(voice.synthesize(pre_jamo, syn_config=syn_config))

    # src == dst (HANGUL == HANGUL) → identity; phonemize receives unchanged text
    assert captured["text"] == pre_jamo
