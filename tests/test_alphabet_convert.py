"""Tests for phoonnx.alphabet_convert — convert_to_alphabet and ALPHABET_CONVERTERS."""
import unicodedata

from phoonnx.alphabet_convert import ALPHABET_CONVERTERS, convert_to_alphabet
from phoonnx.config import Alphabet

GA = unicodedata.normalize("NFC", "가")  # precomposed Hangul syllable


# ---------------------------------------------------------------------------
# convert_to_alphabet unit tests
# ---------------------------------------------------------------------------

def test_hangul_decomposes_to_jamo():
    result = convert_to_alphabet(GA, Alphabet.HANGUL)
    assert result == unicodedata.normalize("NFD", GA)
    assert all("ᄀ" <= c <= "ᇿ" for c in result)


def test_unicode_passthrough():
    text = "hello world"
    assert convert_to_alphabet(text, Alphabet.UNICODE) == text


def test_ipa_passthrough():
    text = "hɛloʊ"
    assert convert_to_alphabet(text, Alphabet.IPA) == text


def test_arpa_passthrough():
    text = "HH AH0 L OW1"
    assert convert_to_alphabet(text, Alphabet.ARPA) == text


def test_cangjie_is_mapped():
    # CANGJIE converter is registered; calling it should not crash for ASCII
    # (ASCII passes through the Lo-category guard unchanged)
    result = convert_to_alphabet("hello", Alphabet.CANGJIE)
    assert isinstance(result, str)


def test_cangjie_in_converters_dict():
    assert Alphabet.CANGJIE in ALPHABET_CONVERTERS


def test_hangul_in_converters_dict():
    assert Alphabet.HANGUL in ALPHABET_CONVERTERS


def test_hira_in_converters_dict():
    assert Alphabet.HIRA in ALPHABET_CONVERTERS


# ---------------------------------------------------------------------------
# Voice preprocessing integration test
# (uses a minimal stub to avoid loading a real ONNX model)
# ---------------------------------------------------------------------------

def test_voice_synthesize_applies_alphabet_conversion(monkeypatch):
    """The synthesize() preprocessing block converts text when alphabet=HANGUL."""
    from unittest.mock import MagicMock
    from phoonnx.config import VoiceConfig, SynthesisConfig, PhonemeType, Alphabet, Engine
    from phoonnx.tokenizer import TTSTokenizer, Vocabulary
    from phoonnx.voice import TTSVoice

    # Minimal vocabulary and tokenizer
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

    # Stub out the heavy parts
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

    # The text passed to encode_text should be jamo, not the original syllable
    assert captured["text"] == unicodedata.normalize("NFD", GA)
