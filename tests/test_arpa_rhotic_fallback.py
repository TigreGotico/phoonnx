"""ARPABET text on an IPA voice: a voice whose map lacks an r-coloured vowel
speaks a spelling its own map holds.

Each test expects ids read straight from the voice's map, in a spelling written
out by hand. The ARPABET table entries are set per test, so the result does not
depend on which scriptconv release the test environment holds.
"""
import types
from unittest.mock import patch

import pytest
from scriptconv.notation import _ARPA_TO_IPA
from scriptconv.phonemizers.base import BasePhonemizer

from phoonnx.config import Alphabet, PhonemeType, SynthesisConfig
from phoonnx.tokenizer import TTSTokenizer, Vocabulary
from phoonnx.voice import TTSVoice

CONSONANTS = "hbt"
# Rows of the English voice index, by what the map holds beyond CONSONANTS.
ESPEAK_NO_RHOTIC = "ɜəɛɹrɚ"     # piper espeak voices: no ɝ
GLOWTTS_NO_NURSE = "əɛɹɚ"       # gruut GlowTTS voices: no ɝ, ɜ or r
MISAKI_NO_RHOTICS = "ɜəɛɹr"     # 3 of the 43 misaki voices: no ɝ or ɚ
FULL_RHOTICS = "ɜəɛɹrɝɚ"


def _voice(symbols, phonemizer=None, phoneme_type=None):
    # The ids are spaced and start high, so an id cannot equal its position.
    phoneme_map = {s: 100 + 7 * i for i, s in enumerate(CONSONANTS + symbols)}
    tokenizer = TTSTokenizer(Vocabulary(char2idx=phoneme_map),
                             add_blank_char=False, add_blank_word=False,
                             use_eos_bos=False, blank_at_end=False, blank_at_start=False)
    voice = TTSVoice.__new__(TTSVoice)
    voice.phonemizer = phonemizer or _FixedPhonemizer("")
    voice.config = types.SimpleNamespace(lang_code="en-US", phonemizer_lang=None, alphabet=Alphabet.IPA, text_casing="ignore",
                                         phoneme_type=phoneme_type, add_diacritics=False,
                                         diacritizer_model=None, tokenizer=tokenizer)
    voice.adapter = types.SimpleNamespace(takes_symbols=True, READS_TEXT=False)
    voice._frontend = voice._frontend_key = voice._frontend_phonemizer = None
    return voice, phoneme_map


def _ids(voice, text, src=Alphabet.ARPA):
    syn_config = SynthesisConfig(input_alphabet=None if src == Alphabet.GRAPHEMES else src)
    return [list(utterance.ids) for utterance in voice.prepare(text, syn_config)]


def _spelled(phoneme_map, spelling):
    return [[phoneme_map[c] for c in spelling]]


@pytest.fixture
def rhotic_table(monkeypatch):
    """ARPABET table with ER as ɝ, ER0 as ɚ and AXR as ɚ, the cited values."""
    for key in ("ER", "ER1", "ER2"):
        monkeypatch.setitem(_ARPA_TO_IPA, key, "ɝ")
    monkeypatch.setitem(_ARPA_TO_IPA, "ER0", "ɚ")
    monkeypatch.setitem(_ARPA_TO_IPA, "AXR", "ɚ")


@pytest.mark.parametrize("arpa, symbols, spelling", [
    ("HH ER1 T", ESPEAK_NO_RHOTIC, "hɜɹt"),
    ("HH ER2 T", ESPEAK_NO_RHOTIC, "hɜɹt"),
    ("HH ER T", MISAKI_NO_RHOTICS, "hɜɹt"),
    ("HH ER1 T", GLOWTTS_NO_NURSE, "hɚt"),
    ("HH ER1 T", "əɹ", "həɹt"),
    ("B ER0", MISAKI_NO_RHOTICS, "bəɹ"),
    ("B AXR", MISAKI_NO_RHOTICS, "bəɹ"),
    ("HH ER1 T B ER0", MISAKI_NO_RHOTICS, "hɜɹtbəɹ"),
])
def test_missing_rhotic_speaks_a_spelling_the_map_holds(rhotic_table, arpa, symbols, spelling):
    voice, phoneme_map = _voice(symbols)
    assert _ids(voice, arpa) == _spelled(phoneme_map, spelling)
    assert voice.tokenizer.not_found_characters == set()


@pytest.mark.parametrize("phoneme_type, spelling", [
    (PhonemeType.ESPEAK, "hɜːt"),      # espeak-ng writes the stressed vowel as ɜː
    (PhonemeType.MISAKI_EN, "hɜɹt"),
    (PhonemeType.MISAKI, "hɜɹt"),
    (PhonemeType.GRUUT, "hɜɹt"),
    (PhonemeType.GORUUT, "hɜɹt"),
    (None, "hɜɹt"),
])
def test_nurse_spelling_follows_the_training_phonemizer(rhotic_table, phoneme_type, spelling):
    voice, phoneme_map = _voice("ɜːəɛɹɚ", phoneme_type=phoneme_type)
    assert _ids(voice, "HH ER1 T") == _spelled(phoneme_map, spelling)


def test_espeak_voice_without_the_long_mark_uses_the_shared_order(rhotic_table):
    voice, phoneme_map = _voice(ESPEAK_NO_RHOTIC, phoneme_type=PhonemeType.ESPEAK)
    assert _ids(voice, "HH ER1 T") == _spelled(phoneme_map, "hɜɹt")


def test_espeak_voice_respells_the_unstressed_vowel_in_the_shared_order(rhotic_table):
    voice, phoneme_map = _voice("ɜːəɛɹ", phoneme_type=PhonemeType.ESPEAK)
    assert _ids(voice, "B ER0") == _spelled(phoneme_map, "bəɹ")


@pytest.mark.parametrize("arpa, spelling", [
    ("HH ER1 T", "hɝt"),
    ("B ER0", "bɚ"),
    ("B AXR", "bɚ"),
])
def test_voice_with_the_symbol_keeps_it(rhotic_table, arpa, spelling):
    voice, phoneme_map = _voice(FULL_RHOTICS)
    assert _ids(voice, arpa) == _spelled(phoneme_map, spelling)


def test_voice_with_one_rhotic_respells_only_the_other(rhotic_table):
    voice, phoneme_map = _voice("ɝəɹ")
    assert _ids(voice, "HH ER1 T B ER0") == _spelled(phoneme_map, "hɝtbəɹ")


@pytest.mark.parametrize("arpa, symbols, missing, spelling", [
    ("HH ER1 T", "ɜɛ", "ɝ", "ht"),      # ɜ with no r sound
    ("HH ER1 T", "ɹɛ", "ɝ", "ht"),      # an r sound with no vowel of the set
    ("B ER0", "ɜɹ", "ɚ", "b"),          # ɜ does not stand in for ə
])
def test_no_fallback_keeps_the_warning(rhotic_table, arpa, symbols, missing, spelling):
    voice, phoneme_map = _voice(symbols)
    with patch("phoonnx.tokenizer.LOG.warning") as warning:
        ids = _ids(voice, arpa)
    assert ids == _spelled(phoneme_map, spelling)
    assert voice.tokenizer.not_found_characters == {missing}
    assert warning.call_count == 1 and repr(missing) in warning.call_args.args[0]


class _FixedPhonemizer(BasePhonemizer):
    """Gives the same IPA for any text, in the shape of a real backend."""

    def __init__(self, ipa):
        super().__init__(Alphabet.IPA)
        self.ipa = ipa

    def phonemize_string(self, text, lang):
        return self.ipa


def test_default_path_never_respells(rhotic_table):
    """Text with no SynthesisConfig.alphabet gets what the phonemizer gave: the
    tokenizer still drops ɝ and warns."""
    voice, phoneme_map = _voice(ESPEAK_NO_RHOTIC, phonemizer=_FixedPhonemizer("hɝt"))
    assert _ids(voice, "hurt", src=Alphabet.GRAPHEMES) == _spelled(phoneme_map, "ht")
    assert voice.tokenizer.not_found_characters == {"ɝ"}


def test_ipa_input_is_never_respelled():
    """IPA text names its symbols itself, so only ARPABET text gets a fallback."""
    voice, phoneme_map = _voice(ESPEAK_NO_RHOTIC)
    assert _ids(voice, "hɝt", src=Alphabet.IPA) == _spelled(phoneme_map, "ht")
    assert voice.tokenizer.not_found_characters == {"ɝ"}


def test_tokenizer_without_a_phoneme_map_is_left_alone(rhotic_table):
    """A BPE tokenizer has no vocabulary attribute, and it reads ɝ as text."""
    voice, _ = _voice(ESPEAK_NO_RHOTIC)
    seen = []
    voice.config.tokenizer = types.SimpleNamespace(tokenize=lambda p: seen.append(p) or [1])
    assert _ids(voice, "HH ER1 T") == [[1]]
    assert seen == [list("hɝt")]
