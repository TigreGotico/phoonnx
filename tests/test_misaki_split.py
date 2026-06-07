"""The per-language misaki phonemizers (split from the legacy dispatcher)."""
from phoonnx.config import get_phonemizer, PhonemeType, Alphabet
from phoonnx.phonemizers.mul import (MisakiPhonemizer, MisakiEnPhonemizer, MisakiJaPhonemizer,
                                     MisakiZhPhonemizer, MisakiKoPhonemizer, MisakiViPhonemizer)


def test_each_type_maps_to_its_class():
    cases = {
        PhonemeType.MISAKI_EN: MisakiEnPhonemizer,
        PhonemeType.MISAKI_JA: MisakiJaPhonemizer,
        PhonemeType.MISAKI_ZH: MisakiZhPhonemizer,
        PhonemeType.MISAKI_KO: MisakiKoPhonemizer,
        PhonemeType.MISAKI_VI: MisakiViPhonemizer,
    }
    for pt, cls in cases.items():
        assert type(get_phonemizer(pt, Alphabet.IPA, None)) is cls


def test_zh_representation_is_alphabet_driven():
    # one class; the alphabet (not a version param) selects IPA vs bopomofo
    assert MisakiZhPhonemizer(Alphabet.IPA).zh_version == "1.0"       # IPA + tone marks
    assert MisakiZhPhonemizer(Alphabet.BOPOMOFO).zh_version == "1.1"  # bopomofo + tone numbers
    assert MisakiZhPhonemizer().alphabet == Alphabet.IPA             # sensible default
    assert MisakiZhPhonemizer.MISAKI_LANGS == ["zh"]


def test_subclasses_narrow_lang_scope():
    assert MisakiJaPhonemizer.MISAKI_LANGS == ["ja"]
    assert set(MisakiEnPhonemizer.MISAKI_LANGS) == {"en-US", "en-GB"}
    # the base remains the broad back-compat dispatcher
    assert set(MisakiPhonemizer.MISAKI_LANGS) >= {"en-US", "ja", "zh", "ko", "vi"}
