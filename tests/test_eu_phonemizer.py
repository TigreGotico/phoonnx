"""Tests for the AhoTTS (ahotts-g2p) Basque phonemizer.

ahotts-g2p is a real dependency (declared in the ``eu`` and ``test`` extras),
so it is imported unconditionally — no importorskip. The phonemizer output is
asserted to be fully tokenizable by the StyleTTS2-eu 178-symbol phoneme map.
The engine variant is chosen by ``phonemizer_model`` (passed to the factory as
``model``): ``classic`` (VITS voices), ``modern`` (StyleTTS2-eu, default) or
``northern`` (Iparralde dialect).
"""
import pytest

from phoonnx.config import PhonemeType, get_phonemizer


# --- StyleTTS2-eu symbol set (single-char-collapsed IPA) -------------------
# Built inline from the StyleTTS2 symbol set, matching the phoneme_id_map of
# the OpenVoiceOS/phoonnx-styletts2/hitz-eu-styletts2 voice. The phonemizer
# output must be tokenizable by this map (every output char is a key here).
_pad = "$"
_punctuation = ';:,.!?¡¿—…"<>“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧ"
    "ʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩ᵻ"
)

# the full set of tokenizable symbols (the StyleTTS2-eu phoneme_id_map keys)
SYMBOLS = set(_pad + _punctuation + _letters + _letters_ipa)


SENTENCES = [
    "Kaixo, mundua.",
    "Euskara hizkuntza zaharra eta ederra da.",
    "Bilbo Euskal Herriko hiri handiena da.",
    "Nik liburu bat irakurri dut atzo arratsaldean.",
    "Etxean txakur bat eta katu bi ditugu.",
]


def test_ahotts_phonemizer_output_is_tokenizable():
    phonemizer = get_phonemizer(PhonemeType.AHOTTS)
    for sentence in SENTENCES:
        out = phonemizer.phonemize_string(sentence, "eu")
        # non-empty string per sentence
        assert isinstance(out, str)
        assert out.strip(), f"empty phonemization for {sentence!r}"
        # every char of the output must be a key in the StyleTTS2-eu map
        unknown = sorted({ch for ch in out if ch not in SYMBOLS})
        assert not unknown, (
            f"phonemes not in StyleTTS2-eu map for {sentence!r}: {unknown}"
        )


def test_ahotts_phonemize_chunks_are_tokenizable():
    """The full phonemize() pipeline (chunking + per-char lists) is tokenizable."""
    phonemizer = get_phonemizer(PhonemeType.AHOTTS)
    for sentence in SENTENCES:
        chunks = phonemizer.phonemize(sentence, "eu")
        assert chunks and any(chunks), f"no phonemes for {sentence!r}"
        for sentence_phones in chunks:
            for ch in sentence_phones:
                assert ch in SYMBOLS, f"untokenizable phoneme {ch!r} in {sentence!r}"


# --- engine selection (phonemizer_model) ----------------------------------

def test_default_engine_is_modern():
    # no phonemizer_model -> the StyleTTS2-eu "modern" engine
    assert get_phonemizer(PhonemeType.AHOTTS).engine == "modern"


@pytest.mark.parametrize("engine", ["classic", "modern", "northern"])
def test_each_engine_is_tokenizable(engine):
    phonemizer = get_phonemizer(PhonemeType.AHOTTS, model=engine)
    assert phonemizer.engine == engine
    for sentence in SENTENCES:
        out = phonemizer.phonemize_string(sentence, "eu")
        assert out.strip()
        unknown = sorted({ch for ch in out if ch not in SYMBOLS})
        assert not unknown, f"{engine}: untokenizable {unknown} in {sentence!r}"


def test_engines_differ():
    s = "Euskara hizkuntza zaharra da."
    classic = get_phonemizer(PhonemeType.AHOTTS, model="classic").phonemize_string(s, "eu")
    modern = get_phonemizer(PhonemeType.AHOTTS, model="modern").phonemize_string(s, "eu")
    northern = get_phonemizer(PhonemeType.AHOTTS, model="northern").phonemize_string(s, "eu")
    assert classic != modern        # modern adds punctuation tokens + dict stress
    assert northern != classic      # Northern pronounces /h/, uvular ʁ, remapped sibilants


def test_unknown_engine_raises():
    with pytest.raises(ValueError):
        get_phonemizer(PhonemeType.AHOTTS, model="v2")
