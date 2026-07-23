"""
Integration tests for the scriptconv delegation layer.

Covers:
- ARPABET → IPA via scriptconv.notation (arpa_to_ipa / arpa_to_ipa_lookup)
- Buckwalter ↔ Arabic script via scriptconv.notation (used inside the Mantoq pipeline)
- normalize_lang MMS script-tag cases (crk-script_syllabics, cmo-khmer-script)
"""
import pytest


# ---------------------------------------------------------------------------
# ARPABET → IPA
# ---------------------------------------------------------------------------

def test_arpa_to_ipa_lookup_importable():
    """arpa_to_ipa_lookup must remain importable from its historic location."""
    from scriptconv.notation import _ARPA_TO_IPA as arpa_to_ipa_lookup
    assert isinstance(arpa_to_ipa_lookup, dict)
    assert len(arpa_to_ipa_lookup) > 50


def test_arpa_to_ipa_lookup_basic_phonemes():
    from scriptconv.notation import _ARPA_TO_IPA as arpa_to_ipa_lookup
    assert arpa_to_ipa_lookup["B"] == "b"
    assert arpa_to_ipa_lookup["IY1"] == "i"
    assert arpa_to_ipa_lookup["AH0"] == "ə"


def test_arpa_to_ipa_function():
    """arpa_to_ipa should produce a non-empty IPA string for a sample word."""
    from scriptconv.notation import arpa_to_ipa
    result = arpa_to_ipa("B IY1 T")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "b" in result
    assert "i" in result


def test_arpa_to_ipa_stress_digits_stripped():
    """Stress-digit variants (AH0, AH1) should both map to IPA without KeyError."""
    from scriptconv.notation import _ARPA_TO_IPA as arpa_to_ipa_lookup
    assert "AH0" in arpa_to_ipa_lookup
    assert "AH1" in arpa_to_ipa_lookup
    assert arpa_to_ipa_lookup["AH0"] == "ə"
    assert arpa_to_ipa_lookup["AH1"] == "ʌ"


def test_en_phonemizer_uses_arpa_lookup():
    """
    G2PEnPhonemizer imports arpa_to_ipa_lookup directly — ensure the import
    still resolves after the scriptconv delegation.
    """
    from scriptconv.phonemizers.en import G2PEnPhonemizer  # noqa: F401 — import-only check


# ---------------------------------------------------------------------------
# Buckwalter ↔ Arabic script (via Mantoq pipeline)
# ---------------------------------------------------------------------------

def test_arabic_to_buckwalter_roundtrip():
    """arabic_to_buckwalter → buckwalter_to_arabic should recover the original."""
    from scriptconv.phonemizers._vendored.mantoq.buck.phonetise_buckwalter import (
        arabic_to_buckwalter,
        buckwalter_to_arabic,
    )
    samples = ["مرحبا", "الشمس", "بيت", "كتاب"]
    for word in samples:
        bw = arabic_to_buckwalter(word)
        recovered = buckwalter_to_arabic(bw)
        assert recovered == word, f"Roundtrip failed for {word!r}: got {recovered!r}"


def test_arabic_to_buckwalter_known_values():
    from scriptconv.phonemizers._vendored.mantoq.buck.phonetise_buckwalter import arabic_to_buckwalter
    assert arabic_to_buckwalter("مرحبا") == "mrHbA"
    assert arabic_to_buckwalter("الشمس") == "Al$ms"


def test_mantoq_phonemizer_arabic_ipa_smoke():
    """MantoqPhonemizer with IPA alphabet should produce a non-empty string."""
    from scriptconv.phonemizers.ar import MantoqPhonemizer
    from phoonnx.config import Alphabet
    p = MantoqPhonemizer(alphabet=Alphabet.IPA)
    result = p.phonemize_string("مرحبا", lang="ar")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# normalize_lang MMS script-tag cases
# ---------------------------------------------------------------------------

def test_normalize_lang_mms_syllabics():
    """crk-script_syllabics should resolve to a BCP-47 tag with Cans script."""
    from phoonnx.util import normalize_lang
    result = normalize_lang("crk-script_syllabics")
    assert "Cans" in result, f"Expected 'Cans' in {result!r}"


def test_normalize_lang_mms_khmer():
    """cmo-khmer-script should resolve to a BCP-47 tag with Khmr script."""
    from phoonnx.util import normalize_lang
    result = normalize_lang("cmo-khmer-script")
    assert "Khmr" in result, f"Expected 'Khmr' in {result!r}"


def test_normalize_lang_plain_lang_unchanged():
    """A plain language code should still pass through normalize_lang."""
    from phoonnx.util import normalize_lang
    result = normalize_lang("en")
    assert result.startswith("en")


def test_normalize_lang_mms_arabic_script():
    """MMS 'arabic' script word should yield Arab subtag."""
    from phoonnx.util import normalize_lang
    result = normalize_lang("arb-script_arabic")
    assert "Arab" in result, f"Expected 'Arab' in {result!r}"


def test_phoneme_type_is_scriptconv_phonemizer_enum():
    """Enum identity across the boundary: values stored in voice configs must
    resolve to the same class in phoonnx and scriptconv."""
    from phoonnx.config import Alphabet, PhonemeType
    from scriptconv.phonemizers.enums import Alphabet as ScAlphabet
    from scriptconv.phonemizers.enums import Phonemizer
    assert PhonemeType is Phonemizer
    assert Alphabet is ScAlphabet


def test_get_phonemizer_injects_normalizer():
    from phoonnx.config import PhonemeType, get_phonemizer
    from phoonnx.util import normalize
    p = get_phonemizer(PhonemeType.GRAPHEMES)
    assert p.normalizer is normalize
    # normalization behavior preserved: digits expand as before the migration
    out = "".join(p.phonemize("2 cats", "en")[0])
    assert "2" not in out


def test_licensed_backends_construct_from_scriptconv_quarantine():
    from phoonnx.config import PhonemeType, get_phonemizer
    from scriptconv.phonemizers.ar import MantoqPhonemizer
    m = get_phonemizer(PhonemeType.MANTOQ)
    assert isinstance(m, MantoqPhonemizer)
    assert type(m).__module__.startswith("scriptconv.")


# ---------------------------------------------------------------------------
# Per-language phonemizer modules re-export the shared contract types
# ---------------------------------------------------------------------------

_PHONEMIZER_MODULES = [
    "ar", "en", "eu", "fa", "gl", "he", "ja", "ko",
    "mwl", "o2ipa", "pt", "vi", "zh", "shami",
]


@pytest.mark.parametrize("module_name", _PHONEMIZER_MODULES)
def test_shim_reexports_alphabet_and_base_phonemizer(module_name):
    """Every per-language phonemizer module carries the shared contract types
    (Alphabet, BasePhonemizer) alongside its language-specific classes."""
    import importlib
    mod = importlib.import_module(f"scriptconv.phonemizers.{module_name}")
    from scriptconv.phonemizers.enums import Alphabet
    from scriptconv.phonemizers.base import BasePhonemizer
    assert mod.Alphabet is Alphabet
    assert mod.BasePhonemizer is BasePhonemizer


def test_diacritization_delegates_to_scriptconv():
    """phoonnx no longer owns any diacritizer: the diacritization path routes
    straight to ``scriptconv.diacritics.diacritize``. Arabic routes to
    scriptconv's tashkeel backend; an unknown language passes through
    unchanged. The heavy text2tashkeel backend is faked so routing is asserted
    without requiring the optional dependency in the test environment."""
    from unittest.mock import patch, MagicMock
    import scriptconv.diacritics as scd

    fake_backend = MagicMock()
    fake_backend.diacritize.return_value = "ARABIC+DIACRITICS"
    with patch.object(scd, "_tashkeel", return_value=fake_backend) as mk:
        assert scd.diacritize("ذهب محمد", "ar") == "ARABIC+DIACRITICS"
        mk.assert_called_once()  # 'ar' routed to scriptconv's tashkeel backend
    # unknown language needs no backend and must pass through unchanged
    assert scd.diacritize("hello world", "und") == "hello world"


def test_voice_diacritization_delegates_to_scriptconv():
    """phoonnx owns no diacritizer: a grapheme voice with add_diacritics=True
    routes text through scriptconv's diacritizer graph edge before phonemization
    (diacritization is a topological edge, not a phoonnx call). Asserted with a
    spy on scriptconv.diacritics.diacritize — no backend/model needed."""
    import types
    from unittest.mock import patch, MagicMock
    import numpy as np
    import scriptconv.diacritics as scd
    from phoonnx.voice import TTSVoice
    from phoonnx.config import Alphabet, SynthesisConfig

    voice = TTSVoice.__new__(TTSVoice)
    voice.phonetic_spellings = None
    voice.phonemizer = MagicMock()
    voice.phonemizer.phonemize_lazy = MagicMock(return_value=iter([list("n_ss")]))
    voice.config = types.SimpleNamespace(alphabet=Alphabet.IPA, lang_code="ar",
                                         add_diacritics=True, diacritizer_model="rawi-ensemble",
                                         sample_rate=22050)
    voice.adapter = MagicMock()
    voice.phonemes_to_ids = lambda p: [1] * len(p)
    voice.phoneme_ids_to_audio = lambda ids, syn_config=None, language_ids=None, include_alignments=False: np.zeros(4, dtype=np.float32)

    seen = {}
    def spy(text, lang="und", **k):
        seen.update(text=text, lang=lang, kwargs=k)
        return text
    with patch.object(scd, "diacritize", side_effect=spy):
        list(voice.synthesize("نص", SynthesisConfig(alphabet=Alphabet.GRAPHEMES,
                                                     normalize_audio=False)))
    assert seen.get("text") == "نص" and seen.get("lang") == "ar"
    assert seen["kwargs"].get("diacritizer_model") == "rawi-ensemble"


def test_en_module_reexports_arpa_to_ipa_lookup():
    from scriptconv.phonemizers import en
    from scriptconv.phonemizers.en import arpa_to_ipa_lookup
    assert en.arpa_to_ipa_lookup is arpa_to_ipa_lookup


def test_shami_module_reexports_frontend_helpers():
    from scriptconv.phonemizers import shami
    from scriptconv.phonemizers.shami import TextFrontend, sentence_tokenize
    from scriptconv.phonemizers.base import PhonemizedChunks
    assert shami.TextFrontend is TextFrontend
    assert shami.PhonemizedChunks is PhonemizedChunks
    assert shami.sentence_tokenize is sentence_tokenize


def _phonemic_voice(model_alphabet):
    """Minimal voice for the already-phonemic dispatch (no phonemizer needed)."""
    import types
    from phoonnx.voice import TTSVoice
    from phoonnx.config import Alphabet
    voice = TTSVoice.__new__(TTSVoice)
    voice.phonemizer = None
    voice.config = types.SimpleNamespace(lang_code="en", alphabet=model_alphabet,
                                         add_diacritics=False, diacritizer_model=None)
    voice.phonemes_to_ids = lambda p: list(range(len(p)))
    return voice


def test_phonemic_input_transcodes_to_model_alphabet_via_scriptconv():
    """Already-phonemic input in a foreign notation is transcoded to the model's
    alphabet through scriptconv's graph, and an ARPA-vocab model receives WHOLE
    symbols (whitespace-tokenised), not characters. Replaces the coverage lost
    with the deleted phoonnx-side conversion module."""
    from phoonnx.config import Alphabet, SynthesisConfig
    voice = _phonemic_voice(Alphabet.ARPA)
    out = list(voice._iter_synthesis_ids(
        "S@", SynthesisConfig(alphabet=Alphabet.XSAMPA), Alphabet.XSAMPA, Alphabet.ARPA))
    assert [p for p, _ in out] == [["SH", "AX"]]


def test_phonemic_input_same_alphabet_passes_through_unchanged():
    """src == tgt is a no-op route: the text reaches _phonemic_chunks untouched."""
    from phoonnx.config import Alphabet, SynthesisConfig
    voice = _phonemic_voice(Alphabet.ARPA)
    out = list(voice._iter_synthesis_ids(
        "SH AX", SynthesisConfig(alphabet=Alphabet.ARPA), Alphabet.ARPA, Alphabet.ARPA))
    assert [p for p, _ in out] == [["SH", "AX"]]


def test_unroutable_phonemic_pair_passes_text_through():
    """When scriptconv has no path between the two notations the input is passed
    through unchanged rather than raising (best-effort, matching the old BFS)."""
    from phoonnx.config import Alphabet, SynthesisConfig
    voice = _phonemic_voice(Alphabet.ARPA)
    out = list(voice._iter_synthesis_ids(
        "abc", SynthesisConfig(alphabet=Alphabet.HANGUL), Alphabet.HANGUL, Alphabet.ARPA))
    assert [p for p, _ in out] == [["abc"]]


def test_hebrew_diacritizer_model_carries_the_phonikud_path():
    """The single diacritizer_model knob is overloaded per language: for Hebrew it
    is the phonikud ONNX path. Assert a voice's value reaches scriptconv's
    diacritizer for he (spy — no model download)."""
    import types
    from unittest.mock import patch, MagicMock
    import numpy as np
    import scriptconv.diacritics as scd
    from phoonnx.voice import TTSVoice
    from phoonnx.config import Alphabet, SynthesisConfig

    voice = TTSVoice.__new__(TTSVoice)
    voice.phonetic_spellings = None
    voice.phonemizer = MagicMock()
    voice.phonemizer.phonemize_lazy = MagicMock(return_value=iter([list("ʃalom")]))
    voice.config = types.SimpleNamespace(alphabet=Alphabet.IPA, lang_code="he",
                                         add_diacritics=True,
                                         diacritizer_model="/models/phonikud.onnx",
                                         sample_rate=22050)
    voice.adapter = MagicMock()
    voice.phonemes_to_ids = lambda p: [1] * len(p)
    voice.phoneme_ids_to_audio = lambda ids, syn_config=None, language_ids=None, include_alignments=False: np.zeros(4, dtype=np.float32)

    seen = {}
    with patch.object(scd, "diacritize",
                      side_effect=lambda t, l="und", **k: seen.update(lang=l, **k) or t):
        list(voice.synthesize("שלום", SynthesisConfig(alphabet=Alphabet.GRAPHEMES,
                                                      normalize_audio=False)))
    assert seen.get("lang") == "he"
    assert seen.get("diacritizer_model") == "/models/phonikud.onnx"
