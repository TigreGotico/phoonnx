"""
Integration hardening for the scriptconv phonemizer stack behind phoonnx's
shipped StyleTTS2-family voices.

phoonnx owns no phonemizer code of its own — every phonemizer used by a
shipped voice (espeak, cotovia/gl, misaki, ahotts/eu, ...) is delegated to
scriptconv (see phonemizers-live-in-scriptconv). scriptconv is under active
refactor upstream and has broken phoonnx CI before (the mantoq->halabi
notation rename, fixed in phoonnx#334). This module exists so the *next*
such break shows up as one clear, named assertion failure here instead of
20 unrelated AttributeErrors scattered across voice.py/config.py call sites.

Two kinds of tests:
  1. An enum-surface test: every scriptconv Notation/Phonemizer enum member
     phoonnx's PhonemeType/Alphabet re-exports must still exist in scriptconv.
  2. Per-phonemizer integration tests, one block per phonemizer used by a
     shipped styletts2-family voice: espeak (es/ca/en), cotovia (gl, default
     tra), misaki (en, zh). Each block asserts:
       (a) output is non-empty and stable-typed (list[list[str]] of symbols)
       (b) every emitted symbol is a key in the *actual* shipped voice's
           phoneme_id_map (vendored fixture in
           tests/fixtures_scriptconv_vocabs.json, extracted from the real
           config.json hosted at OpenVoiceOS/phoonnx-styletts2 on HF — see
           that file's docstring-equivalent comment below for provenance)
       (c) a couple of exact phonemizations are pinned as regression
           anchors — these encode *current* scriptconv behavior, not a
           spec; if scriptconv changes its output on purpose, update the
           anchor and the PR that does it should say so.

Optional-dependency phonemizers (ahotts-g2p, misaki[zh]) are skipped via
``pytest.importorskip`` rather than mocked, per policy: no weights are
downloaded, but a real optional pip package is either present or the test
is skipped outright (never faked to look green).
"""
import json
from pathlib import Path

import pytest

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures_scriptconv_vocabs.json").read_text()
)


def _assert_tokenizable(symbols, vocab, voice_name):
    unknown = [s for s in symbols if s not in vocab and s != " "]
    assert not unknown, (
        f"{voice_name}: scriptconv emitted symbol(s) not in the shipped "
        f"voice's phoneme_id_map: {unknown!r}"
    )


# ---------------------------------------------------------------------------
# 1. Enum-surface test
# ---------------------------------------------------------------------------

# Every scriptconv Phonemizer member that phoonnx.config.PhonemeType relies
# on existing (directly referenced by name somewhere in phoonnx, or backing
# a shipped styletts2-family voice's "phoneme_type" field).
_REQUIRED_PHONEMIZER_MEMBERS = [
    "UNICODE", "GRAPHEMES", "ESPEAK", "GRUUT", "GORUUT",
    "MISAKI", "MISAKI_EN", "MISAKI_JA", "MISAKI_ZH", "MISAKI_KO", "MISAKI_VI",
    "MANTOQ", "COTOVIA", "AHOTTS",
]

_REQUIRED_ALPHABET_MEMBERS = ["IPA"]


def test_required_phonemizer_enum_members_exist():
    """A halabi-class rename (member removed/renamed) fails here by name,
    not as an AttributeError deep inside voice.py."""
    from scriptconv.phonemizers.enums import Phonemizer
    missing = [m for m in _REQUIRED_PHONEMIZER_MEMBERS if not hasattr(Phonemizer, m)]
    assert not missing, f"scriptconv.phonemizers.enums.Phonemizer missing: {missing}"


def test_required_alphabet_enum_members_exist():
    from scriptconv.phonemizers.enums import Alphabet
    missing = [m for m in _REQUIRED_ALPHABET_MEMBERS if not hasattr(Alphabet, m)]
    assert not missing, f"scriptconv.phonemizers.enums.Alphabet missing: {missing}"


def test_phoonnx_phonemetype_is_scriptconv_registry_complete():
    """Every phoonnx.config.PhonemeType member must resolve in scriptconv's
    phonemizer registry (get_phonemizer_class must not raise)."""
    from phoonnx.config import PhonemeType
    from scriptconv.phonemizers.registry import get_phonemizer_class
    failures = {}
    for member in PhonemeType:
        try:
            get_phonemizer_class(member)
        except Exception as e:  # noqa: BLE001 - collecting all failures is the point
            failures[member.name] = repr(e)
    assert not failures, f"PhonemeType members unresolvable in scriptconv registry: {failures}"


# ---------------------------------------------------------------------------
# 2. Per-phonemizer coverage: espeak (es, ca, en)
# ---------------------------------------------------------------------------

_SENTENCES_ES = ["hola amigo", "el gato duerme"]
_SENTENCES_CA = ["hola amic", "el gat dorm"]
_SENTENCES_EN = ["hello friend", "the cat sleeps"]


def _espeak_symbols(text, lang):
    from scriptconv.phonemizers.mul import EspeakPhonemizer
    p = EspeakPhonemizer()
    chunks = p.phonemize(text, lang=lang)
    assert isinstance(chunks, list) and chunks and isinstance(chunks[0], list)
    return [sym for chunk in chunks for sym in chunk]


@pytest.mark.parametrize("text", _SENTENCES_ES)
def test_espeak_es_bsc_voice_coverage(text):
    symbols = _espeak_symbols(text, "es")
    assert symbols
    _assert_tokenizable(symbols, FIXTURES["bsc_es_styletts2"]["vocab"], "bsc/es-styletts2")


@pytest.mark.parametrize("text", _SENTENCES_CA)
def test_espeak_ca_bsc_voice_coverage(text):
    symbols = _espeak_symbols(text, "ca")
    assert symbols
    _assert_tokenizable(symbols, FIXTURES["bsc_ca_styletts2"]["vocab"], "bsc/ca-styletts2")


@pytest.mark.parametrize("text", _SENTENCES_EN)
def test_espeak_en_ddatt_voice_coverage(text):
    symbols = _espeak_symbols(text, "en")
    assert symbols
    _assert_tokenizable(symbols, FIXTURES["ddatt_en_styletts2"]["vocab"], "ddatt/en-styletts2")


def test_espeak_es_regression_anchor():
    """Pins scriptconv's current espeak(es) output. Encodes current
    scriptconv behavior, NOT a spec — if this changes on purpose, update
    the anchor and say so in the PR."""
    assert _espeak_symbols("hola amigo", "es") == list("ˈola amˈiɣo")


def test_espeak_ca_regression_anchor():
    assert _espeak_symbols("hola amic", "ca") == list("ˈɔlə əmˈik")


def test_espeak_en_regression_anchor():
    assert _espeak_symbols("hello friend", "en") == list("həlˈoʊ fɹˈɛnd")


# ---------------------------------------------------------------------------
# Per-phonemizer coverage: cotovia (gl, default tra)
# ---------------------------------------------------------------------------

_SENTENCES_GL = ["Ola amigo", "O gato dorme"]


def _cotovia_output(text):
    from scriptconv.phonemizers.gl import CotoviaPhonemizer
    p = CotoviaPhonemizer()
    return p.phonemize_string(text, lang="gl")


@pytest.mark.parametrize("text", _SENTENCES_GL)
def test_cotovia_gl_proxectonos_voice_smoke(text):
    """proxectonos/*-cotovia voices go through scriptconv's CotoviaPhonemizer,
    which wraps pycotovia at a fixed ``tra=2`` (phonemes + stress marks).
    pycotovia's 0.4.0a1 release renumbered its ``tra`` levels (a new -t3
    prosodic mode at tra=4, raw debug output moved 4->5) but left tra<=3
    semantics — including tra=2 — unchanged, so this call site needs no
    code change; this test is the regression guard for that claim."""
    out = _cotovia_output(text)
    assert isinstance(out, str)
    assert len(out.strip()) > 0


def test_cotovia_gl_regression_anchor():
    """Pins scriptconv's current cotovia(gl) output at pycotovia>=0.4.0a1."""
    assert _cotovia_output("Ola amigo") == "ola amiGo "


def test_cotovia_call_site_uses_stress_level_not_raw():
    """Guards against scriptconv's gl.py accidentally passing what used to
    be raw-debug-level 4 as if it were still raw after the pycotovia
    0.4.0a1 renumbering (old raw=4 -> new raw=5, new tra=4 is the -t3
    prosodic stage). scriptconv currently hardcodes tra=2 (stress marks),
    which is unaffected either way, but pins the exact call so an upstream
    edit that starts asking for a higher tra shows up here."""
    import inspect
    from scriptconv.phonemizers import gl as gl_mod
    src = inspect.getsource(gl_mod)
    assert "tra=2" in src, (
        "scriptconv.phonemizers.gl no longer calls pycotovia with tra=2 — "
        "re-verify against the pycotovia>=0.4.0a1 tra table (1=phonemes, "
        "2=+stress, 3=+syllable-sep, 4=+prosody/-t3, 5=raw) before bumping "
        "the floor further."
    )


# ---------------------------------------------------------------------------
# Per-phonemizer coverage: misaki (en, zh)
# ---------------------------------------------------------------------------

_SENTENCES_EN_MISAKI = ["hello friend", "the cat sleeps"]


def _misaki_en_symbols(text):
    from scriptconv.phonemizers.mul import MisakiEnPhonemizer
    p = MisakiEnPhonemizer()
    chunks = p.phonemize(text, lang="en")
    return [sym for chunk in chunks for sym in chunk]


@pytest.mark.parametrize("text", _SENTENCES_EN_MISAKI)
def test_misaki_en_kokoro_voice_coverage(text):
    symbols = _misaki_en_symbols(text)
    assert symbols
    _assert_tokenizable(symbols, FIXTURES["kokoro_af"]["vocab"], "kokoro/af")


def test_misaki_en_regression_anchor():
    assert _misaki_en_symbols("hello friend") == list("həlˈO fɹˈɛnd")


def test_misaki_zh_kokoro_voice_coverage():
    """misaki's Chinese g2p needs the optional misaki[zh] extra (jieba,
    pypinyin, ...); skip rather than mock if it isn't installed — no
    weights are needed, just an optional pure-python dependency."""
    pytest.importorskip("misaki.zh", reason="misaki[zh] extra not installed")
    from scriptconv.phonemizers.mul import MisakiZhPhonemizer
    p = MisakiZhPhonemizer()
    chunks = p.phonemize("你好朋友", lang="zh")
    symbols = [sym for chunk in chunks for sym in chunk]
    assert symbols
    _assert_tokenizable(symbols, FIXTURES["kokoro_zf_xiaobei"]["vocab"], "kokoro/zf_xiaobei")


# ---------------------------------------------------------------------------
# Per-phonemizer coverage: ahotts (eu) — hitz voices
# ---------------------------------------------------------------------------

def test_ahotts_eu_hitz_voice_coverage():
    """ahotts-g2p is an optional extra (scriptconv[eu]); skip if absent
    rather than mock — no weights involved, just an unvendored pip package."""
    pytest.importorskip("ahotts", reason="ahotts-g2p extra not installed")
    from scriptconv.phonemizers.eu import AhoTTSPhonemizer
    p = AhoTTSPhonemizer()
    chunks = p.phonemize("kaixo laguna", lang="eu")
    symbols = [sym for chunk in chunks for sym in chunk]
    assert symbols
    _assert_tokenizable(symbols, FIXTURES["hitz_eu_antton"]["vocab"], "hitz/eu-antton")
