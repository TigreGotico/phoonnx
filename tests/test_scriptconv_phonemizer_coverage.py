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

ahotts-g2p and (on Python <3.13) misaki[en]/misaki[zh] are installed
unconditionally through the ``[test]`` extra — nothing here is gated on an
``importorskip`` guessing whether an optional dep happens to be present.
ahotts-g2p and misaki[zh]'s own deps (jieba, pypinyin, cn2an) are
pure-Python with no compiled extensions or model-weight downloads, so
there was never a "heavy dependency" reason to make them optional.

misaki itself is a genuine exception, not a policy violation: its PyPI
metadata declares ``Requires-Python <3.13,>=3.8`` — pip/uv refuse to
install ANY misaki extra on 3.13/3.14, full stop (confirmed against
PR#348's CI logs: the 3.13/3.14 jobs died at install time trying to
build ``blis`` — misaki[en]'s spacy chain — with a Cython/NumPy-C-API
error that has no fix on our side). ``tests/pyproject.toml`` marks
``misaki[en]``/``misaki[zh]`` with a ``python_version<'3.13'`` marker
so the extra is either installed-and-tested or (below 3.13) provably
absent from the environment — the misaki-backed tests below assert that
exact ceiling with ``skipif(sys.version_info >= (3, 13))`` rather than
an ``importorskip`` that would also silently swallow a genuine
regression on the supported versions.

espeak-ng itself is a system binary (installed via
``system_deps: espeak-ng`` in the CI workflow, not a Python extra) whose
*exact* output varies by version — see the espeak-anchor note below for
how that's handled without skipping anything.
"""
import json
import sys
from pathlib import Path

import pytest

# misaki's own PyPI metadata caps Requires-Python at <3.13 (verified via
# PyPI JSON API and PR#348's CI logs — misaki[en]'s spacy->blis chain has
# no 3.13/3.14 wheels and fails to build from source). pyproject.toml's
# [test] extra mirrors this with a `python_version<'3.13'` marker on both
# misaki[en] and misaki[zh]. This is a real upstream dependency ceiling,
# not a "skip on missing dep": on <3.13 the package MUST be present and
# these tests run for real; on >=3.13 it cannot be installed at all, so
# skipping here is the only truthful outcome.
_MISAKI_UNSUPPORTED_PY = sys.version_info >= (3, 13)
_misaki_ceiling = pytest.mark.skipif(
    _MISAKI_UNSUPPORTED_PY,
    reason="misaki requires-python is <3.13 (spacy/blis chain has no "
           "3.13+ wheels); not a skip-on-missing-dep, see module docstring",
)

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


# espeak-backed anchors are NOT pinned to an exact symbol sequence: the
# text-to-phoneme mapping comes from the system espeak-ng binary, whose
# version varies across CI runners and dev machines (observed drift:
# ɔ/o and ə/ɐ substitutions between espeak-ng releases on the same input).
# Exact-sequence assertions there test the installed espeak-ng build, not
# phoonnx/scriptconv code, so they replace with version-tolerant structural
# checks: stress marks present, plausible length, no out-of-vocab symbols
# (the real regression guard — already covered by the *_coverage tests
# above). misaki and cotovia are pip-pinned pure-Python engines with no
# such external-binary drift, so their anchors stay exact (below).
_STRESS_MARK = "ˈ"


def _assert_structural_espeak_anchor(symbols, min_len, max_len):
    assert min_len <= len(symbols) <= max_len, (
        f"espeak output length {len(symbols)} outside expected "
        f"[{min_len}, {max_len}] — investigate before assuming this is "
        f"just an espeak-ng version drift"
    )
    assert _STRESS_MARK in symbols, "expected a primary-stress mark in the output"


def test_espeak_es_regression_anchor():
    """Structural (version-tolerant) anchor for scriptconv's espeak(es)
    output — see module-level note on why this isn't an exact-sequence
    assertion."""
    _assert_structural_espeak_anchor(_espeak_symbols("hola amigo", "es"), 9, 13)


def test_espeak_ca_regression_anchor():
    _assert_structural_espeak_anchor(_espeak_symbols("hola amic", "ca"), 8, 12)


def test_espeak_en_regression_anchor():
    _assert_structural_espeak_anchor(_espeak_symbols("hello friend", "en"), 10, 15)


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


@_misaki_ceiling
@pytest.mark.parametrize("text", _SENTENCES_EN_MISAKI)
def test_misaki_en_kokoro_voice_coverage(text):
    symbols = _misaki_en_symbols(text)
    assert symbols
    _assert_tokenizable(symbols, FIXTURES["kokoro_af"]["vocab"], "kokoro/af")


@_misaki_ceiling
def test_misaki_en_regression_anchor():
    assert _misaki_en_symbols("hello friend") == list("həlˈO fɹˈɛnd")


def _misaki_zh_symbols(text):
    from scriptconv.phonemizers.mul import MisakiZhPhonemizer
    p = MisakiZhPhonemizer()
    chunks = p.phonemize(text, lang="zh")
    return [sym for chunk in chunks for sym in chunk]


@_misaki_ceiling
def test_misaki_zh_kokoro_voice_coverage():
    """misaki[zh] (jieba, pypinyin, cn2an — all pure-python, no compiled
    or weight downloads) is installed via the [test] extra on Python
    <3.13, so this runs for real there rather than being skipped."""
    symbols = _misaki_zh_symbols("你好朋友")
    assert symbols
    _assert_tokenizable(symbols, FIXTURES["kokoro_zf_xiaobei"]["vocab"], "kokoro/zf_xiaobei")


@_misaki_ceiling
def test_misaki_zh_regression_anchor():
    """misaki is pip-pinned (pure-Python engine, no external binary), so
    unlike the espeak anchors above this can stay an exact anchor."""
    assert _misaki_zh_symbols("你好朋友") == ["n", "i", "↓", "x", "a", "u", "↓", " ",
                                              "p", "ʰ", "ə", "↗", "ŋ", "j", "o", "u", "↓"]


# ---------------------------------------------------------------------------
# Per-phonemizer coverage: ahotts (eu) — hitz voices
# ---------------------------------------------------------------------------

def _ahotts_eu_symbols(text):
    from scriptconv.phonemizers.eu import AhoTTSPhonemizer
    p = AhoTTSPhonemizer()
    chunks = p.phonemize(text, lang="eu")
    return [sym for chunk in chunks for sym in chunk]


def test_ahotts_eu_hitz_voice_coverage():
    """ahotts-g2p is a tiny zero-dependency pure-Python package, installed
    via the [test] extra — runs for real in CI, not skipped."""
    symbols = _ahotts_eu_symbols("kaixo laguna")
    assert symbols
    _assert_tokenizable(symbols, FIXTURES["hitz_eu_antton"]["vocab"], "hitz/eu-antton")


def test_ahotts_eu_regression_anchor():
    """ahotts-g2p is pip-pinned pure-Python (no external binary), so this
    stays an exact anchor."""
    assert _ahotts_eu_symbols("kaixo laguna") == list("kajʃO laɣUna")
