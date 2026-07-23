"""Diacritization is delegated to ``scriptconv.diacritics``.

``BasePhonemizer.add_diacritics`` no longer carries phoonnx's own Arabic
libtashkeel port nor calls phonikud directly for the decision — it routes
every language through scriptconv, which owns the four diacritization
backends (Hebrew phonikud, Arabic text2tashkeel, stressonnx stress
languages, European-Portuguese sense diacritics).
"""
import pytest

from phoonnx.phonemizers.base import GraphemePhonemizer


def test_unknown_language_passthrough():
    # Languages scriptconv does not recognize are returned verbatim.
    p = GraphemePhonemizer()
    assert p.add_diacritics("this is a test", "en") == "this is a test"


def test_stress_language_routes_to_scriptconv():
    # Russian is a stressonnx stress language phoonnx never handled before.
    # The call must reach scriptconv's stress backend; with the optional
    # stressonnx package absent that backend raises ImportError naming its
    # extra — proving the delegation reaches it rather than silently passing
    # the text through.
    try:
        import stressonnx  # noqa: F401
    except ImportError:
        p = GraphemePhonemizer()
        with pytest.raises(ImportError) as exc:
            p.add_diacritics("замок", "ru")
        assert "stressonnx" in str(exc.value)
    else:  # pragma: no cover - only when the optional backend is installed
        p = GraphemePhonemizer()
        # Backend present: stress marks are added, text changes.
        assert p.add_diacritics("замок", "ru") != "замок"


def test_hebrew_wiring_resolves_model_path(monkeypatch):
    # Hebrew keeps the phonikud_onnx backend, but phoonnx (not scriptconv)
    # fetches the model: add_diacritics must resolve the local model path via
    # PhonikudDiacritizer.download and hand it to scriptconv as phonikud_model.
    import phoonnx.phonemizers.base as base
    import scriptconv.diacritics as sd

    monkeypatch.setattr(
        base.PhonikudDiacritizer, "download",
        classmethod(lambda cls: "/tmp/fake-phonikud.onnx"),
    )
    seen = {}

    def fake_diacritize(text, lang, phonikud_model=None, **kw):
        seen["lang"] = lang
        seen["phonikud_model"] = phonikud_model
        return "מנוקד"  # niqqud-bearing marker

    monkeypatch.setattr(sd, "diacritize", fake_diacritize)

    p = GraphemePhonemizer()
    out = p.add_diacritics("מנוקד", "he")
    assert seen["phonikud_model"] == "/tmp/fake-phonikud.onnx"
    assert seen["lang"] == "he"
    assert out
