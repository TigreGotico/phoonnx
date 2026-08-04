"""Tests for KittenTTS voices routed through the existing StyleTTS2 adapter.

KittenTTS (KittenML/kitten-tts-{nano-0.1,nano-0.2,mini-0.1}) uses the exact
StyleTTS2Adapter single-graph contract (input_ids + style[1,256] + speed ->
waveform @ 24kHz) already implemented for StyleTTS2/Kokoro, so no new adapter
class or ``detect()`` extension is needed -- these tests pin that down and
guard the vocab/index against regressions.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from phoonnx.engines import get_adapter, detect_engine
from phoonnx.engines.styletts2 import StyleTTS2Adapter
from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.config import VoiceConfig

VOICE_INDEX = json.loads(
    (Path(__file__).parent.parent / "phoonnx" / "voice_index" / "styletts2.json").read_text()
)

# Verbatim from kittentts==0.1.3 kittentts/__init__.py (KittenTTS.__init__'s
# inline TextCleaner symbol list); duplicated here only for the test to
# recompute the map independently of scripts/conversion/styletts2/export_kittentts.py.
_KITTENTTS_SYMBOLS = (
    '$;:,.!?¡¿—…"«»"" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    'ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢ'
    'ǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘\'̩\'ᵻ'
)


def _req(n=5, **p):
    return AdapterSynthesisRequest(phoneme_ids=np.arange(1, n + 1, dtype=np.int64)[None, :],
                                    phoneme_lengths=np.array([n], np.int64), params=p)


def _kittentts_entries():
    return {vid: v for vid, v in VOICE_INDEX.items() if vid.startswith("kittentts/")}


# --- engine routing / detect: no new adapter, no detect() change needed ----

def test_kittentts_uses_existing_styletts2_engine_string():
    """engine=='styletts2' must already resolve to StyleTTS2Adapter (Kokoro's
    detect() path), so KittenTTS needs zero adapter/detect() changes."""
    assert isinstance(get_adapter("styletts2"), StyleTTS2Adapter)
    assert StyleTTS2Adapter.detect(config={"engine": "styletts2"}) is True
    assert isinstance(detect_engine(config={"engine": "styletts2"}), StyleTTS2Adapter)


# --- vocab correctness: every TextCleaner symbol is tokenizable -----------

def test_every_kittentts_symbol_is_tokenizable():
    """The config's phoneme_id_map must cover every symbol KittenTTS's own
    TextCleaner can emit -- built programmatically from the same literal, not
    hand-copied, so this is correct-by-construction rather than a fixture."""
    expected_map = {s: i for i, s in enumerate(list(_KITTENTTS_SYMBOLS))}
    entries = _kittentts_entries()
    assert entries, "no kittentts/* entries found in styletts2.json"

    # config.json is not shipped locally (it lives on the HF mirror); rebuild
    # it the same way export_kittentts.py does and check parity here so a
    # vocab regression is caught without a network fetch.
    import importlib.util
    script_path = (Path(__file__).parent.parent / "scripts" / "conversion"
                    / "styletts2" / "export_kittentts.py")
    spec = importlib.util.spec_from_file_location("export_kittentts", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    built_map = mod.build_phoneme_id_map()

    assert built_map == expected_map
    # ids stay within the embedding table range built by the same enumerate();
    # note upstream's literal has duplicate quote glyphs, so a few earlier ids
    # get silently overwritten by later duplicates (dict-comprehension quirk
    # from kittentts==0.1.3 itself) -- ids are valid but not contiguous.
    assert max(built_map.values()) < len(list(_KITTENTTS_SYMBOLS))
    assert min(built_map.values()) == 0


def test_kittentts_vocab_ids_line_up_with_embedding_table_size():
    import importlib.util
    script_path = (Path(__file__).parent.parent / "scripts" / "conversion"
                    / "styletts2" / "export_kittentts.py")
    spec = importlib.util.spec_from_file_location("export_kittentts", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for tier in mod.TIERS:
        cfg = mod.build_config(tier)
        assert cfg["num_symbols"] == len(cfg["phoneme_id_map"])
        assert cfg["engine"] == "styletts2"
        assert cfg["phoneme_type"] == "espeak"
        assert cfg["alphabet"] == "ipa"


# --- config routing: canonical phoonnx config is recognised ----------------

def test_kittentts_config_is_canonical_phoonnx_shape():
    import importlib.util
    script_path = (Path(__file__).parent.parent / "scripts" / "conversion"
                    / "styletts2" / "export_kittentts.py")
    spec = importlib.util.spec_from_file_location("export_kittentts", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg_dict = mod.build_config("nano-0.1")
    assert VoiceConfig.is_phoonnx(cfg_dict)
    cfg = VoiceConfig.from_dict(cfg_dict)
    assert cfg.engine.value == "styletts2"
    assert cfg.sample_rate == 24000


# --- voice index well-formedness -------------------------------------------

def test_kittentts_voice_index_well_formed():
    entries = _kittentts_entries()
    assert len(entries) == 24, "expected 3 tiers x 8 voices"
    tiers = {"nano-0.1", "nano-0.2", "mini-0.1"}
    seen_tiers = set()
    for vid, entry in entries.items():
        assert entry["voice_id"] == vid
        assert entry["engine"] == "styletts2"
        assert entry["phoneme_type"] == "espeak"
        assert entry["alphabet"] == "ipa"
        assert entry["lang"] == "en-US"
        assert entry["model_url"].startswith(
            "https://huggingface.co/OpenVoiceOS/phoonnx-kitten-tts/resolve/main/")
        assert entry["config_url"].startswith(
            "https://huggingface.co/OpenVoiceOS/phoonnx-kitten-tts/resolve/main/")
        assert entry["style_url"] and entry["style_url"].endswith(".bin")
        tier = vid.split("/", 1)[1].rsplit("-expr-voice", 1)[0]
        for t in tiers:
            if vid.startswith(f"kittentts/{t}-"):
                seen_tiers.add(t)
    assert seen_tiers == tiers


def test_kittentts_voice_index_ids_unique():
    entries = _kittentts_entries()
    assert len(entries) == len(set(entries))
