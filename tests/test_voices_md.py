"""VOICES.md is what scripts/voices_md.py writes for the bundled index."""
import sys
from pathlib import Path

from phoonnx.model_manager import TTSModelManager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import voices_md  # noqa: E402


def test_voices_md_is_current():
    assert voices_md.VOICES_MD.read_text(encoding="utf-8") == voices_md.render(), (
        "VOICES.md differs from the bundled index: run python scripts/voices_md.py")


def test_voices_md_lists_the_catalog(tmp_path):
    manager = TTSModelManager(cache_path=str(tmp_path / "voices.json"))
    manager.merge_default_voices()
    text = voices_md.render()
    assert f"**Total Voices:** {len(manager.all_voices)}\n" in text
    assert f"**Total Languages:** {len(manager.supported_langs)}\n" in text
    rows = {line.split("`")[1] for line in text.splitlines() if line.startswith("| `")}
    assert rows == {voice.voice_id for voice in manager.all_voices}
