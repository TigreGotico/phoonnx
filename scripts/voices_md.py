"""Write VOICES.md from the bundled voice index.

Run ``python scripts/voices_md.py`` after a change to ``phoonnx/voice_index``.
``tests/test_voices_md.py`` fails while VOICES.md differs from what this
script writes.
"""
import json
from pathlib import Path

from phoonnx.model_manager import TTSModelManager

VOICES_MD = Path(__file__).resolve().parent.parent / "VOICES.md"


def render() -> str:
    """The VOICES.md text for the bundled index. A voice id that two index
    files publish takes the entry of the later file, as the catalog does."""
    entries = {}
    for path in TTSModelManager.voice_index_files():
        entries.update(json.loads(path.read_text(encoding="utf-8")))
    rows = sorted((entry["lang"], voice_id, entry.get("engine") or "",
                   entry.get("phoneme_type") or "")
                  for voice_id, entry in entries.items())
    lines = [
        "## Supported Voices",
        "",
        f"**Total Voices:** {len(rows)}",
        f"**Total Languages:** {len({lang for lang, *_ in rows})}",
        "",
        "> ⚠️ some languages are duplicated, either using a different script or less specific "
        "language code (eg. Kurdish is available in latin, cyrillic and arabic scripts)",
        "",
        "| Voice ID | Language Code | Engine | Phoneme Type |",
        "| :--- | :--- | :--- | :--- |",
    ]
    lines += ["| " + " | ".join(f"`{cell}`" if cell else "" for cell in
                                (voice_id, lang, engine, phoneme_type)) + " |"
              for lang, voice_id, engine, phoneme_type in rows]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    VOICES_MD.write_text(render(), encoding="utf-8")
