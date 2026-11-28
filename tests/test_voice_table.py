"""docs/voices.md is what tools/build_voice_table.py writes from docs/data/voice_qa.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import build_voice_table  # noqa: E402


def test_voice_table_is_the_result_file():
    records = json.loads(build_voice_table.RESULT.read_text(encoding="utf-8"))
    assert build_voice_table.PAGE.read_text(encoding="utf-8") == build_voice_table.render(records)
