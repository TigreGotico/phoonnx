"""Each docs page is true: the ``verify`` command of its agent hint exits 0."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["README.md", *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs").rglob("*.md"))]


def verify_command(page: str) -> str:
    match = re.search(r"^verify: (.+)$", (ROOT / page).read_text(encoding="utf-8"), re.M)
    assert match, f"{page} has no verify line in its agent hint"
    return match.group(1).replace(".venv/bin/python", sys.executable)


@pytest.mark.parametrize("page", PAGES)
def test_the_verify_command_of_a_docs_page_exits_0(page):
    result = subprocess.run(verify_command(page), shell=True, capture_output=True, text=True,
                            cwd=ROOT, timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


def test_the_download_table_of_choose_voice_is_the_result_file():
    sizes = json.loads((ROOT / "docs/data/download_sizes.json").read_text(encoding="utf-8"))
    rows = re.findall(r"^\| `[\w-]+` \| `([^`]+)` \| (\d+) MB \|$",
                      (ROOT / "docs/choose_voice.md").read_text(encoding="utf-8"), re.M)
    assert rows and {voice for voice, _ in rows} == set(sizes)
    assert {voice: int(mb) for voice, mb in rows} == {voice: round(size / 1e6) for voice, size in sizes.items()}


def test_the_orpheus_speed_is_the_result_file():
    rows = json.loads((ROOT / "scripts/conversion/orpheus/evidence/wer.json").read_text(encoding="utf-8"))
    ms = [row["ms_per_token"] for row in rows]
    rtf = [row["rtf"] for row in rows]
    claim = (f"the backbone needs {min(ms) / 1000:.2f} to {max(ms) / 1000:.2f} s per token. The voice\n"
             f"thus runs {round(min(rtf))} to {round(max(rtf))} times slower than real time, "
             f"{sum(rtf) / len(rtf):.1f} times on average over seven sentences")
    assert len(rows) == 7
    assert claim in (ROOT / "docs/orpheus.md").read_text(encoding="utf-8")
