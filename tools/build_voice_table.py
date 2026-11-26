#!/usr/bin/env python3
"""Write docs/voices.md from docs/data/voice_qa.json.

Run ``python tools/build_voice_table.py`` after ``tools/voice_qa.py``.
``tests/test_voice_table.py`` fails while docs/voices.md differs from ``render()``.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "docs/data/voice_qa.json"
PAGE = ROOT / "docs/voices.md"
PATHS = ("manual", "index", "plugin")


def cell(result: dict) -> str:
    label = result["class"]
    if "wer" in result:
        label += f" {result['mean']:.2f}"
    elif "error" in result:
        label += " (load error)"
    return f"[{label}]({result['issue']})" if "issue" in result else label


def row(record: dict) -> str:
    if record.get("class") == "NO-VOICE":
        cells = [record["lang"], *["NO-VOICE"] * 8]
    else:
        floor = "ASR-UNVERIFIED" if record["asr_floor"] is None else f"{record['asr_floor']:.2f}"
        cells = [record["lang"], f"`{record['voice_id']}`", record["engine"],
                 f"{record['phoneme_type']} / {record['alphabet']}",
                 f"`{record['asr_model']}`" if record["asr_model"] else "ASR-UNVERIFIED", floor,
                 *(cell(record[name]) for name in PATHS)]
    return "| " + " | ".join(cells) + " |"


def render(records: list) -> str:
    versions = sorted({r["phoonnx"] for r in records})
    lines = [
        "# Voice QA",
        "",
        "<!-- agent-hint",
        "covers: the ASR round trip of the default voice of each leader language, through the manual, index and plugin load paths",
        "source: tools/voice_qa.py, docs/data/voice_qa.json, tools/build_voice_table.py",
        "generated: yes, by tools/build_voice_table.py from docs/data/voice_qa.json",
        "verify: .venv/bin/python -I tools/build_voice_table.py --check",
        "see: voice_sweep.md, choose_voice.md",
        "-->",
        "",
        f"This table shows how well the default voice of each leader language speaks. "
        f"`tools/voice_qa.py` measured it with phoonnx {', '.join(versions)} and wrote "
        f"`docs/data/voice_qa.json`. The plugin picks each voice from the language tag alone. "
        f"Each voice speaks two fixed sentences with a digit and punctuation through three load paths. "
        "A path cell gives the class and the mean WER of the two sentences. "
        "The result file holds the WER of each sentence.",
        "",
        "The WER of each language comes from its own onnx-asr model, the one that the "
        "ovos-stt-plugin-onnx-asr registry names. The ASR floor is the WER of that model on two "
        "clips of real speech. A WER of 0.35 or less passes, and 0.5 or less for kab, hi-IN and bn-IN. "
        "When the floor is above that limit, the cell is ASR-UNVERIFIED.",
        "",
        "| Language | Voice | Engine | Phonemes / alphabet | ASR model | ASR floor | Manual | Index | Plugin |",
        "| :--- | :--- | :--- | :--- | :--- | ---: | :--- | :--- | :--- |",
        *(row(r) for r in records),
        "",
        "---",
        "",
        "[← Voice sweep](voice_sweep.md) · [Home](index.md) · [Training quickstart →](training/quickstart.md)",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    text = render(json.loads(RESULT.read_text(encoding="utf-8")))
    if "--check" in sys.argv:
        current = PAGE.read_text(encoding="utf-8") == text
        print("docs/voices.md is current" if current else "docs/voices.md differs from the result file")
        return 0 if current else 1
    PAGE.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
