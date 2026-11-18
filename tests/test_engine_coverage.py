"""Every registered inference adapter is reached by an index voice, or has a reason.

The reasons are the "Engines with no index voice" table in docs/engines.md. When an
adapter gains a voice, its row leaves the table and the allow-list below.
"""
import json
from pathlib import Path

from phoonnx.engines import list_engines

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "phoonnx" / "voice_index"
DOCS = ROOT / "docs" / "engines.md"

# Config formats whose voices the VITS adapters load through detection.
VITS_FORMATS = {"phoonnx", "piper", "mimic3", "coqui", "transformers"}

NO_INDEX_VOICE = {
    "vits_streaming": "a mode of a VITS voice, selected by streaming: true",
    "yourtts": "the Phoonnx/yourtts-multi-graphemes configs wait for an ASR round trip",
    "zipvoice": "clones per request; no converted model in the Phoonnx organisation",
    "tinyvoca": "the Phoonnx/tinyvoca-en_us mirror is not public",
}


def index_entries() -> list[dict]:
    entries = []
    for path in sorted(INDEX.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        entries.extend(data.values() if isinstance(data, dict) else data)
    return entries


def reached_adapters() -> set[str]:
    registered = set(list_engines())
    reached = set()
    for entry in index_entries():
        engine = entry.get("engine")
        if engine in registered:
            reached.add(engine)
        elif engine in VITS_FORMATS:
            prior_split = "decode_path" in (entry.get("aux_model_urls") or {})
            reached.add("vits_prior_split" if prior_split else "vits")
    return reached


def test_every_index_entry_names_its_engine():
    unnamed = [entry.get("voice_id") for entry in index_entries() if not entry.get("engine")]
    assert unnamed == []


def test_every_adapter_has_a_voice_or_a_reason():
    missing = set(list_engines()) - reached_adapters() - set(NO_INDEX_VOICE)
    assert missing == set()


def test_no_reason_is_kept_for_an_adapter_with_a_voice():
    assert set(NO_INDEX_VOICE) & reached_adapters() == set()


def test_every_reason_is_documented():
    section = DOCS.read_text(encoding="utf-8").split("#### Engines with no index voice", 1)[1]
    section = section.split("####", 1)[0]
    rows = {line.split("|")[1].strip().strip("`") for line in section.splitlines() if line.startswith("| `")}
    assert rows == set(NO_INDEX_VOICE)
