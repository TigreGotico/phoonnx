"""The architecture matrix of docs/engines.md is what scripts/engine_matrix.py writes."""
import json
import sys
from pathlib import Path

from phoonnx.engines import list_engines

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import engine_matrix  # noqa: E402


def test_engine_matrix_is_current():
    assert engine_matrix.DOCS.read_text(encoding="utf-8") == engine_matrix.render(), (
        "docs/engines.md differs from the engine registries: run python scripts/engine_matrix.py")


def test_every_inference_engine_has_a_voice_or_a_task():
    counts = engine_matrix.voice_counts()
    missing = {name for name in list_engines() if counts[name] == 0} - set(engine_matrix.VOICE_TASKS)
    assert missing == set()


def test_no_task_is_kept_for_a_closed_gap():
    counts = engine_matrix.voice_counts()
    assert {name for name in engine_matrix.VOICE_TASKS if counts[name] > 0} == set()
    assert set(engine_matrix.VOICE_TASKS) <= set(list_engines())
    rows = set(list_engines()) | set(engine_matrix.training_engines())
    assert set(engine_matrix.TRAINING_TASKS) <= rows - set(engine_matrix.training_engines())


def test_every_conversion_folder_has_an_architecture():
    folders = {path.name for path in engine_matrix.CONVERSION.iterdir() if path.is_dir()}
    assert folders == set(engine_matrix.CONVERSION_DIRS)


def test_known_cells():
    header, _, *body = engine_matrix.render_table().splitlines()
    names = [cell.strip() for cell in header.strip("|").split("|")]
    rows = {}
    for line in body:
        cells = dict(zip(names, (cell.strip() for cell in line.strip("|").split("|"))))
        rows[cells["Architecture"].strip("`")] = cells
    # The id fixture holds only VITS voices (tests/fixtures/id_parity.json).
    assert [arch for arch, row in rows.items() if row["Id fixture"] == "yes"] == ["vits"]
    assert int(rows["vits"]["Index voices"]) > 1000
    assert "`export_piper`" in rows["vits"]["Export"]
    assert rows["yourtts"]["Training"] == "`yourtts`"
    assert rows["shami"]["Training"] == "R4.8"
    assert rows["magpie"]["Export"] == "`scripts/conversion/magpie`"
    assert rows["pockettts"]["Export"] == "none"
    assert rows["orpheus"]["Export"] == "none"
    assert rows["zipvoice"]["Export"] == "`export_onnx`"
    assert "`scripts/conversion/inflect`" in rows["vits"]["Export"]
    assert "conversion" not in rows["vits_prior_split"]["Export"]
    entries = set()
    for path in engine_matrix.INDEX.glob("*.json"):
        entries |= set(json.loads(path.read_text(encoding="utf-8")))
    assert sum(int(row["Index voices"]) for row in rows.values()) == len(entries)
