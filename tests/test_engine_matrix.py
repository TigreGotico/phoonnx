"""The architecture matrix of docs/engines.md is what scripts/engine_matrix.py writes."""
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
