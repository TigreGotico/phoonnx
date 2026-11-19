"""The token ids in tests/fixtures/id_parity.json are what phoonnx computes.

phoonnx.js reads the same fixture and must produce the same ids. Each voice
loads from the config files the fixture stores, with no model and no download.
Regenerate the fixture with scripts/id_parity_fixture.py.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import id_parity_fixture  # noqa: E402
from phoonnx.model_manager import TTSModelManager  # noqa: E402

FIXTURE = json.loads(id_parity_fixture.FIXTURE.read_text(encoding="utf-8"))
MANAGER = TTSModelManager()
MANAGER.merge_default_voices()


@pytest.mark.parametrize("row", FIXTURE["rows"], ids=lambda row: f"{row['voice_id']}|{row['text']}")
def test_ids_match_fixture(row):
    voice = FIXTURE["voices"][row["voice_id"]]
    entry = MANAGER.voices[row["voice_id"]].to_dict()
    assert id_parity_fixture.compute_ids(entry, voice["files"], row["text"]) == row["ids"]


@pytest.mark.parametrize("voice_id", [v for v, voice in FIXTURE["voices"].items()
                                      if voice["phoneme_type"] == "espeak"])
def test_espeak_voice_matches_fixture(voice_id):
    voice = FIXTURE["voices"][voice_id]
    entry = MANAGER.voices[voice_id].to_dict()
    assert id_parity_fixture.espeak_voice(entry, voice["files"]) == voice["espeak_voice"]


def test_fixture_covers_three_phoneme_types():
    types = {voice["phoneme_type"] for voice in FIXTURE["voices"].values()}
    assert {"unicode", "espeak", "graphemes"} <= types
