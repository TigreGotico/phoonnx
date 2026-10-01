import json
from pathlib import Path

from phoonnx.model_manager import TTSModelManager

FIXTURE = Path(__file__).parent / "fixtures" / "piper_voice_sha256.json"


def _load_piper_indices():
    """The piper and piper_community voice-index files, keyed by voice_id."""
    entries = {}
    for path in TTSModelManager.voice_index_files():
        if path.name not in ("piper.json", "piper_community.json"):
            continue
        entries.update(json.loads(path.read_text()))
    return entries


def test_no_duplicate_model_urls_in_piper_indices():
    """piper.json/piper_community.json must not carry two voice_ids pointing
    at the same model_url. Other voice-index files legitimately reuse one
    shared model file across multiple speaker/language entries, so this
    guard is scoped to the piper family, where every voice_id owns its own
    file.
    """
    seen = {}
    dupes = []
    for voice_id, entry in _load_piper_indices().items():
        url = entry.get("model_url")
        if url in seen:
            dupes.append((url, seen[url], voice_id))
        else:
            seen[url] = voice_id
    assert not dupes, f"duplicate model_url entries found: {dupes}"


def test_no_duplicate_voice_artifacts_in_piper_indices():
    """piper.json/piper_community.json must not carry two entries for the
    same underlying model artifact (same sha256), tracked via a fixture of
    known content hashes for the OpenVoiceOS/phoonnx-vits voice files.
    """
    sha_by_voice_id = json.loads(FIXTURE.read_text())
    entries = _load_piper_indices()

    sha_to_voice_ids = {}
    for voice_id in entries:
        sha = sha_by_voice_id.get(voice_id)
        if sha is None:
            continue
        sha_to_voice_ids.setdefault(sha, []).append(voice_id)

    dupes = {sha: ids for sha, ids in sha_to_voice_ids.items() if len(ids) > 1}
    assert not dupes, f"duplicate voice artifacts still indexed: {dupes}"
