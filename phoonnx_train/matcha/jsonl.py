"""Torch-free loader for the shared phoonnx ``dataset.jsonl`` format.

Kept free of heavy imports so the parsing rules (required fields,
malformed-line resilience, phoneme-length filtering) are testable in
environments without torch.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

_LOGGER = logging.getLogger(__name__)


def load_dataset_lines(
    jsonl_path: Path, max_phoneme_ids: Optional[int] = None
) -> Iterable[Dict]:
    """Yield utterance dicts from *jsonl_path*.

    Malformed lines (bad JSON, empty ``phoneme_ids``, missing
    ``audio_norm_path``) are logged and skipped; utterances longer than
    *max_phoneme_ids* are filtered out.
    """
    skipped = 0
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                utt = json.loads(line)
                if not utt.get("phoneme_ids") or "audio_norm_path" not in utt:
                    raise ValueError("missing phoneme_ids/audio_norm_path")
            except Exception:
                _LOGGER.exception("skipping malformed dataset line: %s", line[:120])
                continue
            if max_phoneme_ids is not None and len(utt["phoneme_ids"]) > max_phoneme_ids:
                skipped += 1
                continue
            yield utt
    if skipped:
        _LOGGER.warning("skipped %d utterances (too many phonemes)", skipped)
