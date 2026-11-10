"""
Snapshot every corpus case through ``VoiceConfig.from_dict``.

The snapshot records the whole resulting config — including the tokenizer's
vocabulary and flags, and the caller's dict *after* the call, since loaders are
allowed to normalise it in place — so a refactor that claims to preserve
behaviour has something to prove it against.

Run it as a script to write a snapshot::

    python -m tests.differential.runner snapshot.json

To compare two refs, snapshot each from its own worktree and diff the files::

    git worktree add /tmp/before origin/dev
    (cd /tmp/before && PYTHONPATH=$PWD python -m tests.differential.runner /tmp/before.json)
    python -m tests.differential.runner /tmp/after.json
    python -m tests.differential.runner --diff /tmp/before.json /tmp/after.json

``--corpus-dir`` snapshots a directory of real ``config.json`` files instead of
the synthetic corpus, which is how a catalog-wide sweep is run.
"""
import copy
import dataclasses
import glob
import json
import os
import sys
import tempfile
from typing import Any, Dict, Optional

from tests.differential import corpus


def serialize(voice_config) -> Dict[str, Any]:
    """Every field of a VoiceConfig, enums as values, tokenizer fully expanded."""
    out = {}
    for f in dataclasses.fields(voice_config):
        value = getattr(voice_config, f.name)
        if f.name != "tokenizer":
            out[f.name] = getattr(value, "value", value)
        elif value is None:
            out[f.name] = None
        else:
            vocabulary = value.vocabulary
            out[f.name] = {
                "char2idx": dict(vocabulary.char2idx), "pad": vocabulary.pad,
                "blank": vocabulary.blank, "bos": vocabulary.bos, "eos": vocabulary.eos,
                "add_blank_char": value.add_blank_char,
                "add_blank_word": value.add_blank_word,
                "use_eos_bos": value.use_eos_bos,
                "blank_at_start": value.blank_at_start,
                "blank_at_end": value.blank_at_end,
            }
    return out


def _run_one(config: Dict[str, Any], kwargs: Dict[str, Any]) -> Any:
    from phoonnx.config import VoiceConfig
    config = copy.deepcopy(config)
    try:
        voice_config = VoiceConfig.from_dict(config, **kwargs)
    except Exception as e:
        return {"raised": f"{type(e).__name__}: {e}",
                "config_after": json.loads(json.dumps(config, default=str))}
    return {"voice_config": serialize(voice_config),
            "config_after": json.loads(json.dumps(config, default=str))}


def snapshot(corpus_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run the corpus (or a directory of real configs) and return the results."""
    results = {}
    if corpus_dir:
        for path in sorted(glob.glob(os.path.join(corpus_dir, "*.json"))):
            config = json.load(open(path, encoding="utf-8"))
            for i, kwargs in enumerate(corpus.KWARG_SETS):
                results[f"{os.path.basename(path)}|{i}"] = _run_one(config, kwargs)
        return results

    with tempfile.TemporaryDirectory() as tmpdir:
        for name, config, companions in corpus.build(tmpdir):
            for i, kwargs in enumerate(corpus.KWARG_SETS):
                results[f"{name}|{i}"] = _run_one(config, {**companions, **kwargs})
    return results


GOLDEN = os.path.join(os.path.dirname(__file__), "golden.json")


def main(argv) -> int:
    if argv[:1] == ["--diff"]:
        before, after = (json.load(open(p, encoding="utf-8")) for p in argv[1:3])
        keys = sorted(set(before) | set(after))
        differing = [k for k in keys if before.get(k) != after.get(k)]
        print(f"{len(keys)} cases, {len(differing)} differing")
        for key in differing:
            print(f"--- {key}\n  before: {before.get(key)}\n  after:  {after.get(key)}")
        return 1 if differing else 0

    corpus_dir = None
    if argv[:1] == ["--corpus-dir"]:
        corpus_dir, argv = argv[1], argv[2:]
    results = snapshot(corpus_dir)
    out = argv[0] if argv else GOLDEN
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, sort_keys=True, default=str)
        f.write("\n")
    print(f"wrote {out}: {len(results)} cases")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
