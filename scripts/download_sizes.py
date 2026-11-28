"""Write the download size of each example voice of docs/choose_voice.md to docs/data/download_sizes.json.

    python scripts/download_sizes.py

It reads the size of each pinned Hugging Face file of the voice from the Hub, so it needs the network.
The index pins each file to a commit, so the sizes change only when the index changes.
"""
import json
import re
import sys
from dataclasses import fields
from pathlib import Path

from huggingface_hub import HfApi

from phoonnx.model_manager import TTSModelManager

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "choose_voice.md"
OUT = ROOT / "docs" / "data" / "download_sizes.json"
HF_URL = re.compile(r"https://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[0-9a-f]{40})/(?P<path>[^?]+)")


def example_voices() -> list:
    return re.findall(r"^\| `[\w-]+` \| `([^`]+)` \| \d+ MB \|$", PAGE.read_text(encoding="utf-8"), re.M)


def voice_files(info) -> set:
    urls = [getattr(info, f.name) for f in fields(info) if f.name.endswith("_url")]
    urls += list((info.aux_model_urls or {}).values())
    return {(m["repo"], m["rev"], m["path"]) for url in urls if isinstance(url, str)
            for m in [HF_URL.match(url.split("?")[0])] if m}


def main() -> None:
    manager = TTSModelManager()
    manager.load()
    manager.merge_default_voices()
    api = HfApi()
    sizes = {}
    for voice_id in example_voices():
        by_commit = {}
        for repo, rev, path in voice_files(manager.voices[voice_id]):
            by_commit.setdefault((repo, rev), []).append(path)
        total = 0
        for (repo, rev), paths in by_commit.items():
            tree = {f.path: f.size for f in api.list_repo_tree(repo, revision=rev, recursive=True)
                    if getattr(f, "size", None) is not None}
            # an ONNX graph loads with its external-data files (model.onnx_data, model.onnx.data, ...)
            total += sum(size for name, size in tree.items()
                         if any(name == path or (path.endswith(".onnx") and name.startswith((path + "_data", path + ".data")))
                                for path in paths))
        sizes[voice_id] = total
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(sizes, indent=2) + "\n", encoding="utf-8")
    print(f"{len(example_voices())} voices in, {len(sizes)} sizes out: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
