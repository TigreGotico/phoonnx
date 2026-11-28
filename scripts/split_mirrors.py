"""Plan the split of a Phoonnx mirror repo into one Hugging Face repo per model.

A mirror repo (``Phoonnx/phoonnx-<family>``) holds one model per top-level folder.
Each model folder becomes ``Phoonnx/<family>-<folder>``. The voice index decides
what a new repo holds: every file in the model folder, plus every file that an
index entry of that model references elsewhere in the same mirror repo, at the
same relative path, and the licence files at the top of the mirror repo. A
folder with no ONNX graph is not a model; it is listed,
and its files reach a new repo only through an index reference.

The licence comes from the mirror repo's card: the front-matter ``license``
value, and the lines of the card body that name the folder. A mirror repo
without a licence gets no new repos; it is listed as a gap. Paths inside a new
repo are the paths in the mirror repo, so an index URL changes only its repo id.

Usage: split_mirrors.py plan <repo>... --index-dir DIR --out plan.json
The plan step reads the Hub and writes nothing to it.
"""
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import click
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id

ORG = "Phoonnx"
MIRROR_PREFIX = "phoonnx-"
LICENCE_FILE = re.compile(r"(?i)(licen[cs]e|notice|copying)(\.\w+)?")
HF_FILE_URL = re.compile(r"https://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/(?P<rev>[^/]+)/(?P<path>.+)")


class SplitError(ValueError):
    """The mirror repo cannot be split as planned."""


@dataclass
class RepoPlan:
    target: str
    folder: str
    files: Dict[str, int] = field(default_factory=dict)  # path in the mirror repo -> size in bytes
    shared: List[str] = field(default_factory=list)  # files outside the folder: index references and licence files
    voice_ids: List[str] = field(default_factory=list)
    licence_lines: List[str] = field(default_factory=list)  # mirror-card lines that name the folder

    @property
    def size(self) -> int:
        return sum(self.files.values())


@dataclass
class MirrorPlan:
    source: str
    licence: Optional[str]
    repos: List[RepoPlan] = field(default_factory=list)
    not_models: List[str] = field(default_factory=list)
    top_level_models: List[str] = field(default_factory=list)
    top_level_files: List[str] = field(default_factory=list)  # neither a graph nor the card: in no new repo
    licence_gap: bool = False  # the card records no licence, so the mirror gets no new repos
    unplaced_voices: List[str] = field(default_factory=list)
    orphan_references: List[str] = field(default_factory=list)  # files other repos' voices use, in no new repo
    stays: bool = False  # fewer than two model folders: the repo already holds one model


def target_name(source: str, folder: str) -> str:
    """``Phoonnx/{family}-{folder}``: the folder folded to ASCII, each run of other characters as one ``_``."""
    family = source.split("/", 1)[1].removeprefix(MIRROR_PREFIX)
    folded = unicodedata.normalize("NFKD", folder).encode("ascii", "ignore").decode()
    target = f"{ORG}/{family}-{re.sub(r'(?:[^A-Za-z0-9.-]|_)+', '_', folded)}"
    try:
        validate_repo_id(target)
    except HFValidationError as error:
        raise SplitError(f"{source}/{folder}: {target!r} is not a valid repo name: {error}") from error
    return target


def _split_card(readme: str) -> Tuple[str, str]:
    """The card's front matter and its body."""
    readme = readme.replace("\r\n", "\n")
    front = re.match(r"---\n(.*?)\n---\n", readme, re.S)
    return (front.group(1), readme[front.end():]) if front else ("", readme)


def card_licence(readme: str) -> Optional[str]:
    """The ``license`` value of the card's front matter."""
    match = re.search(r"^license:[ \t]*(\S+)", _split_card(readme)[0], re.M)
    return match.group(1) if match else None


def _urls(value) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _urls(item)
    elif isinstance(value, list):
        for item in value:
            yield from _urls(item)


def _paths_in(source: str, entry: dict) -> List[str]:
    """Every file of ``source`` that an index entry references."""
    paths = []
    for url in _urls(entry):
        match = HF_FILE_URL.fullmatch(url.split("?")[0])
        if match and match["repo"] == source:
            paths.append(match["path"])
    return paths


def plan_mirror(source: str, files: Dict[str, int], readme: str, index: Dict[str, dict]) -> MirrorPlan:
    """The new repos for one mirror repo. ``files`` maps each path in the repo to its size."""
    plan = MirrorPlan(source, card_licence(readme))
    folders: Dict[str, Dict[str, int]] = {}
    licence_files = {path: size for path, size in files.items() if LICENCE_FILE.fullmatch(path)}
    for path, size in files.items():
        if path in licence_files:
            continue
        if "/" in path:
            folders.setdefault(path.split("/", 1)[0], {})[path] = size
        elif path.endswith(".onnx"):
            plan.top_level_models.append(path)
        elif path not in ("README.md", ".gitattributes"):
            plan.top_level_files.append(path)
    models = {name: content for name, content in folders.items() if any(p.endswith(".onnx") for p in content)}
    plan.not_models = sorted(set(folders) - set(models))
    if len(models) < 2:
        plan.stays = True
        return plan
    if not plan.licence:
        plan.licence_gap = True
        return plan
    # a licence file at the top of the mirror covers every model, so it travels with each one
    repos = {name: RepoPlan(target_name(source, name), name, {**content, **licence_files}, sorted(licence_files))
             for name, content in sorted(models.items())}
    targets: Dict[str, str] = {}
    for repo in repos.values():
        if repo.target.lower() in targets:
            raise SplitError(f"{source}: {repo.folder!r} and {targets[repo.target.lower()]!r} "
                             f"fold to the same repo {repo.target}")
        targets[repo.target.lower()] = repo.folder
    for voice_id, entry in sorted(index.items()):
        paths = _paths_in(source, entry)
        if not paths:
            continue
        model = HF_FILE_URL.fullmatch((entry.get("model_url") or "").split("?")[0])
        folder = model["path"].split("/", 1)[0] if model and model["repo"] == source else None
        if model and model["repo"] != source:  # a voice of another repo that uses files of this one
            plan.orphan_references += [p for p in paths if p.split("/", 1)[0] not in repos]
            continue
        if folder not in repos:
            plan.unplaced_voices.append(voice_id)
            continue
        repo = repos[folder]
        repo.voice_ids.append(voice_id)
        for path in paths:
            if path not in files:
                raise SplitError(f"{voice_id}: the index references {source}/{path}, which the repo does not hold")
            # phoonnx fetches "<graph>_data" next to every graph, so a copied graph brings its weights
            for needed in (path, f"{path}_data") if path.endswith(".onnx") else (path,):
                if needed in files and needed not in repo.files:
                    repo.files[needed] = files[needed]
                    repo.shared.append(needed)
    body = _split_card(readme)[1]
    for repo in repos.values():
        repo.licence_lines = [line.strip() for line in body.splitlines()
                              if re.search(rf"(?<![\w-])(?<![\w-]\.){re.escape(repo.folder)}(?![\w-]|\.[\w-])", line)]
    plan.orphan_references = sorted(set(plan.orphan_references))
    plan.repos = list(repos.values())
    return plan


def load_index(index_dir: Path) -> Dict[str, dict]:
    index: Dict[str, dict] = {}
    for index_file in sorted(index_dir.glob("*.json")):
        index.update(json.loads(index_file.read_text(encoding="utf-8")))
    return index


def _summary(plan: MirrorPlan) -> Tuple[int, int, int]:
    return len(plan.repos), sum(r.size for r in plan.repos), sum(len(r.shared) for r in plan.repos)


@click.group()
def main():
    pass


@main.command()
@click.argument("repos", nargs=-1, required=True)
@click.option("--index-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def plan(repos, index_dir, out):
    """Print and save the split plan for each mirror repo. Writes nothing to the Hub."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    index = load_index(index_dir)
    plans, failed = [], []
    for source in repos:
        try:
            files = {item.path: item.size for item in api.list_repo_tree(source, recursive=True)
                     if getattr(item, "size", None) is not None}
            readme = Path(hf_hub_download(source, "README.md")).read_text(encoding="utf-8") \
                if "README.md" in files else ""
            mirror = plan_mirror(source, files, readme, index)
        except Exception as error:  # one repo must not stop the plan; the refusal names the cause
            failed.append(f"{source}: {type(error).__name__}: {error}")
            continue
        plans.append(mirror)
        count, size, shared = _summary(mirror)
        if mirror.stays:
            click.echo(f"{source}: stays, one model")
            continue
        if mirror.licence_gap:
            click.echo(f"{source}: licence gap, the card records no licence; no new repos")
            continue
        named = sum(bool(r.licence_lines) for r in mirror.repos)
        click.echo(f"{source}: models {count} bytes {size} shared copies {shared} licence {mirror.licence or 'GAP'} "
                   f"named in the card {named} not models {mirror.not_models} unplaced voices "
                   f"{len(mirror.unplaced_voices)} orphan references {len(mirror.orphan_references)} "
                   f"top-level graphs {len(mirror.top_level_models)} top-level files left out {mirror.top_level_files}")
    total = sum(_summary(p)[0] for p in plans)
    gaps = [p.source for p in plans if p.licence_gap]
    click.echo(f"mirrors in {len(repos)} planned {len(plans)} refused {len(failed)} new repos {total} "
               f"licence gaps {len(gaps)} {gaps}")
    for error in failed:
        click.echo(f"refused: {error}")
    out.write_text(json.dumps([asdict(p) for p in plans], indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
