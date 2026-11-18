"""Validation of the bundled voice-index JSON files.

The index files are the catalogue phoonnx ships: each entry is the keyword
arguments for one :class:`~phoonnx.model_manager.TTSModelInfo`. A typo in a
field name, a URL where a mapping belongs, or a language tag no tagger can
parse only surfaces when someone tries to load that particular voice, which
may be long after the entry was added.

The schema here is read off the ``TTSModelInfo`` dataclass at import time
rather than restated, so it cannot describe a field the loader does not accept
or miss one the loader gained. Adding a field to the dataclass is all that is
needed for the index files to be allowed to carry it.

Every Hugging Face URL in an entry names a commit, never a branch: a release
then loads the files it was tested with, and an upload to the repo reaches
users only through an index change.
"""

import json
import re
from dataclasses import MISSING, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Union, get_args, get_origin

from langcodes import Language

from phoonnx.model_manager import _HF_URL, VARIANT_FIELDS, TTSModelInfo

_FIELD_TYPES: Dict[str, Any] = {f.name: f.type for f in fields(TTSModelInfo)}

#: Fields an entry must carry: those the dataclass gives no default.
REQUIRED_FIELDS = frozenset(
    f.name for f in fields(TTSModelInfo)
    if f.default is MISSING and f.default_factory is MISSING
)

#: Every field an entry may carry.
KNOWN_FIELDS = frozenset(_FIELD_TYPES)


def _describes(value: Any, annotation: Any) -> bool:
    """Whether ``value``, as it comes out of JSON, fits ``annotation``."""
    origin = get_origin(annotation)
    if origin is Union:
        return any(_describes(value, arg) for arg in get_args(annotation))
    if annotation is type(None):
        return value is None
    if origin is dict:
        if not isinstance(value, dict):
            return False
        key_type, val_type = get_args(annotation)
        return all(_describes(k, key_type) and _describes(v, val_type)
                   for k, v in value.items())
    if origin in (list, tuple, set, frozenset):
        # JSON has only one sequence, so every sequence annotation is an array
        if not isinstance(value, list):
            return False
        args = get_args(annotation)
        if not args:
            return True
        if origin is tuple and Ellipsis not in args:
            return (len(value) == len(args)
                    and all(_describes(v, a) for v, a in zip(value, args)))
        element = args[0]
        return all(_describes(v, element) for v in value)
    if annotation is Any:
        return True
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        # enum-typed fields are written as their plain string value
        return value in {member.value for member in annotation}
    if annotation is bool:
        return isinstance(value, bool)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if origin is None and isinstance(annotation, type):
        return isinstance(value, annotation)
    raise TypeError(f"index_schema cannot describe {annotation!r}")




#: A Hub revision an index URL may name: a full commit sha, so that a later
#: upload to the repo changes nothing a released index loads.
COMMIT_REVISION = re.compile(r"[0-9a-f]{40}")


def _strings(value: Any) -> Iterator[str]:
    """Every string inside ``value``, at any depth."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _revision_problems(voice_id: str, entry: Dict[str, Any]) -> List[str]:
    """Each Hub URL of ``entry`` that names a branch or tag instead of a commit."""
    problems = []
    for value in _strings(entry):
        match = _HF_URL.match(value.split("?")[0])
        if match and not COMMIT_REVISION.fullmatch(match["rev"]):
            problems.append(f"{voice_id}: {value} names revision {match['rev']!r}, "
                            f"expected a 40-hex commit sha")
    return problems


def _precision_problems(voice_id: str, entry: Dict[str, Any]) -> List[str]:
    """How the ``precision`` and ``variants`` fields of ``entry`` are wrong."""
    problems = []
    if entry.get("precision") not in (None, "int8", "full"):
        problems.append(f"{voice_id}: precision is {entry['precision']!r}, expected 'int8' or 'full'")
    variants = entry.get("variants")
    if not isinstance(variants, dict):
        return problems
    for name in sorted(set(variants) - {"int8"}):
        problems.append(f"{voice_id}: unknown variant {name!r}; the only variant is 'int8'")
    variant = variants.get("int8")
    if variant is None:
        return problems
    if entry.get("precision") == "int8":
        problems.append(f"{voice_id}: an int8-only entry cannot carry an int8 variant")
    if not set(variant) & VARIANT_FIELDS:
        problems.append(f"{voice_id}: the int8 variant names no file")
    for name, value in sorted(variant.items()):
        if name == "auto":
            if not isinstance(value, bool):
                problems.append(f"{voice_id}: variant 'auto' is {value!r}, expected true or false")
        elif name not in VARIANT_FIELDS:
            problems.append(f"{voice_id}: variant field {name!r} is not a file field")
        elif value is None:
            problems.append(f"{voice_id}: variant field {name!r} is null; leave it out to keep the base value")
        elif not _describes(value, _FIELD_TYPES[name]):
            problems.append(f"{voice_id}: variant field {name!r} has {value!r}, "
                            f"expected {_FIELD_TYPES[name]}")
    return problems


def validate_entry(voice_id: str, entry: Dict[str, Any]) -> List[str]:
    """Every way ``entry`` fails the schema, as human-readable messages."""
    problems = []

    for name in sorted(set(entry) - set(_FIELD_TYPES)):
        problems.append(f"{voice_id}: unknown field {name!r}")

    for name in sorted(REQUIRED_FIELDS - set(entry)):
        problems.append(f"{voice_id}: missing required field {name!r}")

    for name, value in sorted(entry.items()):
        if name not in _FIELD_TYPES:
            continue
        if not _describes(value, _FIELD_TYPES[name]):
            problems.append(
                f"{voice_id}: field {name!r} has {value!r}, "
                f"expected {_FIELD_TYPES[name]}"
            )

    if entry.get("voice_id") not in (None, voice_id):
        problems.append(
            f"{voice_id}: voice_id field is {entry['voice_id']!r}, "
            f"which does not match the key it is filed under"
        )

    problems += _precision_problems(voice_id, entry)
    problems += _revision_problems(voice_id, entry)

    for name in ("lang", "phonemizer_lang"):
        tag = entry.get(name)
        if isinstance(tag, str):
            try:
                Language.get(tag)
            except Exception as exc:
                problems.append(f"{voice_id}: field {name!r} is not a usable language tag: {tag!r} ({exc})")

    return problems


def validate_index_file(path: Union[str, Path]) -> List[str]:
    """Every schema problem in one voice-index JSON file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index, dict):
        return [f"{path.name}: index must be an object keyed by voice_id"]
    problems = []
    for voice_id, entry in index.items():
        if not isinstance(entry, dict):
            problems.append(f"{path.name}: entry {voice_id!r} is not an object")
            continue
        problems += [f"{path.name}: {p}" for p in validate_entry(voice_id, entry)]
    return problems
