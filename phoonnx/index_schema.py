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
"""

import json
from dataclasses import MISSING, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union, get_args, get_origin

from langcodes import Language

from phoonnx.model_manager import TTSModelInfo

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
