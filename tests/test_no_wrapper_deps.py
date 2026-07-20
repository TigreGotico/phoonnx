"""Golden rule guard: phoonnx must NEVER depend on piper-phonemize or any other
espeak-ng wrapper library (phonemizer, espeakng_loader, ...).

Those libraries link GPL espeak-ng code and are a licensing hazard. The only
allowed espeak routes are the bundled subprocess wrapper (``EspeakPhonemizer``,
which shells out to the ``espeak-ng`` binary) and ``espyak`` (our pure-Python
port fallback).

This test scans every source file under phoonnx/, phoonnx_train/ and tests/ and
asserts none of them *import* a forbidden module, and that none appear in
pyproject.toml dependencies or extras.
"""
import ast
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
SCAN_DIRS = [REPO_ROOT / "phoonnx", REPO_ROOT / "phoonnx_train", REPO_ROOT / "tests"]

# Match module ROOTS exactly (never as substrings): ``mwl_phonemizer`` is an
# unrelated, allowed package that merely contains the string "phonemizer".
FORBIDDEN_MODULE_ROOTS = {"piper_phonemize", "phonemizer", "espeakng_loader"}
# Distribution (PyPI) spellings of the same, for the pyproject scan.
FORBIDDEN_DIST_NAMES = {"piper-phonemize", "piper_phonemize", "phonemizer",
                        "espeakng-loader", "espeakng_loader"}


def _imported_module_roots(py_file: Path):
    """Every module root imported anywhere in the file (module scope, inside
    functions, inside try/except — all of it)."""
    tree = ast.parse(py_file.read_text(encoding="utf-8-sig"), filename=str(py_file))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:  # ignore relative "from . import x" (module is None)
                roots.add(node.module.split(".")[0])
    return roots


def _dist_root(dep_spec: str) -> str:
    dist = dep_spec.split(";")[0]
    for sep in (">=", "<=", "==", "!=", "~=", ">", "<"):
        dist = dist.split(sep)[0]
    return dist.strip().split("[")[0].strip()


class TestNoWrapperDeps(unittest.TestCase):
    def test_no_forbidden_imports_in_source(self):
        offenders = []
        for scan_dir in SCAN_DIRS:
            for py_file in scan_dir.rglob("*.py"):
                hits = _imported_module_roots(py_file) & FORBIDDEN_MODULE_ROOTS
                if hits:
                    offenders.append(f"{py_file.relative_to(REPO_ROOT)}: {sorted(hits)}")
        self.assertFalse(
            offenders,
            "Forbidden espeak-wrapper imports found (GPL linking hazard); use the "
            "bundled EspeakPhonemizer subprocess wrapper or espyak instead:\n"
            + "\n".join(offenders),
        )

    def test_no_forbidden_deps_in_pyproject(self):
        with open(PYPROJECT, "rb") as f:
            data = tomllib.load(f)
        specs = list(data["project"].get("dependencies", []))
        for extra_specs in data["project"].get("optional-dependencies", {}).values():
            specs.extend(extra_specs)
        offenders = sorted(
            {_dist_root(s) for s in specs if _dist_root(s) in FORBIDDEN_DIST_NAMES}
        )
        self.assertFalse(
            offenders,
            "Forbidden espeak-wrapper packages declared in pyproject.toml: "
            + ", ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
