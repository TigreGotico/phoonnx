"""Regression test: `pip install phoonnx` (base deps only) must be enough to
import phoonnx.cli and phoonnx.model_manager, and error messages must not
reference the old phoonnx_cli.py script name.
"""
import ast
import sys
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
PHOONNX_DIR = REPO_ROOT / "phoonnx"

# Distribution (PyPI) name -> importable top-level module name, only needed
# where they differ.
DIST_TO_IMPORT = {
    "json_database": "json_database",
    "unicode_rbnf": "unicode_rbnf",
    "ovos-number-parser": "ovos_number_parser",
    "ovos-date-parser": "ovos_date_parser",
    "quebra-frases": "quebra_frases",
    "scriptconv": "scriptconv",
    "langcodes": "langcodes",
    "onnxruntime": "onnxruntime",
    "numpy": "numpy",
    "click": "click",
    "requests": "requests",
}

STDLIB_MODULES = set(sys.stdlib_module_names) | {"__future__"}


def _dist_names_to_import_names(dep_specs):
    names = set()
    for dep in dep_specs:
        # strip version specifiers / environment markers
        dist_name = dep.split(";")[0]
        for sep in (">=", "<=", "==", "!=", "~=", ">", "<"):
            dist_name = dist_name.split(sep)[0]
        dist_name = dist_name.strip()
        # drop pip "extras" markers, e.g. "gruut[de]" -> "gruut"
        dist_name = dist_name.split("[")[0]
        names.add(DIST_TO_IMPORT.get(dist_name, dist_name.replace("-", "_")))
    return names


def _base_dependency_import_names():
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    deps = data["project"]["dependencies"]
    return _dist_names_to_import_names(deps)


def _all_extras_import_names():
    """Every optional-dependencies extra, pooled together. This is
    deliberately permissive (it does not map a given module to the one
    extra it needs) — the goal is to catch genuinely undeclared third-party
    imports, not to build a full extras-resolution engine."""
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    extras = data["project"].get("optional-dependencies", {})
    names = set()
    for deps in extras.values():
        names |= _dist_names_to_import_names(deps)
    return names


def _unconditional_top_level_imports(py_file: Path):
    """Return the set of top-level module names imported unconditionally
    (i.e. at module scope, not inside try/except or function/class bodies)."""
    tree = ast.parse(py_file.read_text(encoding="utf-8-sig"), filename=str(py_file))
    names = set()
    for node in tree.body:  # only module-level statements, not nested
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


class TestBaseDependencies(unittest.TestCase):
    def test_cli_and_model_manager_import_without_extras(self):
        """phoonnx.cli and phoonnx.model_manager must be importable using
        only base dependencies."""
        import phoonnx.cli  # noqa: F401
        import phoonnx.model_manager  # noqa: F401

    def test_unconditional_third_party_imports_are_base_dependencies(self):
        base_deps = _base_dependency_import_names()
        for module_name in ("cli.py", "model_manager.py"):
            py_file = PHOONNX_DIR / module_name
            imported = _unconditional_top_level_imports(py_file)
            third_party = imported - STDLIB_MODULES
            third_party = {n for n in third_party if n != "phoonnx"}
            missing = third_party - base_deps
            self.assertFalse(
                missing,
                f"{py_file} unconditionally imports {missing}, which "
                f"is not declared in pyproject.toml base dependencies",
            )

    def test_all_phoonnx_modules_only_import_declared_dependencies(self):
        """Every top-level import anywhere under phoonnx/ must resolve to
        stdlib, a base dependency, or something declared in at least one
        optional-dependencies extra — nothing silently undeclared."""
        base_deps = _base_dependency_import_names()
        extras_deps = _all_extras_import_names()
        allowed = base_deps | extras_deps | STDLIB_MODULES | {"phoonnx"}
        failures = []
        for py_file in PHOONNX_DIR.rglob("*.py"):
            imported = _unconditional_top_level_imports(py_file)
            missing = imported - allowed
            if missing:
                failures.append(f"{py_file.relative_to(REPO_ROOT)}: {sorted(missing)}")
        self.assertFalse(
            failures,
            "modules unconditionally import third-party packages missing "
            "from pyproject.toml (base deps or any extra):\n" + "\n".join(failures),
        )

    def test_no_phoonnx_cli_py_self_references(self):
        for py_file in PHOONNX_DIR.glob("*.py"):
            content = py_file.read_text()
            self.assertNotIn(
                "phoonnx_cli.py",
                content,
                f"{py_file} still references the old 'phoonnx_cli.py' script "
                f"name; the installed console script is 'phoonnx-voices'",
            )


if __name__ == "__main__":
    unittest.main()
