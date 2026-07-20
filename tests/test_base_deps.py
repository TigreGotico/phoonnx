"""Regression test: `pip install phoonnx` (base deps only) must be enough to
import phoonnx.cli and phoonnx.model_manager, and error messages must not
reference the old phoonnx_cli.py script name.
"""
import ast
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

STDLIB_MODULES = {
    "os", "sys", "re", "json", "wave", "string", "logging", "typing",
    "dataclasses", "pathlib", "enum", "collections", "datetime",
    "unicodedata", "__future__",
}


def _base_dependency_import_names():
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    deps = data["project"]["dependencies"]
    names = set()
    for dep in deps:
        # strip version specifiers / environment markers
        dist_name = dep.split(";")[0]
        for sep in (">=", "<=", "==", "!=", "~=", ">", "<"):
            dist_name = dist_name.split(sep)[0]
        dist_name = dist_name.strip()
        names.add(DIST_TO_IMPORT.get(dist_name, dist_name.replace("-", "_")))
    return names


def _unconditional_top_level_imports(py_file: Path):
    """Return the set of top-level module names imported unconditionally
    (i.e. at module scope, not inside try/except or function/class bodies)."""
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
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
