"""Static-import check for phoonnx_train's training engines: every
unconditional top-level import in the listed modules must resolve to
stdlib, phoonnx_train.torch_compat, or a dependency declared in one of
the module's expected pyproject.toml extras."""
import ast
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
TRAIN_DIR = REPO_ROOT / "phoonnx_train"

STDLIB_MODULES = {
    "os", "sys", "re", "json", "wave", "string", "logging", "typing",
    "dataclasses", "pathlib", "enum", "collections", "datetime",
    "unicodedata", "__future__", "abc", "subprocess", "hashlib", "base64",
    "csv", "math", "optparse", "functools", "shutil", "io", "itertools",
    "tempfile", "warnings", "copy", "random", "argparse", "time",
    "importlib", "shlex", "glob", "struct", "textwrap", "uuid", "inspect",
    "multiprocessing",
}

DIST_TO_IMPORT = {
    "pytorch-lightning": "pytorch_lightning",
    "huggingface_hub": "huggingface_hub",
    "PyYAML": "yaml",
    "einops-exts": "einops_exts",
    "scikit-learn": "sklearn",
    "opt_einsum": "opt_einsum",
}

# module (relative to phoonnx_train/) -> extras it needs on top of [train]
MODULE_EXTRAS = {
    "engines/vits.py": [],
    "styletts2/export.py": ["train-styletts2"],
    "styletts2/models.py": ["train-styletts2"],
    "styletts2/aligner_module.py": ["train-styletts2"],
    "styletts2/pitch_module.py": ["train-styletts2"],
    "styletts2/Utils/PLBERT/util.py": ["train-styletts2"],
    "vocos/lightning.py": [],
    "engines/yourtts.py": [],
    "engines/zipvoice.py": [],
    "preprocess.py": ["train-data"],
    "matcha/audio.py": [],
    "optispeech/model/generator/modules/attentions/s4d.py": ["train-optispeech"],
}


def _dist_names_to_import_names(dep_specs):
    names = set()
    for dep in dep_specs:
        dist_name = dep.split(";")[0]
        for sep in (">=", "<=", "==", "!=", "~=", ">", "<"):
            dist_name = dist_name.split(sep)[0]
        dist_name = dist_name.strip().split("[")[0]
        names.add(DIST_TO_IMPORT.get(dist_name, dist_name.replace("-", "_")))
    return names


def _extras():
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["optional-dependencies"]


def _unconditional_top_level_imports(py_file: Path):
    tree = ast.parse(py_file.read_text(encoding="utf-8-sig"), filename=str(py_file))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


class TestTrainBaseDependencies(unittest.TestCase):
    def test_training_modules_only_import_declared_extras(self):
        extras = _extras()
        train_deps = _dist_names_to_import_names(extras["train"])
        failures = []
        for rel_path, extra_names in MODULE_EXTRAS.items():
            py_file = TRAIN_DIR / rel_path
            self.assertTrue(py_file.exists(), f"expected file missing: {py_file}")
            allowed = (
                STDLIB_MODULES
                | train_deps
                | {"torch", "phoonnx_train", "phoonnx", "click"}
            )
            for extra_name in extra_names:
                allowed |= _dist_names_to_import_names(extras[extra_name])
            imported = _unconditional_top_level_imports(py_file)
            missing = imported - allowed
            if missing:
                failures.append(f"{rel_path}: {sorted(missing)} (extras checked: {['train'] + extra_names})")
        self.assertFalse(
            failures,
            "training modules unconditionally import packages missing from "
            "their expected pyproject.toml extras:\n" + "\n".join(failures),
        )


if __name__ == "__main__":
    unittest.main()
