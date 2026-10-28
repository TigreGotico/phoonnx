"""Guard against bespoke torch-version-compat logic creeping back outside
``phoonnx_train/torch_compat.py``. All ONNX-export dynamo handling and
trusted-checkpoint loading must go through that one module."""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO_ROOT / "phoonnx_train"
CANONICAL = TRAIN_DIR / "torch_compat.py"

_DYNAMO_KWARG_RE = re.compile(r"\bdynamo\s*=")
# String-style lexicographic comparisons like `torch.__version__ >= "2.5"` or
# `torch.__version__ > '2'` — NOT legitimate int/tuple comparisons such as
# `int(torch.__version__.split(".")[0]) >= 2`.
_VERSION_STRING_CMP_RE = re.compile(
    r"torch\.__version__\s*(==|!=|<=|>=|<|>)\s*[\"']"
)


def _python_files():
    for path in TRAIN_DIR.rglob("*.py"):
        if path == CANONICAL:
            continue
        if "__pycache__" in path.parts:
            continue
        yield path


def test_no_bespoke_dynamo_kwarg_outside_torch_compat():
    offenders = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _DYNAMO_KWARG_RE.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "literal `dynamo=` found outside phoonnx_train/torch_compat.py; "
        "use phoonnx_train.torch_compat.onnx_export_kwargs() instead:\n"
        + "\n".join(offenders)
    )


def test_no_bespoke_torch_version_string_comparison_outside_torch_compat():
    offenders = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _VERSION_STRING_CMP_RE.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "lexicographic torch.__version__ string comparison found outside "
        "phoonnx_train/torch_compat.py; use packaging.version.parse via "
        "phoonnx_train.torch_compat instead:\n" + "\n".join(offenders)
    )
