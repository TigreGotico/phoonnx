"""The built distribution carries the data files the training code loads by path.

Every other test here runs against the source checkout, where a data file is on
disk whether or not packaging ships it, so a missing ``package-data`` entry is
invisible to all of them and only shows up once someone installs a wheel.
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files the training code opens by filesystem path at runtime, as they appear
# inside the distribution.
RUNTIME_DATA_FILES = ["phoonnx_train/norm_audio/models/silero_vad.onnx"]

_IGNORED = shutil.ignore_patterns(".git", "build", "dist", "*.egg-info",
                                  "__pycache__", ".venv", ".pytest_cache")


@pytest.fixture(scope="module")
def wheel_contents(tmp_path_factory):
    """Names inside a wheel built from a pristine copy of the checkout.

    The copy keeps setuptools' ``build/`` scratch tree out of the working
    checkout, where a later build would sweep it up as a second copy of the
    package.
    """
    tmp = tmp_path_factory.mktemp("packaging")
    src = tmp / "src"
    shutil.copytree(REPO_ROOT, src, ignore=_IGNORED)
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation",
         "--outdir", str(tmp / "dist")],
        cwd=src, check=True, capture_output=True,
    )
    wheels = list((tmp / "dist").glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, built {wheels}"
    return set(zipfile.ZipFile(wheels[0]).namelist())


@pytest.mark.parametrize("relpath", RUNTIME_DATA_FILES)
def test_wheel_carries_runtime_data_file(wheel_contents, relpath):
    assert relpath in wheel_contents


def test_runtime_data_files_exist_in_checkout():
    """Guards the list above against naming a file that has moved or gone."""
    for relpath in RUNTIME_DATA_FILES:
        assert (REPO_ROOT / relpath).is_file(), relpath
