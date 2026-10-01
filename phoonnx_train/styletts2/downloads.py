"""Download the pretrained English auxiliary models used by StyleTTS2 training.

The three checkpoints (text aligner, pitch extractor, PL-BERT) are published in
the upstream `yl4579/StyleTTS2 <https://github.com/yl4579/StyleTTS2>`_ repo (MIT)
and fetched on demand instead of being vendored — they total ~135 MB.

For a new language, train replacements with the ``styletts2-aligner`` /
``styletts2-plbert`` / ``styletts2-pitch`` engines and point
``asr_path``/``asr_config``, ``plbert_dir`` and ``f0_path`` at the outputs.
"""
import logging
import os
import urllib.request
from pathlib import Path
from typing import Dict, Optional

LOG = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://github.com/yl4579/StyleTTS2/raw/main"

# relative path in upstream repo -> expected size in bytes (integrity check)
_AUX_FILES = {
    "Utils/ASR/epoch_00080.pth": 94552811,
    "Utils/ASR/config.yml": 481,
    "Utils/JDC/bst.t7": 21029926,
    "Utils/PLBERT/step_1000000.t7": 25185187,
    "Utils/PLBERT/config.yml": 915,
}


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(base) / "phoonnx" / "styletts2_aux_en"


def _download(url: str, dest: Path, expected_size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    LOG.info("downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, tmp)  # nosec B310 - fixed https URL
    size = tmp.stat().st_size
    if size != expected_size:
        tmp.unlink(missing_ok=True)
        raise IOError(f"{url}: downloaded {size} bytes, expected {expected_size}")
    tmp.replace(dest)


def download_aux_models(cache_dir: Optional[str] = None,
                        base_url: str = DEFAULT_BASE_URL) -> Dict[str, str]:
    """Fetch the yl4579 English auxiliary checkpoints (cached).

    Returns a dict with the config keys the StyleTTS2 engine consumes:
    ``asr_path``, ``asr_config``, ``f0_path`` and ``plbert_dir``.
    """
    root = Path(cache_dir) if cache_dir else default_cache_dir()
    for rel, expected_size in _AUX_FILES.items():
        dest = root / rel
        if dest.is_file() and dest.stat().st_size == expected_size:
            continue
        _download(f"{base_url}/{rel}", dest, expected_size)
    return {
        "asr_path": str(root / "Utils/ASR/epoch_00080.pth"),
        "asr_config": str(root / "Utils/ASR/config.yml"),
        "f0_path": str(root / "Utils/JDC/bst.t7"),
        "plbert_dir": str(root / "Utils/PLBERT"),
    }
