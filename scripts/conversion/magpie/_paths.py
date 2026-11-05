"""Shared paths for the Magpie-TTS ONNX export scripts.

The checkpoint comes from the Hugging Face cache. The output directory is
``MAGPIE_ONNX_DIR`` if set, else ``./magpie_onnx``.
"""
import os

REPO = "nvidia/magpie_tts_multilingual_357m"
CHECKPOINT_FILE = "magpie_tts_multilingual_357m.nemo"


def checkpoint_path() -> str:
    """Download (or reuse the cached copy of) the NeMo checkpoint."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, CHECKPOINT_FILE)


def out_dir() -> str:
    """Directory the ONNX graphs are written to. Created if missing."""
    d = os.environ.get("MAGPIE_ONNX_DIR", os.path.abspath("magpie_onnx"))
    os.makedirs(d, exist_ok=True)
    return d


def extract_dir() -> str:
    """Extract the .nemo archive (a tar) once and return the directory.

    The archive carries the tokenizer dictionaries and the model config, which the
    asset dump copies into the export.
    """
    import tarfile
    d = os.path.join(out_dir(), "_nemo_extract")
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
        with tarfile.open(checkpoint_path()) as tar:
            tar.extractall(d)
    return d
