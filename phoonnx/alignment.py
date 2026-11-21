"""Retrofitting a phoneme-duration output onto an exported ONNX voice.

Per-phoneme timings come out of the model as a duration tensor, and standard
exports do not expose one. :func:`export_alignment_model` adds it as a build
step, writing a patched sibling model that any later load picks up; it is what
``phoonnx-voices add-alignment`` runs, and the cheaper option, since the
surgery happens once, offline, for everyone.

:func:`ensure_alignment_session` is the runtime fallback for a voice that was
never patched: the first ``include_alignments=True`` call performs the same
surgery, caches the patched model next to the original, and loads it. It is
best-effort by design — a model with no locatable duration tensor, a missing
``onnx`` install, or a read-only model directory all degrade to "alignments
unavailable" rather than failing synthesis.
"""
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, Optional

import onnxruntime

from phoonnx.providers import make_session
from phoonnx.util import LOG

# Sentinel distinguishing "surgery not attempted yet" from a cached negative
# result (``None`` — tried once, model exposes no locatable duration tensor).
# Public because a caller caching the outcome, TTSVoice included, has to be
# able to tell the two apart.
UNSET = object()

# One lock per destination path. The runtime path is reached from whatever
# thread first asks a shared voice for alignments, and two threads writing the
# same derived model would otherwise interleave: without the lock both run the
# surgery, and without the atomic replace below one can load the half-written
# graph the other is still saving.
#
# Never pruned, like the build locks in ``phoonnx.providers``: an entry is a
# lock object keyed by a model path, the paths come from the voices installed
# on the machine, and dropping one while a thread still holds it would let the
# next caller build a second lock for the same file and race it.
_locks_guard = threading.Lock()
_locks: Dict[str, threading.Lock] = {}


def _lock_for(dest: Path) -> threading.Lock:
    key = str(dest)
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = _locks[key] = threading.Lock()
        return lock


def alignment_model_path(model_path: str) -> Path:
    """Sibling path the alignment-patched copy of ``model_path`` is kept at.

    Next to the original model by default (``<model>.alignment.onnx``). When
    ``PHOONNX_ORT_CACHE_DIR`` is set, that directory is used instead — the
    same env var :func:`phoonnx.providers.make_session` already uses for its
    optimized-graph cache, and the sensible choice here too: it signals
    "phoonnx may write derived ONNX artifacts here" and covers model
    directories that are read-only (e.g. a shared/system voice cache) without
    introducing a second env var.
    """
    src = Path(model_path)
    stem = src.name[:-len(".onnx")] if src.name.endswith(".onnx") else src.stem
    cache_dir = os.environ.get("PHOONNX_ORT_CACHE_DIR")
    directory = Path(cache_dir) if cache_dir else src.parent
    return directory / f"{stem}.alignment.onnx"


def export_alignment_model(model_path: str,
                           dest: Optional[str] = None) -> Path:
    """Write a copy of ``model_path`` with its duration tensor exposed.

    Returns the path written. Raises ``ImportError`` when the ``onnx``
    package needed for the surgery is not installed, ``ValueError`` when the
    graph has no unique duration (``Ceil``-op) tensor to promote, and
    ``OSError`` when the destination cannot be written: as a build step this
    is expected to say why it could not do the job rather than quietly
    produce a model without alignments.
    """
    import onnx

    from phoonnx.onnx_surgery import add_phoneme_alignment_output

    destination = Path(dest) if dest else alignment_model_path(model_path)
    model = onnx.load(str(model_path))
    tensor_name = add_phoneme_alignment_output(model)
    if tensor_name is None:
        raise ValueError(
            f"no unique duration (Ceil-op) tensor found in "
            f"'{model_path}'; this model cannot produce phoneme alignments")
    _save_atomically(model, destination)
    LOG.debug(f"exposed duration tensor '{tensor_name}' in '{destination}'")
    return destination


def ensure_alignment_session(
        model_path: str,
        providers=None,
) -> Optional[onnxruntime.InferenceSession]:
    """Load an alignment-capable session for ``model_path``, patching if needed.

    Returns a session with the duration tensor exposed as an extra output, or
    ``None`` when this model has no locatable duration tensor, ``onnx`` isn't
    installed, or the patched copy could not be written or loaded — every
    failure mode degrades to "no alignment available" rather than raising.

    Safe to call from several threads at once: the surgery for a given
    destination runs under a lock and the patched model is moved into place
    atomically, so a concurrent caller either waits for it or reuses the file
    already written, and never loads a partially saved graph.
    """
    dest = alignment_model_path(model_path)
    session = _load_if_fresh(dest, model_path, providers)
    if session is not None:
        return session

    with _lock_for(dest):
        # Re-checked under the lock: another thread may have written it while
        # this one waited, and redoing the surgery would rewrite the file
        # underneath the session that thread is about to load.
        session = _load_if_fresh(dest, model_path, providers)
        if session is not None:
            return session
        try:
            export_alignment_model(model_path, str(dest))
        except ImportError:
            LOG.info(
                "model has no alignment output and the 'onnx' package isn't "
                "installed to retrofit one at runtime; install "
                "'phoonnx[streaming]' (which pulls in 'onnx'), or re-export "
                "the model with `--add-phoneme-alignment`, to use "
                "include_alignments=True")
            return None
        except ValueError as e:
            LOG.info(f"no alignment output on "
                     f"'{os.path.basename(str(model_path))}': {e}; "
                     f"include_alignments=True will keep returning None for "
                     f"this voice")
            return None
        except OSError as e:
            LOG.info(f"could not write an alignment-patched copy of "
                     f"'{model_path}' to '{dest}' ({e}); "
                     f"include_alignments=True will keep returning None for "
                     f"this voice")
            return None
        except Exception as e:
            LOG.info(f"no alignment output: surgery on '{model_path}' failed "
                     f"({e})")
            return None

        try:
            return make_session(str(dest), providers=providers)
        except Exception as e:
            LOG.info(f"wrote an alignment-patched model to '{dest}' but "
                     f"failed to load it ({e}); include_alignments=True will "
                     f"keep returning None for this voice")
            return None


def _load_if_fresh(dest: Path, model_path: str,
                   providers) -> Optional[onnxruntime.InferenceSession]:
    """Load an already-patched ``dest``, if it exists and predates no edit of
    the model it was derived from. ``None`` means "not usable, do the surgery".
    """
    try:
        if not dest.is_file() or dest.stat().st_mtime < os.path.getmtime(model_path):
            return None
    except OSError:
        return None
    try:
        return make_session(str(dest), providers=providers)
    except Exception as e:
        LOG.debug(f"cached alignment model '{dest}' did not load ({e})")
        return None


def _save_atomically(model, dest: Path) -> None:
    """Save ``model`` to ``dest`` via a uniquely named temporary sibling.

    ``onnx.save`` writes incrementally, so saving straight to the shared
    destination publishes a truncated graph for as long as the write takes.
    Another process reading the name meanwhile gets that truncated graph;
    ``os.replace`` makes the file appear complete or not at all.
    """
    import onnx

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=dest.name + ".", suffix=".tmp",
                               dir=str(dest.parent))
    os.close(fd)
    try:
        onnx.save(model, tmp)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
