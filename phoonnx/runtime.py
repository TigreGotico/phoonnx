"""Where a voice load gets its files and how it opens its ONNX graphs.

A program that holds the files itself, or runs the graphs in another runtime,
passes a :class:`Runtime` to :meth:`phoonnx.model_manager.TTSModelInfo.load` or
:meth:`phoonnx.voice.TTSVoice.load`. phoonnx.js does this in Pyodide: it gives
phoonnx the files it downloaded, and it runs the graphs in onnxruntime-web.
"""
import contextlib
import contextvars
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional


@dataclass(frozen=True)
class Runtime:
    """How a voice load gets files and opens graphs. A field left ``None`` keeps the phoonnx default.

    ``fetch(url)`` gives the local path of the file at ``url``. ``open_session(path,
    providers=..., threads=...)`` gives the session of the graph at ``path``; it may
    return ``None`` when the graph runs elsewhere. A runtime with its own
    ``open_session`` also serves the external-data files of its graphs: phoonnx
    does not read a graph to find them.

    The runtime applies to the files and sessions of one load call, including the
    graphs that an engine opens while the voice is built.
    """
    fetch: Optional[Callable[[str], Path]] = None
    open_session: Optional[Callable[..., Any]] = None


_RUNTIME: contextvars.ContextVar[Optional[Runtime]] = contextvars.ContextVar("phoonnx_runtime", default=None)


def current_runtime() -> Optional[Runtime]:
    """The runtime of the load that is running, or ``None``."""
    return _RUNTIME.get()


@contextlib.contextmanager
def using_runtime(runtime: Optional[Runtime]) -> Iterator[None]:
    """Make *runtime* the runtime of the code in the block. ``None`` keeps the one in force."""
    if runtime is None:
        yield
        return
    token = _RUNTIME.set(runtime)
    try:
        yield
    finally:
        _RUNTIME.reset(token)
