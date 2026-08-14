"""
ONNX Runtime execution-provider selection.

Every ``InferenceSession`` in phoonnx is created through :func:`make_session`,
so a single provider decision applies to the voice model *and* to every
auxiliary graph it pulls in (vocoders, speaker encoders, text encoders,
diacritizers).

A provider list is resolved from the first of these that yields something:

1. an explicit ``providers=[...]`` argument (the caller controls the order,
   including its own fallbacks);
2. the ``PHOONNX_ONNX_PROVIDERS`` environment variable — a comma-separated
   provider list, or ``auto``;
3. auto-detection — :data:`PREFERRED_PROVIDERS` intersected with what the
   installed ``onnxruntime`` build actually offers.

Requested providers that the installed runtime does not offer are dropped with
a warning, and ``CPUExecutionProvider`` is always appended, so a session never
fails because of the provider list.

GPU providers come from the ONNX Runtime build, not from phoonnx: the default
``onnxruntime`` wheel is CPU-only (plus a few platform providers), ``ROCm``
needs ``onnxruntime-rocm``, ``DirectML`` needs ``onnxruntime-directml``, and
CUDA needs ``onnxruntime-gpu``.
"""
import hashlib
import os
import threading
import weakref
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import onnxruntime

from phoonnx.util import LOG

#: A provider is either a name or a ``(name, options)`` pair.
ProviderSpec = Union[str, Tuple[str, Dict[str, Any]]]

CPU_PROVIDER = "CPUExecutionProvider"

#: Environment variable holding a comma-separated provider list, or ``auto``.
PROVIDERS_ENV_VAR = "PHOONNX_ONNX_PROVIDERS"

#: Environment variable holding a directory to cache ORT-optimized graphs in.
#: When set (or a ``cache_dir`` is passed to :func:`make_session`), the
#: optimized model is written once and every subsequent process load reuses
#: it instead of re-running graph optimization from the raw model. Unset by
#: default — behaviour is unchanged unless a caller opts in.
CACHE_DIR_ENV_VAR = "PHOONNX_ORT_CACHE_DIR"

#: Environment variable that turns session sharing off. Set it to ``0``/
#: ``false``/``no`` to make every :func:`make_session` call build its own
#: session, as it used to. Sharing is the default because a catalog entry is
#: not a model: 646 of the bundled omnivoice voices name the same 3 GB graph
#: and differ only in the engine options applied at synthesis time.
SHARE_SESSIONS_ENV_VAR = "PHOONNX_SHARE_ONNX_SESSIONS"

#: Auto-detection preference order, best first. Only providers the installed
#: runtime reports as available are kept.
PREFERRED_PROVIDERS: List[str] = [
    "CUDAExecutionProvider",  # NVIDIA
    "ROCMExecutionProvider",  # AMD
    "MIGraphXExecutionProvider",  # AMD
    "DmlExecutionProvider",  # DirectML (Windows)
    "CoreMLExecutionProvider",  # Apple
    "OpenVINOExecutionProvider",  # Intel
    CPU_PROVIDER,
]

#: Default session options per provider, applied when a provider is requested
#: by name only.
PROVIDER_OPTIONS: Dict[str, Dict[str, Any]] = {
    "CUDAExecutionProvider": {"cudnn_conv_algo_search": "HEURISTIC"},
}


def available_providers() -> List[str]:
    """Providers offered by the installed ONNX Runtime build."""
    try:
        return list(onnxruntime.get_available_providers())
    except Exception as err:  # pragma: no cover - defensive, ORT always answers
        LOG.warning(f"could not query onnxruntime providers: {err}")
        return [CPU_PROVIDER]


def _name(provider: ProviderSpec) -> str:
    return provider[0] if isinstance(provider, tuple) else provider


def _with_options(provider: ProviderSpec) -> ProviderSpec:
    """Attach the default options of a provider given by bare name."""
    if isinstance(provider, tuple):
        return provider
    options = PROVIDER_OPTIONS.get(provider)
    return (provider, dict(options)) if options else provider


def _from_env() -> Optional[Sequence[str]]:
    raw = os.environ.get(PROVIDERS_ENV_VAR, "").strip()
    if not raw or raw.lower() == "auto":
        return None
    return [p.strip() for p in raw.split(",") if p.strip()]


def _autodetect() -> List[ProviderSpec]:
    available = available_providers()
    return [_with_options(p) for p in PREFERRED_PROVIDERS if p in available]


def resolve_providers(
        providers: Optional[Sequence[ProviderSpec]] = None,
        use_cuda: bool = False,
) -> List[ProviderSpec]:
    """
    Resolve the execution providers to run a session with.

    Parameters:
        providers: Ordered provider list; names or ``(name, options)`` pairs.
            When omitted, ``PHOONNX_ONNX_PROVIDERS`` is consulted and otherwise
            the best available provider is auto-detected.
        use_cuda: Deprecated alias for ``providers=["CUDAExecutionProvider"]``.
            Ignored when ``providers`` is given.

    Returns:
        A provider list, filtered to what the runtime offers and always ending
        in ``CPUExecutionProvider``.
    """
    if providers is None and use_cuda:
        LOG.warning("'use_cuda' is deprecated, pass "
                    "providers=['CUDAExecutionProvider'] instead")
        providers = ["CUDAExecutionProvider"]

    if providers is None:
        providers = _from_env()

    if providers is None:
        resolved = _autodetect()
    else:
        available = available_providers()
        resolved = []
        for provider in providers:
            if _name(provider) in available:
                resolved.append(_with_options(provider))
            else:
                LOG.warning(
                    f"'{_name(provider)}' is not available in this onnxruntime "
                    f"build (available: {available}), skipping it. Install the "
                    f"matching onnxruntime package to enable it."
                )

    if not any(_name(p) == CPU_PROVIDER for p in resolved):
        resolved.append(CPU_PROVIDER)

    LOG.debug(f"onnxruntime execution providers: {[_name(p) for p in resolved]}")
    return resolved


def _cache_key_path(
        model_path: Any,
        resolved_providers: Sequence[ProviderSpec],
        cache_dir: Union[str, Path],
) -> Path:
    """Deterministic cache-file path for a (model, provider list) pair.

    The key is cheap to compute (path + size + mtime, no file hashing) but
    changes whenever the model file, the onnxruntime build, or the provider
    list changes, so a stale cache is never served silently.
    """
    stat = os.stat(model_path)
    provider_key = ",".join(repr(p) for p in resolved_providers)
    raw = (
        f"{os.path.abspath(str(model_path))}|{stat.st_size}|{stat.st_mtime_ns}|"
        f"{onnxruntime.__version__}|{provider_key}"
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.ort_optimized.onnx"


def _warn_on_provider_fallback(
        requested: Sequence[ProviderSpec],
        session: onnxruntime.InferenceSession,
) -> None:
    """Warn when the first requested provider silently fell back to another.

    onnxruntime does not raise when a provider fails to initialize (e.g. a
    CUDA build advertising ``CUDAExecutionProvider`` while missing a shared
    library like ``libcublasLt``); it just falls back to the next provider
    in the list with no log line. This surfaces that so a "GPU" run that is
    actually running on CPU is not mistaken for one that is not.
    """
    if not requested:
        return
    try:
        actual = list(session.get_providers())
    except Exception:  # pragma: no cover - defensive, ORT always answers
        return
    requested_first = _name(requested[0])
    if requested_first not in actual:
        LOG.warning(
            f"requested onnxruntime provider '{requested_first}' is not "
            f"active for this session; falling back to "
            f"'{actual[0] if actual else 'unknown'}' (active providers: {actual})"
        )


#: Live shared sessions, keyed by what makes two of them interchangeable.
#: Values are weak, so a session disappears from here the moment nothing is
#: using it any more; the store never keeps a graph alive by itself and needs
#: no release call that a failing request could skip.
_SHARED_SESSIONS: "weakref.WeakValueDictionary[tuple, onnxruntime.InferenceSession]" = \
    weakref.WeakValueDictionary()
#: Guards ``_SHARED_SESSIONS`` and the per-key build locks below.
_SHARED_LOCK = threading.Lock()
#: One lock per key, so two threads asking for the same model build it once
#: while two threads asking for different models still load in parallel.
#: Keyed by model, not by voice, so this stays as small as the catalog's set
#: of distinct graphs.
_BUILD_LOCKS: Dict[tuple, threading.Lock] = {}


def _sharing_enabled() -> bool:
    """Whether identical sessions may be shared (the default)."""
    value = os.environ.get(SHARE_SESSIONS_ENV_VAR)
    if value is None:
        return True
    return value.strip().lower() not in ("0", "false", "no", "off")


def _external_data_path(*model_paths: str) -> Optional[str]:
    """Path of a model's external-weights sidecar, if it names one on disk.

    onnxruntime resolves the sidecar it finds inside the graph against the
    graph's own directory, and phoonnx fetches models with that sidecar named
    ``<model>_data`` (see ``model_manager._sidecar_url``) or, for graphs
    exported with onnxruntime's own convention, ``<model>.data``. Either one,
    if present, holds the weights the graph itself does not carry.

    Takes several candidate locations for the graph, tried in order: the hub
    cache lays a voice out as ``snapshots/<rev>/model.onnx``, a symlink whose
    sidecar sits alongside *it*, resolving to ``blobs/<sha>`` with nothing
    beside the blob at all. Probing only the resolved real path — as this
    used to — never finds that sidecar, so external-data weights fetched
    through the hub never entered the session key at all.
    """
    for model_path in model_paths:
        for candidate in (f"{model_path}_data", f"{model_path}.data"):
            if os.path.isfile(candidate):
                return candidate
    return None


def session_key(model_path: Any,
                resolved: Sequence[ProviderSpec],
                cache_dir: Optional[Union[str, Path]]) -> tuple:
    """Identity of a session: two calls with the same key are interchangeable.

    The model file is identified by its resolved real path plus its size and
    modification time, so a voice that reaches the same cached artifact
    through a different symlink or a non-normalized path shares one session,
    while a file that was replaced on disk gets a new one rather than a stale
    session over weights that are no longer there. A graph with external
    weights keeps the actual weights in a sidecar file the graph itself does
    not change size or mtime when replaced, so that sidecar's (size, mtime)
    is folded into the key too — otherwise a stable stub graph pointing at
    swapped-out weights would keep serving the old session forever.
    """
    try:
        resolved_path = os.path.realpath(str(model_path))
    except OSError:  # pragma: no cover - realpath practically never raises
        resolved_path = str(model_path)
    try:
        stat = os.stat(resolved_path)
        version = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        version = ()
    sidecar = _external_data_path(str(model_path), resolved_path)
    if sidecar is not None:
        try:
            sidecar_stat = os.stat(sidecar)
            version = version + (sidecar_stat.st_size, sidecar_stat.st_mtime_ns)
        except OSError:
            pass
    return (resolved_path, version, repr(list(resolved)), str(cache_dir or ""))


def shared_sessions() -> Dict[tuple, onnxruntime.InferenceSession]:
    """The sessions that are alive right now, for tests and diagnostics."""
    with _SHARED_LOCK:
        return dict(_SHARED_SESSIONS)


def make_session(
        model_path: Any,
        providers: Optional[Sequence[ProviderSpec]] = None,
        sess_options: Optional[onnxruntime.SessionOptions] = None,
        use_cuda: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
) -> onnxruntime.InferenceSession:
    """
    Return an ``InferenceSession`` on the resolved execution providers,
    reusing the one already loaded for this model when there is one.

    An ONNX Runtime session is thread-safe to run and holds the weights, which
    is nearly all of the memory a voice costs. Voices that name the same graph
    therefore share one session rather than each loading their own copy; what
    differs between them (the engine options, the language, the tokenizer)
    lives in the voice, not in the session. Sharing only ever happens between
    calls that would have produced identical sessions — same file, same
    providers, same optimized-graph cache, and default session options. Pass
    an explicit ``sess_options`` (or set ``PHOONNX_SHARE_ONNX_SESSIONS=0``) to
    get a private session.

    Parameters:
        cache_dir: Directory to cache the ORT-optimized graph in. When given
            (or ``PHOONNX_ORT_CACHE_DIR`` is set), the optimized model is
            written to a deterministic file the first time and every later
            session for the same model/provider list is created directly
            from that pre-optimized file with graph optimization disabled,
            skipping the optimization pass. A stale or corrupt cache file
            is deleted and the session falls back to a normal load. Default
            (no env var, no argument): unchanged behaviour.
    """
    resolved = resolve_providers(providers, use_cuda=use_cuda)
    cache_dir = cache_dir if cache_dir is not None else os.environ.get(CACHE_DIR_ENV_VAR)

    if sess_options is not None or not _sharing_enabled():
        return _build_session(model_path, resolved, sess_options, cache_dir)

    key = session_key(model_path, resolved, cache_dir)
    with _SHARED_LOCK:
        session = _SHARED_SESSIONS.get(key)
        if session is not None:
            LOG.debug(f"reusing the loaded onnx session for '{model_path}'")
            return session
        build_lock = _BUILD_LOCKS.setdefault(key, threading.Lock())

    # Built outside the store lock: a cold load of a multi-gigabyte graph takes
    # seconds to minutes, and every other model would wait behind it. The
    # per-key lock still makes two callers for the same model load it once.
    with build_lock:
        with _SHARED_LOCK:
            session = _SHARED_SESSIONS.get(key)
        if session is not None:
            return session
        session = _build_session(model_path, resolved, sess_options, cache_dir)
        with _SHARED_LOCK:
            _SHARED_SESSIONS[key] = session
            # The lock has done its job once the session it guarded is
            # built and published; keeping it around forever would grow
            # _BUILD_LOCKS by one entry per (path, size, mtime, providers,
            # cache_dir) ever seen, including stale keys from models that
            # were since replaced on disk. Drop it only if nothing raced in
            # and replaced it with a lock of its own.
            if _BUILD_LOCKS.get(key) is build_lock:
                del _BUILD_LOCKS[key]
        return session


def _build_session(
        model_path: Any,
        resolved: Sequence[ProviderSpec],
        sess_options: Optional[onnxruntime.SessionOptions],
        cache_dir: Optional[Union[str, Path]],
) -> onnxruntime.InferenceSession:
    """Load a fresh session; see :func:`make_session` for the sharing."""
    if not cache_dir:
        session = onnxruntime.InferenceSession(
            str(model_path),
            sess_options=sess_options or onnxruntime.SessionOptions(),
            providers=resolved,
        )
        _warn_on_provider_fallback(resolved, session)
        return session

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _cache_key_path(model_path, resolved, cache_dir)

    if cache_path.is_file():
        try:
            cached_options = onnxruntime.SessionOptions()
            cached_options.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            session = onnxruntime.InferenceSession(
                str(cache_path), sess_options=cached_options, providers=resolved,
            )
            _warn_on_provider_fallback(resolved, session)
            return session
        except Exception as err:
            LOG.warning(
                f"cached optimized model '{cache_path}' failed to load "
                f"({err}); removing it and rebuilding from '{model_path}'"
            )
            try:
                cache_path.unlink()
            except OSError:
                pass

    build_options = sess_options or onnxruntime.SessionOptions()
    build_options.optimized_model_filepath = str(cache_path)
    session = onnxruntime.InferenceSession(
        str(model_path), sess_options=build_options, providers=resolved,
    )
    _warn_on_provider_fallback(resolved, session)
    return session
