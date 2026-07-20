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


def make_session(
        model_path: Any,
        providers: Optional[Sequence[ProviderSpec]] = None,
        sess_options: Optional[onnxruntime.SessionOptions] = None,
        use_cuda: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
) -> onnxruntime.InferenceSession:
    """
    Create an ``InferenceSession`` on the resolved execution providers.

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
