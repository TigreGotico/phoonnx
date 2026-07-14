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
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import onnxruntime

from phoonnx.util import LOG

#: A provider is either a name or a ``(name, options)`` pair.
ProviderSpec = Union[str, Tuple[str, Dict[str, Any]]]

CPU_PROVIDER = "CPUExecutionProvider"

#: Environment variable holding a comma-separated provider list, or ``auto``.
PROVIDERS_ENV_VAR = "PHOONNX_ONNX_PROVIDERS"

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


def make_session(
        model_path: Any,
        providers: Optional[Sequence[ProviderSpec]] = None,
        sess_options: Optional[onnxruntime.SessionOptions] = None,
        use_cuda: bool = False,
) -> onnxruntime.InferenceSession:
    """Create an ``InferenceSession`` on the resolved execution providers."""
    return onnxruntime.InferenceSession(
        str(model_path),
        sess_options=sess_options or onnxruntime.SessionOptions(),
        providers=resolve_providers(providers, use_cuda=use_cuda),
    )
