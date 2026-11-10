"""Warm-start a vendored :class:`~phoonnx_train.mosstts.model.MossTTSNano` from a checkpoint.

Supported sources:

* an upstream Hugging Face checkpoint directory (``model.safetensors``, a sharded
  ``model.safetensors.index.json``, or ``pytorch_model.bin``) — read with ``safetensors``
  / ``torch.load`` directly, never through ``transformers`` and never with
  ``trust_remote_code``;
* a single ``.safetensors`` / ``.bin`` / ``.pt`` file;
* a Lightning ``.ckpt`` written by a previous phoonnx run (keys are ``model.*``).

The vendored module names were chosen to match upstream, so the mapping is close to the
identity — but it is applied *explicitly* here (rather than relying on it) and the
matched-parameter fraction is reported, so a checkpoint that silently fails to load is
impossible to miss.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

_LOGGER = logging.getLogger("mosstts.warmstart")

#: Renames applied to source keys before matching. Empty today (upstream and the vendored
#: module agree), kept explicit so future upstream renames land in one place.
KEY_RENAMES: Dict[str, str] = {}

#: Source keys that are expected to be absent from the vendored module.
IGNORED_SOURCE_KEYS = (
    "local_transformer.wte.weight",  # upstream replaces the local wte with nn.Identity
)


@dataclass
class WarmStartReport:
    """What the mapping actually did, in tensors and in parameters."""

    source: str
    matched: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)
    shape_mismatch: List[Tuple[str, tuple, tuple]] = field(default_factory=list)
    matched_parameters: int = 0
    total_parameters: int = 0
    tied_from_source: List[str] = field(default_factory=list)

    @property
    def matched_fraction(self) -> float:
        if self.total_parameters == 0:
            return 0.0
        return self.matched_parameters / float(self.total_parameters)

    def summary(self) -> str:
        return (
            f"{self.source}: matched {len(self.matched)} tensors / "
            f"{self.matched_parameters:,} of {self.total_parameters:,} parameters "
            f"({100.0 * self.matched_fraction:.2f}%), "
            f"missing={len(self.missing)} unexpected={len(self.unexpected)} "
            f"shape_mismatch={len(self.shape_mismatch)}"
        )


def _load_safetensors(path: Path) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return dict(load_file(str(path), device="cpu"))


def _load_torch(path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a state dict")
    return {key: value for key, value in payload.items() if isinstance(value, torch.Tensor)}


def load_source_state_dict(source: Union[str, Path]) -> Tuple[Dict[str, torch.Tensor], str]:
    """Read a checkpoint into a flat ``{key: tensor}`` mapping, whatever its container."""
    path = Path(source).expanduser()
    if path.is_dir():
        index_path = path / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            state: Dict[str, torch.Tensor] = {}
            for shard in sorted(set(index["weight_map"].values())):
                state.update(_load_safetensors(path / shard))
            return state, str(index_path)
        for candidate in ("model.safetensors", "pytorch_model.bin", "model.ckpt"):
            candidate_path = path / candidate
            if candidate_path.exists():
                return load_source_state_dict(candidate_path)
        raise FileNotFoundError(
            f"no model.safetensors / model.safetensors.index.json / pytorch_model.bin under {path}"
        )
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".safetensors":
        return _load_safetensors(path), str(path)
    return _load_torch(path), str(path)


def normalize_source_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip Lightning's ``model.`` prefix and apply :data:`KEY_RENAMES`."""
    keys = list(state)
    # Only strip the prefix when it is universal; a bare upstream checkpoint has none.
    strip = bool(keys) and all(key.startswith("model.") for key in keys)
    normalized: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key[len("model."):] if strip else key
        new_key = KEY_RENAMES.get(new_key, new_key)
        normalized[new_key] = value
    return normalized


def warm_start(
    model: torch.nn.Module,
    source: Union[str, Path],
    strict_shapes: bool = True,
    logger: Optional[logging.Logger] = None,
) -> WarmStartReport:
    """Copy every compatible tensor from *source* into *model*, in place.

    Tied heads (``text_lm_head.weight`` / ``audio_lm_heads.*``) share storage with their
    embedding, so a checkpoint that only stores the embedding still warm-starts them; the
    report lists those under ``tied_from_source`` and counts them as matched.
    """
    logger = logger or _LOGGER
    raw_state, resolved_source = load_source_state_dict(source)
    source_state = normalize_source_keys(raw_state)
    target_state = model.state_dict()
    tied = getattr(model, "tied_weight_keys", {})

    report = WarmStartReport(source=resolved_source)
    report.total_parameters = sum(int(tensor.numel()) for tensor in target_state.values())

    to_load: Dict[str, torch.Tensor] = {}
    for key, target_tensor in target_state.items():
        candidate = source_state.get(key)
        if candidate is None and key in tied:
            candidate = source_state.get(tied[key])
            if candidate is not None:
                report.tied_from_source.append(key)
        if candidate is None:
            report.missing.append(key)
            continue
        if tuple(candidate.shape) != tuple(target_tensor.shape):
            report.shape_mismatch.append((key, tuple(candidate.shape), tuple(target_tensor.shape)))
            if strict_shapes:
                continue
            continue
        to_load[key] = candidate.to(dtype=target_tensor.dtype)
        report.matched.append(key)
        report.matched_parameters += int(target_tensor.numel())

    consumed = set(report.matched) | {tied[key] for key in report.tied_from_source}
    for key in source_state:
        if key in consumed or key in IGNORED_SOURCE_KEYS:
            continue
        report.unexpected.append(key)

    model.load_state_dict(to_load, strict=False)
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    logger.info("warm start %s", report.summary())
    if report.missing:
        logger.warning("warm start: %d parameters left at their initial values: %s",
                       len(report.missing), ", ".join(report.missing[:8]))
    if report.shape_mismatch:
        logger.warning("warm start: %d shape mismatches: %s",
                       len(report.shape_mismatch), report.shape_mismatch[:4])
    return report


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m phoonnx_train.mosstts.warmstart --checkpoint <dir> [--config <json>]``"""
    import argparse

    from phoonnx_train.mosstts.config import MossTTSNanoConfig
    from phoonnx_train.mosstts.model import MossTTSNano

    parser = argparse.ArgumentParser(description="Report the warm-start coverage of a checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="upstream checkpoint dir/file or a .ckpt")
    parser.add_argument("--config", default=None, help="config JSON (defaults to <checkpoint>/config.json)")
    args = parser.parse_args(argv)

    config_path = Path(args.config) if args.config else Path(args.checkpoint) / "config.json"
    config = MossTTSNanoConfig.from_json_file(config_path)
    model = MossTTSNano(config)
    report = warm_start(model, args.checkpoint)
    print(report.summary())
    if report.missing:
        print("missing:", *report.missing, sep="\n  ")
    if report.unexpected:
        print("unexpected:", *report.unexpected, sep="\n  ")
    return 0 if not report.missing and not report.shape_mismatch else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["warm_start", "WarmStartReport", "load_source_state_dict", "normalize_source_keys"]
