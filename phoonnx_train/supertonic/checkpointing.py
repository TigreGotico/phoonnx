"""Full-state checkpointing for the SuperTonic training stages.

A checkpoint carries everything needed to resume bit-for-bit: model weights,
optimizer state(s), LR-scheduler state(s), the global step, and the config +
tokenizer needed to rebuild the model. Writes are atomic (write to a temp file
in the same directory, then ``os.replace``) so an interrupted save never
corrupts the previous good checkpoint. Loading a truncated or corrupt file
raises a clear :class:`CheckpointError`.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be read or is missing required fields."""


def save_checkpoint(path: str, *, step: int, models: Dict[str, torch.nn.Module],
                    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
                    schedulers: Optional[Dict[str, Any]] = None,
                    extra: Optional[Dict[str, Any]] = None) -> None:
    """Atomically write a full checkpoint.

    ``models``/``optimizers``/``schedulers`` are name->object maps so a
    multi-network stage (e.g. the autoencoder generator + discriminators) can
    round-trip every optimizer.
    """
    payload: Dict[str, Any] = {
        "format": "supertonic-checkpoint-v1",
        "step": int(step),
        "models": {k: m.state_dict() for k, m in models.items()},
        "optimizers": {k: o.state_dict() for k, o in (optimizers or {}).items()},
        "schedulers": {k: s.state_dict() for k, s in (schedulers or {}).items()
                       if hasattr(s, "state_dict")},
        "extra": dict(extra or {}),
    }
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".ckpt-", dir=str(Path(path).parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    path = str(path)
    if not os.path.exists(path):
        raise CheckpointError(f"checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:  # truncated / corrupt / non-torch file
        raise CheckpointError(f"failed to read checkpoint {path}: {exc}") from exc
    if not isinstance(ckpt, dict) or "models" not in ckpt or "step" not in ckpt:
        raise CheckpointError(f"checkpoint {path} is missing required fields (models/step)")
    return ckpt


def resume_into(path: str, *, models: Dict[str, torch.nn.Module],
                optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
                schedulers: Optional[Dict[str, Any]] = None,
                map_location: str = "cpu") -> int:
    """Restore model/optimizer/scheduler state in place and return the step."""
    ckpt = load_checkpoint(path, map_location=map_location)
    for name, model in models.items():
        if name not in ckpt["models"]:
            raise CheckpointError(f"checkpoint {path} has no state for model {name!r}")
        model.load_state_dict(ckpt["models"][name])
    for name, opt in (optimizers or {}).items():
        if name in ckpt.get("optimizers", {}):
            opt.load_state_dict(ckpt["optimizers"][name])
    for name, sched in (schedulers or {}).items():
        if name in ckpt.get("schedulers", {}) and hasattr(sched, "load_state_dict"):
            sched.load_state_dict(ckpt["schedulers"][name])
    return int(ckpt["step"])


def load_state_dict_grow_vocab(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load ``state_dict`` into ``model``, tolerating a larger dim-0 (e.g. an
    embedding grown for a fine-tuning tokenizer with extra characters). Every
    other dimension must match exactly; extra rows keep the model's fresh init.
    """
    target = model.state_dict()
    missing = state_dict.keys() - target.keys()
    unexpected = target.keys() - state_dict.keys()
    if missing or unexpected:
        raise CheckpointError(f"key mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}")
    with torch.no_grad():
        for key, src in state_dict.items():
            dst = target[key]
            if src.shape == dst.shape:
                dst.copy_(src)
            elif src.dim() == dst.dim() and src.shape[1:] == dst.shape[1:] and src.shape[0] <= dst.shape[0]:
                dst[:src.shape[0]].copy_(src)
            else:
                raise CheckpointError(f"incompatible shape for {key}: ckpt={tuple(src.shape)} model={tuple(dst.shape)}")
