"""Conditional flow matching (CFM) objective.

ZipVoice (and flow-matching TTS generally — Voicebox, Matcha-TTS, E2/F5-TTS)
trains a vector field ``v_theta(x_t, t, cond)`` that transports noise
``x_0 ~ N(0, I)`` to data ``x_1`` along a straight-line (optimal-transport)
probability path::

    x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1        t ~ U(0, 1)
    u_t = x_1 - (1 - sigma_min) * x_0                       (the target field)
    loss = || v_theta(x_t, t, cond) - u_t ||^2

At inference this field is integrated from ``t=0`` to ``t=1`` with a small
number of Euler (or higher-order) ODE steps — see the ``fm_decoder`` sampling
loop in ``phoonnx/engines/zipvoice.py`` for the runtime side of the same
math. This module is intentionally architecture-agnostic: it does not know
about Zipformer, mel spectrograms, or phoneme tokens, so it can be reused
against any backbone that exposes a ``model(x_t, t, cond) -> v_t`` callable.
"""
from typing import Callable, Optional

import torch

DEFAULT_SIGMA_MIN = 1e-5


def sample_flow_path(
    x1: torch.Tensor,
    sigma_min: float = DEFAULT_SIGMA_MIN,
    t: Optional[torch.Tensor] = None,
    x0: Optional[torch.Tensor] = None,
):
    """Sample a point on the conditional (optimal-transport) flow path.

    Args:
        x1: target data, shape ``[batch, ..., dim]``.
        sigma_min: minimum noise scale of the path (keeps ``x_t`` full rank
            at ``t=1``); ZipVoice/Voicebox use a small value like ``1e-5``.
        t: optional pre-sampled timesteps, shape ``[batch]`` in ``[0, 1]``.
            Sampled uniformly if not given.
        x0: optional pre-sampled noise, same shape as ``x1``. Sampled from
            ``N(0, I)`` if not given.

    Returns:
        ``(x_t, t, u_t)`` — the interpolated point, the timesteps used
        (broadcastable to ``x1``'s leading batch dim), and the target
        vector field ``u_t`` the model is trained to predict at ``x_t``.
    """
    if x1.dim() < 1:
        raise ValueError("x1 must have at least a batch dimension")
    batch = x1.shape[0]
    device, dtype = x1.device, x1.dtype

    if x0 is None:
        x0 = torch.randn_like(x1)
    elif x0.shape != x1.shape:
        raise ValueError(f"x0 shape {tuple(x0.shape)} != x1 shape {tuple(x1.shape)}")

    if t is None:
        t = torch.rand(batch, device=device, dtype=dtype)
    elif t.shape != (batch,):
        raise ValueError(f"t must have shape ({batch},), got {tuple(t.shape)}")

    t_bcast = t.view(batch, *([1] * (x1.dim() - 1)))
    x_t = (1 - (1 - sigma_min) * t_bcast) * x0 + t_bcast * x1
    u_t = x1 - (1 - sigma_min) * x0
    return x_t, t, u_t


def cfm_loss(
    model: Callable[..., torch.Tensor],
    x1: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    condition_mask: Optional[torch.Tensor] = None,
    sigma_min: float = DEFAULT_SIGMA_MIN,
    t: Optional[torch.Tensor] = None,
    x0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Conditional flow-matching loss for a vector-field model.

    Args:
        model: callable ``model(x_t, t, cond) -> v_t`` (called with ``cond``
            as a keyword only if ``cond is not None``, so this also accepts
            unconditional models). Must return a tensor shaped like ``x1``.
        x1: target data (e.g. a batch of prompt+target mel frames),
            shape ``[batch, frames, channels]`` (or any ``[batch, ...]``).
        cond: optional conditioning tensor forwarded to ``model`` (e.g. the
            text/speech condition from the Zipformer text encoder).
        mask: optional boolean/float mask, broadcastable to ``x1``, marking
            valid (non-padding) positions. Padded positions do not
            contribute to the loss.
        condition_mask: optional boolean/float mask, broadcastable to
            ``x1``, marking the frames that are *generation targets* —
            i.e. NOT part of the reference/prompt region the model sees as
            speech condition. When given, the loss is restricted to
            ``condition_mask & mask``: the reference frames the model can
            copy from its input must not contribute to the objective, or
            the flow is trained to reconstruct the prompt instead of
            generating the continuation. Build it with
            :func:`target_region_mask` from an in-context pair's
            ``ref_frames``.
        sigma_min: see :func:`sample_flow_path`.
        t, x0: optional pre-sampled timesteps/noise (mainly for tests that
            need determinism); sampled internally otherwise.

    Returns:
        A scalar loss tensor.
    """
    x_t, t, u_t = sample_flow_path(x1, sigma_min=sigma_min, t=t, x0=x0)
    v_t = model(x_t, t, cond) if cond is not None else model(x_t, t)

    if v_t.shape != u_t.shape:
        raise ValueError(f"model output shape {tuple(v_t.shape)} != target shape {tuple(u_t.shape)}")

    sq_err = (v_t - u_t) ** 2

    def _expand(m: torch.Tensor) -> torch.Tensor:
        m = m.to(sq_err.dtype)
        while m.dim() < sq_err.dim():
            m = m.unsqueeze(-1)
        return m.expand_as(sq_err)

    combined: Optional[torch.Tensor] = None
    if mask is not None:
        combined = _expand(mask)
    if condition_mask is not None:
        cm = _expand(condition_mask)
        combined = cm if combined is None else combined * cm
    if combined is None:
        return sq_err.mean()
    denom = combined.sum().clamp_min(1.0)
    return (sq_err * combined).sum() / denom


def target_region_mask(
    ref_frames: torch.Tensor,
    total_frames: int,
) -> torch.Tensor:
    """Boolean ``[batch, total_frames]`` mask that is True on the target
    (generated) region and False on the leading reference/prompt frames.

    Args:
        ref_frames: ``[batch]`` long tensor — per-item prompt length in
            frames (an in-context pair's ``ref_frames``).
        total_frames: padded frame count of the batch.
    """
    ar = torch.arange(total_frames, device=ref_frames.device).unsqueeze(0)
    return ar >= ref_frames.unsqueeze(1)


def drop_condition(
    cond: torch.Tensor,
    drop_ratio: float,
    training: bool = True,
) -> torch.Tensor:
    """Classifier-free-guidance condition dropout.

    Zeroes the *text* condition for a random ``drop_ratio`` fraction of the
    batch during training (the speech/prompt condition is never dropped),
    so the model also learns the unconditional field that CFG interpolates
    against at inference.

    Args:
        cond: ``[batch, ...]`` condition tensor.
        drop_ratio: probability of dropping each batch item's condition.
        training: no-op when False (inference/validation).
    """
    if not training or drop_ratio <= 0:
        return cond
    keep = (torch.rand(cond.shape[0], device=cond.device) >= drop_ratio)
    keep = keep.view(cond.shape[0], *([1] * (cond.dim() - 1)))
    return cond * keep.to(cond.dtype)
