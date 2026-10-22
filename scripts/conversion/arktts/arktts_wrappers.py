"""Export wrappers that put the ArkTTS DualAR model into the phoonnx ONNX contract.

ArkTTS (``itzune/zortzi-tts``, ``Audio8/Audio8-TTS-Preview-0.6b``) ships its model code
on the Hub, so nothing is vendored here: ``AutoModel(trust_remote_code=True)`` builds the
graph and these wrappers only re-plumb the KV cache, which the upstream code hides inside
``nn.Module`` buffers that ``torch.onnx.export`` cannot carry.

Three graphs come out, and they match the contract of the official export at
``itzune/zortzi-tts-onnx`` name for name, so either set of weights drives the same runtime:

``slow_ar``
    ``codes[1, 11, T] int64``, ``input_pos[T] int64``, ``cache_key_{0..23}`` and
    ``cache_value_{0..23}``, each ``[1, 2, 2048, 64]``
    -> ``logits[1, T, 4097]``, ``slow_hidden[1, T, 896]``,
    ``key_delta_{i}`` / ``value_delta_{i}`` ``[1, 2, T, 64]``

``fast_ar``
    ``slow_hidden[1, 1, 896]``, ``token_id[1, 1] int64``, ``use_slow_hidden[1] bool``,
    ``input_pos[1] int64``, ``cache_key_{0..3}`` / ``cache_value_{0..3}`` ``[1, 2, 10, 64]``
    -> ``logits[1, 1, 4096]``, ``key_delta_{i}`` / ``value_delta_{i}`` ``[1, 2, 1, 64]``

``codec_decoder``
    ``codes[1, 10, T] int64`` -> ``audio[1, 1, samples]`` at 44.1 kHz

Two decisions in the contract are worth spelling out, because getting either wrong gives
audio that sounds plausible and is wrong:

* **The cache is a fixed 2048-wide window, not a growing tensor.** The graph writes the
  new keys and values into a copy of the cache at ``input_pos``, then attends over the
  whole window. Positions that were never written are masked out by the causal rule
  ``key <= input_pos[query]`` — this is the ``"valid_prefix"`` layout the official
  ``runtime_manifest.json`` declares. The caller still gets the deltas back so it can
  keep its own copy of the cache in sync.
* **The slow logits are sliced, not full.** Upstream masks every token outside the
  semantic range and the EOS token before it samples, so a full ``[1, T, 155776]`` output
  would be ~600 MB of prefill that sampling never reads. The graph emits the 4096 semantic
  logits followed by the EOS logit at index 4096 — the ``"semantic_then_eos"`` layout.

The wrappers hold no state: every cache tensor is an input and every update is an output,
which is what lets one graph serve both prefill and decode.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SLOW_CACHE_WIDTH = 2048
"""Positions the slow cache holds — ``config.max_seq_len``, the model's hard ceiling."""

EOS_LOGIT_INDEX = 4096
"""Where the EOS logit sits in the sliced slow output, right after the 4096 semantic ones."""


def precompute_rope(length: int, head_dim: int, base: float) -> Tensor:
    """The rotary table upstream builds, in float32 instead of bfloat16.

    Upstream casts the table to bfloat16 in ``_precompute_rope``. Keeping float32 here
    costs nothing at export time and removes a quantisation step that would otherwise be
    baked into the graph as a constant, so the exported model is never *worse* than the
    checkpoint it came from.
    """
    frequencies = 1.0 / (base ** (torch.arange(0, head_dim, 2).float()[: head_dim // 2] / head_dim))
    phases = torch.outer(torch.arange(length).float(), frequencies)
    return torch.stack((torch.cos(phases), torch.sin(phases)), dim=-1)


def apply_rope(x: Tensor, rope: Tensor) -> Tensor:
    """Rotate ``[B, T, H, D]`` by the ``[T, D/2, 2]`` table, as upstream's ``_apply_rope``.

    Upstream brackets this in ``x.float()`` / ``.to(x.dtype)`` to protect a bfloat16
    checkpoint. Export runs in float32, where both are no-ops, and they are left out on
    purpose: they survive tracing as ``Cast`` nodes with hard-coded target types, and those
    are what make the half-precision converter produce a graph ONNX Runtime will not load.
    """
    shaped = x.reshape(*x.shape[:-1], -1, 2)
    rope = rope[None, :, None]
    rotated = torch.stack(
        (
            shaped[..., 0] * rope[..., 0] - shaped[..., 1] * rope[..., 1],
            shaped[..., 1] * rope[..., 0] + shaped[..., 0] * rope[..., 1],
        ),
        dim=-1,
    )
    return rotated.flatten(3)


def attention_with_cache(
    module: nn.Module,
    x: Tensor,
    rope: Tensor,
    input_pos: Tensor,
    cache_key: Tensor,
    cache_value: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """One attention block against an explicit fixed-width cache.

    Returns the block output plus the new keys and values, so the caller can emit them as
    graph outputs. ``module`` is an upstream ``ArkttsAttention``; only its weights are used.

    The mask is ``key <= input_pos[query]`` over the whole cache window. Slots the loop has
    not written yet hold zeros, and this rule is what keeps them out of the softmax — there
    is no separate validity mask, which is why a caller that writes the deltas back to the
    wrong offset gets silently wrong attention rather than an error.
    """
    batch, length, _ = x.shape
    query_size = module.n_head * module.head_dim
    kv_size = module.n_local_heads * module.head_dim
    query, key, value = module.wqkv(x).split((query_size, kv_size, kv_size), dim=-1)
    query = query.view(batch, length, module.n_head, module.head_dim)
    key = key.view(batch, length, module.n_local_heads, module.head_dim)
    value = value.view(batch, length, module.n_local_heads, module.head_dim)
    if module.qk_norm:
        query = module.q_norm(query)
        key = module.k_norm(key)
    query = apply_rope(query, rope).transpose(1, 2)
    key = apply_rope(key, rope).transpose(1, 2)
    value = value.transpose(1, 2)

    full_key = cache_key.index_copy(2, input_pos, key)
    full_value = cache_value.index_copy(2, input_pos, value)

    repeats = module.n_head // module.n_local_heads
    attend_key = full_key.repeat_interleave(repeats, dim=1)
    attend_value = full_value.repeat_interleave(repeats, dim=1)

    key_positions = torch.arange(full_key.shape[2], device=x.device)
    mask = (key_positions[None, :] <= input_pos[:, None])[None, None]
    scores = query @ attend_key.transpose(-2, -1) / math.sqrt(module.head_dim)
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    output = torch.softmax(scores, dim=-1) @ attend_value
    output = output.transpose(1, 2).contiguous().view(batch, length, query_size)
    return module.wo(output), key, value


def block_with_cache(block: nn.Module, x, rope, input_pos, cache_key, cache_value):
    """An upstream ``ArkttsTransformerBlock`` driven through :func:`attention_with_cache`."""
    attended, key, value = attention_with_cache(
        block.attention, block.attention_norm(x), rope, input_pos, cache_key, cache_value
    )
    hidden = x + attended
    return hidden + block.feed_forward(block.ffn_norm(hidden)), key, value


class SlowARWrapper(nn.Module):
    """The 24-layer backbone: one step over ``T`` positions, cache in and cache out.

    ``T`` is the whole prompt on the first call and 1 on every call after it. The same
    graph serves both, so the runtime never loads a second copy of 600 M parameters just
    to prefill.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        config = model.config
        self.num_codebooks = int(config.num_codebooks)
        self.codebook_size = int(config.codebook_size)
        self.semantic_begin_id = int(config.semantic_begin_id)
        self.semantic_end_id = int(config.semantic_end_id)
        self.eos_token_id = int(config.eos_token_id)
        self.norm_fastlayer_input = bool(config.norm_fastlayer_input)
        self.register_buffer(
            "rope_table",
            precompute_rope(SLOW_CACHE_WIDTH, config.head_dim, config.rope_base),
            persistent=False,
        )

    def embed(self, codes: Tensor) -> Tensor:
        """Row 0 is the semantic token; rows 1..10 are the codebooks of the *same* frame.

        The codebook rows only contribute when row 0 names a semantic token. On the text
        part of the prompt they are zero-filled padding, and adding their embeddings there
        would corrupt the text conditioning — hence the ``torch.where``, which upstream
        also applies in ``ArkttsModel._embed``.
        """
        model = self.model
        codebook_embeds = [
            model.codebook_embeddings(codes[:, index + 1] + index * self.codebook_size)
            for index in range(self.num_codebooks)
        ]
        codebook_sum = torch.stack(codebook_embeds, dim=1).sum(dim=1)
        semantic = (codes[:, 0] >= self.semantic_begin_id) & (codes[:, 0] <= self.semantic_end_id)
        codebook_sum = torch.where(semantic.unsqueeze(-1), codebook_sum, torch.zeros_like(codebook_sum))
        return model.embeddings(codes[:, 0]) + codebook_sum

    def forward(self, codes: Tensor, input_pos: Tensor, *cache: Tensor) -> Tuple[Tensor, ...]:
        model = self.model
        hidden = self.embed(codes)
        rope = self.rope_table[input_pos]
        deltas: List[Tensor] = []
        for index, layer in enumerate(model.layers):
            hidden, key, value = block_with_cache(
                layer, hidden, rope, input_pos, cache[2 * index], cache[2 * index + 1]
            )
            deltas.extend((key, value))
        normalized = model.norm(hidden)
        full_logits = F.linear(normalized, model.embeddings.weight)
        logits = torch.cat(
            (
                full_logits[..., self.semantic_begin_id : self.semantic_end_id + 1],
                full_logits[..., self.eos_token_id : self.eos_token_id + 1],
            ),
            dim=-1,
        )
        slow_hidden = normalized if self.norm_fastlayer_input else hidden
        return (logits, slow_hidden, *deltas)


class FastARWrapper(nn.Module):
    """The 4-layer depth transformer that writes codebooks 1..9 of one frame.

    Position 0 reads the backbone's hidden state and exists only to seed the cache — its
    logits are discarded, exactly as upstream's ``_generate_codebooks`` discards them.
    Positions 1..9 read the previous codebook's token. ``use_slow_hidden`` selects between
    the two, so one graph covers all ten positions.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        config = model.config
        self.register_buffer(
            "rope_table",
            precompute_rope(config.num_codebooks, config.fast_head_dim, config.rope_base),
            persistent=False,
        )

    def forward(
        self,
        slow_hidden: Tensor,
        token_id: Tensor,
        use_slow_hidden: Tensor,
        input_pos: Tensor,
        *cache: Tensor,
    ) -> Tuple[Tensor, ...]:
        model = self.model
        projected = model.fast_project_in(slow_hidden)
        embedded = model.fast_embeddings(token_id)
        hidden = torch.where(use_slow_hidden.reshape(-1, 1, 1), projected, embedded)
        rope = self.rope_table[input_pos]
        deltas: List[Tensor] = []
        for index, layer in enumerate(model.fast_layers):
            hidden, key, value = block_with_cache(
                layer, hidden, rope, input_pos, cache[2 * index], cache[2 * index + 1]
            )
            deltas.extend((key, value))
        logits = model.fast_output(model.fast_norm(hidden))
        return (logits, *deltas)


class CodecDecoderWrapper(nn.Module):
    """``[1, 10, T]`` residual-VQ codes -> ``[1, 1, samples]`` waveform at 44.1 kHz.

    Both checkpoints carry a byte-identical ``codec.pth``, so this graph is interchangeable
    between them; it is exported per mirror anyway so each mirror stands alone.
    """

    def __init__(self, codec: nn.Module):
        super().__init__()
        self.codec = codec

    def forward(self, codes: Tensor) -> Tensor:
        return self.codec.decode(codes)
