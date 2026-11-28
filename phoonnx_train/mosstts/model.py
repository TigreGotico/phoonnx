"""Vendored MOSS-TTS-Nano backbone (global-local / RQ-Transformer codec LM).

Architecture, per arXiv:2603.18090 ("MOSS-TTS: ...") and the upstream reference
implementation, is the **global-latent + local autoregressive** scheme, *not* the
delay-pattern parallel-head scheme the report also describes:

* the **global** transformer (12 layers, 768 hidden, RoPE, GPT-2 blocks) consumes one
  row of ``1 + n_vq`` token ids per timestep — column 0 is a text/slot token, columns
  ``1..n_vq`` are RVQ codebook ids — and sums their embeddings into a single vector. It
  emits one *global hidden state* per timestep.
* the **local** transformer (1 layer, same width) then runs over ``n_vq + 1`` positions
  for that timestep: position 0 is the global hidden state, position 1 is the embedding
  of the *text* target, positions ``2..n_vq`` are the embeddings of audio channels
  ``0..n_vq-2``. Its outputs feed ``text_lm_head`` (position 0) and
  ``audio_lm_heads[c]`` (position ``c + 1``). At training time all of those inputs are
  teacher-forced, which makes one local forward per timestep enough for the whole frame.

Module and parameter names are chosen to match the upstream checkpoint 1:1 so
:mod:`phoonnx_train.mosstts.warmstart` is a near-identity mapping. Nothing here imports
``transformers``; the GPT-2 pieces are re-implemented from the published architecture.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from phoonnx_train.mosstts.config import GPT2DecoderConfig, MossTTSNanoConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]


def _activation(name: str):
    if name == "gelu_new":
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    raise ValueError(f"unsupported activation_function={name!r}")


class RotaryEmbedding(nn.Module):
    """Interleaved-pair RoPE, matching the upstream ``repeat_interleave(2)`` layout."""

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        freqs = torch.einsum(
            "bs,d->bsd",
            position_ids.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype),
            self.inv_freq,
        )
        cos = freqs.cos().repeat_interleave(2, dim=-1).unsqueeze(2).to(dtype=dtype)
        sin = freqs.sin().repeat_interleave(2, dim=-1).unsqueeze(2).to(dtype=dtype)
        return cos, sin


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    even = hidden_states[..., ::2]
    odd = hidden_states[..., 1::2]
    return torch.stack((-odd, even), dim=-1).reshape_as(hidden_states)


def apply_rotary_pos_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (hidden_states * cos) + (rotate_half(hidden_states) * sin)


class MossMLP(nn.Module):
    def __init__(self, config: GPT2DecoderConfig) -> None:
        super().__init__()
        self.fc_in = nn.Linear(config.hidden_size, config.inner_size)
        self.fc_out = nn.Linear(config.inner_size, config.hidden_size)
        self.act = _activation(config.activation_function)
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(self.act(self.fc_in(hidden_states))))


class MossAttention(nn.Module):
    """Causal self-attention with RoPE and an optional KV cache.

    Q/K/V are kept in ``[batch, seq, heads, head_dim]`` layout (upstream's convention,
    and the one the exported ONNX caches use) and only transposed inside the kernels.
    """

    def __init__(self, config: GPT2DecoderConfig, layer_idx: int, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.layer_idx = layer_idx
        self.attn_implementation = attn_implementation
        self.attn_dropout = float(config.attn_pdrop)
        self.scale_attn_weights = bool(config.scale_attn_weights)
        self.scale_attn_by_inverse_layer_idx = bool(config.scale_attn_by_inverse_layer_idx)
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.rotary_emb: Optional[RotaryEmbedding] = None
        if config.position_embedding_type == "rope":
            self.rotary_emb = RotaryEmbedding(self.head_dim, base=config.rope_base)

    @property
    def scale(self) -> float:
        scale = 1.0
        if self.scale_attn_weights:
            scale /= math.sqrt(self.head_dim)
        if self.scale_attn_by_inverse_layer_idx:
            scale /= float(self.layer_idx + 1)
        return scale

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tensor.shape[0], tensor.shape[1]
        return tensor.reshape(batch_size, seq_len, self.embed_dim)

    @staticmethod
    def _causal_mask(attention_mask: torch.Tensor, query_length: int, key_length: int) -> torch.Tensor:
        device = attention_mask.device
        query_positions = torch.arange(query_length, device=device) + max(key_length - query_length, 0)
        key_positions = torch.arange(key_length, device=device)
        causal = (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        return causal & attention_mask[:, None, None, :].to(torch.bool)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        layer_past: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        query, key, value = self.c_attn(hidden_states).split(self.embed_dim, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(position_ids, dtype=query.dtype)
            query = apply_rotary_pos_emb(query, cos, sin)
            key = apply_rotary_pos_emb(key, cos, sin)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key.to(key.dtype), key], dim=1)
            value = torch.cat([past_value.to(value.dtype), value], dim=1)
        present = (key, value) if use_cache else None

        mask = self._causal_mask(attention_mask, query.shape[1], key.shape[1])
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        if self.attn_implementation == "sdpa":
            # A fully masked query row makes SDPA emit NaNs; unmask one aligned key for
            # padded query positions and zero their output afterwards.
            query_mask = attention_mask[:, -query.shape[1]:].to(torch.bool)
            sdpa_mask = mask
            if not bool(query_mask.all()):
                sdpa_mask = mask.expand(q.shape[0], -1, -1, -1).clone()
                bad_batch, bad_query = torch.nonzero(~query_mask, as_tuple=True)
                aligned = bad_query + max(key.shape[1] - query.shape[1], 0)
                sdpa_mask[bad_batch, :, bad_query, aligned] = True
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=sdpa_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
                scale=self.scale,
            )
            if not bool(query_mask.all()):
                output = output.masked_fill(~query_mask[:, None, :, None], 0.0)
        else:
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            if self.training and self.attn_dropout > 0:
                probs = F.dropout(probs, self.attn_dropout)
            output = torch.matmul(probs, v)

        output = self._merge_heads(output.transpose(1, 2).contiguous())
        return self.resid_dropout(self.c_proj(output)), present


class MossBlock(nn.Module):
    def __init__(self, config: GPT2DecoderConfig, layer_idx: int, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = MossAttention(config, layer_idx=layer_idx, attn_implementation=attn_implementation)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MossMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        layer_past: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        attn_output, present = self.attn(
            self.ln_1(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states, present


class MossDecoder(nn.Module):
    """Embedding-in / hidden-out GPT-2 stack, used for both the global and local halves.

    ``wte`` exists only on the global stack; the local stack is fed pre-built embeddings,
    so upstream replaces its ``wte`` with :class:`torch.nn.Identity` (and the checkpoint
    carries no ``local_transformer.wte.weight``).
    """

    def __init__(
        self,
        config: GPT2DecoderConfig,
        attn_implementation: str = "sdpa",
        with_wte: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.attn_implementation = attn_implementation
        self.position_embedding_type = config.position_embedding_type
        self.wte: nn.Module = nn.Embedding(config.vocab_size, config.hidden_size) if with_wte else nn.Identity()
        self.wpe: nn.Module = (
            nn.Embedding(config.n_positions, config.hidden_size)
            if self.position_embedding_type == "absolute"
            else nn.Identity()
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [MossBlock(config, layer_idx=i, attn_implementation=attn_implementation) for i in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = float(self.config.initializer_range)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def set_attention_implementation(self, attn_implementation: str) -> None:
        self.attn_implementation = attn_implementation
        for block in self.h:
            block.attn.attn_implementation = attn_implementation

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[KVCache, ...]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[KVCache, ...]]]:
        batch_size, seq_len, _ = inputs_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=inputs_embeds.device)
        attention_mask = attention_mask.to(dtype=torch.bool, device=inputs_embeds.device)
        query_mask = attention_mask[:, -seq_len:]

        if position_ids is None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(~attention_mask, 0)[:, -seq_len:]

        hidden_states = inputs_embeds
        if self.position_embedding_type == "absolute":
            hidden_states = hidden_states + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)
        hidden_states = hidden_states * query_mask.unsqueeze(-1).to(hidden_states.dtype)

        presents: Optional[List[KVCache]] = [] if use_cache else None
        for layer_index, block in enumerate(self.h):
            layer_past = None if past_key_values is None else past_key_values[layer_index]
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    raise ValueError("use_cache=True is incompatible with gradient checkpointing")
                hidden_states = torch.utils.checkpoint.checkpoint(
                    lambda h, m, p, b=block: b(h, attention_mask=m, position_ids=p, use_cache=False)[0],
                    hidden_states, attention_mask, position_ids,
                    use_reentrant=False,
                )
                present = None
            else:
                hidden_states, present = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    layer_past=layer_past,
                    use_cache=use_cache,
                )
            hidden_states = hidden_states * query_mask.unsqueeze(-1).to(hidden_states.dtype)
            if presents is not None:
                presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states * query_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states, (tuple(presents) if presents is not None else None)


class MossTTSNano(nn.Module):
    """The full global-local codec LM.

    ``forward`` returns only the global hidden states, matching upstream
    ``MossTTSNanoForCausalLM.forward``; the per-channel logits are produced by
    :meth:`local_logits`, which is what the training loss consumes.
    """

    def __init__(self, config: MossTTSNanoConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = MossDecoder(config.gpt2, attn_implementation=config.attn_implementation)
        hidden_size = config.hidden_size

        self.audio_embeddings = nn.ModuleList(
            [nn.Embedding(size, hidden_size) for size in config.audio_codebook_sizes]
        )
        self.text_lm_head = nn.Linear(hidden_size, config.gpt2.vocab_size, bias=False)
        self.audio_lm_heads = nn.ModuleList(
            [nn.Linear(hidden_size, size, bias=False) for size in config.audio_codebook_sizes]
        )
        self.local_transformer = MossDecoder(
            config.local_gpt2(),
            attn_implementation=config.local_transformer_attn_implementation,
            with_wte=False,
        )

        std = float(config.initializer_range)
        for module in list(self.audio_embeddings) + [self.text_lm_head] + list(self.audio_lm_heads):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        self.tie_weights()

    # ------------------------------------------------------------------
    def tie_weights(self) -> None:
        """Every head shares its input embedding, exactly as upstream does."""
        self.text_lm_head.weight = self.transformer.wte.weight
        for embedding, head in zip(self.audio_embeddings, self.audio_lm_heads):
            head.weight = embedding.weight

    @property
    def tied_weight_keys(self) -> dict:
        tied = {"text_lm_head.weight": "transformer.wte.weight"}
        tied.update(
            {f"audio_lm_heads.{i}.weight": f"audio_embeddings.{i}.weight" for i in range(self.config.n_vq)}
        )
        return tied

    def set_attention_implementation(self, attn_implementation: str, local: Optional[str] = None) -> None:
        self.transformer.set_attention_implementation(attn_implementation)
        self.local_transformer.set_attention_implementation(local or attn_implementation)

    def gradient_checkpointing_enable(self, enabled: bool = True) -> None:
        self.transformer.gradient_checkpointing = enabled

    # ------------------------------------------------------------------
    def build_inputs_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Sum the text/slot embedding and the ``n_vq`` codebook embeddings of each row.

        Columns holding ``audio_pad_token_id`` (every audio column of a text row)
        contribute nothing.
        """
        if input_ids.ndim != 3 or input_ids.shape[-1] != self.config.row_width:
            raise ValueError(
                f"expected input_ids of shape [batch, seq, {self.config.row_width}], "
                f"got {tuple(input_ids.shape)}"
            )
        inputs_embeds = self.transformer.wte(input_ids[..., 0])
        pad = int(self.config.audio_pad_token_id)
        for channel_index, embedding in enumerate(self.audio_embeddings):
            channel_ids = input_ids[..., channel_index + 1]
            valid = channel_ids.ne(pad)
            out_of_range = valid & ((channel_ids < 0) | (channel_ids >= embedding.num_embeddings))
            if bool(out_of_range.any()):
                bad = channel_ids[out_of_range]
                raise ValueError(
                    f"out-of-range audio token ids for channel {channel_index}: "
                    f"min={int(bad.min())} max={int(bad.max())} "
                    f"codebook_size={embedding.num_embeddings} audio_pad_token_id={pad}"
                )
            safe = channel_ids.masked_fill(~valid, 0)
            inputs_embeds = inputs_embeds + embedding(safe) * valid.unsqueeze(-1).to(inputs_embeds.dtype)
        return inputs_embeds

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[KVCache, ...]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[KVCache, ...]]]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("either input_ids or inputs_embeds must be given")
            inputs_embeds = self.build_inputs_embeds(input_ids)
        return self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    # ------------------------------------------------------------------
    def build_local_inputs(self, global_hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Teacher-forced local-transformer inputs for a flat batch of timesteps.

        ``global_hidden`` is ``[N, hidden]`` and ``labels`` is ``[N, n_vq + 1]`` (column 0
        the text target, columns ``1..n_vq`` the codebook targets, ``-100`` where ignored).
        The returned ``[N, n_vq + 1, hidden]`` stack is the RQ-Transformer teacher forcing:
        channel ``c``'s prediction is conditioned on the *true* channel ``c - 1``.
        """
        n_vq = self.config.n_vq
        if labels.shape[-1] != n_vq + 1:
            raise ValueError(f"labels must be [N, {n_vq + 1}], got {tuple(labels.shape)}")
        dtype = self.local_transformer.ln_f.weight.dtype
        local_inputs = torch.zeros(
            (global_hidden.shape[0], n_vq + 1, self.config.hidden_size),
            dtype=dtype,
            device=global_hidden.device,
        )
        local_inputs[:, 0, :] = global_hidden.to(dtype)

        text_targets = labels[:, 0]
        safe_text = text_targets.masked_fill(text_targets.lt(0), int(self.config.pad_token_id))
        local_inputs[:, 1, :] = self.transformer.wte(safe_text).to(dtype)

        audio_targets = labels[:, 1:]
        for channel_index in range(n_vq - 1):
            teacher_ids = audio_targets[:, channel_index]
            embedding = self.audio_embeddings[channel_index]
            valid = (teacher_ids >= 0) & (teacher_ids < embedding.num_embeddings)
            safe = teacher_ids.masked_fill(~valid, 0)
            channel_embeds = embedding(safe) * valid.unsqueeze(-1).to(embedding.weight.dtype)
            local_inputs[:, channel_index + 2, :] = channel_embeds.to(dtype)
        return local_inputs

    def local_logits(
        self,
        global_hidden: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """One teacher-forced local pass -> ``(text_logits, [audio_logits per channel])``."""
        local_inputs = self.build_local_inputs(global_hidden, labels)
        local_hidden, _ = self.local_transformer(
            inputs_embeds=local_inputs,
            attention_mask=torch.ones(local_inputs.shape[:2], dtype=torch.bool, device=local_inputs.device),
            use_cache=False,
        )
        text_logits = self.text_lm_head(local_hidden[:, 0, :])
        audio_logits = [
            head(local_hidden[:, channel_index + 1, :])
            for channel_index, head in enumerate(self.audio_lm_heads)
        ]
        return text_logits, audio_logits

    def num_parameters(self) -> int:
        seen = set()
        total = 0
        for parameter in self.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            total += parameter.numel()
        return total


__all__ = ["MossTTSNano", "MossDecoder", "MossBlock", "MossAttention", "RotaryEmbedding"]
