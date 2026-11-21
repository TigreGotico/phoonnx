"""Export a trained MOSS-TTS-Nano checkpoint to the multi-graph ONNX layout.

The layout is the one :class:`phoonnx.engines.mosstts.MossTTSNanoAdapter` consumes, and
the one upstream's ``onnx/export_hf_to_tts_onnx.py`` produces:

``moss_tts_prefill.onnx``
    ``input_ids [B, S, n_vq+1] i32``, ``attention_mask [B, S] i32``
    -> ``global_hidden [B, S, H]`` + ``present_key_i`` / ``present_value_i``
``moss_tts_decode_step.onnx``
    one row + ``past_valid_lengths`` + past KV -> ``global_hidden`` + present KV
``moss_tts_local_decoder.onnx``
    ``global_hidden``, ``text_token_id``, ``audio_prefix_token_ids`` -> all-channel logits
``moss_tts_local_cached_step.onnx``
    a single cached local step, so the host can sample with arbitrary parameters
``moss_tts_local_fixed_sampled_frame.onnx``
    a whole frame sampled in-graph at the upstream default temperature / top-p / top-k /
    repetition penalty, driven by host-supplied uniform randoms

The KV caches are ``[batch, seq, heads, head_dim]``, matching the adapter.

The attention math is written out explicitly here rather than reusing
:class:`~phoonnx_train.mosstts.model.MossAttention`, because the exported decode graphs
mask by a scalar ``past_valid_lengths`` (a fixed-capacity cache with a valid prefix)
instead of a boolean mask tensor. The weights are shared with the live module — nothing
is copied — so the graphs stay in sync with training by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

from phoonnx_train.mosstts.config import MossTTSNanoConfig
from phoonnx_train.mosstts.model import MossDecoder, MossTTSNano

_LOGGER = logging.getLogger("mosstts.export")

# Sampling constants baked into moss_tts_local_fixed_sampled_frame.onnx (upstream defaults).
FIXED_SAMPLED_TEXT_TEMPERATURE = 1.0
FIXED_SAMPLED_TEXT_TOP_P = 1.0
FIXED_SAMPLED_TEXT_TOP_K = 50
FIXED_SAMPLED_AUDIO_TEMPERATURE = 0.8
FIXED_SAMPLED_AUDIO_TOP_P = 0.95
FIXED_SAMPLED_AUDIO_TOP_K = 25
FIXED_SAMPLED_AUDIO_REPETITION_PENALTY = 1.2

KVCache = Tuple[torch.Tensor, torch.Tensor]


def _flatten_kv(past_key_values: Iterable[KVCache]) -> Tuple[torch.Tensor, ...]:
    flat: List[torch.Tensor] = []
    for key, value in past_key_values:
        flat.extend([key.to(torch.float32), value.to(torch.float32)])
    return tuple(flat)


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """Export-safe ``rotate_half``.

    ``torch.stack(..., dim=-1)`` is mis-lowered by the legacy ONNX tracer (it emits a
    Concat on the wrong axis, which silently breaks RoPE); an explicit unsqueeze + cat on
    a positive axis is not.
    """
    neg_odd = (-hidden_states[..., 1::2]).unsqueeze(4)
    even = hidden_states[..., ::2].unsqueeze(4)
    return torch.cat((neg_odd, even), dim=4).reshape_as(hidden_states)


def _apply_rope(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (hidden_states * cos) + (_rotate_half(hidden_states) * sin)


class ExportDecoderCore(nn.Module):
    """Trace-friendly re-expression of :class:`~phoonnx_train.mosstts.model.MossDecoder`."""

    def __init__(self, decoder: MossDecoder) -> None:
        super().__init__()
        self.decoder = decoder
        self.hidden_size = decoder.config.hidden_size
        self.num_heads = decoder.config.num_attention_heads
        self.head_dim = decoder.config.head_dim

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tensor.shape[0], tensor.shape[1]
        return tensor.reshape(batch_size, seq_len, self.hidden_size)

    @staticmethod
    def _positions_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = attention_mask.to(torch.long).cumsum(dim=-1) - 1
        return position_ids.masked_fill(~attention_mask, 0)

    @staticmethod
    def _scale(attn: nn.Module, head_dim: int) -> float:
        scale = 1.0
        if attn.scale_attn_weights:
            scale /= math.sqrt(head_dim)
        if attn.scale_attn_by_inverse_layer_idx:
            scale /= float(attn.layer_idx + 1)
        return scale

    def _qkv(self, attn: nn.Module, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        query, key, value = attn.c_attn(hidden_states).split(self.hidden_size, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        if attn.rotary_emb is not None:
            cos, sin = attn.rotary_emb(position_ids.to(torch.long), dtype=query.dtype)
            query = _apply_rope(query, cos, sin)
            key = _apply_rope(key, cos, sin)
        return query, key, value

    def _finish(self, attn: nn.Module, probs: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        attn_output = torch.matmul(probs, value.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).contiguous()
        return attn.resid_dropout(attn.c_proj(self._merge_heads(attn_output)))

    # ------------------------------------------------------------------
    def run_prefill(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[KVCache, ...]]]:
        query_mask = attention_mask[:, -inputs_embeds.shape[1]:]
        position_ids = self._positions_from_mask(attention_mask)[:, -inputs_embeds.shape[1]:]

        hidden_states = inputs_embeds
        if self.decoder.position_embedding_type == "absolute":
            hidden_states = hidden_states + self.decoder.wpe(position_ids)
        hidden_states = self.decoder.drop(hidden_states)
        hidden_states = hidden_states * query_mask.unsqueeze(-1).to(hidden_states.dtype)

        query_length = hidden_states.shape[1]
        key_length = attention_mask.shape[1]
        device = hidden_states.device
        query_positions = torch.arange(query_length, device=device) + (key_length - query_length)
        key_positions = torch.arange(key_length, device=device)
        causal = (key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        mask = causal & attention_mask[:, None, None, :]

        presents: List[KVCache] = []
        for block in self.decoder.h:
            normed = block.ln_1(hidden_states)
            query, key, value = self._qkv(block.attn, normed, position_ids)
            if use_cache:
                presents.append((key, value))
            scores = torch.matmul(
                query.permute(0, 2, 1, 3), key.permute(0, 2, 1, 3).transpose(2, 3)
            )
            scores = scores * self._scale(block.attn, self.head_dim)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            # positive axis: the tracer mis-lowers a negative softmax axis here
            attn_output = self._finish(block.attn, torch.softmax(scores, dim=3), value)
            hidden_states = hidden_states + attn_output
            hidden_states = hidden_states + block.mlp(block.ln_2(hidden_states))
            hidden_states = hidden_states * query_mask.unsqueeze(-1).to(hidden_states.dtype)

        hidden_states = self.decoder.ln_f(hidden_states)
        hidden_states = hidden_states * query_mask.unsqueeze(-1).to(hidden_states.dtype)
        return hidden_states, (tuple(presents) if use_cache else None)

    def run_decode_step(
        self,
        inputs_embeds: torch.Tensor,
        past_valid_lengths: torch.Tensor,
        past_key_values: Optional[Tuple[KVCache, ...]],
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[KVCache, ...]]]:
        """One row against a fixed-capacity cache whose valid prefix is ``past_valid_lengths``."""
        query_position_ids = past_valid_lengths.to(torch.long).unsqueeze(1)
        hidden_states = inputs_embeds
        if self.decoder.position_embedding_type == "absolute":
            hidden_states = hidden_states + self.decoder.wpe(query_position_ids)
        hidden_states = self.decoder.drop(hidden_states)
        batch_size = hidden_states.shape[0]

        presents: List[KVCache] = []
        for layer_index, block in enumerate(self.decoder.h):
            normed = block.ln_1(hidden_states)
            query, key, value = self._qkv(block.attn, normed, query_position_ids)
            if past_key_values is not None:
                past_key, past_value = past_key_values[layer_index]
                key = torch.cat([past_key.to(key.dtype), key], dim=1)
                value = torch.cat([past_value.to(value.dtype), value], dim=1)
            if use_cache:
                presents.append((key, value))

            key_length = key.shape[1]
            key_positions = torch.arange(key_length, device=key.device).view(1, 1, 1, key_length)
            valid = past_valid_lengths.to(torch.long).view(batch_size, 1, 1, 1) + 1
            queries = query_position_ids.to(torch.long).view(batch_size, 1, 1, 1)
            mask = (key_positions < valid) & (key_positions <= queries)

            scores = torch.matmul(
                query.permute(0, 2, 1, 3), key.permute(0, 2, 1, 3).transpose(2, 3)
            )
            scores = scores * self._scale(block.attn, self.head_dim)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            # positive axis: the tracer mis-lowers a negative softmax axis here
            attn_output = self._finish(block.attn, torch.softmax(scores, dim=3), value)
            hidden_states = hidden_states + attn_output
            hidden_states = hidden_states + block.mlp(block.ln_2(hidden_states))

        hidden_states = self.decoder.ln_f(hidden_states)
        return hidden_states, (tuple(presents) if use_cache else None)


def _build_inputs_embeds(model: MossTTSNano, input_ids_i32: torch.Tensor) -> torch.Tensor:
    input_ids = input_ids_i32.to(torch.long)
    inputs_embeds = model.transformer.wte(input_ids[..., 0])
    pad = int(model.config.audio_pad_token_id)
    for channel_index, embedding in enumerate(model.audio_embeddings):
        channel_ids = input_ids[..., channel_index + 1]
        valid = channel_ids.ne(pad)
        safe = torch.where(valid, channel_ids, torch.zeros_like(channel_ids))
        inputs_embeds = inputs_embeds + embedding(safe) * valid.unsqueeze(-1).to(inputs_embeds.dtype)
    return inputs_embeds.to(torch.float32)


class PrefillWrapper(nn.Module):
    def __init__(self, model: MossTTSNano) -> None:
        super().__init__()
        self.model = model
        self.core = ExportDecoderCore(model.transformer)

    def forward(self, input_ids_i32: torch.Tensor, attention_mask_i32: torch.Tensor):
        hidden_states, present = self.core.run_prefill(
            inputs_embeds=_build_inputs_embeds(self.model, input_ids_i32),
            attention_mask=attention_mask_i32.to(torch.bool),
            use_cache=True,
        )
        return (hidden_states.to(torch.float32), *_flatten_kv(present or ()))


class DecodeStepWrapper(nn.Module):
    def __init__(self, model: MossTTSNano) -> None:
        super().__init__()
        self.model = model
        self.core = ExportDecoderCore(model.transformer)
        self.num_layers = int(model.config.gpt2.n_layer)

    def forward(self, input_ids_i32: torch.Tensor, past_valid_lengths_i32: torch.Tensor, *past: torch.Tensor):
        if len(past) != self.num_layers * 2:
            raise ValueError(f"expected {self.num_layers * 2} KV tensors, got {len(past)}")
        rebuilt = tuple(
            (past[i * 2].to(torch.float32), past[i * 2 + 1].to(torch.float32))
            for i in range(self.num_layers)
        )
        hidden_states, present = self.core.run_decode_step(
            inputs_embeds=_build_inputs_embeds(self.model, input_ids_i32),
            past_valid_lengths=past_valid_lengths_i32.to(torch.int32).reshape(-1),
            past_key_values=rebuilt,
            use_cache=True,
        )
        return (hidden_states.to(torch.float32), *_flatten_kv(present or ()))


class LocalDecoderWrapper(nn.Module):
    """Whole-frame local pass with a teacher-forced audio prefix (no cache)."""

    def __init__(self, model: MossTTSNano) -> None:
        super().__init__()
        self.model = model
        self.core = ExportDecoderCore(model.local_transformer)
        self.max_audio_prefix_length = int(model.config.n_vq - 1)

    def forward(
        self,
        global_hidden: torch.Tensor,
        text_token_id_i32: torch.Tensor,
        audio_prefix_token_ids_i32: torch.Tensor,
    ):
        batch_size = global_hidden.shape[0]
        pieces = [
            global_hidden.unsqueeze(1),
            self.model.transformer.wte(text_token_id_i32.to(torch.long).reshape(batch_size)).unsqueeze(1),
        ]
        prefix_ids = audio_prefix_token_ids_i32.to(torch.long)
        pad = int(self.model.config.audio_pad_token_id)
        for prefix_index in range(self.max_audio_prefix_length):
            current = prefix_ids[:, prefix_index]
            valid = current.ne(pad)
            safe = torch.where(valid, current, torch.zeros_like(current))
            embed = self.model.audio_embeddings[prefix_index](safe)
            pieces.append((embed * valid.unsqueeze(-1).to(embed.dtype)).unsqueeze(1))

        local_inputs = torch.cat(pieces, dim=1)
        local_hidden, _ = self.core.run_prefill(
            inputs_embeds=local_inputs,
            attention_mask=torch.ones(local_inputs.shape[:2], dtype=torch.bool, device=local_inputs.device),
            use_cache=False,
        )
        text_logits = self.model.text_lm_head(local_hidden[:, 0, :]).to(torch.float32)
        audio_logits = torch.stack(
            [head(local_hidden[:, index + 1, :]).to(torch.float32)
             for index, head in enumerate(self.model.audio_lm_heads)],
            dim=1,
        )
        return text_logits, audio_logits


class LocalCachedStepWrapper(nn.Module):
    """One cached local step. ``step_type`` picks the input: 0 global, 1 text, 2 audio."""

    def __init__(self, model: MossTTSNano) -> None:
        super().__init__()
        self.model = model
        self.core = ExportDecoderCore(model.local_transformer)
        self.num_layers = int(model.config.local_transformer_layers)

    def _audio_embedding(self, audio_token_ids: torch.Tensor, channel_indices: torch.Tensor) -> torch.Tensor:
        pieces = []
        for channel_index, embedding in enumerate(self.model.audio_embeddings):
            active = channel_indices.eq(channel_index)
            safe = torch.where(active, audio_token_ids, torch.zeros_like(audio_token_ids))
            embed = embedding(safe)
            pieces.append(embed * active.unsqueeze(-1).to(embed.dtype))
        return torch.stack(pieces, dim=0).sum(dim=0)

    def forward(
        self,
        global_hidden: torch.Tensor,
        text_token_id_i32: torch.Tensor,
        audio_token_id_i32: torch.Tensor,
        channel_index_i32: torch.Tensor,
        step_type_i32: torch.Tensor,
        past_valid_lengths_i32: torch.Tensor,
        *past: torch.Tensor,
    ):
        if len(past) != self.num_layers * 2:
            raise ValueError(f"expected {self.num_layers * 2} KV tensors, got {len(past)}")
        batch_size = global_hidden.shape[0]
        rebuilt = tuple(
            (past[i * 2].to(torch.float32), past[i * 2 + 1].to(torch.float32))
            for i in range(self.num_layers)
        )
        step_type = step_type_i32.to(torch.long).reshape(batch_size)
        text_embed = self.model.transformer.wte(text_token_id_i32.to(torch.long).reshape(batch_size))
        audio_embed = self._audio_embedding(
            audio_token_id_i32.to(torch.long).reshape(batch_size),
            channel_index_i32.to(torch.long).reshape(batch_size),
        )
        dtype = text_embed.dtype
        input_embed = (
            global_hidden.to(dtype) * step_type.eq(0).unsqueeze(-1).to(dtype)
            + text_embed * step_type.eq(1).unsqueeze(-1).to(dtype)
            + audio_embed * step_type.eq(2).unsqueeze(-1).to(dtype)
        )
        local_hidden, present = self.core.run_decode_step(
            inputs_embeds=input_embed.unsqueeze(1),
            past_valid_lengths=past_valid_lengths_i32.to(torch.int32).reshape(-1),
            past_key_values=rebuilt,
            use_cache=True,
        )
        last_hidden = local_hidden[:, 0, :]
        text_logits = self.model.text_lm_head(last_hidden).to(torch.float32)
        audio_logits = torch.stack(
            [head(last_hidden).to(torch.float32) for head in self.model.audio_lm_heads], dim=1
        )
        return (text_logits, audio_logits, *_flatten_kv(present or ()))


def apply_repetition_penalty_from_seen_mask(
    logits: torch.Tensor, seen_mask_i32: torch.Tensor, penalty: torch.Tensor
) -> torch.Tensor:
    penalty = penalty.to(dtype=logits.dtype).reshape(-1, 1)
    penalized = torch.where(logits < 0, logits * penalty, logits / penalty)
    return torch.where(seen_mask_i32.to(torch.bool), penalized, logits)


def sample_from_topk_topp_with_random_u(
    logits: torch.Tensor, random_u: torch.Tensor, *, temperature: float, top_k: int, top_p: float
) -> torch.Tensor:
    """Inverse-CDF sampling over the top-k/top-p tail, with the RNG supplied by the host."""
    scores = logits.to(torch.float32)
    if temperature != 1.0:
        scores = scores / float(temperature)
    topk_scores, topk_indices = torch.topk(scores, k=int(top_k), dim=1, largest=True, sorted=True)
    if 0.0 < top_p < 1.0:
        probs = torch.softmax(topk_scores, dim=1)
        prev_cumsum = torch.cumsum(probs, dim=1) - probs
        topk_scores = topk_scores.masked_fill(~(prev_cumsum < float(top_p)), float("-inf"))
    cdf = torch.cumsum(torch.softmax(topk_scores, dim=1), dim=1)
    clamped = torch.clamp(random_u.to(cdf.dtype).reshape(-1, 1), min=0.0, max=0.99999994)
    positions = torch.clamp(torch.sum((cdf < clamped).to(torch.int64), dim=1), max=topk_indices.shape[1] - 1)
    return topk_indices.gather(1, positions.unsqueeze(1)).squeeze(1).to(torch.long)


class LocalFixedSampledFrameWrapper(nn.Module):
    """A whole frame, sampled in-graph at the upstream default parameters."""

    def __init__(self, model: MossTTSNano) -> None:
        super().__init__()
        self.model = model
        self.core = ExportDecoderCore(model.local_transformer)
        self.num_layers = int(model.config.local_transformer_layers)
        self.num_heads = model.local_transformer.config.num_attention_heads
        self.head_dim = model.local_transformer.config.head_dim
        self.num_channels = int(model.config.n_vq)
        sizes = set(int(size) for size in model.config.audio_codebook_sizes)
        if len(sizes) != 1:
            raise ValueError("the fixed-frame graph requires one codebook size across all channels")
        self.audio_codebook_size = sizes.pop()
        self.top_k = min(FIXED_SAMPLED_AUDIO_TOP_K, self.audio_codebook_size)

    def _empty_past(self, batch_size: int, device: torch.device) -> Tuple[KVCache, ...]:
        return tuple(
            (
                torch.zeros((batch_size, 0, self.num_heads, self.head_dim), dtype=torch.float32, device=device),
                torch.zeros((batch_size, 0, self.num_heads, self.head_dim), dtype=torch.float32, device=device),
            )
            for _ in range(self.num_layers)
        )

    def forward(
        self,
        global_hidden: torch.Tensor,
        repetition_seen_mask_i32: torch.Tensor,
        assistant_random_u_f32: torch.Tensor,
        audio_random_u_f32: torch.Tensor,
    ):
        batch_size = global_hidden.shape[0]
        device = global_hidden.device
        past_valid_lengths = torch.zeros((batch_size,), dtype=torch.int32, device=device)
        present = self._empty_past(batch_size, device)
        penalty = torch.full(
            (batch_size,), FIXED_SAMPLED_AUDIO_REPETITION_PENALTY, dtype=torch.float32, device=device
        )

        local_hidden, present = self.core.run_decode_step(
            inputs_embeds=global_hidden.unsqueeze(1),
            past_valid_lengths=past_valid_lengths,
            past_key_values=present,
            use_cache=True,
        )
        text_logits = self.model.text_lm_head(local_hidden[:, 0, :]).to(torch.float32)
        candidates = torch.stack(
            [
                text_logits[:, int(self.model.config.audio_assistant_slot_token_id)],
                text_logits[:, int(self.model.config.audio_end_token_id)],
            ],
            dim=1,
        )
        if FIXED_SAMPLED_TEXT_TEMPERATURE != 1.0:
            candidates = candidates / float(FIXED_SAMPLED_TEXT_TEMPERATURE)
        assistant_probs = torch.softmax(candidates, dim=1)[:, 0]
        assistant_u = torch.clamp(
            assistant_random_u_f32.to(assistant_probs.dtype).reshape(-1), min=0.0, max=0.99999994
        )
        assistant_selected = assistant_u <= assistant_probs
        next_text_token = torch.where(
            assistant_selected,
            torch.full((batch_size,), int(self.model.config.audio_assistant_slot_token_id),
                       dtype=torch.long, device=device),
            torch.full((batch_size,), int(self.model.config.audio_end_token_id),
                       dtype=torch.long, device=device),
        )

        generated: List[torch.Tensor] = []
        current_embed = self.model.transformer.wte(next_text_token).unsqueeze(1)
        current_valid = past_valid_lengths + 1
        audio_u = audio_random_u_f32.to(torch.float32)

        for channel_index in range(self.num_channels):
            local_hidden, present = self.core.run_decode_step(
                inputs_embeds=current_embed,
                past_valid_lengths=current_valid,
                past_key_values=present,
                use_cache=True,
            )
            audio_logits = self.model.audio_lm_heads[channel_index](local_hidden[:, 0, :]).to(torch.float32)
            penalized = apply_repetition_penalty_from_seen_mask(
                audio_logits, repetition_seen_mask_i32[:, channel_index, : self.audio_codebook_size], penalty
            )
            sampled = sample_from_topk_topp_with_random_u(
                penalized,
                audio_u[:, channel_index],
                temperature=FIXED_SAMPLED_AUDIO_TEMPERATURE,
                top_k=self.top_k,
                top_p=FIXED_SAMPLED_AUDIO_TOP_P,
            )
            generated.append(sampled.to(torch.int32))
            current_valid = current_valid + 1
            if channel_index + 1 < self.num_channels:
                current_embed = self.model.audio_embeddings[channel_index](sampled).unsqueeze(1)

        return assistant_selected.to(torch.int32).reshape(batch_size, 1), torch.stack(generated, dim=1)


# ----------------------------------------------------------------------
# external data handling (ported from upstream export_hf_to_tts_onnx.py)
# ----------------------------------------------------------------------
def externalize_onnx_file(onnx_path: Path) -> Tuple[Path, Path]:
    import onnx
    from onnx import external_data_helper

    resolved = onnx_path.expanduser().resolve()
    data_path = resolved.with_suffix(".data")
    temp_path = resolved.with_suffix(".onnx.tmp")
    model = onnx.load_model(str(resolved), load_external_data=True)
    for path in (data_path, temp_path):
        if path.exists():
            path.unlink()
    external_data_helper.convert_model_to_external_data(
        model, all_tensors_to_one_file=True, location=data_path.name,
        size_threshold=1024, convert_attribute=False,
    )
    onnx.save_model(model, str(temp_path))
    temp_path.replace(resolved)
    return resolved, data_path


def merge_shared_external_data(onnx_paths: Sequence[Path], shared_data_path: Path) -> None:
    """De-duplicate identical initializers across graphs into one ``.data`` blob."""
    import onnx
    from onnx import TensorProto, external_data_helper

    resolved_paths = [path.expanduser().resolve() for path in onnx_paths]
    if not resolved_paths:
        raise ValueError("onnx_paths must not be empty")
    shared_data_path = shared_data_path.expanduser().resolve()
    models = [
        (path,
         onnx.load_model(str(path), load_external_data=False),
         onnx.load_model(str(path), load_external_data=True))
        for path in resolved_paths
    ]

    def is_external(tensor) -> bool:
        return tensor.data_location == TensorProto.EXTERNAL or bool(tensor.external_data)

    unique: Dict[str, Tuple[int, int]] = {}
    blob = bytearray()
    for _path, meta, data in models:
        for tensor_meta, tensor_data in zip(meta.graph.initializer, data.graph.initializer):
            if not is_external(tensor_meta):
                continue
            raw = bytes(tensor_data.raw_data)
            digest = hashlib.sha256(raw).hexdigest()
            if digest in unique:
                continue
            unique[digest] = (len(blob), len(raw))
            blob.extend(raw)

    shared_data_path.parent.mkdir(parents=True, exist_ok=True)
    shared_data_path.write_bytes(bytes(blob))

    for path, meta, data in models:
        for tensor_meta, tensor_data in zip(meta.graph.initializer, data.graph.initializer):
            if not is_external(tensor_meta):
                continue
            raw = bytes(tensor_data.raw_data)
            offset, length = unique[hashlib.sha256(raw).hexdigest()]
            # set_external_data reads raw_data before clearing it, so it has to be present
            tensor_meta.raw_data = raw
            tensor_meta.data_location = TensorProto.EXTERNAL
            external_data_helper.set_external_data(
                tensor_meta, location=shared_data_path.name, offset=offset, length=length
            )
            tensor_meta.ClearField("raw_data")
        path.write_bytes(meta.SerializeToString())


# ----------------------------------------------------------------------
def export_moss_tts_onnx(
    model: MossTTSNano,
    output_dir: Path,
    opset: int = 17,
    sample_seq_len: int = 24,
    sample_past_len: int = 24,
    external_data: bool = False,
    checkpoint_path: str = "",
) -> Path:
    """Write the five graphs plus ``tts_browser_onnx_meta.json``; returns the meta path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device="cpu", dtype=torch.float32).eval()
    model.set_attention_implementation("eager")
    config: MossTTSNanoConfig = model.config

    n_layer = int(config.gpt2.n_layer)
    n_head = int(config.gpt2.n_head)
    head_dim = int(config.gpt2.head_dim)
    hidden = int(config.hidden_size)
    row_width = int(config.row_width)
    local_layers = int(config.local_transformer_layers)
    local_heads = int(model.local_transformer.config.num_attention_heads)
    local_head_dim = int(model.local_transformer.config.head_dim)

    present_names = ["global_hidden"] + [
        name for i in range(n_layer) for name in (f"present_key_{i}", f"present_value_{i}")
    ]
    decode_inputs = ["input_ids", "past_valid_lengths"] + [
        name for i in range(n_layer) for name in (f"past_key_{i}", f"past_value_{i}")
    ]
    local_cached_inputs = [
        "global_hidden", "text_token_id", "audio_token_id", "channel_index",
        "step_type", "past_valid_lengths",
    ] + [name for i in range(local_layers) for name in (f"local_past_key_{i}", f"local_past_value_{i}")]
    local_cached_outputs = ["text_logits", "audio_logits"] + [
        name for i in range(local_layers) for name in (f"local_present_key_{i}", f"local_present_value_{i}")
    ]

    prefill_axes = {
        "input_ids": {0: "batch", 1: "prefill_seq"},
        "attention_mask": {0: "batch", 1: "prefill_seq"},
        "global_hidden": {0: "batch", 1: "prefill_seq"},
    }
    decode_axes = {
        "input_ids": {0: "batch", 1: "step_seq"},
        "past_valid_lengths": {0: "batch"},
        "global_hidden": {0: "batch", 1: "step_seq"},
    }
    for i in range(n_layer):
        prefill_axes[f"present_key_{i}"] = {0: "batch", 1: "prefill_seq"}
        prefill_axes[f"present_value_{i}"] = {0: "batch", 1: "prefill_seq"}
        decode_axes[f"past_key_{i}"] = {0: "batch", 1: "past_seq"}
        decode_axes[f"past_value_{i}"] = {0: "batch", 1: "past_seq"}
        decode_axes[f"present_key_{i}"] = {0: "batch", 1: "total_seq"}
        decode_axes[f"present_value_{i}"] = {0: "batch", 1: "total_seq"}
    local_cached_axes = {
        name: {0: "batch"}
        for name in ("global_hidden", "text_token_id", "audio_token_id", "channel_index",
                     "step_type", "past_valid_lengths", "text_logits", "audio_logits")
    }
    for i in range(local_layers):
        local_cached_axes[f"local_past_key_{i}"] = {0: "batch", 1: "local_past_seq"}
        local_cached_axes[f"local_past_value_{i}"] = {0: "batch", 1: "local_past_seq"}
        local_cached_axes[f"local_present_key_{i}"] = {0: "batch", 1: "local_total_seq"}
        local_cached_axes[f"local_present_value_{i}"] = {0: "batch", 1: "local_total_seq"}
    fixed_axes = {
        name: {0: "batch"}
        for name in ("global_hidden", "repetition_seen_mask", "assistant_random_u",
                     "audio_random_u", "should_continue", "frame_token_ids")
    }

    prefill_ids = torch.full((1, sample_seq_len, row_width), int(config.audio_pad_token_id), dtype=torch.int32)
    prefill_ids[:, :, 0] = int(config.pad_token_id)
    prefill_mask = torch.ones((1, sample_seq_len), dtype=torch.int32)
    decode_ids = torch.full((1, 1, row_width), int(config.audio_pad_token_id), dtype=torch.int32)
    decode_ids[:, :, 0] = int(config.pad_token_id)
    past = tuple(
        tensor
        for _ in range(n_layer)
        for tensor in (
            torch.zeros((1, sample_past_len, n_head, head_dim), dtype=torch.float32),
            torch.zeros((1, sample_past_len, n_head, head_dim), dtype=torch.float32),
        )
    )
    local_past = tuple(
        tensor
        for _ in range(local_layers)
        for tensor in (
            torch.zeros((1, sample_past_len, local_heads, local_head_dim), dtype=torch.float32),
            torch.zeros((1, sample_past_len, local_heads, local_head_dim), dtype=torch.float32),
        )
    )

    from phoonnx_train.torch_compat import onnx_export_kwargs

    def export(module, name, args, input_names, output_names, dynamic_axes):
        path = output_dir / name
        with torch.no_grad():
            torch.onnx.export(
                module, args, str(path),
                input_names=input_names, output_names=output_names,
                dynamic_axes=dynamic_axes, opset_version=int(opset),
                do_constant_folding=True, **onnx_export_kwargs(),
            )
        _LOGGER.info("exported %s", path)
        return path

    export(PrefillWrapper(model), "moss_tts_prefill.onnx",
           (prefill_ids, prefill_mask), ["input_ids", "attention_mask"], present_names, prefill_axes)
    export(DecodeStepWrapper(model), "moss_tts_decode_step.onnx",
           (decode_ids, torch.full((1,), sample_past_len, dtype=torch.int32), *past),
           decode_inputs, present_names, decode_axes)
    export(LocalDecoderWrapper(model), "moss_tts_local_decoder.onnx",
           (torch.zeros((1, hidden), dtype=torch.float32),
            torch.zeros((1,), dtype=torch.int32),
            torch.full((1, config.n_vq - 1), int(config.audio_pad_token_id), dtype=torch.int32)),
           ["global_hidden", "text_token_id", "audio_prefix_token_ids"],
           ["text_logits", "audio_logits"], {})
    export(LocalCachedStepWrapper(model), "moss_tts_local_cached_step.onnx",
           (torch.zeros((1, hidden), dtype=torch.float32),
            torch.zeros((1,), dtype=torch.int32), torch.zeros((1,), dtype=torch.int32),
            torch.zeros((1,), dtype=torch.int32), torch.zeros((1,), dtype=torch.int32),
            torch.full((1,), sample_past_len, dtype=torch.int32), *local_past),
           local_cached_inputs, local_cached_outputs, local_cached_axes)
    export(LocalFixedSampledFrameWrapper(model), "moss_tts_local_fixed_sampled_frame.onnx",
           (torch.zeros((1, hidden), dtype=torch.float32),
            torch.zeros((1, config.n_vq, config.audio_codebook_sizes[0]), dtype=torch.int32),
            torch.full((1,), 0.5, dtype=torch.float32),
            torch.full((1, config.n_vq), 0.5, dtype=torch.float32)),
           ["global_hidden", "repetition_seen_mask", "assistant_random_u", "audio_random_u"],
           ["should_continue", "frame_token_ids"], fixed_axes)

    files = {
        "prefill": "moss_tts_prefill.onnx",
        "decode_step": "moss_tts_decode_step.onnx",
        "local_decoder": "moss_tts_local_decoder.onnx",
        "local_cached_step": "moss_tts_local_cached_step.onnx",
        "local_fixed_sampled_frame": "moss_tts_local_fixed_sampled_frame.onnx",
    }
    external_data_files: Dict[str, List[str]] = {}
    if external_data:
        for name in files.values():
            externalize_onnx_file(output_dir / name)
        merge_shared_external_data(
            [output_dir / files["prefill"], output_dir / files["decode_step"]],
            output_dir / "moss_tts_global_shared.data",
        )
        merge_shared_external_data(
            [output_dir / files["local_decoder"], output_dir / files["local_cached_step"],
             output_dir / files["local_fixed_sampled_frame"]],
            output_dir / "moss_tts_local_shared.data",
        )
        for key in ("prefill", "decode_step"):
            external_data_files[files[key]] = ["moss_tts_global_shared.data"]
        for key in ("local_decoder", "local_cached_step", "local_fixed_sampled_frame"):
            external_data_files[files[key]] = ["moss_tts_local_shared.data"]
        for name in files.values():
            stale = (output_dir / name).with_suffix(".data")
            if stale.exists():
                stale.unlink()

    metadata = {
        "format_version": 1,
        "checkpoint_path": checkpoint_path,
        "files": files,
        "external_data_files": external_data_files,
        "model_config": {
            "n_vq": config.n_vq,
            "row_width": row_width,
            "hidden_size": hidden,
            "global_layers": n_layer,
            "global_heads": n_head,
            "head_dim": head_dim,
            "local_layers": local_layers,
            "local_heads": local_heads,
            "local_head_dim": local_head_dim,
            "vocab_size": int(config.gpt2.vocab_size),
            "audio_codebook_sizes": list(config.audio_codebook_sizes),
            "audio_pad_token_id": config.audio_pad_token_id,
            "pad_token_id": config.pad_token_id,
            "im_start_token_id": config.im_start_token_id,
            "im_end_token_id": config.im_end_token_id,
            "audio_start_token_id": config.audio_start_token_id,
            "audio_end_token_id": config.audio_end_token_id,
            "audio_user_slot_token_id": config.audio_user_slot_token_id,
            "audio_assistant_slot_token_id": config.audio_assistant_slot_token_id,
        },
        "onnx": {
            "opset": int(opset),
            "prefill_output_names": present_names,
            "decode_input_names": decode_inputs,
            "decode_output_names": present_names,
            "local_cached_input_names": local_cached_inputs,
            "local_cached_output_names": local_cached_outputs,
            "fixed_sampled_frame_constants": {
                "text_temperature": FIXED_SAMPLED_TEXT_TEMPERATURE,
                "text_top_p": FIXED_SAMPLED_TEXT_TOP_P,
                "text_top_k": FIXED_SAMPLED_TEXT_TOP_K,
                "audio_temperature": FIXED_SAMPLED_AUDIO_TEMPERATURE,
                "audio_top_p": FIXED_SAMPLED_AUDIO_TOP_P,
                "audio_top_k": FIXED_SAMPLED_AUDIO_TOP_K,
                "audio_repetition_penalty": FIXED_SAMPLED_AUDIO_REPETITION_PENALTY,
            },
        },
    }
    meta_path = output_dir / "tts_browser_onnx_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


def load_model_for_export(
    checkpoint: Path, config_path: Optional[Path] = None
) -> Tuple[MossTTSNano, MossTTSNanoConfig]:
    """Build a model from a Lightning ``.ckpt`` or an upstream checkpoint directory."""
    from phoonnx_train.mosstts.warmstart import warm_start

    checkpoint = Path(checkpoint)
    if config_path is not None:
        config = MossTTSNanoConfig.from_json_file(config_path)
    elif checkpoint.is_dir():
        config = MossTTSNanoConfig.from_pretrained_dir(checkpoint)
    else:
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        hparams = payload.get("hyper_parameters") or {}
        if "config" not in hparams:
            raise ValueError(
                f"{checkpoint} has no `config` hyper-parameter; pass --config explicitly"
            )
        config = MossTTSNanoConfig.from_dict(hparams["config"])
    model = MossTTSNano(config)
    report = warm_start(model, checkpoint)
    if report.missing:
        raise RuntimeError(
            f"refusing to export a partially loaded model: {len(report.missing)} parameters "
            f"missing from {checkpoint} ({report.summary()})"
        )
    return model, config


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export MOSS-TTS-Nano to phoonnx's ONNX layout.")
    parser.add_argument("--checkpoint", required=True, help="Lightning .ckpt or upstream checkpoint dir")
    parser.add_argument("--config", default=None, help="config JSON override")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--sample-seq-len", type=int, default=24)
    parser.add_argument("--sample-past-len", type=int, default=24)
    parser.add_argument("--external-data", action="store_true",
                        help="store weights in shared .data blobs (upstream layout)")
    parser.add_argument("--tokenizer-model", default=None, help="tokenizer.model to copy next to the graphs")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model, _config = load_model_for_export(
        Path(args.checkpoint), Path(args.config) if args.config else None
    )
    meta_path = export_moss_tts_onnx(
        model,
        Path(args.output_dir),
        opset=args.opset,
        sample_seq_len=args.sample_seq_len,
        sample_past_len=args.sample_past_len,
        external_data=args.external_data,
        checkpoint_path=str(args.checkpoint),
    )
    if args.tokenizer_model:
        import shutil

        shutil.copy2(args.tokenizer_model, Path(args.output_dir) / "tokenizer.model")
    print(f"export complete: {meta_path.parent}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "export_moss_tts_onnx",
    "load_model_for_export",
    "PrefillWrapper",
    "DecodeStepWrapper",
    "LocalDecoderWrapper",
    "LocalCachedStepWrapper",
    "LocalFixedSampledFrameWrapper",
]
