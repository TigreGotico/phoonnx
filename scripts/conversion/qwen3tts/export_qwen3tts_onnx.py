#!/usr/bin/env python3
"""Export Qwen3-TTS-12Hz CustomVoice to the seven ONNX graphs phoonnx loads.

Usage::

    pip install qwen-tts torch onnx onnxruntime onnxscript
    python export_qwen3tts_onnx.py --out ./onnx

Default source model: ``Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`` (Apache-2.0).

Graphs written
--------------

``talker.onnx``
    The 28-layer talker. One step per 80 ms audio frame; reads summed text and
    codec embeddings, writes the logits of code group 0 and the hidden state the
    code predictor conditions on.
``text_embed.onnx`` / ``codec_embed.onnx``
    The two embedding tables the prompt is built from. The text table is the
    largest single file: 151936 rows of 2048, projected down to 1024.
``code_predictor_prefill.onnx`` / ``code_predictor_step.onnx``
    The 5-layer code predictor that writes code groups 1..15 of a frame. Each
    group has its own embedding table and its own output head, so the step graph
    takes the group index as an input and gathers both from a stacked weight.
``sub_codec_embed.onnx``
    A flat gather over those stacked per-group embedding tables.
``codec_decoder.onnx``
    The 12.5 Hz codec decoder: 16 code groups per frame to 24 kHz audio.

Three things in the upstream model do not trace
-----------------------------------------------

1. **mRoPE.** ``apply_interleaved_rope`` assigns into strided slices, which bakes
   the traced sequence length into the graph. phoonnx runs batch 1 with no left
   padding, so the three mRoPE position rows are always equal, and when they are
   the whole thing reduces to plain RoPE. The script checks that before it swaps
   the implementation.
2. **Mask builders.** ``create_causal_mask`` and its sliding-window sibling use
   ``torch.vmap`` and reshape ``cache_position`` with the traced length baked in.
   Every mask here is built with plain tensor ops and checked against the
   transformers one first.
3. **The codec decoder's transposed convolutions** trim with a computed stop
   index. Trimming from the right instead keeps the output length dynamic; the
   decoder is exported with the dynamo exporter, which follows the length.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models import modeling_qwen3_tts as MOD
from qwen_tts.core.tokenizer_12hz import modeling_qwen3_tts_tokenizer_v2 as TOK

OPSET = 17


def past_names(n):
    return sum(([f"past_key_values.{i}.key", f"past_key_values.{i}.value"]
                for i in range(n)), [])


def present_names(n):
    return sum(([f"present.{i}.key", f"present.{i}.value"] for i in range(n)), [])


def make_cache(flat):
    cache = DynamicCache(config=None)
    for i in range(len(flat) // 2):
        cache.update(flat[2 * i], flat[2 * i + 1], i)
    return cache


def flatten_cache(cache, n):
    return sum(([cache.layers[i].keys, cache.layers[i].values] for i in range(n)), [])


def causal_mask(query_positions, total, dtype):
    """Additive mask where every query attends to every earlier key."""
    kv = torch.arange(total).view(1, -1)
    mask = torch.zeros(1, 1, query_positions.numel(), total, dtype=dtype)
    return mask.masked_fill(kv > query_positions.view(-1, 1), torch.finfo(dtype).min)


def export(module, args, path, input_names, output_names, dynamic_axes):
    module.eval()
    with torch.inference_mode():
        torch.onnx.export(module, args, path, input_names=input_names,
                          output_names=output_names, dynamic_axes=dynamic_axes,
                          opset_version=OPSET, do_constant_folding=True, dynamo=False)
    print(f"wrote {path} {os.path.getsize(path) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    ap.add_argument("--out", default="./onnx")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = args.out
    os.makedirs(out, exist_ok=True)

    wrapper = Qwen3TTSModel.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu", attn_implementation="eager")
    model = wrapper.model
    talker = model.talker
    cfg = model.config.talker_config
    predictor = talker.code_predictor
    pcfg = cfg.code_predictor_config
    decoder = model.speech_tokenizer.model.decoder

    n_talker = cfg.num_hidden_layers
    kv_talker, hd_talker = cfg.num_key_value_heads, cfg.head_dim
    n_pred = pcfg.num_hidden_layers
    kv_pred, hd_pred = pcfg.num_key_value_heads, pcfg.head_dim
    groups = cfg.num_code_groups
    vocab_pred = pcfg.vocab_size

    assert isinstance(predictor.small_to_mtp_projection, nn.Identity), (
        "the sub-talker projection is no longer the identity; sub_codec_embed "
        "would have to apply it")

    # ---- 1. mRoPE reduces to plain RoPE at batch 1 -----------------------
    original_rope = MOD.apply_multimodal_rotary_pos_emb

    def plain_rope(q, k, cos, sin, mrope_section, mrope_interleaved=False, unsqueeze_dim=1):
        c, s = cos[0].unsqueeze(unsqueeze_dim), sin[0].unsqueeze(unsqueeze_dim)
        return ((q * c) + (MOD.rotate_half(q) * s),
                (k * c) + (MOD.rotate_half(k) * s))

    for length in (1, 7, 33):
        pos = torch.arange(length).view(1, 1, length).expand(3, 1, length)
        cos, sin = talker.model.rotary_emb(torch.zeros(1, length, cfg.hidden_size), pos)
        q = torch.randn(1, cfg.num_attention_heads, length, hd_talker)
        k = torch.randn(1, kv_talker, length, hd_talker)
        a = original_rope(q, k, cos, sin, cfg.rope_scaling["mrope_section"], True)
        b = plain_rope(q, k, cos, sin, cfg.rope_scaling["mrope_section"], True)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), \
            f"mRoPE differs from plain RoPE at length {length}"
    print("mRoPE equals plain RoPE for equal position rows: ok")
    MOD.apply_multimodal_rotary_pos_emb = plain_rope

    # ---- 2. embeddings ---------------------------------------------------
    class TextEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = talker.model.text_embedding
            self.projection = talker.text_projection

        def forward(self, input_ids):
            return self.projection(self.embedding(input_ids))

    class CodecEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = talker.model.codec_embedding

        def forward(self, input_ids):
            return self.embedding(input_ids)

    export(TextEmbed(), (torch.zeros(1, 5, dtype=torch.long),),
           f"{out}/text_embed.onnx", ["input_ids"], ["hidden"],
           {"input_ids": {1: "T"}, "hidden": {1: "T"}})
    export(CodecEmbed(), (torch.zeros(1, 5, dtype=torch.long),),
           f"{out}/codec_embed.onnx", ["input_ids"], ["hidden"],
           {"input_ids": {1: "T"}, "hidden": {1: "T"}})

    class SubCodecEmbed(nn.Module):
        """Gather over the stacked per-group tables: table i holds code group i+1."""

        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.stack(
                [e.weight.detach() for e in predictor.model.codec_embedding]
            ).reshape(-1, cfg.hidden_size))

        def forward(self, input_ids, tables):
            return self.weight[tables * vocab_pred + input_ids].unsqueeze(0)

    sub_embed = SubCodecEmbed().eval()
    with torch.inference_mode():
        ids = torch.randint(0, vocab_pred, (groups - 1,))
        ref = torch.cat([predictor.model.codec_embedding[i](ids[i].view(1, 1))
                         for i in range(groups - 1)], dim=1)
        assert torch.equal(sub_embed(ids, torch.arange(groups - 1)), ref)
    export(sub_embed, (ids, torch.arange(groups - 1)), f"{out}/sub_codec_embed.onnx",
           ["input_ids", "tables"], ["hidden"],
           {"input_ids": {0: "G"}, "tables": {0: "G"}, "hidden": {1: "G"}})

    # ---- 3. talker -------------------------------------------------------
    class Talker(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = talker.model
            self.head = talker.codec_head

        def forward(self, inputs_embeds, position_ids, *past):
            cache = make_cache(list(past))
            positions = position_ids[0]
            total = past[0].shape[2] + inputs_embeds.shape[1]
            mask = causal_mask(positions, total, inputs_embeds.dtype)
            hidden = inputs_embeds
            embeddings = self.model.rotary_emb(
                hidden, position_ids.unsqueeze(0).expand(3, -1, -1))
            for layer in self.model.layers:
                hidden = layer(hidden, attention_mask=mask,
                               position_ids=positions.view(1, -1), past_key_values=cache,
                               use_cache=True, cache_position=positions,
                               position_embeddings=embeddings)[0]
            hidden = self.model.norm(hidden)
            return ((self.head(hidden[:, -1:]), hidden[:, -1:])
                    + tuple(flatten_cache(cache, n_talker)))

    talker_module = Talker().eval()
    with torch.inference_mode():
        for width, cached in ((7, 0), (1, 11)):
            embeds = torch.randn(1, width, cfg.hidden_size)
            positions = torch.arange(cached, cached + width).unsqueeze(0)
            zeros = [torch.zeros(1, kv_talker, cached, hd_talker)
                     for _ in range(2 * n_talker)]
            mine = talker_module(embeds, positions, *zeros)[1]
            stock = talker.model(
                inputs_embeds=embeds,
                position_ids=positions.unsqueeze(0).expand(3, -1, -1),
                attention_mask=None, past_key_values=make_cache(zeros),
                use_cache=True, cache_position=positions[0]
            ).last_hidden_state[:, -1:]
            assert torch.equal(mine, stock), "talker mask differs from the stock model"
    print("talker mask matches the stock model: ok")

    width, cached = 7, 3
    zeros = [torch.zeros(1, kv_talker, cached, hd_talker) for _ in range(2 * n_talker)]
    axes = {"inputs_embeds": {1: "S"}, "position_ids": {1: "S"}}
    axes.update({n: {2: "P"} for n in past_names(n_talker)})
    axes.update({n: {2: "P_S"} for n in present_names(n_talker)})
    export(talker_module,
           (torch.randn(1, width, cfg.hidden_size),
            torch.arange(cached, cached + width).unsqueeze(0), *zeros),
           f"{out}/talker.onnx",
           ["inputs_embeds", "position_ids"] + past_names(n_talker),
           ["logits", "last_hidden"] + present_names(n_talker), axes)

    # ---- 4. code predictor ----------------------------------------------
    class PredictorPrefill(nn.Module):
        """Two input positions, output head 0: gives code group 1."""

        def __init__(self):
            super().__init__()
            self.model = predictor.model
            self.projection = predictor.small_to_mtp_projection
            self.head = predictor.lm_head[0]

        def forward(self, inputs_embeds):
            embeds = self.projection(inputs_embeds)
            cache = DynamicCache(config=None)
            positions = torch.arange(embeds.shape[1])
            result = self.model(inputs_embeds=embeds, attention_mask=None,
                                past_key_values=cache, use_cache=True,
                                cache_position=positions,
                                position_ids=positions.unsqueeze(0))
            logits = self.head(result.last_hidden_state[:, -1:])[:, 0]
            return (logits,) + tuple(flatten_cache(result.past_key_values, n_pred))

    axes = {n: {2: "P"} for n in present_names(n_pred)}
    export(PredictorPrefill(), (torch.randn(1, 2, cfg.hidden_size),),
           f"{out}/code_predictor_prefill.onnx", ["inputs_embeds"],
           ["logits"] + present_names(n_pred), axes)

    class PredictorStep(nn.Module):
        """One later group: the group index picks the output head."""

        def __init__(self):
            super().__init__()
            self.model = predictor.model
            self.register_buffer("head", torch.stack(
                [h.weight.detach() for h in predictor.lm_head]))

        def forward(self, inputs_embeds, step, position_ids, *past):
            cache = make_cache(list(past))
            positions = position_ids[0]
            total = past[0].shape[2] + inputs_embeds.shape[1]
            mask = causal_mask(positions, total, inputs_embeds.dtype)
            hidden = inputs_embeds
            embeddings = self.model.rotary_emb(hidden, position_ids)
            for layer in self.model.layers:
                hidden = layer(hidden, attention_mask=mask, position_ids=position_ids,
                               past_key_values=cache, use_cache=True,
                               cache_position=positions,
                               position_embeddings=embeddings)[0]
            hidden = self.model.norm(hidden)[:, -1:]
            logits = torch.matmul(hidden, self.head[step].transpose(0, 1))[:, 0]
            return (logits,) + tuple(flatten_cache(cache, n_pred))

    step_module = PredictorStep().eval()
    with torch.inference_mode():
        for cached in (2, 6):
            embeds = torch.randn(1, 1, pcfg.hidden_size)
            positions = torch.tensor([[cached]])
            zeros = [torch.zeros(1, kv_pred, cached, hd_pred) for _ in range(2 * n_pred)]
            mine = step_module(embeds, torch.tensor(1), positions, *zeros)[0]
            stock_hidden = predictor.model(
                inputs_embeds=embeds, attention_mask=None,
                past_key_values=make_cache(zeros), use_cache=True,
                cache_position=positions[0], position_ids=positions
            ).last_hidden_state[:, -1:]
            stock = torch.matmul(stock_hidden,
                                 predictor.lm_head[1].weight.transpose(0, 1))[:, 0]
            assert torch.equal(mine, stock), "code-predictor mask differs from stock"
    print("code-predictor mask matches the stock model: ok")

    zeros = [torch.zeros(1, kv_pred, 2, hd_pred) for _ in range(2 * n_pred)]
    axes = {n: {2: "P"} for n in past_names(n_pred)}
    axes.update({n: {2: "P1"} for n in present_names(n_pred)})
    export(step_module,
           (torch.randn(1, 1, pcfg.hidden_size), torch.tensor(1),
            torch.tensor([[2]]), *zeros),
           f"{out}/code_predictor_step.onnx",
           ["inputs_embeds", "step", "position_ids"] + past_names(n_pred),
           ["logits"] + present_names(n_pred), axes)

    # ---- 5. codec decoder ------------------------------------------------
    pre = decoder.pre_transformer
    pre.config._attn_implementation = "eager"
    window = pre.config.sliding_window

    def sliding_mask(positions, dtype):
        q, kv = positions.unsqueeze(1), positions.unsqueeze(0)
        allowed = (kv <= q) & (kv > q - window)
        return torch.zeros_like(allowed, dtype=dtype).masked_fill(
            ~allowed, torch.finfo(dtype).min).unsqueeze(0).unsqueeze(0)

    from transformers.masking_utils import create_sliding_window_causal_mask
    for length in (5, 40, 300):
        stock = create_sliding_window_causal_mask(
            config=pre.config, input_embeds=torch.zeros(1, length, pre.config.hidden_size),
            attention_mask=None, cache_position=torch.arange(length),
            past_key_values=None, position_ids=torch.arange(length).unsqueeze(0))
        mine = sliding_mask(torch.arange(length), torch.float32)
        assert torch.equal(stock == 0, mine == 0), \
            f"sliding-window mask differs at length {length}"
    print("sliding-window mask matches transformers: ok")

    def trim_from_the_right(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        if self.right_pad > 0:
            hidden_state = hidden_state[..., : -self.right_pad]
        return hidden_state.contiguous()

    def upstream_trim(hidden_state, right_pad):
        """Upstream indexes with an explicit shape subtraction; ``[..., :-right_pad]``
        is the dynamo/onnx-export-friendly equivalent used above."""
        if right_pad > 0:
            hidden_state = hidden_state[..., : hidden_state.shape[-1] - right_pad]
        return hidden_state.contiguous()

    class _StubConv:
        def __init__(self, right_pad):
            self.right_pad = right_pad
            self.conv = lambda x: x

    for right_pad, length in ((0, 4), (1, 4), (3, 9), (5, 40), (5, 137)):
        conv_out = torch.randn(1, 4, length + right_pad)
        mine = trim_from_the_right(_StubConv(right_pad), conv_out)
        stock = upstream_trim(conv_out, right_pad)
        assert torch.equal(mine, stock), \
            f"transposed-conv trim differs from upstream at right_pad={right_pad}, length={length}"
    print("transposed-conv trim matches upstream slicing: ok")

    TOK.Qwen3TTSTokenizerV2CausalTransConvNet.forward = trim_from_the_right

    class CodecDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = decoder

        def forward(self, codes):
            hidden = self.decoder.quantizer.decode(codes)
            hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            mask = sliding_mask(positions, hidden.dtype)
            hidden = self.decoder.pre_transformer(
                inputs_embeds=hidden,
                attention_mask={"sliding_attention": mask, "full_attention": mask},
                position_ids=positions.unsqueeze(0), cache_position=positions,
                use_cache=False).last_hidden_state
            hidden = hidden.permute(0, 2, 1)
            for blocks in self.decoder.upsample:
                for block in blocks:
                    hidden = block(hidden)
            wav = hidden
            for block in self.decoder.decoder:
                wav = block(wav)
            return wav.clamp(min=-1, max=1)

    frames = torch.export.Dim("T", min=4, max=4096)
    with torch.inference_mode():
        program = torch.onnx.export(
            CodecDecoder().eval(), (torch.randint(0, 2000, (1, groups, 40)),),
            dynamo=True, opset_version=18, input_names=["codes"], output_names=["wav"],
            dynamic_shapes={"codes": {2: frames}})
        program.optimize()
        program.save(f"{out}/codec_decoder.onnx")
    print(f"wrote {out}/codec_decoder.onnx "
          f"{os.path.getsize(f'{out}/codec_decoder.onnx') / 1e6:.1f} MB")

    import onnxruntime as ort
    session = ort.InferenceSession(f"{out}/codec_decoder.onnx",
                                   providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        for length in (40, 137):
            codes = torch.randint(0, 2000, (1, groups, length))
            stock = decoder(codes).numpy()
            got = session.run(None, {"codes": codes.numpy()})[0]
            assert got.shape == stock.shape, "codec decoder length is not dynamic"
            max_diff = np.abs(got - stock).max()
            print(f"codec decoder T={length}: max abs diff {max_diff:.3e}")
            assert max_diff < 1e-5, \
                f"codec decoder ONNX output diverges from stock at T={length}: {max_diff:.3e}"

    # ---- 6. tokenizer ----------------------------------------------------
    wrapper.processor.tokenizer.backend_tokenizer.save(f"{out}/tokenizer.json")
    print(f"wrote {out}/tokenizer.json")


if __name__ == "__main__":
    main()
