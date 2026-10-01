#!/usr/bin/env python3
"""Export a Llasa LLaMA backbone to a single KV-cached ONNX decoder graph.

Llasa (HKUSTAudio) is ``LlamaForCausalLM`` with the vocabulary extended by the
65,536 XCodec2 speech tokens (``<|s_0|>`` .. ``<|s_65535|>``).  Generation is a
plain autoregressive loop, so one ONNX graph serves both the prefill step
(``sequence_length`` > 1, empty past) and every decode step
(``sequence_length`` == 1, non-empty past).

Graph signature::

    inputs : input_ids        [batch, sequence_length]  int64
             attention_mask   [batch, past_length + sequence_length] int64
             position_ids     [batch, sequence_length]  int64
             past_key_values.<i>.key    [batch, kv_heads, past_length, head_dim] float32
             past_key_values.<i>.value  [batch, kv_heads, past_length, head_dim] float32
    outputs: logits           [batch, 1, vocab]  float32   (last position only)
             present.<i>.key   / present.<i>.value

Only the last position's logits are returned: the vocabulary is 193,800 wide, so
returning the full prefill logits would cost ~78 MB per 100 prompt tokens for a
value the sampler never looks at.

Usage::

    python export_llm.py --model HKUSTAudio/Llasa-1B --output out/llasa-1b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


class LlasaOnnxWrapper(nn.Module):
    """Flattens the ``DynamicCache`` API into plain tensor inputs/outputs."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.num_layers = model.config.num_hidden_layers

    def forward(self, input_ids, attention_mask, position_ids, *past):
        from transformers import DynamicCache

        layers = [(past[2 * i], past[2 * i + 1]) for i in range(self.num_layers)]
        cache = DynamicCache(layers)
        # Call the decoder stack directly and project only the last position:
        # the vocabulary is 193,800 wide, so running ``lm_head`` over a whole
        # prefill would cost a 193,800-column matmul per prompt token for a
        # value the sampler never reads.
        out = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        logits = self.model.lm_head(out.last_hidden_state[:, -1:, :])
        present = []
        for layer in out.past_key_values.layers:
            present.extend([layer.keys, layer.values])
        return (logits, *present)


def io_names(num_layers: int):
    past = []
    present = []
    for i in range(num_layers):
        past += [f"past_key_values.{i}.key", f"past_key_values.{i}.value"]
        present += [f"present.{i}.key", f"present.{i}.value"]
    return past, present


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HKUSTAudio/Llasa-1B")
    ap.add_argument("--output", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    cfg = model.config
    kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    wrapper = LlasaOnnxWrapper(model).eval()

    # A 2-token prefill with a 3-token past exercises both the prefill and the
    # decode shapes, so the traced graph stays valid for either.
    batch, seq, past_len = 1, 2, 3
    input_ids = torch.randint(0, 1000, (batch, seq), dtype=torch.int64)
    attention_mask = torch.ones((batch, past_len + seq), dtype=torch.int64)
    position_ids = torch.arange(past_len, past_len + seq, dtype=torch.int64)[None]
    past = []
    for _ in range(cfg.num_hidden_layers):
        past += [
            torch.randn(batch, kv_heads, past_len, head_dim),
            torch.randn(batch, kv_heads, past_len, head_dim),
        ]

    past_names, present_names = io_names(cfg.num_hidden_layers)
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence_length"},
        "attention_mask": {0: "batch", 1: "total_length"},
        "position_ids": {0: "batch", 1: "sequence_length"},
        "logits": {0: "batch"},
    }
    for n in past_names:
        dynamic_axes[n] = {0: "batch", 2: "past_length"}
    for n in present_names:
        dynamic_axes[n] = {0: "batch", 2: "total_length"}

    onnx_path = out_dir / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask, position_ids, *past),
            str(onnx_path),
            input_names=["input_ids", "attention_mask", "position_ids", *past_names],
            output_names=["logits", *present_names],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    meta = {
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": cfg.vocab_size,
        "speech_token_offset": tok.convert_tokens_to_ids("<|s_0|>"),
        "speech_generation_start": tok.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>"),
        "speech_generation_end": tok.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>"),
        "text_understanding_start": tok.convert_tokens_to_ids("<|TEXT_UNDERSTANDING_START|>"),
        "text_understanding_end": tok.convert_tokens_to_ids("<|TEXT_UNDERSTANDING_END|>"),
        "num_speech_tokens": 65536,
    }
    (out_dir / "llm_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("wrote", onnx_path)


if __name__ == "__main__":
    main()
