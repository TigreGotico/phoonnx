#!/usr/bin/env python3
"""Compare the exported Llasa ONNX decoder against the torch reference.

Reports, for a real Llasa prompt:

* prefill logit difference (max absolute / mean absolute) on the last position;
* per-step decode logit difference over a greedy run;
* greedy token agreement between torch ``generate(do_sample=False)`` and the
  ONNX KV-cache loop.

Usage::

    python parity_llm.py --model HKUSTAudio/Llasa-1B --onnx out/llasa-1b/model.onnx
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch


PROMPT_TEXTS = [
    "Dealing with family secrets is never easy.",
    "突然，身边一阵笑声。",
]


def build_prompt(tok, text: str) -> np.ndarray:
    chat = [
        {"role": "user",
         "content": "Convert the text to speech:<|TEXT_UNDERSTANDING_START|>"
                    + text + "<|TEXT_UNDERSTANDING_END|>"},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"},
    ]
    return tok.apply_chat_template(chat, tokenize=True, return_tensors="pt",
                                  continue_final_message=True,
                                  return_dict=True)["input_ids"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HKUSTAudio/Llasa-1B")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()

    import onnxruntime
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    tok = AutoTokenizer.from_pretrained(args.model)
    ref = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager").eval()
    sess = onnxruntime.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    past_names = [i.name for i in sess.get_inputs() if i.name.startswith("past_key_values")]
    n_layers = len(past_names) // 2
    cfg = ref.config
    kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    end_id = tok.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

    report = []
    for text in PROMPT_TEXTS:
        ids = build_prompt(tok, text)
        n = ids.shape[1]

        # ---- torch greedy reference -----------------------------------
        with torch.no_grad():
            out = ref.generate(ids, max_length=n + args.steps,
                               eos_token_id=end_id, do_sample=False)
        torch_tokens = out[0][n:].tolist()

        # ---- torch per-step logits ------------------------------------
        torch_logits = []
        with torch.no_grad():
            cache = DynamicCache()
            cur = ids
            pos = torch.arange(n)[None]
            mask = torch.ones((1, n), dtype=torch.long)
            for step in range(len(torch_tokens)):
                o = ref(input_ids=cur, attention_mask=mask, position_ids=pos,
                        past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                torch_logits.append(o.logits[0, -1].numpy().copy())
                cur = torch.tensor([[torch_tokens[step]]])
                pos = torch.tensor([[n + step]])
                mask = torch.cat([mask, torch.ones((1, 1), dtype=torch.long)], 1)

        # ---- onnx greedy loop -----------------------------------------
        onnx_tokens = []
        onnx_logits = []
        past = {k: np.zeros((1, kv_heads, 0, head_dim), np.float32) for k in past_names}
        cur_ids = ids.numpy().astype(np.int64)
        pos_ids = np.arange(n, dtype=np.int64)[None]
        att = np.ones((1, n), np.int64)
        for step in range(args.steps):
            feed = dict(input_ids=cur_ids, attention_mask=att, position_ids=pos_ids, **past)
            outs = sess.run(None, feed)
            logits = outs[0][0, -1]
            onnx_logits.append(logits)
            nxt = int(np.argmax(logits))
            onnx_tokens.append(nxt)
            for j, name in enumerate(past_names):
                past[name] = outs[1 + j]
            if nxt == end_id:
                break
            cur_ids = np.array([[nxt]], np.int64)
            pos_ids = np.array([[n + step]], np.int64)
            att = np.concatenate([att, np.ones((1, 1), np.int64)], 1)

        k = min(len(torch_logits), len(onnx_logits))
        diffs = [float(np.abs(torch_logits[i] - onnx_logits[i]).max()) for i in range(k)]
        means = [float(np.abs(torch_logits[i] - onnx_logits[i]).mean()) for i in range(k)]
        agree = sum(1 for i in range(min(len(torch_tokens), len(onnx_tokens)))
                    if torch_tokens[i] == onnx_tokens[i])
        report.append({
            "text": text,
            "prompt_tokens": n,
            "steps_compared": k,
            "prefill_logit_max_abs_diff": diffs[0],
            "prefill_logit_mean_abs_diff": means[0],
            "decode_logit_max_abs_diff": max(diffs[1:]) if k > 1 else None,
            "decode_logit_mean_abs_diff": float(np.mean(means[1:])) if k > 1 else None,
            "torch_tokens": len(torch_tokens),
            "onnx_tokens": len(onnx_tokens),
            "greedy_agreement": f"{agree}/{min(len(torch_tokens), len(onnx_tokens))}",
        })
        print(json.dumps(report[-1], indent=2), flush=True)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
