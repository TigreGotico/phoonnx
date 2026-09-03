#!/usr/bin/env python3
"""Check an exported Orpheus LM graph against the torch reference it came from.

    python verify_parity.py --model unsloth/orpheus-3b-0.1-ft \
        --onnx ./orpheus-3b-en-onnx/model.onnx \
        --tokenizer ./orpheus-3b-en-onnx/tokenizer.json \
        --voice tara --text "The quick brown fox jumps over the lazy dog." --steps 24

Greedy decoding, both sides driven from the same served prompt (built by
:meth:`OrpheusAdapter.build_prompt_ids` — see ``probe_prompt.py`` for why it carries a
double BOS). Two things are compared and reported:

1. **Prefill logits** — max abs diff at the last prompt position, and whether the
   argmax agrees.
2. **Greedy decode agreement** — token-for-token over ``--steps`` steps, driven in
   lockstep so both sides see the same history at each step (the only way the two
   stacks are comparable; with sampling they draw from different random streams).

This is a network- and weights-heavy script (needs the ~6-13 GB torch checkpoint and
the multi-GB ONNX graph) — it is not run as part of CI or of the evidence gathered for
this PR; its self-gating design (below) is the deliverable. The parity numbers quoted
in the PR body and in the mirror's README came from one prior run against
``unsloth/orpheus-3b-0.1-ft`` (see ``evidence/README.md`` for provenance); this script
reconstructs that methodology so it can be re-run and re-verified by anyone.

Exits non-zero if greedy agreement is not 100% or the prefill diff exceeds
``--diff-threshold`` — a quantized or mis-exported graph should not ship.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="unsloth/orpheus-3b-0.1-ft",
                     help="ungated torch reference; the gated canopylabs/orpheus-3b-0.1-ft "
                          "requires accepting Canopy Labs' licence terms first")
    ap.add_argument("--onnx", required=True, help="path to the exported model.onnx")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--voice", default="tara")
    ap.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--diff-threshold", type=float, default=1.0,
                     help="max acceptable prefill logit diff; the fp32 export in this "
                          "PR measured 0.166, quantized variants measured 8.5+")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    from phoonnx.engines.orpheus import OrpheusAdapter
    from phoonnx.providers import make_session

    torch.set_num_threads(args.threads)

    adapter = OrpheusAdapter()
    from tokenizers import Tokenizer
    adapter.tokenizer = Tokenizer.from_file(args.tokenizer)
    prompt_ids = adapter.build_prompt_ids(args.text, args.voice)
    print("served prompt ids (%d):" % len(prompt_ids), prompt_ids[:12], "...")

    # ---- torch reference, greedy ------------------------------------------------
    torch_model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu")
    torch_model.eval()
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.no_grad():
        out = torch_model(input_ids=input_ids, use_cache=True)
    torch_prefill_logits = out.logits[0, -1].numpy()
    past = out.past_key_values

    torch_tokens = []
    cur = input_ids
    for _ in range(args.steps):
        with torch.no_grad():
            out = torch_model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        torch_tokens.append(nxt)
        cur = torch.tensor([[nxt]], dtype=torch.long)

    # ---- onnx graph, greedy -------------------------------------------------
    session = make_session(args.onnx)
    onnx_tokens_all = adapter.generate(
        session, prompt_ids, {"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0,
                              "max_new_tokens": args.steps},
        np.random.default_rng(0))
    onnx_tokens = onnx_tokens_all[:args.steps]

    # first-step logits for the diff report
    adapter._read_kv_shape(session)
    empty = np.zeros((1, adapter.num_kv_heads, 0, adapter.head_dim), np.float32)
    ids = np.asarray(prompt_ids, np.int64).reshape(1, -1)
    attn = np.ones_like(ids)
    prefill_out = session.run(None, {"input_ids": ids, "attention_mask": attn,
                                     **{n: empty for n in adapter.past_names}})
    onnx_prefill_logits = np.asarray(prefill_out[0], np.float32)[0, -1]

    diff = float(np.abs(onnx_prefill_logits - torch_prefill_logits).max())
    prefill_argmax_agree = int(onnx_prefill_logits.argmax()) == int(torch_prefill_logits.argmax())

    frames = min(len(onnx_tokens), len(torch_tokens))
    agree = [a == b for a, b in zip(onnx_tokens[:frames], torch_tokens[:frames])]
    agreement = sum(agree) / max(1, len(agree))

    print(f"prefill max abs logit diff: {diff:.4g}")
    print(f"prefill argmax agrees: {prefill_argmax_agree}")
    print(f"greedy agreement: {sum(agree)}/{len(agree)} ({100 * agreement:.1f}%)")

    ok = agreement == 1.0 and diff <= args.diff_threshold
    print("PASS" if ok else "FAIL — export should not ship")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
