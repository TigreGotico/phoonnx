"""Check an ArkTTS ONNX export against the PyTorch checkpoint it came from.

    python verify_parity.py --repo itzune/zortzi-tts \
        --onnx-dir ~/zortzi-tts-onnx --precision fp16 \
        --voice-codes ~/zortzi-tts-onnx/voices/maider/codes.npy \
        --reference-text "Aurrelaria prest dago jokatzeko." \
        --text "Kaixo mundua, gaur eguraldi ona dago." --steps 24

Four things are compared, and all four have to hold before a mirror is published:

1. **Prompt** — the ``[1, 11, T]`` matrix this repo's adapter builds against the one
   upstream's own processor and ``_prepare_prompt`` build. Exact equality, no tolerance.
   Everything downstream is meaningless if the prompts differ.
2. **Slow AR** — logits and hidden states, at prefill and at every decode step, driven in
   lockstep by greedy decoding so both sides see the same history.
3. **Fast AR** — logits for all nine predicted codebooks of every frame, again in lockstep.
4. **Codec decoder** — the waveform for a fixed code matrix.

Greedy decoding is used *only here*. Both model cards say greedy loops forever in real
synthesis; that does not matter for a fixed step budget, and it is the only way to make
the two stacks follow the same path. Agreement rate is the gate that matters — a logit
difference that never changes an argmax is a rounding artefact, and one that does is a
defect no matter how small it looks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from transformers import AutoModel, AutoProcessor

from phoonnx.engines.arktts import (
    FAST_HEAD_DIM, FAST_KV_HEADS, FAST_LAYERS, MAX_SEQ_LEN, NUM_CODEBOOKS,
    SEMANTIC_BEGIN_ID, SLOW_HEAD_DIM, SLOW_KV_HEADS, SLOW_LAYERS, ArkTTSAdapter,
)


def summarize(name: str, diffs: list[float], agree: list[bool],
              gaps: list[float] | None = None) -> dict:
    """Report the differences, the greedy agreement, and how close the misses were.

    ``gaps`` holds, for each disagreement, the reference's own top-1 minus top-2 margin.
    A miss whose margin is under the logit difference is the two stacks splitting a tie
    that precision alone decided; a miss with a wide margin is a real defect. The
    distinction is the whole point of the gate, so it is reported rather than averaged
    away.
    """
    row = {
        "tensor": name,
        "max_abs_diff": float(max(diffs)) if diffs else 0.0,
        "mean_abs_diff": float(np.mean(diffs)) if diffs else 0.0,
        "greedy_agreement": f"{sum(agree)}/{len(agree)}" if agree else "n/a",
    }
    tail = ""
    if gaps:
        row["miss_margins"] = [round(float(g), 5) for g in sorted(gaps)]
        row["max_miss_margin"] = float(max(gaps))
        tail = f"  worst miss margin {max(gaps):.4g}"
    print(f"  {row['tensor']:<24} max {row['max_abs_diff']:.4g}  "
          f"mean {row['mean_abs_diff']:.4g}  greedy {row['greedy_agreement']}{tail}")
    return row


def margin(logits: np.ndarray) -> float:
    """The reference's top-1 minus top-2 logit — how decided this argmax was."""
    top = np.partition(logits, -2)[-2:]
    return float(abs(top[1] - top[0]))


def build_reference_prompt(model, processor, text: str, codes: np.ndarray,
                           reference_text: str) -> torch.Tensor:
    """The prompt upstream builds, through its own processor and packer."""
    batch = processor(text=text, reference_text=reference_text,
                      reference_codes=torch.as_tensor(codes, dtype=torch.long))
    prompt, _ = model._prepare_prompt(**batch)
    return prompt


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--onnx-dir", required=True, type=Path)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--voice-codes", required=True, type=Path, help=".npy or voice JSON")
    parser.add_argument("--reference-text", default="")
    parser.add_argument("--text", required=True)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="tokenizer.json for the adapter's prompt builder")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if args.voice_codes.suffix == ".npy":
        codes = np.load(args.voice_codes).astype(np.int64)
        reference_text = args.reference_text
    else:
        data = json.loads(args.voice_codes.read_text())
        codes = np.asarray(data["codes"], np.int64)
        reference_text = args.reference_text or data["reference_text"]
    if not reference_text:
        raise SystemExit("--reference-text is required with a .npy voice")

    print(f"torch reference: {args.repo}")
    model = AutoModel.from_pretrained(args.repo, dtype=torch.float32, trust_remote_code=True).eval()
    processor = AutoProcessor.from_pretrained(args.repo, trust_remote_code=True)

    # --- 1. prompt -------------------------------------------------------------
    reference_prompt = build_reference_prompt(model, processor, args.text, codes, reference_text)
    adapter = ArkTTSAdapter()
    adapter.tokenizer = __import__("phoonnx.tokenizer", fromlist=["BPETokenizer"]).BPETokenizer(
        str(args.tokenizer or Path(args.onnx_dir) / "tokenizer" / "tokenizer.json"))
    adapter.reference_codes = codes
    adapter.reference_text = reference_text
    suffix = adapter.encode_text(args.text, None, None)[0]
    prompt = adapter.build_prompt(np.asarray(suffix, np.int64))
    prompt_ok = bool(np.array_equal(prompt, reference_prompt.cpu().numpy()))
    print(f"prompt: adapter {prompt.shape} vs upstream {tuple(reference_prompt.shape)} -> "
          f"{'MATCH' if prompt_ok else 'MISMATCH'}")
    if not prompt_ok:
        raise SystemExit("prompt mismatch — nothing below this line is meaningful")

    # --- sessions --------------------------------------------------------------
    slow = onnxruntime.InferenceSession(
        str(args.onnx_dir / f"slow_ar_{args.precision}.onnx"), providers=["CPUExecutionProvider"])
    fast = onnxruntime.InferenceSession(
        str(args.onnx_dir / f"fast_ar_{args.precision}.onnx"), providers=["CPUExecutionProvider"])
    adapter.fast_ar = fast

    slow_names = [o.name for o in slow.get_outputs()]
    fast_names = [o.name for o in fast.get_outputs()]
    cache = adapter.empty_cache(slow, SLOW_LAYERS, MAX_SEQ_LEN, SLOW_KV_HEADS, SLOW_HEAD_DIM)

    model._setup_generation_caches(1, MAX_SEQ_LEN, torch.float32)
    width = prompt.shape[2]
    torch_mask = torch.ones((1, width), dtype=torch.long)
    torch_pos = torch_mask.cumsum(-1).sub(1).clamp_min(0)
    torch_logits, torch_hidden = model._slow_step(
        reference_prompt, torch.arange(width), torch_pos, torch_mask)

    input_pos = np.arange(width, dtype=np.int64)
    outputs = dict(zip(slow_names, slow.run(None, {"codes": prompt, "input_pos": input_pos, **cache})))
    adapter.scatter_cache(cache, outputs, SLOW_LAYERS, input_pos)

    def slice_torch(logits: torch.Tensor) -> np.ndarray:
        semantic = logits[..., SEMANTIC_BEGIN_ID:model.config.semantic_end_id + 1]
        eos = logits[..., model.config.eos_token_id:model.config.eos_token_id + 1]
        return torch.cat((semantic, eos), dim=-1).float().numpy().reshape(-1)

    slow_diffs, slow_agree, hidden_diffs, slow_gaps = [], [], [], []
    fast_diffs, fast_agree, fast_gaps = [], [], []
    rows = []

    print(f"\nlockstep greedy decode, {args.steps} steps")
    for step in range(args.steps):
        reference_logits = slice_torch(torch_logits)
        onnx_logits = np.asarray(outputs["logits"][0, -1], np.float32)
        slow_diffs.append(float(np.abs(reference_logits - onnx_logits).max()))
        slow_agree.append(int(reference_logits.argmax()) == int(onnx_logits.argmax()))
        if not slow_agree[-1]:
            slow_gaps.append(margin(reference_logits))
        hidden_diffs.append(float(np.abs(
            torch_hidden[:, -1:].float().numpy()
            - np.asarray(outputs["slow_hidden"][:, -1:], np.float32)).max()))

        semantic = int(reference_logits.argmax())
        if semantic == model.config.codebook_size:
            print(f"  EOS at step {step}")
            break

        # fast AR, both stacks driven by the same codebook history
        torch_fast_hidden = model.fast_project_in(torch_hidden[:, -1:])
        model._fast_step(torch_fast_hidden, 0)
        current = torch.tensor([semantic], dtype=torch.long)
        codebooks = [semantic]

        fast_cache = adapter.empty_cache(fast, FAST_LAYERS, NUM_CODEBOOKS,
                                         FAST_KV_HEADS, FAST_HEAD_DIM)
        fast_dtype = (np.float16 if fast.get_inputs()[0].type == "tensor(float16)"
                      else np.float32)
        onnx_out = dict(zip(fast_names, fast.run(None, {
            "slow_hidden": np.asarray(outputs["slow_hidden"][:, -1:], fast_dtype),
            "token_id": np.zeros((1, 1), np.int64),
            "use_slow_hidden": np.asarray([True]),
            "input_pos": np.asarray([0], np.int64), **fast_cache})))
        adapter.scatter_cache(fast_cache, onnx_out, FAST_LAYERS, np.asarray([0], np.int64))

        for position in range(1, NUM_CODEBOOKS):
            torch_fast_logits = model._fast_step(
                model.fast_embeddings(current)[:, None], position)
            onnx_out = dict(zip(fast_names, fast.run(None, {
                "slow_hidden": np.zeros((1, 1, model.config.dim), fast_dtype),
                "token_id": np.asarray([[codebooks[-1]]], np.int64),
                "use_slow_hidden": np.asarray([False]),
                "input_pos": np.asarray([position], np.int64), **fast_cache})))
            adapter.scatter_cache(fast_cache, onnx_out, FAST_LAYERS,
                                  np.asarray([position], np.int64))
            reference = torch_fast_logits.float().numpy().reshape(-1)
            produced = np.asarray(onnx_out["logits"][0, -1], np.float32)
            fast_diffs.append(float(np.abs(reference - produced).max()))
            fast_agree.append(int(reference.argmax()) == int(produced.argmax()))
            if not fast_agree[-1]:
                fast_gaps.append(margin(reference))
            current = torch.tensor([int(reference.argmax())], dtype=torch.long)
            codebooks.append(int(reference.argmax()))

        # advance both stacks with the same frame
        column = torch.tensor([[SEMANTIC_BEGIN_ID + semantic] + codebooks],
                              dtype=torch.long).T[None]
        torch_mask = torch.cat((torch_mask, torch.ones((1, 1), dtype=torch.long)), dim=1)
        torch_logits, torch_hidden = model._slow_step(
            column, torch.tensor([width + step]), torch.tensor([[width + step]]), torch_mask)
        input_pos = np.asarray([width + step], np.int64)
        outputs = dict(zip(slow_names, slow.run(
            None, {"codes": column.numpy(), "input_pos": input_pos, **cache})))
        adapter.scatter_cache(cache, outputs, SLOW_LAYERS, input_pos)

    print("\nresults")
    rows.append(summarize("slow_ar logits", slow_diffs, slow_agree, slow_gaps))
    rows.append(summarize("slow_ar hidden", hidden_diffs, []))
    rows.append(summarize("fast_ar logits", fast_diffs, fast_agree, fast_gaps))

    # --- 4. codec decoder ------------------------------------------------------
    decoder_path = args.onnx_dir / "codec_decoder_fp16.onnx"
    if decoder_path.is_file():
        decoder = onnxruntime.InferenceSession(str(decoder_path),
                                               providers=["CPUExecutionProvider"])
        feed = codes.reshape(1, NUM_CODEBOOKS, -1)
        produced = np.asarray(decoder.run(None, {"codes": feed})[0], np.float32).reshape(-1)
        reference = model.decode_audio(
            torch.as_tensor(feed, dtype=torch.long))[0][0].float().numpy()
        length = min(produced.size, reference.size)
        diff = float(np.abs(produced[:length] - reference[:length]).max())
        correlation = float(np.corrcoef(produced[:length], reference[:length])[0, 1])
        print(f"  {'codec decoder':<24} max {diff:.4g}  corr {correlation:.6f}  "
              f"samples onnx {produced.size} torch {reference.size}")
        rows.append({"tensor": "codec_decoder", "max_abs_diff": diff,
                     "correlation": correlation})

    if args.report:
        args.report.write_text(json.dumps(
            {"repo": args.repo, "onnx_dir": str(args.onnx_dir),
             "precision": args.precision, "prompt_match": prompt_ok, "rows": rows},
            indent=2) + "\n")
        print("wrote", args.report)


if __name__ == "__main__":
    main()
