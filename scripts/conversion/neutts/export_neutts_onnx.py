#!/usr/bin/env python3
"""Export a NeuTTS Air / VieNeu-TTS / Akiti-TTS backbone to ONNX.

The backbone is a plain ``Qwen3ForCausalLM`` that autoregressively emits NeuCodec
audio tokens (``<|speech_N|>``). One graph serves the whole loop:

``neutts_lm.onnx``
    inputs
        ``input_ids``       int64 ``[1, S]``      — ``S`` prompt tokens, or 1 per decode step
        ``attention_mask``  int64 ``[1, P + S]``  — ones over past **and** current tokens
        ``position_ids``    int64 ``[1, S]``      — absolute positions (``P .. P+S-1``)
        ``past_key_<i>``    fp32  ``[1, 4, P, 64]`` for ``i`` in ``0..27``
        ``past_value_<i>``  fp32  ``[1, 4, P, 64]``
    outputs
        ``logits``          fp32  ``[1, V]``      — **last position only**
        ``present_key_<i>`` / ``present_value_<i>`` fp32 ``[1, 4, P + S, 64]``

Prefill is ``P = 0`` with the full prompt; every decode step is ``S = 1`` with the
returned cache. Emitting one graph rather than a separate prefill/decode pair keeps a
single copy of the ~0.3 B weights on disk — the KV-cache contract is identical either
way, and it is the same shape ``phoonnx.engines.chatterbox`` already drives.

Only the last position's logits are returned: a full ``[1, S, 66938]`` tensor is ~250 MB
of useless prefill output, and sampling only ever reads the final row.

Usage::

    python export_neutts_onnx.py --repo afrispeech/Akiti-TTS --out-dir ./onnx
    python export_neutts_onnx.py --repo afrispeech/Akiti-TTS --out-dir ./onnx --quantize
    python export_neutts_onnx.py --repo afrispeech/Akiti-TTS --out-dir ./onnx --check-parity

Upstream weights keep their own license; see the model card of the source repo.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


class NeuTTSExportWrapper(torch.nn.Module):
    """Flattens the HF cache API into positional tensors the ONNX graph can carry."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_layers = model.config.num_hidden_layers

    def forward(self, input_ids, attention_mask, position_ids, past):
        # ``past`` is one flat list of 2*L tensors rather than varargs: torch.export
        # needs a fixed arity to attach per-input dynamic shapes to.
        from transformers.cache_utils import DynamicCache

        legacy = tuple(
            (past[2 * i], past[2 * i + 1]) for i in range(self.num_layers)
        )
        cache = DynamicCache.from_legacy_cache(legacy)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        present = out.past_key_values.to_legacy_cache()
        flat = [t for layer in present for t in layer]
        return (out.logits[:, -1, :], *flat)


def _io_names(num_layers: int):
    past = [n for i in range(num_layers) for n in (f"past_key_{i}", f"past_value_{i}")]
    present = [n for i in range(num_layers) for n in (f"present_key_{i}", f"present_value_{i}")]
    return past, present


def _dummy_inputs(config, seq: int, past_len: int, dtype=torch.float32):
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    input_ids = torch.randint(0, 300, (1, seq), dtype=torch.int64)
    attention_mask = torch.ones((1, past_len + seq), dtype=torch.int64)
    position_ids = torch.arange(past_len, past_len + seq, dtype=torch.int64)[None, :]
    past = [
        torch.zeros((1, kv_heads, past_len, head_dim), dtype=dtype)
        for _ in range(2 * config.num_hidden_layers)
    ]
    return input_ids, attention_mask, position_ids, past


def export(repo: str, out_dir: Path, opset: int = 17) -> Path:
    from transformers import AutoModelForCausalLM

    out_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32)
    model.eval()
    config = model.config
    wrapper = NeuTTSExportWrapper(model)

    past_names, present_names = _io_names(config.num_hidden_layers)
    input_names = ["input_ids", "attention_mask", "position_ids"] + past_names
    output_names = ["logits"] + present_names

    seq = torch.export.Dim("seq")
    dynamic_shapes = {
        "input_ids": {1: seq},
        "attention_mask": {1: torch.export.Dim("total")},
        "position_ids": {1: seq},
        "past": [{2: torch.export.Dim("past")} for _ in past_names],
    }

    # A non-empty past in the export sample is what puts the cache-concat path (rather
    # than the "no cache yet" branch) in the graph; the dynamic axes then let past=0
    # work at prefill time.
    ids, mask, pos, past = _dummy_inputs(config, seq=4, past_len=3)
    path = out_dir / "neutts_lm.onnx"
    torch.onnx.export(
        wrapper,
        (ids, mask, pos, past),
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        opset_version=opset,
        dynamo=True,
    )
    _write_meta(out_dir, repo, config)
    return path


def _write_meta(out_dir: Path, repo: str, config) -> None:
    (out_dir / "neutts_onnx_meta.json").write_text(json.dumps({
        "source_repo": repo,
        "architecture": config.architectures[0],
        "num_hidden_layers": config.num_hidden_layers,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "vocab_size": config.vocab_size,
        "eos_token_id": config.eos_token_id,
    }, indent=2) + "\n")


def quantize(fp32_path: Path) -> Path:
    """Dynamic int8 quantization of the MatMuls (the weights dominate this graph)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = fp32_path.with_name(fp32_path.stem + "_int8.onnx")
    quantize_dynamic(str(fp32_path), str(out), weight_type=QuantType.QInt8,
                     extra_options={"MatMulConstBOnly": True})
    return out


def check_parity(repo: str, onnx_path: Path, seq: int = 24, steps: int = 8) -> dict:
    """Compare ONNX against the torch model on a fixed prompt: the prefill logits and
    every decode step, threading the ONNX cache through exactly as the engine does."""
    import onnxruntime
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).eval()
    config = model.config
    past_names, present_names = _io_names(config.num_hidden_layers)

    sess = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]

    ids = torch.randint(0, 60000, (1, seq), dtype=torch.int64)

    # --- prefill
    feed = {
        "input_ids": ids.numpy(),
        "attention_mask": np.ones((1, seq), np.int64),
        "position_ids": np.arange(seq, dtype=np.int64)[None, :],
    }
    kv_shape = (1, config.num_key_value_heads, 0, config.head_dim)
    feed.update({n: np.zeros(kv_shape, np.float32) for n in past_names})
    onnx_out = dict(zip(out_names, sess.run(None, feed)))

    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=True)
    diffs = {"prefill": float(np.abs(onnx_out["logits"] - ref.logits[:, -1, :].numpy()).max())}

    # --- decode steps, greedy so both stacks walk the same token path
    past = {p: onnx_out[q] for p, q in zip(past_names, present_names)}
    cache = ref.past_key_values
    token = int(np.argmax(onnx_out["logits"]))
    worst = 0.0
    for step in range(steps):
        pos = seq + step
        feed = {
            "input_ids": np.array([[token]], np.int64),
            "attention_mask": np.ones((1, pos + 1), np.int64),
            "position_ids": np.array([[pos]], np.int64),
            **past,
        }
        onnx_out = dict(zip(out_names, sess.run(None, feed)))
        with torch.no_grad():
            ref = model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True)
        cache = ref.past_key_values
        worst = max(worst, float(np.abs(onnx_out["logits"] - ref.logits[:, -1, :].numpy()).max()))
        past = {p: onnx_out[q] for p, q in zip(past_names, present_names)}
        token = int(np.argmax(onnx_out["logits"]))
    diffs["decode"] = worst
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="afrispeech/Akiti-TTS",
                    help="HF repo id or local directory holding the merged checkpoint")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--quantize", action="store_true", help="also write an int8 graph")
    ap.add_argument("--check-parity", action="store_true",
                    help="compare the exported graph against torch and print max abs diff")
    args = ap.parse_args()

    path = export(args.repo, args.out_dir, args.opset)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    if args.quantize:
        q = quantize(path)
        print(f"wrote {q} ({os.path.getsize(q) / 1e6:.1f} MB)")
    if args.check_parity:
        for name, diff in check_parity(args.repo, path).items():
            print(f"parity {name}: max abs diff {diff:.3e}")


if __name__ == "__main__":
    main()
