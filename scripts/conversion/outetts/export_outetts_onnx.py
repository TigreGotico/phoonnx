#!/usr/bin/env python3
"""Export an OuteTTS 1.0 backbone to ONNX with the KV-cache contract phoonnx drives.

OuteTTS 1.0 ships two checkpoints — ``OuteTTS-1.0-0.6B`` (Qwen3) and
``Llama-OuteTTS-1.0-1B`` (Llama-3.2). Both are plain causal LMs that emit DAC.speech
codec tokens (``<|c1_N|>``/``<|c2_N|>``). One graph serves prefill *and* decode::

    model.onnx
        inputs
            ``input_ids``                 int64 ``[1, S]``      prompt, or 1 token/step
            ``attention_mask``            int64 ``[1, P + S]``  ones over past + current
            ``position_ids``              int64 ``[1, S]``      absolute, ``P .. P+S-1``
            ``past_key_values.<i>.key``   fp32  ``[1, H, P, D]`` for ``i`` in ``0..L-1``
            ``past_key_values.<i>.value`` fp32  ``[1, H, P, D]``
        outputs
            ``logits``                    fp32  ``[1, 1, V]``   last position only
            ``present.<i>.key`` / ``present.<i>.value``  fp32 ``[1, H, P + S, D]``

The names match OuteAI's own ONNX exports, so ``phoonnx.engines.outetts`` drives a graph
from this script and one from ``OuteAI/*-ONNX`` with the same code.

``logits`` carries only the final row. OuteAI's exports return every position, which is a
~1 GB tensor on a 1843-token prefill that the sampler never reads;
``OuteTTSAdapter.generate`` indexes ``logits[0, -1]``, which is correct either way.

Why re-export the 1B
~~~~~~~~~~~~~~~~~~~~
``OuteAI/Llama-OuteTTS-1.0-1B-ONNX`` does not reproduce its own torch weights. Measured
against ``OuteAI/Llama-OuteTTS-1.0-1B`` in float32 on a real 1843-token OuteTTS prompt,
the published float32 graph is off by **12 logits** at the last position with a logit
correlation of **0.48**, and greedy decoding diverges. The gap is already there at 32
tokens, so it is the export and not accumulation. The 0.6B export, by contrast, matches
to 3.8e-05 — so this script exists for the 1B and is verified on both.

Usage::

    python export_outetts_onnx.py --repo OuteAI/Llama-OuteTTS-1.0-1B --out-dir ./onnx
    python export_outetts_onnx.py --repo OuteAI/OuteTTS-1.0-0.6B --out-dir ./onnx \\
        --check-parity

``--check-parity`` prints the diagnostic (prefill/decode max abs diff, greedy agreement)
and then exits non-zero if greedy agreement is below 100% or either max diff exceeds
1e-4, so a broken re-export fails CI instead of scrolling past in a log.

Upstream weights keep their own license: ``OuteTTS-1.0-0.6B`` is Apache-2.0, and
``Llama-OuteTTS-1.0-1B`` is CC-BY-NC-SA-4.0 (no commercial use).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


class OuteTTSExportWrapper(torch.nn.Module):
    """Flattens the HuggingFace cache API into positional tensors an ONNX graph carries."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_layers = model.config.num_hidden_layers

    def forward(self, input_ids, attention_mask, position_ids, past):
        # ``past`` is one flat list of 2*L tensors rather than varargs: torch.export
        # needs a fixed arity to attach per-input dynamic shapes to.
        from transformers.cache_utils import DynamicCache

        legacy = tuple((past[2 * i], past[2 * i + 1]) for i in range(self.num_layers))
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
        # keep the sequence axis so ``logits[0, -1]`` reads the same row here as it does
        # on an OuteAI export that returns every position
        return (out.logits[:, -1:, :], *flat)


def _load_fp32(repo: str):
    """Load *repo* in float32 across transformers versions.

    The keyword was renamed ``torch_dtype`` -> ``dtype`` in transformers 4.56; both
    spellings exist in the wild, so pick whichever this install accepts rather than
    pinning the caller to one version.
    """
    import inspect

    from transformers import AutoModelForCausalLM

    keyword = "dtype" if "dtype" in inspect.signature(
        AutoModelForCausalLM.from_pretrained).parameters else "torch_dtype"
    return AutoModelForCausalLM.from_pretrained(repo, **{keyword: torch.float32}).eval()


def _io_names(num_layers: int):
    past = [f"past_key_values.{i}.{k}"
            for i in range(num_layers) for k in ("key", "value")]
    present = [f"present.{i}.{k}"
               for i in range(num_layers) for k in ("key", "value")]
    return past, present


def _kv_geometry(config):
    """(heads, head_dim) of one KV cache entry.

    ``head_dim`` is explicit on Qwen3 and on recent Llama configs, and derived from the
    hidden size on older ones.
    """
    head_dim = getattr(config, "head_dim", None) or \
        config.hidden_size // config.num_attention_heads
    return config.num_key_value_heads, head_dim


def _dummy_inputs(config, seq: int, past_len: int, dtype=torch.float32):
    kv_heads, head_dim = _kv_geometry(config)
    input_ids = torch.randint(0, 300, (1, seq), dtype=torch.int64)
    attention_mask = torch.ones((1, past_len + seq), dtype=torch.int64)
    position_ids = torch.arange(past_len, past_len + seq, dtype=torch.int64)[None, :]
    past = [
        torch.zeros((1, kv_heads, past_len, head_dim), dtype=dtype)
        for _ in range(2 * config.num_hidden_layers)
    ]
    return input_ids, attention_mask, position_ids, past


def export(repo: str, out_dir: Path, opset: int = 18,
           filename: str = "model.onnx") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = _load_fp32(repo)
    config = model.config
    wrapper = OuteTTSExportWrapper(model)

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
    path = out_dir / filename
    # 1.2 B float32 parameters do not fit in a protobuf, so the weights always go to the
    # ``model.onnx_data`` sidecar; phoonnx's model manager fetches that sidecar by name.
    torch.onnx.export(
        wrapper,
        (ids, mask, pos, past),
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        opset_version=opset,
        dynamo=True,
        external_data=True,
    )
    _rename_external_data(path)
    _write_meta(out_dir, repo, config)
    return path


def _rename_external_data(path: Path) -> Path:
    """Move the weight sidecar to ``<graph name>_data`` and repoint the graph at it.

    ``torch.onnx.export`` writes ``model.onnx.data``. phoonnx's model manager derives the
    sidecar URL from the graph URL by appending ``_data``
    (:meth:`phoonnx.model_manager.TTSModelInfo._fetch_onnx`), which is also the
    convention every OuteAI/optimum export uses, so a graph exported here has to say
    ``model.onnx_data`` or a downloaded voice will not find its weights.

    Only the protobuf is rewritten — the multi-GB sidecar is renamed on disk, never
    loaded — so this costs no memory.
    """
    import onnx

    target = path.name + "_data"
    written = path.with_name(path.name + ".data")
    if not written.exists():
        return path

    model = onnx.load(str(path), load_external_data=False)

    def repoint(graph):
        for tensor in graph.initializer:
            for entry in tensor.external_data:
                if entry.key == "location":
                    entry.value = target
        for node in graph.node:
            for attr in node.attribute:
                if attr.HasField("g"):
                    repoint(attr.g)
                for sub in attr.graphs:
                    repoint(sub)

    repoint(model.graph)
    onnx.save(model, str(path), save_as_external_data=False)
    written.rename(path.with_name(target))
    return path


def _write_meta(out_dir: Path, repo: str, config) -> None:
    kv_heads, head_dim = _kv_geometry(config)
    (out_dir / "outetts_onnx_meta.json").write_text(json.dumps({
        "source_repo": repo,
        "architecture": config.architectures[0],
        "num_hidden_layers": config.num_hidden_layers,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "vocab_size": config.vocab_size,
        "eos_token_id": config.eos_token_id,
    }, indent=2) + "\n")


def check_parity(repo: str, onnx_path: Path, seq: int = 64, steps: int = 16) -> dict:
    """Compare ONNX against torch on a fixed prompt: the prefill logits and every decode
    step, threading the ONNX cache through exactly as the engine does."""
    import onnxruntime

    torch.manual_seed(0)
    model = _load_fp32(repo)
    config = model.config
    kv_heads, head_dim = _kv_geometry(config)
    past_names, present_names = _io_names(config.num_hidden_layers)

    sess = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]

    ids = torch.randint(0, config.vocab_size, (1, seq), dtype=torch.int64)

    feed = {
        "input_ids": ids.numpy(),
        "attention_mask": np.ones((1, seq), np.int64),
        "position_ids": np.arange(seq, dtype=np.int64)[None, :],
    }
    feed.update({n: np.zeros((1, kv_heads, 0, head_dim), np.float32) for n in past_names})
    onnx_out = dict(zip(out_names, sess.run(None, feed)))

    with torch.no_grad():
        ref = model(input_ids=ids, use_cache=True)
    reference = ref.logits[:, -1, :].numpy()
    got = np.asarray(onnx_out["logits"]).reshape(1, -1)
    diffs = {"prefill": float(np.abs(got - reference).max())}

    past = {p: onnx_out[q] for p, q in zip(past_names, present_names)}
    cache = ref.past_key_values
    token = int(np.argmax(got))
    worst = 0.0
    agree = 0
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
            ref = model(input_ids=torch.tensor([[token]]), past_key_values=cache,
                        use_cache=True)
        cache = ref.past_key_values
        reference = ref.logits[:, -1, :].numpy()
        got = np.asarray(onnx_out["logits"]).reshape(1, -1)
        worst = max(worst, float(np.abs(got - reference).max()))
        agree += int(np.argmax(got) == np.argmax(reference))
        past = {p: onnx_out[q] for p, q in zip(past_names, present_names)}
        token = int(np.argmax(got))
    diffs["decode"] = worst
    diffs["greedy_agreement"] = agree / steps
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="OuteAI/Llama-OuteTTS-1.0-1B",
                    help="HF repo id or local directory holding the checkpoint")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--filename", default="model.onnx")
    ap.add_argument("--check-parity", action="store_true",
                    help="compare the exported graph against torch, print max abs diff, "
                         "and exit non-zero if greedy_agreement < 100%% or a max diff "
                         "exceeds the 1e-4 tolerance")
    args = ap.parse_args()

    path = export(args.repo, args.out_dir, args.opset, args.filename)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB + sidecar)")
    if args.check_parity:
        tolerance = 1e-4
        diffs = check_parity(args.repo, path)
        for name, value in diffs.items():
            print(f"parity {name}: {value:.3e}" if name != "greedy_agreement"
                  else f"parity {name}: {value:.0%}")
        if diffs["greedy_agreement"] < 1.0 or diffs["prefill"] > tolerance \
                or diffs["decode"] > tolerance:
            print(f"parity check FAILED (tolerance={tolerance:.0e})")
            sys.exit(1)


if __name__ == "__main__":
    main()
