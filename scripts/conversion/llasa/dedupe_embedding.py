#!/usr/bin/env python3
"""Drop the duplicated tied embedding from an exported Llasa graph.

Llasa ties ``lm_head`` to ``embed_tokens``, but the ONNX exporter writes the
193,800 x 2,048 matrix twice: once as the ``Gather`` table and once, transposed,
as the ``lm_head`` ``MatMul`` operand. That is 3.2 GB of pure duplication in an
fp32 export.

This rewrites ``MatMul(x, W_t)`` into ``Reshape -> Gemm(x, W, transB=1) ->
Unsqueeze``, so the head reuses the embedding initializer, and deletes the
transposed copy. ``Gemm`` needs a 2-D left operand, hence the reshape; the
sliced hidden state is ``[batch, 1, hidden]``, so the reshape is free.

The rewrite is arithmetically identical and is verified by re-running
``parity_llm.py`` against the deduplicated graph.

Usage::

    python dedupe_embedding.py --input out/llasa-1b/model_fp32.onnx \
                               --output out/llasa-1b/model.onnx
"""
from __future__ import annotations

import argparse

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--data-file", default=None,
                    help="external data filename (default: <output basename>_data)")
    args = ap.parse_args()

    model = onnx.load(args.input)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}

    gather = next(n for n in graph.node
                  if n.op_type == "Gather" and n.input[0] in inits
                  and len(inits[n.input[0]].dims) == 2)
    embed_name = gather.input[0]
    vocab, hidden = inits[embed_name].dims
    print(f"embedding initializer {embed_name} [{vocab}, {hidden}]")

    head = next(n for n in graph.node
                if n.op_type == "MatMul" and len(n.input) > 1 and n.input[1] in inits
                and list(inits[n.input[1]].dims) == [hidden, vocab])
    head_w = head.input[1]
    print(f"lm_head initializer {head_w} [{hidden}, {vocab}]")

    a = numpy_helper.to_array(inits[embed_name])
    b = numpy_helper.to_array(inits[head_w])
    if not np.array_equal(a, b.T):
        raise SystemExit("lm_head weight is not the transposed embedding — refusing to dedupe")
    print("verified: lm_head weight == embedding transposed")

    x, out = head.input[0], head.output[0]
    shape_2d = helper.make_tensor(f"{head.name}_shape2d", onnx.TensorProto.INT64, [2], [-1, hidden])
    axes = helper.make_tensor(f"{head.name}_axes", onnx.TensorProto.INT64, [1], [1])
    graph.initializer.extend([shape_2d, axes])

    new_nodes = [
        helper.make_node("Reshape", [x, shape_2d.name], [f"{out}_2d"], name=f"{head.name}_reshape"),
        helper.make_node("Gemm", [f"{out}_2d", embed_name], [f"{out}_gemm"],
                         name=f"{head.name}_gemm", transB=1),
        helper.make_node("Unsqueeze", [f"{out}_gemm", axes.name], [out],
                         name=f"{head.name}_unsqueeze"),
    ]
    idx = list(graph.node).index(head)
    graph.node.remove(head)
    for offset, node in enumerate(new_nodes):
        graph.node.insert(idx + offset, node)
    graph.initializer.remove(inits[head_w])

    data_file = args.data_file or args.output.rsplit("/", 1)[-1] + "_data"
    onnx.save_model(model, args.output, save_as_external_data=True,
                    all_tensors_to_one_file=True, location=data_file, size_threshold=1024)
    # The checker cannot serialise a >2 GB proto in memory, so it runs on the
    # saved path, where external data is read lazily.
    onnx.checker.check_model(args.output, full_check=False)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
