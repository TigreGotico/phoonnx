#!/usr/bin/env python3
"""Check the exported XCodec2 decoder against the torch reference.

Two comparisons are made on the same speech tokens:

1. the real-valued ISTFT head against the upstream complex head, in torch;
2. the ONNX ``decoder.onnx`` against the upstream torch ``decode_code``.

Usage::

    python parity_codec.py --onnx out/xcodec2/decoder.onnx --tokens ref_greedy_tokens.npy
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from export_xcodec2 import DecoderWrapper, load_xcodec2, patch_istft

# Measured on the shipped export: onnx-vs-torch max abs sample diff ~1.2e-4 against
# a signal RMS of 0.258, ~66 dB below signal (see PR #366). 1e-2 (~28 dB) leaves
# broad headroom over that noise floor while still catching an export that has
# actually broken (a wrong ISTFT sign or a mis-wired fold shows up orders of
# magnitude above this).
MAX_ABS_DIFF_TOLERANCE = 1e-2
MIN_CORRELATION = 0.999


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return {
        "samples": int(n),
        "max_abs_diff": float(np.abs(a - b).max()),
        "mean_abs_diff": float(np.abs(a - b).mean()),
        "rms_ref": float(np.sqrt((a ** 2).mean())),
        "corr": float(np.corrcoef(a, b)[0, 1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tokens", required=True, help=".npy of raw speech token ids (0..65535)")
    args = ap.parse_args()

    import onnxruntime

    codes_np = np.load(args.tokens).astype(np.int64).reshape(1, 1, -1)
    codes = torch.from_numpy(codes_np)

    model = load_xcodec2()
    with torch.no_grad():
        complex_audio = model.decode_code(codes[:, 0, :].unsqueeze(1))[0, 0].numpy()

    patch_istft(model)
    dec = DecoderWrapper(model).eval()
    with torch.no_grad():
        real_audio = dec(codes)[0, 0].numpy()

    sess = onnxruntime.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    onnx_audio = np.asarray(sess.run(None, {"codes": codes_np})[0]).reshape(-1)

    report = {
        "real_istft_vs_complex_istft": stats(complex_audio, real_audio),
        "onnx_vs_torch_complex": stats(complex_audio, onnx_audio),
        "onnx_vs_torch_real": stats(real_audio, onnx_audio),
    }
    print(json.dumps(report, indent=2))

    failed = False
    for name, row in report.items():
        if row["max_abs_diff"] > MAX_ABS_DIFF_TOLERANCE:
            print(f"parity FAILED: {name} max_abs_diff={row['max_abs_diff']:.3e} "
                  f"> tolerance {MAX_ABS_DIFF_TOLERANCE:.0e}")
            failed = True
        if row["corr"] < MIN_CORRELATION:
            print(f"parity FAILED: {name} corr={row['corr']:.6f} "
                  f"< tolerance {MIN_CORRELATION}")
            failed = True
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
