#!/usr/bin/env python3
"""Check an exported SNAC decoder against torch — correctly, despite SNAC being
stochastic.

    python verify_snac.py --torch-model hubertsiuzdak/snac_24khz \
        --onnx ./orpheus-3b-en-onnx/snac_decoder.onnx --n-decodes 40

A naive ``max|onnx - torch|`` diff on SNAC's decoder looks like a broken export even
for a byte-correct fp32 ONNX conversion: SNAC's decoder contains a noise-injection
block, so **two torch decodes of the same input codes already differ** from each
other. A one-shot torch-vs-onnx diff cannot tell a bad export from that expected
spread.

Methodology, matching the one used to produce the numbers in the PR body and mirror
README:

1. Decode the same fixed code matrix through torch **N times** (default 40) to build
   the model's own run-to-run noise floor: relative RMS of each decode against the
   mean of all N.
2. Decode the ONNX candidate once and compute its relative RMS against that same
   torch mean.
3. Report the candidate's RMS as a **ratio to the floor** — a ratio under ~1.1x means
   the candidate sits inside the model's own stochastic spread and is not
   distinguishable from "another torch decode"; a ratio several times the floor
   (the int8/uint8 variants measured 4.8-6.0x in this PR) is a real defect, not noise.

Weights-heavy (needs the torch SNAC checkpoint and the ONNX decoder) — not run for
this PR's evidence; see ``evidence/README.md`` for where the quoted ratios came from.
Self-gating: exits non-zero if the candidate's ratio exceeds ``--max-ratio``.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def relative_rms(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    denom = np.sqrt(np.mean(b.astype(np.float64) ** 2)) + 1e-12
    return float(np.sqrt(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)) / denom)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--torch-model", default="hubertsiuzdak/snac_24khz")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--n-decodes", type=int, default=40,
                     help="torch decodes used to estimate the noise floor")
    ap.add_argument("--n-frames", type=int, default=50,
                     help="SNAC frames of fixed random codes to decode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-ratio", type=float, default=1.5,
                     help="candidate RMS / floor RMS above this fails; fp32 measured "
                          "~0.9-1.0x in this PR, int8/uint8 measured 4.8-6.0x")
    args = ap.parse_args()

    import torch
    from snac import SNAC

    from phoonnx.providers import make_session

    rng = np.random.default_rng(args.seed)
    # fixed code matrix: three streams at rates 1/2/4, matching the adapter's layout
    s0 = rng.integers(0, 4096, args.n_frames)
    s1 = rng.integers(0, 4096, args.n_frames * 2)
    s2 = rng.integers(0, 4096, args.n_frames * 4)
    codes = [torch.tensor(s0, dtype=torch.long)[None],
             torch.tensor(s1, dtype=torch.long)[None],
             torch.tensor(s2, dtype=torch.long)[None]]

    model = SNAC.from_pretrained(args.torch_model).eval()
    with torch.no_grad():
        decodes = [model.decode(codes).numpy().reshape(-1) for _ in range(args.n_decodes)]
    torch_mean = np.mean(np.stack(decodes), axis=0)

    floor_rms = float(np.mean([relative_rms(d, torch_mean) for d in decodes]))
    print(f"torch run-to-run noise floor (mean relative RMS over {args.n_decodes} "
          f"decodes vs their own mean): {floor_rms:.4f}")

    session = make_session(args.onnx)
    names = [i.name for i in session.get_inputs()]
    feed = {n: np.asarray(c[0], np.int64).reshape(1, -1) for n, c in zip(names, codes)}
    onnx_wav = np.asarray(session.run(None, feed)[0], np.float32).reshape(-1)

    candidate_rms = relative_rms(onnx_wav, torch_mean)
    ratio = candidate_rms / max(floor_rms, 1e-12)
    print(f"candidate relative RMS vs torch mean: {candidate_rms:.4f}")
    print(f"ratio to noise floor: {ratio:.2f}x")

    ok = ratio <= args.max_ratio
    print("PASS — inside the model's own stochastic spread" if ok
          else "FAIL — candidate diverges beyond SNAC's own noise floor")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
