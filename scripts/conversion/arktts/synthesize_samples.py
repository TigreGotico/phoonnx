"""Synthesize a sentence list through the ArkTTS adapter and report per-voice CPU RTF.

    python synthesize_samples.py --onnx-dir ~/zortzi-tts-onnx --precision fp16 \
        --tokenizer ~/zortzi-tts-onnx/tokenizer/tokenizer.json \
        --voice maider=voices/maider.json --voice antton=voices/antton.json \
        --sentences eu.txt --out-dir ./samples --report rtf.json

This drives :class:`phoonnx.engines.arktts.ArkTTSAdapter` exactly as the voice layer does,
so what it measures is what a phoonnx user gets: one slow-AR step and nine fast-AR steps
per 46 ms frame, on CPU, single stream.

Every call is seeded, so a rerun reproduces the same audio. That matters because ArkTTS
*must* sample — greedy decoding never terminates — and an unseeded benchmark would compare
different utterances between runs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile

from phoonnx.engines.arktts import SAMPLE_RATE, ArkTTSAdapter
from phoonnx.engines.base import AdapterSynthesisRequest
from phoonnx.tokenizer import BPETokenizer


def build_adapter(onnx_dir: Path, precision: str, tokenizer: Path, codec: Path):
    """Open the three graphs and return the adapter plus the slow-AR session."""
    def session(path: Path):
        return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    adapter = ArkTTSAdapter()
    adapter.fast_ar = session(onnx_dir / f"fast_ar_{precision}.onnx")
    adapter.decoder = session(codec)
    adapter.tokenizer = BPETokenizer(str(tokenizer))
    return adapter, session(onnx_dir / f"slow_ar_{precision}.onnx")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-dir", required=True, type=Path)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--codec", type=Path, default=None,
                        help="codec decoder graph (defaults to codec_decoder_fp16.onnx)")
    parser.add_argument("--voice", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--sentences", required=True, type=Path,
                        help="one sentence per line, optionally prefixed 'lang|'; "
                             "blank lines and '#' comments ignored")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    codec = args.codec or args.onnx_dir / "codec_decoder_fp16.onnx"
    adapter, slow = build_adapter(args.onnx_dir, args.precision, args.tokenizer, codec)
    # A line may carry its own language tag as "en|The quick brown fox ...". The tag is not
    # used for synthesis — ArkTTS infers the language from the text itself and has no
    # language token — it travels to the report so the WER gate knows how to score the clip.
    sentences = []
    for line in args.sentences.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        language, separator, text = line.partition("|")
        sentences.append((language if separator else "", text if separator else line))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in args.voice:
        name, _, path = spec.partition("=")
        adapter.load_voice(path)
        audio_total = wall_total = 0.0
        for index, (language, sentence) in enumerate(sentences):
            ids = adapter.encode_text(sentence, None, None)[0]
            request = AdapterSynthesisRequest(
                phoneme_ids=np.asarray(ids, np.int64).reshape(1, -1),
                phoneme_lengths=np.asarray([len(ids)], np.int64),
                params={"seed": args.seed + index})
            started = time.perf_counter()
            result = adapter.synthesize(request, slow)
            elapsed = time.perf_counter() - started
            duration = result.audio.size / SAMPLE_RATE
            audio_total += duration
            wall_total += elapsed
            out = args.out_dir / f"{name}_{index:02d}.wav"
            soundfile.write(str(out), result.audio, SAMPLE_RATE)
            rows.append({"voice": name, "index": index, "text": sentence,
                         "language": language,
                         "wav": out.name, "frames": result.extras["frame_count"],
                         "audio_seconds": round(duration, 3),
                         "wall_seconds": round(elapsed, 3),
                         "rtf": round(elapsed / max(duration, 1e-9), 3)})
            print(f"  {out.name}  {duration:5.2f}s audio  {elapsed:6.1f}s wall  "
                  f"RTF {elapsed / max(duration, 1e-9):5.2f}")
        print(f"{name}: aggregate RTF {wall_total / max(audio_total, 1e-9):.2f} "
              f"over {audio_total:.1f}s of audio")
        rows.append({"voice": name, "aggregate_rtf": round(wall_total / max(audio_total, 1e-9), 3),
                     "audio_seconds": round(audio_total, 2)})

    if args.report:
        args.report.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        print("wrote", args.report)


if __name__ == "__main__":
    main()
