#!/usr/bin/env python3
"""Synthesize the WER/RTF probe set through the Orpheus adapter and score it.

    python bench_wer.py --onnx ./orpheus-3b-en-onnx --out ./evidence/wer.json

Runs a fixed set of sentences across three voices (``tara``, ``leo``, ``zoe``) plus one
emotion-tag smoke test, times the LM loop to get RTF, transcribes each clip with
OpenVoiceOS's qwen3-asr-0.6b-onnx (never an unlabeled whisper, per house policy), and
writes a report shaped like ``evidence/wer.json`` — the file this PR's ``evidence/``
directory carries, fetched from the mirror's original run.

This is the same probe set and methodology that produced the PR body's RTF and WER
tables: mean 38.7x realtime, mean WER 0.0000 over the six tag-free utterances. Running
this script downloads the ~13 GB mirrored ONNX graphs plus the ASR model, so it is not
part of this PR's laptop-light evidence gathering; ``evidence/wer.json`` is the prior
run's output, fetched rather than regenerated, with provenance in ``evidence/README.md``.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore every summer morning.",
]
VOICES = ["tara", "leo", "zoe"]
EMOTION_PROBE = ("tara", "That is really funny <laugh> I cannot believe it.")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, want in enumerate(reference, start=1):
        current = [i]
        for j, got in enumerate(hypothesis, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (want != got)))
        previous = current
    return previous[-1]


def wer_cer(reference: str, hypothesis: str) -> tuple[float, float]:
    ref_words, hyp_words = normalize(reference).split(), normalize(hypothesis).split()
    ref_chars = list(normalize(reference).replace(" ", ""))
    hyp_chars = list(normalize(hypothesis).replace(" ", ""))
    wer = edit_distance(ref_words, hyp_words) / max(len(ref_words), 1)
    cer = edit_distance(ref_chars, hyp_chars) / max(len(ref_chars), 1)
    return round(wer, 4), round(cer, 4)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", default="./orpheus-3b-en-onnx",
                     help="dir with model.onnx(+_data shards), snac_decoder.onnx, "
                          "tokenizer.json — the layout of OpenVoiceOS/phoonnx-orpheus")
    ap.add_argument("--asr", default="OpenVoiceOS/qwen3-asr-0.6b-onnx")
    ap.add_argument("--wavdir", type=Path, default=Path("./samples"))
    ap.add_argument("--out", type=Path, default=Path("./evidence/wer.json"))
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    import numpy as np
    import onnx_asr
    import soundfile

    from phoonnx.engines.base import AdapterSynthesisRequest
    from phoonnx.engines.orpheus import OrpheusAdapter
    from phoonnx.providers import make_session

    args.wavdir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    adapter = OrpheusAdapter()
    adapter.configure(SimpleNamespace(engine_params={
        "snac_decoder_path": f"{args.onnx}/snac_decoder.onnx",
        "tokenizer_path": f"{args.onnx}/tokenizer.json",
        "voices": VOICES + ["dan", "leah", "jess", "mia", "zac"],
        "default_voice": "tara",
    }))
    session = make_session(f"{args.onnx}/model.onnx")
    asr = onnx_asr.load_model(args.asr, providers=["CPUExecutionProvider"])

    probes = [(v, s) for v in VOICES for s in SENTENCES] + [EMOTION_PROBE]
    rows = []
    for i, (voice, text) in enumerate(probes):
        ids = adapter.build_prompt_ids(text, voice)
        t0 = time.monotonic()
        tokens = adapter.generate(session, ids, adapter.default_params(),
                                   np.random.default_rng(0))
        lm_s = time.monotonic() - t0
        streams = adapter.token_ids_to_codes(tokens)
        audio = adapter.decode_codes(streams)
        audio_s = len(audio) / 24000
        wav = args.wavdir / f"wav_{i:02d}_{voice}.wav"
        soundfile.write(str(wav), audio, 24000)

        mono16k = audio if True else audio  # ASR resample handled by onnx_asr if needed
        hyp = str(asr.recognize(audio, sample_rate=24000, language="en"))
        wer, cer = wer_cer(text.replace("<laugh>", "").strip(), hyp)

        row = {"wav": wav.name, "voice": voice, "text": text, "tokens": len(tokens),
               "audio_s": round(audio_s, 2), "lm_s": round(lm_s, 1),
               "rtf": round(lm_s / max(audio_s, 1e-6), 1),
               "ms_per_token": round(1000 * lm_s / max(len(tokens), 1)),
               "hyp": hyp, "wer": wer, "cer": cer}
        rows.append(row)
        print(f"  {row['wav']}  rtf={row['rtf']}  wer={wer}  {hyp!r}")

    args.out.write_text(json.dumps(rows, indent=1) + "\n")
    mean_wer = sum(r["wer"] for r in rows[:-1]) / max(len(rows) - 1, 1)
    mean_rtf = sum(r["rtf"] for r in rows) / len(rows)
    print(f"\nmean WER (tag-free utterances): {mean_wer:.4f}")
    print(f"mean RTF: {mean_rtf:.1f}x")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
