"""Transcribe synthesized samples with a CPU ASR model and score them against the prompts.

    python wer_gate.py --samples ./samples --report rtf.json --language eu \
        --asr onnx-community/whisper-large-v3-turbo --out wer.json

This is an intelligibility check, not an ASR benchmark. The question it answers is whether
the words the engine was asked to say come back out, which catches the failure modes a
logit-parity check cannot see: a wrong prompt layout, a mis-scattered KV cache, or a
sampler that drifts into a repetition loop.

**ASR coverage for Basque.** ``nemo-canary-1b-v2`` covers the 25 official EU languages;
Basque is not one of them, and no other model ``onnx-asr`` ships recognises ``eu`` either.
There is no capable Basque CPU model available here, so Basque is scored with
``whisper-large-v3-turbo`` as an explicit best-effort: Whisper lists ``eu`` among its
languages, but it is a low-resource one for the model and its own error rate is high. A
Basque number here therefore bounds the engine's intelligibility from below and must not
be read as a measurement of the voice's quality.

For Chinese, Cantonese and Japanese the same run reports CER rather than WER, since
whitespace does not delimit words in those scripts.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

CHARACTER_SCORED = {"zh", "yue", "ja"}
"""Languages scored by character: their orthography has no whitespace word boundaries."""


def normalize(text: str) -> str:
    """Casefold, strip punctuation and collapse whitespace — the usual WER preprocessing."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance over tokens, the numerator of both WER and CER."""
    previous = list(range(len(hypothesis) + 1))
    for i, want in enumerate(reference, start=1):
        current = [i]
        for j, got in enumerate(hypothesis, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (want != got)))
        previous = current
    return previous[-1]


def tokenize(text: str, language: str) -> list[str]:
    return list(text.replace(" ", "")) if language in CHARACTER_SCORED else text.split()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path,
                        help="the JSON synthesize_samples.py wrote, for wav -> text")
    parser.add_argument("--language", required=True,
                        help="fallback for clips whose report row carries no language tag")
    parser.add_argument("--asr", default="onnx-community/whisper-large-v3-turbo")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import onnx_asr
    import soundfile

    rows = [row for row in json.loads(args.report.read_text()) if "wav" in row]
    model = onnx_asr.load_model(args.asr, providers=["CPUExecutionProvider"])

    scored: list[dict] = []
    tally: dict[str, list[int]] = {}
    for row in rows:
        path = args.samples / row["wav"]
        if not path.is_file():
            continue
        waveform, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        mono = waveform.mean(axis=1)
        if rate != 16000:
            from math import gcd

            from scipy.signal import resample_poly

            divisor = gcd(rate, 16000)
            mono = resample_poly(mono, 16000 // divisor, rate // divisor).astype("float32")
        language = row.get("language") or args.language
        hypothesis = model.recognize(mono, sample_rate=16000, language=language)
        want = tokenize(normalize(row["text"]), language)
        got = tokenize(normalize(str(hypothesis)), language)
        distance = edit_distance(want, got)
        counts = tally.setdefault(language, [0, 0, 0])
        counts[0] += distance
        counts[1] += len(want)
        counts[2] += 1
        scored.append({"wav": row["wav"], "voice": row["voice"], "language": language,
                       "reference": row["text"], "hypothesis": str(hypothesis),
                       "rate": round(distance / max(len(want), 1), 4)})
        print(f"  {row['wav']}  [{language}]  {distance}/{len(want)}  {hypothesis!r}")

    summary = {}
    print()
    for language, (errors, total, clips) in sorted(tally.items()):
        metric = "CER" if language in CHARACTER_SCORED else "WER"
        rate = errors / max(total, 1)
        summary[language] = {"metric": metric, "aggregate": round(rate, 4),
                             "errors": errors, "tokens": total, "clips": clips}
        print(f"{language}: {metric} {rate:.3f} ({errors}/{total}) over {clips} clips")
    print(f"ASR {args.asr}")

    if args.out:
        args.out.write_text(json.dumps(
            {"asr": args.asr, "by_language": summary, "clips": scored},
            ensure_ascii=False, indent=2) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
