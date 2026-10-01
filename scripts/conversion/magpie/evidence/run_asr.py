"""Transcribe the Magpie-TTS WER-gate samples and score them against the prompts.

    python synth.py                # writes ./wav/<lang>_<i>.wav
    python run_asr.py              # scores every language, writes results.json, gates

This is an intelligibility check, not an ASR benchmark: it answers whether the words the
engine was asked to say come back out. Exits non-zero if any language's WER (CER for ko,
whose script the ASR only handles at the character level, and where a WER of 0.0 is
scored below) rises above ``--threshold`` (default 30%), so a regression fails CI/manual
gating instead of only showing up as a smaller number in a JSON file nobody reads.

**Korean ASR caveat.** No OpenVoiceOS-org ONNX ASR model covers Korean, and the org has
not shipped one. ``ko`` is scored with ``onnx-community/whisper-large-v3-turbo`` as an
explicit fallback — it is not the calibrated in-house pipeline the other four languages
use, so the ko number is a best-effort intelligibility check, not a like-for-like
comparison with fr/it/vi/ar.

**Vietnamese diacritics.** The vi reference sentences were hand-corrected for diacritics
before this run (Vietnamese distinguishes words by tone marks the source draft dropped or
mis-typed); ``sentences.json`` here already carries the corrected forms.
"""
import argparse
import json
import sys
from pathlib import Path

import jiwer
import onnx_asr

HERE = Path(__file__).parent
WAV_DIR = HERE / "wav"
SENTENCES = json.load(open(HERE / "sentences.json"))

ASR_REPOS = {
    "fr": "OpenVoiceOS/nvidia-fr-conformer-transducer-large-onnx",
    "it": "OpenVoiceOS/nvidia-it-conformer-transducer-large-onnx",
    "vi": "OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx",
    "ar": "OpenVoiceOS/stt_ar_fastconformer_hybrid_large_pc_v1.0_onnx",
    "ko": "onnx-community/whisper-large-v3-turbo",  # no ko ASR in OpenVoiceOS org; explicit fallback
}


def normalize(s):
    return " ".join(s.lower().replace(",", "").replace(".", "").replace("?", "").replace("!", "").split())


def main(langs, threshold):
    results = {}
    for lang in langs:
        repo = ASR_REPOS[lang]
        if repo.startswith("onnx-community/whisper"):
            m = onnx_asr.load_model(repo, quantization=None)
        else:
            m = onnx_asr.load_model(repo)
        hyps, refs = [], []
        for i, ref in enumerate(SENTENCES[lang]):
            wav_path = WAV_DIR / f"{lang}_{i}.wav"
            if lang == "ko":
                hyp = m.recognize(str(wav_path), language="ko")
            else:
                hyp = m.recognize(str(wav_path))
            hyps.append(normalize(hyp))
            refs.append(normalize(ref))
            print(f"[{lang}] ref={ref!r} hyp={hyp!r}", flush=True)
        wer = jiwer.wer(refs, hyps)
        cer = jiwer.cer(refs, hyps)
        results[lang] = {"wer": wer, "cer": cer, "refs": refs, "hyps": hyps, "asr": repo}
        print(f"[{lang}] WER={wer:.3f} CER={cer:.3f}", flush=True)

    out = HERE / "results.json"
    merged = json.loads(out.read_text()) if out.exists() else {}
    merged.update(results)
    out.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    print("wrote", out)

    failures = [(lang, r["wer"]) for lang, r in results.items() if r["wer"] > threshold]
    if failures:
        for lang, wer in failures:
            print(f"GATE FAIL: {lang} WER {wer:.3f} > threshold {threshold:.2f}", file=sys.stderr)
        sys.exit(1)
    print(f"GATE PASS: all languages under WER {threshold:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("langs", nargs="*", default=list(ASR_REPOS.keys()))
    parser.add_argument("--threshold", type=float, default=0.30, help="max acceptable WER before the gate fails")
    args = parser.parse_args()
    main(args.langs, args.threshold)
