#!/usr/bin/env python3
"""Self-gating benchmark for the ``vc_reference`` voice-conversion post-stage.

Runs the full matrix (utterances x source voices x target speakers), scores it,
writes a run JSON plus a markdown table, and exits non-zero when a gate fails.

What it measures
----------------
speaker similarity
    Cosine between speakeronnx embeddings.  Three numbers per pair:

    * ``sim_to_target`` — converted output vs the target-speaker reference.
      This is the number the feature exists for.
    * ``sim_to_source`` — converted output vs the same sentence in the source
      voice.  The **negative control**.  A pass-through "conversion" scores ~1.0
      here (see the mimi/quickvc failure mode), so this has to fall well below
      1.0.  Note it does **not** have to fall below ``sim_to_target``: when the
      source and the target are already similar speakers (same sex, similar
      pitch) the converted audio legitimately resembles both.
    * ``floor`` — source vs target with no conversion at all.  The baseline any
      real conversion has to beat.

intelligibility
    WER of an ASR pass (parakeet) against the reference text, before and after
    conversion.  Conversion may not wreck the words.

Gates
-----
1. Every (source voice, target speaker) pair moves toward the target: its mean
   ``sim_to_target - floor`` margin is positive, and the margin over the whole
   matrix is >= 0.10.  Aggregating per pair rather than per utterance is
   deliberate — the acoustic models are stochastic, so a single utterance can
   sit at the floor by chance without the pair being broken.
2. ``sim_to_source`` <= 0.85 for every pair — no source leakage.  A stage that
   silently passed audio through would score ~1.0.
3. Mean WER degradation <= 5 percentage points absolute.

Usage
-----
::

    python scripts/vc_gate.py --out vc-run.json

Everything is CPU-only and offline after the first model download.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --- matrix -----------------------------------------------------------------

UTTERANCES = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Please turn on the kitchen lights and set the thermostat to twenty degrees.",
    "She sold seashells by the seashore every summer morning without fail.",
    "Recording the weather forecast for tomorrow afternoon in central Portugal.",
    "An open source voice assistant should never send your speech to the cloud.",
]

# Three fast, architecturally different sources: a piper VITS voice, a kokoro
# StyleTTS2 voice, and a kittentts nano voice.
SOURCE_VOICES = [
    "piper/en_US-amy-low",
    "kokoro/af_heart",
    "kittentts/nano-0.2-expr-voice-2-m",
]

# Two target speakers, deliberately far apart (male / female) and from engines
# that are not in the source set, so no source-target pair is trivially close.
TARGET_VOICES = {
    "target_male": "piper/en_US-ryan-medium",
    "target_female": "supertonic/F1/en",
}

# The target reference clip: long enough for every VC engine's speaker encoder.
TARGET_REF_TEXT = (
    "This is a reference recording of my voice. I am speaking clearly and at a "
    "steady pace so that the speaker encoder has enough material to work with. "
    "The weather today is mild, with a light breeze from the north."
)

SIM_MARGIN_GATE = 0.10   # mean (sim_to_target - floor)
LEAKAGE_GATE = 0.85      # max sim_to_source; a pass-through scores ~1.0
WER_DELTA_GATE = 5.0     # percentage points absolute


# --- helpers ----------------------------------------------------------------

def _pkg_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _norm(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate in percent (Levenshtein over words)."""
    r, h = _norm(reference), _norm(hypothesis)
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return 100.0 * prev[-1] / len(r)


def write_wav(path: str, audio: np.ndarray, sr: int) -> str:
    pcm = np.clip(np.asarray(audio, np.float32).reshape(-1), -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes((pcm * 32767.0).astype("<i2").tobytes())
    return path


def load_voice(voice_id: str):
    from phoonnx.model_manager import TTSModelManager

    mgr = TTSModelManager()
    mgr.merge_default_voices()
    for info in mgr.all_voices:
        if info.voice_id == voice_id:
            return info.load()
    raise SystemExit(f"unknown voice id: {voice_id}")


def synth(voice, text, syn_config=None) -> tuple[np.ndarray, int]:
    """Concatenate every chunk of one synthesis call into a single waveform."""
    from phoonnx.config import SynthesisConfig

    parts, sr = [], None
    for chunk in voice.synthesize(text, syn_config=syn_config or SynthesisConfig()):
        parts.append(chunk.audio_float_array)
        sr = chunk.sample_rate
    if not parts:
        raise RuntimeError(f"no audio for {text!r}")
    return np.concatenate(parts).astype(np.float32), sr


# --- run --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="vc-run.json")
    ap.add_argument("--vc-engine", default="openvoice")
    ap.add_argument("--speaker-model", default="wespeaker-resnet34")
    ap.add_argument("--asr-model", default="nemo-parakeet-tdt-0.6b-v2")
    ap.add_argument("--audio-dir", default=None,
                    help="keep the generated WAVs here instead of a temp dir")
    args = ap.parse_args()

    import onnx_asr
    import speakeronnx
    from phoonnx.config import SynthesisConfig
    from phoonnx.version import VERSION_STR

    embedder = speakeronnx.SpeakerEmbedder(args.speaker_model)
    asr = onnx_asr.load_model(args.asr_model)

    workdir = Path(args.audio_dir) if args.audio_dir else Path(tempfile.mkdtemp(prefix="vc_gate_"))
    workdir.mkdir(parents=True, exist_ok=True)

    def embed(path: str) -> np.ndarray:
        return np.asarray(embedder.embed(path)).reshape(-1)

    def transcribe(path: str) -> str:
        return asr.recognize(path)

    # 1. Build the target-speaker reference clips.
    refs = {}
    for name, voice_id in TARGET_VOICES.items():
        print(f"[ref] {name} <- {voice_id}", flush=True)
        voice = load_voice(voice_id)
        audio, sr = synth(voice, TARGET_REF_TEXT)
        refs[name] = {
            "voice_id": voice_id,
            "path": write_wav(str(workdir / f"ref_{name}.wav"), audio, sr),
            "sample_rate": sr,
        }
        refs[name]["embedding"] = embed(refs[name]["path"])
        del voice

    # 2. The matrix.
    rows = []
    for source_id in SOURCE_VOICES:
        print(f"[source] {source_id}", flush=True)
        voice = load_voice(source_id)
        for u_idx, text in enumerate(UTTERANCES):
            base_audio, base_sr = synth(voice, text)
            base_path = write_wav(
                str(workdir / f"base_{u_idx}_{source_id.replace('/', '_')}.wav"),
                base_audio, base_sr)
            base_emb = embed(base_path)
            base_hyp = transcribe(base_path)
            base_wer = wer(text, base_hyp)

            for tgt_name, ref in refs.items():
                cfg = SynthesisConfig(vc_reference=ref["path"],
                                      vc_engine=args.vc_engine)
                t0 = time.time()
                conv_audio, conv_sr = synth(voice, text, cfg)
                elapsed = time.time() - t0
                conv_path = write_wav(
                    str(workdir /
                        f"conv_{u_idx}_{source_id.replace('/', '_')}_{tgt_name}.wav"),
                    conv_audio, conv_sr)
                conv_emb = embed(conv_path)
                conv_hyp = transcribe(conv_path)
                conv_wer = wer(text, conv_hyp)

                rows.append({
                    "utterance_index": u_idx,
                    "text": text,
                    "source_voice": source_id,
                    "target_speaker": tgt_name,
                    "target_voice": ref["voice_id"],
                    "sim_to_target": float(speakeronnx.cosine(conv_emb, ref["embedding"])),
                    "sim_to_source": float(speakeronnx.cosine(conv_emb, base_emb)),
                    "floor": float(speakeronnx.cosine(base_emb, ref["embedding"])),
                    "wer_before": base_wer,
                    "wer_after": conv_wer,
                    "asr_before": base_hyp,
                    "asr_after": conv_hyp,
                    "source_sample_rate": base_sr,
                    "converted_sample_rate": conv_sr,
                    "rtf_seconds": elapsed,
                    "audio_before": os.path.basename(base_path),
                    "audio_after": os.path.basename(conv_path),
                })
                print(f"  {source_id} -> {tgt_name} [{u_idx}] "
                      f"tgt={rows[-1]['sim_to_target']:.3f} "
                      f"src={rows[-1]['sim_to_source']:.3f} "
                      f"floor={rows[-1]['floor']:.3f} "
                      f"wer {base_wer:.1f} -> {conv_wer:.1f}", flush=True)
        del voice

    # 3. Gates.
    margins = [r["sim_to_target"] - r["floor"] for r in rows]
    pair_margins = {}
    for r in rows:
        pair_margins.setdefault((r["source_voice"], r["target_speaker"]), []).append(
            r["sim_to_target"] - r["floor"])
    pair_means = {f"{k[0]} -> {k[1]}": float(np.mean(v)) for k, v in pair_margins.items()}
    leakage = [r["sim_to_source"] for r in rows]
    wer_deltas = [r["wer_after"] - r["wer_before"] for r in rows]

    gates = {
        "similarity_beats_floor": {
            "pass": (all(m > 0 for m in pair_means.values())
                     and float(np.mean(margins)) >= SIM_MARGIN_GATE),
            "mean_margin": float(np.mean(margins)),
            "min_pair_mean_margin": float(min(pair_means.values())),
            "min_utterance_margin": float(np.min(margins)),
            "pair_mean_margins": pair_means,
            "threshold": SIM_MARGIN_GATE,
        },
        "no_source_leakage": {
            "pass": float(np.max(leakage)) <= LEAKAGE_GATE,
            "max_sim_to_source": float(np.max(leakage)),
            "mean_sim_to_source": float(np.mean(leakage)),
            "threshold": LEAKAGE_GATE,
        },
        "intelligibility_preserved": {
            "pass": float(np.mean(wer_deltas)) <= WER_DELTA_GATE,
            "mean_wer_delta_pp": float(np.mean(wer_deltas)),
            "max_wer_delta_pp": float(np.max(wer_deltas)),
            "threshold_pp": WER_DELTA_GATE,
        },
    }

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phoonnx_version": VERSION_STR,
        "voiceclonnx_version": _pkg_version("voiceclonnx"),
        "speakeronnx_version": _pkg_version("speakeronnx"),
        "onnx_asr_version": _pkg_version("onnx-asr"),
        "vc_engine": args.vc_engine,
        "speaker_model": args.speaker_model,
        "asr_model": args.asr_model,
        "n_utterances": len(UTTERANCES),
        "source_voices": SOURCE_VOICES,
        "target_voices": TARGET_VOICES,
        "rows": rows,
        "gates": gates,
        "pass": all(g["pass"] for g in gates.values()),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    _write_markdown(report, out.with_suffix(".md"))

    print(json.dumps(gates, indent=2))
    print(f"\nwrote {out} and {out.with_suffix('.md')}")
    if not args.audio_dir:
        print(f"(audio in {workdir} — delete when done)")
    return 0 if report["pass"] else 1


def _write_markdown(report: dict, path: Path) -> None:
    rows = report["rows"]
    lines = [
        f"# Voice-conversion gate — {report['vc_engine']}",
        "",
        f"phoonnx {report['phoonnx_version']} · voiceclonnx {report['voiceclonnx_version']} · "
        f"speaker model {report['speaker_model']} · ASR {report['asr_model']}",
        f"Generated {report['generated_utc']}",
        "",
        "## Speaker similarity (cosine)",
        "",
        "| source voice | target | sim to target | sim to source | floor (no VC) | margin |",
        "|---|---|---|---|---|---|",
    ]
    by_pair: dict = {}
    for r in rows:
        by_pair.setdefault((r["source_voice"], r["target_speaker"]), []).append(r)
    for (src, tgt), rs in by_pair.items():
        lines.append(
            f"| `{src}` | {tgt} | {np.mean([r['sim_to_target'] for r in rs]):.3f} "
            f"| {np.mean([r['sim_to_source'] for r in rs]):.3f} "
            f"| {np.mean([r['floor'] for r in rs]):.3f} "
            f"| **{np.mean([r['sim_to_target'] - r['floor'] for r in rs]):+.3f}** |")
    lines += [
        "",
        "## Intelligibility (WER %, parakeet)",
        "",
        "| source voice | target | WER before | WER after | delta (pp) |",
        "|---|---|---|---|---|",
    ]
    for (src, tgt), rs in by_pair.items():
        b = np.mean([r["wer_before"] for r in rs])
        a = np.mean([r["wer_after"] for r in rs])
        lines.append(f"| `{src}` | {tgt} | {b:.1f} | {a:.1f} | {a - b:+.1f} |")
    lines += ["", "## Gates", ""]
    for name, g in report["gates"].items():
        lines.append(f"- **{name}**: {'PASS' if g['pass'] else 'FAIL'} — "
                     + ", ".join(f"{k}={v}" for k, v in g.items() if k != "pass"))
    lines += ["", f"Overall: **{'PASS' if report['pass'] else 'FAIL'}**", ""]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
