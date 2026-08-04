# Magpie-TTS WER-gate evidence

Intelligibility evidence for the Magpie-TTS engine (PR #362), covering the five
languages that shipped voices for at scan time: French, Italian, Vietnamese, Arabic,
Korean.

## Provenance

- **Date:** 2026-08-02
- **Machine:** laptop (CPU-only, single process; not the `.200` GPU box — this is an
  intelligibility smoke check, not a throughput benchmark)
- **Engine under test:** `phoonnx` Magpie-TTS engine, branch `feat/magpie-tts`
- **Method:** synthesize each sentence with the real engine (`synth.py`), transcribe the
  resulting WAV with a CPU ONNX ASR model per language (`run_asr.py`), score WER with
  `jiwer` against the (normalized) source sentence.

## ASR models used

| Lang | Model | Notes |
|---|---|---|
| fr | `OpenVoiceOS/nvidia-fr-conformer-transducer-large-onnx` | in-house OVOS ONNX ASR |
| it | `OpenVoiceOS/nvidia-it-conformer-transducer-large-onnx` | in-house OVOS ONNX ASR |
| vi | `OpenVoiceOS/nvidia-parakeet-ctc-0.6b-vietnamese-onnx` | in-house OVOS ONNX ASR |
| ar | `OpenVoiceOS/stt_ar_fastconformer_hybrid_large_pc_v1.0_onnx` | in-house OVOS ONNX ASR |
| ko | `onnx-community/whisper-large-v3-turbo` | **fallback, not an OVOS-org model** |

**Korean ASR caveat.** The OpenVoiceOS org does not ship a Korean ASR model — none of the
models in that org cover `ko`. `ko` is therefore scored with Whisper-large-v3-turbo as an
explicit best-effort fallback. This is labeled deliberately: the ko WER (0.0 on this run)
should be read as "intelligible to a general-purpose ASR," not as a measurement against
the same calibrated pipeline the other four languages use. If OVOS ever mirrors a Korean
ONNX ASR model, this gate should switch to it.

**Vietnamese diacritics correction.** The vi reference sentences in `sentences.json` were
hand-corrected for diacritics before this run — Vietnamese distinguishes words by tone
marks that an earlier draft of the sentence list dropped or mis-typed. The committed
`sentences.json` already contains the corrected forms; the refs recorded in
`results.json` are the normalized (lowercased, punctuation-stripped) versions of those
corrected sentences, not the original draft.

## Running it

```bash
cd scripts/conversion/magpie/evidence
python synth.py            # writes ./wav/<lang>_<i>.wav using the real engine
python run_asr.py          # scores WER per language, writes results.json, gates
```

`run_asr.py` exits non-zero if any language's WER exceeds `--threshold` (default 0.30 —
30%). This is a self-gating script: a regression that pushes a language's WER over the
threshold fails the run instead of only showing up as a smaller number in a JSON file.

## Files

- `sentences.json` — the five source sentences per language (fr/it/vi/ar/ko)
- `synth.py` — synthesizes each sentence with the Magpie-TTS engine, writes WAVs
- `run_asr.py` — transcribes the WAVs, scores WER/CER, writes `results.json`, gates at 30%
- `results.json` — the committed raw ref/hyp pairs and per-language WER/CER from the
  2026-08-02 run (the `wav/` directory itself is not committed — regenerate it with
  `synth.py` to reproduce)

## Results (2026-08-02 run)

| Lang | WER | CER | Clips |
|---|---|---|---|
| fr | 0.108 | 0.019 | 5 |
| it | 0.061 | 0.005 | 5 |
| vi | 0.089 | 0.039 | 5 |
| ar | 0.103 | 0.031 | 5 |
| ko | 0.000 | 0.000 | 5 (Whisper fallback, see caveat above) |

All five languages pass the 30% WER gate with wide margin.
