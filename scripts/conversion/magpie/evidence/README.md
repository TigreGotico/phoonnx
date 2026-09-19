# Magpie-TTS WER smoke gate

Intelligibility **smoke gate** for the Magpie-TTS engine (PR #362), covering the five
languages that shipped voices at scan time: French, Italian, Vietnamese, Arabic, Korean.

This is a regression tripwire, not a benchmark. Five sentences per language is enough to
catch a broken pipeline (garbled audio, wrong language routing, a codec desync) — it is
not enough to characterize the engine's real-world word error rate, and the numbers below
should not be quoted as such. Anyone wanting a benchmark-grade WER should run the standard
per-language eval sets (e.g. Common Voice, FLEURS) through the same `run_asr.py` scoring
path with a much larger sample.

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

Smoke-gate numbers, 5 clips per language — read as "did the pipeline break," not as a WER
benchmark (see framing note above).

| Lang | WER | CER | Clips |
|---|---|---|---|
| fr | 0.108 | 0.019 | 5 |
| it | 0.061 | 0.005 | 5 |
| vi | 0.089 | 0.039 | 5 |
| ar | 0.103 | 0.031 | 5 |

| Lang | WER | CER | Clips | Methodology |
|---|---|---|---|---|
| ko | 0.000 | 0.000 | 5 | **Whisper fallback — not comparable to the rows above.** Scored with a general-purpose ASR (Whisper-large-v3-turbo), not the calibrated in-house OVOS ONNX ASR pipeline the other four languages use. Treat 0.000 as "intelligible to a general-purpose ASR," not as a WER measured on the same footing as fr/it/vi/ar. |

All five languages pass the 30% smoke-gate threshold with wide margin; this confirms the
pipeline produces intelligible speech, not that the engine has been benchmarked.

## Codebook/frame ordering: validated against real NeMo output, not just self-consistent

The WER smoke gate above only proves the *output* is intelligible speech. It says nothing
about whether the ONNX pipeline's codebook and frame-stacking order matches what the
original NeMo checkpoint actually produces — a self-consistent-but-wrong ordering could
still decode to some audio and pass an ASR check by accident.

That ordering claim is validated separately, by `parity.py`, against ground truth: it runs
the *same* input through the real NeMo PyTorch checkpoint (`MagpieTTSModel.restore_from`)
and through the ONNX pipeline, both with greedy decoding (temperature 0.01, top-k 1, fixed
seed) so the two runs are deterministic and comparable token-for-token, and measures exact
per-frame, per-codebook agreement between the two code streams. It gates at 99% agreement
(`MAGPIE_PARITY_THRESHOLD`, default 0.99) and is run in **both** decoder modes NeMo
supports (`use_kv_cache_for_inference` on and off — see `parity.py`'s `kv` argument),
because the two modes are documented in `phoonnx/engines/magpie.py` to produce different
sample paths (a KV cache re-applies each step's attention prior only to new positions,
while the no-cache default re-applies it to the whole history), so agreement had to be
checked against real NeMo in both, not assumed to transfer from one to the other.

That 100% (well above the 99% gate) greedy-token agreement against the real NeMo torch
model, in both decoder modes, is the ground truth behind the codebook/frame ordering used
throughout this engine — it is not inferred from the ONNX outputs being self-consistent
with each other.
