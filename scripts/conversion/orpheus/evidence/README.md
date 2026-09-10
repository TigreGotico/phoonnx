# Evidence provenance

Honest account of what ran where, split into the original PR-authoring run and this
reconstruction.

## Original run (produced the PR body's tables and `wer.json`)

The PR body's RTF table (mean 38.7x realtime), parity table (fp32 25/25 greedy
agreement, 0.166 max prefill logit diff; `model_q4` 23/25, `model_q4f16` 10/25), SNAC
noise-floor table (fp32 0.89x the floor; int8 6.02x; uint8/quantized 4.83x) and the WER
table (mean 0.0000 over six tag-free utterances) were produced during the original
conversion/verification work that authored this PR and the
[`OpenVoiceOS/phoonnx-orpheus`](https://huggingface.co/OpenVoiceOS/phoonnx-orpheus)
mirror.

- **Machine**: a `.200`-class CPU worker (12 cores, fp32) — the box the RTF numbers in
  the PR body are stated against. The exact hostname and run timestamp were not
  preserved; the scratch directory that held the run's logs and the original
  `scripts/conversion/orpheus/` tooling (`.200` scratch) was deleted before this PR's
  documentation citation was checked against what's actually committed — the gap this
  `scripts/conversion/orpheus/` directory now closes.
- **Reference repos used**:
  - LM: [`unsloth/orpheus-3b-0.1-ft`](https://huggingface.co/unsloth/orpheus-3b-0.1-ft)
    (ungated copy of the gated `canopylabs/orpheus-3b-0.1-ft`).
  - ONNX candidates: [`onnx-community/orpheus-3b-0.1-ft-ONNX`](https://huggingface.co/onnx-community/orpheus-3b-0.1-ft-ONNX)
    (`model`, `model_q4`, `model_q4f16`).
  - SNAC: [`hubertsiuzdak/snac_24khz`](https://huggingface.co/hubertsiuzdak/snac_24khz)
    torch, vs. [`onnx-community/snac_24khz-ONNX`](https://huggingface.co/onnx-community/snac_24khz-ONNX)
    (`decoder_model` fp32/fp16/int8/uint8/q4/bnb4).
  - ASR for the WER gate: [`OpenVoiceOS/qwen3-asr-0.6b-onnx`](https://huggingface.co/OpenVoiceOS/qwen3-asr-0.6b-onnx).
  - **The gated `canopylabs` originals were never downloaded** — all 7 research repos
    plus `orpheus-3b-0.1-ft` return 403 (gating is `"auto"`/accept-terms, and accepting
    a licence agreement on someone else's behalf is not this agent's call to make). All
    parity work used the ungated third-party mirrors above.
- **Output**: `wer.json` in this directory is fetched verbatim from
  `OpenVoiceOS/phoonnx-orpheus/samples/wer.json` — the original run's own output file,
  re-hosted alongside the mirrored weights — not regenerated. Its numbers match the PR
  body's WER/RTF table exactly (mean WER 0.0000, mean RTF ~38.7x).

## This reconstruction (2026-08-04)

The scripts in `scripts/conversion/orpheus/` (`probe_prompt.py`, `verify_parity.py`,
`verify_snac.py`, `bench_wer.py`) did not exist as committed files before this PR — the
original tooling lived in `.200` scratch and was deleted. They are reconstructed here,
post hoc, from the methodology the PR body and module docstring already describe, built
to the house standard (`scripts/conversion/{arktts,qwen3tts}/`): runnable, self-gating,
not stubs. This is stated honestly rather than presented as if it were the original
run's untouched tooling.

- **`probe_prompt.py` was actually run** against `tokenizer.json` fetched from
  `OpenVoiceOS/phoonnx-orpheus/orpheus-3b-en-onnx/tokenizer.json` (15.7 MB, the only
  artifact this check needs) on this machine, `miro-asustufgamingf15fx506hmfx506hm`,
  2026-08-04T18:50Z, Python 3.12.13. Output committed at `probe_prompt_output.txt`.
  Result: **double-BOS confirmed** — `served_ids` reproduces
  `[128000, 128259, 128000, 83, ...]` against the real tokenizer, matching the PR body's
  quoted trace exactly, and `OrpheusAdapter.build_prompt_ids`'s own construction matches
  the served form byte-for-byte.
- **`verify_parity.py` and `verify_snac.py` were not run** — they need the multi-GB
  torch and ONNX weights, which this laptop-light task explicitly excludes downloading.
  Their value is the self-gating logic (exit non-zero on parity/floor-ratio failure),
  reconstructed faithfully from the numbers and methodology the PR body already reports,
  so a future run — laptop-heavy or on a GPU box — can reproduce and re-gate the
  original findings rather than take them on faith.
- **`bench_wer.py` was not run** — same reason; `wer.json` is fetched from the mirror
  instead of regenerated (see above).
