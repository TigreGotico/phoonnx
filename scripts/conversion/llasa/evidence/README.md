# Evidence provenance

This directory holds the durable measurement artifacts backing the claims in PR #366
("feat: Llasa engine (XCodec2 AR codec-LM)"). None of it is reproduced automatically by
CI — it documents what was actually run, when, and on what.

## What is here

* `wer.json` — the ONNX-side intelligibility/speed gate (`bench_wer.py`, no
  `--torch-floor`): 8 English + 8 Mandarin sentences synthesised through
  `TTSVoice`/`LlasaAdapter` on the **published mirror**
  (`https://huggingface.co/OpenVoiceOS/phoonnx-llasa`), transcribed by
  `nemo-parakeet-tdt-0.6b-v3` (English) and `OpenVoiceOS/omnilingual-asr-ctc-1b-onnx`
  (Mandarin).

## Torch-parity numbers: not re-run here

The torch-vs-ONNX parity tables in the PR body (`parity_llm.py` / `parity_codec.py`,
48-step greedy agreement, logit diffs, codec sample diffs) come from the **original
export run** and are not reproduced as a committed JSON in this directory. Re-running
full torch parity needs both export environments described in
`scripts/conversion/llasa/README.md` (a `transformers` 5.x env for the LM, a pinned
`transformers==4.47.1` + `torchao==0.8.0` env for the codec) plus the upstream fp32
checkpoints, and is cost-prohibitive to repeat on every review pass.

Instead, `parity_llm.py` and `parity_codec.py` are now **self-gating**: both exit
non-zero when greedy agreement drops below 100% or a measured diff exceeds a fixed
tolerance (`LOGIT_DIFF_TOLERANCE = 1e-3` for the LM, `MAX_ABS_DIFF_TOLERANCE = 1e-2` /
`MIN_CORRELATION = 0.999` for the codec — both roughly two orders of magnitude of
headroom over the noise floor actually measured on the shipped export, per the PR
body's tables). Any future re-export that regresses parity fails the script itself
instead of relying on a human reading a printed diff.

## Provenance of `wer.json`

| Field | Value |
|---|---|
| Date | 2026-08-04 |
| Machine | ser9 (192.168.1.111) |
| phoonnx commit | `feat/llasa` @ the commit that added this README |
| Model mirror | `OpenVoiceOS/phoonnx-llasa` (fp32 `model.onnx` + `xcodec2_decoder.onnx`) |
| Voices used | `en_female_a` (English), `zh_male_a` (Mandarin) |
| ASR (en) | `nemo-parakeet-tdt-0.6b-v3` via `onnx_asr` |
| ASR (zh) | `OpenVoiceOS/omnilingual-asr-ctc-1b-onnx` via `onnx_asr` (`omnilingual-ctc`,
  needs the TigreGotico `onnx-asr` fork's `integration` branch — upstream PyPI
  `onnx-asr` does not have this recognizer) |
| Command | `python bench_wer.py --bundle <mirror dir> --output wer.json` (no
  `--torch-floor`: only the ONNX side was re-measured; see above) |

This is the same 8×2 sentence set and the same two recognisers used for the numbers in
the PR body, so `wer.json`'s `onnx_en` / `onnx_zh` rows are directly comparable to the
"phoonnx ONNX" column of the PR's intelligibility table. Any material drift between
this file and that table across future re-runs is a real regression signal, not noise.
