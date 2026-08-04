# Orpheus conversion & verification tooling

There is no `export_orpheus_onnx.py` in this directory: the ONNX graphs mirrored at
[`OpenVoiceOS/phoonnx-orpheus`](https://huggingface.co/OpenVoiceOS/phoonnx-orpheus) are
re-hosted from [`onnx-community/orpheus-3b-0.1-ft-ONNX`](https://huggingface.co/onnx-community/orpheus-3b-0.1-ft-ONNX)
(LM) and [`onnx-community/snac_24khz-ONNX`](https://huggingface.co/onnx-community/snac_24khz-ONNX)
(SNAC decoder), unmodified — nothing here re-exports them. What this directory holds is
the *verification* that those community exports are safe to serve through this adapter,
and the probe that pinned the adapter's most surprising design decision.

```bash
pip install tokenizers                                    # probe_prompt.py only
python probe_prompt.py --tokenizer ./tokenizer.json        # no weights needed

pip install torch transformers onnxruntime                # verify_parity.py
python verify_parity.py --onnx ./model.onnx --tokenizer ./tokenizer.json

pip install torch snac onnxruntime                         # verify_snac.py
python verify_snac.py --onnx ./snac_decoder.onnx

pip install onnx_asr soundfile                              # bench_wer.py
python bench_wer.py --onnx ./orpheus-3b-en-onnx --out ./evidence/wer.json
```

## `probe_prompt.py` — the double-BOS finding

Builds the served prompt two ways — literal source reading vs. upstream's actual
decode-then-re-tokenize round trip — and diffs the token ids. Needs only
`tokenizer.json`, no weights, runs in under a second. This is the script the module
docstring in `phoonnx/engines/orpheus.py` and `OrpheusAdapter.build_prompt_ids` cite.
Its output for this PR is committed at `evidence/probe_prompt_output.txt`.

## `verify_parity.py` — LM prefill/greedy agreement

Greedy decode, both sides (torch reference vs. the ONNX graph through the phoonnx
adapter) driven from the same served prompt. Self-gating: exits non-zero unless greedy
agreement is 100% and the prefill logit diff stays under threshold. Needs the ~6-13 GB
torch reference and the ONNX graph — not run for this PR (laptop-light constraint); its
gating logic, not a fresh run, is the deliverable. See `evidence/README.md` for where
the quoted numbers (25/25 greedy agreement, 0.166 max prefill diff for fp32; rejections
for both int4 variants) came from.

## `verify_snac.py` — SNAC's stochastic decoder, ranked correctly

SNAC's decoder has a noise-injection block, so two torch decodes of the same codes
already differ (`hubertsiuzdak/snac_24khz`). A plain torch-vs-onnx diff cannot tell a
bad export from that expected spread, so this script first estimates the model's own
run-to-run noise floor (N torch decodes vs. their mean) and then reports the ONNX
candidate's distance as a **ratio to that floor**. Self-gating on the ratio. Needs the
torch SNAC checkpoint and the ONNX decoder — not run for this PR; see `evidence/README.md`.

## `bench_wer.py` — intelligibility + RTF

Synthesizes the fixed probe set (3 voices x 2 sentences + one `<laugh>` smoke test)
through the live adapter, times the LM loop for RTF, and transcribes with
[`OpenVoiceOS/qwen3-asr-0.6b-onnx`](https://huggingface.co/OpenVoiceOS/qwen3-asr-0.6b-onnx)
(never an unlabeled whisper, per house policy). Needs the ~13 GB mirrored graphs and the
ASR model — not run for this PR; `evidence/wer.json` is the original run's output,
fetched from the mirror rather than regenerated (see `evidence/README.md`).

## `evidence/`

- `wer.json` — fetched from `OpenVoiceOS/phoonnx-orpheus/samples/wer.json`, the original
  run's WER/RTF numbers quoted in the PR body and mirror README.
- `probe_prompt_output.txt` — this PR's actual, freshly reproduced `probe_prompt.py` run
  against the mirrored `tokenizer.json`.
- `README.md` — provenance: what ran where, what didn't run and why, what was
  reconstructed after the fact.

Source model: [`canopylabs/orpheus-3b-0.1-ft`](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft)
(gated; verification used the ungated [`unsloth/orpheus-3b-0.1-ft`](https://huggingface.co/unsloth/orpheus-3b-0.1-ft)
mirror instead), Apache-2.0, Canopy Labs.
SNAC: [`hubertsiuzdak/snac_24khz`](https://huggingface.co/hubertsiuzdak/snac_24khz), Apache-2.0.
Mirror: [`OpenVoiceOS/phoonnx-orpheus`](https://huggingface.co/OpenVoiceOS/phoonnx-orpheus).
