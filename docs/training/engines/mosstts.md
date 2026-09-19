# MOSS-TTS-Nano — training and finetuning

MOSS-TTS-Nano is a ~100M-parameter autoregressive **codec LM**: a 12-layer GPT-2-style
backbone with RoPE predicts frames of RVQ-16 audio tokens, which the frozen
MOSS-Audio-Tokenizer-Nano decodes to 48 kHz stereo audio. It clones a voice zero-shot from
a reference clip, with no speaker table and no transcript of the reference.

The pipeline lives in `phoonnx_train/mosstts/` and is fully vendored: the architecture is
implemented in-repo, no `transformers`, no `trust_remote_code`, no upstream package at
runtime.

## Which scheme this implements

The MOSS-TTS technical report (arXiv:2603.18090) describes two ways to model the 16 RVQ
channels. Nano uses the **global-local (RQ-Transformer) scheme**, and that is what this
trainer implements:

* the **global** transformer consumes one row of `1 + n_vq = 17` token ids per frame —
  column 0 is a text or slot token, columns 1..16 are the codebook ids — sums their
  embeddings, and emits one hidden state per frame;
* a 1-layer **local** transformer then walks the `n_vq + 1 = 17` channels of that frame:
  position 0 is the global hidden state, position 1 is the text target's embedding,
  positions 2..16 are the embeddings of codebook channels 0..14. Its outputs feed
  `text_lm_head` (position 0) and `audio_lm_heads[c]` (position `c + 1`).

There is **no delay pattern**: channels stay time-aligned, and no per-codebook loss
schedule is applied.

## Data format

Training consumes a JSONL where the audio has already been turned into codec tokens. The
codec is frozen and never part of training — it runs exactly once, offline, through the
published ONNX encode graph.

```json
{"audio": "wavs/0001.wav", "text": "Bom dia.", "language": "pt",
 "ref_audio": "wavs/ref.wav",
 "audio_codes": [[c0, ..., c15], ...],
 "ref_audio_codes": [[c0, ..., c15], ...]}
```

`instruction`, `tokens`, `quality`, `sound_event`, `ambient_sound` and `language` are
optional metadata lines of the chat template; anything missing renders as `None`.

Build it from a manifest — either a JSONL with `audio` / `text`, or an LJSpeech-style
`metadata.csv` plus `--wav-dir`:

```bash
python -m phoonnx_train.mosstts.prepare_data \
  --codec-encode-onnx models/moss_audio_tokenizer_encode.onnx \
  --input-manifest data/metadata.csv --wav-dir data/wavs \
  --output-jsonl data/train.codes.jsonl
```

Audio is resampled to 48 kHz and duplicated to stereo automatically. `--n-vq N` keeps only
the first `N` codebook layers, for models configured with a narrower stack.

Each record is packed into a `[T, 17]` row matrix — instruction prefix, optional reference
frames (user slot), text, assistant prefix, then the target frames (assistant slot) and a
closing `audio_end`. Labels are that matrix shifted left by one, with everything before the
assistant turn masked to `-100`, so the loss only sees the audio the model must generate.

## Finetuning from the released weights

```bash
python -m phoonnx_train.mosstts.train \
  --train-jsonl data/train.codes.jsonl \
  --tokenizer-model models/MOSS-TTS-Nano/tokenizer.model \
  --config models/MOSS-TTS-Nano/config.json \
  --warm-start-from models/MOSS-TTS-Nano \
  --output-dir runs/moss-pt \
  --max-steps 20000 --batch-size 1 --max-length 1024
```

`--warm-start-from` accepts an upstream checkpoint directory (`config.json` +
`pytorch_model.bin` or `model.safetensors`), a single weights file, or a previous phoonnx
`.ckpt`. Weights are copied through an explicit key mapping and the matched-parameter
fraction is logged:

```
warm start: .../pytorch_model.bin: matched 194 tensors / 142,477,056 of 142,477,056
parameters (100.00%), missing=0 unexpected=0 shape_mismatch=0
```

Anything below 100% means part of the model is still at its random initialisation — treat
it as a failure, not a warning.

To check coverage without training:

```bash
python -m phoonnx_train.mosstts.warmstart --checkpoint models/MOSS-TTS-Nano
```

## Resuming

`--warm-start-from` copies weights only; the optimizer and LR schedule start fresh. To
continue an interrupted run *exactly* — optimizer moments, scheduler position and global
step included — use `--resume-from`:

```bash
python -m phoonnx_train.mosstts.train ... --resume-from runs/moss-pt/last.ckpt
```

The two flags are mutually exclusive.

## Objective and defaults

Loss is the weighted mean of per-head cross-entropies:
`sum(w_i * CE_i) / sum(w_i)` over the text head and the 16 audio heads, all teacher-forced
through the local transformer. `--channelwise-loss-weight` accepts either 17 explicit
weights or the shorthand `text,total_audio` — the default `1,32` gives the text head 1.0
and splits 32 evenly across the 16 audio heads (2.0 each).

Optimizer defaults match upstream: AdamW `lr=1e-5`, `betas=(0.9, 0.95)`, `eps=1e-8`,
`weight_decay=0.1`, linear decay with a 3% warmup ratio, gradient-norm clip 1.0. Upstream
measured ~3.23 GiB peak VRAM at batch size 1, sequence length 1024, bf16.

`--prompt-style` selects the chat template. The default, `inference`, is the one the
checkpoint's own `prompting.py` (and `phoonnx.engines.mosstts`) builds, including the
`im_end` token that closes the user turn. `finetuning` reproduces upstream's
`finetuning/dataset.py`, which omits that token — kept for bit-comparison with upstream
runs, but it desynchronises the finetune from the prompt used at synthesis time.

## Export

```bash
python -m phoonnx_train.mosstts.export_onnx \
  --checkpoint runs/moss-pt/last.ckpt \
  --output-dir exported/moss-pt \
  --external-data \
  --tokenizer-model models/MOSS-TTS-Nano/tokenizer.model
```

This writes the same multi-graph layout the runtime engine loads:

| file | role |
|---|---|
| `moss_tts_prefill.onnx` | prompt rows → `global_hidden` + 12 present KV pairs |
| `moss_tts_decode_step.onnx` | one row + past KV → `global_hidden` + present KV |
| `moss_tts_local_fixed_sampled_frame.onnx` | `global_hidden` → `should_continue` + 16 sampled tokens (sampling baked in at the upstream defaults) |
| `moss_tts_local_cached_step.onnx` | per-channel local step with its own KV cache, for host-side sampling |
| `moss_tts_local_decoder.onnx` | whole-frame local pass with a teacher-forced audio prefix |
| `tts_browser_onnx_meta.json` | file map, model geometry, baked sampling constants |

`--external-data` de-duplicates the weights into `moss_tts_global_shared.data` and
`moss_tts_local_shared.data`, as the released export does. Drop the flag for a small model
where inlined weights are simpler.

The exported graphs are validated against the live PyTorch module in
`tests/test_mosstts_export.py`, including the full
`prefill → local frame → decode_step` loop.

## Licensing

The upstream repository (github.com/OpenMOSS/MOSS-TTS-Nano) now carries an Apache-2.0
`LICENSE` file at its root, matching the Hugging Face model cards. This pipeline is an
independent implementation guided by the published architecture and the upstream reference
code; no upstream source file is copied.
