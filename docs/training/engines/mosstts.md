# MOSS-TTS-Nano: training and fine-tuning

<!-- agent-hint
covers: how to prepare data for, fine-tune, resume and export MOSS-TTS-Nano
source: phoonnx_train/mosstts/config.py, phoonnx_train/mosstts/dataset.py, phoonnx_train/mosstts/prepare_data.py, phoonnx_train/mosstts/train.py, phoonnx_train/mosstts/warmstart.py, phoonnx_train/mosstts/export_onnx.py
generated: no
verify: ! comm -23 <(grep -oP -- '(?<![\w-])--[a-z][a-z0-9-]*' docs/training/engines/mosstts.md | sort -u) <(for m in prepare_data train export_onnx warmstart; do .venv/bin/python -m phoonnx_train.mosstts.$m --help; done | grep -oP -- '(?<![\w-])--[a-z][a-z0-9-]*' | sort -u) | grep . && .venv/bin/python -W ignore -c "import re, sys; from phoonnx_train.mosstts.train import build_argparser; d = {a.option_strings[0]: a.default for a in build_argparser()._actions if a.option_strings}; p = open('docs/training/engines/mosstts.md').read(); table = p.split(chr(10) + '| Flag | Default |')[1].split(chr(10) + chr(10))[0]; rows = re.findall(r'^\| \x60(--[a-z0-9-]+)\x60 \| \x60([^\x60]+)\x60 \|', table, re.M); bad = [(f, v, d.get(f)) for f, v in rows if str(d.get(f)) != v]; print(len(rows), bad); sys.exit(bool(bad) or not rows)"
see: ../training.md, ../../cloning.md
-->

This page shows how to prepare data for MOSS-TTS-Nano, fine-tune it, resume a run and export the result, with the pipeline in `phoonnx_train/mosstts/`.

MOSS-TTS-Nano is an autoregressive codec language model. A 12-layer GPT-2-style backbone with RoPE predicts frames of 16 RVQ audio tokens, and the frozen MOSS-Audio-Tokenizer-Nano decodes them to 48 kHz stereo audio. With the default configuration, the model has about 117 million parameters. It clones a voice zero-shot from a reference clip, with no speaker table and no transcript of the reference.

The code in `phoonnx_train/mosstts/` implements the architecture itself. It needs no `transformers`, no `trust_remote_code` and no upstream package.

## 1. Prepare the data

Training reads a JSONL file in which the audio is already codec tokens. The codec is frozen and is not part of training. It runs once, offline, through the published ONNX encode graph.

Build the file from a manifest. The manifest is a JSONL file with `audio` and `text`, or an LJSpeech `metadata.csv` with `--wav-dir`:

```bash
python -m phoonnx_train.mosstts.prepare_data \
  --codec-encode-onnx models/moss_audio_tokenizer_encode.onnx \
  --input-manifest data/metadata.csv --wav-dir data/wavs \
  --output-jsonl data/train.codes.jsonl
```

The command resamples the audio to 48 kHz and makes it stereo. `--sample-rate` and `--channels` change these values. `--n-vq N` keeps only the first `N` codebook layers, for a model with fewer layers.

Each line of the output has this form:

```json
{"audio": "wavs/0001.wav", "text": "Bom dia.", "language": "pt",
 "ref_audio": "wavs/ref.wav",
 "audio_codes": [[c0, ..., c15], ...],
 "ref_audio_codes": [[c0, ..., c15], ...]}
```

`instruction`, `tokens`, `quality`, `sound_event`, `ambient_sound` and `language` are optional metadata lines of the chat template. A missing value becomes `None`.

## 2. Fine-tune from the released weights

```bash
python -m phoonnx_train.mosstts.train \
  --train-jsonl data/train.codes.jsonl \
  --tokenizer-model models/MOSS-TTS-Nano/tokenizer.model \
  --config models/MOSS-TTS-Nano/config.json \
  --warm-start-from models/MOSS-TTS-Nano \
  --output-dir runs/moss-pt \
  --max-steps 20000 --batch-size 1 --max-length 1024
```

`--warm-start-from` takes an upstream checkpoint folder (`config.json` with `pytorch_model.bin` or `model.safetensors`), a single weights file, or an earlier phoonnx `.ckpt`. The command copies the weights through an explicit key map. It logs the number of tensors and parameters that matched, the percentage, and the counts of missing, unexpected and wrong-shape keys. A value below 100 percent means that part of the model keeps its random start values. Treat it as a failure.

To check the coverage without training:

```bash
python -m phoonnx_train.mosstts.warmstart --checkpoint models/MOSS-TTS-Nano
```

## 3. Resume a run

`--warm-start-from` copies weights only, so the optimizer and the schedule start again. To continue an interrupted run with its optimizer moments, scheduler position and global step, use `--resume-from`:

```bash
python -m phoonnx_train.mosstts.train ... --resume-from runs/moss-pt/last.ckpt
```

You cannot give both flags.

## 4. Export

```bash
python -m phoonnx_train.mosstts.export_onnx \
  --checkpoint runs/moss-pt/last.ckpt \
  --output-dir exported/moss-pt \
  --external-data \
  --tokenizer-model models/MOSS-TTS-Nano/tokenizer.model
```

This writes the graphs that the runtime engine loads:

| File | Role |
|---|---|
| `moss_tts_prefill.onnx` | Prompt rows to `global_hidden` and 12 present KV pairs |
| `moss_tts_decode_step.onnx` | One row and the past KV to `global_hidden` and the present KV |
| `moss_tts_local_fixed_sampled_frame.onnx` | `global_hidden` to `should_continue` and 16 sampled tokens, with sampling at the upstream defaults |
| `moss_tts_local_cached_step.onnx` | A local step for each channel with its own KV cache, for sampling in the host |
| `moss_tts_local_decoder.onnx` | A local pass over the whole frame with a teacher-forced audio prefix |
| `tts_browser_onnx_meta.json` | The file map, the model geometry and the sampling constants |

`--external-data` puts the shared weights in `moss_tts_global_shared.data` and `moss_tts_local_shared.data`, as the released export does. For a small model, leave out the flag to keep the weights in the graphs.

`tests/test_mosstts_export.py` compares the exported graphs with the PyTorch module, and runs the full `prefill`, local frame and `decode_step` loop.

## Which scheme this implements

The MOSS-TTS technical report (arXiv:2603.18090) describes two ways to model the 16 RVQ channels. Nano uses the global-local (RQ-Transformer) scheme, and this trainer implements that scheme.

The global transformer reads one row of `1 + n_vq = 17` token ids per frame. Column 0 is a text or slot token, and columns 1 to 16 are the codebook ids. It adds their embeddings and gives one hidden state per frame.

A one-layer local transformer then reads the `n_vq + 1 = 17` channels of that frame. Position 0 is the global hidden state, position 1 is the embedding of the text target, and positions 2 to 16 are the embeddings of codebook channels 0 to 14. Its outputs go to `text_lm_head` (position 0) and `audio_lm_heads[c]` (position `c + 1`).

There is no delay pattern. The channels stay aligned in time, and no loss schedule per codebook applies.

Each record becomes a `[T, 17]` row matrix: the instruction prefix, the optional reference frames (user slot), the text, the assistant prefix, the target frames (assistant slot) and a closing `audio_end`. The labels are that matrix shifted left by one, with everything before the assistant turn masked to `-100`. The loss thus sees only the audio that the model must generate.

## Objective and defaults

The loss is the weighted mean of the cross-entropies of the heads, `sum(w_i * CE_i) / sum(w_i)`, over the text head and the 16 audio heads. All heads are teacher-forced through the local transformer. `--channelwise-loss-weight` takes 17 weights, or the short form `text,total_audio`. The default `1,32` gives the text head 1.0 and splits 32 evenly over the 16 audio heads, 2.0 each.

The optimizer is AdamW with a linear schedule. These are the defaults of `phoonnx_train.mosstts.train`:

| Flag | Default |
|---|---|
| `--learning-rate` | `1e-05` |
| `--adam-beta1` | `0.9` |
| `--adam-beta2` | `0.95` |
| `--adam-eps` | `1e-08` |
| `--weight-decay` | `0.1` |
| `--warmup-ratio` | `0.03` |
| `--lr-scheduler-type` | `linear` |
| `--max-grad-norm` | `1.0` |
| `--channelwise-loss-weight` | `1,32` |
| `--prompt-style` | `inference` |
| `--batch-size` | `1` |
| `--max-length` | `1024` |
| `--max-epochs` | `3` |

`--prompt-style` selects the chat template. The default, `inference`, is the template that the checkpoint's own `prompting.py` and `phoonnx.engines.mosstts` build, with the `im_end` token that closes the user turn. `finetuning` gives the template of upstream `finetuning/dataset.py`, which has no such token. It exists for comparison with upstream runs, but the fine-tuned model then sees a prompt that differs from the prompt at synthesis.

## Licensing

The upstream repository (github.com/OpenMOSS/MOSS-TTS-Nano) has an Apache-2.0 `LICENSE` file at its root, as the Hugging Face model cards state. This pipeline is an independent implementation that follows the published architecture and the upstream reference code. It copies no upstream source file.

---

[Docs index](../../index.md) · Previous: [F5-TTS](f5tts.md) · Next: [Shami](shami.md)
