# Training reference

<!-- agent-hint
covers: every phoonnx_train.train flag, the training engines, their quality presets and the evaluation and vocoder tools, and the features.json generator
source: phoonnx_train/train.py:main, phoonnx_train/engines/__init__.py:_register_builtins, phoonnx_train/engines/*.py:_QUALITY_PRESETS, phoonnx_train/eval_loop.py:main, phoonnx_train/train_vocos.py:main, phoonnx_train/export_vocos.py:main, phoonnx_train/features_json.py:cli
generated: no
verify: ! comm -3 <(grep '^| `--' docs/training/training.md | grep -oP -- '(?<![\w-])--[a-z][a-z0-9-]*' | sort -u) <(python -m phoonnx_train.train --help | grep -oP -- '(?<![\w-])--[a-z][a-z0-9-]*' | grep -vx -- --help | sort -u) | grep .; the eval_loop and Vocos flags and the preset tables: none, checked by reading phoonnx_train/eval_loop.py:main, phoonnx_train/train_vocos.py:main, phoonnx_train/export_vocos.py:main and phoonnx_train/engines/*.py:_QUALITY_PRESETS
see: preprocess.md, export.md, engines/matcha.md, ../vocoders.md
-->

This page lists every `phoonnx_train.train` flag, the training engines and the multi-stage recipes, for readers who finished the [training quickstart](quickstart.md).

The [preprocess reference](preprocess.md) covers preprocessing, and [Export](export.md) covers
the ONNX conversion. `--engine` selects an architecture. The command gives model creation,
checkpoint loading and export to that engine.

```bash
python -m phoonnx_train.train --dataset-dir DIR [options]
```

## train.py options

| Option | Default | Description |
|---|---|---|
| `--dataset-dir DIR` | required | Pre-processed dataset directory (contains `config.json`, `dataset.jsonl`) |
| `--engine NAME` | `vits` | Architecture to train (see [Engines](#engines)) |
| `--quality TEXT` | `medium` | Quality/size preset; validated against the engine's `quality_presets()` (falls back to `medium` or the first preset if unknown) |
| `--batch-size INT` | `16` | Training batch size |
| `--max-epochs INT` | `1000` | Stop after this many epochs |
| `--checkpoint-epochs INT` | `1` | Save a checkpoint every N epochs (all are kept) |
| `--validation-split FLOAT` | `0.1` | Fraction of data held out for validation |
| `--num-workers INT` | `4` | Data-loader worker processes |
| `--accelerator TEXT` | `auto` | `cpu`, `gpu`, `tpu`, `mps`, … |
| `--devices TEXT` | `1` | Number of devices or a list of device IDs (a string) |
| `--precision TEXT` | `32` | Precision: `16`, `32`, `bf16`, `16-mixed`, … (a string) |
| `--default-root-dir DIR` | none | Root for logs and checkpoints |
| `--seed INT` | `1234` | Random seed |
| `--resume-from-checkpoint PATH` | none | Resume the run of a Lightning checkpoint, a URL or a remote path. The engine loads a local file that Lightning did not write as a weight-only warm start, and the log gives a warning. `vits` needs a Lightning checkpoint and stops with an error on any other file |
| `--resume-from-single-speaker-checkpoint PATH` | none | Convert a single-speaker checkpoint to multi-speaker and resume |
| `--discard-encoder` | off | Discard encoder weights from the base checkpoint (not supported by all engines) |
| `--log-audio-samples / --no-log-audio-samples` | off | Log synthesized validation audio each epoch (needs a TensorBoard logger) |
| `--compile` | off | Compile the model with `torch.compile` for faster training |
| `--compile-mode {default,reduce-overhead,max-autotune,max-autotune-no-cudagraphs}` | `default` | `torch.compile` mode; only relevant with `--compile` |
| `--eval-sentences FILE` | none | Held-out sentences that the in-training scoreboard speaks (see [In-training mode](#in-training-mode-early-stopping-callback)) |
| `--eval-every INT` | `0` | Score a checkpoint every N epochs; `0` turns the scoreboard off |
| `--early-stop-patience INT` | `0` | Stop after N scored epochs with no better score; `0` turns early stopping off |
| `--eval-speaker-ref-dir DIR` | none | Reference recordings of the target speaker, for speaker similarity |
| `--min-spk-sim FLOAT` | none | Lowest speaker similarity a best checkpoint must reach |
| `--eval-speaker-id INT` | none | Speaker id that the scoreboard uses for a multi-speaker model |
| `--eval-seed INT` | `1234` | Seed for the scoreboard synthesis |

> There is **no `--learning-rate` flag.** The optimizer and schedule are owned by the engine
> and its quality preset.

**Outputs.** PyTorch Lightning writes the checkpoints under
`<default-root-dir>/lightning_logs/version_<N>/checkpoints/` as `epoch=<E>-step=<S>.ckpt`.
It keeps every checkpoint and writes no `last.ckpt`. Without `--default-root-dir`, the run
folder is the current folder. TensorBoard logs live under the same `lightning_logs` tree:

```bash
tensorboard --logdir <default-root-dir>/lightning_logs
```

## Engines

`--engine` accepts any registered training engine:

| `--engine` | Architecture | Notes / guide |
|---|---|---|
| `vits` | VITS end-to-end | Default; single-graph waveform |
| `glowtts` | GlowTTS (flow-based mel) | Two-stage; [guide](engines/glowtts.md) |
| `matcha` | Matcha (flow-matching mel) | Two-stage; [guide](engines/matcha.md) |
| `fastpitch` | ForwardTTS / FastPitch | Two-stage; [guide](engines/fastpitch.md) |
| `speedyspeech` | ForwardTTS / SpeedySpeech | Same engine as `fastpitch`, no pitch |
| `mixer` / `mixertts` | Mixer-TTS | Two-stage; [guide](engines/mixertts.md) |
| `zipvoice` | ZipVoice (flow-matching, in-context cloning) | [guide](engines/zipvoice.md) |
| `supertonic` | SuperTonic (three-stage: autoencoder + flow-matching + duration) | [guide](engines/supertonic.md) |
| `optispeech` | OptiSpeech (lightweight end-to-end, S4D attention) | Install `phoonnx[train,train-optispeech]` |
| `yourtts` | YourTTS (VITS + d-vector cloning) | See [Cloning](../cloning.md) |
| `styletts2` | StyleTTS2 / Kokoro | See [StyleTTS2](#styletts2-engine---engine-styletts2) below |
| `styletts2-aligner` | StyleTTS2 text aligner (ASRCNN) | Auxiliary model |
| `styletts2-plbert` | StyleTTS2 prosodic text encoder (PL-BERT) | Auxiliary model |
| `styletts2-pitch` | StyleTTS2 JDC pitch extractor | Auxiliary model |
| `rapido` | Rapido | [guide](engines/rapido.md) |

`--quality` selects a preset that each engine defines:

| Presets | Engines |
|---|---|
| `x-low`, `medium`, `high` | `vits`, `glowtts`, `matcha`, `fastpitch`, `speedyspeech`, `mixer`, `optispeech`, `yourtts` |
| `low`, `medium`, `high` | `styletts2` |
| `base`, `low` | `zipvoice`, `supertonic`, `rapido` |
| `low`, `medium` | `styletts2-aligner`, `styletts2-plbert` |
| `medium` | `styletts2-pitch` |

An unknown `--quality` value logs a warning. The run then uses `medium`, or the first preset
when the engine has no `medium`.

## Resuming and fine-tuning

Resume an interrupted run (restores optimizer, scheduler and epoch):

```bash
python -m phoonnx_train.train --dataset-dir train_out --default-root-dir train_out/runs \
  --resume-from-checkpoint train_out/runs/lightning_logs/version_0/checkpoints/epoch=50-step=12750.ckpt
```

**Fine-tune onto another speaker or dataset:** preprocess that data with `--prev-config` pointing
at the base voice's `config.json` so the phoneme-to-ID map is preserved (see the
[fine-tuning options](preprocess.md#fine-tuning-phoneme-map)), then train with
`--resume-from-checkpoint` on the base checkpoint.

**Single-speaker → multi-speaker:** `--resume-from-single-speaker-checkpoint` adapts a
single-speaker checkpoint into a multi-speaker model and resumes.

## engine_params and precedence

Engine-specific knobs (`asr_path`, `plbert_dir`, `stage`, `backbone`, `download_aux`,
`vocoder_path`, …) ride in an `"engine_params"` dict in the dataset's `config.json`. When an
engine reads a parameter, the effective value is resolved with this precedence:

**explicit CLI flag > `config.json` `engine_params` > CLI default > quality preset.**

## ZipVoice engine (`--engine zipvoice`)

Trains the ZipVoice flow-matching TTS (Zipformer text encoder + flow-matching decoder),
vendored in `phoonnx_train/zipvoice/`. It consumes the same preprocessed dataset as every other
engine; audio is resampled to 24 kHz internally and turned into 100-bin Vocos log-mel features
(cached next to the audio cache).

- `--quality base` (default fallback) uses the upstream model size; `--quality low` is a tiny
  smoke-test tier.
- Training uses the upstream recipe: ScaledAdam + the Eden schedule, 70–100% target-span
  infilling masks, and text-condition dropout for classifier-free guidance.
- `--resume-from-checkpoint` accepts both Lightning checkpoints and the upstream `{"model": …}`
  layout, so a released ZipVoice checkpoint can be a fine-tuning start point.

```bash
python -m phoonnx_train.train --dataset-dir train_out --engine zipvoice --quality base \
  --default-root-dir train_out/runs

python -m phoonnx_train.export_onnx \
  train_out/runs/lightning_logs/version_0/checkpoints/epoch=99-step=25000.ckpt \
  --config train_out/config.json --engine zipvoice --output-dir ./exported
# -> text_encoder.onnx + fm_decoder.onnx
```

See the [ZipVoice guide](engines/zipvoice.md).

## MOSS-TTS-Nano (standalone pipeline)

MOSS-TTS-Nano is an autoregressive codec LM, so it does not consume the phoneme-based
preprocessed dataset the `--engine` trainers share — it trains on pre-encoded RVQ tokens
and raw text, from its own CLI in `phoonnx_train/mosstts/`:

```bash
python -m phoonnx_train.mosstts.prepare_data --codec-encode-onnx ... \
  --input-manifest data/metadata.csv --output-jsonl data/train.codes.jsonl
python -m phoonnx_train.mosstts.train --train-jsonl data/train.codes.jsonl \
  --tokenizer-model tokenizer.model --warm-start-from models/MOSS-TTS-Nano \
  --output-dir runs/moss-pt
python -m phoonnx_train.mosstts.export_onnx --checkpoint runs/moss-pt/last.ckpt \
  --output-dir exported/moss-pt --external-data
```

See the [MOSS-TTS-Nano guide](engines/mosstts.md) for the data format, warm-start, resume
and export details.

## GlowTTS engine (`--engine glowtts`)

Trains GlowTTS, a flow-based text→mel acoustic model with Monotonic Alignment Search, vendored
in `phoonnx_train/glowtts/`. It consumes the same preprocessed dataset (mel features only — no
waveform/vocoder training).

- `--quality x-low|medium|high` scales encoder/flow widths; `medium` is the paper configuration.
- Recipe: MLE (flow NLL) + log-duration MSE losses, Adam with Noam warmup (4000 steps).
- Export produces the **mel model only**; synthesis pairs it with a separate vocoder ONNX (see
  [Vocoders](../vocoders.md)). The mel basis is pinned to fmin 0 / fmax 8000 Hz and recorded in
  the ONNX metadata.

```bash
python -m phoonnx_train.train --dataset-dir train_out --engine glowtts \
  --default-root-dir train_out/runs
python -m phoonnx_train.export_onnx \
  train_out/runs/lightning_logs/version_0/checkpoints/epoch=999-step=250000.ckpt \
  --config train_out/config.json --engine glowtts --output-dir ./exported
```

See the [GlowTTS guide](engines/glowtts.md).

## FastPitch / SpeedySpeech (`--engine fastpitch` / `--engine speedyspeech`)

The Coqui **ForwardTTS** family: a non-autoregressive text→mel model where an encoder produces
per-token states, a duration predictor expands them to frame rate, and a decoder renders an
80-channel mel. Durations are learned **unsupervised** (attention alignment + monotonic
alignment search — no external aligner). The model/loss code is vendored in
`phoonnx_train/fastpitch/` (pure-torch, no `TTS` dependency) and reuses the shared VITS
spectrogram cache, so switching between `vits` and `fastpitch`/`speedyspeech` on the same
phonemized dataset needs no re-preprocessing.

| | FastPitch | SpeedySpeech |
|---|---|---|
| encoder/decoder | FFT-transformer | residual conv-BN |
| pitch predictor | on | off |
| relative size | larger, higher quality | smaller, faster |

Both are the same engine class (`ForwardTTSTrainingEngine`); the `--engine` name only changes
the default `variant`, which can also be set via the engine's `extra` config
(`variant: speedyspeech`).

**Pitch (F0).** The FastPitch pitch predictor needs a frame-level F0 target. The engine's
`extra_preprocess` extracts F0 and caches it, frame-aligned with the mel. The preprocess flag
`--f0-method` selects the extractor. The default `pyin` uses `librosa.pyin`, which `[train]`
installs. `dio` and `harvest` use WORLD through `pyworld`, from the `train-pyworld` extra, and
they are about 50 times faster. When `pyworld` is missing for `dio` or `harvest`, preprocess
logs a warning and skips F0 extraction. Training then runs, but the pitch predictor trains
without its target loss. SpeedySpeech uses no pitch.

`--quality` (`x-low` / `medium` / `high`) scales `hidden_channels`, `hidden_channels_ffn`, and
the encoder/decoder layer count. FastPitch/SpeedySpeech are **two-stage** — a separate vocoder
is required at inference. See the [FastPitch guide](engines/fastpitch.md).

## StyleTTS2 engine (`--engine styletts2`)

A full two-stage engine porting the yl4579/StyleTTS2 recipe onto the shared framework. It
trains models **from scratch in other languages** and fine-tunes existing checkpoints onto
other speakers. The upstream model/loss/data code is vendored in `phoonnx_train/styletts2/`.
Install `phoonnx[train,train-styletts2]`.

**Stages** (select with `stage` in `engine_params` / `StyleTTS2Config`):

| Stage | What trains |
|---|---|
| `first` | text aligner (TMA), text encoder, style encoder, decoder — mel reconstruction with ground-truth F0/energy; adversarial + WavLM feature-matching after `tma_epoch` |
| `second` | PL-BERT + duration/prosody predictors, style diffusion (after `diff_epoch`), joint decoder + SLM adversarial (after `joint_epoch`); starts from the stage-1 checkpoint (`first_stage_path`) |
| `finetune` | the `second` recipe with diffusion + joint enabled from epoch 0 — adapt a checkpoint to another speaker or dataset |

**Auxiliary models** (all configurable via `engine_params`):

| Key | Model | Where to get it |
|---|---|---|
| `asr_path` + `asr_config` | text aligner (ASRCNN) | upstream `Utils/ASR/`; trained further during TMA |
| `f0_path` | JDC pitch extractor | upstream `Utils/JDC/bst.t7` |
| `plbert_dir` | PL-BERT | upstream `Utils/PLBERT/` (English); pretrained es/ca ([PL-BERT-wp-es](https://huggingface.co/BSC-LT/PL-BERT-wp-es)), gl ([PL-ModernBERT-gl](https://huggingface.co/proxectonos/PL-ModernBERT-gl)); or train with `--engine styletts2-plbert` |
| `slm.model` | WavLM for SLM losses | HF hub, default `microsoft/wavlm-base-plus`; disable with `use_slm: false` |

Set `download_aux: true` to fetch the English aligner/pitch/PL-BERT automatically (cached under
`~/.cache/phoonnx/styletts2_aux_en`) for any unset path. Missing auxiliaries are otherwise
randomly initialized with a warning.

```json
{"engine_params": {"stage": "second", "first_stage_path": "…", "plbert_dir": "…", "download_aux": true}}
```

**Data layout.** StyleTTS2 uses the upstream list format inside the dataset dir:
`train_list.txt` / `val_list.txt` with `filename.wav|phonemes|speaker_id` lines and audio under
`wavs/` (24 kHz).

**Quality presets.** `low` (halved widths, iSTFTNet), `medium` (LJSpeech recipe, iSTFTNet),
`high` (LibriTTS recipe, HiFi-GAN decoder, multispeaker-ready).

**Export.** `export_onnx` (via `phoonnx_train/styletts2/export.py`) accepts an upstream `.pth`
or a Lightning `.ckpt` and emits the two-graph zero-shot contract the `StyleTTS2Adapter`
consumes — `model.onnx` (tokens + style + speed → waveform) and `style_encoder.onnx` (reference
waveform → style) — plus `config.json`. StyleTTS2 exports are self-contained; no external
vocoder is required.

### Training every StyleTTS2 step for another language

A from-scratch model in another language needs a language-specific text aligner and prosodic text
encoder trained first. `phoonnx_train` ships an engine for each:

| Engine | Trains | Consumed via |
|---|---|---|
| `styletts2-aligner` | ASRCNN text aligner (CTC + s2s CE) | `asr_path` + `asr_config` |
| `styletts2-plbert` | PL-BERT (`backbone: albert` or `modernbert`) | `plbert_dir` |
| `styletts2-pitch` | JDC pitch extractor (pyworld ground truth) | `f0_path` |

All three fine-tune too (warm-start via `pretrained_path` / `pretrained_dir`).

```bash
# 0. phonemize with phoonnx's own phonemizers
python -m phoonnx_train.styletts2.phonemize_corpus list \
    raw_list.txt dataset/train_list.txt --lang pt --phonemizer espeak
python -m phoonnx_train.styletts2.phonemize_corpus plbert \
    corpus.txt plbert_data --lang pt

# 1. text aligner   2. PL-BERT   3. pitch (optional)
python -m phoonnx_train.train --dataset-dir dataset   --engine styletts2-aligner
python -m phoonnx_train.train --dataset-dir plbert_data --engine styletts2-plbert
python -m phoonnx_train.train --dataset-dir dataset   --engine styletts2-pitch

# 4. StyleTTS2 stage first, then second (paths via engine_params in config.json)
python -m phoonnx_train.train --dataset-dir dataset --engine styletts2

# 5. export the two-graph ONNX contract
python -m phoonnx_train.export_onnx --engine styletts2 ...
```

## Evaluating checkpoints during training

`phoonnx_train/eval_loop.py` watches the training directory for Lightning checkpoints it has not scored
(`epoch=*.ckpt`) and scores each on a fixed set of held-out sentences, synthesized on CPU so
training keeps the GPU. Results are appended to `metrics.csv` (one row per epoch), per-utterance
scores are written for **every** epoch under `perutt/epoch<N>.csv` (columns `sentence`, `utmos`,
`spk_sim` — so a single sentence degrading while the mean holds is visible across epochs), and
the best epoch's wavs are kept under `samples/epoch<N>/`. The loop is idempotent — already-scored
(and permanently failed) epochs are skipped on restart. Needs `phoonnx[train-eval]`.

```bash
python -m phoonnx_train.eval_loop \
  --train-dir train_out --config train_out/config.json \
  --sentences heldout.txt --output-dir ./eval \
  --speaker-ref-dir my_dataset/wavs
```

Useful flags: `--once` (single pass), `--poll N` (seconds between scans, default 60),
`--dry-run`, and `--noise-scale` / `--length-scale` / `--noise-w` overrides.

Two metrics are reported: **UTMOS** (`utmos_*`, an automatic 1–5 MOS predictor — treat as a
relative ranker between checkpoints of the same voice, especially on non-English) and
**speaker similarity** (`spk_sim_*`, cosine similarity to a centroid of the target speaker's
reference recordings; needs `speakeronnx` + `--speaker-ref-dir`, otherwise only UTMOS is
scored).

## Evaluation and early stopping

The evaluation logic lives in the reusable `phoonnx_train.evaluation` package
(`CheckpointScorer`, `SelectionPolicy`, `MetricsTracker`, Lightning callbacks). The same code
runs in two modes.

### Sidecar mode (out-of-process scoreboard)

`phoonnx_train/eval_loop.py` is a small command over the package, with the flags above.
It loads each checkpoint through the training-engine registry. `--engine` defaults to `auto`:
the `engine` key in `config.json` when it is present, else `vits`. It synthesizes on CPU with
**per-utterance deterministic seeding**: `torch.manual_seed(seed + i)` runs just before
utterance `i`, so the same checkpoint scored twice gives identical wavs and scores. These
flags control the scoreboard:

- `--early-stop-patience N` — after `N` scored epochs pass with no better score, write a
  `stop.flag` into `--train-dir`.
- `--min-spk-sim FLOAT` — the similarity gate floor (see below).
- `--speaker-id INT` — speaker id for multi-speaker models (a warning is logged when
  `num_speakers > 1` and no id is given; synthesis falls back to the engine default voice).
- `--engine NAME` — engine override (default `auto`).
- `--vocoder PATH` — passed through to the engine's synthesis (engine-dependent; ignored by
  the end-to-end VITS engine).
- `--seed INT` — the base seed of the per-utterance seeding (default `1234`).
- `--num-ref-wavs INT` — how many reference recordings make the speaker centroid (default `60`).

### In-training mode (early stopping callback)

`train.py` has opt-in flags: `--eval-sentences FILE`, `--eval-every N` (default `0` = off),
`--early-stop-patience N`, `--eval-speaker-ref-dir DIR`, `--min-spk-sim FLOAT`,
`--eval-speaker-id INT` and `--eval-seed`. When they are set, `EvalScoreboardCallback` runs the same `CheckpointScorer` in-process every `N`
epochs (after `ModelCheckpoint` saves), writes the same scoreboard under `<run-dir>/eval/`, and
sets `trainer.should_stop` once patience is exhausted. **Scoring failures never crash training**
— they are logged and swallowed.

`StopFileCallback` is **always** added to the trainer (even with evaluation off): it watches
`<run-dir>/stop.flag` and stops training when it appears. This is the sidecar → trainer bridge —
a sidecar `eval_loop` with `--early-stop-patience` can stop a training run it does not share a
process with. With no flag present it is a no-op.

### The similarity gate

`SelectionPolicy(metric="utmos_mean", min_spk_sim=...)` selects the best checkpoint by UTMOS
**gated on speaker similarity**: UTMOS alone can prefer a checkpoint whose voice has drifted
away from the target speaker (higher naturalness, wrong identity). When speaker scoring is
active and `--min-spk-sim` is set, a candidate must clear that floor to be eligible at all;
among eligible candidates the higher UTMOS wins. With no speaker score available (no reference
dir / `speakeronnx` missing) selection falls back to UTMOS-only and logs a warning.

### Artifacts and the stop.flag contract

- `metrics.csv` — one row per scored epoch. A scoreboard with fewer columns is read and
  appended to.
- `best.json` — epoch, step, checkpoint path, and all scores of the current best.
- `best.ckpt` — a **copy** (not a symlink) of the best checkpoint, so it survives checkpoint
  pruning.
- `perutt/epoch<N>.csv` — per-utterance scores for every evaluated epoch (`sentence`, `utmos`,
  `spk_sim`), for tracking per-sentence trends / overfit across epochs.
- `samples/epoch<N>/` — the best epoch's synthesized wavs plus `perutt.csv`.
- `failed.json` — a checkpoint that fails to load/score 3 times is recorded failed and skipped
  forever (ERROR logged); no infinite retry.
- `stop.flag` — a single reason line. Its presence means "stop training"; the trainer's
  `StopFileCallback` reads the reason, logs it and stops at the next epoch boundary.

## Writing a feature table (features.json)

A feature-input text encoder reads one distinctive-feature vector per phone id from a `features.json` file. `phoonnx_train.features_json` writes that file from a voice `config.json` or from a bare `phoneme_id_map` JSON. Install its feature sources first:

```bash
uv pip install --prerelease=allow 'phoonnx[train-features]'
```

The `--source` option selects the feature table: `phonematcher` (36 features) or `orthography2ipa` (the 23 features of its `feats.py`). Each value is 1 (plus), -1 (minus) or 0 (does not apply). The file also records the source version and the sha256 of the map, so a text encoder can refuse a table made for a different map.

Seven prosody features end every vector. A space sets `pros_word_boundary`, and `ˈ` and `ˌ` set the two stress rows. A punctuation character sets rows by its Unicode name, so the rule reads every script:

- COMMA, SEMICOLON, COLON, ANO TELEIA or SECTION MARK sets `pros_pause_minor`.
- FULL STOP, QUESTION, EXCLAMATION, INTERROBANG, DANDA or ELLIPSIS sets `pros_pause_major`, unless the name also has INVERTED.
- QUESTION or INTERROBANG sets `pros_question`, and EXCLAMATION or INTERROBANG sets `pros_exclamation`.

So `?` and `؟` set a major pause and a question, and `¿` sets only a question. `⁈`, `⁉` and `‽` set a major pause, a question and an exclamation. U+1361 ETHIOPIC WORDSPACE sets `pros_word_boundary`. The IPA major-group bar `‖` sets `pros_pause_major`. The click letters `ǀ` and `ǁ` are phones. `.` sets `pros_pause_major`; for a voice that writes `.` as the IPA syllable break, `--non-phone .` makes it `null`. A phone is 0 on all seven rows, and a prosody symbol is 0 on every phone feature.

Four tone features come after the prosody block. The generator maps tone numerals and left-stem letters to Chao letters through phonematcher's `NUMERALS` table, and `is_tone` then finds a tone id. Numerals read as Chao levels, where 5 is high. A corpus that writes 1 as the high tone, as some Africanist and Otomanguean sources do, must be converted before generation, or its pitch comes out inverted. A tone id has `tone_is` 1 and its pitch at the start, middle and end, where each Chao level 1 to 5 becomes (level − 3) / 2. So `˥` gives 1, 1, 1, `˧˥` gives 0, 0.5, 1, and `⁵¹` gives 1, 0, −1. The orthography2ipa source uses the same tone block, from phonematcher, so the tone block always needs phonematcher, even with `--source orthography2ipa`. The `train-features` extra installs both. A tone id is 0 on every phone and prosody feature. A vowel or a syllabic consonant (marked with U+0329) written with IPA tone diacritics, such as `á`, `ǎ`, `ȁ` or `ḿ̩`, gets the phone features of the phone without its marks and the tone block of the marks, read with phonematcher's diacritic table. So `á` gets the features of `a` and the tone 1, 0.5, 0.5, 0.5. A tone mark on any other consonant, such as Slavic `ś`, is not read as a tone, and the command refuses the symbol. Every other id is 0 on the tone block. The `source` object records the phonematcher version under `tone`, for both sources.

Pad, BOS, EOS, digits, `|` and punctuation that matches no rule get `null`. The command prints each punctuation character with its name and rows, and each null id with its reason. When a symbol looks like a phone and the source cannot read it, the command writes nothing and lists each such symbol with its code points. Fix the source, or rule the symbol a non-phone with `--non-phone SYMBOL` (repeatable).

U+00B7 MIDDLE DOT gets `null`, because a voice map uses it more often for a syllable boundary or the Catalan l·l than for a pause. NFC turns U+0387 GREEK ANO TELEIA into U+00B7. For a voice where it is a pause, `--pause SYMBOL` (repeatable) rules the symbol `pros_pause_minor`. The same applies to `|`, which voice maps also use as a word or syllable separator. A `--pause` value is one character. The command refuses an empty or longer value before it reads the map, and it writes the file only after every check passes.

```bash
echo '{"_":[0],"a":[1],"ɐ̃":[2],"ˈ":[3]," ":[4],"?":[5],"،":[6],"§":[7],"˧˥":[8]}' > small_map.json
python -m phoonnx_train.features_json small_map.json --source phonematcher --out small_features.json
```

```text
ids in: 9
non-phone allowlist: '_' pad, '^' BOS, '$' EOS, '|' null by default; rule it a pause with --pause |, Pc punctuation, Pd punctuation, Ps punctuation, Pe punctuation, Pi punctuation, Pf punctuation, Po punctuation, Nd digit
prosody ids: 3
  3 'ˈ' MODIFIER LETTER VERTICAL LINE: pros_stress_primary
  5 '?' QUESTION MARK: pros_pause_major pros_question
  6 '،' ARABIC COMMA: pros_pause_minor
null ids: 2
  0 '_' pad
  7 '§' punctuation SECTION SIGN, no prosody rule
ids out: 9
wrote small_features.json (phonematcher 2.3.10a3, 47 features, 7 prosody, 4 tone)
```

A codepoint-level map has a separate id for each combining mark and modifier letter, and the source reads none of them alone. Such a map is refused. Use a map with one id per whole segment. orthography2ipa's `feats.py` does not read tie bars, so with `--source orthography2ipa` the command refuses a tied segment such as `d͡ʒ`, `t͡ʃ` or `k͜p`. A tied affricate or double articulation reads with `--source phonematcher`. phonematcher reads tied affricates, but it refuses a tied diphthong such as `a͡i`, because a diphthong is two segments.

## Training a Vocos vocoder

Two-stage engines (Matcha and other mel-emitting engines) produce a mel and rely on a separate
vocoder. `phoonnx_train/train_vocos.py` trains a Vocos-style vocoder in-house so a voice can
ship with a vocoder trained (or fine-tuned) on the same speaker's audio.

**Data:** audio only, no transcripts — any directory tree of `.wav`/`.flac`/`.ogg`/`.mp3`
(including a preprocess `cache/` folder). `--audio-dir` is repeatable. A few hours of clean
speech is enough to fine-tune. Training from scratch needs much more audio, or a `--warm-start`.

> **Mel settings are locked.** The vocoder is trained on the exact mel configuration phoonnx
> acoustic models use (`n_fft=1024`, `hop=256`, `win=1024`, 80 mels, `fmax=8000`, log-mel). A
> vocoder trained with different settings gives noise for every phoonnx mel model.

```bash
python -m phoonnx_train.train_vocos \
  --audio-dir /path/to/voice/wavs \
  --sample-rate 22050 --batch-size 16 --crop-seconds 1.0 \
  --warm-start charactr/vocos-mel-24khz \
  --max-epochs 100 --default-root-dir /tmp/vocos_train
```

`--warm-start` accepts a local `.ckpt`/`.bin` or a HuggingFace repo id with reference-Vocos
weights; matching parameters are copied and the matched fraction logged.
`--resume-from-checkpoint` continues an interrupted run. The command also takes the
`train.py` flags `--checkpoint-epochs`, `--devices`, `--accelerator`, `--precision`,
`--num-workers` and `--seed`, with the same defaults.

**Export and wire up:**

```bash
python -m phoonnx_train.export_vocos \
  /tmp/vocos_train/lightning_logs/version_0/checkpoints/epoch=99-step=12345.ckpt \
  vocoder.onnx
```

This writes `vocoder.onnx` (+ `vocoder.onnx.json` with the STFT parameters) in the layout the
vocoder registry auto-detects as `vocos`. `--seconds` (default `2.0`) sets the length of the
random mel that the export uses for its parity check. Point any mel-emitting voice at it via
`engine_params={"vocoder_path": "vocoder.onnx"}`.

## config.json reference

Preprocess writes a `config.json` that both training and inference read. Key fields:

| Field | Meaning |
|---|---|
| `dataset` | Dataset name (from `--dataset-name` or the parent dir) |
| `audio.sample_rate` | Training/inference sample rate |
| `audio.quality` | Arbitrary label (from `--audio-quality` or the output dir name) |
| `lang_code` | Language code used for phonemization (normalized with `langcodes`) |
| `inference.noise_scale` / `length_scale` / `noise_w` | Default inference scales |
| `inference.add_diacritics` | Apply diacritics at inference (Arabic/Hebrew only) |
| `alphabet` | Phoneme alphabet (`ipa`, `unicode`, `arpa`, `pinyin`, …) |
| `phoneme_type` | Phonemizer used (`espeak`, `gruut`, `byt5`, …) |
| `phonemizer_model` | Model/variant for phonemizers that take one |
| `phoneme_id_map` | Phoneme symbol → integer id |
| `num_symbols` | Size of the phoneme map |
| `num_speakers` | Number of speakers (1 for single-speaker) |
| `speaker_id_map` | Speaker label → id (multi-speaker) |
| `phoonnx_version` | Version of the preprocessing pipeline |
| `keep_punctuation` | The phonemes keep clause punctuation (`, . ? ! : ;`) |
| `text_casing` | The casing applied to the text before phonemization (`ignore`, `lower`, `upper`, `casefold`, from `--text-casing`). Synthesis applies the same casing |
| `phonemizer_registry` / `phonemizer_registry_version` | The distribution that resolved the phonemizer, and its version |
| `phonemizer_identity` | The engine that phonemized the rows: `backend`, `distribution`, `version` (for espeak-ng, the binary), `lang` (the resolved lect), `revision` (the scriptconv hash of the source, package versions, engine release and data files that make the phonemes) and `frontend` (null, because preprocess calls the backend directly and not through `TextFrontend`). Empty when every row came from a phonemes column |
| `phonemes_phonemized` / `phonemes_from_column` | How many manifest rows were phonemized, and how many came from a phonemes column |

The full loaded form (tokenizer flags, special tokens, `engine_params`) is documented in the
[Configuration reference](../configuration.md).

---

[Docs index](../index.md) · Previous: [Preprocess reference](preprocess.md) · Next: [Export to ONNX](export.md)
