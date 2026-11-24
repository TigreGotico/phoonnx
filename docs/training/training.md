# Training reference

<!-- agent-hint
covers: every phoonnx_train.train flag, the training engines, their quality presets and the evaluation and vocoder tools, and the features.json generator
source: phoonnx_train/train.py:main, phoonnx_train/train.py:_build_extra, phoonnx_train/engines/__init__.py:_register_builtins, phoonnx_train/engines/*.py:_QUALITY_PRESETS, phoonnx_train/eval_loop.py:main, phoonnx_train/train_vocos.py:main, phoonnx_train/export_vocos.py:main, phoonnx_train/features_json.py:cli, phoonnx_train/vits/input_features.py
generated: no
verify: ! comm -3 <(sed -n '/^## train.py options/,/^## Engines/p' docs/training/training.md | grep -P '^\| \x60--' | grep -oP -- '(?<![\w-])--[a-z][a-z0-9-]*' | sort -u) <(.venv/bin/python -m phoonnx_train.train --help | grep -oP -- '(?<![\w-])--[a-z][a-z0-9-]*' | grep -vx -- --help | sort -u) | grep . && .venv/bin/python -W ignore -c "import re, subprocess, sys; from phoonnx_train.engines import list_engines, get_engine; p = open('docs/training/training.md').read(); table = p.split(chr(10) + '| Presets | Engines |')[1].split(chr(10) + chr(10))[0]; rows = {e: tuple(re.findall(r'\x60([a-z-]+)\x60', r.split('|')[1])) for r in table.splitlines()[1:] for e in re.findall(r'\x60([a-z0-9-]+)\x60', r.split('|')[2])}; bad = [e for e in list_engines() if rows.get(e) != tuple(get_engine(e).quality_presets())] + sorted(set(rows) - set(list_engines())); fence = chr(96) * 3; m = re.search(r\"^echo '(.*)' > small_map.json$\", p, re.M).group(1); want = p.split(fence + 'text' + chr(10))[1].split(fence)[0]; got = subprocess.run([sys.executable, '-W', 'ignore', '-m', 'phoonnx_train.features_json', '/dev/stdin', '--source', 'phonematcher', '--out', '.venv/small_features.json'], input=m, capture_output=True, text=True).stdout; got = re.sub(r'wrote \S+ \(phonematcher [^,]+,', 'wrote small_features.json (phonematcher <version>,', got); bad += [] if got == want else ['features_json report differs']; print(bad); sys.exit(bool(bad))" && .venv/bin/python -W ignore -c "import re, sys, click; from phoonnx_train.train import _build_extra; p = open('docs/training/training.md').read().split('## engine_params and precedence')[-1].split(chr(10) + '## ')[0]; keys = {'command line': 'flag', 'engine_params': 'engine_params', 'quality preset': 'preset', 'default value': 'default'}; page = [next(v for k, v in keys.items() if k in item) for item in re.findall(r'^[0-9]+[.] (.*)$', p, re.M)]; on = {'flag', 'engine_params', 'preset', 'default'}; got = []; out = {}; cmd = click.Command('c', params=[click.Option(['--k'], default='default')], callback=lambda k: out.update(r=_build_extra({'k': 'preset'} if 'preset' in on else {}, {'k': 'engine_params'} if 'engine_params' in on else {}, k=k)['k'])); [(cmd.main(['--k', 'flag'] if 'flag' in on else [], standalone_mode=False), got.append(out['r']), on.discard(out['r'])) for _ in range(4)]; print(page, got); sys.exit(page != got)"
see: preprocess.md, export.md, engines/matcha.md, ../vocoders.md
-->

This page lists every `phoonnx_train.train` flag, the training engines and their quality presets, the evaluation tools and the feature table generator. Read it after the [training quickstart](quickstart.md).

The [preprocess reference](preprocess.md) covers preprocessing, and [Export](export.md) covers the ONNX conversion. `--engine` selects an architecture. The engine creates the model, loads checkpoints and exports the result.

```bash
python -m phoonnx_train.train --dataset-dir DIR [options]
```

## Outputs

PyTorch Lightning writes the checkpoints to `<default-root-dir>/lightning_logs/version_<N>/checkpoints/` as `epoch=<E>-step=<S>.ckpt`. It keeps every checkpoint and writes no `last.ckpt`. Without `--default-root-dir`, the run folder is the current folder. The TensorBoard logs are in the same `lightning_logs` folder:

```bash
tensorboard --logdir <default-root-dir>/lightning_logs
```

## Resuming and fine-tuning

To resume an interrupted run, with its optimizer, scheduler and epoch, give the last checkpoint:

```bash
python -m phoonnx_train.train --dataset-dir train_out --default-root-dir train_out/runs \
  --resume-from-checkpoint train_out/runs/lightning_logs/version_0/checkpoints/epoch=50-step=12750.ckpt
```

To fine-tune a voice on another speaker or dataset, first preprocess that data with `--prev-config` set to the `config.json` of the base voice. This keeps the phoneme id map (see the [fine-tuning options](preprocess.md#fine-tuning-phoneme-map)). Then train with `--resume-from-checkpoint` set to the base checkpoint.

To turn a single-speaker model into a multi-speaker model, give its checkpoint to `--resume-from-single-speaker-checkpoint`.

## train.py options

| Option | Default | Description |
|---|---|---|
| `--dataset-dir DIR` | required | The preprocessed dataset folder, with `config.json` and `dataset.jsonl` |
| `--engine NAME` | `vits` | The architecture to train (see [Engines](#engines)) |
| `--quality TEXT` | `medium` | The quality preset of the engine (see [Engines](#engines)) |
| `--batch-size INT` | `16` | The training batch size |
| `--max-epochs INT` | `1000` | Stop after this number of epochs |
| `--checkpoint-epochs INT` | `1` | Save a checkpoint every N epochs. Training keeps every checkpoint |
| `--validation-split FLOAT` | `0.1` | The fraction of the data kept for validation |
| `--num-workers INT` | `4` | The number of data loader processes |
| `--accelerator TEXT` | `auto` | `cpu`, `gpu`, `tpu`, `mps` and the other Lightning accelerators |
| `--devices TEXT` | `1` | The number of devices, or a list of device ids |
| `--precision TEXT` | `32` | `16`, `32`, `bf16`, `16-mixed` and the other Lightning precisions |
| `--default-root-dir DIR` | none | The folder for logs and checkpoints |
| `--seed INT` | `1234` | The random seed |
| `--resume-from-checkpoint PATH` | none | Resume the run of a Lightning checkpoint, a URL or a remote path. A local file that Lightning did not write loads as weights only, and the log gives a warning. `vits` needs a Lightning checkpoint and stops with an error on another file |
| `--resume-from-single-speaker-checkpoint PATH` | none | Change a single-speaker checkpoint into a multi-speaker model and continue |
| `--discard-encoder` | off | Do not load the encoder weights of the base checkpoint. Not every engine supports it |
| `--input-layer ids\|residual\|composition` | `ids` | The input of the VITS text encoder. `ids` learns one vector per id. `residual` and `composition` read `--input-features` (see [Training with a feature table](#training-with-a-feature-table)) |
| `--input-features PATH` | none | A `features.json` made from the `phoneme_id_map` of this dataset. It needs `--input-layer residual` or `composition`, and only `vits` reads it |
| `--log-audio-samples / --no-log-audio-samples` | off | Log synthesized validation audio after each epoch. It needs a TensorBoard logger |
| `--compile` | off | Compile the model with `torch.compile`. If the compilation fails, training continues without it |
| `--compile-mode {default,reduce-overhead,max-autotune,max-autotune-no-cudagraphs}` | `default` | The `torch.compile` mode. It applies only with `--compile` |
| `--eval-sentences FILE` | none | Held-out sentences that the in-training scoreboard speaks (see [In-training mode](#in-training-mode-early-stopping-callback)) |
| `--eval-every INT` | `0` | Score a checkpoint every N epochs. `0` turns the scoreboard off |
| `--early-stop-patience INT` | `0` | Stop after N scored epochs with no better score. `0` turns early stopping off |
| `--eval-speaker-ref-dir DIR` | none | Recordings of the target speaker, for speaker similarity |
| `--min-spk-sim FLOAT` | none | The lowest speaker similarity that a best checkpoint must have |
| `--eval-speaker-id INT` | none | The speaker id that the scoreboard uses for a multi-speaker model |
| `--eval-seed INT` | `1234` | The seed of the scoreboard synthesis |

There is no `--learning-rate` flag. The engine and its quality preset set the optimizer and the schedule.

## Engines

`--engine` takes the name of a training engine:

| `--engine` | Architecture | Notes |
|---|---|---|
| `vits` | VITS, end to end | The default. One graph that gives the waveform |
| `glowtts` | GlowTTS, flow-based mel | Two stages. See the [GlowTTS guide](engines/glowtts.md) |
| `matcha` | Matcha, flow-matching mel | Two stages. See the [Matcha guide](engines/matcha.md) |
| `fastpitch` | ForwardTTS, FastPitch variant | Two stages. See the [FastPitch guide](engines/fastpitch.md) |
| `speedyspeech` | ForwardTTS, SpeedySpeech variant | The `fastpitch` engine with no pitch predictor |
| `mixer`, `mixertts` | Mixer-TTS | Two stages. See the [Mixer-TTS guide](engines/mixertts.md) |
| `zipvoice` | ZipVoice, flow matching with in-context cloning | See the [ZipVoice guide](engines/zipvoice.md) |
| `supertonic` | SuperTonic, with an autoencoder, a flow-matching model and a duration model | See the [SuperTonic guide](engines/supertonic.md) |
| `optispeech` | OptiSpeech, a small end-to-end model | Install `phoonnx[train,train-optispeech]` |
| `yourtts` | YourTTS, VITS with d-vector cloning | See [Cloning](../cloning.md) |
| `styletts2` | StyleTTS2 and Kokoro | See [StyleTTS2](#styletts2-engine---engine-styletts2) |
| `styletts2-aligner` | StyleTTS2 text aligner (ASRCNN) | An auxiliary model |
| `styletts2-plbert` | StyleTTS2 prosodic text encoder (PL-BERT) | An auxiliary model |
| `styletts2-pitch` | StyleTTS2 JDC pitch extractor | An auxiliary model |
| `rapido` | Rápido | See the [Rápido guide](engines/rapido.md) |

`--quality` selects a preset that the engine defines:

| Presets | Engines |
|---|---|
| `x-low`, `medium`, `high` | `vits`, `glowtts`, `matcha`, `fastpitch`, `speedyspeech`, `mixer`, `mixertts`, `optispeech`, `yourtts` |
| `low`, `medium`, `high` | `styletts2` |
| `base`, `low` | `zipvoice`, `supertonic`, `rapido` |
| `low`, `medium` | `styletts2-aligner`, `styletts2-plbert` |
| `medium` | `styletts2-pitch` |

An unknown `--quality` value logs a warning. The run then uses `medium`, or the first preset when the engine has no `medium`.

## engine_params and precedence

The settings of one engine, such as `asr_path`, `plbert_dir`, `stage`, `backbone`, `download_aux` and `vocoder_path`, go in an `"engine_params"` object in the `config.json` of the dataset. The engine reads them together with its quality preset and the values of `--batch-size`, `--validation-split`, `--num-workers` and `--log-audio-samples`. When two of these give the same key, this order applies:

1. a flag that you give on the command line,
2. `engine_params` in `config.json`,
3. the quality preset,
4. the default value of the flag.

## ZipVoice engine (`--engine zipvoice`)

This engine trains the ZipVoice flow-matching model, a Zipformer text encoder and a flow-matching decoder. The code is in `phoonnx_train/zipvoice/`. It reads the same preprocessed dataset as the other engines. It resamples the audio to 24 kHz and makes 100-bin Vocos log-mel features, which it caches beside the audio cache.

`--quality base` has the size of the upstream model, and `--quality low` is a very small model for tests. The training follows the upstream recipe: ScaledAdam with the Eden schedule, a random 70 to 100 percent span of each utterance as the target, and text-condition dropout for classifier-free guidance. `--resume-from-checkpoint` reads Lightning checkpoints and the upstream `{"model": ...}` layout, so a released ZipVoice checkpoint can start a fine-tune.

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

MOSS-TTS-Nano is an autoregressive codec language model. It does not read the phoneme dataset that the `--engine` trainers share. It trains on encoded codec tokens and raw text, with its own commands in `phoonnx_train/mosstts/`:

```bash
python -m phoonnx_train.mosstts.prepare_data --codec-encode-onnx ... \
  --input-manifest data/metadata.csv --output-jsonl data/train.codes.jsonl
python -m phoonnx_train.mosstts.train --train-jsonl data/train.codes.jsonl \
  --tokenizer-model tokenizer.model --warm-start-from models/MOSS-TTS-Nano \
  --output-dir runs/moss-pt
python -m phoonnx_train.mosstts.export_onnx --checkpoint runs/moss-pt/last.ckpt \
  --output-dir exported/moss-pt --external-data
```

The [MOSS-TTS-Nano guide](engines/mosstts.md) describes the data format, the warm start, the resume and the export.

## GlowTTS engine (`--engine glowtts`)

This engine trains GlowTTS, a flow-based text-to-mel model with Monotonic Alignment Search. The code is in `phoonnx_train/glowtts/`. It reads the same preprocessed dataset, and trains the mel model only.

`--quality x-low|medium|high` scales the widths of the encoder and the flow. `medium` is the configuration of the paper. The losses are the flow negative log-likelihood and the mean squared error of the log durations. The optimizer is Adam with a Noam schedule of 4000 warmup steps.

The export gives the mel model only. Synthesis pairs it with a separate vocoder ONNX (see [Vocoders](../vocoders.md)). The mel basis uses fmin 0 and fmax 8000 Hz, and the ONNX metadata records both values.

```bash
python -m phoonnx_train.train --dataset-dir train_out --engine glowtts \
  --default-root-dir train_out/runs
python -m phoonnx_train.export_onnx \
  train_out/runs/lightning_logs/version_0/checkpoints/epoch=999-step=250000.ckpt \
  --config train_out/config.json --engine glowtts --output-dir ./exported
```

See the [GlowTTS guide](engines/glowtts.md).

## FastPitch / SpeedySpeech (`--engine fastpitch` / `--engine speedyspeech`)

These engines train the Coqui ForwardTTS family, a text-to-mel model that is not autoregressive. An encoder makes a state for each token, a duration predictor expands the states to the frame rate, and a decoder makes an 80-channel mel. The model learns the durations by itself, through attention alignment and monotonic alignment search, with no external aligner. The code is in `phoonnx_train/fastpitch/` and needs no `TTS` package. It uses the spectrogram cache of VITS, so you can switch between `vits`, `fastpitch` and `speedyspeech` on one dataset with no new preprocessing.

| | FastPitch | SpeedySpeech |
|---|---|---|
| Encoder and decoder | FFT transformer | residual conv-BN |
| Pitch predictor | on | off |
| Size | larger | smaller |

Both names use the same engine class, `ForwardTTSTrainingEngine`. The `--engine` name only changes the default `variant`. You can also set `variant: speedyspeech` in `engine_params`.

The FastPitch pitch predictor needs an F0 target for each frame. The `extra_preprocess` of the engine extracts F0 and caches it, aligned with the mel frames. The preprocess flag `--f0-method` selects the extractor. The default, `pyin`, uses `librosa.pyin`, which the `train` extra installs. `dio` and `harvest` use WORLD through `pyworld`, from the `train-pyworld` extra, and are faster. When `pyworld` is missing for `dio` or `harvest`, preprocess logs a warning and skips the F0 extraction. Training then runs, but the pitch predictor trains without its target loss. SpeedySpeech uses no pitch.

`--quality` (`x-low`, `medium` or `high`) scales `hidden_channels`, `hidden_channels_ffn` and the number of encoder and decoder layers. FastPitch and SpeedySpeech have two stages, so synthesis needs a separate vocoder. See the [FastPitch guide](engines/fastpitch.md).

## StyleTTS2 engine (`--engine styletts2`)

This engine ports the yl4579/StyleTTS2 recipe. It trains models from the start in other languages, and fine-tunes checkpoints on other speakers. The upstream model, loss and data code is in `phoonnx_train/styletts2/`. Install `phoonnx[train,train-styletts2]`.

Select the stage with `stage` in `engine_params`:

| Stage | What trains |
|---|---|
| `first` | The text aligner (TMA), the text encoder, the style encoder and the decoder, with mel reconstruction from the true F0 and energy. Adversarial and WavLM feature-matching losses start after `tma_epoch` |
| `second` | PL-BERT, the duration and prosody predictors, style diffusion after `diff_epoch`, and the joint decoder with SLM adversarial loss after `joint_epoch`. It starts from the stage 1 checkpoint (`first_stage_path`) |
| `finetune` | The `second` recipe with diffusion and the joint stage on from epoch 0, to adapt a checkpoint to another speaker or dataset |

These `engine_params` keys set the auxiliary models:

| Key | Model | Source |
|---|---|---|
| `asr_path` and `asr_config` | Text aligner (ASRCNN) | Upstream `Utils/ASR/`. TMA trains it further |
| `f0_path` | JDC pitch extractor | Upstream `Utils/JDC/bst.t7` |
| `plbert_dir` | PL-BERT | Upstream `Utils/PLBERT/` (English). Pretrained es and ca: [PL-BERT-wp-es](https://huggingface.co/BSC-LT/PL-BERT-wp-es). gl: [PL-ModernBERT-gl](https://huggingface.co/proxectonos/PL-ModernBERT-gl). Or train one with `--engine styletts2-plbert` |
| `slm.model` | WavLM for the SLM losses | The Hugging Face Hub, `microsoft/wavlm-base-plus` by default. Turn it off with `use_slm: false` |

Set `download_aux: true` to download the English aligner, pitch extractor and PL-BERT for each path that you do not set. The cache is `~/.cache/phoonnx/styletts2_aux_en`, or the same folder under `$XDG_CACHE_HOME`. Without it, a missing auxiliary model starts from random weights, and the log gives a warning.

```json
{"engine_params": {"stage": "second", "first_stage_path": "…", "plbert_dir": "…", "download_aux": true}}
```

StyleTTS2 uses the upstream list format in the dataset folder: `train_list.txt` and `val_list.txt`, with `filename.wav|phonemes|speaker_id` lines, and the audio in `wavs/` at 24 kHz.

The quality presets are `low` (half widths, iSTFTNet decoder), `medium` (the LJSpeech recipe, iSTFTNet decoder) and `high` (the LibriTTS recipe, HiFi-GAN decoder, ready for many speakers).

`export_onnx`, through `phoonnx_train/styletts2/export.py`, reads an upstream `.pth` or a Lightning `.ckpt`. It writes the two graphs that the `StyleTTS2Adapter` reads: `model.onnx` (tokens, style and speed to waveform) and `style_encoder.onnx` (reference waveform to style), and a `config.json`. A StyleTTS2 export needs no external vocoder.

### Training every StyleTTS2 step for another language

A model in another language needs a text aligner and a prosodic text encoder for that language first. `phoonnx_train` has an engine for each:

| Engine | Trains | Read through |
|---|---|---|
| `styletts2-aligner` | ASRCNN text aligner (CTC and s2s cross-entropy) | `asr_path` and `asr_config` |
| `styletts2-plbert` | PL-BERT (`backbone: albert` or `modernbert`) | `plbert_dir` |
| `styletts2-pitch` | JDC pitch extractor, with pyworld targets | `f0_path` |

All three can also fine-tune, with a warm start from `pretrained_path` or `pretrained_dir`.

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

`phoonnx_train/eval_loop.py` watches the training folder for Lightning checkpoints (`epoch=*.ckpt`) that it has not scored. It scores each one on a fixed set of held-out sentences, and synthesizes on the CPU so that training keeps the GPU. It needs `phoonnx[train-eval]`.

```bash
python -m phoonnx_train.eval_loop \
  --train-dir train_out --config train_out/config.json \
  --sentences heldout.txt --output-dir ./eval \
  --speaker-ref-dir my_dataset/wavs
```

The loop adds one row per epoch to `metrics.csv`. It writes the scores of each sentence for every epoch to `perutt/epoch<N>.csv`, with the columns `sentence`, `utmos` and `spk_sim`, so you can see one sentence get worse while the mean stays the same. It keeps the audio of the best epoch in `samples/epoch<N>/`. After a restart it skips the epochs that it scored and the epochs that failed.

It reports two metrics. UTMOS (`utmos_*`) predicts a 1 to 5 mean opinion score. Speaker similarity (`spk_sim_*`) is the cosine similarity to the centroid of the reference recordings of the target speaker. Speaker similarity needs `speakeronnx` and `--speaker-ref-dir`. Without them, the loop scores UTMOS only.

## Evaluation and early stopping

The `phoonnx_train.evaluation` package holds the evaluation code: `CheckpointScorer`, `SelectionPolicy`, `MetricsTracker` and the Lightning callbacks. The same code runs in two modes.

### Sidecar mode (out-of-process scoreboard)

`phoonnx_train/eval_loop.py` is a command that runs the package in its own process. It loads each checkpoint through the training engine registry. `--engine` defaults to `auto`, which reads the `engine` key in `config.json` and uses `vits` when the key is missing. It synthesizes on the CPU and seeds each sentence: `torch.manual_seed(seed + i)` runs before sentence `i`, so one checkpoint scored twice gives the same audio and the same scores.

| Option | Default | Description |
|---|---|---|
| `--once` | off | Do one pass and stop |
| `--poll SECONDS` | `60` | The time between two scans for checkpoints |
| `--dry-run` | off | Load the config, phonemize and encode the sentences, and stop |
| `--noise-scale`, `--length-scale`, `--noise-w` | from `config.json` | Replace the inference scales of the config |
| `--early-stop-patience N` | `0` | After N scored epochs with no better score, write a `stop.flag` into `--train-dir`. `0` turns it off |
| `--min-spk-sim FLOAT` | none | The floor of the similarity gate (see [The similarity gate](#the-similarity-gate)) |
| `--speaker-id INT` | none | The speaker id for a multi-speaker model. Without it, the log gives a warning and synthesis uses the default voice of the engine |
| `--engine NAME` | `auto` | The training engine |
| `--vocoder PATH` | none | A vocoder that the engine synthesis uses. The VITS engine ignores it |
| `--seed INT` | `1234` | The base seed of the seeding for each sentence |
| `--num-ref-wavs INT` | `60` | The number of reference recordings in the speaker centroid |

### In-training mode (early stopping callback)

`train.py` has the opt-in flags `--eval-sentences FILE`, `--eval-every N`, `--early-stop-patience N`, `--eval-speaker-ref-dir DIR`, `--min-spk-sim FLOAT`, `--eval-speaker-id INT` and `--eval-seed`. When you give `--eval-sentences` and an `--eval-every` above 0, `EvalScoreboardCallback` runs the same `CheckpointScorer` in the training process every N epochs, after `ModelCheckpoint` saves. It writes the same scoreboard to `<run-dir>/eval/`, and sets `trainer.should_stop` when the patience runs out. A scoring failure does not stop training. The callback logs it and continues. An engine that cannot synthesize for evaluation makes `train.py` stop at the start with an error that lists the engines that can.

The trainer always has `StopFileCallback`, also with evaluation off. It watches `<run-dir>/stop.flag` and stops training when the file appears. So an `eval_loop` sidecar with `--early-stop-patience` can stop a training run in another process. With no flag file, it does nothing.

### The similarity gate

`SelectionPolicy(metric="utmos_mean", min_spk_sim=...)` selects the best checkpoint by UTMOS, gated on speaker similarity. UTMOS alone can prefer a checkpoint whose voice moved away from the target speaker: more natural, but the wrong speaker. When speaker scoring is on and you give `--min-spk-sim`, a checkpoint must reach that floor to be a candidate. Among the candidates, the higher UTMOS wins. With no speaker score, because there is no reference folder or no `speakeronnx`, the selection uses UTMOS only and logs a warning.

### Artifacts and the stop.flag contract

| File | Content |
|---|---|
| `metrics.csv` | One row per scored epoch. The loop also reads and extends a file with fewer columns |
| `best.json` | The epoch, the step, the checkpoint path and all scores of the best checkpoint |
| `best.ckpt` | A copy of the best checkpoint, not a symbolic link, so it stays when old checkpoints are deleted |
| `perutt/epoch<N>.csv` | The scores of each sentence for every scored epoch (`sentence`, `utmos`, `spk_sim`) |
| `samples/epoch<N>/` | The synthesized audio of the best epoch, and its `perutt.csv` |
| `failed.json` | The checkpoints that failed. A checkpoint that fails to load or score 3 times is recorded here, logged as an ERROR, and skipped from then on |
| `stop.flag` | One line with the reason. The file means "stop training". `StopFileCallback` reads the reason, logs it and stops at the end of the epoch |

## Writing a feature table (features.json)

A feature-input text encoder reads one distinctive-feature vector per phone id from a `features.json` file. `phoonnx_train.features_json` writes that file from a voice `config.json` or from a bare `phoneme_id_map` JSON. Install its feature sources first:

```bash
uv pip install --prerelease=allow 'phoonnx[train-features]'
```

The `--source` option selects the feature table: `phonematcher` (40 features, the last four for tone) or `orthography2ipa` (the 23 features of its `feats.py`). Each value is 1 (plus), -1 (minus) or 0 (does not apply). The file also records the source version and the sha256 of the map, so a text encoder can refuse a table made for a different map.

Seven prosody features end every vector. A space sets `pros_word_boundary`, and `ˈ` and `ˌ` set the two stress rows. A punctuation character sets rows by its Unicode name, so the rule reads every script:

- COMMA, SEMICOLON, COLON, ANO TELEIA or SECTION MARK sets `pros_pause_minor`.
- FULL STOP, QUESTION, EXCLAMATION, INTERROBANG, DANDA or ELLIPSIS sets `pros_pause_major`, unless the name also has INVERTED.
- QUESTION or INTERROBANG sets `pros_question`, and EXCLAMATION or INTERROBANG sets `pros_exclamation`.

So `?` and `؟` set a major pause and a question, and `¿` sets only a question. `⁈`, `⁉` and `‽` set a major pause, a question and an exclamation. U+1361 ETHIOPIC WORDSPACE sets `pros_word_boundary`. The IPA major-group bar `‖` sets `pros_pause_major`. The click letters `ǀ` and `ǁ` are phones. `.` sets `pros_pause_major`. For a voice that writes `.` as the IPA syllable break, `--non-phone .` makes it `null`. A phone is 0 on all seven rows, and a prosody symbol is 0 on every phone feature.

Four tone features, `tone_is`, `tone_start`, `tone_mid` and `tone_end`, give the pitch of a tone. A tone has `tone_is` 1 and its pitch at the start, middle and end, where each Chao level 1 to 5 becomes (level − 3) / 2. So `˥` gives 1, 1, 1, `˧˥` gives 0, 0.5, 1, and `⁵¹` gives 1, 0, −1. Tone numerals and left-stem letters read as the Chao letters of phonematcher's `NUMERALS` table. Numerals read as Chao levels, where 5 is high. A corpus that writes 1 as the high tone, as some Africanist and Otomanguean sources do, must be converted before generation, or its pitch comes out inverted.

The phonematcher source has the four tone features in its own vector, before the prosody block. Its vector of a vowel or a syllabic consonant (marked with U+0329) holds the tone of that phone. The tone comes from the tone letters after the phone, as in `a˥˩`, or from IPA tone diacritics, as in `á`, `ǎ`, `ȁ` or `ḿ̩`. So `á` gets the tone 1, 0.5, 0.5, 0.5, and `a˥˩` gets 1, 0, −1. A tone mark on any other consonant, such as Slavic `ś`, is not a tone, and the command refuses the symbol. A phone with no tone is 0 on all four.

A tone id holds tone letters and no phone, such as `˧˧` after a coda in `tam˧˧`, or the second tone `꜔꜒` of the sandhi form `ni˨˩˦꜔꜒`. phonematcher's `tone_levels` reads it. The id gets its tone in the four tone features, and 0 on every phone and prosody feature.

The orthography2ipa source has no tone features, so the table adds the four after the prosody block. A toned phone gets the orthography2ipa features of the phone without its tone, and the tone of its marks or tone letters. phonematcher reads that tone, so the tone features need phonematcher with both sources. The `train-features` extra installs both. The `source` object records the phonematcher version under `tone`, for both sources.

Pad, BOS, EOS, digits, `|` and punctuation that matches no rule get `null`. The command prints each punctuation character with its name and rows, and each null id with its reason. When a symbol looks like a phone and the source cannot read it, the command writes nothing and lists each such symbol with its code points. Fix the source, or rule the symbol a non-phone with `--non-phone SYMBOL` (repeatable).

U+00B7 MIDDLE DOT gets `null`, because a voice map uses it more often for a syllable boundary or the Catalan l·l than for a pause. NFC turns U+0387 GREEK ANO TELEIA into U+00B7. For a voice where it is a pause, `--pause SYMBOL` (repeatable) rules the symbol `pros_pause_minor`. The same applies to `|`, which voice maps also use as a word or syllable separator. A `--pause` value is one character. The command refuses an empty or longer value before it reads the map, and it writes the file only after every check passes.

```bash
echo '{"_":[0],"a":[1],"ɐ̃":[2],"ˈ":[3]," ":[4],"?":[5],"،":[6],"§":[7],"˧˥":[8]}' > small_map.json
python -m phoonnx_train.features_json small_map.json --source phonematcher --out small_features.json
```

The command prints this report. The last line names the installed phonematcher version.

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
wrote small_features.json (phonematcher <version>, 47 features, 7 prosody, 4 tone)
```

A codepoint-level map has a separate id for each combining mark and modifier letter, and the source reads none of them alone. Such a map is refused. Use a map with one id per whole segment. orthography2ipa's `feats.py` does not read tie bars, so with `--source orthography2ipa` the command refuses a tied segment such as `d͡ʒ`, `t͡ʃ` or `k͜p`. A tied affricate or double articulation reads with `--source phonematcher`. phonematcher reads tied affricates, but it refuses a tied diphthong such as `a͡i`, because a diphthong is two segments.

### Training with a feature table

Give the table with `--input-features` and choose a layer with `--input-layer`. Only the `vits` engine reads them:

```bash
python -m phoonnx_train.train --dataset-dir train_out --default-root-dir train_out/runs \
  --input-layer residual --input-features train_out/features.json
```

| Layer | Input vector of an id |
|---|---|
| `residual` | A linear projection of its features, plus one learned vector per id |
| `composition` | A linear projection of its features, plus a learned vector for its base symbol and one for each of its marks. `ɐ̃` is `ɐ` plus U+0303, and `ç` is a base of its own |

Every learned table starts at zero. A phone that never occurs in training keeps the projection of its features, so it lands near the trained phones that share them. A `null` id, such as pad, BOS, EOS or the `|` and `‖` group marks, gets only its learned part. The learned part is scaled as the id embedding is.

The table must come from the dataset's own map. Training stops when the table's `phoneme_id_map_sha256` differs from the hash of the map in `config.json`, when the map has an id that the table does not list, or when a vector has the wrong length.

Checkpoints and export follow these rules:

- The checkpoint keeps the table. A resume with another table or another input layer stops with a message.
- A feature model warm-starts from an id checkpoint with `--discard-encoder`. Every weight except the text-encoder input loads, and the feature layer starts as built.
- An id model reads a feature checkpoint through `--resume-from-single-speaker-checkpoint` or a weights-only file, and then loads the layer as its id table. A Lightning checkpoint given to `--resume-from-checkpoint` resumes the run, so the input layer must match. A feature model loads the projection when the feature names match. It loads the per-id, base and mark tables only when the symbols match. `--discard-encoder` keeps the whole layer as built.
- Export turns the layer into a plain id embedding with the same values and checks that the two match. The exported voice runs as any VITS voice, and the runtime needs no feature table.

## Training a Vocos vocoder

A two-stage engine, such as Matcha, makes a mel and needs a separate vocoder. `phoonnx_train/train_vocos.py` trains a Vocos vocoder, so a voice can have a vocoder trained or fine-tuned on the audio of the same speaker.

The data is audio only, with no transcripts: any folder tree of `.wav`, `.flac`, `.ogg` or `.mp3` files, such as the `cache/` folder of preprocess. You can give `--audio-dir` more than once. A fine-tune needs less audio than a model trained from the start, or use `--warm-start`.

The vocoder trains on the mel of the phoonnx acoustic models: `n_fft=1024`, `hop=256`, `win=1024`, 80 mels, `fmax=8000`, log-mel. A vocoder trained on other settings gives noise for every phoonnx mel model.

```bash
python -m phoonnx_train.train_vocos \
  --audio-dir /path/to/voice/wavs \
  --sample-rate 22050 --batch-size 16 --crop-seconds 1.0 \
  --warm-start charactr/vocos-mel-24khz \
  --max-epochs 100 --default-root-dir vocos_train
```

`--warm-start` takes a local `.ckpt` or `.bin` file, or a Hugging Face repo id with reference Vocos weights. The command copies the parameters that match and logs the fraction that matched. `--resume-from-checkpoint` continues an interrupted run. The command also takes the `train.py` flags `--checkpoint-epochs`, `--devices`, `--accelerator`, `--precision`, `--num-workers` and `--seed`.

Export the vocoder:

```bash
python -m phoonnx_train.export_vocos \
  vocos_train/lightning_logs/version_0/checkpoints/epoch=99-step=12345.ckpt \
  vocoder.onnx
```

This writes `vocoder.onnx` and `vocoder.onnx.json`, which holds the STFT parameters. The vocoder registry detects this layout as `vocos`. `--seconds` (default `2.0`) sets the length of the random mel that the export uses for its parity check. To use the vocoder with a voice that makes a mel, set `engine_params={"vocoder_path": "vocoder.onnx"}`.

## config.json reference

Preprocess writes a `config.json` that training and inference read. These are its fields:

| Field | Meaning |
|---|---|
| `dataset` | The dataset name, from `--dataset-name` or the parent folder of the output |
| `audio.sample_rate` | The sample rate of training and inference |
| `audio.quality` | A free label, from `--audio-quality` or the name of the output folder |
| `lang_code` | The language code of the phonemizer, as `--language` gives it |
| `inference.noise_scale`, `length_scale`, `noise_w` | The default inference scales |
| `inference.add_diacritics` | Add diacritics at inference, as preprocess did with `--add-diacritics` |
| `alphabet` | The phoneme alphabet (`ipa`, `unicode`, `arpa`, `pinyin` and others) |
| `phoneme_type` | The phonemizer (`espeak`, `gruut`, `byt5` and others) |
| `phonemizer_model` | The model or variant, for the phonemizers that take one |
| `phoneme_id_map` | The map from phoneme symbol to integer id |
| `num_symbols` | The size of the phoneme map |
| `num_speakers` | The number of speakers, 1 for a single speaker |
| `speaker_id_map` | The map from speaker label to id, for many speakers |
| `phoonnx_version` | The phoonnx version that preprocessed the data |
| `keep_punctuation` | Whether the phonemes keep clause punctuation (`, . ? ! : ;`) |
| `text_casing` | The casing of the text before phonemization (`ignore`, `lower`, `upper` or `casefold`, from `--text-casing`). Synthesis applies the same casing |
| `fold_compounds` | `true` with `--segment-map`, so the tokenizer finds the longest segment first |
| `phonemizer_registry`, `phonemizer_registry_version` | The distribution that resolved the phonemizer, and its version |
| `phonemizer_identity` | The engine that phonemized the rows. Its keys are `backend`, `distribution`, `version` (for espeak-ng, the binary), `lang` (the resolved lect), `revision` and `frontend`. `revision` is the scriptconv hash of the source, package versions, engine release and data files that make the phonemes. `frontend` is null, because preprocess reads the identity from the backend. The object is empty when every row came from a phonemes column |
| `phonemes_phonemized`, `phonemes_from_column` | The number of rows that preprocess phonemized, and the number that came from a phonemes column |

The [Configuration reference](../configuration.md) describes the full loaded form, with the tokenizer flags, the special tokens and `engine_params`.

---

[Docs index](../index.md) · Previous: [Preprocess reference](preprocess.md) · Next: [Export to ONNX](export.md)
