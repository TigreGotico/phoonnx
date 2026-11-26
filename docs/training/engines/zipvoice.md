# ZipVoice Engine

<!-- agent-hint
covers: how to clone a voice with ZipVoice, configure its graphs, and train and export a ZipVoice model
source: phoonnx/engines/zipvoice.py, phoonnx/voice.py, phoonnx_train/engines/zipvoice.py, phoonnx_train/zipvoice/lightning.py, scripts/conversion/zipvoice/
generated: no
verify: ! comm -23 <(grep -oP -- '(?<![\w-])--[a-z][a-z0-9-]*' docs/training/engines/zipvoice.md | grep -vx -- --pre | sort -u) <(for m in preprocess train export_onnx; do .venv/bin/python -m phoonnx_train.$m --help; done | grep -oP -- '(?<![\w-])--[a-z][a-z0-9-]*' | sort -u) | grep . && .venv/bin/python -W ignore -c "import inspect, re, sys; from phoonnx.engines.zipvoice import ZipVoiceAdapter; from phoonnx_train.zipvoice.lightning import ZipVoiceModule; p = open('docs/training/engines/zipvoice.md').read(); src = open('phoonnx/engines/zipvoice.py').read(); d = {**ZipVoiceAdapter().default_params(), 't_shift': float(re.search(r'get\(.t_shift., ([0-9.]+)\)', src).group(1)), 'target_rms': ZipVoiceAdapter.TARGET_RMS, 'feat_scale': ZipVoiceAdapter.FEAT_SCALE}; d.update({k: v.default for k, v in inspect.signature(ZipVoiceModule).parameters.items()}); rows = re.findall(r'^\| \x60(\w+)\x60 \| \x60([0-9.]+)\x60 \|', p, re.M); bad = [(k, v, d.get(k)) for k, v in rows if d.get(k) is None or float(v) != float(d[k])]; keys = re.findall(r'^    .(\w+).: .', p, re.M); bad += [k for k in keys if k not in re.findall(r'ep(?:\.get\(|\[).(\w+).', src)]; print(len(rows), len(keys), bad); sys.exit(bool(bad) or not rows or not keys)"
see: ../../cloning.md, f5tts.md, chatterbox.md, ../training.md
-->

This page shows how to clone a voice with ZipVoice, how to configure its graphs, and how to train and export a ZipVoice model.

See also the [training reference](../training.md), the [adapter architecture](../../engines.md), the [configuration reference](../../configuration.md), [vocoders](../../vocoders.md) and [voice cloning](../../cloning.md).

## What it is

ZipVoice is a zero-shot flow-matching text-to-speech model from [k2-fsa](https://github.com/k2-fsa), with about 123 million parameters. It clones a voice in context: from a short reference clip and its transcript, it speaks other text in that voice. It is an iterative engine. It does not make audio in one pass through one graph. It runs a short ODE sampling loop over a flow-matching vector field.

Use ZipVoice for zero-shot cloning when you have the transcript of the reference. The d-vector engines (YourTTS, StyleTTS2) and [Chatterbox](chatterbox.md) clone from the audio alone. See [Voice cloning](../../cloning.md).

## Extras needed

Cloning needs `pip install --pre "phoonnx[cloning]"` (`soundfile` and `scipy`) to load and resample the reference. Training needs `pip install --pre "phoonnx[train]"`. Add `train-resample` when you preprocess the dataset at a sample rate other than 24 kHz.

## Clone a voice

ZipVoice clones in context, so it needs the reference audio and its transcript:

```python
voice.synthesize("A line the reference never spoke.", SynthesisConfig(reference=SpeakerReference(
    audio="alice.wav",
    text="hello, this is the reference clip",
    lang="en",   # set for a non-target-language reference
)))
```

`TTSVoice` gives the cloning fields to the adapter as two parameters:

- `reference.audio` becomes `reference_audio`. The mel frontend turns it into the `speech_condition` (the prompt mel).
- `reference.text` becomes `prompt_tokens`. The `text_encoder` reads them with the target `tokens`, as the alignment anchor.

`reference.lang` selects the phonemizer of the reference text when it differs from the target language. Without a reference audio or a transcript, the adapter raises an error. Audio alone cannot replace the transcript.

## Configure the graphs

`ZipVoiceAdapter` (`phoonnx/engines/zipvoice.py`) overrides `BaseOnnxAdapter.synthesize()`. This replaces the default flow (`build_feed_dict`, one run, `parse_outputs`) with a loop over several graphs. The primary session of the voice is the `fm_decoder`. `configure()` loads the mel graph, the text encoder and the Vocos vocoder from `engine_params`:

```python
engine_params = {
    "text_encoder_path": "text_encoder.onnx",
    "mel_path": "zipvoice_mel.onnx",
    "vocoder_path": "vocos_24khz.onnx",
    "vocoder_type": "vocos",
}
```

`engine_params` can also set `num_step` and `guidance_scale`. A synthesis call can set `speed`, `num_step`, `guidance_scale` and `t_shift` through `SynthesisConfig.extra`. The defaults are:

| Parameter | Default | Effect |
|---|---|---|
| `num_step` | `16` | Number of ODE steps |
| `guidance_scale` | `1.0` | Classifier-free guidance scale |
| `speed` | `1.0` | Speaking rate, given to the text encoder |
| `t_shift` | `0.5` | Warp of the time grid |

The upstream ONNX weights are in [`k2-fsa/ZipVoice`](https://huggingface.co/k2-fsa/ZipVoice): `zipvoice/text_encoder.onnx`, `zipvoice/fm_decoder.onnx` and `tokens.txt`, a vocabulary of 360 espeak and pinyin tokens. The `zipvoice_distill/` folder holds ZipVoice-Distill, a version trained for 4 to 8 steps. Set `num_step` to a value in that range for it. The Vocos vocoder is `vocos_24khz.onnx` in `jasonzhang76/zipvoice`. `scripts/conversion/zipvoice/export_mel.py` writes the mel graph.

## Architecture

ZipVoice joins a Zipformer backbone (the k2-fsa Transformer variant that gives ZipVoice its name) with conditional flow matching. It has no speaker encoder. The model puts the reference mel and the reference text before the target. It rebuilds the reference part and makes the target part in the same voice. The reference text anchors the alignment between the reference audio and its phonemes.

Three ONNX graphs and the vocoder make the runtime pipeline:

```
text  ─ espeak / pypinyin ─►  tokens ─┐
ref.wav ─► mel (VocosFbank) ──────────┤
                                       ▼
            text_encoder ─► text_condition ─┐
            speech_condition (prompt mel) ──┼─► fm_decoder ×N (Euler ODE) ─► mel ─► Vocos ─► audio
            x₀ ~ N(0,1) ────────────────────┘
```

### Graph inputs and outputs

The `text_encoder` graph reads the tokens and gives `text_condition`:

| Name | Type | Shape |
|---|---|---|
| `tokens` | int64 | `[1, T_text]` |
| `prompt_tokens` | int64 | `[1, T_prompt_text]` |
| `prompt_features_len` | int64 | scalar |
| `speed` | float32 | scalar |
| output `text_condition` | float32 | `[1, T, 100]` |

The `fm_decoder` graph is the flow-matching vector field. Classifier-free guidance is inside the graph:

| Name | Type | Shape |
|---|---|---|
| `t` | float32 | scalar (the ODE time) |
| `x` | float32 | `[1, T, 100]` (the sample) |
| `text_condition` | float32 | `[1, T, 100]` |
| `speech_condition` | float32 | `[1, T, 100]` (the prompt mel, padded) |
| `guidance_scale` | float32 | scalar |
| output `v` | float32 | `[1, T, 100]` (the vector field) |

The Vocos vocoder reads `mels[1, 100, T]` and gives `mag`, `x` and `y`, the STFT parts. The [`vocos` vocoder](../../vocoders.md) of the registry inverts them.

### The sampling loop

The adapter warps a uniform time grid with `t_shift`, so more steps fall early in the path, and then takes forward Euler steps:

```python
steps = linspace(0, 1, num_step + 1)
steps = t_shift * steps / (1 + (t_shift - 1) * steps)
x = randn(1, T, 100)
for i in range(num_step):
    v = fm_decoder(t=steps[i], x, text_condition, speech_condition, guidance_scale)
    x = x + v * (steps[i+1] - steps[i])      # forward Euler
```

The adapter seeds the noise with 0, so one input always gives the same audio.

### The mel frontend

The reference mel is the Vocos log-mel: `log(clamp(MelSpectrogram(24 kHz, n_fft 1024, hop 256, n_mels 100, power=1), 1e-7))`. `scripts/conversion/zipvoice/export_mel.py` writes it as an ONNX graph that computes the DFT with a conv1d, so the runtime needs no torch. On one second of noise, its output has a cosine similarity of 1.0 with the torchaudio mel.

Two constants control the level. The adapter normalizes the prompt audio to an RMS of `target_rms` before the mel. The model works in a feature space scaled by `feat_scale`: the adapter multiplies the prompt mel by it and divides the generated mel by it before the vocoder. When the prompt is quieter than `target_rms`, the adapter scales the output audio down by the same ratio.

| Constant | Value |
|---|---|
| `target_rms` | `0.1` |
| `feat_scale` | `0.1` |

## Train

The training engine is `--engine zipvoice`. `phoonnx_train/zipvoice/` holds the upstream model and recipe (Apache-2.0): the TTSZipformer text encoder and the flow-matching decoder (`model.py`, `zipformer.py`, `scaling.py`, `solver.py`), the ScaledAdam optimizer and the Eden schedule (`optim.py`, `lr_scheduler.py`), and the Vocos log-mel features (`feature.py`, 24 kHz, 100 bins, hop 256). The imports are package-relative, and the phoonnx `dataset.jsonl` replaces the upstream `lhotse` manifests.

```bash
# 1. shared preprocessing (any sample rate; audio is resampled to 24 kHz
#    for feature extraction)
python -m phoonnx_train.preprocess --language en-US --input-dir data \
    --output-dir training --sample-rate 24000

# 2. train (quality: base = upstream ZipVoice size, low = CI/smoke tier)
python -m phoonnx_train.train --dataset-dir training --engine zipvoice --quality base

# 3. export the two graphs that the adapter reads
python -m phoonnx_train.export_onnx --engine zipvoice \
    --config training/config.json --output-dir exported <checkpoint>
```

Each training batch follows upstream `train_zipvoice.py`:

1. It samples `t` from U(0, 1) and Gaussian noise.
2. It masks a random 70 to 100 percent span of each utterance as the target. The rest is the speech prompt that the model fills in from.
3. It drops the text condition for a `condition_drop_ratio` share of the items, for classifier-free guidance.
4. It regresses the straight-line vector field. The loss covers only the target frames that are not padding.

The optimizer is ScaledAdam with the Eden schedule. These are the defaults of `ZipVoiceModule`:

| Option | Default |
|---|---|
| `base_lr` | `0.02` |
| `clipping_scale` | `2.0` |
| `lr_batches` | `7500` |
| `lr_epochs` | `10` |
| `condition_drop_ratio` | `0.2` |
| `batch_size` | `8` |

To fine-tune from the released checkpoint, give it to `--resume-from-checkpoint`. A file that Lightning did not write loads as a warm start through the engine's `load_checkpoint`. It accepts the upstream `{"model": ...}` layout and loads each tensor whose name and shape match.

### Export

`export_onnx` writes `text_encoder.onnx` and `fm_decoder.onnx`, with the inputs and outputs in the tables above and classifier-free guidance inside the decoder graph. It follows upstream `onnx_export.py`: opset 13, with the scaled modules converted before tracing. When the dataset `config.json` exists, it also writes a voice config with `engine` set to `zipvoice`, the sample rate 24000 and the phoneme map.

The export does not write the mel graph or the vocoder. Add `zipvoice_mel.onnx` and `vocos_24khz.onnx`, and set the `engine_params` keys as in [Configure the graphs](#configure-the-graphs).

## Upstream

| | |
|---|---|
| Repo | [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) |
| Paper | *ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching*, [arXiv:2506.13053](https://arxiv.org/abs/2506.13053) |
| Languages | English, Chinese |
| ONNX weights | [`k2-fsa/ZipVoice`](https://huggingface.co/k2-fsa/ZipVoice) (text encoder and flow decoder), [`jasonzhang76/zipvoice`](https://huggingface.co/jasonzhang76/zipvoice) (Vocos vocoder) |

---

[← x-vits](xvits.md) · [Home](../../index.md) · [Chatterbox →](chatterbox.md)
