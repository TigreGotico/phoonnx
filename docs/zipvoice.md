# ZipVoice Engine

**ZipVoice** is a compact (~123M parameter) **flow-matching**, zero-shot
text-to-speech model from [k2-fsa](https://github.com/k2-fsa) (the team behind
sherpa-onnx / icefall / next-gen Kaldi). It does **in-context voice cloning**: given a
short reference clip *and its transcription*, it speaks new text in that voice with
state-of-the-art speaker similarity. It is phoonnx's first **iterative** engine — instead
of a single `tokens → graph → audio` pass, it runs a short ODE sampling loop over a
flow-matching vector field.

## Upstream

| | |
|---|---|
| Repo | <https://github.com/k2-fsa/ZipVoice> |
| Paper | *ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching* — [arXiv:2506.13053](https://arxiv.org/abs/2506.13053) |
| Dialogue variant | *ZipVoice-Dialog* — [arXiv:2507.09318](https://arxiv.org/abs/2507.09318) |
| Languages | English, Chinese |
| ONNX weights | [`k2-fsa/ZipVoice`](https://huggingface.co/k2-fsa/ZipVoice) (text encoder + flow decoder), Vocos vocoder on HF |

## Architecture

ZipVoice pairs a **Zipformer** backbone (k2-fsa's efficient Transformer — the "Zip" in
ZipVoice) with **conditional flow matching**. Rather than a speaker encoder / d-vector,
cloning is **in-context (infilling)**: the reference mel + reference text are prepended
to the target, the model reconstructs the reference region and generates the target
region in the same voice. The reference's text is the alignment anchor between its audio
and phonemes.

Three ONNX graphs make up the runtime pipeline:

```
text  ─ espeak / pypinyin ─►  tokens ─┐
ref.wav ─► mel (VocosFbank) ──────────┤
                                       ▼
            text_encoder ─► text_condition ─┐
            speech_condition (prompt mel) ──┼─► fm_decoder ×N (Euler ODE) ─► mel ─► Vocos ─► audio
            x₀ ~ N(0,1) ────────────────────┘
```

### ONNX I/O

**`text_encoder`** — `tokens, prompt_tokens, prompt_features_len, speed → text_condition`

| Name | Type | Shape |
|---|---|---|
| `tokens` | int64 | `[1, T_text]` |
| `prompt_tokens` | int64 | `[1, T_prompt_text]` |
| `prompt_features_len` | int64 | scalar |
| `speed` | float32 | scalar |
| → `text_condition` | float32 | `[1, T, 100]` |

**`fm_decoder`** — the flow-matching vector field (classifier-free guidance is internal)

| Name | Type | Shape |
|---|---|---|
| `t` | float32 | scalar (current ODE time) |
| `x` | float32 | `[1, T, 100]` (current sample) |
| `text_condition` | float32 | `[1, T, 100]` |
| `speech_condition` | float32 | `[1, T, 100]` (prompt mel, padded) |
| `guidance_scale` | float32 | scalar |
| → `v` | float32 | `[1, T, 100]` (the vector field) |

**Vocos vocoder** — `mels[1, 100, T] → mag, x, y` (STFT coefficients, inverted by the
existing [`vocos` vocoder](vocoders.md) in the registry).

### The sampling loop

Flow matching integrates `x` from noise (`t=0`) to data (`t=1`):

```python
steps = linspace(0, 1, num_step + 1)
x = randn(1, T, 100)
for i in range(num_step):
    v = fm_decoder(t=steps[i], x, text_condition, speech_condition, guidance_scale)
    x = x + v * (steps[i+1] - steps[i])      # forward Euler
```

`num_step` defaults to **16** (configurable); **ZipVoice-Distill** is trained for few-step
sampling and runs well at **4–8**.

### The mel front end

The reference mel is the yl4579/Vocos log-mel — `log(clamp(MelSpectrogram(24 kHz,
n_fft 1024, hop 256, n_mels 100, power=1), 1e-7))` — reimplemented in
`scripts/conversion/zipvoice/export_mel.py` as a conv1d-DFT ONNX so the runtime needs no
torch (numerically identical to torchaudio, cosine 1.0).

> **Two normalization constants matter.** The prompt wav is RMS-normalized to
> `target_rms = 0.1` before the mel, and the model works in feature space scaled by
> `feat_scale = 0.1` — the prompt mel is multiplied by it and the generated mel is
> **divided** by it before the vocoder. Both are handled inside the adapter.

## The adapter

`ZipVoiceAdapter` (`phoonnx/engines/zipvoice.py`) overrides
`BaseOnnxAdapter.synthesize()` — the hook that lets an engine replace the default
single-graph `build_feed_dict → run → parse_outputs` with its own multi-ONNX loop. The
voice's primary session is the `fm_decoder`; the mel + text-encoder graphs and the Vocos
vocoder are loaded from `engine_params` in `configure()`:

```python
engine_params = {
    "text_encoder_path": "text_encoder.onnx",
    "mel_path": "zipvoice_mel.onnx",
    "vocoder_path": "vocos_24khz.onnx",
    "vocoder_type": "vocos",
}
```

## Cloning

ZipVoice is an in-context engine, so it needs the reference **and its transcription**:

```python
voice.synthesize("A line the reference never spoke.", SynthesisConfig(
    speaker_reference="alice.wav",
    speaker_reference_text="hello, this is the reference clip",
    speaker_reference_lang="en",   # set for a non-target-language reference
))
```

See [Voice Cloning](cloning.md) for the full API, cross-lingual cloning, and how it
contrasts with the d-vector engines (YourTTS, StyleTTS2).

## Variants

| Variant | Notes |
|---|---|
| **ZipVoice** | base flow-matching model (`num_step` ~16) |
| **ZipVoice-Distill** | distilled for few-step sampling (`num_step` 4–8), minimal quality loss |
| **ZipVoice-Dialog** | two-party spoken-dialogue generation ([arXiv:2507.09318](https://arxiv.org/abs/2507.09318)) |

## Training

`phoonnx_train --engine zipvoice` trains ZipVoice end-to-end. The upstream
model and recipe are vendored in `phoonnx_train/zipvoice/` (Apache-2.0):
the TTSZipformer text encoder + flow-matching decoder (`model.py`,
`zipformer.py`, `scaling.py`, `solver.py`), the ScaledAdam optimizer and
Eden schedule (`optim.py`, `lr_scheduler.py`), and the Vocos log-mel
feature extractor (`feature.py`, 24 kHz / 100 bins / hop 256) — imports
made package-relative, the `lhotse` manifest pipeline replaced by phoonnx's
own `dataset.jsonl`.

### Recipe

```bash
# 1. shared preprocessing (any sample rate; audio is resampled to 24 kHz
#    for feature extraction)
python -m phoonnx_train.preprocess --language en-US --input-dir data \
    --output-dir training --sample-rate 24000

# 2. train (quality: base = upstream ZipVoice size, low = CI/smoke tier)
python -m phoonnx_train.train --dataset-dir training --engine zipvoice

# 3. export the two-graph ONNX contract the adapter consumes
python -m phoonnx_train.export_onnx --engine zipvoice ...
```

Per training batch (matching upstream `train_zipvoice.py`): sample
`t ~ U(0,1)` and Gaussian noise, mask a random 70–100% span of each
utterance as the generation target (the remainder is the speech prompt the
model in-fills from), drop the text condition for `condition_drop_ratio`
(default 0.2) of items for classifier-free guidance, and regress the
straight-line vector field with the loss restricted to target ∩ non-padded
frames. Optimizer: ScaledAdam (`base_lr` 0.02, `clipping_scale` 2.0) with
the Eden schedule (`lr_batches` 7500, `lr_epochs` 10) and the Zipformer
batch-count schedules driven from the global step.

Fine-tuning from the released 123M checkpoint: convert the upstream
checkpoint keys with `load_checkpoint` (`--resume-from-checkpoint` accepts
upstream `{"model": ...}` layouts) and train on your data.


### Export

`export_onnx` emits `text_encoder.onnx` (tokens + prompt_tokens +
prompt_features_len + speed → text_condition) and `fm_decoder.onnx`
(t, x, text_condition, speech_condition, guidance_scale → v, with
classifier-free guidance folded into the graph) — the exact contract
`ZipVoiceAdapter` (`phoonnx/engines/zipvoice.py`) consumes, matching
upstream `onnx_export.py` (opset 13, scaled-module conversion applied
before tracing).

### Downstream config

Nothing changes for inference/downstream consumers: a ZipVoice voice — whether the
stock HF checkpoint or a fine-tune produced via the path above — is configured exactly
as described in [The adapter](#the-adapter) and [Cloning](cloning.md); training status
has no bearing on the runtime `engine_params` shape.
