# Streaming VITS Engine

Standard Piper/VITS voices synthesize a whole sentence in one ONNX call, so the
first audio sample is only available once the entire sentence is decoded. The
**streaming** engine lowers *time-to-first-audio* (TTFA) by decoding the
sentence in chunks and emitting each chunk as soon as it is ready.

The `VitsStreamingAdapter` handles this. It is auto-selected for any voice whose
config declares `"streaming": true` **and** ships a separate decoder graph.

## The core requirement: you need a *split* model

> **Streaming only works with a model that is split into two ONNX graphs:**
> a separate **encoder** (`encoder.onnx`) and **decoder** (`decoder.onnx`),
> with `"streaming": true` in the voice config.
>
> A normal single-file Piper/VITS `.onnx` **cannot stream** — there is no point
> in the pipeline to cut. Loading a combined model will simply fall through to
> the ordinary [`VitsAdapter`](./engines.md), which is correct but not streaming.

There are three ways to get a split model:

0. **Let phoonnx split a normal model at load time (recommended).** You do *not*
   need a specially-exported voice. If a voice config just sets
   `"streaming": true` on an ordinary single-graph Piper/VITS model, phoonnx
   splits it into an encoder/decoder pair on first load and caches the two
   graphs next to the model (`<model>.encoder.onnx` / `<model>.decoder.onnx`).
   See [Auto-split](#auto-split-no-re-export-needed). This is lossless by
   construction and sidesteps the re-export hazards below.

1. **Use a pre-split "+RT" voice.** The Sonata project publishes fast/streaming
   variants of many Piper voices in the
   [`mush42/piper-rt`](https://huggingface.co/datasets/mush42/piper-rt) dataset.
   Each `<name>+RT-<quality>.tar` contains `encoder.onnx`, `decoder.onnx` and a
   `<name>.json` config with `"streaming": true`. Extract it and point the voice
   config at the two graphs (see [Loading](#loading-a-streaming-voice)).

2. **Export your own split model from a checkpoint.** If you have a trained
   Piper checkpoint you can export it as a split encoder/decoder pair. This is
   the same route the `+RT` voices are produced by. **Verify the result** — see
   [Verifying an exported split model](#verifying-an-exported-split-model), it is
   easy to produce a subtly broken split (see the `ryan+RT` case study below).

## Auto-split (no re-export needed)

A monolithic VITS graph has exactly one natural cut point: the input to the
HiFiGAN waveform decoder, which is the already-masked latent `(z * y_mask)` of
shape `[B, 192, T]`. Everything before it (text encoder, duration predictor,
flow, length regulation) is the *encoder*; the HiFiGAN generator is the
*decoder*. phoonnx finds that tensor (the data input of the decoder's 192-channel
`conv_pre`, ignoring the flow's own 192-channel convs) and extracts the two
subgraphs with `onnx.utils.extract_model`.

Because the split is **the same ops as the original graph, just cut in two**, it
is lossless by construction — verified bit-for-bit against a one-shot decode on
real voices (`maxabs ~2e-8`). Crucially, it **cannot** introduce the duration
drift that breaks some re-exported `+RT` voices (see the `ryan+RT` case study):
no weights are re-exported, so nothing can shift.

To use it, ship (or hand-write) a config that turns streaming on for an ordinary
voice — no `decoder_path` required:

```json
{ "streaming": true, "engine": "vits", "phoneme_type": "espeak" }
```

The auto-split encoder emits a single latent output (no separate `y_mask`), and
its decoder takes that one tensor (plus `sid` on multi-speaker models); the
adapter matches these by name/shape, so pre-split `+RT` voices and auto-split
voices both work through the same code path.

Splitting needs the full `onnx` package (not just `onnxruntime`), which is an
optional extra: `pip install phoonnx[streaming]`. It is only needed to *create*
a split — a pre-split `+RT` voice streams without it. If `onnx` is missing, or
the model is not a splittable VITS, phoonnx logs a warning and loads the voice
as a normal, non-streaming model.

## How it works: discard-and-stitch

The decoder is convolutional. If you naively slice the latent `z` on the time
axis and decode each slice, the **edges** of every chunk are contaminated
(~16-19 frames deep) because the convolutions near the boundary are missing
their neighbours. A crossfade between chunks does *not* fix this — the error is
in the samples, not the seam.

The fix is **discard-and-stitch**: decode each chunk with a *context margin* of
`M` extra frames on both sides, then throw the contaminated margins away and
keep only the clean middle. Because the kept region had full convolutional
context, it is **bit-identical** to a one-shot decode. Concatenating the clean
middles reproduces the exact one-shot audio, seam-free.

```
latent frames:  ....[  margin | KEEP  | margin ]....
decode this span --------^                ^------ discard both margins
```

## ONNX interface

**Encoder** (the primary session), runs once per sentence:

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `input` | int64 | `[B, T]` | Phoneme IDs |
| `input_lengths` | int64 | `[B]` | Sequence length |
| `scales` | float32 | `[3]` | `[noise_scale, length_scale, noise_w]` |
| `sid` | int64 | `[B]` | Speaker ID (multi-speaker only) |
| → `z` | float32 | `[B, 192, T]` | Latent sequence |
| → `y_mask` | float32 | `[B, 1, T]` | Frame mask |

**Decoder** (auxiliary graph), run once per chunk:

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `z` | float32 | `[B, 192, T_slice]` | Latent slice (with margins) |
| `y_mask` | float32 | `[B, 1, T_slice]` | Matching mask slice |
| → `output` | float32 | `[B, 1, N]` or `[N]` | Waveform samples |

The **hop length** (decoder samples per latent frame, almost always 256) is read
from the config if present, otherwise measured from the first decode.

## Loading a streaming voice

Point the config at the two graphs via `engine_params`. The primary model path
is the **encoder**; the decoder is an auxiliary graph:

```json
{
    "streaming": true,
    "engine": "vits_streaming",
    "engine_params": {
        "decoder_path": "decoder.onnx"
    },
    "inference": { "noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8 }
}
```

```python
from phoonnx import TTSVoice
voice = TTSVoice.load(model_path="encoder.onnx")  # config sits next to it
```

Detection is strict: the adapter claims a voice **only** when `"streaming": true`
*and* `engine_params.decoder_path` are both present, so it never hijacks a normal
single-graph voice.

## Parameters

All streaming parameters live under `engine_params.streaming` and have proven
defaults — you only set them to tune. They were tuned empirically on high-tier
voices (see [Tuning notes](#tuning-notes)).

| Param | Default | Description |
|-------|---------|-------------|
| `context_margin` | 32 | `M`: frames of context decoded and discarded each side. **Safe on every voice tested.** 24 usually works and is ~6% faster; 16 corrupts edges. |
| `first_chunk` | 8 | Frames in the first chunk. Smaller → lower TTFA. |
| `next_chunk` | 160 | Frames per subsequent chunk. Balances per-chunk overhead vs. throughput. |
| `fallback_frames` | 128 | Sentences this short or shorter are decoded one-shot (streaming can't help). |
| `intra_op_num_threads` | `cores / 2` | onnxruntime threads for the decoder. See below. |

## Tuning notes

These defaults come from measurements on `en_US-lessac+RT-high` (a heavy
high-tier decoder — the worst case for latency). Your mileage varies with model
size and CPU, but the *shape* of the results is general.

**Time-to-first-audio wins scale with sentence length.** Streaming pays off more
the longer the sentence, because a one-shot decode has to finish the whole thing
before any audio plays:

| Sentence | one-shot TTFA | streaming TTFA | speedup |
|----------|---------------|----------------|---------|
| short (~1.3 s audio) | 1399 ms | 1382 ms | 1.0x (falls back) |
| mid (~3.9 s) | 3342 ms | 1552 ms | 2.2x |
| long (~11.7 s) | 4801 ms | 1965 ms | 2.4x |

**Latency vs. throughput — an honest reading.** Streaming does *not* make total
synthesis faster; it trades a little extra compute for a much lower, and *bounded*,
time-to-first-audio. Measured on an auto-split `celtia-cotovia` (16 kHz), same
onnxruntime threading for both paths:

| Audio | monolithic TTFA | RTF (mono) | streaming TTFA | RTF (stream) | TTFA speedup |
|-------|-----------------|-----------|----------------|--------------|--------------|
| 1.4 s | 348 ms | 0.24 | 254 ms | — | 1.4x (falls back) |
| 3.9 s | 763 ms | 0.20 | 188 ms | 0.25 | 4.1x |
| 7.1 s | 1320 ms | 0.19 | 253 ms | 0.25 | 5.2x |
| 12.6 s | 2063 ms | 0.16 | 277 ms | 0.24 | 7.5x |

Two things to read off this: (1) monolithic TTFA grows linearly with sentence
length, while streaming TTFA stays flat (~200–280 ms) — that flatness is the
point. (2) Streaming's *total* RTF is ~25–30% higher (the discarded margins are
real work), so for **batch/offline file generation streaming is a net loss** and
should be left off. It is a win only for interactive, real-time speech, where
first-audio latency is what a user feels — and the win grows on slower hardware,
where monolithic RTF approaches or exceeds 1.0 and a long sentence otherwise
stalls for seconds before any sound.

**Thread count.** onnxruntime defaults to using *all* logical cores, which
measured **slower** on the heavy decoder — thread-coordination overhead and
contention. On an 8-core machine, 4 threads was optimal; 8 was worse than 4 on
both encoder and decoder. The default is therefore `cores / 2`. Override with
`engine_params.intra_op_num_threads` if profiling says otherwise.

**Context margin.** `M = 32` was bit-identical (float noise, maxabs ~3e-6) on
every voice tested. `M = 24` was also bit-identical on lessac/libritts and ~6%
faster, but on some models (e.g. the Claudio NL/EN voice) 24 left a tiny audible
residual, so **32 is the conservative default**. `M = 16` visibly corrupts the
audio (maxabs ~3e-3). If you lower `M`, verify with the maxabs check below.

**Pipelining does not help.** Running the encoder for sentence N+1 while
decoding sentence N was tested and *hurt* TTFA (the encoder thread steals cores
from the decoder that is producing the first chunk). It is deliberately not
implemented.

## The `ryan+RT` case study: verify your split models

Not every published `+RT` voice is correct. `en_US-ryan+RT-medium` produces
**visibly wrong prosody** — stress lands too early (e.g. "COOL and calm" instead
of "cool and CALM"). Investigation showed:

- The phonemization is **correct** — espeak places the stress marks right.
- The streaming split is **not** at fault — discard-and-stitch is bit-identical
  to a one-shot decode of the same model.
- The `+RT` model's **duration predictor** produces durations ~12% shorter than
  the original `ryan-high` model, *deterministically* (same difference with
  `noise_w = 0`). By contrast, `lessac+RT` and `libritts_r+RT` matched their
  originals to **0.0%**.

The root cause: `ryan` was trained with **piper 0.2.0** (PyTorch 1.11), then
converted through the newer **piper 1.0.0** (PyTorch 2.2) export used for the
`+RT` set. The duration predictor did not survive that version jump cleanly.
`lessac`/`libritts` were already 1.0.0, so their re-export was clean.

**Lessons:**

- Streaming and the encoder/decoder split are prosody-neutral. If a `+RT` voice
  sounds wrong, suspect the **export**, not the streaming.
- Do not mix piper versions: export `+RT` models from a checkpoint whose piper
  version matches the export pipeline.
- **Always verify a split model** after exporting or before trusting a
  downloaded one.

## Verifying an exported split model

Two checks catch the two failure modes.

**1. Streaming is lossless** (the split + discard-and-stitch reproduce a one-shot
decode). Encode once, then compare a full decode against the chunked decode of
the *same* `z` (the encoder samples noise internally, so you must reuse one `z`):

```python
import numpy as np, onnxruntime as ort
enc = ort.InferenceSession("encoder.onnx"); dec = ort.InferenceSession("decoder.onnx")
z, ym = enc.run(["z", "y_mask"], feed)              # ONE encode
ref = dec.run(["output"], {"z": z, "y_mask": ym})[0].squeeze()   # one-shot
# ... run the discard-and-stitch loop over the same z, M=32 ...
assert np.abs(stream[:len(ref)] - ref).max() < 1e-4   # float noise only
```

**2. The split preserves prosody** (duration predictor unchanged vs. the original
single-graph model). With `noise_w = 0` (deterministic), the frame count `T`
from the split encoder must match the original model's:

```python
# split:    z, _ = enc.run([...], scales=[0.667, 1.0, 0.0]); T_split = z.shape[2]
# original: audio = original_voice(...); T_orig = len(audio) // hop
assert T_split == T_orig    # a >1-2% difference means the export drifted (see ryan+RT)
```

## See also

- [engines.md](./engines.md) — the adapter registry and auto-detection
- [configuration.md](./configuration.md) — voice config schema
