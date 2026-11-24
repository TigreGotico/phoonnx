# Streaming VITS Engine

<!-- agent-hint
covers: how split encoder and decoder VITS voices stream, how phoonnx splits a model, and how to tune and check it
source: phoonnx/engines/vits_streaming.py:VitsStreamingAdapter, phoonnx/engines/vits_split.py:ensure_split_vits, phoonnx/voice.py:TTSVoice.load, phoonnx/voice.py:TTSVoice.stream
generated: no
verify: .venv/bin/python -m doctest -o NORMALIZE_WHITESPACE docs/streaming.md && .venv/bin/python -c "import re, sys; from phoonnx.engines.vits_streaming import _DEFAULT_STREAMING as D; rows = re.findall(r'^\| \x60(\w+)\x60 \| (\d+) \|', open('docs/streaming.md').read(), re.M); sys.exit(0 if {k: int(v) for k, v in rows} == D else 1)"
see: engines.md, benchmarking.md, alignment.md
-->

This page explains how VITS voices with a split encoder and decoder stream, for developers
who use or export such voices. It covers when streaming applies, how the split works, and
how to tune and check it.

A standard Piper or VITS voice synthesizes a whole sentence in one ONNX call, so the first
audio sample is ready only when the whole sentence is decoded. The streaming engine
decodes the sentence in chunks, with a small first chunk. The first audio of a long
sentence is therefore ready sooner.

The `VitsStreamingAdapter` does this. phoonnx selects it for a voice whose config sets
`"streaming": true` and names a separate decoder graph.

## Get a split model

Streaming needs a model in two ONNX graphs: an encoder and a decoder, with
`"streaming": true` in the voice config. A single-graph Piper or VITS model cannot stream
on its own. If a config names no decoder, the voice loads with the ordinary
[`VitsAdapter`](engines.md), which does not stream.

There are three ways to get a split model:

1. Let phoonnx split an ordinary model when it loads it. This is the recommended way. Set
   `"streaming": true` in the config of a single-graph Piper or VITS model. On the first
   load, phoonnx splits the model into an encoder and a decoder and caches the two graphs
   next to the model. See [Auto-split](#auto-split).
2. Use a pre-split "+RT" voice. The Sonata project publishes streaming variants of many
   Piper voices in the [`mush42/piper-rt`](https://huggingface.co/datasets/mush42/piper-rt)
   dataset. Each `<name>+RT-<quality>.tar` holds `encoder.onnx`, `decoder.onnx` and a
   `<name>.json` config with `"streaming": true`. Extract it and point the voice config at
   the two graphs. See [Load a streaming voice](#load-a-streaming-voice).
3. Export your own split model from a trained Piper checkpoint. Check the result with the
   steps in [Verify a split model](#verify-a-split-model).

## Auto-split

A single VITS graph has one natural cut point: the input of the HiFiGAN waveform decoder,
which is the masked latent `z * y_mask` of shape `[B, 192, T]`. The text encoder, the
duration predictor, the flow and the length regulation come before it, and they become the
encoder. The HiFiGAN generator becomes the decoder. phoonnx finds that tensor, the data
input of the 192-channel `conv_pre` of the decoder, and extracts the two subgraphs with
`onnx.utils.extract_model`.

The split holds the same operations as the original graph, cut in two. It re-exports no
weights, so it cannot change the durations. The chunked decode of a split model gives the
same audio as the original model, to float precision.

To use it, set streaming on in the config of an ordinary voice. You do not need a
`decoder_path`:

```json
{ "streaming": true, "engine": "vits", "phoneme_type": "espeak" }
```

For `model.onnx`, the split files are `model.encoder.onnx` and `model.decoder.onnx`, in the
folder of the model. A later load reuses them:

```python
>>> from phoonnx.engines.vits_split import split_paths
>>> [path.name for path in split_paths("voices/model.onnx")]
['model.encoder.onnx', 'model.decoder.onnx']

```

The encoder of an auto-split model gives one latent output, with no separate `y_mask`.
Its decoder takes that one tensor, and `sid` on a multi-speaker model. The adapter matches
the inputs and outputs by name and shape, so pre-split "+RT" voices and auto-split voices
use the same code.

The split needs the full `onnx` package, not only `onnxruntime`. Install it with
`pip install --pre "phoonnx[streaming]"`. You need it only to make a split. A pre-split
"+RT" voice streams without it. If `onnx` is missing, or the model is not a VITS model that
phoonnx can split, phoonnx logs a warning and loads the voice as a normal model that does
not stream.

## How it works: discard and stitch

The decoder is convolutional. If you cut the latent `z` on the time axis and decode each
slice alone, the edges of each chunk are wrong, because the convolutions near the edge do
not see their neighbours. A crossfade between chunks does not fix this, because the error
is in the samples, not at the seam.

The adapter therefore decodes each chunk with a context margin of `M` extra frames on both
sides. Then it discards the margins and keeps the middle. The kept region had its full
convolutional context, so it is the same as a one-shot decode. The middles joined in order
give the one-shot audio, with no seams.

```
latent frames:  ....[  margin | KEEP  | margin ]....
decode this span --------^                ^------ discard both margins
```

## ONNX interface

The encoder is the primary session. It runs once per sentence:

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `input` | int64 | `[B, T]` | Phoneme IDs |
| `input_lengths` | int64 | `[B]` | Sequence length |
| `scales` | float32 | `[3]` | `[noise_scale, length_scale, noise_w]` |
| `sid` | int64 | `[B]` | Speaker ID (multi-speaker only) |
| → `z` | float32 | `[B, 192, T]` | Latent sequence |
| → `y_mask` | float32 | `[B, 1, T]` | Frame mask |

The decoder is an auxiliary graph. It runs once per chunk:

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `z` | float32 | `[B, 192, T_slice]` | Latent slice (with margins) |
| `y_mask` | float32 | `[B, 1, T_slice]` | Matching mask slice |
| → `output` | float32 | `[B, 1, N]` or `[N]` | Waveform samples |

The hop length is the number of decoder samples per latent frame. It comes from the
`hop_length` of the voice config, which is 256 when the config does not set it. The adapter
measures it from the first decode only when it runs without a voice config.

## Load a streaming voice

Point the config at the two graphs through `engine_params`. The primary model path is the
encoder, and the decoder is an auxiliary graph. A relative `decoder_path` in the config is
relative to the folder of the config, so the voice loads from any working directory. A
`decoder_path` that the caller passes in `engine_params` is used as written.

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
from phoonnx.voice import TTSVoice

voice = TTSVoice.load(model_path="encoder.onnx", config_path="voice.json")
print(type(voice.adapter).__name__)  # VitsStreamingAdapter
```

Without `config_path`, `TTSVoice.load` reads `encoder.onnx.json`. A "+RT" archive names its
config `<name>.json`, so pass that file as `config_path`.

The adapter claims a voice only when `"streaming": true` and `engine_params.decoder_path`
are both in the config, so it never takes a normal single-graph voice:

```python
>>> from phoonnx.engines.vits_streaming import VitsStreamingAdapter
>>> VitsStreamingAdapter.detect({"streaming": True, "engine_params": {"decoder_path": "decoder.onnx"}})
True
>>> VitsStreamingAdapter.detect({"streaming": True})
False

```

The decoder runs on the `providers` of the voice, the same list as the encoder. See
[Execution Providers](configuration.md#execution-providers).

## Stream the audio

`TTSVoice.stream()` yields each decoded chunk as soon as the decoder makes it. Peak
normalization needs the whole sentence, so turn it off. With `audio.normalize=True` (the
default), a target loudness, voice conversion or super-resolution, `stream()` yields one
chunk per sentence.

```python
from phoonnx.config import AudioPost, SynthesisConfig

config = SynthesisConfig(audio=AudioPost(normalize=False))
for chunk in voice.stream("A long sentence that the decoder reads in chunks.", config):
    play(chunk.audio_int16_bytes)  # your audio sink
```

The first chunk holds `first_chunk` frames, 2048 samples at a hop length of 256. The chunks
join to the audio that `synthesize()` gives with the same settings. `synthesize()` waits for
the last chunk of a sentence and yields the sentence whole.

## Parameters

The chunk parameters are under `engine_params.streaming`. The decoder thread count is a key
of `engine_params` itself. Each one has a default, so set them only to tune:

```json
{
    "engine_params": {
        "decoder_path": "decoder.onnx",
        "intra_op_num_threads": 4,
        "streaming": {"context_margin": 32, "first_chunk": 8}
    }
}
```

| Param | Default | Description |
|-------|---------|-------------|
| `context_margin` | 32 | `M`: the frames of context decoded and discarded on each side of a chunk |
| `first_chunk` | 8 | The frames in the first chunk. A smaller value gives the first audio sooner |
| `next_chunk` | 160 | The frames in each later chunk. A larger value costs less per chunk |
| `fallback_frames` | 128 | A sentence of this many frames or fewer is decoded in one call |
| `intra_op_num_threads` | the thread count of the voice, else half the CPU cores | The onnxruntime threads of the decoder. `0` gives the ONNX Runtime default. Set it in `engine_params`, not in `engine_params.streaming` |

## Tuning notes

Streaming does not make the total synthesis faster. The discarded margins are extra
decoder work, so the total real-time factor is a little higher than a one-shot decode. In
exchange, the time to the first audio stays about the same as a sentence gets longer,
while a one-shot decode makes the listener wait for the whole sentence. Use streaming for
interactive speech, where a person waits for the first sound. Leave it off for batch file
generation. The gain is larger on slow hardware, where a one-shot decode of a long sentence
takes seconds.

A short sentence of `fallback_frames` frames or fewer is decoded in one call, so streaming
gives it no gain.

onnxruntime uses all logical cores by default. On the heavy decoder, that can be slower,
because the threads compete. The decoder therefore uses half the cores, unless the voice
has a [thread count](configuration.md#thread-count) of its own. Set
`engine_params.intra_op_num_threads` to override it.

A `context_margin` of 32 is the default. A smaller margin can leave errors at the chunk
edges on some voices. If you lower it, check the result with the first test in
[Verify a split model](#verify-a-split-model).

## Verify a split model

An export from a checkpoint can give a duration predictor that differs from the original
model, for example when the checkpoint and the export pipeline come from different Piper
versions. The auto-split cannot cause this, and the chunked decode cannot cause it either.
If a "+RT" voice sounds wrong, suspect its export. Two checks catch the two failure modes.

1. Check that streaming is lossless. Encode once, then compare a full decode with the
   chunked decode of the same `z`. The encoder samples noise inside, so reuse one `z`:

   ```python
   import numpy as np, onnxruntime as ort
   enc = ort.InferenceSession("encoder.onnx"); dec = ort.InferenceSession("decoder.onnx")
   z, ym = enc.run(["z", "y_mask"], feed)              # ONE encode
   ref = dec.run(["output"], {"z": z, "y_mask": ym})[0].squeeze()   # one-shot
   # ... run the discard-and-stitch loop over the same z, M=32 ...
   assert np.abs(stream[:len(ref)] - ref).max() < 1e-4   # float noise only
   ```

2. Check that the split keeps the durations of the original single-graph model. With
   `noise_w = 0`, the encoder is deterministic, and the frame count `T` from the split
   encoder must match the frame count of the original model:

   ```python
   # split:    z, _ = enc.run([...], scales=[0.667, 1.0, 0.0]); T_split = z.shape[2]
   # original: audio = original_voice(...); T_orig = len(audio) // hop
   assert T_split == T_orig    # a difference means the export changed the durations
   ```

---

[Docs index](index.md) · Previous: [Voice conversion](voice_conversion.md) · Next: [Phoneme alignment](alignment.md)
