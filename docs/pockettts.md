# Pocket TTS

<!-- agent-hint
covers: how to use the Pocket TTS voices, how the engine runs, and its controls and precision
source: phoonnx/engines/pockettts.py:PocketTTSAdapter, phoonnx/voice_index/pockettts.json
generated: no
verify: pytest tests/test_pockettts.py, plus the pockettts/en/alba example on the page
see: precision.md, cloning.md
-->

This page shows how to use the Pocket TTS voices and how the engine works, for developers who want a small multilingual model on CPU.

Pocket TTS is a small text-to-speech model from [Kyutai](https://kyutai.org/tts/). It has
100 million parameters and speaks faster than real time on two CPU cores. It covers six
languages: English, French, German, Italian, Portuguese and Spanish.

Each language is a separate weight bundle. phoonnx serves them from the
[`Phoonnx/phoonnx-pocket-tts`](https://huggingface.co/Phoonnx/phoonnx-pocket-tts)
mirror, with 26 speakers per language.

## Use a voice

```python
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.merge_default_voices()
manager.download_voice_by_id("pockettts/en/alba")

voice = manager.voices["pockettts/en/alba"].load()
for chunk in voice.synthesize("Hello world."):
    print(chunk.sample_rate, chunk.audio_float_array.dtype, chunk.audio_float_array.ndim)
```

```text
24000 float32 1
```

The English bundle is about 160 MB. Install the `pockettts` extra first:
`pip install --pre "phoonnx[pockettts]"`.

Voice ids follow `pockettts/<language>/<speaker>`. The languages are `en`, `fr`, `de`,
`it`, `pt` and `es`.

## How it works

Pocket TTS is a flow-matching latent language model on top of the Mimi neural codec. Five
ONNX graphs make one voice:

| Graph | Inputs | Outputs |
|---|---|---|
| `text_conditioner` | `token_ids` | `embeddings` |
| `flow_lm_main` | `sequence`, `text_embeddings`, `state_*` | `conditioning`, `eos_logit`, `out_state_*` |
| `flow_lm_flow` | `c`, `s`, `t`, `x` | `flow_dir` |
| `mimi_decoder` | `latent`, `state_*` | `audio_frame`, `out_state_*` |
| `mimi_encoder` | `audio` | `latents` |

Synthesis of one chunk runs like this:

1. The bundle's SentencePiece tokenizer turns the text into token ids, and
   `text_conditioner` embeds them.
2. `flow_lm_main` consumes the embeddings once. This primes its state.
3. Each following step feeds the previous latent frame back in. The graph returns a
   conditioning vector and an end-of-speech logit.
4. A short Euler loop over `flow_lm_flow` turns a Gaussian sample into the next latent
   frame. One step is enough; more steps trade speed for smoothness.
5. The loop stops a few frames after the end-of-speech logit crosses its threshold.
6. `mimi_decoder` turns the latent frames into 24 kHz audio. It runs on small, fixed-size
   sub-chunks of 15 frames and carries its state forward between them, so the join inside
   one text chunk's audio has no audible seam.

The model produces 12.5 latent frames per second. Each frame decodes to 1920 samples.

### Stream the audio

`TTSVoice.stream` yields the audio of each 15-frame sub-chunk when `mimi_decoder` has
made it, so playback can start before the model finishes the sentence. Peak
normalisation needs the whole sentence, so turn it off to get the sub-chunks:

```python
from phoonnx.config import AudioPost, SynthesisConfig

config = SynthesisConfig(audio=AudioPost(normalize=False), extra={"seed": 7})
text = "Hello world. This is a longer second sentence that takes a while to say."
for chunk in voice.stream(text, config):
    print(len(chunk.audio_float_array))
```

```text
28800
28800
28800
5760
```

Each full part is 15 frames of 1920 samples. Joined, the parts give the same audio as `voice.synthesize(text, config)`. With the default config, `stream` yields one chunk for each text chunk, as `synthesize` does. See [Streaming the audio of a sentence](api.md#streaming-the-audio-of-a-sentence).

Each synthesis call starts `flow_lm_main` from the voice's saved state, not from wherever
the previous text chunk left off. So state only carries forward within a chunk's own
15-frame sub-chunks, not across the sentence-level chunks described below. This is
deliberate: the model was trained on single sentences and drifts on longer input.

### Stream state

The `state_*` tensors are the contract between the graphs and phoonnx. `bundle.json`
holds a manifest for `flow_lm_main` and one for `mimi_decoder`. Each entry gives the input
name, the matching output name, the shape, the dtype and how to fill the tensor at the
start of a stream. phoonnx creates the tensors from the manifest, feeds them in, and
copies the returned tensors back before the next step.

### Voices are states

A Pocket TTS voice is not an embedding vector. It is the transformer state after the model
has consumed a speaker's audio. Kyutai publishes one such state per speaker per language,
as a safetensors file of module states. The manifest maps those module states onto the
`flow_lm_main` state tensors.

A saved state can be shorter than the cache the exported graph declares. phoonnx copies it
into the leading part of a freshly filled tensor, so a state saved from a short recording
still loads.

## Text handling

The adapter owns text to token conversion, so no phonemizer runs. It applies Pocket TTS's
own frontend: capitalize the first letter, add a final period, then split on sentence-final
punctuation. A sentence over the bundle's token limit is split again on commas, semicolons
and colons. Short pieces are packed back together up to the limit.

The model was trained on single sentences and drifts on longer input, which is why the
chunking is part of the engine rather than a general text utility.

## Voice cloning

Pass a reference recording to clone a voice at run time:

```python
from phoonnx.config import SpeakerReference, SynthesisConfig

config = SynthesisConfig(reference=SpeakerReference(audio="reference.wav"))
for chunk in voice.synthesize("Hello world.", config):
    ...
```

`mimi_encoder` turns the clip into latent frames, phoonnx prepends the bundle's
beginning-of-sequence embedding, and `flow_lm_main` consumes the result to build the state.
Give the model a few seconds of clean mono speech.

Clone voices only with the speaker's consent.

## Controls

| Parameter | Default | Effect |
|---|---|---|
| `temperature` | 0.7 | Sampling spread. 0 makes generation deterministic. |
| `lsd_steps` | 1 | Euler steps per latent frame. More steps are slower. |
| `eos_threshold` | -4.0 | When the model is judged to have finished speaking. |
| `seed` | none | Fixes the random draw, so a run repeats exactly. |

Set them per call through `SynthesisConfig.extra`, or per voice through
`engine_params`.

## Precision

The voice index points at the 8-bit graphs, which make the model real time on a small CPU. These entries carry `"precision": "int8"`, so a load with `precision="full"` raises `PrecisionUnavailable`:

```text
PrecisionUnavailable: voice 'pockettts/en/alba' has no full files; precision 'auto' or 'int8' loads it
```
 The mirror also holds full-precision graphs under the same names without the `_int8` suffix. To trade speed for the reference numerics, write your own index entry whose `model_url` and auxiliary URLs point at those graphs.

The Italian voices are different. Their 8-bit graphs repeat words on some sentences, so
the Italian entries load the full-precision graphs. Load an Italian voice with
`precision="int8"` to get the 8-bit graphs.

## Licensing

The weights are Kyutai's, under Creative Commons Attribution 4.0. Credit Kyutai when you
use them. Kyutai asks that the model is not used for impersonation without consent, for
deception, or for unlawful content.

---

[Docs index](index.md) · Previous: [Use phoonnx offline](offline.md) · Next: [Orpheus](orpheus.md)
