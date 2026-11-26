# Pocket TTS

<!-- agent-hint
covers: how to use the Pocket TTS voices, how the engine runs, and its controls and precision
source: phoonnx/engines/pockettts.py:PocketTTSAdapter, phoonnx/engines/pockettts.py:AVAILABLE_LANGS, phoonnx/voice.py:TTSVoice.stream, phoonnx/voice_index/pockettts.json
generated: no
verify: .venv/bin/python -c "import re, sys, json; from collections import Counter; from phoonnx.engines.pockettts import PocketTTSAdapter, AVAILABLE_LANGS, DECODE_CHUNK_FRAMES; q = chr(96); page = re.sub('<!-- agent-hint.*?\n-->', '', open('docs/pockettts.md').read(), flags=re.S); a = PocketTTSAdapter(); code = {'temperature': str(a.temperature), 'lsd_steps': str(a.lsd_steps), 'eos_threshold': str(a.eos_threshold), 'seed': 'none'}; table = {k: v for k, v in re.findall('^[|] ' + q + '([a-z_]+)' + q + ' [|] ([^ |]+) [|]', page, re.M) if k in code}; idx = json.load(open('phoonnx/voice_index/pockettts.json')); per = Counter(k.split('/')[1] for k in idx); langs = re.findall(q + '([a-z]{2})' + q, re.search('The languages are (.*)', page).group(1)); speakers = int(re.search('with ([0-9]+) speakers per language', page).group(1)); int8 = all((v.get('precision') == 'int8') != k.startswith('pockettts/it/') for k, v in idx.items()); frames = str(DECODE_CHUNK_FRAMES) + ' frames of 1920 samples' in page; bad = [table != code, langs != AVAILABLE_LANGS, set(per.values()) != {speakers}, not int8, not frames]; print(bad, table); sys.exit(1 if any(bad) else 0)"
see: precision.md, cloning.md
-->

This page shows how to use the Pocket TTS voices and how the engine works, for developers who want a small multilingual model on CPU.

Pocket TTS is a small text-to-speech model from [Kyutai](https://kyutai.org/tts/). It has
100 million parameters and speaks faster than real time on two CPU cores. It covers six
languages: English, French, German, Italian, Portuguese and Spanish.

Each language is a separate weight bundle. phoonnx loads each bundle from its own
repository in the [Phoonnx](https://huggingface.co/Phoonnx) organization on the Hugging Face
Hub, with 26 speakers per language.

## Use a voice

Install the `pockettts` extra first:

```bash
pip install --pre "phoonnx[pockettts]"
```

Then download and load a voice:

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

The download of an English voice is 171 MB.

Voice ids follow `pockettts/<language>/<speaker>`. The languages are `en`, `fr`, `de`, `it`, `pt` and `es`.

## How it works

Pocket TTS is a flow-matching latent language model on top of the Mimi neural codec. Five
ONNX graphs make one voice:

| Graph | Inputs | Role |
|---|---|---|
| `text_conditioner` | `token_ids` | Embeds the text tokens. |
| `flow_lm_main` | `sequence`, `text_embeddings`, `state_*` | Gives a conditioning vector and an end-of-speech logit for each step. |
| `flow_lm_flow` | `c`, `s`, `t`, `x` | Gives the flow direction of the next latent frame. |
| `mimi_decoder` | `latent`, `state_*` | Turns latent frames into audio. |
| `mimi_encoder` | `audio` | Turns a reference clip into latent frames, for cloning. |

Synthesis of one chunk runs like this:

1. The SentencePiece tokenizer of the bundle turns the text into token ids, and
   `text_conditioner` embeds them.
2. `flow_lm_main` reads the embeddings once. This primes its state.
3. Each following step feeds the previous latent frame back in. The graph returns a
   conditioning vector and an end-of-speech logit.
4. A short Euler loop over `flow_lm_flow` turns a Gaussian sample into the next latent
   frame. One step is enough. More steps are slower and give smoother output.
5. The loop stops a few frames after the end-of-speech logit crosses its threshold.
6. `mimi_decoder` turns the latent frames into 24 kHz audio. It decodes sub-chunks of 15
   frames and carries its state from one sub-chunk to the next, so the joins inside the
   audio of one text chunk have no audible seam.

The model makes 12.5 latent frames per second. Each frame decodes to 1920 samples.

### Stream the audio

`TTSVoice.stream` yields the audio of each 15-frame sub-chunk when `mimi_decoder` has
made it, so playback can start before the model finishes the sentence. Peak
normalization needs the whole sentence, so turn it off to get the sub-chunks:

```python
from phoonnx.config import AudioPost, SynthesisConfig

config = SynthesisConfig(audio=AudioPost(normalize=False), extra={"seed": 7})
text = "Hello world. This is a longer second sentence that takes a while to say."
for chunk in voice.stream(text, config):
    print(len(chunk.audio_float_array))
```

Each full part is 15 frames of 1920 samples, which is 28800 samples. The last part of a text
chunk can be shorter. Joined, the parts give the same audio as `voice.synthesize(text, config)`.
With the default config, `stream` yields one chunk for each text chunk, as `synthesize`
does. See [Streaming the audio of a sentence](api.md#streaming-the-audio-of-a-sentence).

Each text chunk starts `flow_lm_main` from the saved state of the voice, not from the state
that the previous text chunk left. So the state carries forward only within the 15-frame
sub-chunks of one text chunk. The model was trained on single sentences and drifts on
longer input.

### Stream state

The `state_*` tensors are the contract between the graphs and phoonnx. `bundle.json`
holds a manifest for `flow_lm_main` and one for `mimi_decoder`. Each entry gives the input
name, the matching output name, the shape, the dtype and how to fill the tensor at the
start of a stream. phoonnx creates the tensors from the manifest, feeds them in, and
copies the returned tensors back before the next step.

### Voices are states

A Pocket TTS voice is not an embedding vector. It is the transformer state after the model
has read the audio of a speaker. Kyutai publishes one such state per speaker per language,
as a safetensors file of module states. The manifest maps those module states onto the
`flow_lm_main` state tensors.

A saved state can be shorter than the cache that the exported graph declares. phoonnx
copies it into the start of a tensor that holds the start-of-stream values, so a state saved from a short recording still
loads.

## Text handling

The adapter turns the text into tokens itself, so no phonemizer runs. It applies the text
frontend of Pocket TTS: capitalize the first letter, add a final period, then split on
sentence-final punctuation. A sentence over the token limit of the bundle is split again
on commas, semicolons and colons. Short pieces are packed back together up to the limit.

The model was trained on single sentences and drifts on longer input. This is why the
chunking is part of the engine and not a general text utility.

## Voice cloning

Pass a reference recording to clone a voice at run time:

```python
from phoonnx.config import SpeakerReference, SynthesisConfig

config = SynthesisConfig(reference=SpeakerReference(audio="reference.wav"))
for chunk in voice.synthesize("Hello world.", config):
    ...
```

`mimi_encoder` turns the clip into latent frames. phoonnx puts the beginning-of-sequence
embedding of the bundle before them, and `flow_lm_main` reads the result to build the state.
Give the model a few seconds of clean mono speech.

Clone voices only with the speaker's consent.

## Controls

| Parameter | Default | Effect |
|---|---|---|
| `temperature` | 0.7 | Sampling spread. 0 makes generation deterministic. |
| `lsd_steps` | 1 | Euler steps per latent frame. More steps are slower. |
| `eos_threshold` | -4.0 | The logit above which the model has finished speaking. |
| `seed` | none | Fixes the random draw, so a run repeats exactly. |

Set them per call through `SynthesisConfig.extra`, or per voice through
`engine_params`.

## Precision

The index entries of English, French, German, Portuguese and Spanish point at the 8-bit
graphs, which make the model real time on a small CPU. These entries carry
`"precision": "int8"`, so a load with `precision="full"` raises `PrecisionUnavailable`:

```text
PrecisionUnavailable: voice 'pockettts/en/alba' has no full files; precision 'auto' or 'int8' loads it
```

Each language repository also holds the full-precision graphs, under the same names
without the `_int8` suffix. To load them, write your own index entry whose `model_url` and
auxiliary URLs point at those graphs.

The Italian voices are different. Their 8-bit graphs repeat words on some sentences, so
the Italian entries load the full-precision graphs. Load an Italian voice with
`precision="int8"` to get the 8-bit graphs. See [Precision variants](precision.md).

## Licensing

The weights are Kyutai's, under Creative Commons Attribution 4.0. Credit Kyutai when you
use them. Kyutai asks that the model is not used for impersonation without consent, for
deception, or for unlawful content.

---

[← Use phoonnx offline](offline.md) · [Home](index.md) · [Orpheus →](orpheus.md)
