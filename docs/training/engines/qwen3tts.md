# Qwen3-TTS Engine

This page is for integrators who want Qwen3-TTS voices in phoonnx. After reading it you
can pick a timbre and a language, and you know what each of the seven ONNX graphs does
and what one second of audio costs.

> Related: [adapter architecture](../../engines.md) ·
> [configuration](../../configuration.md) ·
> [Spark-TTS — the sibling codec-LM engine](sparktts.md)

## What it is

**Qwen3-TTS** (Alibaba Qwen) predicts the tokens of a 12.5 Hz neural codec and turns them
into a waveform at 24 kHz. It has two stacked autoregressive models, not one:

* the **talker** — 28 decoder layers that emit one token per audio frame. That token is
  code group 0, and the talker's hidden state conditions the second model;
* the **code predictor** — 5 decoder layers that read the talker's hidden state and write
  the other 15 code groups of the *same* frame, one group per step.

One 80 ms frame therefore costs one talker step and 15 code-predictor steps. The 16
groups of a frame are summed into a single embedding and fed back to the talker together
with the next slice of the text.

The talker reads embeddings, never token ids: at every prompt position a text hidden
state and a codec hidden state are added together. The prompt holds the assistant role
tokens, a language block, the speaker, the whole text against codec padding, and finally
the codec BOS that starts the audio.

phoonnx ships the **0.6B CustomVoice** checkpoint. It has nine trained timbres and no
speaker encoder, so a voice is a name, not a reference clip, and this engine does not do
[cloning](../../cloning.md).

## When to pick it

Pick Qwen3-TTS when one model has to cover many languages with a fixed, high-quality
voice. It speaks Chinese, English, Japanese, Korean, German, French, Russian, Portuguese,
Spanish and Italian, which is the widest language set of any codec-LM engine in phoonnx.

Do not pick it when you need to clone a speaker — use [Spark-TTS](sparktts.md) or
[Chatterbox](chatterbox.md) — or when you need real-time synthesis on a small CPU. Two
autoregressive stages cost far more than a single-pass engine: measured on six cores,
one second of audio takes about five seconds of CPU time.

## Voices

| Voice | Timbre | Native language |
| --- | --- | --- |
| `vivian` | bright young female | Chinese |
| `serena` | warm gentle young female | Chinese |
| `uncle_fu` | seasoned male, mellow | Chinese |
| `dylan` | youthful male | Chinese (Beijing) |
| `eric` | lively male | Chinese (Sichuan) |
| `ryan` | dynamic male | English |
| `aiden` | sunny American male | English |
| `ono_anna` | playful female | Japanese |
| `sohee` | warm female | Korean |

Any timbre speaks any of the ten languages; the table gives the language each timbre was
recorded in, which is where it sounds best. `dylan` and `eric` are dialect voices: they
always speak their dialect when the language is Chinese or unset, whatever the caller
asks for.

Voice ids follow `qwen3tts/<timbre>/<language>`:

```python
from phoonnx.voice import TTSVoice

voice = TTSVoice.load("qwen3tts/ryan/en")
audio = voice.synthesize("The quick brown fox jumps over the lazy dog.")
```

## Graphs

| File | What it does |
| --- | --- |
| `talker.onnx` | the 28-layer talker, one KV-cached step per frame (the voice's `model_url`) |
| `text_embed.onnx` | text ids to projected text hidden states |
| `codec_embed.onnx` | talker codec ids to hidden states |
| `code_predictor_prefill.onnx` | talker hidden plus code group 0, gives code group 1 |
| `code_predictor_step.onnx` | code group n gives code group n+1 |
| `sub_codec_embed.onnx` | code-group token to its group's embedding |
| `codec_decoder.onnx` | 16 code groups per frame to 24 kHz audio |

Every group has its own embedding table and its own output head, so the step graph takes
the group index as an input and gathers both from a stacked weight. The graphs are
float32 and total about 4.4 GB, which is what a voice downloads once.

The mirror is [`OpenVoiceOS/phoonnx-qwen3-tts`](https://huggingface.co/OpenVoiceOS/phoonnx-qwen3-tts).
The export script is `scripts/conversion/qwen3tts/export_qwen3tts_onnx.py`.

## Control parameters

Defaults come from the checkpoint's own `generation_config.json`:

| Parameter | Default | What it does |
| --- | --- | --- |
| `temperature` | 0.9 | talker sampling temperature |
| `top_k` | 50 | talker top-k |
| `top_p` | 1.0 | talker nucleus |
| `repetition_penalty` | 1.05 | penalty over the code-group-0 tokens already emitted |
| `subtalker_temperature` | 0.9 | code-predictor temperature |
| `subtalker_top_k` | 50 | code-predictor top-k |
| `subtalker_top_p` | 1.0 | code-predictor nucleus |

Two more parameters are accepted per call: `do_sample` and `seed`. `do_sample=False`
gives a deterministic result that matches the upstream PyTorch model token for token,
which is what the parity checker uses. Do not use it for synthesis: greedy decoding makes
this model loop and talk forever on some inputs, which is why upstream samples by
default.

Sampling is stochastic, and a rare draw also makes the model keep talking past the text.
Cap `max_new_tokens` when you need a bounded result: at 12.5 frames per second, 250
frames is 20 seconds of audio.

## Licence

Apache-2.0, the licence of the upstream model. The weights, the architecture and the nine
timbres are the work of the Alibaba Qwen team; cite the Qwen3-TTS technical report
(arXiv 2601.15621) when you use them.
