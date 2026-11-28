# Spark-TTS Engine

<!-- agent-hint
covers: how to use a Spark-TTS preset speaker or clone a voice, and what the five graphs do
source: phoonnx/engines/sparktts.py, phoonnx/voice_index/sparktts.json
generated: no
verify: .venv/bin/python -c "import re,sys,json; import phoonnx.engines.sparktts as S; p=open('docs/training/engines/sparktts.md').read().replace(chr(96),''); d={k:float(v) for k,v in re.findall(r'^\| ([a-z_]+) \| ([\d.]+) \| [A-Z]',p,re.M)}; ids=set(re.findall(r'(sparktts/(?:fe)?male/\w\w)',p)); m=re.search(r'longer than (\d+) characters.*?budget of (\d+) tokens',p,re.S).groups(); sys.exit(0 if d==S.SparkTTSAdapter().default_params() and ids==set(json.load(open('phoonnx/voice_index/sparktts.json'))) and tuple(map(int,m))==(S.MAX_CHUNK_CHARS,S.MAX_NEW_TOKENS) else 1)"
see: chatterbox.md, ../../cloning.md, ../../engines.md
-->

This page shows how to use the Spark-TTS preset speakers and how to clone a voice with Spark-TTS.

## What it is

Spark-TTS (SparkAudio) is a decoder-only language model on a Qwen2.5-0.5B backbone.
It does not predict audio. It predicts BiCodec tokens, and BiCodec turns those tokens
into a waveform. BiCodec splits a voice into two streams:

* Global tokens are 32 tokens that hold the speaker: the timbre, not the content.
* Semantic tokens are one stream at 50 Hz that holds what is said.

The language model reads a prompt made of control tokens, the text, and the 32 global
tokens, and then writes the semantic stream. Spark-TTS is trained on English and
Mandarin.

## When to pick it

Pick Spark-TTS when you want one small model for English and Mandarin, with a speaker that you fix or clone. Like [Chatterbox](chatterbox.md), it is an autoregressive codec language model. Chatterbox always needs a reference clip, but a Spark-TTS voice can ship its speaker as 32 numbers and run with no reference.

Spark-TTS is slower than the non-autoregressive engines. One token is one pass through
the language model, and one second of speech is 50 tokens.

## Architecture

A voice has five ONNX graphs. A preset voice needs only the first two:

```
text ─► BPE ─► content ids ─┐
speaker: 32 global tokens ──┤
                            ▼
              prompt = task + content + speaker
                            │
                            ▼
     model.onnx (Qwen2, KV-cached) ──loop──► semantic tokens
                            │
                            ▼
     bicodec_vocoder.onnx(semantic, global) ─► wav @ 16 kHz

cloning only:
  ref.wav@16k ─► |STFT| ─► speaker_encoder_tokenizer.onnx ─► 32 global tokens
  ref.wav@16k ─► wav2vec2_model.onnx ─► bicodec_encoder_quantizer.onnx ─► semantic tokens
```

| Graph | Input | Output |
| :--- | :--- | :--- |
| `model.onnx` | `input_ids`, `attention_mask`, `position_ids`, `past_key_values.*` | `logits`, `present.*` |
| `bicodec_vocoder.onnx` | `semantic_tokens` `[1,T]`, `global_tokens` `[1,1,32]` | waveform `[1,1,N]` |
| `speaker_encoder_tokenizer.onnx` | `spec` `[1,513,T]` | `global_tokens` `[1,1,32]` |
| `wav2vec2_model.onnx` | `wav` `[1,N]` | `feat` `[1,T,1024]` |
| `bicodec_encoder_quantizer.onnx` | `feat` `[1,T,1024]` | `semantic_tokens` `[1,T]` |

The adapter reads the KV-cache shape from the input signature of the language model. The three cloning graphs open on first use, so a preset voice never loads the 860 MB wav2vec2 graph.

The short-time Fourier transform in front of the speaker encoder runs in NumPy, because ONNX has no complex dtype and the torch exporters cannot lower `torch.stft`. The mel filterbank projection is inside `speaker_encoder_tokenizer.onnx`.

## Speakers

A Spark-TTS voice gets its speaker in one of two ways.

### Preset speaker

The voice ships a small JSON file with its 32 global tokens, so the speaker is fixed. The bundled index holds four preset voices: `sparktts/female/en`, `sparktts/male/en`, `sparktts/female/zh` and `sparktts/male/zh`.

```python
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.merge_default_voices()
voice = manager.voices["sparktts/female/en"].load()
for chunk in voice.synthesize("Spark-TTS speaks English and Mandarin."):
    audio = chunk.audio_float_array
```

### Zero-shot clone

Give a reference clip, and the speaker comes from the audio:

```python
from phoonnx.config import SpeakerReference, SynthesisConfig

syn = SynthesisConfig(reference=SpeakerReference(audio="reference.wav"))
for chunk in voice.synthesize("This sentence uses the cloned voice.", syn):
    audio = chunk.audio_float_array
```

For the in-context variant, also set `reference.text` to the transcription of the clip. The prompt then also holds the semantic tokens of the clip, and the model continues a real utterance. Only this variant needs the wav2vec2 and semantic encoder graphs.

## Decoding parameters

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `temperature` | 0.8 | Sampling temperature. 0 decodes greedily. |
| `top_k` | 50 | Keep the 50 highest-scoring tokens |
| `top_p` | 0.95 | Keep the smallest set of tokens that covers 95 % of the probability |

The adapter applies temperature, then top-k, then top-p, in the order that HuggingFace `generate` uses and that the Spark-TTS inference code uses. Spark-TTS applies no repetition penalty, because normal speech contains repeated codec tokens. [Chatterbox](chatterbox.md) applies one.

The adapter splits text longer than 200 characters at sentence boundaries. It synthesizes each chunk in one autoregressive pass, with a budget of 3000 tokens per pass (about 60 seconds of audio).

## Weights and licensing

The graphs live at
[`Phoonnx/sparktts-multi-0.5b-unicode`](https://huggingface.co/Phoonnx/sparktts-multi-0.5b-unicode).
The language model comes from `Fhrozen/Spark-TTS-0.5B-ONNX`, in float32. The BiCodec and wav2vec2 graphs come from `SparkAudio/Spark-TTS-0.5B`.

SparkAudio releases the Spark-TTS 0.5B weights under CC-BY-NC-SA-4.0, so you may not use them commercially. The upstream code is Apache-2.0.

---

[← Chatterbox](chatterbox.md) · [Home](../../index.md) · [Qwen3-TTS →](qwen3tts.md)
