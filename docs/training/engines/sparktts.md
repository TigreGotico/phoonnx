# Spark-TTS Engine

This page is for integrators who want Spark-TTS voices in phoonnx. After reading it you
can use a preset speaker, clone a voice from a reference clip, and understand what each
of the five ONNX graphs does.

> Related: [adapter architecture](../../engines.md) ·
> [configuration](../../configuration.md) · [voice cloning](../../cloning.md) ·
> [Chatterbox — the sibling codec-LM engine](chatterbox.md)

## What it is

**Spark-TTS** (SparkAudio) is a decoder-only language model on a Qwen2.5-0.5B backbone.
It does not predict audio. It predicts **BiCodec** tokens, and BiCodec turns those tokens
into a waveform. BiCodec splits a voice into two streams:

* **global tokens** — 32 tokens that hold the speaker: timbre, not content;
* **semantic tokens** — one stream at 50 Hz that holds what is said.

The language model reads a prompt made of control tokens, the text, and the 32 global
tokens, and then writes the semantic stream. Spark-TTS is trained on English and
Mandarin.

## When to pick it

Pick Spark-TTS when you want one small model to cover English and Mandarin with a
speaker you can either freeze or clone. It is the second autoregressive codec-LM engine
in phoonnx, after [Chatterbox](chatterbox.md). The difference that matters in practice:
Chatterbox always needs a reference clip, while a Spark-TTS voice can ship its speaker as
32 numbers and run with no reference at all.

Spark-TTS is slower than the non-autoregressive engines. One token is one pass through
the language model, and one second of speech is 50 tokens.

## Architecture

Five ONNX graphs. Only the first two are needed for a preset voice:

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

The KV-cache shape comes from the language model's own input signature, so no layer count
is written into phoonnx. The three cloning graphs open on first use, so a preset voice
never pays for the 860 MB wav2vec2 front end.

The short-time Fourier transform in front of the speaker encoder runs in NumPy, not in a
graph: ONNX has no complex dtype, so neither torch exporter can lower `torch.stft`. The
mel filterbank projection is inside `speaker_encoder_tokenizer.onnx`, and the NumPy
magnitude spectrogram matches torchaudio to 5e-7.

## Speakers

A Spark-TTS voice gets its speaker in one of two ways.

**Preset.** The voice ships a small JSON with its 32 global tokens, so the speaker is
fixed and every call sounds the same. The four bundled voices were minted with
Spark-TTS's controllable mode (gender, moderate pitch, moderate speed) and then frozen.

```python
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.merge_default_voices()
voice = manager.voices["sparktts/female/en"].load()
for chunk in voice.synthesize("Spark-TTS speaks English and Mandarin."):
    ...  # chunk.audio_float_array
```

**Zero-shot clone.** Give a reference clip and the speaker comes from the audio:

```python
from phoonnx.config import SynthesisConfig

syn = SynthesisConfig(speaker_reference="reference.wav")
for chunk in voice.synthesize("This sentence uses the cloned voice.", syn):
    ...
```

Add `speaker_reference_text` — the transcription of the clip — for the in-context
variant. The prompt then also carries the clip's own semantic tokens, so the model
continues a real utterance instead of starting cold, which follows the voice more
closely. This is the only path that needs the two extra cloning graphs.

## Decoding parameters

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `temperature` | `0.8` | sampling temperature; `0` decodes greedily |
| `top_k` | `50` | keep the 50 highest-scoring tokens |
| `top_p` | `0.95` | nucleus: keep the smallest set covering 95 % of the mass |

The warpers run in the order HuggingFace `generate` builds them — temperature, then
top-k, then top-p — because that is the stack Spark-TTS's own inference code uses.
Spark-TTS sets **no repetition penalty**: a penalty would suppress the repeated codec
tokens that normal speech contains. This differs from [Chatterbox](chatterbox.md), whose
own generation config does apply one.

Text longer than 200 characters is split on sentence boundaries and synthesized one chunk
per autoregressive pass, with a budget of 3000 tokens per pass (about 60 seconds of
audio).

## Weights and licensing

The graphs live at
[`OpenVoiceOS/phoonnx-spark-tts`](https://huggingface.co/OpenVoiceOS/phoonnx-spark-tts).
The language model is mirrored from `Fhrozen/Spark-TTS-0.5B-ONNX` after being verified
against the SparkAudio torch weights; the four BiCodec and wav2vec2 graphs were exported
from `SparkAudio/Spark-TTS-0.5B` at opset 17.

Only the float32 language model is mirrored. The quantized exports published upstream
(`q4`, `q4f16`, `int8`) disagree with the torch model by tens of logits and change greedy
decoding, so they are not safe defaults.

Spark-TTS 0.5B is released by SparkAudio under CC-BY-NC-SA-4.0; the upstream code is
Apache-2.0. Check that licence before any commercial use.
