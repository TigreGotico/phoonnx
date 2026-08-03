# OuteTTS Engine

This page is for integrators who want OuteTTS voices in phoonnx. After reading it you can
pick a language, understand what the two ONNX graphs do, and know why phoonnx ships the
0.6B model and not the 1B.

> Related: [adapter architecture](../../engines.md) ·
> [configuration](../../configuration.md) · [voice cloning](../../cloning.md) ·
> [NeuTTS — the sibling codec-LM engine](../../../phoonnx/engines/neutts.py)

## What it is

**OuteTTS 1.0** (OuteAI) is a decoder-only language model that predicts audio codec
tokens. The vocabulary of an ordinary text model was extended with the two codebooks of
**DAC.speech.v1.0**, a 24 kHz codec by IBM Research that runs at 1.5 kbps. The model
reads text and writes interleaved `<|c1_N|><|c2_N|>` pairs; the codec decoder turns that
pair stream into a waveform.

There is no phonemizer, no duration model and no vocoder. The model consumes raw text in
the language's own script, which is why one checkpoint covers 14 languages.

Two checkpoints share the interface:

| Checkpoint | Backbone | Languages | License |
| :--- | :--- | :--- | :--- |
| `OuteTTS-1.0-0.6B` | Qwen3-0.6B | 14 | Apache-2.0 |
| `Llama-OuteTTS-1.0-1B` | Llama-3.2-1B | 23 | CC-BY-NC-SA-4.0 |

phoonnx indexes the **0.6B only**. See [Weights and licensing](#weights-and-licensing).

## When to pick it

Pick OuteTTS when you want one small model to speak many languages from raw text, with no
per-language phonemizer to install. It is the fourth autoregressive codec-LM engine in
phoonnx, after Chatterbox, NeuTTS and Spark-TTS.

Do not pick it when you need speed. One codec frame is two passes through the language
model, and one second of speech is about 47 frames, so one second of audio costs about 94
model calls. On a modern desktop CPU that is a real-time factor around 20 to 30. The
non-autoregressive engines are two orders of magnitude faster.

## Languages

English, Chinese, Dutch, French, Georgian, German, Hungarian, Italian, Japanese, Korean,
Latvian, Polish, Russian, Spanish.

The model can also produce speech in languages it was not trained on, with variable
quality. The voice index lists only the trained set.

## Architecture

```
text ─► normalize ─► chunk (10-30 words)
                          │
speaker profile ──────────┤   transcript + per-word DAC codes
                          ▼
      prompt = <|im_start|> <|text_start|>...<|text_end|>
               <|audio_start|> <word blocks> <|word_start|>
                          │
                          ▼
   model.onnx (Qwen3, KV-cached) ──loop──► <|c1_N|><|c2_N|> ...
                          │
                          ▼
   decoder_model.onnx(audio_codes[1,2,T]) ─► wav @ 24 kHz
                          │
                          ▼
              fade, then loudness normalize to -18 LUFS
```

| Graph | Input | Output |
| :--- | :--- | :--- |
| `model.onnx` | `input_ids`, `attention_mask`, `position_ids`, `past_key_values.<i>.<key\|value>` | `logits` `[1,S,V]`, `present.<i>.<key\|value>` |
| `decoder_model.onnx` | `audio_codes` `[1,2,T]` int64 | `audio_values` `[1,1,T*512]` |

The KV-cache shape (layers, heads, head dim) comes from the language model's own input
signature, so no layer count is written into phoonnx.

Two details differ from the other codec-LM engines:

* `logits` covers **every** position, not only the last one, so the decode step reads
  `logits[0, -1]`.
* The waveform is **loudness normalized** to -18 LUFS with a -1 dBFS peak ceiling, and
  every decoder chunk is faded in and out over 15 ms. Upstream does this inside its codec
  wrapper, so it is part of the model's output level, not a phoonnx preference. phoonnx
  reimplements ITU-R BS.1770-4 in NumPy rather than adding `pyloudnorm` as a dependency.

## Speakers

An OuteTTS voice needs a **speaker profile** or it invents random vocal characteristics.
A profile is an in-context audio prompt: a transcript whose words each carry a duration,
three prosody buckets (energy, spectral centroid, pitch) and their DAC codes. The
profile's transcript is glued in front of your text, and its codes are the start of the
generated stream, so the model continues a real utterance.

The bundled voices all use OuteAI's `en-female-1-neutral` profile. The model carries the
speaker across languages, so this English profile leaves an English accent on the other
13 languages. For production in one language, build a profile from a clip in that
language.

Select a profile per call when a voice ships more than one:

```python
from phoonnx.config import SynthesisConfig
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.merge_default_voices()
voice = manager.voices["outetts/0.6B/de"].load()
syn = SynthesisConfig(extra_params={"voice": "en-female-1-neutral"})
for chunk in voice.synthesize("Der alte Leuchtturm steht an der Küste.", syn):
    ...  # chunk.audio_float_array, 24 kHz mono
```

phoonnx **cannot build a profile from a reference clip**. That needs Whisper word
alignment plus the DAC encoder and a prosody analysis pass; `speaker_reference` therefore
raises. Build the profile with the upstream `outetts` package and ship the JSON.

## Decoding parameters

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `temperature` | `0.4` | sampling temperature; `0` decodes greedily |
| `top_k` | `40` | keep the 40 highest-scoring tokens |
| `top_p` | `0.9` | nucleus: keep the smallest set covering 90 % of the mass |
| `min_p` | `0.05` | drop every token below 5 % of the top token's probability |
| `repetition_penalty` | `1.1` | applied over the **last 64 tokens** only |
| `max_new_tokens` | `4096` | hard stop, about 44 seconds |
| `max_chunk_words` | `30` | words per autoregressive pass |

The warpers run in the order HuggingFace `generate` builds them — temperature, then
top-k, top-p and min-p — because `outetts` drives this checkpoint through the HuggingFace
backend by default. This is **not** the order NeuTTS uses: NeuTTS runs on llama.cpp,
which truncates the raw distribution and applies temperature last.

The repetition penalty is windowed. Upstream ships a patched
`RepetitionPenaltyLogitsProcessor` that looks back only 64 tokens instead of over the
whole context, because the unwindowed penalty destroys long generations for this model.
The window spans the prompt as well as what has been generated.

Text is split into chunks of 10 to 30 words on sentence boundaries and synthesized one
chunk per pass. Upstream segments Chinese and Japanese with MeCab; phoonnx counts CJK per
character instead, which is a tighter bound and needs no extra dependency. This changes
only where a long passage is cut, never the tokens inside a chunk.

## Weights and licensing

The graphs live at
[`OpenVoiceOS/phoonnx-outetts`](https://huggingface.co/OpenVoiceOS/phoonnx-outetts),
mirrored unchanged from OuteAI and IBM Research.

**Only the float32 0.6B is indexed.** Two findings drove that:

* Every quantized 0.6B export published upstream (`fp16`, `int8`, `uint8`, `q4`, `q4f16`,
  `bnb4`, `quantized`) changes greedy decoding against the torch weights. The float32
  export matches it exactly.
* The float32 **1B** export also disagrees with its torch weights — by 12 logits at the
  end of a 1843-token prompt, with a correlation of 0.48 at that position, and greedy
  decoding diverges. The 0.6B export at the same prompt length differs by 3.8e-05. The 1B
  is mirrored for reference but is not indexed.

That costs the nine languages only the 1B covers: Arabic, Belarusian, Bengali,
Lithuanian, Persian, Portuguese, Swahili, Tamil and Ukrainian. They come back as soon as
a 1B export that verifies against its weights exists.

Licenses differ per artifact:

| Artifact | License |
| :--- | :--- |
| `OuteTTS-1.0-0.6B` | Apache-2.0 |
| `Llama-OuteTTS-1.0-1B` | CC-BY-NC-SA-4.0 — **no commercial use** |
| `DAC.speech.v1.0` (IBM Research) | CDLA-Permissive-2.0 |
| speaker profiles (from the `outetts` package) | Apache-2.0 |

Model authorship and research credit belong to OuteAI. Do not clone a voice without the
speaker's permission.
