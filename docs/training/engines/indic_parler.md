# Indic Parler-TTS

[Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts) (AI4Bharat) speaks
20 Indic languages and English. Set `engine: indic_parler`.

It is the Indic fine-tune of [Parler-TTS](https://github.com/huggingface/parler-tts)
(Yoach Lacombe, Vaibhav Srivastav, Sanchit Gandhi). You do not pick a voice with a
speaker id or a reference clip. You **describe** it:

> *"Rohit's voice is clear and expressive, with a moderate speed and pitch, recorded in a
> very close-sounding environment with no background noise."*

## Why this engine is different

Every other autoregressive engine in phoonnx — Chatterbox, NeuTTS, Spark-TTS, OuteTTS,
ArkTTS, Qwen3-TTS — is **decoder-only**: one token stream, one KV cache. Indic Parler is
phoonnx's first **encoder-decoder** engine, and the conditioning arrives on two different
paths:

1. The **description** goes through a frozen Flan-T5 encoder. All 24 decoder layers
   **cross-attend** to the result. Those cross-attention keys and values never change
   while decoding, so the prefill graph computes them once and the decode graph reads
   them back unchanged, step after step.
2. The **text to speak** is embedded by `embed_prompts` and **prepended to the decoder's
   input embeddings**. It is part of the self-attention stream, not the cross-attention
   stream. Upstream calls this `prompt_cross_attention: false`.

Two tokenizers follow from that split, and they are **different vocabularies**: the
checkpoint's own tokenizer for the prompt, the Flan-T5 tokenizer for the description.
Mixing them up produces speech, just not the speech you asked for.

## Graphs

Published at
[`OpenVoiceOS/phoonnx-indic-parler`](https://huggingface.co/OpenVoiceOS/phoonnx-indic-parler).

| `engine_params` key | Graph | Contract |
|---|---|---|
| *(the voice's own model)* | `decoder_prefill.onnx` | codec ids + prompt ids + encoder states → `logits[9,1088]`, self-KV (24×2), cross-KV (24×2) |
| `decoder_decode_path` | `decoder_decode.onnx` | one codec frame + both caches → `logits[9,1088]`, self-KV (24×2) |
| `text_encoder_path` | `text_encoder.onnx` | description ids + mask → encoder states `[1,S,1024]` |
| `dac_decoder_path` | `dac_decoder.onnx` | 9 codebooks → waveform @ 44.1 kHz |
| `prompt_tokenizer_path` | `tokenizer.json` | the checkpoint's prompt vocabulary |
| `description_tokenizer_path` | `description_tokenizer.json` | the Flan-T5 vocabulary |

## The delay pattern

The decoder writes 9 DAC codebooks at once, but staggered: codebook *k* starts *k* steps
late. A frame is only complete 9 steps after it begins, and the first and last 8 steps of
a generation are partly filler.

Two rules keep that staircase well-formed, both ported from upstream:

* forced cells — the lower triangle is BOS, the upper triangle is PAD, and only the `-1`
  cells are ever predicted (`build_delay_pattern_mask` / `apply_delay_pattern_mask`);
* end-of-speech walks one codebook at a time, in order. Codebook *k+1* may not stop before
  codebook *k* has. This is `ParlerTTSLogitsProcessor` upstream and
  `_eos_delay_constraint` here.

So a generation does not end the moment the model wants to stop: it takes 8 more steps to
unwind the staircase.

## Sampling

Upstream defaults to `do_sample: true`, and so does this adapter. Greedy decoding is
available (`do_sample: 0`) and is what the parity tests use, but it tends to run past the
end of the sentence rather than emitting end-of-speech.

| Parameter | Default | Meaning |
|---|---|---|
| `temperature` | `1.0` | sampling temperature |
| `top_k` | `0` | top-k cutoff, `0` disables |
| `do_sample` | `1` | `0` selects greedy |
| `max_new_tokens` | `1500` | frame ceiling; upstream's hard limit is 2610 |
| `seed` | *(none)* | fixes the sampler for reproducible output |
| `description` | *(from the voice)* | overrides the indexed description for one call |

## Voices

The bundled index carries **32 voices**: every speaker AI4Bharat lists as *recommended*
for its language, across 18 languages. The description of each voice is the wording from
AI4Bharat's own model card, so a voice id resolves to exactly the sentence the model was
trained to answer to.

```python
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.merge_default_voices()
voice = manager.voices["indic_parler/rohit/hi"].load()
for chunk in voice.synthesize("नमस्ते, आप कैसे हैं?"):
    audio = chunk.audio_float_array          # float32 @ 44.1 kHz
```

Any of the 69 speakers upstream lists can be reached by passing a `description` at call
time; the index covers the recommended subset.

## Cost

The checkpoint is ~880M parameters and the decoder runs one step per 22 ms of audio, so
this is the slowest engine phoonnx ships. Measured on a 24-core CPU across 12 languages,
real-time factor is **4.1 to 4.7** (mean 4.5). Use it where quality matters more than
latency.

Per-language samples and the intelligibility report live in the mirror under
[`samples/`](https://huggingface.co/OpenVoiceOS/phoonnx-indic-parler/tree/main/samples).
