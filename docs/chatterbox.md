# Chatterbox Engine

**Chatterbox** (Resemble AI) is an **autoregressive codec-LM** TTS — a Llama-based
language model that generates discrete speech tokens conditioned on text and a reference
speaker, then decodes them to a waveform. It is phoonnx's first **autoregressive**
engine (the second iterative subtype after the flow-matching [ZipVoice](zipvoice.md)),
driven through the overridable `BaseOnnxAdapter.synthesize()`.

Its standout features: **zero-shot cloning from a reference clip with no transcription**
(d-vector style) and an **`exaggeration`** control for expressiveness.

## Upstream

| | |
|---|---|
| Model | [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox) |
| ONNX | [`onnx-community/chatterbox-ONNX`](https://huggingface.co/onnx-community/chatterbox-ONNX) (4 graphs, external-data weights) |
| Converter | [VladOS95-cyber/onnx_conversion_scripts](https://github.com/VladOS95-cyber/onnx_conversion_scripts) — the LM is built with [onnxruntime-genai's `builder.py`](https://github.com/microsoft/onnxruntime-genai) |
| Variants | base, **multilingual**, **turbo** |

## Architecture

Four ONNX graphs:

```
ref.wav@24k ─► speech_encoder ─► (cond_emb, prompt_token, x_vector, prompt_feat)
text ─► BPE ─► ids ─┐
                    ▼
       embed_tokens(ids, position_ids, exaggeration) ─► embeds ─┐
       cond_emb ─────────────────────────────────────────────────┤ (prefill)
                                                                  ▼
       language_model (Llama, KV-cached) ──loop──► speech tokens ─► conditional_decoder(+x_vector,feat) ─► wav
```

The generation loop is standard codec-LM autoregression: prefill (cond embedding + text
embeddings), then step-by-step — `embed_tokens` the last token, run the KV-cached Llama,
apply a repetition penalty, greedily pick the next token, stop at the speech-EOS
(`6562`). The KV-cache shape (layers / heads / head-dim) is read from the LM's own input
signature, so no model config is hardcoded. The `language_model` is the voice's primary
`session`; the other three graphs load from `engine_params`.

## Text tokenization

Chatterbox tokenizes **raw text with its own subword BPE**, not phonemes — so it uses
`phoonnx.tokenizer.BPETokenizer` (the vocab-lookup `TTSTokenizer` is the *other*
implementation of the same tokenizer role). Because phoneme front ends normalize
(strip punctuation, expand numbers), the adapter sets `tokenizes_raw_text = True` and
`TTSVoice` feeds it the raw text untouched; Chatterbox's BPE does its own normalization.

## Cloning + exaggeration

```python
voice.synthesize("Any sentence in the cloned voice.", SynthesisConfig(
    speaker_reference="reference.wav",   # no transcription needed (d-vector)
    exaggeration=0.6,                    # 0.0–1.0, default 0.5; higher = more expressive
))
```

See [Voice Cloning](cloning.md). Unlike [ZipVoice](zipvoice.md) (in-context, needs the
reference's transcription), Chatterbox is **d-vector** — the `speech_encoder` summarizes
the voice from audio alone, in any language.

## A note on performance

It's autoregressive, so synthesis cost scales with output length (one Llama step per
speech token). Use the quantized `language_model_q4.onnx` for the smallest/fastest
build (~350 MB vs ~2 GB).
