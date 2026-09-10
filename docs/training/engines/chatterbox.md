# Chatterbox Engine

This page is for integrators cloning voices with Chatterbox in phoonnx. After
reading it you can clone a voice from a reference clip (no transcription needed),
control expressiveness, and wire up the multi-graph runtime.

> Related: [adapter architecture](../../engines.md) ·
> [configuration](../../configuration.md) · [voice cloning](../../cloning.md) ·
> [ZipVoice — the in-context iterative engine](zipvoice.md)

## What it is

**Chatterbox** (Resemble AI) is an **autoregressive codec-LM** TTS — a
Llama-based language model that generates discrete speech tokens conditioned on
text and a reference speaker, then decodes them to a waveform. It is phoonnx's
autoregressive iterative engine, driven through the overridable
`BaseOnnxAdapter.synthesize()`. Its standout features: **zero-shot cloning from a
reference clip with no transcription** (d-vector style) and an **`exaggeration`**
control for expressiveness.

## When to pick it

Choose Chatterbox for expressive zero-shot cloning when you have only a reference
clip (no transcription) — unlike [ZipVoice](zipvoice.md) (in-context, needs the
reference's transcription). The `speech_encoder` summarizes the voice from audio
alone, in any language. See [Voice Cloning](../../cloning.md).

## Extras needed

Cloning (reference loading) uses `pip install phoonnx[cloning]`. The multilingual
variant's ja/zh script transforms need
`pip install phoonnx[chatterbox-multilingual]` (`pykakasi`, `spacy-pkuseg`); ko
is pure-Python and always on; he/ru degrade gracefully when their optional
stressors/diacritizers are absent.

## Architecture

Four ONNX graphs — `language_model` + `speech_encoder` + `embed_tokens` +
`conditional_decoder`:

```
ref.wav@24k ─► speech_encoder ─► (cond_emb, prompt_token, x_vector, prompt_feat)
text ─► BPE ─► ids ─┐
                    ▼
       embed_tokens(ids, position_ids, exaggeration) ─► embeds ─┐
       cond_emb ─────────────────────────────────────────────────┤ (prefill)
                                                                  ▼
       language_model (Llama, KV-cached) ──loop──► speech tokens ─► conditional_decoder(+x_vector,feat) ─► wav
```

The generation loop is standard codec-LM autoregression: prefill (cond embedding
+ text embeddings), then step-by-step — `embed_tokens` the last token, run the
KV-cached Llama, apply a repetition penalty, then **sample the next token
(temperature / top-p) or pick greedily when `temperature <= 0`**, stopping at the
speech-EOS token (`6562`) or after `MAX_NEW_TOKENS`. The KV-cache shape
(layers / heads / head-dim) is read from the LM's own input signature, so no
model config is hardcoded. The `language_model` is the voice's primary
`session`; the other three graphs load from `engine_params`.

### Decoding parameters and caps

Defaults come from `SynthesisConfig` (`temperature=0.8`, `top_p=0.95`,
`exaggeration=0.5`), read per call by the adapter:

| Param | Default | Meaning |
|---|---|---|
| `exaggeration` | 0.5 | Expressiveness (0.0–1.0), fed to `embed_tokens` |
| `temperature` | 0.8 | Sampling temperature; **`0` (or ≤0) = greedy decoding** |
| `top_p` | 0.95 | Nucleus-sampling cutoff (only when sampling) |

Two internal caps are fixed by the adapter, not configurable:

| Constant | Value | Meaning |
|---|---|---|
| `REPETITION_PENALTY` | 1.2 | Applied over all emitted tokens each step, before sampling, to suppress codec-LM babble |
| `MAX_NEW_TOKENS` | 1000 | Hard cap on generated speech tokens (stops runaway generation) |

A trailing silence token is appended before the `conditional_decoder`.

## Cloning + exaggeration

```python
voice.synthesize("Any sentence in the cloned voice.", SynthesisConfig(
    speaker_reference="reference.wav",   # no transcription needed (d-vector)
    exaggeration=0.6,                    # 0.0–1.0, default 0.5; higher = more expressive
    temperature=0.8,                     # sampling temperature (0 = greedy)
    top_p=0.95,                          # nucleus sampling cutoff
))
```

## Runtime configuration

Chatterbox splits inference across **four ONNX graphs** and needs a
`tokenizer.json` (subword BPE). The `language_model` is the primary voice
session; the other three graphs load from `engine_params`. The adapter raises a
`RuntimeError` at synthesis if any of the auxiliary paths is missing:

> `Chatterbox voice missing embed_tokens / speech_encoder / conditional_decoder
> paths in engine_params`

so `engine_params` must supply `embed_tokens_path`, `speech_encoder_path` and
`conditional_decoder_path`. A synthesis call without `reference_audio` (the
reference clip) also raises `RuntimeError` — Chatterbox always needs a reference.

Use the quantized `language_model_q4.onnx` for the smallest/fastest build
(~350 MB vs ~2 GB); synthesis cost scales with output length (one Llama step per
speech token).

## Text tokenization

Chatterbox tokenizes **raw text with its own subword BPE**, not phonemes. The
adapter overrides `BaseOnnxAdapter.encode_text` to BPE the text directly (phoneme
front ends would strip punctuation / expand numbers); the tokenizer owns
normalization. Base + turbo use `phoonnx.tokenizer.BPETokenizer`; the
multilingual variant uses its `ChatterboxMTLTokenizer` subclass, which adds the
language-aware front end (below). **Each variant ships its own `tokenizer.json`**
— base/multilingual a custom BPE, turbo a GPT-2 BPE — so a voice must point its
`BPETokenizer` at the matching model's tokenizer.

### Multilingual language selection

`ChatterboxMTLTokenizer` prefixes a `[<lang>]` token from the voice's
`lang_code`, lowercases + NFKD-normalises, and replaces spaces with `[SPACE]`.
Latin/Greek/Cyrillic languages (en, pt, es, fr, de, it, nl, el, tr, sv, …) work
directly. Four scripts also need a per-language transform (in
`phoonnx.lang_preprocess`, dispatched via `SCRIPT_TRANSFORMS`):

| lang | transform | dependency |
|---|---|---|
| **ko** | Hangul → Jamo | none (pure Python, always on) |
| **ja** | kanji → hiragana | `pykakasi` |
| **zh** | Cangjie codes | `spacy-pkuseg` (+ HF `Cangjie5_TC.json`) |
| **ru** | add stress marks | `stressonnx` (pure-onnxruntime stressor, no torch) |

Install the ja/zh deps with `pip install phoonnx[chatterbox-multilingual]`.
Hebrew/Arabic use the universal `add_diacritics` SynthesisConfig flag (set it on
those voices), not a tokenizer transform. Any missing dependency degrades to the
raw text with a warning (so the rest of the pipeline still runs).

## Variants

All three variants run on one adapter, which reads each graph's I/O signature:
base + multilingual use a Llama LM (positions in `embed_tokens`), turbo uses
GPT-2 (positions fed to the LM).

## Gotchas / aliases

- **Decoding is sampling by default**, not greedy: with the default
  `temperature=0.8` the adapter samples with top-p `0.95`. Greedy decoding
  happens only when `temperature <= 0`.
- **Four graphs are mandatory:** missing `embed_tokens` / `speech_encoder` /
  `conditional_decoder` paths, or a missing `reference_audio`, each raise a
  `RuntimeError`.
- **Match the tokenizer to the model:** each variant's `tokenizer.json` differs.

## Upstream

| | |
|---|---|
| Model | [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox) |
| ONNX | [`onnx-community/chatterbox-ONNX`](https://huggingface.co/onnx-community/chatterbox-ONNX) (4 graphs, external-data weights) |
| Converter | [VladOS95-cyber/onnx_conversion_scripts](https://github.com/VladOS95-cyber/onnx_conversion_scripts) — the LM is built with [onnxruntime-genai's `builder.py`](https://github.com/microsoft/onnxruntime-genai) |
| Variants | base, multilingual (Llama), turbo (GPT-2 + meanflow) — all supported by the I/O-driven adapter |
