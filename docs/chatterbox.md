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
| Variants | base ✓, **multilingual** ✓ (Llama); **turbo** ✓ (GPT-2 + meanflow) — all supported by the I/O-driven adapter |

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

Chatterbox tokenizes **raw text with its own subword BPE**, not phonemes. The adapter
overrides `BaseOnnxAdapter.encode_text` to BPE the text directly (phoneme front ends
would strip punctuation / expand numbers); the tokenizer owns normalization. Base +
turbo use `phoonnx.tokenizer.BPETokenizer`; the multilingual variant uses its
`ChatterboxMTLTokenizer` subclass, which adds the language-aware front end (below). Both
are the subword complement to the vocab-lookup `TTSTokenizer`.

## Cloning + exaggeration

```python
voice.synthesize("Any sentence in the cloned voice.", SynthesisConfig(
    speaker_reference="reference.wav",   # no transcription needed (d-vector)
    exaggeration=0.6,                    # 0.0–1.0, default 0.5; higher = more expressive
    temperature=0.8,                     # sampling temperature (0 = greedy)
    top_p=0.95,                          # nucleus sampling cutoff
))
```

See [Voice Cloning](cloning.md). Unlike [ZipVoice](zipvoice.md) (in-context, needs the
reference's transcription), Chatterbox is **d-vector** — the `speech_encoder` summarizes
the voice from audio alone, in any language.

## Variants + tokenizers

All three variants run on one adapter, which reads each graph's I/O signature: base +
multilingual use a Llama LM (positions in embed_tokens), turbo uses GPT-2 (positions fed
to the LM). **Each variant ships its own `tokenizer.json`** — base/multilingual a custom
BPE, **turbo a GPT-2 BPE** — so a voice must point its `BPETokenizer` at the matching
model's tokenizer. The repetition penalty runs over all emitted tokens and decoding is
temperature/top-p sampling (greedy at `temperature=0`); a trailing silence token is
appended before the decoder.

**Multilingual language selection:** `ChatterboxMTLTokenizer` prefixes a `[<lang>]`
token from the voice's `lang_code`, lowercases + NFKD-normalises, and replaces spaces
with `[SPACE]`. Latin/Greek/Cyrillic languages (en, pt, es, fr, de, it, nl, el, tr, sv,
…) work directly. Five scripts also need a per-language transform (in
`phoonnx.lang_preprocess`, dispatched via `SCRIPT_TRANSFORMS`):

| lang | transform | dependency |
|---|---|---|
| **ko** | Hangul → Jamo | none (pure Python, always on) |
| **ja** | kanji → hiragana | `pykakasi` |
| **zh** | Cangjie codes | `spacy-pkuseg` (+ HF `Cangjie5_TC.json`) |
| **ru** | add stress marks | `russian_text_stresser` (heavy: spaCy + Wiktionary DB; not on PyPI — install manually) |

Install the ja/zh deps with `pip install phoonnx[chatterbox-multilingual]`. Hebrew/Arabic
use the universal `add_diacritics` SynthesisConfig flag (set it on those voices), not a
tokenizer transform. `ru` needs `russian_text_stresser` installed manually. Any missing dependency degrades to the raw text
with a warning (so the rest of the pipeline still runs).

## A note on performance

It's autoregressive, so synthesis cost scales with output length (one Llama step per
speech token). Use the quantized `language_model_q4.onnx` for the smallest/fastest
build (~350 MB vs ~2 GB).
