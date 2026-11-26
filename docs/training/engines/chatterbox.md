# Chatterbox Engine

<!-- agent-hint
covers: how to clone a voice with Chatterbox, its decoding parameters and its four-graph runtime
source: phoonnx/engines/chatterbox.py, phoonnx/voice_index/chatterbox.json, phoonnx/tokenizer.py:ChatterboxMTLTokenizer, phoonnx/lang_preprocess.py:SCRIPT_TRANSFORMS
generated: no
verify: .venv/bin/python -c "import re,sys,json; from phoonnx.engines.chatterbox import ChatterboxAdapter as A; from phoonnx.lang_preprocess import SCRIPT_TRANSFORMS as S; p=open('docs/training/engines/chatterbox.md').read().replace(chr(96),''); d={k:float(v) for k,v in re.findall(r'^\| ([a-z_]+) \| ([\d.]+) \| [A-Z]',p,re.M)}; c={k:float(v) for k,v in re.findall(r'^\| ([A-Z_]+) \| ([\d.]+) \|',p,re.M)}; s=set(re.findall(r'^\| (\w\w) \| [A-Z]',p,re.M)); idx=json.load(open('phoonnx/voice_index/chatterbox.json')); n=int(re.search(r'holds (\d+) Chatterbox voices',p).group(1)); sys.exit(0 if d==A().default_params() and c=={'REPETITION_PENALTY':A.REPETITION_PENALTY,'MAX_NEW_TOKENS':A.MAX_NEW_TOKENS} and s==set(S) and n==len(idx) else 1)"
see: ../../cloning.md, zipvoice.md, ../../engines.md
-->

This page shows how to clone a voice with Chatterbox and control how expressive it sounds.

## What it is

Chatterbox (Resemble AI) is an autoregressive codec language model. A Llama language model makes discrete speech tokens from the text and a reference speaker, and a decoder turns the tokens into a waveform. The adapter runs this loop in its own `synthesize()`. Chatterbox clones a voice from a reference clip with no transcription, and its `exaggeration` setting controls how expressive the speech is.

Pick Chatterbox for expressive zero-shot cloning when you have only a reference clip. [ZipVoice](zipvoice.md) also clones, but it needs the transcription of the clip. The `speech_encoder` reads the voice from the audio alone, in any language. See [Voice cloning](../../cloning.md).

## Install

Reference loading needs the `cloning` extra:

```bash
pip install --pre "phoonnx[cloning]"
```

The Japanese and Chinese transforms of the multilingual voices need `pip install --pre "phoonnx[chatterbox-multilingual]"`, which installs `pykakasi` and `spacy-pkuseg`.

## Clone a voice

Every Chatterbox voice needs a reference clip. Give it in `SynthesisConfig.reference`:

```python
from phoonnx.config import SpeakerReference, SynthesisConfig
from phoonnx.model_manager import TTSModelManager

m = TTSModelManager()
m.merge_default_voices()
voice = m.voices["chatterbox/multilingual/pt"].load()
for chunk in voice.synthesize("Qualquer frase na voz clonada.", SynthesisConfig(
        reference=SpeakerReference(audio="reference.wav"),
        exaggeration=0.6,
        temperature=0.8,
        top_p=0.95)):
    audio = chunk.audio_float_array
```

`exaggeration` goes from 0.0 to 1.0, and a higher value is more expressive. A `SynthesisConfig` field that you set overrides the default of the adapter for one call. With the default `temperature` of 0.8, the adapter samples with top-p. Set `temperature` to 0 or below for greedy decoding.

## Indexed voices

The bundled index holds 42 Chatterbox voices in `phoonnx/voice_index/chatterbox.json`:

- `chatterbox/base/en` and `chatterbox/turbo/en`, the English base and turbo models.
- `chatterbox/multilingual/<lang>`, one entry per language of the multilingual model.
- `chatterbox/lahgtna/<region>`, Arabic dialect voices. Their index entries map the dialect tag to a language token of the model with `lang_tokens`, for example `arz-EG` to `ms`.

The index entries use the q4 quantized `language_model_q4.onnx`. Synthesis time grows with the output length, because the model runs one Llama step per speech token.

## How generation works

A voice has four ONNX graphs. The `language_model` is the primary session of the voice. `embed_tokens`, `speech_encoder` and `conditional_decoder` load from `engine_params`.

```
ref.wav@24k ─► speech_encoder ─► (cond_emb, prompt_token, x_vector, prompt_feat)
text ─► BPE ─► ids ─┐
                    ▼
       embed_tokens(ids, position_ids, exaggeration) ─► embeds ─┐
       cond_emb ─────────────────────────────────────────────────┤ (prefill)
                                                                  ▼
       language_model (Llama, KV-cached) ──loop──► speech tokens ─► conditional_decoder(+x_vector,feat) ─► wav
```

The prefill feeds the conditioning embedding and the text embeddings. Then each step embeds the last token, runs the KV-cached language model, applies a repetition penalty, and samples the next token, or picks the best one when `temperature <= 0`. The loop stops at the speech end token (6562) or after `MAX_NEW_TOKENS`. The adapter reads the KV-cache shape (layers, heads, head size) from the input signature of the language model. Before the `conditional_decoder`, the adapter adds a short run of silence tokens.

All three variants run on one adapter, which reads the inputs and outputs of each graph. The base and multilingual variants use a Llama language model with the positions in `embed_tokens`. The turbo variant uses GPT-2, with the positions fed to the language model.

## Required files

`engine_params` must give `embed_tokens_path`, `speech_encoder_path` and `conditional_decoder_path`. If one is missing, synthesis raises `RuntimeError`:

```
Chatterbox voice missing embed_tokens / speech_encoder / conditional_decoder paths in engine_params
```

A call with no reference clip also raises `RuntimeError`.

## Text tokenization

Chatterbox reads raw text with its own subword BPE, not phonemes. The adapter overrides `BaseOnnxAdapter.encode_text` to tokenize the text directly, and the tokenizer owns the normalization. The base and turbo variants use `phoonnx.tokenizer.BPETokenizer`. The multilingual variant uses its subclass `ChatterboxMTLTokenizer`. Each variant ships its own `tokenizer.json` (a custom BPE for base and multilingual, a GPT-2 BPE for turbo), so a voice must use the tokenizer of its own model.

`ChatterboxMTLTokenizer` lowercases and NFKD-normalizes the text, puts a `[<lang>]` token from the `lang_code` of the voice in front, and replaces each space with `[SPACE]`. Latin, Greek and Cyrillic languages need nothing more. Four languages also need a script transform from `SCRIPT_TRANSFORMS` in `phoonnx.lang_preprocess`:

| Language | Transform | Dependency |
|---|---|---|
| ko | Hangul to Jamo | none |
| ja | Kanji to hiragana | `pykakasi` |
| zh | Cangjie codes | `spacy-pkuseg` and the Hub file `Cangjie5_TC.json` |
| ru | Add stress marks | `stressonnx` |

If a dependency is missing, the transform logs a warning and returns the text unchanged. For Hebrew and Arabic voices, set `diacritize` in the `stages` field of `SynthesisConfig`. It is not a tokenizer transform.

## Reference

### Decoding parameters

| Param | Default | Meaning |
|---|---|---|
| `exaggeration` | 0.5 | Expressiveness (0.0 to 1.0), fed to `embed_tokens` |
| `temperature` | 0.8 | Sampling temperature. 0 or below gives greedy decoding. |
| `top_p` | 0.95 | Nucleus-sampling cutoff, used only when sampling |

### Fixed limits

These constants are class attributes of `ChatterboxAdapter`. No setting changes them.

| Constant | Value | Meaning |
|---|---|---|
| `REPETITION_PENALTY` | 1.2 | Applied over all emitted tokens at each step, before sampling |
| `MAX_NEW_TOKENS` | 1000 | Maximum number of generated speech tokens |

### Upstream

| | |
|---|---|
| Model | [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox) |
| ONNX | [`onnx-community/chatterbox-ONNX`](https://huggingface.co/onnx-community/chatterbox-ONNX) (4 graphs, external-data weights) |
| Converter | [VladOS95-cyber/onnx_conversion_scripts](https://github.com/VladOS95-cyber/onnx_conversion_scripts). The language model is built with [onnxruntime-genai's `builder.py`](https://github.com/microsoft/onnxruntime-genai). |
| Variants | base and multilingual (Llama), turbo (GPT-2 and meanflow) |

---

[← ZipVoice](zipvoice.md) · [Home](../../index.md) · [Spark-TTS →](sparktts.md)
