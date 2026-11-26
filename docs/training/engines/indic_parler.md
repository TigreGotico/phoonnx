# Indic Parler-TTS

<!-- agent-hint
covers: how Indic Parler-TTS picks a voice from a text description, which voices exist and how intelligible they are
source: phoonnx/engines/indic_parler.py, phoonnx/voice_index/indic_parler.json
generated: no
verify: .venv/bin/python -c "import re,sys,json; from phoonnx.engines.indic_parler import IndicParlerAdapter as A; p=open('docs/training/engines/indic_parler.md').read().replace(chr(96),''); d={k:float(v) for k,v in re.findall(r'^\| ([a-z_]+) \| ([\d.]+) \| [A-Z]',p,re.M)}; idx=json.load(open('phoonnx/voice_index/indic_parler.json')); v={n.strip()+'/'+l for l,ns in re.findall(r'^\| [^|]+ \(([a-z]+)\) \| ([^|]+) \|',p,re.M) for n in ns.split(',')}; n=int(re.search(r'holds (\d+) voices',p).group(1)); sys.exit(0 if d==A().default_params() and n==len(idx) and v=={k.split('/',1)[1] for k in idx} and A.MAX_LENGTH==int(re.search(r'hard limit of (\d+)',p).group(1)) else 1)"
see: ../../engines.md, ../../configuration.md
-->

This page shows how to use the Indic Parler-TTS voices, which you select with a description of the voice in words.

## What it is

[Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts) (AI4Bharat) is the Indic fine-tune of [Parler-TTS](https://github.com/huggingface/parler-tts) (Yoach Lacombe, Vaibhav Srivastav, Sanchit Gandhi). It speaks Indic languages and English, at 44.1 kHz. A voice config selects the adapter with `engine: indic_parler`.

You do not select a voice with a speaker ID or a reference clip. You describe it:

> *"Rohit's voice is clear and expressive, with a moderate speed and pitch, recorded in a very close-sounding environment with no background noise."*

## Speak with a voice

The bundled index holds 32 voices: the speakers that AI4Bharat recommends for each language, in 18 languages. The description of each voice is the wording of the AI4Bharat model card, so a voice id gives the sentence that the model was trained on.

```python
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.merge_default_voices()
voice = manager.voices["indic_parler/rohit/hi"].load()
for chunk in voice.synthesize("नमस्ते, आप कैसे हैं?"):
    audio = chunk.audio_float_array  # float32 at 44.1 kHz
```

To use another speaker of the upstream list, or a voice of your own words, pass a `description` for one call:

```python
from phoonnx.config import SynthesisConfig

syn = SynthesisConfig(extra={"description": "Divya's voice is calm and slow."})
for chunk in voice.synthesize("नमस्ते, आप कैसे हैं?", syn):
    audio = chunk.audio_float_array
```

The checkpoint also supports Urdu (`ur`), Sindhi (`sd`), Maithili (`mai`), Konkani (`kok`) and Santali (`sat`). AI4Bharat lists no recommended speaker for them, so the index has no voice for them. Reach them with a custom `description`.

The decoder runs one step per codec frame, plus 8 steps for the delay pattern, so synthesis is slow on a CPU. Samples of each language are in the [`samples/`](https://huggingface.co/Phoonnx/parlertts-multi-indic-unicode/tree/main/samples) folder of the mirror.

## How it works

Indic Parler is an encoder-decoder model, and the conditioning arrives on two paths:

1. The description goes through a frozen Flan-T5 encoder. Every decoder layer cross-attends to the result. Those cross-attention keys and values do not change during decoding, so the prefill graph computes them once and the decode graph reads them at each step.
2. `embed_prompts` embeds the text to speak, and the adapter puts it in front of the input embeddings of the decoder. The text is part of the self-attention stream, not the cross-attention stream. Upstream calls this `prompt_cross_attention: false`.

So there are two tokenizers with different vocabularies: the tokenizer of the checkpoint for the text, and the Flan-T5 tokenizer for the description. If you swap them, the model speaks, but it does not speak your text.

### Graphs

All voices use the graphs of [`Phoonnx/parlertts-multi-indic-unicode`](https://huggingface.co/Phoonnx/parlertts-multi-indic-unicode).

| `engine_params` key | Graph | Contract |
|---|---|---|
| the model of the voice | `decoder_prefill.onnx` | codec ids, prompt ids and encoder states to `logits[9,1088]`, the self-attention cache and the cross-attention cache |
| `decoder_decode_path` | `decoder_decode.onnx` | one codec frame and both caches to `logits[9,1088]` and the self-attention cache |
| `text_encoder_path` | `text_encoder.onnx` | description ids and mask to encoder states |
| `dac_decoder_path` | `dac_decoder.onnx` | 9 codebooks to a waveform at 44.1 kHz |
| `prompt_tokenizer_path` | `tokenizer.json` | the prompt vocabulary of the checkpoint |
| `description_tokenizer_path` | `description_tokenizer.json` | the Flan-T5 vocabulary |

### The delay pattern

The decoder writes 9 DAC codebooks at once, but staggered: codebook *k* starts *k* steps late. A frame is complete 9 steps after it begins, and the first and last 8 steps of a generation are partly filler.

Two rules from upstream keep that staircase correct:

* The lower triangle holds BOS, the upper triangle holds PAD, and the model predicts only the `-1` cells (`build_delay_pattern_mask`, `apply_delay_pattern_mask`).
* End-of-speech moves one codebook at a time, in order. Codebook *k+1* cannot stop before codebook *k*. Upstream calls this `ParlerTTSLogitsProcessor`, and the adapter calls it `_eos_delay_constraint`.

So a generation takes 8 more steps after the model decides to stop.

## Sampling

Upstream samples by default (`do_sample: true`), and so does the adapter. Greedy decoding (`do_sample: 0`) is available, but it tends to continue past the end of the sentence instead of emitting end-of-speech.

## Reference

### Decoding parameters

| Parameter | Default | Meaning |
|---|---|---|
| `temperature` | 1.0 | Sampling temperature |
| `top_k` | 0 | Top-k cutoff. 0 turns it off. |
| `do_sample` | 1 | Set 0 for greedy decoding |
| `max_new_tokens` | 1500 | Frame ceiling, under the upstream hard limit of 2610 positions |

`seed` (no default) fixes the sampler, and `description` (default: the description of the voice) overrides the voice for one call.

### Indexed voices

| Language | Voices |
|---|---|
| Assamese (as) | amit, sita |
| Bengali (bn) | arjun, aditi |
| Bodo (brx) | bikram, maya |
| Chhattisgarhi (hne) | bhanu, champa |
| Dogri (doi) | karan |
| English (en) | thoma, mary |
| Gujarati (gu) | yash, neha |
| Hindi (hi) | rohit, divya |
| Kannada (kn) | suresh, anu |
| Malayalam (ml) | anjali, harish |
| Manipuri (mni) | laishram, ranjit |
| Marathi (mr) | sanjay, sunita |
| Nepali (ne) | amrita |
| Odia (or) | manas, debjani |
| Punjabi (pa) | divjot, gurpreet |
| Sanskrit (sa) | aryan |
| Tamil (ta) | jaya |
| Telugu (te) | prakash, lalitha |

A voice id has the form `indic_parler/<voice>/<language>`, for example `indic_parler/rohit/hi`.

### Licence

The mirror is Apache-2.0, the licence of the upstream checkpoint.

---

[← Rápido](rapido.md) · [Home](../../index.md) · [Architecture →](../../architecture.md)
