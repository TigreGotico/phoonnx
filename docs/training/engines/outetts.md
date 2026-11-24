# OuteTTS Engine

<!-- agent-hint
covers: which OuteTTS languages and speakers exist, what the two graphs do and the decoding defaults
source: phoonnx/engines/outetts.py, phoonnx/voice_index/outetts.json
generated: no
verify: .venv/bin/python -c "import re,sys,json; import phoonnx.engines.outetts as O; p=open('docs/training/engines/outetts.md').read().replace(chr(96),''); d={k:float(v) for k,v in re.findall(r'^\| ([a-z_]+) \| ([\d.]+) \| [A-Z]',p,re.M)}; idx=set(json.load(open('phoonnx/voice_index/outetts.json'))); l={m:set(re.findall(r'^\| (\w\w) \|',p.split('### '+m+' languages')[1].split('###')[0],re.M)) for m in ('0.6B','1B')}; w=int(re.search(r'last (\d+) tokens',p).group(1)); sys.exit(0 if d=={k:float(x) for k,x in O.OuteTTSAdapter().default_params().items()} and {f'outetts/{m}/{c}' for m in l for c in l[m]}==idx and w==O.REPETITION_WINDOW else 1)"
see: sparktts.md, ../../cloning.md, ../../engines.md
-->

This page shows how to use the OuteTTS voices, with their languages, speaker profiles and decoding parameters.

## What it is

OuteTTS 1.0 (OuteAI) is a decoder-only language model that predicts audio codec tokens. Its vocabulary holds the tokens of a text model and the two codebooks of DAC.speech.v1.0, a 24 kHz codec by IBM Research at 1.5 kbps. The model reads text and writes interleaved `<|c1_N|><|c2_N|>` pairs, and the codec decoder turns that stream into a waveform.

OuteTTS has no phonemizer, no duration model and no vocoder. The model reads raw text in the script of each language, so one checkpoint covers many languages.

Two checkpoints share the interface:

| Checkpoint | Backbone | Index voices | Licence |
| :--- | :--- | :--- | :--- |
| `OuteTTS-1.0-0.6B` | Qwen3-0.6B | 14 | Apache-2.0 |
| `Llama-OuteTTS-1.0-1B` | Llama-3.2-1B | 9 | CC-BY-NC-SA-4.0 |

The index uses the 0.6B checkpoint for its 14 languages, and the 1B checkpoint only for the nine languages that the 0.6B checkpoint does not cover.

Pick OuteTTS when you want one small model to speak many languages from raw text, with no phonemizer to install. Do not pick it when you need speed. One codec frame is two passes through the language model, and one second of speech is about 47 frames, so one second of audio costs about 94 model calls. The non-autoregressive engines are much faster.

## Speak with a voice

A voice id names the checkpoint and the language, for example `outetts/0.6B/de` or `outetts/1B/sw`. So a 0.6B voice never uses the non-commercial 1B weights.

```python
from phoonnx.config import SynthesisConfig
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.merge_default_voices()
voice = manager.voices["outetts/0.6B/de"].load()
syn = SynthesisConfig(extra={"voice": "en-female-1-neutral"})
for chunk in voice.synthesize("Der alte Leuchtturm steht an der Küste.", syn):
    audio = chunk.audio_float_array  # 24 kHz mono
```

The model can also speak languages that it was not trained on, with variable quality. The voice index lists only the trained languages.

## Speakers

An OuteTTS voice needs a speaker profile, or it makes up a random voice. A profile is an in-context audio prompt: a transcript in which each word has a duration, three prosody buckets (energy, spectral centroid, pitch) and its DAC codes. The transcript of the profile goes in front of your text, and its codes start the generated stream, so the model continues a real utterance.

All bundled voices use the OuteAI profile `en-female-1-neutral`. The model carries the speaker across languages, so this English profile gives an English accent to the other languages. For production in one language, build a profile from a clip in that language. When a voice ships more than one profile, select one per call with `extra={"voice": ...}`. An unknown profile name raises `ValueError`.

phoonnx cannot build a profile from a reference clip, because that needs Whisper word alignment, the DAC encoder and a prosody analysis. A call with `reference` raises `RuntimeError`. Build the profile with the upstream `outetts` package and ship the JSON file.

Sampling can make the model repeat part of a sentence. Greedy decoding (`temperature=0`) does not loop, but it is not the default.

## How generation works

```
text ─► normalize ─► chunk (10-30 words)
                          │
speaker profile ──────────┤   transcript + per-word DAC codes
                          ▼
      prompt = <|im_start|> <|text_start|>...<|text_end|>
               <|audio_start|> <word blocks> <|word_start|>
                          │
                          ▼
   model.onnx (KV-cached) ──loop──► <|c1_N|><|c2_N|> ...
                          │
                          ▼
   decoder_model.onnx(audio_codes[1,2,T]) ─► wav @ 24 kHz
                          │
                          ▼
              fade, then loudness normalize to -18 LUFS
```

| Graph | Input | Output |
| :--- | :--- | :--- |
| `model.onnx` | `input_ids`, `attention_mask`, `position_ids`, `past_key_values.<i>.<key\|value>` | `logits`, `present.<i>.<key\|value>` |
| `decoder_model.onnx` | `audio_codes` `[1,2,T]` int64 | `audio_values` `[1,1,T*512]` |

The adapter reads the KV-cache shape (layers, heads, head size) from the input signature of the language model. It reads only `logits[0, -1]`, so a graph that returns the logits of every position and a graph that returns only the last one both work.

The adapter fades each decoder chunk in and out over 15 ms. Then it normalizes the loudness to -18 LUFS with a -1 dBFS peak ceiling, as upstream does in its codec wrapper. phoonnx implements ITU-R BS.1770-4 in NumPy, so it needs no `pyloudnorm`.

The adapter applies temperature, then top-k, top-p and min-p, in the order of HuggingFace `generate`, because `outetts` runs this checkpoint through the HuggingFace backend by default. The repetition penalty looks back over the last 64 tokens only, as in the patched `RepetitionPenaltyLogitsProcessor` of upstream. The window includes the prompt.

The adapter splits text into chunks of 10 to 30 words at sentence boundaries, and it synthesizes one chunk per pass. For Chinese and Japanese, it counts each CJK character as a word, where upstream segments with MeCab. This changes only where a long passage is cut.

## Weights and licensing

The index voices use three Hub repositories:

- `Phoonnx/outetts-multi-outeai-0.6b-unicode`: the 0.6B language model, its tokenizer and the speaker profiles.
- `Phoonnx/outetts-multi-outeai-1b-unicode`: the 1B language model. `scripts/conversion/outetts/export_outetts_onnx.py` exported it from `OuteAI/Llama-OuteTTS-1.0-1B`, because the float32 1B ONNX of OuteAI does not reproduce its torch weights.
- `Phoonnx/outetts-dac`: the DAC decoder.

The index uses float32 graphs only.

| Artifact | Licence |
| :--- | :--- |
| `OuteTTS-1.0-0.6B` | Apache-2.0 |
| `Llama-OuteTTS-1.0-1B` | CC-BY-NC-SA-4.0, no commercial use (the nine `outetts/1B/*` voices) |
| `DAC.speech.v1.0` (IBM Research) | CDLA-Permissive-2.0 |
| Speaker profiles (from the `outetts` package) | Apache-2.0 |

OuteAI owns the model and the research credit. Do not clone a voice without the permission of the speaker.

## Reference

### 0.6B languages

| Code | Language |
| :--- | :--- |
| en | English |
| zh | Chinese |
| nl | Dutch |
| fr | French |
| ka | Georgian |
| de | German |
| hu | Hungarian |
| it | Italian |
| ja | Japanese |
| ko | Korean |
| lv | Latvian |
| pl | Polish |
| ru | Russian |
| es | Spanish |

### 1B languages

| Code | Language |
| :--- | :--- |
| ar | Arabic |
| be | Belarusian |
| bn | Bengali |
| lt | Lithuanian |
| fa | Persian |
| pt | Portuguese |
| sw | Swahili |
| ta | Tamil |
| uk | Ukrainian |

### Decoding parameters

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `temperature` | 0.4 | Sampling temperature. 0 decodes greedily. |
| `top_k` | 40 | Keep the 40 highest-scoring tokens |
| `top_p` | 0.9 | Keep the smallest set of tokens that covers 90 % of the probability |
| `min_p` | 0.05 | Drop every token below 5 % of the probability of the top token |
| `repetition_penalty` | 1.1 | Applied over the last 64 tokens |
| `max_new_tokens` | 4096 | Hard stop, about 44 seconds |
| `max_chunk_words` | 30 | Words per autoregressive pass |

---

[Docs index](../../index.md) · Previous: [Shami](shami.md) · Next: [OmniVoice](omnivoice.md)
