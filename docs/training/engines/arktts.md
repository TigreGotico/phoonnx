# ArkTTS Engine (Zortzi + Audio8)

<!-- agent-hint
covers: which ArkTTS checkpoints and voices exist, how the engine samples, and what it costs
source: phoonnx/engines/arktts.py, phoonnx/voice_index/arktts.json
generated: no
verify: .venv/bin/python -c "import re,sys,json; import phoonnx.engines.arktts as K; p=open('docs/training/engines/arktts.md').read().replace(chr(96),''); d={k:float(v) for k,v in re.findall(r'^\| ([a-z_]+) \| ([\d.]+) \| [A-Z]',p,re.M)}; ids=set(re.findall(r'(arktts/[a-z0-9-]+/[a-z]+)',p)); idx=set(json.load(open('phoonnx/voice_index/arktts.json'))); n=int(re.search(r'holds (\d+) ArkTTS voices',p).group(1)); r=re.search(r'last (\d+) emitted.*?top-p (\d+\.\d+) at temperature (\d+\.\d+)',p,re.S).groups(); sys.exit(0 if d==K.ArkTTSAdapter().default_params() and ids<=idx and n==len(idx) and (int(r[0]),float(r[1]),float(r[2]))==(K.RAS_WINDOW_SIZE,K.RAS_TOP_P,K.RAS_TEMPERATURE) else 1)"
see: qwen3tts.md, sparktts.md, ../../engines.md
-->

This page shows how to use the ArkTTS voices, with their checkpoints, reference voices and sampling settings.

## What it is

ArkTTS predicts the tokens of a 21.5 Hz neural codec and turns them into a waveform at 44.1 kHz. Like Qwen3-TTS, it has two stacked autoregressive models:

* The slow AR has 24 decoder layers and emits one token per audio frame. That token is codebook 0, and the hidden state of the slow AR conditions the second model.
* The fast AR has 4 decoder layers. It reads the slow hidden state and writes codebooks 1 to 9 of the same frame, one codebook per step.

One 46 ms frame therefore costs one slow step and nine fast steps. The ten codebooks of a
frame are summed into a single embedding and fed back to the slow AR at the next position.

The prompt is a `[1, 11, T]` matrix, not a flat token sequence. Row 0 holds the token stream: a system block, then the codes of the reference clip shifted into the semantic range of the model, then the text. Rows 1 to 10 hold the ten codebooks of the reference clip, aligned under those semantic positions, and zero everywhere else.

Three checkpoints share this architecture, and one adapter runs all of them:

| Checkpoint | Mirror | Index languages |
| --- | --- | --- |
| [`Audio8/Audio8-TTS-Preview-0.6b`](https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6b) | `Phoonnx/arktts-multi-audio8_preview-0.6b-unicode` | Cantonese, Chinese, Dutch, English, French, German, Italian, Japanese, Korean, Polish, Spanish |
| [`itzune/zortzi-tts`](https://huggingface.co/itzune/zortzi-tts) | `Phoonnx/arktts-eu_es-zortzi-unicode` | Basque |
| [`Edge0/Audio8-TTS-Realistic-Air`](https://huggingface.co/Edge0/Audio8-TTS-Realistic-Air) | `Phoonnx/arktts-multi-realistic-unicode` | English |

All three are Apache-2.0. Zortzi is a Basque fine-tune of Audio8 and speaks only Basque. Realistic-Air is a fine-tune of Audio8.

## When to pick it

Pick ArkTTS when you want a Basque codec-LM voice, or 44.1 kHz output. OmniVoice also has a Basque voice, `omnivoice/eu`.

Do not pick ArkTTS for real-time synthesis on a CPU. Two autoregressive stages with ten steps per frame are expensive, and the upstream model cards recommend a GPU for real-time use. For a cheaper multilingual voice, use [Qwen3-TTS](qwen3tts.md). For cloning on a CPU, use [Spark-TTS](sparktts.md).

## Voices

An ArkTTS voice is not a speaker ID. ArkTTS has no speaker table and always emits the `<|speaker:0|>` token. The codec codes of a short reference clip carry the speaker, and the prompt embeds them directly. So the mirrors ship the codes, not the audio, in one small JSON file per voice that also holds the transcription of the clip.

| Voice | Timbre | Reference clip |
| --- | --- | --- |
| `maider` | female | HiTZ-Aholab Basque TTS |
| `antton` | male | HiTZ-Aholab Basque TTS |
| `lj` | female | LJ Speech LJ001-0009 (public domain) |

Audio8 has no reference voices of its own, because upstream ships it as a cloning-only model. Its mirror holds the two Basque clips. They set the timbre, not the language, so the index lists both voices for every Audio8 language. The Realistic-Air mirror holds the `lj` voice.

Voice ids have the form `arktts/<checkpoint>-<voice>/<language>`, for example `arktts/zortzi-maider/eu`, `arktts/audio8-antton/ja` or `arktts/realistic-air-lj/en`. The bundled index holds 25 ArkTTS voices.

```python
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.merge_default_voices()
voice = manager.voices["arktts/zortzi-maider/eu"].load()
for chunk in voice.synthesize("Kaixo, zer moduz?"):
    audio = chunk.audio_float_array
```

To add a voice, encode a clip offline with `scripts/conversion/arktts/mint_voice.py`. Use a short clip with flat intonation, because the intonation of the reference carries into every sentence.

## Sampling

ArkTTS must sample. Both model cards state that greedy decoding runs into repetition loops that never reach the end-of-speech token. Sampling is on by default. The `do_sample` parameter can turn it off, but do not do that. The defaults are the values that the model cards give:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `temperature` | 0.8 | Sampling temperature |
| `top_k` | 50 | Keep the 50 highest-scoring tokens |
| `top_p` | 0.95 | Nucleus cutoff |
| `max_new_tokens` | 512 | Frames per pass, about 24 seconds of audio |

The sampler follows the upstream order, which is not the HuggingFace order. The semantic mask runs first. Then top-k and top-p run together against the softmax of the unscaled logits, and the temperature divides last. If the temperature divides first, as in HuggingFace, other tokens survive the nucleus cut.

The sampler also applies repetition-aware sampling. If a draw lands on a semantic token that is in the last 10 emitted tokens, the sampler discards it and draws again with top-p 0.9 at temperature 1.0. That second draw stops the loops. With sampling off, both draws give the same argmax, so greedy decoding cannot stop.

To make a call reproducible, pass a `seed` in `engine_params` or in `SynthesisConfig.extra`.

## Graphs

| File | Role |
| --- | --- |
| `slow_ar_fp16.onnx` | 24-layer backbone with a KV cache. It is the `model_url` of the voice. |
| `fast_ar_fp16.onnx` | 4-layer depth transformer over the ten codebooks |
| `codec_decoder_fp16.onnx` | `(1, 10, T)` codes to a 44.1 kHz waveform |
| `tokenizer.json` | the model's own Qwen2 subword BPE |
| `voices/<name>.json` | one voice's reference codes and transcription |

The Audio8 and Zortzi mirrors hold fp16 graphs. The Realistic-Air mirror holds fp32 graphs with the same names and an `_fp32` suffix. The upstream `int4` graphs are not mirrored.

The KV cache is a fixed window of 2048 positions, not a growing tensor. Each graph writes its keys and values for the step at `input_pos` and attends over the whole window. The logits of the slow AR are sliced to 4097 entries: the 4096 semantic logits, then end-of-speech. So an index below 4096 is already the value of codebook 0.

The adapter splits text longer than 200 characters at sentence boundaries and synthesizes one chunk per autoregressive pass, because the frame budget of the model runs out before a long chunk ends.

## Limitations

These limits come from the upstream models:

* The models do not always speak numbers correctly. The model cards report that digits come out in a mix of languages, even when written as words. phoonnx expands numbers, dates and units to words with text2spoken before tokenization, so listen to the result when the text has numbers.
* The text normalization does not expand an acronym, so write it as it is spoken. The upstream example writes "TTS" as "te te ese".
* Zortzi speaks only Basque.
* Cloning is offline. phoonnx ships the codec decoder but not the encoder, so a reference clip becomes a voice through `mint_voice.py`, not at synthesis time.

## Licence and attribution

The three checkpoints are Apache-2.0. The `maider` and `antton` clips come from the [HiTZ-Aholab Basque TTS dataset](https://doi.org/10.5281/zenodo.17952596), which is CC BY 4.0. If you share these voices, or audio made with them, give that attribution.

---

[← Qwen3-TTS](qwen3tts.md) · [Home](../../index.md) · [F5-TTS →](f5tts.md)
