# Qwen3-TTS Engine

<!-- agent-hint
covers: which Qwen3-TTS timbres and languages exist, what the seven graphs do and the control parameters
source: phoonnx/engines/qwen3tts.py, phoonnx/voice_index/qwen3tts.json
generated: no
verify: .venv/bin/python -c "import re,sys,json; import phoonnx.engines.qwen3tts as Q; p=open('docs/training/engines/qwen3tts.md').read().replace(chr(96),''); d={k:float(v) for k,v in re.findall(r'^\| ([a-z_]+) \| ([\d.]+) \| [A-Z]',p,re.M)}; s=set(re.findall(r'^\| ([a-z_]+) \| [a-z][^|]* \| [A-Z]',p,re.M)); ids=set(re.findall(r'(qwen3tts/[a-z_]+/[a-z]{2})',p)); sys.exit(0 if d==Q.Qwen3TTSAdapter().default_params() and s==set(Q.SPEAKER_IDS) and ids<=set(json.load(open('phoonnx/voice_index/qwen3tts.json'))) and len(re.findall(r'^\| qwen3tts/',p,re.M))==len(json.load(open('phoonnx/voice_index/qwen3tts.json'))) else 1)"
see: sparktts.md, arktts.md, ../../engines.md
-->

This page shows how to use the Qwen3-TTS voices, with their timbres, languages and decoding parameters.

## What it is

Qwen3-TTS (Alibaba Qwen) predicts the tokens of a 12.5 Hz neural codec and turns them
into a waveform at 24 kHz. It has two stacked autoregressive models, not one:

* The talker has 28 decoder layers and emits one token per audio frame. That token is code group 0, and the hidden state of the talker conditions the second model.
* The code predictor has 5 decoder layers. It reads the hidden state of the talker and writes the other 15 code groups of the same frame, one group per step.

One 80 ms frame therefore costs one talker step and 15 code-predictor steps. The 16
groups of a frame are summed into a single embedding and fed back to the talker together
with the next slice of the text.

The talker reads embeddings, never token ids: at every prompt position a text hidden
state and a codec hidden state are added together. The prompt holds the assistant role
tokens, a language block, the speaker, the whole text against codec padding, and finally
the codec BOS that starts the audio.

phoonnx ships the 0.6B CustomVoice checkpoint. It has nine trained timbres and no
speaker encoder, so a voice is a name, not a reference clip, and this engine does not do
[cloning](../../cloning.md).

## When to pick it

Pick Qwen3-TTS when one model must speak many languages with a fixed voice. It speaks Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish and Italian.

To clone a speaker, use [Spark-TTS](sparktts.md) or [Chatterbox](chatterbox.md) instead. Qwen3-TTS is also a poor fit for real-time synthesis on a small CPU, because two autoregressive stages cost much more than a single-pass engine.

## Voices

| Voice | Timbre | Native language |
| --- | --- | --- |
| `vivian` | bright young female | Chinese |
| `serena` | warm gentle young female | Chinese |
| `uncle_fu` | seasoned male, mellow | Chinese |
| `dylan` | youthful male | Chinese (Beijing) |
| `eric` | lively male | Chinese (Sichuan) |
| `ryan` | dynamic male | English |
| `aiden` | sunny American male | English |
| `ono_anna` | playful female | Japanese |
| `sohee` | warm female | Korean |

Any timbre speaks any of the ten languages. The table gives the language of each timbre's recordings, where it sounds best. `dylan` and `eric` are dialect voices. When the language is Chinese or not set, they speak their dialect, whatever language the caller asks for.

Voice ids have the form `qwen3tts/<timbre>/<language>`. The bundled index holds these 15 voices:

| voice_id |
| --- |
| `qwen3tts/vivian/zh` |
| `qwen3tts/serena/zh` |
| `qwen3tts/uncle_fu/zh` |
| `qwen3tts/dylan/zh` |
| `qwen3tts/eric/zh` |
| `qwen3tts/ryan/en` |
| `qwen3tts/ryan/de` |
| `qwen3tts/ryan/es` |
| `qwen3tts/ryan/fr` |
| `qwen3tts/ryan/it` |
| `qwen3tts/ryan/pt` |
| `qwen3tts/ryan/ru` |
| `qwen3tts/aiden/en` |
| `qwen3tts/ono_anna/ja` |
| `qwen3tts/sohee/ko` |

Load one like any other index voice:

```python
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.merge_default_voices()
voice = manager.voices["qwen3tts/ryan/en"].load()
for chunk in voice.synthesize("The quick brown fox jumps over the lazy dog."):
    audio = chunk.audio_float_array
```

## Graphs

| File | What it does |
| --- | --- |
| `talker.onnx` | the 28-layer talker, one KV-cached step per frame (the voice's `model_url`) |
| `text_embed.onnx` | text ids to projected text hidden states |
| `codec_embed.onnx` | talker codec ids to hidden states |
| `code_predictor_prefill.onnx` | talker hidden plus code group 0, gives code group 1 |
| `code_predictor_step.onnx` | code group n gives code group n+1 |
| `sub_codec_embed.onnx` | code-group token to its group's embedding |
| `codec_decoder.onnx` | 16 code groups per frame to 24 kHz audio |

Every group has its own embedding table and its own output head, so the step graph takes the group index as an input and gathers both from a stacked weight. The graphs are float32 and hold about 4.4 GB in total. All 15 voices share them, so the first voice downloads them once.

The mirror is [`Phoonnx/qwen3tts-multi-customvoice-0.6b-unicode`](https://huggingface.co/Phoonnx/qwen3tts-multi-customvoice-0.6b-unicode).
The export script is `scripts/conversion/qwen3tts/export_qwen3tts_onnx.py`.

## Control parameters

Defaults come from the checkpoint's own `generation_config.json`:

| Parameter | Default | What it does |
| --- | --- | --- |
| `temperature` | 0.9 | Talker sampling temperature |
| `top_k` | 50 | Talker top-k |
| `top_p` | 1.0 | Talker nucleus |
| `repetition_penalty` | 1.05 | Penalty over the code-group-0 tokens already emitted |
| `subtalker_temperature` | 0.9 | Code-predictor temperature |
| `subtalker_top_k` | 50 | Code-predictor top-k |
| `subtalker_top_p` | 1.0 | Code-predictor nucleus |

The adapter also reads `do_sample`, `seed` and `max_new_tokens` from `SynthesisConfig.extra`. `do_sample=False` decodes greedily and gives the same result on each call. `scripts/conversion/qwen3tts/verify_parity.py` uses it to compare with the upstream PyTorch model. Do not use greedy decoding for synthesis: on some inputs it makes this model repeat itself and never stop, so upstream samples by default.

A rare sampled draw can also make the model continue past the text. To bound the output, set `max_new_tokens`. At 12.5 frames per second, 250 frames is 20 seconds of audio.

## Licence

The model is Apache-2.0, the licence of the upstream model. The weights, the architecture and the nine timbres are the work of the Alibaba Qwen team. Cite the Qwen3-TTS technical report ([arXiv 2601.15621](https://arxiv.org/abs/2601.15621)) when you use them.

---

[← Spark-TTS](sparktts.md) · [Home](../../index.md) · [ArkTTS →](arktts.md)
