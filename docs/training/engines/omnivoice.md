# OmniVoice

<!-- agent-hint
covers: how OmniVoice decodes, how to clone with it, its parameters and its cost
source: phoonnx/engines/omnivoice.py, phoonnx/voice_index/omnivoice.json
generated: no
verify: .venv/bin/python -c "import re,sys,json; from phoonnx.engines.omnivoice import OmniVoiceAdapter as A; p=open('docs/training/engines/omnivoice.md').read().replace(chr(96),''); d={k:float(v) for k,v in re.findall(r'^\| ([a-z_]+) \| ([\d.]+) \| [A-Z]',p,re.M)}; n=int(re.search(r'holds (\d+) OmniVoice voices',p).group(1)); sys.exit(0 if d=={k:float(x) for k,x in A().default_params().items()} and n==len(json.load(open('phoonnx/voice_index/omnivoice.json'))) else 1)"
see: ../../cloning.md, zipvoice.md, f5tts.md
-->

This page shows how to use the OmniVoice voices and how to clone a voice with OmniVoice.

[OmniVoice](https://github.com/k2-fsa/OmniVoice) (k2-fsa) is a zero-shot text-to-speech model for more than 600 languages. The bundled index holds 646 OmniVoice voices, one per language, with ids such as `omnivoice/en`. A voice config selects the adapter with `engine: omnivoice`.

OmniVoice is a masked diffusion language model. The audio starts as a grid of MASK tokens, and a fixed number of steps replace the most confident MASK slots with codec codes until none are left. The codec decoder then makes the waveform. This has three effects on the adapter:

1. Attention is bidirectional. A slot in the middle must see the slots after it, so the backbone reads the whole sequence at once. There is no causal mask and no KV cache.
2. Every step is a full forward pass. Nothing is cached between steps, so the cost grows with the number of steps times the sequence length.
3. The length is set before decoding. The model never emits an end-of-speech token, so the adapter estimates the number of frames from the text.

## Graphs

All voices use the graphs of [`Phoonnx/omnivoice-multi-unicode`](https://huggingface.co/Phoonnx/omnivoice-multi-unicode).

| `engine_params` key | Graph | Contract |
|---|---|---|
| *(the voice's own model)* | `omnivoice_backbone.onnx` | `(input_ids[B,8,S] int64, audio_mask[B,S] bool)` → `logits[B,8,S,1025]` |
| `acoustic_encoder_path` | `acoustic_encoder.onnx` | reference wav @24 kHz `(1,1,T)` → `(1,256,T')` |
| `semantic_encoder_path` | `semantic_encoder.onnx` | reference wav @16 kHz `(1,T)` → `(1,768,T')` |
| `quantizer_encoder_path` | `quantizer_encoder.onnx` | acoustic + semantic → codes `(8,1,T')` |
| `decoder_path` | `higgs_decoder.onnx` | codes `(8,1,T')` → waveform @24 kHz |
| `bpe_tokenizer_path` | `tokenizer.json` | the model's own Qwen3 subword BPE |

The backbone is one graph that holds the audio embeddings, the Qwen3 core and the heads of each codebook. The three encoder graphs open only when a call clones.

The codec is the Higgs Audio V2 tokenizer. It has a DAC acoustic branch at 24 kHz and a HuBERT semantic branch at 16 kHz, quantized into 8 residual codebooks at 25 frames per second.

## Prompt layout

```
<|denoise|>                                   (cloning only)
<|lang_start|>{language or None}<|lang_end|>
<|instruct_start|>{instruct or None}<|instruct_end|>
<|text_start|>{reference transcription + " " + text}<|text_end|>
{reference codec codes}                       (cloning only)
{MASK × target_len}                           ← what gets decoded
```

All eight codebook rows hold the same text IDs. `audio_mask` marks where the rows stop being copies of the text and start being codec streams, so the embedding layer knows which table to use for each position.

The adapter joins the reference transcription and the target text into one string, not two token sequences.

## Cloning

Like ZipVoice, OmniVoice clones in context: it needs the reference clip and its transcription. See [Voice cloning](../../cloning.md). The adapter trims the silence of the reference clip with `pydub`, which the `omnivoice` extra installs (`pip install --pre "phoonnx[omnivoice,cloning]"`).

```python
from phoonnx.config import SynthesisConfig
from phoonnx.model_manager import TTSModelManager

manager = TTSModelManager()
manager.merge_default_voices()
voice = manager.voices["omnivoice/en"].load()
syn = SynthesisConfig(
    speaker_reference="reference.wav",
    speaker_reference_text="Transcription of the reference clip.",
)
for chunk in voice.synthesize("Speech models speak in hundreds of languages.", syn):
    audio = chunk.audio_float_array  # 24 kHz
```

Use a reference clip of 3 to 10 seconds. A longer clip makes every step slower, because its codes are in the sequence that the backbone reads at each step.

Without a reference, the model picks a voice. In the voice-design mode, `instruct` describes the voice in words, for example "a calm older man".

## Parameters

| Parameter | Default | What it does |
|---|---|---|
| `num_step` | 32 | Unmasking steps. Fewer is faster and rougher. |
| `guidance_scale` | 2.0 | Classifier-free guidance. 0 disables it and halves the cost. |
| `t_shift` | 0.1 | Warps the time grid. Smaller leaves more slots for the late steps. |
| `layer_penalty_factor` | 5.0 | Biases unmasking towards the low codebooks, which carry more of the signal. |
| `position_temperature` | 5.0 | Randomness in *which* slot is filled next. 0 is greedy. |
| `class_temperature` | 0.0 | Randomness in *which code* is chosen. 0 is greedy. |
| `length_scale` | 1.0 | Scales the estimated duration. Above 1 is slower speech. |

These parameters have no numeric default:

| Parameter | Default | What it does |
|---|---|---|
| `lang` | the language of the voice | Fills the language slot of the prompt. An unknown tag gives a prompt with no language. |
| `instruct` | none | Voice-design description |
| `seed` | none | Makes the random path reproducible |

With both temperatures at 0, generation is deterministic.

## Classifier-free guidance

Each step runs the backbone twice: once over the full prompt, and once over the target span alone, with no text, language or reference. Upstream batches both rows behind a `[2B,1,S,S]` block mask. For a single item, that mask lets each row attend only within its own length, so the adapter runs two separate forwards of different lengths and gets the same result.

## A warning about the community export

`onnx-community/OmniVoice-Onnx` routes the backbone through `com.microsoft::GroupQueryAttention`, which is always causal. OmniVoice needs bidirectional attention, so do not use that backbone with this adapter. The backbone in `Phoonnx/omnivoice-multi-unicode` keeps the bidirectional mask. Its codec graphs are copies of the community codec graphs.

## Cost

OmniVoice is slower than real time on a CPU. Each call costs `num_step × 2` full forward passes over the whole prompt. A shorter reference clip and shorter text make it faster, and `guidance_scale: 0` halves the cost, with some loss of quality.

## Licence

The OmniVoice weights are CC-BY-NC-4.0, so you may not use them commercially.

---

[Docs index](../../index.md) · Previous: [OuteTTS](outetts.md) · Next: [SuperTonic](supertonic.md)
